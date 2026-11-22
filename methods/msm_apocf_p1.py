"""Stage 1 of the MS MARCO APO-CF run: the Base model (Qwen3-4B or Llama-3.1-8B-Instruct) is judge AND proposer, one refinement per original instruction Tk, then freeze.
Port of esci_apocf_p1.py / esci_llama_apocf_p1.py (rules unchanged) with the passage domain and the MS MARCO pair sets:
  1. Diagnostic confusion: Base + original Tk on the 400 frozen diagnostic pairs (200 queries, positive + one basic negative, SFT rule; data/msmarco_po/diag200); decision = label + terminator margin > 0 (FP32 head).
  2. Proposer: the UNCHANGED APO-CF meta-prompt (rewriter.py build_apocf_prompt, domain='passage', level='full') once per variant (shorter, similar, bare), the model's own chat template, temperature 1.0, top_p 1.0, top_k disabled,
     max_new_tokens 512, torch.manual_seed(0) once before the first call, order shorter -> similar -> bare, batch 1, bf16. Prompts and raw replies are written to disk BEFORE anything is scored. No regeneration, no repair, no human edit.
  3. Fixed parser (as ESCI). 4. Validity: mipro_core.BodyChecker (empty / more than one paragraph / output-format instruction / more than 128 tokens) + unbalanced quotes + generation ended without an end token + duplicates
     + ADDITIONAL INPUT TRUNCATION: a candidate is invalid if, with the candidate, more diagnostic / selection inputs (both labels) or more SFT training inputs (both labels, the content the trainers see) need passage shortening than with the original
     (tolerance 0). TREC-DL inputs are NOT used here (no TREC data in any PO decision); their shortening is reported after the selection is frozen.
  5. Selection: binary Cohen kappa over the 400 frozen selection pairs (200 queries, val200 queries with SFT-rule pairs; data/msmarco_po/val200); original wins ties, then slot order shorter / similar / bare; undefined kappa stops the run.
SFT is a separate step.
    CUDA_VISIBLE_DEVICES=0 python methods/msm_apocf_p1.py --model qwen --orig_id T1 --out_dir cpmr_experiment/msm_po/qwen/p1_T1 --local_files_only"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = next((d for d in Path(__file__).resolve().parents if (d / ".repo-root").is_file()), None)
for rel in ("", "methods", "evaluation", "cpmr_experiment/scripts", "cpmr_experiment/instructions_v2/scripts"):
    sys.path.insert(0, str(ROOT / rel))

import torch  # noqa: E402
import yaml  # noqa: E402

import mipro_core as C  # noqa: E402
from apocf_online_lib import load_apocf_module  # noqa: E402

VARIANTS = ("shorter", "similar", "bare")
CUT = re.compile(r"\s*Output exactly one label")
PREAMBLE = re.compile(r"^\s*(here\b|here's|sure\b|certainly|i have|i've|i revised|the revised|revised\b|below\b|this instruction|note\b)", re.IGNORECASE)
MODELS = {
    "qwen": dict(config="configs/qwen3_4b_msm_loss.yaml", eos_ids=[151645, 151643], pad_id=151643, template_kwargs={"enable_thinking": False}, terminator="<|im_end|>"),
    "llama": dict(config="configs/llama31_8b_msm.yaml", eos_ids=[128009, 128001], pad_id=128004, template_kwargs={}, terminator="<|eot_id|>"),
}
PASSAGE_PREFIX = "Passage: "


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse_reply(raw):
    t = raw.strip()
    t = re.sub(r"^(revised\s+instruction\s*:?\s*)", "", t, flags=re.IGNORECASE).strip()
    m = CUT.search(t)
    body = re.sub(r"\s+", " ", t[: m.start()] if m else t).strip()
    if body.startswith('"') and body.count('"') % 2 == 1:
        body = body[1:].strip()
    elif len(body) > 1 and body[0] == '"' and body[-1] == '"' and body.count('"') == 2:
        body = body[1:-1].strip()
    return body


def kappa(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    return None if pe == 1 else (po - pe) / (1 - pe)


def load_pairs(path):
    """SFT-format pair rows -> {query_id: {"query", "docs": [(doc_id, passage text WITHOUT the 'Passage: ' marker, is_relevant)]}} in file order"""
    out = {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        t = r["product_text"]
        assert t.startswith(PASSAGE_PREFIX), r["product_id"]
        e = out.setdefault(r["query_id"], {"query": r["query"], "docs": []})
        e["docs"].append((r["product_id"], t[len(PASSAGE_PREFIX):], r["label"] == "Relevant"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--orig_id", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--manifest", default="cpmr_experiment/instructions_v2/manifest/instructions_v2.json")
    ap.add_argument("--custom_instruction_file", default="methods/custom_instructions_T6_T7.json")
    ap.add_argument("--diag_pairs", default="data/msmarco_po/diag200/pairs_text_basic_diag200.jsonl")
    ap.add_argument("--sel_pairs", default="data/msmarco_po/val200/pairs_text_basic_seed0.jsonl")
    ap.add_argument("--train_pairs", default="data/msmarco_train_v2_5k/pairs_text_basic_5000.jsonl")
    ap.add_argument("--passage_cache", default="cpmr_experiment/instructions_v2/cache_5k/msmarco5k_basic__cache.json", help="Qwen only: the cached passage content the Qwen trainer feeds")
    ap.add_argument("--rewriter_path", default=str(ROOT / "src" / "apocf_prompt.py"))
    ap.add_argument("--max_body_tokens", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--gen_seed", type=int, default=0)
    ap.add_argument("--max_sel_queries", type=int, default=0, help="debug only: nothing durable is written")
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()
    spec = MODELS[args.model]
    t0 = time.time()
    out = ROOT / args.out_dir
    if not args.max_sel_queries:
        out.mkdir(parents=True, exist_ok=True)
    from src.model import load_model_and_tokenizer
    cfg = yaml.safe_load(open(ROOT / spec["config"]))
    loaded = load_model_and_tokenizer(model_name=cfg["model_name"], lora_r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"], target_modules=cfg["lora"]["target_modules"],
                                      gradient_checkpointing=False, local_files_only=args.local_files_only)
    model, tok = loaded.model, loaded.tokenizer
    model.eval()
    inner = model.base_model.model
    core = inner.model
    norm = core.norm
    w_head = inner.lm_head.weight.detach().float()
    device = next(model.parameters()).device
    revision = "local:" + hashlib.sha256(open(Path(cfg["model_name"]) / "model.safetensors.index.json", "rb").read()).hexdigest()[:16] + "/" + hashlib.sha256(open(Path(cfg["model_name"]) / "config.json", "rb").read()).hexdigest()[:16]
    assert max(float(p.detach().abs().max()) for n, p in model.named_parameters() if "lora_B" in n) == 0.0, "not the untrained Base"
    store = {}
    norm.register_forward_pre_hook(lambda m, a: store.__setitem__("h", a[0].detach()))
    # ---------------------------------------------------------------- instruction and data
    man = json.load(open(ROOT / args.manifest, encoding="utf-8"))
    instr = {k: v["body"] + man["output_suffix"] for k, v in man["entries"].items() if "body" in v}
    instr.update(json.load(open(ROOT / args.custom_instruction_file, encoding="utf-8")))
    orig_full = instr[args.orig_id]
    assert orig_full.endswith(C.SUFFIX) and orig_full.count("Output exactly one label") == 1
    start_body = C.split_start(orig_full)
    diag, sel = load_pairs(ROOT / args.diag_pairs), load_pairs(ROOT / args.sel_pairs)
    train = load_pairs(ROOT / args.train_pairs)
    assert not (set(diag) & set(sel)) and not (set(diag) & set(train)) and not (set(sel) & set(train)), "diagnostic / selection / training queries must be disjoint"
    if args.max_sel_queries:
        sel = dict(list(sel.items())[: args.max_sel_queries])
    if args.model == "qwen":
        from rescore_fp32 import score_both

        def margins_of(full, qid, e):
            ids = [d for d, t, g in e["docs"]]
            _, s32 = score_both(model, tok, full, e["query"], ids, {d: t for d, t, g in e["docs"]}, args.max_length, device, norm, w_head, store, 4)
            return [float(s32[d]) for d in ids]
    else:
        from llama_r7_eval import score_batch

        def margins_of(full, qid, e):
            ids = [d for d, t, g in e["docs"]]
            dt = {d: t for d, t, g in e["docs"]}
            res = {}
            for i in range(0, len(ids), 4):
                r, s = score_batch(core, norm, w_head, store, tok, full, e["query"], ids[i:i + 4], dt, args.max_length, device)
                res.update(r)
            return [res[d]["Relevant"]["label"] + res[d]["Relevant"]["eot"] - res[d]["Irrelevant"]["label"] - res[d]["Irrelevant"]["eot"] for d in ids]

    def score_set(full, data):
        with torch.no_grad():
            return {qid: margins_of(full, qid, e) for qid, e in data.items()}

    def confusion(margins, data):
        tp = fp = fn = tn = 0
        for qid, e in data.items():
            for m, (d, t, g) in zip(margins[qid], e["docs"]):
                pred = m > 0
                tp += pred and g; fp += pred and not g; fn += (not pred) and g; tn += (not pred) and (not g)
        return int(tp), int(fp), int(fn), int(tn)

    # ---------------------------------------------------------------- input-truncation counters (CPU, the model's own template, both labels)
    resp_len = {lab: len(tok(lab, add_special_tokens=False)["input_ids"]) for lab in ("Relevant", "Irrelevant")}
    budgets = {lab: args.max_length - resp_len[lab] - (1 if args.model == "llama" else 1) for lab in resp_len}   # label tokens + one terminator / EOS (as the builders)
    tk = spec["template_kwargs"]

    def prefix_len(full, query, ptext):
        t = tok.apply_chat_template([{"role": "user", "content": f"{full}\n\nQuery: {query}\n{PASSAGE_PREFIX}{ptext}"}], tokenize=False, add_generation_prompt=True, **tk)
        return len(tok(t, add_special_tokens=False)["input_ids"])

    def n_shortened(full, data, text_of=None):
        n = 0
        for qid, e in data.items():
            for d, t, g in e["docs"]:
                pt = text_of(e["query"], d, t) if text_of else t
                L = prefix_len(full, e["query"], pt)
                n += sum(L > budgets[lab] for lab in budgets)
        return n
    train_text_of = None
    if args.model == "qwen":
        sys.path.insert(0, str(ROOT / "cpmr_experiment/instructions_v2/scripts"))
        cache = json.load(open(ROOT / args.passage_cache, encoding="utf-8"))["entries"]
        by_pair = {(v["query_id"], v["doc_id"]): v["passage_fixed"] for v in cache.values()}
        qid_of = {}
        for qid, e in train.items():
            qid_of[e["query"]] = qid

        def train_text_of(query, d, t):
            fixed = by_pair[(qid_of[query], d)]
            assert fixed.startswith(PASSAGE_PREFIX)
            return fixed[len(PASSAGE_PREFIX):]

    def val_check(full):
        return n_shortened(full, diag) + n_shortened(full, sel)

    def train_check(full):
        return n_shortened(full, train, train_text_of)
    # ---------------------------------------------------------------- 1. diagnostic confusion
    conf_path = out / "p1_confusion.json"
    if conf_path.exists():
        conf = json.load(open(conf_path, encoding="utf-8"))
        assert conf["original_sha256"] == hashlib.sha256(orig_full.encode()).hexdigest()
    else:
        t1 = time.time()
        tp, fp, fn, tn = confusion(score_set(orig_full, diag), diag)
        conf = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_pairs": tp + fp + fn + tn, "rule": f"label+{spec['terminator']} margin > 0 -> Relevant (FP32 head); truth = positive / basic negative of the SFT rule; 400 diagnostic pairs (200 queries)",
                "original_sha256": hashlib.sha256(orig_full.encode()).hexdigest(), "diag_pairs_sha256": sha_file(ROOT / args.diag_pairs), "seconds": time.time() - t1}
        if not args.max_sel_queries:
            C._atomic_write(conf_path, conf)
    print(f"[p1] confusion of Base + {args.orig_id} on {conf['n_pairs']} diagnostic pairs: TP {conf['tp']} FP {conf['fp']} FN {conf['fn']} TN {conf['tn']}", flush=True)
    # ---------------------------------------------------------------- 2. proposer
    mod = load_apocf_module(args.rewriter_path)
    ex_path = out / "p1_exchange.json"
    if ex_path.exists():
        records = json.load(open(ex_path, encoding="utf-8"))
    else:
        records = []
        torch.manual_seed(args.gen_seed); torch.cuda.manual_seed_all(args.gen_seed)
        for v in VARIANTS:
            prompt = mod.build_apocf_prompt(orig_full, conf["tn"], conf["fp"], conf["fn"], conf["tp"], domain="passage", level="full", constraint_variant=v)
            chat = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, **tk)
            ids = tok(chat, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            t1 = time.time()
            with torch.no_grad():
                gen = inner.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=args.max_new_tokens, eos_token_id=spec["eos_ids"], pad_token_id=spec["pad_id"])
            new = gen[0, ids.shape[1]:].tolist()
            finished = any(t in spec["eos_ids"] for t in new)
            raw = tok.decode([t for t in new if t not in spec["eos_ids"]], skip_special_tokens=True)
            records.append({"variant": v, "system_and_user_prompt_as_rendered": chat, "meta_prompt": prompt, "meta_prompt_sha256": C.sha(prompt), "prompt_tokens": int(ids.shape[1]), "n_new_tokens": len(new), "finished_with_end_token": finished,
                            "raw_reply": raw, "seconds": time.time() - t1,
                            "sampling": {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": args.max_new_tokens, "eos_token_ids": spec["eos_ids"], "seed": args.gen_seed,
                                         "rng": "torch.manual_seed(seed) (cpu + cuda) once before the first call; calls shorter, similar, bare draw from the same global stream; batch size 1; bf16",
                                         "model": cfg["model_name"], "revision": revision, "adapter": "none (LoRA B = 0, untrained)", "chat_template_kwargs": tk or "model default", "domain": "passage"}})
            print(f"[p1] proposer {v}: {len(new)} new tokens, finished {finished}, {time.time() - t1:.0f}s", flush=True)
        if not args.max_sel_queries:
            C._atomic_write(ex_path, records)
    for r in records:
        print(f"[p1] raw reply ({r['variant']}): {r['raw_reply']!r}", flush=True)
    # ---------------------------------------------------------------- 3./4. parse and validity
    checker = C.BodyChecker(tok, start_body, val_check=val_check, train_check=train_check, max_body_tokens=args.max_body_tokens, val_tolerance=0, train_tolerance=0)
    cand_rows = [{"index": 0, "variant": "unchanged", "body": start_body, "body_sha256": C.sha(start_body), "body_tokens": checker.n_tokens(start_body), "violations": [], "valid": True, "preamble_flag": False,
                  "shortened_diag_sel": checker.base_val, "shortened_train": checker.base_train}]
    seen = {C.norm_text(start_body): 0}
    for i, rec in enumerate(records, start=1):
        body = parse_reply(rec["raw_reply"]) if rec["raw_reply"].strip() else ""
        bad = (checker.violations(body) if body else ["empty"])
        if body and body.count('"') % 2:
            bad = bad + ["unbalanced double quotes"]
        if not rec["finished_with_end_token"]:
            bad = bad + ["truncated output (max_new_tokens reached without an end-of-turn token)"]
        if body and not bad and C.norm_text(body) in seen:
            bad = [f"duplicate of candidate {seen[C.norm_text(body)]}"]
        row = {"index": i, "variant": rec["variant"], "body": body, "body_sha256": C.sha(body) if body else None, "body_tokens": checker.n_tokens(body) if body else 0, "violations": bad, "valid": not bad,
               "preamble_flag": bool(PREAMBLE.search(body)) if body else False, "meta_prompt_sha256": rec["meta_prompt_sha256"], "n_new_tokens": rec["n_new_tokens"]}
        if body:
            full_c = body.strip() + C.SUFFIX
            row["shortened_diag_sel"], row["shortened_train"] = val_check(full_c), train_check(full_c)
        if not bad:
            seen[C.norm_text(body)] = i
        cand_rows.append(row)
    for r in cand_rows:
        print(f"[p1] candidate {r['index']} {r['variant']}: {r['body_tokens']} tokens; shortened diag+sel {r.get('shortened_diag_sel')}, train {r.get('shortened_train')}; " + ("valid" if r["valid"] else "INVALID: " + "; ".join(r["violations"])) + (" [preamble-like start]" if r["preamble_flag"] else ""), flush=True)
    # ---------------------------------------------------------------- 5. selection by kappa on the selection pairs
    for r in cand_rows:
        if not r["valid"]:
            continue
        full = r["body"].strip() + C.SUFFIX
        t1 = time.time()
        margins = score_set(full, sel)
        tp, fp, fn, tn = confusion(margins, sel)
        k = kappa(tp, fp, fn, tn)
        if k is None:
            raise SystemExit(f"kappa undefined for candidate {r['index']} ({tp},{fp},{fn},{tn}); stopping, reporting required")
        r.update({"kappa_selection": k, "tp_fp_fn_tn": [tp, fp, fn, tn], "rel_pct": 100 * (tp + fp) / (tp + fp + fn + tn)})
        print(f"[p1] candidate {r['index']} scored on {len(sel)} selection queries ({tp + fp + fn + tn} pairs): kappa {k:.4f}, Rel {r['rel_pct']:.1f}% ({time.time() - t1:.0f}s)", flush=True)
    valid = [r for r in cand_rows if r["valid"]]
    best = max(r["kappa_selection"] for r in valid)
    sel0 = next(r for r in valid if r["kappa_selection"] == best)              # valid is in slot order (unchanged, shorter, similar, bare): first maximal wins, the original wins ties
    new_full = sel0["body"].strip() + C.SUFFIX
    res = {"stage": f"MS MARCO APO-CF stage 1 ({cfg['model_name']} Base judge + Base proposer, one refinement)", "model": args.model, "original_id": args.orig_id, "original_full": orig_full, "original_sha256": hashlib.sha256(orig_full.encode()).hexdigest(),
           "candidates": cand_rows, "selection": {"index": sel0["index"], "variant": sel0["variant"], "kappa": sel0["kappa_selection"], "rule": "max kappa on the 400 selection pairs over valid candidates; original wins ties, then slot order"},
           "selected_full": new_full, "selected_sha256": hashlib.sha256(new_full.encode()).hexdigest(), "changed": sel0["index"] != 0, "n_valid": len(valid),
           "data": {"diag_pairs_sha256": sha_file(ROOT / args.diag_pairs), "sel_pairs_sha256": sha_file(ROOT / args.sel_pairs), "train_pairs_sha256": sha_file(ROOT / args.train_pairs), "max_length": args.max_length},
           "model_info": {"name": cfg["model_name"], "revision": revision}, "seconds": time.time() - t0, "torch": torch.__version__}
    print(f"[p1] selected candidate {sel0['index']} ({sel0['variant']}), selection kappa {sel0['kappa_selection']:.4f}; changed {res['changed']}; {time.time() - t0:.0f}s", flush=True)
    if not args.max_sel_queries:
        C._atomic_write(out / "p1_result.json", res)
        if res["changed"]:
            C._atomic_write(out / "po_instruction.json", {f"{args.orig_id}_APOCF": new_full})


if __name__ == "__main__":
    main()
