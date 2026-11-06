"""Stage 1 of the ESCI-T Llama APO-CF run (experiment log 166; Llama-3.1-8B-Instruct Base is judge AND proposer, one refinement per instruction, then freeze; port of antique_apocf_p1.py with the product domain, rules unchanged except the data and the selection metric):
  1. Diagnostic confusion: Base + original Pk on ALL judged rows of the 200 frozen diagnostic queries (truth: ESCI label E = Relevant, S/C/I = Irrelevant); decision = label+<|eot_id|> margin > 0.
  2. Proposer: the UNCHANGED APO-CF meta-prompt (sftpo/src/rewriter.py build_apocf_prompt, domain='product', level='full') once per variant (shorter, similar, bare) through the official Qwen3 chat template with enable_thinking=False.
     temperature 1.0, top_p 1.0, top_k disabled, max_new_tokens 512, torch.manual_seed(0) once before the first call, order shorter -> similar -> bare, batch 1, bf16. Prompts and raw replies are written to disk BEFORE anything is scored.
     No addendum, no regeneration, no repair, no human edit.
  3. Fixed parser (as in 147/151/154). 4. Validity: mipro_core.BodyChecker (empty / more than one paragraph / output-format instruction / more than 128 tokens) + unbalanced quotes + generation ended without an end token + duplicates.
     Input truncation cannot occur by construction (max_length reserves 128 body tokens + suffix; the scorer counts shortenings and the count is stored). Invalid candidates are kept in the record.
  5. Selection: binary Cohen kappa over ALL judged rows of the 200 frozen selection queries (E = Relevant; margin > 0 = Relevant); original wins ties, then slot order shorter / similar / bare; an undefined kappa stops the run (reported, no other metric is substituted).
     nDCG is not used. Decision + frozen manifest are written; SFT is a separate step.
    CUDA_VISIBLE_DEVICES=0 python methods/esci_apocf_p1.py --orig_id P3 --out_dir cpmr_experiment/esci/p1_P3 --local_files_only"""
import argparse
import gzip
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = next((d for d in Path(__file__).resolve().parents if (d / ".repo-root").is_file()), None)
for rel in ("", "methods", "evaluation", "cpmr_experiment/scripts"):
    sys.path.insert(0, str(ROOT / rel))

import torch  # noqa: E402
import yaml  # noqa: E402

import mipro_core as C  # noqa: E402
from apocf_online_lib import load_apocf_module  # noqa: E402
from esci_llama_core import score_batch  # noqa: E402
from esci_llama_eval import by_query  # noqa: E402

VARIANTS = ("shorter", "similar", "bare")
EOS_IDS = [128009, 128001]
CUT = re.compile(r"\s*Output exactly one label")
PREAMBLE = re.compile(r"^\s*(here\b|here's|sure\b|certainly|i have|i've|i revised|the revised|revised\b|below\b|this instruction|note\b)", re.IGNORECASE)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None   # the 1,000-query test / 20-per-query SFT files of the earlier freeze are not part of the public package


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", default="configs/llama31_8b_esci.yaml")
    ap.add_argument("--instr_file", default="cpmr_experiment/esci/audit/esci_p_manifest.json")
    ap.add_argument("--orig_id", required=True)
    ap.add_argument("--audit", default="cpmr_experiment/esci/audit")
    ap.add_argument("--rewriter_path", default=str(ROOT / "src" / "apocf_prompt.py"))
    ap.add_argument("--max_body_tokens", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--gen_seed", type=int, default=0)
    ap.add_argument("--max_sel_queries", type=int, default=0, help="debug only: nothing durable is written")
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    from src.model import load_model_and_tokenizer
    cfg = yaml.safe_load(open(ROOT / args.config))
    A = ROOT / args.audit
    max_length = json.load(open(ROOT / "cpmr_experiment/esci_llama/audit/max_length.json"))["max_length"]
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
    man = json.load(open(ROOT / args.instr_file, encoding="utf-8"))
    e0 = next(e for e in man["entries"] if e["display_id"] == args.orig_id)
    orig_full = e0["full"]
    assert hashlib.sha256(orig_full.encode()).hexdigest() == e0["sha256_full"]
    start_body = C.split_start(orig_full)
    diag, sel = by_query(A / "diag_rows.jsonl"), by_query(A / "sel_rows.jsonl")
    if args.max_sel_queries:
        sel = dict(list(sel.items())[: args.max_sel_queries])
    n_shr_total = [0]

    def score_set(full, data):
        """margins [lp_label(R), lp_eot(R), lp_label(I), lp_eot(I)] per row, queries in file order, batches of 4 (as the evaluation)"""
        res_all = {}
        for qid, e in data.items():
            dt = {d: t for d, t, g in e["docs"]}
            ids = [d for d, t, g in e["docs"]]
            res = {}
            for i in range(0, len(ids), 4):
                r, s = score_batch(core, norm, w_head, store, tok, full, e["query"], ids[i:i + 4], dt, max_length, device)
                res.update(r); n_shr_total[0] += s
            res_all[qid] = {d: [res[d]["Relevant"]["label"], res[d]["Relevant"]["eot"], res[d]["Irrelevant"]["label"], res[d]["Irrelevant"]["eot"]] for d in ids}
        return res_all

    def confusion(margins, data):
        tp = fp = fn = tn = 0
        for qid, e in data.items():
            g = {d: x for d, t, x in e["docs"]}
            for d, v in margins[qid].items():
                pred, truth = (v[0] + v[1] - v[2] - v[3]) > 0, g[d] == "E"
                tp += pred and truth; fp += pred and not truth; fn += (not pred) and truth; tn += (not pred) and (not truth)
        return int(tp), int(fp), int(fn), int(tn)
    # ---------------------------------------------------------------- 1. diagnostic confusion
    conf_path = out / "p1_confusion.json"
    if conf_path.exists():
        conf = json.load(open(conf_path, encoding="utf-8"))
        assert conf["original_sha256"] == hashlib.sha256(orig_full.encode()).hexdigest()
    else:
        t1 = time.time()
        tp, fp, fn, tn = confusion(score_set(orig_full, diag), diag)
        by_label = {}
        conf = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_pairs": tp + fp + fn + tn, "rule": "label+<|eot_id|> margin > 0 -> Relevant (FP32 head); truth = ESCI label E; all judged rows of the 200 diagnostic queries",
                "original_sha256": hashlib.sha256(orig_full.encode()).hexdigest(), "diag_rows_sha256": sha_file(A / "diag_rows.jsonl"), "seconds": time.time() - t1}
        if not args.max_sel_queries:
            C._atomic_write(conf_path, conf)
    print(f"[p1] confusion of Base + {args.orig_id} on {conf['n_pairs']} diagnostic rows: TP {conf['tp']} FP {conf['fp']} FN {conf['fn']} TN {conf['tn']}", flush=True)
    # ---------------------------------------------------------------- 2. proposer
    mod = load_apocf_module(args.rewriter_path)
    ex_path = out / "p1_exchange.json"
    if ex_path.exists():
        records = json.load(open(ex_path, encoding="utf-8"))
    else:
        records = []
        torch.manual_seed(args.gen_seed); torch.cuda.manual_seed_all(args.gen_seed)
        for v in VARIANTS:
            prompt = mod.build_apocf_prompt(orig_full, conf["tn"], conf["fp"], conf["fn"], conf["tp"], domain="product", level="full", constraint_variant=v)
            chat = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            ids = tok(chat, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            t1 = time.time()
            with torch.no_grad():
                gen = inner.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=args.max_new_tokens, eos_token_id=EOS_IDS, pad_token_id=128004)
            new = gen[0, ids.shape[1]:].tolist()
            finished = any(t in EOS_IDS for t in new)
            raw = tok.decode([t for t in new if t not in EOS_IDS], skip_special_tokens=True)
            records.append({"variant": v, "system_and_user_prompt_as_rendered": chat, "meta_prompt": prompt, "meta_prompt_sha256": C.sha(prompt), "prompt_tokens": int(ids.shape[1]), "n_new_tokens": len(new), "finished_with_end_token": finished,
                            "raw_reply": raw, "seconds": time.time() - t1,
                            "sampling": {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": args.max_new_tokens, "eos_token_ids": EOS_IDS, "seed": args.gen_seed,
                                         "rng": "torch.manual_seed(seed) (cpu + cuda) once before the first call; calls shorter, similar, bare draw from the same global stream; batch size 1; bf16",
                                         "model": cfg["model_name"], "revision": revision, "adapter": "none (LoRA B = 0, untrained)", "chat_template": "Llama-3.1 official"}})
            print(f"[p1] proposer {v}: {len(new)} new tokens, finished {finished}, {time.time() - t1:.0f}s", flush=True)
        if not args.max_sel_queries:
            C._atomic_write(ex_path, records)
    for r in records:
        print(f"[p1] raw reply ({r['variant']}): {r['raw_reply']!r}", flush=True)
    # ---------------------------------------------------------------- 3./4. parse and validity
    checker = C.BodyChecker(tok, start_body, val_check=lambda full: 0, train_check=None, max_body_tokens=args.max_body_tokens, val_tolerance=0, train_tolerance=0)
    cand_rows = [{"index": 0, "variant": "unchanged", "body": start_body, "body_sha256": C.sha(start_body), "body_tokens": checker.n_tokens(start_body), "violations": [], "valid": True, "preamble_flag": False}]
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
        if not bad:
            seen[C.norm_text(body)] = i
        cand_rows.append(row)
    for r in cand_rows:
        print(f"[p1] candidate {r['index']} {r['variant']}: {r['body_tokens']} tokens; " + ("valid" if r["valid"] else "INVALID: " + "; ".join(r["violations"])) + (" [preamble-like start]" if r["preamble_flag"] else ""), flush=True)
    # ---------------------------------------------------------------- 5. selection by kappa on the selection queries
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
        print(f"[p1] candidate {r['index']} scored on {len(sel)} selection queries ({tp + fp + fn + tn} rows): kappa {k:.4f}, Rel {r['rel_pct']:.1f}% ({time.time() - t1:.0f}s)", flush=True)
        if not args.max_sel_queries:
            tmp = out / f"p1_raw_cand{r['index']}.tmp"
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                json.dump({"full_sha256": C.sha(full), "fields": "[lp_label(R), lp_eot(R), lp_label(I), lp_eot(I)] per row; margin = f0+f1-f2-f3", "margins": margins}, f)
            tmp.replace(out / f"p1_raw_cand{r['index']}.json.gz")
    valid = [r for r in cand_rows if r["valid"]]
    by_k = sorted(valid, key=lambda r: (-round(r["kappa_selection"], 12), r["index"]))
    sel0 = by_k[0]
    res = {"stage": "ESCI-T Llama APO-CF stage 1 (Llama-3.1-8B-Instruct Base judge + Base proposer, one refinement)", "original_id": args.orig_id, "original_full": orig_full, "original_sha256": e0["sha256_full"],
           "confusion": {k: conf[k] for k in ("tp", "fp", "fn", "tn")}, "variants": list(VARIANTS), "max_body_tokens": args.max_body_tokens, "max_length": max_length, "candidates": cand_rows, "n_valid_unique": len(valid),
           "n_selection_queries": len(sel), "n_selection_rows": sum(len(e["docs"]) for e in sel.values()), "inputs_shortened_total": n_shr_total[0],
           "selection": {"index": sel0["index"], "variant": sel0["variant"], "body_sha256": sel0["body_sha256"], "kappa": sel0["kappa_selection"], "original_kappa": cand_rows[0]["kappa_selection"], "replaced": sel0["index"] != 0,
                         "gap_to_second": (abs(by_k[0]["kappa_selection"] - by_k[1]["kappa_selection"]) if len(by_k) > 1 else None),
                         "rule": "max binary kappa over all judged rows of the selection queries (E = Relevant); ties -> original, then slot order shorter / similar / bare; nDCG not used"},
           "model": cfg["model_name"], "model_revision": revision, "torch": torch.__version__, "code_sha256": sha_file(__file__),
           "file_sha256": {"diag": sha_file(A / "diag_rows.jsonl"), "sel": sha_file(A / "sel_rows.jsonl"), "sft": sha_file(A / "sft_rows.jsonl"), "test": sha_file(A / "test_rows.jsonl"), "rewriter": sha_file(args.rewriter_path)}, "seconds": time.time() - t0}
    print(f"[p1] selected slot {sel0['index']} ({sel0['variant']}): kappa {sel0['kappa_selection']:.4f} vs original {cand_rows[0]['kappa_selection']:.4f}; replaced={sel0['index'] != 0}; shortened {n_shr_total[0]}; {time.time() - t0:.0f}s", flush=True)
    if args.max_sel_queries:
        return 0
    C._atomic_write(out / "p1_result.json", res)
    entries = [{"display_id": args.orig_id, "source_id": e0["source_id"], "full": orig_full, "sha256_full": e0["sha256_full"]}]
    if sel0["index"] != 0:
        new_full = sel0["body"].strip() + C.SUFFIX
        entries.append({"display_id": f"{args.orig_id}_APOCF", "source_id": f"{args.orig_id}_APOCF(esci-llama,llama-base-proposer,slot{sel0['index']},{sel0['variant']})", "full": new_full, "sha256_full": C.sha(new_full)})
    C._atomic_write(out / "esci_apocf_manifest.json", {"note": "frozen before any test evaluation of the refined instruction; selection by kappa on the selection set", "entries": entries})
    return 0


if __name__ == "__main__":
    sys.exit(main())
