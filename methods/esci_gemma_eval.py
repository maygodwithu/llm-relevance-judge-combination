"""Gemma-3-4B-it ESCI evaluation (experiment log 217; second server): port of esci_llama_eval.py with the Gemma scorer esci_gemma_core.score_batch (label tokens + <end_of_turn>, FP32 final norm / tied head / log_softmax, text path of Gemma3ForConditionalGeneration, max_length 704 -> no shortening; the shortened count is recorded). Score files esciGemmaEval_<model>__<instr>__<tag>.json.gz with the same 12-field layout as the Qwen / Llama ESCI files. Original docstring of the Llama evaluator follows.
ESCI-T evaluation (experiment log 166; Llama-3.1-8B-Instruct, second server): every judged row of a frozen row file (default: the 1,000 fixed test queries, 20,739 rows), label + <|eot_id|> sequence scoring, FP32 head (tied embedding matrix), own instruction.
Same scorer as the ANTIQUE / Qwen3-8B / Llama experiments (esci_core.score_batch: BF16 blocks, FP32 final norm / head / log_softmax at the supervised positions, template with enable_thinking=False, batch of 4 rows of one query in file order).
Raw record per row: [lp_label(R), lp_eot(R), lp_eos(R), lp_label(I), lp_eot(I), lp_eos(I), lp_first(R), lp_first(I), n_label_tokens(R), n_label_tokens(I), binary truth (3 = E = Relevant, 0 = S/C/I), ESCI code (0 E, 1 S, 2 C, 3 I)];
primary margin m = [lp_label + lp_eot](Relevant) - same(Irrelevant) > 0 -> Relevant.
    CUDA_VISIBLE_DEVICES=0 python methods/esci_eval.py --jobs Base:P0 E4_P0_s0_step518:P0 ... --instr_file <manifest> --ckpt_map <json> --out_dir <dir> --local_files_only"""
import argparse
import collections
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = next((d for d in Path(__file__).resolve().parents if (d / ".repo-root").is_file()), None)
for rel in ("", "methods", "evaluation", "cpmr_experiment/scripts"):
    sys.path.insert(0, str(ROOT / rel))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from esci_gemma_core import score_batch  # noqa: E402

CODE = {"E": 0, "S": 1, "C": 2, "I": 3}


def by_query(path):
    q = collections.OrderedDict()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        e = q.setdefault(r["nq"], {"query": r["query"], "docs": []})
        e["docs"].append((r["product_id"], r["product_text"], r["esci"]))
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--instr_file", required=True)
    ap.add_argument("--ckpt_map", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rows", default="cpmr_experiment/esci/audit/test300_rows.jsonl")
    ap.add_argument("--config", default="configs/gemma3_4b_esci.yaml")
    ap.add_argument("--max_length", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--tag", default="test")
    ap.add_argument("--max_queries", type=int, default=0, help="debug: nothing durable is written")
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()
    from layer_drift_repr import load_adapter
    from src.model import load_model_and_tokenizer
    t0 = time.time()
    cfg = yaml.safe_load(open(ROOT / args.config))
    max_length = args.max_length or 704
    loaded = load_model_and_tokenizer(model_name=cfg["model_name"], lora_r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"], target_modules=cfg["lora"]["target_modules"],
                                      gradient_checkpointing=False, local_files_only=args.local_files_only)
    model, tok = loaded.model, loaded.tokenizer
    model.eval()
    inner = model.base_model.model
    core = inner.model
    norm = core.language_model.norm                                  # Gemma3RMSNorm of the language model
    w_head = inner.lm_head.weight.detach().float()
    device = next(model.parameters()).device
    store = {}
    norm.register_forward_pre_hook(lambda m, a: store.__setitem__("h", a[0].detach()))
    revision = "local:" + hashlib.sha256(open(Path(cfg["model_name"]) / "model.safetensors.index.json", "rb").read()).hexdigest()[:16] + "/" + hashlib.sha256(open(Path(cfg["model_name"]) / "config.json", "rb").read()).hexdigest()[:16]
    ins = {e["display_id"]: e["full"] for e in json.load(open(ROOT / args.instr_file, encoding="utf-8"))["entries"]}
    cmap = json.load(open(ROOT / args.ckpt_map)) if args.ckpt_map else {}
    data = by_query(ROOT / args.rows)
    rows_sha = hashlib.sha256((ROOT / args.rows).read_bytes()).hexdigest()
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    loaded_tag, ck = None, {}
    for job in args.jobs:
        mname, ri = job.split(":")
        text = ins[ri]
        f = out / f"esciGemmaEval_{mname}__{ri}__{args.tag}.json.gz"
        if f.exists() and not args.max_queries:
            print(f"[esci] skip {job}", flush=True)
            continue
        if loaded_tag != mname:
            if mname == "Base":
                assert loaded_tag in (None, "Base")
                assert max(float(p.detach().abs().max()) for n, p in model.named_parameters() if "lora_B" in n) == 0.0
                ck = {"seed": None, "global_step": 0}
            else:
                ck = load_adapter(model, ROOT / cmap[mname])
            loaded_tag = mname
        raw, n_shr, n_rows, t1 = {}, 0, 0, time.time()
        for n, (qid, e) in enumerate(data.items()):
            if args.max_queries and n >= args.max_queries:
                break
            dt = {d: t for d, t, g in e["docs"]}
            ids = [d for d, t, g in e["docs"]]
            res = {}
            for i in range(0, len(ids), args.batch_size):
                r, s = score_batch(core, norm, w_head, store, tok, text, e["query"], ids[i:i + args.batch_size], dt, max_length, device)
                res.update(r); n_shr += s
            raw[qid] = {d: [res[d]["Relevant"]["label"], res[d]["Relevant"]["eot"], res[d]["Relevant"]["eos"], res[d]["Irrelevant"]["label"], res[d]["Irrelevant"]["eot"], res[d]["Irrelevant"]["eos"],
                            res[d]["Relevant"]["first"], res[d]["Irrelevant"]["first"], res[d]["Relevant"]["n"], res[d]["Irrelevant"]["n"], 3 if g == "E" else 0, CODE[g]] for d, t, g in e["docs"]}
            n_rows += len(ids)
        mj = np.array([v[0] + v[1] - v[3] - v[4] for q in raw for v in raw[q].values()])
        hj = np.array([v[10] >= 2 for q in raw for v in raw[q].values()])
        pr = mj > 0
        po = (pr == hj).mean(); pe = pr.mean() * hj.mean() + (1 - pr.mean()) * (1 - hj.mean())
        print(f"[esci] {job} {args.tag}: {len(raw)} queries, {n_rows} rows; kappa {(po - pe) / (1 - pe):.4f}, Rel {100 * pr.mean():.1f}% (E {100 * hj.mean():.1f}%), ties {(mj == 0).sum()}, shortened {n_shr} ({time.time() - t1:.0f}s)", flush=True)
        if not args.max_queries:
            meta = {"model": mname, "instruction": ri, "instruction_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "set": args.tag, "rows_file_sha256": rows_sha, "max_length": max_length, "n_queries_scored": len(raw), "n_rows_scored": n_rows,
                    "n_input_shortenings_counted_over_two_labels": n_shr, "model_name": cfg["model_name"], "model_revision": revision, "terminator_primary": "<end_of_turn> (106)", "batch_size": args.batch_size,
                    "fields": "raw[nq][product_id] = [lp_label(R), lp_eot(R), lp_eos(R), lp_label(I), lp_eot(I), lp_eos(I), lp_first(R), lp_first(I), n_label_tokens(R), n_label_tokens(I), 3 if E else 0, ESCI code 0E 1S 2C 3I]", "seed": ck.get("seed"),
                    "global_step": ck.get("global_step"), "torch": torch.__version__, "dtype_forward": "bfloat16", "head_norm": "float32 (tied embedding matrix)", "chat_template": "Gemma-3 official (user turn only)"}
            tmp = f.with_suffix(".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                json.dump({"meta": meta, "raw": raw}, fh)
            tmp.replace(f)
    print(f"[esci] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
