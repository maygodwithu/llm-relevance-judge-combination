"""Validation-pair scoring for the MS MARCO separate-validation threshold (user choice A0 = the EXISTING val200 selection pairs, unchanged): msm_final_eval2.py (the main-experiment Qwen evaluator) with exactly these changes:
the evaluation pool is the frozen val200 pair file (200 queries x [1 positive, 1 uniform-drawn BM25 top-1000 negative], labels Relevant / Irrelevant -> grade 3 / 0) instead of TREC-DL; same instruction texts, same FP32 scorer evaluation/rescore_fp32.py score_both, same max_length 384, same label + terminator, no threshold search.
Gate (first cell): (a) reproduction: the same process scores the first N TREC-DL 2019 queries of the same checkpoint and compares with the stored main-experiment evaluation file (max |difference| and number of identical values are printed); (b) input shortening on the val pairs
(prefix length vs the 384 budget, both labels) is counted; (c) every score is finite.
    CUDA_VISIBLE_DEVICES=<gpu> python methods/msm_val_eval.py --models tag:instr:checkpoint ... --out_dir <dir> --local_files_only [--repro_trec2019 N]"""
import argparse
import glob
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = next((d for d in Path(__file__).resolve().parents if (d / ".repo-root").is_file()), None)
for rel in ("", "evaluation", "cpmr_experiment/scripts", "cpmr_experiment/instructions_v2/scripts"):
    sys.path.insert(0, str(ROOT / rel))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import rescore_fp32 as R  # noqa: E402
from layer_drift_lib import locate_blocks  # noqa: E402
from layer_drift_repr import load_adapter  # noqa: E402
from src.eval_rerank import load_test_pool  # noqa: E402
from v2_input_path import load_v2_instructions  # noqa: E402

PASSAGE_PREFIX = "Passage: "


def load_val_pairs(path):
    queries, judged, doc_text, qrels = {}, {}, {}, {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        t = r["product_text"]
        assert t.startswith(PASSAGE_PREFIX), r["product_id"]
        q, d = r["query_id"], r["product_id"]
        queries[q] = r["query"]
        judged.setdefault(q, []).append(d)
        doc_text[d] = t[len(PASSAGE_PREFIX):]
        qrels.setdefault(q, {})[d] = 3 if r["label"] == "Relevant" else 0
    return queries, doc_text, qrels, judged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen3_4b_msm_loss.yaml")
    ap.add_argument("--models", nargs="+", required=True, help="tag:instr:checkpoint path (or tag:instr:BASE)")
    ap.add_argument("--custom_instruction_file", nargs="+", default=[])
    ap.add_argument("--manifest", default="cpmr_experiment/instructions_v2/manifest/instructions_v2.json")
    ap.add_argument("--val_pairs", default="data/msmarco_po/val200/pairs_text_basic_seed0.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--repro_trec2019", type=int, default=0, help="gate: for the FIRST job of the process also score the first N TREC-DL 2019 queries and compare with the stored evaluation file (nothing durable from this part)")
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()
    from src.model import load_model_and_tokenizer
    t0 = time.time()
    cfg = yaml.safe_load(open(ROOT / args.config))
    loaded = load_model_and_tokenizer(model_name=cfg["model_name"], lora_r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"], target_modules=cfg["lora"]["target_modules"],
                                      gradient_checkpointing=False, local_files_only=args.local_files_only)
    model, tok = loaded.model, loaded.tokenizer
    model.eval()
    device = next(model.parameters()).device
    _, _, norm, lm_head = locate_blocks(model)
    store = {}
    norm.register_forward_pre_hook(lambda m, a: store.__setitem__("h", a[0].detach()))
    w_head = lm_head.weight.detach().float()
    instr, _, _ = load_v2_instructions(ROOT / args.manifest, roles=("TRAIN",))
    instr = dict(instr)
    for cf in args.custom_instruction_file:
        instr.update(json.load(open(ROOT / cf, encoding="utf-8")))
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    queries, doc_text, qrels, judged = load_val_pairs(ROOT / args.val_pairs)
    pairs_sha = hashlib.sha256(open(ROOT / args.val_pairs, "rb").read()).hexdigest()
    n_pairs = sum(len(v) for v in judged.values())
    print(f"[msm-val] val pairs: {len(judged)} queries, {n_pairs} pairs, file sha256 {pairs_sha[:16]}", flush=True)
    resp = {lab: len(tok(lab, add_special_tokens=False)["input_ids"]) for lab in ("Relevant", "Irrelevant")}
    budget = {lab: args.max_length - resp[lab] - 1 for lab in resp}

    def n_short(text):
        n = 0
        for q, ds in judged.items():
            for d in ds:
                t = tok.apply_chat_template([{"role": "user", "content": f"{text}\n\nQuery: {queries[q]}\nPassage: {doc_text[d]}"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                L_ = len(tok(t, add_special_tokens=False)["input_ids"])
                n += sum(L_ > budget[lab] for lab in budget)
        return n

    repro_done = False
    for job in args.models:
        tag, ins_id, path = job.split(":", 2)
        text = instr[ins_id]
        if path == "BASE":
            assert max(float(p_.detach().abs().max()) for n_, p_ in model.named_parameters() if "lora_B" in n_) == 0.0, "adapter not zero: load order"
            ck, ck_sha = {"global_step": 0, "seed": None}, "BASE"
        else:
            p = ROOT / path
            ck = load_adapter(model, p)
            ck_sha = hashlib.sha256(p.read_bytes()).hexdigest()
        f = out / f"msmValEval_{tag}__{ins_id}__val200.json.gz"
        if f.exists():
            print(f"[msm-val] skip {tag}", flush=True)
            continue
        t1 = time.time()
        shortened = n_short(text)
        run32, runbf = {}, {}
        for qid, cids in judged.items():
            runbf[qid], run32[qid] = R.score_both(model, tok, text, queries[qid], cids, doc_text, args.max_length, device, norm, w_head, store, args.batch_size)
        m_all = np.array([run32[q][d] for q in run32 for d in run32[q]])
        assert np.isfinite(m_all).all(), "nonfinite score"
        grades = {q: {d: qrels[q][d] for d in run32[q]} for q in run32}
        tp = fp = fn = tn = 0
        for q in run32:
            for d, m in run32[q].items():
                pr, tr = m > 0, grades[q][d] >= 2
                tp += pr and tr; fp += pr and not tr; fn += (not pr) and tr; tn += (not pr) and (not tr)
        po = (tp + tn) / n_pairs; pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n_pairs ** 2
        print(f"[msm-val] {tag} val200: {n_pairs} pairs, kappa(t=0) {(po - pe) / (1 - pe):.4f}, predicted Relevant {100 * (tp + fp) / n_pairs:.1f}%, inputs needing shortening {shortened}, nonfinite 0 ({time.time() - t1:.0f}s)", flush=True)
        rec = {"meta": {"tag": tag, "set": "val200", "checkpoint": path, "checkpoint_sha256": ck_sha, "global_step": ck.get("global_step"), "seed": ck.get("seed"), "instruction": ins_id,
                        "instruction_sha256": hashlib.sha256(text.encode()).hexdigest(), "max_length": args.max_length, "n_pairs": n_pairs, "val_pairs_sha256": pairs_sha, "inputs_shortened": shortened,
                        "scorer": "evaluation/rescore_fp32.py score_both (FP32 norm/head/log_softmax at the supervised positions, label + terminator)", "model_name": cfg["model_name"], "torch": torch.__version__},
               "run_fp32": run32, "run_bf16": runbf, "grades": grades}
        tmp = f.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(rec, fh)
        tmp.replace(f)
        if args.repro_trec2019 and not repro_done:
            repro_done = True
            tq, tcand, tdoc, tqrels = load_test_pool(year="2019")
            tj = {q: [d for d in cids if tqrels.get(q, {}).get(d, -1) >= 0] for q, cids in tcand.items() if q in tq}
            tj = {q: v for q, v in tj.items() if v}
            stored = None
            for sf in glob.glob(str(ROOT / "cpmr_experiment" / "**" / f"msmEval_{tag}__{ins_id}__2019.json.gz"), recursive=True):
                stored = json.load(gzip.open(sf, "rt", encoding="utf-8"))["run_fp32"]
                break
            assert stored is not None, "no stored TREC-DL 2019 file for this checkpoint"
            diffs = []
            for n, (qid, cids) in enumerate(tj.items()):
                if n >= args.repro_trec2019:
                    break
                _, s32 = R.score_both(model, tok, text, tq[qid], cids, tdoc, args.max_length, device, norm, w_head, store, args.batch_size)
                diffs += [abs(float(s32[d]) - float(stored[qid][d])) for d in cids]
            diffs = np.array(diffs)
            print(f"[msm-val] GATE reproduction {tag}: {len(diffs)} TREC-DL 2019 pairs, max |new - stored| {diffs.max():.3e}, identical {int((diffs == 0).sum())}/{len(diffs)}", flush=True)
    print(f"[msm-val] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
