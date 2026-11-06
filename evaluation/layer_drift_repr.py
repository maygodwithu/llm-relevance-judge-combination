"""Measure B of the layer drift diagnostic (experiment log 090): the same input through the base model + a seed's
adapter ON and OFF (eval mode, no dropout, no grad), the change of the residual stream after every block at two
token positions. Needs a GPU for ~1,000 short forwards and the Qwen3-4B weights.

    CUDA_VISIBLE_DEVICES=0 python evaluation/layer_drift_repr.py --local_files_only
    CUDA_VISIBLE_DEVICES=0 python evaluation/layer_drift_repr.py --local_files_only --smoke   # tiny, own out dir

Samples are fixed by rule BEFORE any number is looked at:
  trained : 64 MS MARCO basic training queries, drawn with a string-seeded RNG (not the checkpoint seed), the
            positive and the negative training document of each query -> 128 inputs
  unseen  : TREC-DL 2019 queries not in the training set (query id or normalised text), the first doc (sorted by doc
            id) with qrel >= 2 and the first with qrel == 0; a query lacking either is dropped and counted. qrel 1
            and unjudged documents are never relabelled as 0.
Only the prefix up to the answer position is fed: no gold label, no EOS. The same inputs and token positions are
used for every seed. OFF is computed once and reused; its bypass of the adapter is checked, not assumed.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = next((d for d in Path(__file__).resolve().parents if (d / ".repo-root").is_file()), None)
if ROOT is None:
    raise RuntimeError("no .repo-root marker found in any parent directory")
for rel in ("", "evaluation", "cpmr_experiment/scripts", "cpmr_experiment/instructions_v2/scripts"):
    sys.path.insert(0, str(ROOT / rel))

import torch  # noqa: E402
import yaml  # noqa: E402

from layer_drift_lib import (collect_hidden, cos_dist, find_positions, rel_l2,  # noqa: E402
                             select_trec, sample_train_queries, sha256_head, write_csv, write_json)

POS_NAMES = ("instruction", "answer")


def load_adapter(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    named = dict(model.named_parameters())
    missing = [k for k in ck["adapter_state"] if k not in named]
    if missing:
        raise RuntimeError(f"{len(missing)} adapter tensors match no parameter, e.g. {missing[0]}")
    with torch.no_grad():
        for k, t in ck["adapter_state"].items():
            named[k].copy_(t.to(named[k].device, dtype=named[k].dtype))
    return ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen3_4b_pilot_2x2.yaml")
    ap.add_argument("--manifest", default="cpmr_experiment/instructions_v2/manifest/instructions_v2.json")
    ap.add_argument("--train_pairs", default="data/msmarco_train_v2_5k/pairs_text_basic_5000.jsonl")
    ap.add_argument("--passage_cache", default="cpmr_experiment/instructions_v2/cache_5k/msmarco5k_basic__cache.json")
    ap.add_argument("--trec_pool", default="data/scenarios/msmarco_trec2019_bm25top100_pool.jsonl")
    ap.add_argument("--ckpt_root", default="cpmr_experiment/pilot_tokenonly/checkpoints")
    ap.add_argument("--tags", nargs="+", default=[f"tokonly_single_s{s}_t100" for s in (0, 1, 2)])
    ap.add_argument("--out_dir", default="cpmr_experiment/single_layer_drift")
    ap.add_argument("--n_train_queries", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prompt_id", default="T1", help="instruction fed to every input (default T1, as in all earlier runs)")
    args = ap.parse_args()

    from cpmr_query_dataset import load_query_groups
    from src.model import load_model_and_tokenizer
    from v2_input_path import V2InputPath

    t0 = time.time()
    cfg = yaml.safe_load(open(ROOT / args.config))
    out = ROOT / args.out_dir / ("smoke" if args.smoke else "")
    tags = args.tags[:1] if args.smoke else args.tags
    n_train = 2 if args.smoke else args.n_train_queries

    v2 = V2InputPath(cache_paths=[ROOT / args.passage_cache], manifest_path=ROOT / args.manifest,
                     max_length=args.max_length, enable_thinking=False, roles=("TRAIN",), profile="diagnostic")
    t1 = v2.instructions[args.prompt_id]
    groups, _ = load_query_groups(ROOT / args.train_pairs, 0)

    # ---------------- samples (fixed by rule)
    train_ids = sample_train_queries(groups.keys(), n_train)
    inputs = []  # dicts: group, query_id, kind, query, ptext
    for qid in train_ids:
        g = groups[qid]
        for kind, doc in (("pos", g["pos"]), ("neg", g["neg"])):
            inputs.append({"group": "trained", "query_id": str(qid), "kind": kind, "query": g["query"],
                           "ptext": v2.fixed_passage(g["query"], doc)})
    rows = [json.loads(l) for l in open(ROOT / args.trec_pool, encoding="utf-8")]
    picked, report = select_trec(rows, [g["query"] for g in groups.values()], groups.keys())
    if args.smoke:
        picked = picked[:2]
    for p in picked:
        for kind in ("pos", "neg"):
            inputs.append({"group": "unseen", "query_id": p["query_id"], "kind": kind, "query": p["query"],
                           "ptext": f"Passage: {p[kind]['doc_text']}"})
    report["n_inputs_trained"] = sum(i["group"] == "trained" for i in inputs)
    report["n_inputs_unseen"] = sum(i["group"] == "unseen" for i in inputs)
    print(f"[repr] {report['n_inputs_trained']} trained inputs ({n_train} queries), "
          f"{report['n_inputs_unseen']} unseen inputs ({report['n_used']} queries; "
          f"{report['n_excluded_train_overlap']} excluded as seen, {report['n_without_pair']} without a qrel>=2/0 pair)")

    # ---------------- model
    loaded = load_model_and_tokenizer(
        model_name=cfg["model_name"], lora_r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"], target_modules=cfg["lora"]["target_modules"],
        gradient_checkpointing=False, local_files_only=args.local_files_only)
    model, tok = loaded.model, loaded.tokenizer
    model.eval()
    device = next(model.parameters()).device

    pos_info = []
    for it in inputs:
        info = find_positions(tok, t1, it["query"], it["ptext"], args.max_length, enable_thinking=False)
        it["ids"], it["positions"] = info["input_ids"], [info["pos_instr"], info["pos_answer"]]
        pos_info.append({k: v for k, v in info.items() if k != "input_ids"})
    n_spill = sum(p["instr_token_spills_past_instruction"] for p in pos_info)
    print(f"[repr] instruction token: {pos_info[0]['instr_token_text']!r} (spills past the instruction in "
          f"{n_spill}/{len(pos_info)} inputs); all precede the query: "
          f"{all(p['instr_token_precedes_query'] for p in pos_info)}; "
          f"prefix tokens {min(p['n_prefix_tokens'] for p in pos_info)}-{max(p['n_prefix_tokens'] for p in pos_info)}")

    def hid(it, off):
        if off:
            with model.disable_adapter():
                return collect_hidden(model, it["ids"], it["positions"], device)
        return collect_hidden(model, it["ids"], it["positions"], device)

    # ---------------- checks that the comparison means what it says (before any adapter is loaded)
    n_b_nonzero = sum(int(p.detach().abs().max() > 0) for n, p in model.named_parameters() if "lora_B" in n)
    n_b_total = sum(1 for n, _ in model.named_parameters() if "lora_B" in n)
    probe = inputs[:4]
    init_on = [hid(it, False) for it in probe]
    init_off = [hid(it, True) for it in probe]
    init_diff = max(float((a - b).abs().max()) for a, b in zip(init_on, init_off))
    print(f"[repr] init: lora_B nonzero tensors {n_b_nonzero}/{n_b_total}; max|ON-OFF| at init = {init_diff:.3e}")
    if n_b_nonzero or init_diff != 0.0:
        print("[repr] WARNING: the initial adapter is not a no-op; ON-OFF is then NOT the net training change")

    # OFF once, reused for every seed
    h_off = [hid(it, True) for it in inputs]
    off_norm = [h.double().norm(dim=-1) for h in h_off]  # (2, L+1)
    n_layers_plus = h_off[0].shape[1]
    # causal-invariance floor of the instruction position: that token cannot depend on the query or passage, so any
    # difference between inputs is numerical (bf16, different sequence lengths), not signal. Relative, per layer.
    off_inv = torch.stack([rel_l2(h_off[i][0], h_off[0][0]) for i in range(len(inputs))]).max(0).values
    n_fwd = len(probe) * 2 + len(inputs)

    per_input, per_query_acc, checks, summaries = [], {}, [], []
    for tag in tags:
        path = ROOT / args.ckpt_root / tag / "ckpt_final.pt"
        ck = load_adapter(model, path)
        seed = ck.get("seed")
        # the OFF path must equal the state before this adapter was loaded
        off_again = max(float((hid(it, True) - b).abs().max()) for it, b in zip(probe, init_off))
        on_vs_off = max(float((hid(it, False) - b).abs().max()) for it, b in zip(probe, init_off))
        n_fwd += len(probe) * 2
        print(f"[repr] {tag}: OFF after loading vs OFF before = {off_again:.3e} (must be 0); "
              f"max|ON-OFF| = {on_vs_off:.3e} (must be > 0)")
        h0_instr, inv_max, on_inv = None, 0.0, torch.zeros(n_layers_plus, dtype=torch.float64)
        for idx, it in enumerate(inputs):
            h_on = hid(it, False)
            n_fwd += 1
            r2, cd = rel_l2(h_on, h_off[idx]), cos_dist(h_on, h_off[idx])  # (2, L+1)
            if h0_instr is None:
                h0_instr = h_on[0].clone()
            else:
                inv_max = max(inv_max, float((h_on[0] - h0_instr).abs().max()))
                on_inv = torch.maximum(on_inv, rel_l2(h_on[0], h0_instr))
            for pi, pname in enumerate(POS_NAMES):
                for layer in range(n_layers_plus):
                    per_input.append({"tag": tag, "seed": seed, "group": it["group"], "query_id": it["query_id"],
                                      "doc_kind": it["kind"], "position": pname, "layer": layer,
                                      "rel_l2": float(r2[pi, layer]), "cos_dist": float(cd[pi, layer]),
                                      "h_off_norm": float(off_norm[idx][pi, layer])})
            key = (tag, seed, it["group"], it["query_id"])
            per_query_acc.setdefault(key, []).append((r2, cd))
        checks.append({"tag": tag, "seed": seed, "global_step": ck.get("global_step"), "sha256_16": sha256_head(path),
                       "off_after_load_vs_before_max_abs": off_again, "on_vs_off_probe_max_abs": on_vs_off,
                       "instruction_position_max_abs_diff_across_inputs": inv_max,
                       "instruction_position_ON_rel_L2_across_inputs_max_per_layer": [round(float(x), 6) for x in on_inv],
                       "instruction_position_OFF_rel_L2_across_inputs_max_per_layer": [round(float(x), 6) for x in off_inv]})
        print(f"[repr] {tag}: instruction-position hidden state across inputs: max|diff| {inv_max:.3e}; relative L2 max over "
              f"layers ON {float(on_inv.max()):.3e}, OFF {float(off_inv.max()):.3e} (numerical floor, not signal)")

    per_query = []
    for (tag, seed, group, qid), vals in per_query_acc.items():
        assert len(vals) == 2, (tag, group, qid, len(vals))  # one positive and one negative document
        r2 = torch.stack([v[0] for v in vals]).mean(0)
        cd = torch.stack([v[1] for v in vals]).mean(0)
        for pi, pname in enumerate(POS_NAMES):
            for layer in range(n_layers_plus):
                per_query.append({"tag": tag, "seed": seed, "group": group, "query_id": qid, "position": pname,
                                  "layer": layer, "rel_l2": float(r2[pi, layer]), "cos_dist": float(cd[pi, layer])})
    for tag in tags:
        for group in ("trained", "unseen"):
            for pname in POS_NAMES:
                for layer in range(n_layers_plus):
                    xs = [r for r in per_query if r["tag"] == tag and r["group"] == group and r["position"] == pname
                          and r["layer"] == layer]
                    if not xs:
                        continue
                    rv = torch.tensor([r["rel_l2"] for r in xs]); cv = torch.tensor([r["cos_dist"] for r in xs])
                    summaries.append({"tag": tag, "seed": xs[0]["seed"], "group": group, "position": pname,
                                      "layer": layer, "n_queries": len(xs), "mean_rel_l2": float(rv.mean()),
                                      "sd_rel_l2": float(rv.std(unbiased=True)) if len(xs) > 1 else float("nan"),
                                      "mean_cos_dist": float(cv.mean())})

    write_csv(out / "repr_drift_input.csv", per_input,
              ["tag", "seed", "group", "query_id", "doc_kind", "position", "layer", "rel_l2", "cos_dist", "h_off_norm"])
    write_csv(out / "repr_drift_query.csv", per_query,
              ["tag", "seed", "group", "query_id", "position", "layer", "rel_l2", "cos_dist"])
    write_csv(out / "repr_drift_summary.csv", summaries,
              ["tag", "seed", "group", "position", "layer", "n_queries", "mean_rel_l2", "sd_rel_l2", "mean_cos_dist"])
    write_json(out / "samples.json", {"rule": __doc__, "trec_report": report,
                                       "trained_query_ids": train_ids,
                                       "unseen_query_ids": [p["query_id"] for p in picked],
                                       "unseen_docs": [{"query_id": p["query_id"], "pos_doc": p["pos"]["doc_id"],
                                                        "pos_qrel": p["pos"]["relevance"], "neg_doc": p["neg"]["doc_id"],
                                                        "neg_qrel": p["neg"]["relevance"]} for p in picked]})
    write_json(out / "repr_drift_meta.json", {
        "model": cfg["model_name"], "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "dtype": str(next(model.parameters()).dtype), "instruction": args.prompt_id, "max_length": args.max_length,
        "enable_thinking": False, "eval_mode_dropout_off": True, "no_grad": True, "batch_size": 1,
        "layer_index": "0 = embedding output; 1..L = decoder block outputs taken with a forward hook (before the final norm)",
        "positions": {"instruction": "token containing the last character of the instruction text",
                      "answer": "last prefix token (just before the answer); no label, no EOS fed"},
        "per_query": "the positive and the negative document of a query are averaged first",
        "init_check": {"lora_B_tensors_nonzero": n_b_nonzero, "lora_B_tensors": n_b_total,
                       "max_abs_ON_minus_OFF_at_init": init_diff},
        "checkpoints": checks, "n_inputs": len(inputs), "n_forwards": n_fwd,
        "position_details_first_input": pos_info[0],
        "n_instruction_tokens_spilling_past_instruction": n_spill,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": __import__("transformers").__version__,
                     "peft": __import__("peft").__version__},
        "seconds": round(time.time() - t0, 1),
        "limits": "hidden-state change at a high layer includes change propagated from lower layers; it is not "
                  "evidence that the high layer itself learned more. Exploratory split of trained vs unseen inputs "
                  "(domain, length, composition differ); the gap is not causal evidence of overfitting.",
    })
    print(f"[repr] wrote {out} ({n_fwd} forwards, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
