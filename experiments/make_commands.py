"""Print the training / evaluation commands of the experiments of the paper (one command per line, nothing is executed).
    python experiments/make_commands.py --model qwen --dataset esci --paths ce,soft,any --instr 1-7 --seeds 0,1,2 [--stage train|eval|prep|all] [--gpu 0]
    python experiments/make_commands.py --all --out experiments/commands          (one file per model and dataset)
    python experiments/make_commands.py --pilot245 | --gemma-fp32
Conditions ("paths"):  base  = no fine-tuning;  ce = token CE;  soft = CE + 1.0 soft kappa;  any = CE + 1.0 AnyLoss-style kappa (alpha 10);
                       po-base / po-ce / po-soft = the same with the APO-CF-selected instruction (ESCI, Qwen3-4B and Llama-3.1-8B-Instruct, T1-T3; Llama keeps the original text for T1, so only T2 and T3 are separate runs).
Seeds 0, 1, 2; 625 updates; effective batch 16 (ESCI) / 8 query pairs = 16 sequences (MS MARCO); the final checkpoint only is evaluated.
The output directories follow the original run layout under cpmr_experiment/; the analysis loaders find the score files by name (cpmr_experiment/**/esciEval_*, msmEval_*, ...)."""
import argparse
from pathlib import Path

C = "cpmr_experiment"
MAN = f"{C}/esci_t/esci_t_manifest.json"
AUD = f"{C}/esci/audit"
CUSTOM = "methods/custom_instructions_T6_T7.json"


def esci_cfg(m):
    return {
        "qwen": dict(root="esci_t", ce_train="methods/esci_train.py", mix_train="methods/esci_loss_train_245.py", ev="methods/esci_eval.py", ce_tag="E4_{i}_s{s}", mix_prefix="EQ", mix_root="esci_mix", po=(1, 2, 3), p1="methods/esci_apocf_p1.py", ce_is_mix=False),
        "llama": dict(root="esci_llama", ce_train="methods/esci_llama_train.py", mix_train="methods/esci_llama_loss_train_mix.py", ev="methods/esci_llama_eval.py", ce_tag="EL_{i}_s{s}", mix_prefix="ELQ", mix_root="esci_llama", po=(2, 3), p1="methods/esci_llama_apocf_p1.py", ce_is_mix=False),
        "gemma": dict(root="esci_gemma", ce_train="methods/esci_gemma_loss_train_mix.py", mix_train="methods/esci_gemma_loss_train_mix.py", ev="methods/esci_gemma_eval.py", ce_tag="GQ_ce_{i}_s{s}", mix_prefix="GQ", mix_root="esci_gemma", po=(), p1=None, ce_is_mix=True),
    }[m]


def esci(m, paths, instr, seeds, stage, gpu):
    g = f"CUDA_VISIBLE_DEVICES={gpu} python"
    c = esci_cfg(m)
    root = f"{C}/{c['root']}"
    mixroot = f"{C}/{c['mix_root']}"
    out = []
    p1root = f"{C}/{'esci_t' if m == 'qwen' else 'esci_llama'}"
    if stage in ("prep", "all") and c["p1"]:
        out.append("# OPTIONAL regeneration of the APO-CF proposer / selection stage into a separate directory. The frozen outputs of the paper (selected instructions, Base selection scores) are shipped in cpmr_experiment/*/p1_T*/ and are what the analysis reads;")
        out.append("# the proposer samples at temperature 1.0, so a new run produces other candidate texts and is not bit-reproducible.")
        for T in instr:
            out.append(f"# {g} {c['p1']} --orig_id T{T} --instr_file {MAN} --out_dir {p1root}/p1_regen/p1_T{T} --local_files_only")
    if stage in ("prep", "all"):
        out.append(f"python scripts/make_ckpt_map.py --train-root {root}/train" + (f" --train-root {mixroot}/train" if mixroot != root else "") + f" --out {root}/ckpt_map.json     # rerun after training, before the evaluation commands")
    variants = []                      # (instruction id, instruction file, is_po)
    if any(p in paths for p in ("base", "ce", "soft", "any")):
        variants += [(f"T{T}", MAN, False) for T in instr]
    if any(p.startswith("po") for p in paths):
        variants += [(f"T{T}_APOCF", f"{p1root}/p1_T{T}/esci_apocf_manifest.json", True) for T in instr if T in c["po"]]
    for iid, man, po in variants:
        for s in seeds:
            for path in ("ce", "soft", "any"):
                key = ("po-" + path) if po else path
                if key not in paths or (po and path == "any"):
                    continue
                if stage in ("train", "all"):
                    if path == "ce" and not c["ce_is_mix"]:
                        out.append(f"{g} {c['ce_train']} --instr {iid} --seed {s} --instr_file {man} --rows {AUD}/sft2_rows.jsonl --out_root {root}/train --local_files_only")
                    else:
                        loss = {"ce": "ce", "soft": "mix", "any": "mixany"}[path]
                        prefix = " --tag_prefix " + c["mix_prefix"] if m != "gemma" else ""
                        if m == "qwen" and path in ("soft", "any"):
                            out.append(f"{g} {c['mix_train']} --loss {loss} --lam 1.0 --alpha 10 --final_only --instr {iid} --seed {s} --instr_file {man} --out_root {mixroot}/train --local_files_only")
                        else:
                            out.append(f"{g} {c['mix_train']} --loss {loss} --lam 1.0 --final_only --instr {iid} --seed {s} --instr_file {man} --rows {AUD}/sft2_rows.jsonl --out_root {mixroot}/train{prefix} --local_files_only")
    if stage in ("eval", "all"):
        for iid, man, po in variants:
            ev_sets = [("test300", "test300_rows.jsonl")] + ([("sel", "sel_rows.jsonl")] if True else [])
            for tagname, rows in ev_sets:
                jobs = []
                if (not po and "base" in paths) or (po and "po-base" in paths):
                    if not (m == "qwen" or m == "llama") or tagname == "test300":                 # Qwen / Llama Base validation scores come from the p1 stage
                        jobs.append(f"Base:{iid}")
                for s in seeds:
                    for path in ("ce", "soft", "any"):
                        key = ("po-" + path) if po else path
                        if key not in paths or (po and path == "any"):
                            continue
                        if path == "ce":
                            tag = c["ce_tag"].format(i=iid, s=s)
                        else:
                            tag = f"{c['mix_prefix']}_{'mix' if path == 'soft' else 'mixany'}_{iid}_s{s}"
                        jobs.append(f"{tag}_step625:{iid}")
                if jobs:
                    out.append(f"{g} {c['ev']} --jobs {' '.join(jobs)} --instr_file {man} --rows {AUD}/{rows} --tag {tagname} --ckpt_map {root}/ckpt_map.json --out_dir {root}/eval --local_files_only")
    return out


def msm(m, paths, instr, seeds, stage, gpu):
    g = f"CUDA_VISIBLE_DEVICES={gpu} python"
    cfg = {"qwen": dict(root="msm_t7", train="methods/msm_loss_train4.py", ev="methods/msm_final_eval2.py", config="configs/qwen3_4b_msm_loss.yaml", tag="msq_{l}_T{T}_s{s}"),
           "llama": dict(root="msm_llama", train="methods/msm_llama_loss_train.py", ev="methods/msm_llama_eval.py", config="configs/llama31_8b_msm.yaml", tag="mlq_{l}_T{T}_s{s}"),
           "gemma": dict(root="msm_gemma", train="methods/msm_gemma_loss_train.py", ev="methods/msm_gemma_eval.py", config="configs/gemma3_4b_msm.yaml", tag="gmq_{l}_T{T}_s{s}")}[m]
    root = f"{C}/{cfg['root']}"
    L = {"ce": "B", "soft": "E", "any": "M"}
    out = []

    def tag(path, T, s):
        if m == "qwen" and T == 1 and path in ("ce", "soft"):
            return f"msm_{L[path]}_s{s}"                          # the Qwen T1 CE / soft-kappa models of the first MS MARCO loss study
        return cfg["tag"].format(l=L[path], T=T, s=s)

    def ckpt(path, T, s):
        if m == "qwen" and T == 1 and path in ("ce", "soft"):
            d = "msm_loss" if path == "ce" else "msm_loss2"
            return f"{C}/{d}/checkpoints/{tag(path, T, s)}/ckpt_final.pt"
        return f"{root}/train/checkpoints/{tag(path, T, s)}/ckpt_final.pt"
    if stage in ("prep", "all") and m == "qwen":
        out.append("python scripts/restore_data.py msmarco-cache      # passage cache of the Qwen3-4B trainers (needs the pinned Qwen3-4B tokenizer)")
    if stage in ("train", "all"):
        for T in instr:
            for s in seeds:
                for path in ("ce", "soft", "any"):
                    if path not in paths:
                        continue
                    custom = f" --custom_instruction_file {CUSTOM}"
                    if m == "qwen" and T == 1 and path == "ce":
                        out.append(f"{g} methods/msm_loss_train.py --loss B --seed {s} --tag {tag(path, T, s)} --config {cfg['config']} --out_root {C}/msm_loss --local_files_only")
                    elif m == "qwen" and T == 1 and path == "soft":
                        out.append(f"{g} methods/msm_loss_train2.py --loss E --seed {s} --tag {tag(path, T, s)} --config {cfg['config']} --out_root {C}/msm_loss2 --local_files_only")
                    elif m == "qwen":
                        out.append(f"{g} {cfg['train']} --loss {L[path]} --instr T{T} --seed {s} --tag {tag(path, T, s)} --config {cfg['config']} --out_root {root}/train{custom} --local_files_only")
                    else:
                        out.append(f"{g} {cfg['train']} --loss {L[path]} --instr T{T} --seed {s} --tag {tag(path, T, s)} --out_root {root}/train{custom} --grad_ckpt 0 --local_files_only")
    if stage in ("eval", "all"):
        if "base" in paths:
            out.append(f"{g} {cfg['ev']} --models " + " ".join(f"Base:T{T}:BASE" for T in instr) + f" --out_dir {root}/eval --custom_instruction_file {CUSTOM} --local_files_only")
        for T in instr:
            models = [f"{tag(p, T, s)}:T{T}:{ckpt(p, T, s)}" for p in ("ce", "soft", "any") if p in paths for s in seeds]
            if models:
                out.append(f"{g} {cfg['ev']} --models {' '.join(models)} --out_dir {root}/eval --custom_instruction_file {CUSTOM} --local_files_only")
    return out


def pilot245(gpu):
    g = f"CUDA_VISIBLE_DEVICES={gpu} python"
    runs = [("Q245S03", "mix", "--lam 0.3"), ("Q245S3", "mix", "--lam 3"), ("Q245A5", "mixany", "--lam 1 --alpha 5"), ("Q245A20", "mixany", "--lam 1 --alpha 20"), ("Q245B", "mixbce", "--lam 1")]
    out = ["# Qwen3-4B, ESCI, original T1, seed 0, final 625 update; the CE / CE+soft kappa / CE+AnyLoss-style kappa (lambda 1, alpha 10) references are the T1 seed-0 runs of the main commands"]
    for pre, loss, a in runs:
        out.append(f"{g} methods/esci_loss_train_245.py --loss {loss} {a} --final_only --instr T1 --seed 0 --instr_file {MAN} --out_root {C}/esci245/train --tag_prefix {pre} --local_files_only")
        t = f"{pre}_{loss}_T1_s0"
        out.append(f'echo \'{{"{t}_step625": "{C}/esci245/train/checkpoints/{t}/ckpt_step625.pt"}}\' > {C}/esci245/ckpt_map_{pre}.json')
        for tagname, rows in (("sel", "sel_rows.jsonl"), ("test300", "test300_rows.jsonl")):
            out.append(f"{g} methods/esci_eval.py --jobs {t}_step625:T1 --instr_file {MAN} --rows {AUD}/{rows} --tag {tagname} --ckpt_map {C}/esci245/ckpt_map_{pre}.json --out_dir {C}/esci245/eval --local_files_only")
    out.append("python analysis/esci245_analysis.py --dir <directory with the 16 esciEval_*_step625__T1__{sel,test300}.json.gz files of the 8 conditions> --out <output dir>")
    return out


def gemma_fp32(gpu):
    g = f"CUDA_VISIBLE_DEVICES={gpu} python"
    out = [f"# Appendix: FP32 re-evaluation of the representative Gemma-3-4B-it checkpoints (ESCI and MS MARCO, T1 and T7, seed 0, CE / soft / Any); the manifest lists the checkpoints and their stored thresholds",
           "# edit the 'path' entries of cpmr_experiment/gemma231/sample_manifest.json if the checkpoints were trained elsewhere"]
    for ds in ("ESCI", "MS"):
        for T in (1, 7):
            for loss in ("CE", "SOFT", "ANY"):
                out.append(f"{g} methods/gemma231_audit.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds {ds} --T {T} --loss {loss} --out cpmr_experiment/gemma231/out")
                out.append(f"{g} methods/gemma234_fp32_eval.py --manifest cpmr_experiment/gemma231/sample_manifest.json --ds {ds} --T {T} --loss {loss} --mode full --out cpmr_experiment/gemma234/out")
    return out


def rng(s):
    r = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            r += list(range(int(a), int(b) + 1))
        else:
            r.append(int(part))
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qwen", "llama", "gemma"])
    ap.add_argument("--dataset", choices=["esci", "msmarco"])
    ap.add_argument("--paths", default="base,ce,soft,any,po-base,po-ce,po-soft")
    ap.add_argument("--instr", default="1-7")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--stage", default="all", choices=["prep", "train", "eval", "all"])
    ap.add_argument("--gpu", default="0", help="physical GPU index to expose (CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pilot245", action="store_true")
    ap.add_argument("--gemma-fp32", action="store_true")
    a = ap.parse_args()
    paths, instr, seeds = set(a.paths.split(",")), rng(a.instr), rng(a.seeds)
    if a.pilot245:
        print("\n".join(pilot245(a.gpu)))
    elif a.gemma_fp32:
        print("\n".join(gemma_fp32(a.gpu)))
    elif a.all:
        d = Path(a.out or "experiments/commands")
        d.mkdir(parents=True, exist_ok=True)
        for m in ("qwen", "llama", "gemma"):
            for ds, fn in (("esci", esci), ("msmarco", msm)):
                cmds = fn(m, paths, instr, seeds, "all", a.gpu)
                (d / f"{m}_{ds}.sh").write_text("#!/bin/bash\n# generated by experiments/make_commands.py; run from the repository root\nset -e\n" + "\n".join(cmds) + "\n", encoding="utf-8", newline="\n")
                print(f"{m}_{ds}.sh: {len(cmds)} lines")
        (d / "pilot245.sh").write_text("#!/bin/bash\nset -e\n" + "\n".join(pilot245(a.gpu)) + "\n", encoding="utf-8", newline="\n")
        (d / "gemma_fp32.sh").write_text("#!/bin/bash\nset -e\n" + "\n".join(gemma_fp32(a.gpu)) + "\n", encoding="utf-8", newline="\n")
    else:
        assert a.model and a.dataset, "give --model and --dataset (or --all)"
        fn = esci if a.dataset == "esci" else msm
        print("\n".join(fn(a.model, paths, instr, seeds, a.stage, a.gpu)))
