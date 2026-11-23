"""CPU-only smoke checks of the reproduction package (no GPU, no model download, no network).
    python checks/run_checks.py [--verbose]
Checks: syntax of every Python file; configs; manifest checksums and counts; frozen instructions; the APO-CF meta-prompt builder against the recorded prompts; the threshold tie rule; kappa / AUC on known cases;
the loss identities (r = 0.5 soft-kappa, BCE); the command generator (every referenced script, config and instruction file exists, tags are unique); absolute paths; restored data (if present).
A skipped check is reported as SKIP; GPU training / scoring is NOT exercised here."""
import argparse
import csv
import glob
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "methods"))
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("--verbose", action="store_true")
a = ap.parse_args()
res = []


def check(name, ok, detail=""):
    res.append((name, "OK" if ok else "FAIL"))
    print(("OK    " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail and (a.verbose or not ok) else ""))


def skip(name, why):
    res.append((name, "SKIP"))
    print(f"SKIP  {name}  [{why}]")


# 1. syntax
bad = []
for f in ROOT.rglob("*.py"):
    try:
        compile(f.read_text(encoding="utf-8"), str(f), "exec")
    except Exception as e:  # noqa: BLE001
        bad.append((str(f.relative_to(ROOT)), str(e)[:80]))
check("python syntax of all files", not bad, str(bad[:3]))

# 2. configs
try:
    import yaml
    exp = {"qwen3_4b_esci.yaml": ("Qwen/Qwen3-4B", 766), "gemma3_4b_esci.yaml": ("google/gemma-3-4b-it", 704), "llama31_8b_esci.yaml": ("meta-llama/Llama-3.1-8B-Instruct", 0),
           "qwen3_4b_msm_loss.yaml": ("Qwen/Qwen3-4B", 384), "llama31_8b_msm.yaml": ("meta-llama/Llama-3.1-8B-Instruct", 384), "gemma3_4b_msm.yaml": ("google/gemma-3-4b-it", 384)}
    ok = True
    for fn, (mn, ml) in exp.items():
        c = yaml.safe_load(open(ROOT / "configs" / fn, encoding="utf-8"))
        ok &= c["model_name"] == mn and c["lora"]["r"] == 8 and c["lora"]["alpha"] == 16 and abs(c["lora"]["dropout"] - 0.05) < 1e-12 and abs(c["optim"]["lr"] - 2e-5) < 1e-12 and c["data"]["max_length"] == ml
    check("configs (model id, LoRA 8/16/0.05, lr 2e-5, input limit)", ok)
except ImportError:
    skip("configs", "PyYAML missing")

# 3. manifests
M = ROOT / "data" / "manifests"
sums = {}
for l in open(M / "SHA256SUMS", encoding="utf-8"):
    h, n = l.split()
    sums[n] = h
check("manifest SHA-256 sums", all(hashlib.sha256((M / n).read_bytes()).hexdigest() == h for n, h in sums.items()) and len(sums) >= 11)
em = json.load(open(M / "esci_manifest.json", encoding="utf-8"))
ok = True
for s, meta in em["sets"].items():
    rows = list(csv.DictReader(open(M / f"esci_rows_{s}.csv", encoding="utf-8")))
    ok &= len(rows) == meta["rows"] and sum(r["esci"] == "E" for r in rows) == meta["E"] and len({(r["query_id"], r["product_id"]) for r in rows}) == len(rows)
check("ESCI manifests: row counts, Relevant counts, unique (query, product)", ok)
check("ESCI sizes of Table 1 (10,000 / 3,879 / 4,047 / 6,078; 5,000 / 200 / 200 / 300 queries)", [em["sets"][s]["rows"] for s in ("sft2", "diag", "sel", "test300")] == [10000, 3879, 4047, 6078] and [em["sets"][s]["queries"] for s in ("sft2", "diag", "sel", "test300")] == [5000, 200, 200, 300])
check("ESCI independent calibration set: 200 queries / 3,939 rows / 1,682 Relevant", [em["sets"]["calib"][k] for k in ("queries", "rows", "E")] == [200, 3939, 1682])
# warm-up: 10% of the 625 updates; MS MARCO trainers round(0.1 x 625) = 62 (warmup_ratio of the configs), ESCI trainers ceil(0.1 x 625) = 63 (fixed in the trainers)
import math as _m
import re as _re
_esci_tr = ["esci_train.py", "esci_llama_train.py", "esci_loss_train_245.py", "esci_loss_train_384.py", "esci_llama_loss_train_mix.py", "esci_gemma_loss_train_mix.py"]
_ok_w = round(0.1 * 625) == 62 and _m.ceil(0.1 * 625) == 63
_ok_w &= all("ceil(0.1 * B)" in (ROOT / "methods" / f_).read_text(encoding="utf-8") for f_ in _esci_tr)
_ok_w &= all('round(cfg["optim"]["warmup_ratio"] * total' in (ROOT / "methods" / f_).read_text(encoding="utf-8") for f_ in ("msm_llama_loss_train.py", "msm_gemma_loss_train.py", "train_apocf_online.py"))
_ok_w = bool(_ok_w) and all(bool(_re.search(r"warmup_ratio:\s*0\.1", (ROOT / "configs" / c_).read_text(encoding="utf-8"))) for c_ in ("qwen3_4b_msm_loss.yaml", "llama31_8b_msm.yaml", "gemma3_4b_msm.yaml"))
check("warm-up: MS MARCO 62 updates (round of 0.1 x 625 from warmup_ratio 0.1), ESCI 63 (ceil)", _ok_w)
q_ids = {s: {r["query_id"] for r in csv.DictReader(open(M / f"esci_rows_{s}.csv", encoding="utf-8"))} for s in em["sets"]}
names = list(q_ids)
check("ESCI query ids disjoint across splits", all(not (q_ids[x] & q_ids[y]) for i, x in enumerate(names) for y in names[i + 1:]))
mm = json.load(open(M / "msmarco_manifest.json", encoding="utf-8"))
tp = list(csv.DictReader(open(M / "msmarco_train_pairs.csv", encoding="utf-8")))
byq = {}
for r in tp:
    byq.setdefault(r["query_id"], []).append(r["label"])
check("MS MARCO train manifest: 10,000 rows, 5,000 queries, one positive and one negative each", len(tp) == 10000 and len(byq) == 5000 and all(sorted(v) == ["Irrelevant", "Relevant"] for v in byq.values()))
for nm in ("val200", "diag200"):
    pr = list(csv.DictReader(open(ROOT / mm[nm]["manifest"], encoding="utf-8")))
    qq = {}
    for r in pr:
        qq.setdefault(r["query_id"], []).append(r["label"])
    check(f"MS MARCO {nm} manifest: {mm[nm]['rows']} rows, {mm[nm]['queries']} queries, one positive and one negative each", len(pr) == mm[nm]["rows"] == 400 and len(qq) == mm[nm]["queries"] == 200 and all(sorted(v) == ["Irrelevant", "Relevant"] for v in qq.values()))
sets_q = {n_: {r["query_id"] for r in csv.DictReader(open(ROOT / mm[n_]["manifest"], encoding="utf-8"))} for n_ in ("val200", "diag200")}
sets_q["train"] = set(byq)
check("MS MARCO query ids disjoint: train / val200 / diag200", not (sets_q["train"] & sets_q["val200"]) and not (sets_q["train"] & sets_q["diag200"]) and not (sets_q["val200"] & sets_q["diag200"]))
jp = list(csv.DictReader(open(M / "trec_dl_judged_pairs.csv", encoding="utf-8")))
check("TREC-DL judged pairs: 2019 42 q / 1,102, 2020 54 q / 1,432", all(len([r for r in jp if r["year"] == y]) == mm["years"][y]["judged_pairs"] and len({r["query_id"] for r in jp if r["year"] == y}) == mm["years"][y]["queries"] for y in ("2019", "2020")) and (mm["years"]["2019"]["judged_pairs"], mm["years"]["2020"]["judged_pairs"]) == (1102, 1432))

# 4. instructions
man = json.load(open(ROOT / "cpmr_experiment/esci_t/esci_t_manifest.json", encoding="utf-8"))["entries"]
check("ESCI T1-T7: seven instructions, sha256_full matches the text, one-label suffix", [e["display_id"] for e in man] == [f"T{k}" for k in range(1, 8)] and all(hashlib.sha256(e["full"].encode()).hexdigest() == e["sha256_full"] and e["full"].endswith('Output exactly one label: "Relevant" or "Irrelevant".') for e in man))
im = json.load(open(ROOT / "cpmr_experiment/instructions_v2/manifest/instructions_v2.json", encoding="utf-8"))
cu = json.load(open(ROOT / "methods/custom_instructions_T6_T7.json", encoding="utf-8"))
check("MS MARCO instructions: T1-T5 in the manifest, T6 / T7 in the custom file", {"T6", "T7"} <= set(cu) and "T1" in json.dumps(im))

# 5. APO-CF prompt builder against the recorded exchanges
try:
    import apocf_prompt as AP
    n = ok_n = 0
    full = {e["display_id"]: e["full"] for e in man}
    for m in ("esci_t", "esci_llama"):
        for T in [f"T{k}" for k in range(1, 8)]:
            c = json.load(open(ROOT / f"cpmr_experiment/{m}/p1_{T}/p1_confusion.json"))
            for r in json.load(open(ROOT / f"cpmr_experiment/{m}/p1_{T}/p1_exchange.json", encoding="utf-8")):
                p = AP.build_apocf_prompt(full[T], c["tn"], c["fp"], c["fn"], c["tp"], domain="product", level="full", constraint_variant=r["variant"])
                n += 1
                ok_n += p == r["meta_prompt"] and hashlib.sha256(p.encode()).hexdigest() == r["meta_prompt_sha256"]
    check("APO-CF meta-prompts rebuilt = recorded: ESCI, Qwen and Llama, T1-T7, three variants (42)", n == 42 and ok_n == 42, f"{ok_n}/{n}")
    n = ok_n = 0
    for m in ("qwen", "llama"):
        for k in range(1, 8):
            d = ROOT / f"cpmr_experiment/msm_po/{m}/p1_T{k}"
            rr_ = json.load(open(d / "p1_result.json", encoding="utf-8")); c = json.load(open(d / "p1_confusion.json"))
            for r in json.load(open(d / "p1_exchange.json", encoding="utf-8")):
                p = AP.build_apocf_prompt(rr_["original_full"], c["tn"], c["fp"], c["fn"], c["tp"], domain="passage", level="full", constraint_variant=r["variant"])
                n += 1
                ok_n += p == r["meta_prompt"] and hashlib.sha256(p.encode()).hexdigest() == r["meta_prompt_sha256"]
    check("APO-CF meta-prompts rebuilt = recorded: MS MARCO, Qwen and Llama, T1-T7, three variants (42)", n == 42 and ok_n == 42, f"{ok_n}/{n}")
    # PO selection summary: changed / kept instructions of the four model-dataset combinations (the paper's PO scope: Qwen and Llama, ESCI and MS MARCO, T1-T7; no Gemma)
    exp = {("esci_t", ""): {1, 2, 3, 4, 5, 6, 7}, ("esci_llama", ""): {2, 3, 5, 7}, ("msm_po", "qwen"): {1, 2, 3, 4, 5, 6, 7}, ("msm_po", "llama"): {2, 3}}
    ok_po = True
    for (root_, sub), want in exp.items():
        got = {k for k in range(1, 8) if json.load(open(ROOT / f"cpmr_experiment/{root_}/{sub}/p1_T{k}/p1_result.json".replace("//", "/"), encoding="utf-8"))["selection"]["index"] != 0}
        ok_po &= got == want
    check("PO selections: changed instructions = Qwen ESCI T1-T7, Llama ESCI T2/T3/T5/T7, Qwen MS MARCO T1-T7, Llama MS MARCO T2/T3", ok_po)
except Exception as e:  # noqa: BLE001
    check("APO-CF meta-prompt builder", False, str(e)[:100])

# 6. threshold tie rule and metrics
try:
    import numpy as np
    import esci204_common as C
    m = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]); y = np.array([False, False, False, True, True, True])
    t, k = C.best_threshold(m, y)
    check("threshold search: separable scores -> kappa 1; t = 0 already separates them and wins the tie over the midpoint 0.25", abs(k - 1.0) < 1e-12 and t == 0.0, f"{t} {k}")
    m2 = np.array([-1.0, 1.0]); y2 = np.array([False, True])
    t2, _ = C.best_threshold(m2, y2)
    check("threshold tie rule: equal kappa -> t = 0 preferred", t2 == 0.0, str(t2))
    m3 = np.array([1.0, 2.0, 3.0, 4.0]); y3 = np.array([True, False, True, False])
    best3 = C.best_threshold(m3, y3)
    cand = [(round(C.kap(int(((m3 > t) & y3).sum()), int(((m3 > t) & ~y3).sum()), int(((m3 <= t) & y3).sum()), int(((m3 <= t) & ~y3).sum())), 12), t) for t in np.concatenate([[0.0], (m3[:-1] + m3[1:]) / 2, [m3.min() - 1, m3.max() + 1]])]
    mx = max(k_ for k_, _ in cand)
    ties = sorted((0 if t_ == 0.0 else 1, abs(t_), t_) for k_, t_ in cand if k_ == mx)
    check("threshold tie rule: 0, then smaller |t|, then smaller t (brute force)", ties[0][2] == best3[0], f"{ties[0]} vs {best3}")
    rng = np.random.default_rng(0)
    same = 0
    import esci245_analysis as E
    for _ in range(100):
        mm_ = rng.normal(size=60) * 2; yy = rng.random(60) < 0.45
        if yy.all() or (~yy).all():
            continue
        same += abs(E.best_threshold(mm_, yy) - C.best_threshold(mm_, yy)[0]) < 1e-12
    check("two implementations of the threshold search agree on 100 random sets", same >= 99, str(same))
    check("kappa on a known confusion matrix (40, 10, 10, 40 -> 0.6)", abs(C.kap(40, 10, 10, 40) - 0.6) < 1e-12)
    check("AUC on known cases (perfect = 1, reversed = 0, ties = 0.5)", abs(C.auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1], bool)) - 1) < 1e-12 and abs(C.auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1], bool))) < 1e-12 and abs(C.auc(np.array([1.0, 1.0, 1.0, 1.0]), np.array([0, 0, 1, 1], bool)) - 0.5) < 1e-12)
except ImportError as e:
    skip("threshold / metric checks", str(e))

# 7. loss identities
try:
    import torch
    import esci_loss_lib as L
    g = torch.Generator().manual_seed(0)
    mv = (torch.randn(16, generator=g) * 3).float(); yv = torch.tensor([1.0] * 8 + [0.0] * 8)
    lk = L.loss_from_m("softk", mv, yv)
    u = torch.sigmoid(mv)
    check("r = 0.5: soft-kappa loss = 2 x mean |y - u|", abs(float(lk) - float(2 * (yv - u).abs().mean())) < 1e-6, f"{float(lk)}")
    check("BCEWithLogits = softplus(m) - y m", float((torch.nn.functional.binary_cross_entropy_with_logits(mv, yv, reduction="none") - (torch.nn.functional.softplus(mv) - yv * mv)).abs().max()) < 1e-5)
    g1, _ = L.grad_wrt_m("anyk", mv, yv)
    mm2 = mv.clone().double().requires_grad_(True)
    l2 = L.loss_from_m("anyk", mm2, yv.double())
    (g2,) = torch.autograd.grad(l2, mm2)
    check("AnyLoss-style kappa gradient (alpha 10) in FP32 = FP64 autograd", float((g1.double() - g2).abs().max()) < 1e-4)
except ImportError as e:
    skip("loss identities", str(e))

# 8. command generator
try:
    sys.path.insert(0, str(ROOT / "experiments"))
    import make_commands as MC
    allc = []
    for mdl in ("qwen", "llama", "gemma"):
        for fn in (MC.esci, MC.msm):
            allc += [c for c in fn(mdl, {"base", "ce", "soft", "any", "po-base", "po-ce", "po-soft"}, list(range(1, 8)), [0, 1, 2], "all", "0") if not c.startswith("#")]
    scripts = {m_ for c in allc for m_ in re.findall(r"(?:methods|scripts|analysis)/[A-Za-z0-9_]+\.py", c)}
    check("command generator: every referenced script exists", all((ROOT / s).exists() for s in scripts), str([s for s in scripts if not (ROOT / s).exists()]))
    cfg_ok = all((ROOT / c).exists() for c in {m_ for c in allc for m_ in re.findall(r"configs/[A-Za-z0-9_]+\.yaml", c)})
    check("command generator: configs exist", cfg_ok)
    tags = [m_ for c in allc if " --tag " in c for m_ in re.findall(r"--tag (\S+)", c) if not m_.startswith(("sel", "test"))]
    check("command generator: MS MARCO training tags unique", len(tags) == len(set(tags)), f"{len(tags)} tags")
    n_train = sum(1 for c in allc if "loss_train" in c or "esci_train.py" in c or "esci_llama_train.py" in c)
    check("command generator: number of training runs", n_train > 300, str(n_train))
    n_po = {}
    for mdl in ("qwen", "llama", "gemma"):
        for ds_, fn in (("esci", MC.esci), ("msmarco", MC.msm)):
            cs = [c for c in fn(mdl, {"po-base", "po-ce", "po-soft"}, list(range(1, 8)), [0, 1, 2], "train", "0") if not c.startswith("#")]
            n_po[(mdl, ds_)] = sum("_APOCF" in c for c in cs)
    check("command generator: PO training runs (Qwen ESCI 42, Llama ESCI 24, Qwen MS MARCO 42, Llama MS MARCO 12, none for Gemma)", n_po == {("qwen", "esci"): 42, ("qwen", "msmarco"): 42, ("llama", "esci"): 24, ("llama", "msmarco"): 12, ("gemma", "esci"): 0, ("gemma", "msmarco"): 0}, str(n_po))
    vcs = [c for c in MC.msm("qwen", {"base", "ce", "soft", "any", "po-base", "po-ce", "po-soft"}, list(range(1, 8)), [0, 1, 2], "val", "0") if "msm_val_eval" in c]
    check("command generator: val200 scoring commands for the MS MARCO models", len(vcs) >= 8, str(len(vcs)))
    pil = MC.pilot245("0")
    check("pilot generator: 5 training runs", sum("esci_loss_train_245.py" in c for c in pil) == 5)
except Exception as e:  # noqa: BLE001
    check("command generator", False, str(e)[:120])

# 9. absolute paths / secrets
pat = re.compile(r"(/mnt/|/home/[a-z]|/data/[a-z]|[A-Z]:\\[A-Za-z]|GPU-[0-9a-f]{8}-|sk-[A-Za-z0-9]{20})")
hits = []
for f in ROOT.rglob("*"):
    if f.is_file() and f.suffix in {".py", ".sh", ".md", ".yaml", ".json", ".txt", ".csv"} and "checks" not in f.parts and f.name != "run_checks.py":
        t = f.read_text(encoding="utf-8", errors="ignore")
        for mt in pat.finditer(t):
            hits.append((str(f.relative_to(ROOT)), mt.group(0)))
check("no absolute server paths, device ids or key patterns in text files", not hits, str(hits[:5]))

# 10. restored data (optional)
if (ROOT / "cpmr_experiment/esci/audit/sft2_rows.jsonl").exists() and (ROOT / "data/msmarco_train_v2_5k/pairs_text_basic_5000.jsonl").exists():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/verify_data.py")], capture_output=True, text=True)
    check("restored data verified by scripts/verify_data.py", r.returncode == 0, r.stdout[-300:])
else:
    skip("restored data", "run scripts/restore_data.py first")

fails = [n for n, s in res if s == "FAIL"]
print(f"\n{sum(s == 'OK' for _, s in res)} OK, {sum(s == 'SKIP' for _, s in res)} SKIP, {len(fails)} FAIL")
sys.exit(1 if fails else 0)
