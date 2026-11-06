"""Chapter 6.3 threshold-gain panels (experiment log 243): nine independent point/interval panels, per-draw delta kappa_b = kappa_test(t_b) - kappa_test(t=0) of Base / CE / CE+soft kappa / CE+AnyLoss-like kappa (T1-T7, 3-seed mean, 500 validation-query resamples), from shared/242 err63_draws.json only (ESCI: reuse_rng1 draws, MS MARCO: new_rng63); no recomputation of thresholds.
   python thrfig_plot.py --draws <err63_draws.json> --res <err63_results.json> --out <dir>"""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--draws", required=True); ap.add_argument("--res", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
rawd, rawr = open(a.draws, "rb").read(), open(a.res, "rb").read()
DR, RS = json.loads(rawd), json.loads(rawr)["sets"]
ROWS = [("esci", "ESCI", "ESCI-{m}"), ("dl2019", "MS", "MS-{m}-2019"), ("dl2020", "MS", "MS-{m}-2020")]
MODELS = [("qwen", "Qwen"), ("llama", "Llama"), ("gemma", "Gemma")]
CONDS = [("BASE", "Base", "#555555"), ("CE", "CE", "#0072B2"), ("SOFT", "Soft", "#D55E00"), ("ANY", "Any", "#009E73")]
panels = {}
for row, _, tmpl in ROWS:
    for mk, mname in MODELS:
        sid = tmpl.format(m=mname)
        spec = "reuse_rng1" if sid.startswith("ESCI") else "new_rng63"
        d = DR[sid][spec]
        sp = RS[sid]["specs"][spec]
        assert sp["n_valid_draws"] == 500 and len(d["valid_draw_index"]) == 500 and d["valid_draw_index"] == sorted(d["valid_draw_index"]) == list(range(500))
        items = {}
        for c, _, _ in CONDS:
            x = np.array(d["delta_tb_minus_t0"][c], float)
            assert x.shape == (500,) and np.isfinite(x).all()
            kb, k0 = np.array(d["kappa_tb"][c], float), d["kappa_t0"][c]
            assert np.abs((kb - k0) - x).max() < 1e-12, (sid, c)             # direct check kappa(t_b) - kappa(0)
            s = sp["cond_stats"][f"T1-T7|{c}"]["delta_tb_t0"]
            mean, lo, hi = float(x.mean()), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))
            assert abs(mean - s["mean"]) < 5e-9 and abs(lo - s["p2_5"]) < 5e-9 and abs(hi - s["p97_5"]) < 5e-9, (sid, c)   # = 242 summary
            items[c] = dict(n=500, mean=mean, p2_5=lo, p97_5=hi, summary_mean=s["mean"], summary_p2_5=s["p2_5"], summary_p97_5=s["p97_5"], kappa_t0=k0)
        panels[(row, mk)] = dict(set=sid, draw_set=spec, conds=items)
# sanity (data-mapping detection only, not a value criterion)
m = lambda row, mk, c: panels[(row, mk)]["conds"][c]["mean"]
for mk, v in (("qwen", -0.0080), ("llama", -0.0061), ("gemma", -0.0028)):
    assert abs(m("esci", mk, "SOFT") - v) < 6e-4, (mk, m("esci", mk, "SOFT"))
assert m("dl2019", "qwen", "SOFT") < 0
for row, mk in (("dl2019", "llama"), ("dl2019", "gemma"), ("dl2020", "qwen"), ("dl2020", "llama"), ("dl2020", "gemma")):
    assert m(row, mk, "SOFT") > 0, (row, mk)


def nice(lo, hi):
    """shared limits = data range (incl. 0) plus 4 % padding; ticks at multiples of the smallest step giving <= 6 ticks inside the limits"""
    for step in (0.005, 0.01, 0.02, 0.025, 0.05, 0.1):
        t = np.arange(math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9) + 1) * step
        if len(t) <= 6:
            return lo, hi, step
    raise RuntimeError


AX = {}
for row, _, _ in ROWS:
    vals = [0.0] + [panels[(row, mk)]["conds"][c][k] for mk, _ in MODELS for c, _, _ in CONDS for k in ("mean", "p2_5", "p97_5")]
    lo, hi = min(vals), max(vals)
    pad = 0.04 * (hi - lo)
    AX[row] = nice(lo - pad, hi + pad)
    print(row, "shared y", AX[row], "(data", round(lo, 4), round(hi, 4), ")")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42})
W_MM, H_MM = 46.0, 40.0
for (row, mk), pn in panels.items():
    lo, hi, step = AX[row]
    fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4), dpi=300, facecolor="white")
    ax = fig.add_axes([0.27, 0.15, 0.70, 0.82])
    ax.set_ylim(lo, hi); ax.set_xlim(-0.6, 3.6)
    ax.yaxis.grid(True, color="#e6e6e6", lw=0.4); ax.set_axisbelow(True)
    ax.axhline(0, color="#222222", lw=0.9, zorder=2)
    for i, (c, lab, col) in enumerate(CONDS):
        v = pn["conds"][c]
        ax.plot([i, i], [v["p2_5"], v["p97_5"]], color=col, lw=1.1, zorder=3, solid_capstyle="butt")
        for y_ in (v["p2_5"], v["p97_5"]):
            ax.plot([i - 0.14, i + 0.14], [y_, y_], color=col, lw=1.1, zorder=3, solid_capstyle="butt")
        ax.plot([i], [v["mean"]], "o", color="black", ms=3.0, mfc=col, mec="black", mew=0.5, zorder=4)
    ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l, _ in CONDS])
    ticks = np.round(np.arange(math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9) + 1) * step, 6)
    ax.set_yticks(ticks)
    ax.set_yticklabels([("%.3f" if step < 0.01 else "%.2f") % t if abs(t) > 1e-12 else "0" for t in ticks])
    ax.tick_params(length=2, width=0.4, pad=1.5)
    ax.set_ylabel(r"$\Delta\kappa$ (test, $t_b$ vs. $t{=}0$)", labelpad=1.5, fontsize=6.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    base = f"threshold_gain_{row}_{mk}"
    fig.savefig(out / (base + ".pdf"), facecolor="white")
    fig.savefig(out / (base + ".png"), dpi=400, facecolor="white")
    plt.close(fig)
meta = dict(source_draws=Path(a.draws).name, source_draws_sha256=hashlib.sha256(rawd).hexdigest(), source_results=Path(a.res).name, source_results_sha256=hashlib.sha256(rawr).hexdigest(),
            panel_mm=[W_MM, H_MM], shared_y_axes={r: dict(lo=v[0], hi=v[1], step=v[2]) for r, v in AX.items()}, x_order=[l for _, l, _ in CONDS],
            panels=[dict(file=f"threshold_gain_{r}_{k}", **p) for (r, k), p in panels.items()])
json.dump(meta, open(out / "thrfig_plot_data.json", "w"), indent=1)
md = ["| panel | set | draws | condition | n | mean Δκ | 2.5% | 97.5% |", "|---|---|---|---|---|---|---|---|"]
for (r, k), p in panels.items():
    for c, lab, _ in CONDS:
        v = p["conds"][c]
        md.append(f"| {r}_{k} | {p['set']} | {p['draw_set']} | {lab} | {v['n']} | {v['mean']:+.4f} | {v['p2_5']:+.4f} | {v['p97_5']:+.4f} |")
open(out / "thrfig_plot_data.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
print("done", len(panels), "panels")
