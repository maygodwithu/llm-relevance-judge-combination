"""Chapter 6.2 error-transition panels (experiment log 239): six independent panels, ESCI CE -> CE+soft kappa, T1-T7 x 3 seeds, default / selected thresholds, from shared/238 err62_results.json only (no recomputation).
   python errfig_plot.py --res <shared/238/results/err62_results.json> --out <dir>"""
import argparse, hashlib, json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--res", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
raw = open(a.res, "rb").read()
R = json.loads(raw)["sets"]
MODELS = [("qwen", "ESCI-Qwen", "Qwen3-4B"), ("llama", "ESCI-Llama", "Llama-3.1-8B-Instruct"), ("gemma", "ESCI-Gemma", "Gemma-3-4B-it")]
MODES = ("default", "selected")
GR = ("E", "S", "C", "I")
BLUE, ORANGE = "#0072B2", "#D55E00"

data = {}
for key, sid, nm in MODELS:
    for mode in MODES:
        cs = [c for c in R[sid]["comparisons"] if c["comparison"] == f"CE->SOFT [{mode}]" and c["scope"] == "main T1-T7" and c["kind"] == "orig"]
        assert len(cs) == 1, (sid, mode, len(cs))
        c = cs[0]
        assert c["instructions"] == list(range(1, 8)) or len(c["instructions"]) == 7, c["instructions"]
        assert len(c["seeds"]) == 3
        rows = {}
        for g in GR:
            x = c["groups"][g]
            wr, rw, net = x["WR"]["point"], x["RW"]["point"], x["net"]["point"]
            assert abs((wr - rw) - net) < 1e-12, (sid, mode, g)
            rows[g] = dict(N=x["N"], WR=wr, RW=rw, net=net, net_ci95=x["net"]["ci95"], WR_ci95=x["WR"]["ci95"], RW_ci95=x["RW"]["ci95"], valid_draws=x["valid_draws"])
        data[(key, mode)] = dict(set=sid, model=nm, mode=mode, comparison=c["comparison"], scope=c["scope"], instructions=c["instructions"], seeds=c["seeds"], groups=rows)

# direction checks (data-mapping sanity, not a value criterion)
chk = []
d = lambda k, m, g: data[(k, m)]["groups"][g]["net"]
chk.append(("default Qwen E net > 0", d("qwen", "default", "E") > 0)); chk.append(("default Llama E net > 0", d("llama", "default", "E") > 0))
chk.append(("default Gemma E net < 0", d("gemma", "default", "E") < 0))
for g in ("S", "C", "I"):
    chk.append((f"default Gemma {g} net > 0", d("gemma", "default", g) > 0))
for n, ok in chk:
    assert ok, "direction check failed: " + n

lo = min(min(-v["RW"] * 100 for v in e["groups"].values()) for e in data.values())
lo = min(lo, min(v["net_ci95"][0] * 100 for e in data.values() for v in e["groups"].values()))
hi = max(max(v["WR"] * 100 for v in e["groups"].values()) for e in data.values())
hi = max(hi, max(v["net_ci95"][1] * 100 for e in data.values() for v in e["groups"].values()))
STEP = 5
XLO, XHI = STEP * math.floor(lo / STEP), STEP * math.ceil(hi / STEP)
print("shared x range", XLO, XHI, "(raw", round(lo, 3), round(hi, 3), ")")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
W_MM, H_MM = 72.0, 56.0
for (key, mode), e in data.items():
    fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4), dpi=300, facecolor="white")
    ax = fig.add_axes([0.105, 0.2, 0.84, 0.79])
    ax.set_xlim(XLO, XHI); ax.set_ylim(3.62, -0.62)
    ax.xaxis.grid(True, color="#e6e6e6", lw=0.4); ax.set_axisbelow(True)
    ax.axvline(0, color="#444444", lw=0.5, zorder=1)
    for i, g in enumerate(GR):
        v = e["groups"][g]
        ax.barh(i + 0.1, v["WR"] * 100, height=0.34, color=BLUE, zorder=2, lw=0)
        ax.barh(i + 0.1, -v["RW"] * 100, height=0.34, color=ORANGE, zorder=2, lw=0)
        ci = v["net_ci95"]
        yy = i - 0.27
        ax.plot([ci[0] * 100, ci[1] * 100], [yy, yy], color="black", lw=0.9, zorder=3, solid_capstyle="butt")
        ax.plot([v["net"] * 100], [yy], "o", color="black", ms=3.2, zorder=4)
    ax.set_yticks(range(4)); ax.set_yticklabels(GR)
    ax.set_xticks(range(XLO, XHI + 1, STEP))
    ax.tick_params(length=2.5, width=0.5, pad=2)
    ax.set_xlabel("% of gold class (net: pp)", labelpad=2)
    ax.set_ylabel("Gold class", labelpad=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    base = f"error_ce_soft_{key}_{mode}"
    fig.savefig(out / (base + ".pdf"), facecolor="white")
    fig.savefig(out / (base + ".png"), dpi=400, facecolor="white")
    plt.close(fig)

meta = dict(source_file=Path(a.res).name, source_sha256=hashlib.sha256(raw).hexdigest(), x_range=[XLO, XHI], x_step=STEP, panel_mm=[W_MM, H_MM],
            direction_checks=[dict(name=n, ok=ok) for n, ok in chk], panels=[])
for (key, mode), e in data.items():
    meta["panels"].append(dict(file=f"error_ce_soft_{key}_{mode}", **e))
json.dump(meta, open(out / "errfig_plot_data.json", "w", indent=1) if False else open(out / "errfig_plot_data.json", "w"), indent=1)
md = ["| model | threshold | group | N | W→R % | R→W % | net pp | net 95% CI |", "|---|---|---|---|---|---|---|---|"]
for (key, mode), e in data.items():
    for g in GR:
        v = e["groups"][g]
        md.append(f"| {e['model']} | {mode} | {g} | {v['N']} | {100*v['WR']:.2f} | {100*v['RW']:.2f} | {100*v['net']:+.2f} | [{100*v['net_ci95'][0]:+.2f}, {100*v['net_ci95'][1]:+.2f}] |")
open(out / "errfig_plot_data.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
print("done", len(data), "panels")
