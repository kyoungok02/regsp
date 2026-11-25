"""Figure S6. Original versus corrected pre-grasp trajectory (handover_block). Real recorded traces only (ACL_TRACE development runs; privileged object poses are used for analysis panels only, never by the policy).
Development comparison: freeze3 executor (label 'original') vs CLC_APPROACH=2 ('corrected') on the same development seeds, blocks D/E/F, restricted to seeds present in BOTH runs.
Frozen final (fresh seeds 8200000+/8700000+) appears only in panel C, labelled separately.  usage: python make_figure_S6_pregrasp_trajectory.py"""
import csv, glob, json, math, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os; N = os.environ["REGSP_RESEARCH_TREE"]; OUT = os.environ.get("REGSP_FIG_OUT", f"{N}/figures_supp"); STEM = "Figure_S6_pregrasp_trajectory"; TASK = "handover_block"  # N = research-tree output directory (outputs/robotwin_v21_4task_counterfactual)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from clc_diag import load as diag_load
BLOCKS = {"D": ("clc_diag", "clc_ap2"), "E": ("clcE_base", "clcE_ap2"), "F": ("clcF_base", "clcF_ap2")}; CFGS = ("rand", "clean")
def ep(lab, cfg):
    return {int(json.load(open(f))["seed"]): json.load(open(f)) for f in glob.glob(f"{N}/eval/{lab}/{cfg}/{TASK}/seed*.json")}
E = {}; common = {}
for b, (a, c) in BLOCKS.items():
    for cfg in CFGS:
        A, C = ep(a, cfg), ep(c, cfg); cs = sorted(set(A) & set(C)); common[(b, cfg)] = cs; E[("orig", b, cfg)] = {s: A[s] for s in cs}; E[("corr", b, cfg)] = {s: C[s] for s in cs}
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return 100 * (c - h) / d, 100 * (c + h) / d

# ---------------------------------------------------------------- trajectories (first approach of the first cycle)
def approach(d, mode):
    tr = d.get("trace_analysis_only", []); cyc = d.get("cycles", [])
    if not cyc or not tr: return None
    c = cyc[0]; arm = c["arm"]; G = np.array(c["g_final"], float)
    i0 = next((i for i, t in enumerate(tr) if t["tag"] == "pre0_9"), None)
    if i0 is None or i0 == 0: return None
    idx, pts, cmds = [], [], {}
    j = i0
    while j < len(tr) and tr[j]["tag"].startswith("pre0_"):
        sfx = tr[j]["tag"].split("_")[1]
        if sfx.isdigit():  # a suffixed tag (e.g. 34r) is the standoff re-attempt, not a new chunk sample: skipped
            k = int(sfx); idx.append(k); pts.append(np.array(tr[j]["tcp"][arm])[:3, 3]); cmds[k] = np.array(tr[j]["cmd"][arm])[:3, 3] if arm in tr[j]["cmd"] else None
        j += 1
    start = np.array(tr[i0 - 1]["tcp"][arm])[:3, 3]
    if mode == "corr":
        S = cmds.get(34)
        if S is None: return None
    else:
        v2 = c["v2"]; S = G - float(v2["pre_dist"]) * np.array(v2["approach"], float)  # intended standoff of the unchanged StagePlan chunk (sample 34), reconstructed from the logged stage variables
    u = (G - S)[:2]; h = np.linalg.norm(u)
    if h < 0.01: return None
    u = u / h; nrm = np.array([-u[1], u[0]])
    def rel(p): d_ = (p - G)[:2]; return float(d_ @ u) * 100, float(d_ @ nrm) * 100   # (along-axis cm, lateral cm); G at origin, +x = direction of the final approach
    P = np.array([rel(p) for p in [start] + pts]); return {"idx": [-1] + idx, "along": P[:, 0], "lat": P[:, 1], "S": rel(S), "seed": d["seed"], "success": bool(d["success"])}
A = {"orig": [], "corr": []}
for (m, b, cfg), eps in E.items():
    for s, d in eps.items():
        r = approach(d, m)
        if r: r.update({"cfg": cfg, "block": b}); A[m].append(r)

# ---------------------------------------------------------------- grasp rows (first closure of the first cycle): grounding error, hold, displacement
R = {"orig": [], "corr": []}
for b, (a, c) in BLOCKS.items():
    for cfg in CFGS:
        for m, lab in (("orig", a), ("corr", c)):
            rows, _, _ = diag_load(f"{N}/eval/{lab}/{cfg}/{TASK}"); cs = set(common[(b, cfg)])
            R[m] += [r for r in rows if r["cyc"] == 0 and r["att"] == 0 and int(r["seed"]) in cs]
nrm = lambda v: float(np.linalg.norm(v))

# ---------------------------------------------------------------- style
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2, "axes.linewidth": 0.7, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
C_O, C_C = "#9A9A9A", "#0072B2"; COL = {"orig": C_O, "corr": C_C}; NAME = {"orig": "Original (freeze3 executor)", "corr": "Corrected (lead-in → standoff → straight approach)"}
def tag(ax, s): ax.text(-0.02, 1.05, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")
fig = plt.figure(figsize=(7.8, 7.4)); fig.suptitle("Figure S6. Original versus corrected pre-grasp trajectory", fontsize=10.5, fontweight="bold", y=0.985)
gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 0.14, 1.0], hspace=0.38, wspace=1.25, left=0.075, right=0.97, top=0.9, bottom=0.085)

# A: top-down approach-aligned plane
axA = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4])]
for ax, m in zip(axA, ("orig", "corr")):
    L = A[m]
    for r in L: ax.plot(r["along"], r["lat"], "-", color=COL[m], lw=0.5, alpha=0.18, zorder=1)
    # representative: median max-lateral-deviation episode over samples before the last one
    dev = [np.max(np.abs(r["lat"][:-1])) for r in L]; rep = L[int(np.argsort(dev)[len(dev) // 2])]
    ax.plot(rep["along"], rep["lat"], "-o", color=COL[m], lw=1.4, ms=3.2, zorder=3, label="representative episode (median deviation)")
    for k, x, y in zip(rep["idx"], rep["along"], rep["lat"]): ax.annotate("start" if k < 0 else str(k), (x, y), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=4.8)
    Sx = np.array([r["S"][0] for r in L]); medS = float(np.median(Sx)); ax.plot(Sx, [0] * len(Sx), "*", color="#D55E00", ms=3.5, alpha=0.5, zorder=2, label="intended standoff (sample 34)")
    ax.plot([medS, 0], [0, 0], "k--", lw=1.0, zorder=2, label="final approach axis (standoff → target)"); ax.plot(0, 0, "kx", ms=6, mew=1.4, zorder=4, label="grasp target G")
    ax.set_xlim(-22, 4); ax.set_ylim(-14, 14); ax.set_aspect("equal", adjustable="box"); ax.set_xlabel("Along approach axis (cm, G = 0)"); ax.set_title(("Original" if m == "orig" else "Corrected") + f" (n = {len(L)} episodes)", pad=3)
    if m == "orig": ax.set_ylabel("Lateral offset in XY plane (cm)")
tag(axA[0], "A"); axL = fig.add_subplot(gs[1, 0:4]); axL.axis("off")
from matplotlib.lines import Line2D
axL.legend(handles=[Line2D([], [], color="k", marker="o", ms=3.5, lw=1.4, label="representative episode (median deviation), labels = sample index"), Line2D([], [], color="#D55E00", marker="*", ls="", ms=5, label="intended standoff (sample 34)"), Line2D([], [], color="k", ls="--", lw=1.0, label="final approach axis"), Line2D([], [], color="k", marker="x", ls="", ms=6, mew=1.4, label="grasp target G"), Line2D([], [], color="0.5", lw=0.8, label="thin lines: all episodes (grey = original, blue = corrected)")], loc="center left", ncol=3, frameon=False, fontsize=5.4, columnspacing=1.0, handlelength=1.6, bbox_to_anchor=(0.0, 0.5))
axA[0].text(0.0, 1.17, "Development comparison, matched seeds (D/E/F, Clean + Randomized)", transform=axA[0].transAxes, fontsize=6.2, style="italic")

# B: lateral deviation per executed pre-grasp sample
axB = fig.add_subplot(gs[0, 4:6]); tag(axB, "B")
for m in ("orig", "corr"):
    ks = sorted({k for r in A[m] for k in r["idx"] if k >= 0}); med, q1, q3, nn = [], [], [], []
    for k in ks:
        v = [abs(r["lat"][r["idx"].index(k)]) for r in A[m] if k in r["idx"]]; med.append(np.median(v)); q1.append(np.percentile(v, 25)); q3.append(np.percentile(v, 75)); nn.append(len(v))
    axB.fill_between(ks, q1, q3, color=COL[m], alpha=0.2, lw=0); axB.plot(ks, med, "-o", color=COL[m], ms=3.2, lw=1.4, label="Original" if m == "orig" else "Corrected")
axB.axvline(34, color="#D55E00", ls=":", lw=1.0); axB.text(33.4, 14.2, "standoff\n(34)", fontsize=5.5, color="#D55E00", va="top", ha="right")
axB.set_xlabel("Executed chunk sample index"); axB.set_ylabel("|lateral offset| from approach axis (cm)"); axB.set_xticks([9, 19, 29, 34, 39, 44, 49]); axB.legend(loc="center right", frameon=False, fontsize=5.6, bbox_to_anchor=(1.0, 0.5)); axB.set_title("Measured TCP: median and IQR", pad=3)

# C: matched success + frozen final
axC = fig.add_subplot(gs[2, 0:2]); tag(axC, "C"); pos = 0; xt, xl = [], []
def counts(m, cfg): s = sum(bool(d["success"]) for b in BLOCKS for d in E[(m, b, cfg)].values()); n = sum(len(E[(m, b, cfg)]) for b in BLOCKS); return s, n
fin = {r["config"] + "|" + r["task"]: r for r in csv.DictReader(open(f"{N}/CLC_FINAL_RESULTS.csv"))}
groups = [("Dev\nClean", lambda m: counts(m, "clean")), ("Dev\nRand.", lambda m: counts(m, "rand")), ("Final\nClean", lambda m: (int(fin["clean|handover_block"]["prev_success" if m == "orig" else "clc_success"]), int(fin["clean|handover_block"]["prev_n" if m == "orig" else "n"]))), ("Final\nRand.", lambda m: (int(fin["rand|handover_block"]["prev_success" if m == "orig" else "clc_success"]), int(fin["rand|handover_block"]["prev_n" if m == "orig" else "n"])))]
for gi, (lab, f) in enumerate(groups):
    x0 = gi + (0.4 if gi >= 2 else 0)
    for mi, m in enumerate(("orig", "corr")):
        k, n = f(m); v = 100 * k / n; lo, hi = wilson(k, n); x = x0 + (mi - 0.5) * 0.36
        axC.bar(x, v, 0.34, color=COL[m], ec="k", lw=0.5, hatch="////" if gi >= 2 else None); axC.errorbar(x, v, yerr=[[v - lo], [hi - v]], color="k", lw=0.7, capsize=1.6); axC.text(x, hi + 1.5, f"{k}/{n}", ha="center", va="bottom", fontsize=5.3, rotation=90)
    xt.append(x0); xl.append(lab)
axC.set_xticks(xt); axC.set_xticklabels(xl, fontsize=6); axC.set_ylim(0, 125); axC.set_yticks(range(0, 101, 25)); axC.set_ylabel("Native success (%)"); axC.set_xlabel("Domain")
axC.legend(handles=[Patch(fc=C_O, ec="k", lw=0.5, label="Original"), Patch(fc=C_C, ec="k", lw=0.5, label="Corrected"), Patch(fc="w", ec="k", lw=0.5, label="Dev: matched seeds"), Patch(fc="w", ec="k", lw=0.5, hatch="////", label="Frozen final: fresh seeds,\nprevious frozen vs CLC")], loc="upper left", ncol=2, frameon=False, fontsize=5.0, columnspacing=0.6, handlelength=1.2)
axC.set_title("Handover success, same seeds before/after", pad=3)

# D: hold rate versus grounding error
axD = fig.add_subplot(gs[2, 2:4]); tag(axD, "D"); bins = [0, 0.5, 1.0, 1.5, 2.5, 99]; labs = ["<0.5", "0.5–1", "1–1.5", "1.5–2.5", ">2.5"]
for mi, m in enumerate(("orig", "corr")):
    g = np.array([nrm(r["ground"]) for r in R[m]]); h = np.array([bool(r["hold"]) for r in R[m]])
    for bi in range(5):
        s = (g >= bins[bi]) & (g < bins[bi + 1]); n = int(s.sum()); k = int(h[s].sum()); x = bi + (mi - 0.5) * 0.36
        if n: axD.bar(x, 100 * k / n, 0.34, color=COL[m], ec="k", lw=0.5, label=("Original" if m == "orig" else "Corrected") if bi == 0 else None); axD.text(x, 100 * k / n + 1.5, f"{k}/{n}", ha="center", va="bottom", fontsize=5.0, rotation=90)
axD.axvline(2.5, color="k", ls=":", lw=0.9); axD.text(2.45, 120, "1.5 cm", fontsize=5.5, ha="right")
axD.set_xticks(range(5)); axD.set_xticklabels(labs); axD.set_ylim(0, 128); axD.set_yticks(range(0, 101, 25)); axD.set_xlabel("Grounding error of the grasp point, xy (cm)"); axD.set_ylabel("First grasp held (%)"); axD.legend(loc="upper right", frameon=False, fontsize=5.6); axD.set_title("Matched dev seeds (object pose: analysis only)", pad=3, fontsize=6.6)

# E: first-grasp outcome categories
axE = fig.add_subplot(gs[2, 4:6]); tag(axE, "E"); cats = [("Held", "#009E73"), ("Missed, grounding ≤ 1.5 cm\n(execution / alignment)", "#E69F00"), ("Missed, grounding > 1.5 cm\n(perception)", "#CC79A7")]
for mi, m in enumerate(("orig", "corr")):
    g = np.array([nrm(r["ground"]) for r in R[m]]); h = np.array([bool(r["hold"]) for r in R[m]]); n = len(g); cnt = [int(h.sum()), int((~h & (g <= 1.5)).sum()), int((~h & (g > 1.5)).sum())]; b = 0
    for (lab, c), v in zip(cats, cnt):
        axE.bar(mi, v, 0.55, bottom=b, color=c, ec="k", lw=0.5, label=lab if mi == 0 else None); axE.text(mi, b + v / 2, str(v), ha="center", va="center", fontsize=6.2); b += v
    axE.text(mi, b + 2, f"n = {n}", ha="center", fontsize=5.8)
axE.set_xticks([0, 1]); axE.set_xticklabels(["Original", "Corrected"]); axE.set_ylabel("First grasps (count)"); axE.set_xlabel("Executor"); axE.set_ylim(0, 200); axE.legend(loc="upper left", frameon=False, fontsize=5.0, bbox_to_anchor=(-0.02, 1.02)); axE.set_title("First-grasp outcome, matched dev seeds", pad=3)
for ext in ("svg", "pdf", "png"): fig.savefig(f"{OUT}/{STEM}.{ext}", dpi=300 if ext == "png" else None)
sm = {"n_matched_episodes": {f"{b}_{c}": len(common[(b, c)]) for b in BLOCKS for c in CFGS}, "traj_episodes": {m: len(A[m]) for m in A}, "grasp_rows": {m: len(R[m]) for m in R},
      "dev_success": {f"{m}_{cfg}": counts(m, cfg) for m in ("orig", "corr") for cfg in CFGS},
      "first_grasp_held": {m: int(sum(bool(r["hold"]) for r in R[m])) for m in R}, "median_lateral_at_sample39_cm": {m: float(np.median([abs(r["lat"][r["idx"].index(39)]) for r in A[m] if 39 in r["idx"]])) for m in A}, "median_lateral_at_sample29_cm": {m: float(np.median([abs(r["lat"][r["idx"].index(29)]) for r in A[m] if 29 in r["idx"]])) for m in A}}
json.dump(sm, open(f"{OUT}/Figure_S6_numbers.json", "w"), indent=1); print(json.dumps(sm, indent=1))
