"""Supplementary figure 'Counterfactual Language Grounding across Manipulation Stages'.
Reads ONLY saved result files of the 4-task V21 counterfactual study (paths below); every plotted number is parsed from them (Wilson intervals are computed from the plotted counts).
usage: python make_counterfactual_language_figure.py   -> SVG / PDF / PNG next to this script."""
import csv, json, math, glob, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import os; N = os.environ["REGSP_RESEARCH_TREE"]; OUT = os.environ.get("REGSP_FIG_OUT", f"{N}/figures_supp"); STEM = "Counterfactual_Language_Grounding"  # N = research-tree output directory (outputs/robotwin_v21_4task_counterfactual)
SRC_CSV, SRC_MD = f"{N}/PIC_LANGUAGE_DEPENDENCE.csv", f"{N}/COUNTERFACTUAL_REFERENT_REPORT.md"
G_NEW, G_PREV = "FINAL3 fresh seeds, PIC", "FINAL3 fresh seeds, previous frozen model"; CFC = "counterfactual (roles exchanged)"

# ---------------------------------------------------------------- data (parsed, not typed)
rows = list(csv.DictReader(open(SRC_CSV)))
def pick(group, cfg, cond):
    r = [x for x in rows if x["group"] == group and x["task"] == "stack_blocks_two" and x["config"] == cfg and x["instruction_condition"] == cond]; assert len(r) == 1, (group, cfg, cond); return r[0]
D = {}
for m, g in (("prev", G_PREV), ("new", G_NEW)):
    for cfg in ("clean", "rand"):
        o, c = pick(g, cfg, "orig"), pick(g, cfg, CFC); n = int(c["n"]); assert int(o["n"]) == n
        D[(m, cfg)] = {"n": n, "std": int(o["native_success"]), "cf_goal": int(c["goal_red_on_green"]), "first_green": int(c["first_grasp_green"])}
tab = {}
for line in open(SRC_MD):
    p = [x.strip() for x in line.strip().strip("|").split("|")]
    if len(p) >= 13 and p[0] in ("click_bell", "stack_blocks_two"): tab[(p[0], p[1], p[2])] = p
def sc(task, mode, tag):
    p = tab[(task, mode, tag)]; return {"orig_val_n": int(p[4].split("/")[1]), "cf_val_n": int(p[5].split("/")[1]), "orig_acc": float(p[6]), "cf_acc": float(p[7]), "swap": float(p[8]), "chance": float(p[10])}
S = {t: {"Baseline scorer (no counterfactual pairs)": sc(t, "base", "v1"), "Counterfactual-paired scorer (same data)": sc(t, "cf", "v1"), "Counterfactual-paired scorer (frozen, +rand. frames)": sc(t, "cf", "u1")} for t in ("click_bell", "stack_blocks_two")}
# same-seed check of the native fresh-seed cells (reported in the sources file)
seedsets = {k: sorted(os.path.basename(f) for f in glob.glob(f"{N}/eval/{d}/{cfg}/stack_blocks_two/seed*.json")) for k, (d, cfg) in {"new_cf_clean": ("final3_pic_cf", "clean"), "prev_cf_clean": ("final3_prev_cf", "clean"), "new_cf_rand": ("final3_pic_cf", "rand"), "prev_cf_rand": ("final3_prev_cf", "rand"), "new_clean": ("final3_pic", "clean"), "prev_clean": ("final3_prev", "clean"), "new_rand": ("final3_pic", "rand"), "prev_rand": ("final3_prev", "rand")}.items()}
same = {a: seedsets["new_" + a] == seedsets["prev_" + a] for a in ("cf_clean", "cf_rand", "clean", "rand")}; json.dump({"seed_sets_identical": same, "n_files": {k: len(v) for k, v in seedsets.items()}}, open(f"{OUT}/seed_check.json", "w"), indent=1)

# ---------------------------------------------------------------- style
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3, "axes.linewidth": 0.7, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
C_PREV, C_NEW, C_BASE, C_CF, C_CFU = "#9A9A9A", "#0072B2", "#9A9A9A", "#56B4E9", "#0072B2"; RED, GREEN = "#C0392B", "#2E8B57"
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return 100 * (c - h) / d, 100 * (c + h) / d
def tag(ax, s): ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")
fig = plt.figure(figsize=(7.6, 7.4)); gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.0], hspace=0.68, wspace=1.3, left=0.095, right=0.985, top=0.925, bottom=0.085)
fig.suptitle("Counterfactual Language Grounding across Manipulation Stages", fontsize=10.5, fontweight="bold", y=0.985)

# ---------------------------------------------------------------- A: schematic
axA = fig.add_subplot(gs[0, 0:3]); axA.set_xlim(0, 10); axA.set_ylim(0, 10); axA.axis("off"); tag(axA, "A")
axA.text(5, 9.95, "stack_blocks_two: same scene, exchanged block roles", ha="center", va="top", fontsize=7.6, fontweight="bold")
def blk(x, y, c, w=1.0, h=0.7): axA.add_patch(Rectangle((x - w / 2, y), w, h, fc=c, ec="k", lw=0.6))
cols = {"Original instruction": 3.9, "Counterfactual (roles exchanged)": 8.0}
ins = {"Original instruction": "\u2026stack green block\non red block\u2026", "Counterfactual (roles exchanged)": "\u2026stack red block\non green block\u2026"}
axA.text(0.0, 6.5, "Stage 1\n(first grasp,\nplaced at centre)", fontsize=6.0, fontweight="bold", va="center"); axA.text(0.0, 2.6, "Stage 2\n(stack target)", fontsize=6.0, fontweight="bold", va="center")
for name, cx in cols.items():
    orig = name.startswith("Original"); bot, top = (RED, GREEN) if orig else (GREEN, RED)
    axA.text(cx, 8.95, name.replace(" (", "\n("), ha="center", va="center", fontsize=6.4, fontweight="bold", linespacing=1.1); axA.text(cx, 7.7, ins[name], ha="center", va="center", fontsize=6.2, style="italic")
    axA.plot([cx - 1.8, cx + 1.8], [5.0, 5.0], color="#555", lw=0.8); blk(cx - 0.8, 5.0, RED); blk(cx + 0.8, 5.0, GREEN)
    xs = cx - 0.8 if orig else cx + 0.8
    axA.add_patch(FancyArrowPatch((xs, 6.75), (xs, 5.85), arrowstyle="-|>", mutation_scale=7, lw=1.0, color="k"))
    axA.text(cx, 6.95, "grasp %s block" % ("red" if orig else "green"), ha="center", va="bottom", fontsize=5.8)
    axA.plot([cx - 1.8, cx + 1.8], [1.4, 1.4], color="#555", lw=0.8); blk(cx, 1.4, bot); blk(cx, 2.1, top)
    axA.text(cx, 0.95, "goal: %s on %s" % (("green", "red") if orig else ("red", "green")), ha="center", va="top", fontsize=5.8)
axA.plot([5.95, 5.95], [1.0, 9.2], color="#BBBBBB", lw=0.6, ls=":"); axA.text(5, 0.0, "Only the instruction differs; the scorer must re-assign which block is the referent at each stage.", ha="center", va="bottom", fontsize=5.6)

# ---------------------------------------------------------------- B: offline scorer accuracy (held-out scenes)
mets = [("orig_acc", "Orig.\nreferent"), ("cf_acc", "CF\nreferent"), ("swap", "Swap:\nboth correct")]
cl = {"Baseline scorer (no counterfactual pairs)": C_BASE, "Counterfactual-paired scorer (same data)": C_CF, "Counterfactual-paired scorer (frozen, +rand. frames)": C_CFU}
sg = gs[0, 3:6].subgridspec(1, 2, wspace=0.28); axB = [fig.add_subplot(sg[0, 0]), fig.add_subplot(sg[0, 1])]
for ax, task in zip(axB, ("click_bell", "stack_blocks_two")):
    names = list(S[task]); bw = 0.26
    for j, nm in enumerate(names):
        xs = [i + (j - 1) * bw for i in range(3)]; vals = [100 * S[task][nm][k] for k, _ in mets]
        ax.bar(xs, vals, bw, color=cl[nm], ec="k", lw=0.4, label=nm if task == "click_bell" else None)
        for x, v in zip(xs, vals): ax.text(x, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=4.8, rotation=90)
    ax.set_xticks(range(3)); ax.set_xticklabels([m[1] for m in mets]); ax.set_ylim(0, 118); ax.set_yticks(range(0, 101, 25)); ax.set_xlabel("Held-out scorer metric", fontsize=6.5); ax.tick_params(axis="x", labelsize=5.4)
    s0 = S[task][names[0]]; ax.set_title(f"{task}\nn = {s0['orig_val_n']} orig / {s0['cf_val_n']} CF scenes", fontsize=6.2, pad=3)
    if task == "click_bell": ax.set_ylabel("Referent accuracy (%)")
axB[0].legend(loc="upper left", bbox_to_anchor=(-0.1, -0.27), ncol=1, frameon=False, fontsize=5.8); tag(axB[0], "B")
axB[0].text(0.0, 1.27, "Offline scorer-level results (no rollouts)", transform=axB[0].transAxes, fontsize=6.4, style="italic")

# ---------------------------------------------------------------- C: native fresh-seed exchanged-goal completion
def grouped(ax, key, ylabel, title):
    for gi, cfg in enumerate(("clean", "rand")):
        for mi, (m, c, lab) in enumerate((("prev", C_PREV, "Previous frozen model"), ("new", C_NEW, "Updated model (instruction-conditioned scorer + progress head)"))):
            d = D[(m, cfg)]; k, n = d[key], d["n"]; v = 100 * k / n; lo, hi = wilson(k, n); x = gi + (mi - 0.5) * 0.36
            ax.bar(x, v, 0.34, color=c, ec="k", lw=0.5, label=lab if gi == 0 else None); ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color="k", lw=0.7, capsize=1.8)
            ax.text(x, hi + 2, f"{k}/{n}", ha="center", va="bottom", fontsize=6.2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Clean", "Randomized"]); ax.set_ylim(0, 135); ax.set_yticks(range(0, 101, 25)); ax.set_xlabel("Evaluation domain (fresh seeds)"); ax.set_ylabel(ylabel)
axC = fig.add_subplot(gs[1, 0:3]); grouped(axC, "cf_goal", "Exchanged goal completed (%)", ""); tag(axC, "C")
axC.set_title("Native fresh-seed rollouts, counterfactual instruction", fontsize=7, pad=3); axC.legend(loc="upper left", frameon=False, fontsize=5.8)

# ---------------------------------------------------------------- D: standard success vs counterfactual-goal completion
axD = fig.add_subplot(gs[1, 3:6]); tag(axD, "D")
xl = []; xt = []; pos = 0
for cfg in ("clean", "rand"):
    for key, nm in (("std", "Standard task\nsuccess"), ("cf_goal", "Exchanged-goal\ncompletion")):
        for mi, (m, c) in enumerate((("prev", C_PREV), ("new", C_NEW))):
            d = D[(m, cfg)]; k, n = d[key], d["n"]; v = 100 * k / n; lo, hi = wilson(k, n); x = pos + (mi - 0.5) * 0.36
            axD.bar(x, v, 0.34, color=c, ec="k", lw=0.5, hatch=None if key == "std" else "////"); axD.errorbar(x, v, yerr=[[v - lo], [hi - v]], color="k", lw=0.7, capsize=1.8); axD.text(x, hi + 2, f"{k}", ha="center", va="bottom", fontsize=6.2)
        xt.append(pos); xl.append(("Clean" if cfg == "clean" else "Rand.") + "\n" + nm); pos += 1
    pos += 0.5
axD.set_xticks(xt); axD.set_xticklabels(xl, fontsize=5.8); axD.set_ylim(0, 135); axD.set_yticks(range(0, 101, 25)); axD.set_xlabel("Domain / metric (n = 20 episodes per bar; counts above bars)"); axD.set_ylabel("Success (%)")
axD.set_title("Standard success vs. counterfactual-goal completion", fontsize=7, pad=3)
from matplotlib.patches import Patch
axD.legend(handles=[Patch(fc=C_PREV, ec="k", lw=0.5, label="Previous frozen"), Patch(fc=C_NEW, ec="k", lw=0.5, label="Updated"), Patch(fc="w", ec="k", lw=0.5, label="Standard success (original instruction)"), Patch(fc="w", ec="k", lw=0.5, hatch="////", label="Exchanged-goal completion")], loc="upper left", ncol=2, frameon=False, fontsize=5.6, columnspacing=0.8, handlelength=1.4)
for ext in ("svg", "pdf", "png"): fig.savefig(f"{OUT}/{STEM}.{ext}", dpi=300 if ext == "png" else None)
print(json.dumps({"stack_native": {f"{m}_{c}": v for (m, c), v in D.items()}, "seed_sets_identical": same}, indent=1))
