"""Redraw the body-text panels of Figure 2 (decomposition) and Figure 4 (refusal)
at their printed size, so that 8 pt text in the file is 8 pt on the page.

Usage (repository root):  python code/make_main_figures.py  [outdir]
Inputs: analysis_tables/P2_main_model_idata.nc, sensitivity_posteriors.csv,
        S1_refusal_units.csv, S1_condition_map.csv, the cell tables.
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import arviz as az

AT = "analysis_tables"
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
FS, FT = 8.0, 7.5
C_BARE, C_REP, C_PUB = "#2E6DA4", "#3E9A70", "#C0392B"
FOC, REF, REV, GREY = "#C2410C", "#2563EB", "#7C3AED", "#888888"
mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
    "legend.fontsize": FT, "xtick.labelsize": FT, "ytick.labelsize": FT,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.dpi": 300})


def save(fig, stem):
    fig.savefig(os.path.join(OUT, f"{stem}.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, f"{stem}.png"), bbox_inches="tight", pad_inches=0.02, dpi=200)


# ---------------- Figure 2a: three coefficients ----------------
idata = az.from_netcdf(f"{AT}/P2_main_model_idata.nc")
rows = [("b_bare", "$\\beta_{\\mathrm{bare}}$\nown bias", C_BARE),
        ("b_rep", "$\\beta_{\\mathrm{rep}}$\nreplication", C_REP),
        ("b_pub", "$\\beta_{\\mathrm{pub}}$\ngap", C_PUB)]
fig, ax = plt.subplots(figsize=(2.25, 2.1))
ax.axvspan(-0.10, 0.10, color="0.88", zorder=0, lw=0)
ax.axvline(0, color="0.35", lw=0.7, ls="--", zorder=1)
for k, (vn, lab, col) in enumerate(rows):
    v = idata.posterior[vn].values.ravel()
    y = len(rows) - 1 - k
    lo, hi = az.hdi(v, hdi_prob=0.95); l5, h5 = az.hdi(v, hdi_prob=0.50); m = v.mean()
    ax.plot([lo, hi], [y, y], color=col, lw=1.4, solid_capstyle="round", zorder=2)
    ax.plot([l5, h5], [y, y], color=col, lw=4.0, solid_capstyle="butt", zorder=3)
    ax.plot(m, y, "o", color=col, ms=5, zorder=4)
    ax.text(m, y + 0.30, f"{m:.2f} [{lo:.2f}, {hi:.2f}]", va="center", ha="center", fontsize=FT, zorder=5,
            bbox=dict(fc="white", ec="none", pad=0.6))
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows][::-1])
ax.set_ylim(-0.5, len(rows) - 0.5); ax.set_xlim(-0.6, 2.4)
ax.set_xticks([0, 1, 2])
ax.set_xlabel("coefficient (Fisher-z slope)\n95% and 50% HDI")
save(fig, "p2_verdict_a")

# ---------------- Figure 2b: gap coefficient by subset ----------------
sp = pd.read_csv(f"{AT}/sensitivity_posteriors.csv")
subs = [("full", "full (71)"), ("excl_ceiling", "no ceiling (51)"),
        ("high_precision", "$n_r \\geq 500$ (45)"), ("uniform_wave", "uniform (43)")]
fig, ax = plt.subplots(figsize=(1.95, 2.1))
ax.axvspan(-0.10, 0.10, color="0.88", zorder=0, lw=0)
ax.axvline(0, color="0.35", lw=0.7, ls="--", zorder=1)
for k, (key, lab) in enumerate(subs):
    r = sp[(sp.analysis == key) & (sp.param == "b_pub")].iloc[0]
    p = sp[(sp.analysis == key) & (sp.param == "P_pub_gt0")]["mean"].iloc[0]
    y = len(subs) - 1 - k
    ax.plot([r["hdi_2.5"], r["hdi_97.5"]], [y, y], color=C_PUB, lw=1.4)
    ax.plot(r["mean"], y, "o", color=C_PUB, ms=5)
    ax.text(r["hdi_97.5"] + 0.1, y, f"{p:.2f}", va="center", fontsize=FT)
ax.set_yticks(range(len(subs))); ax.set_yticklabels([s[1] for s in subs][::-1])
ax.set_ylim(-0.5, len(subs) - 0.5); ax.set_xlim(-1.5, 2.6); ax.set_xticks([-1, 0, 1, 2])
ax.set_xlabel("$\\beta_{\\mathrm{pub}}$, 95% HDI\n(label: $P(\\beta_{\\mathrm{pub}}>0)$)")
save(fig, "p2_verdict_b")

# ---------------- Figure 4a: within-effect arm contrast ----------------
u = pd.read_csv(f"{AT}/S1_refusal_units.csv"); cm = pd.read_csv(f"{AT}/S1_condition_map.csv")
cand = cm.groupby("effect").aversive_pole.nunique(); cand = cand[cand > 1].index.tolist()
AV = u[(u.model == "sonnet45") & (u.effect.isin(cand))]
pb = AV.groupby(["effect", "aversive_pole"]).agg(n=("n_rollouts", "sum"), r=("n_refusal", "sum")).reset_index()
pb["rate"] = pb.r / pb.n
rates = pb.pivot(index="effect", columns="aversive_pole", values="rate"); rates.columns = ["benign", "aversive"]
pool = AV.groupby("aversive_pole").agg(n=("n_rollouts", "sum"), r=("n_refusal", "sum")); pool["rate"] = pool.r / pool.n
up = int((rates.aversive > rates.benign).sum()); dn = int((rates.aversive < rates.benign).sum()); tie = len(rates) - up - dn
fig, ax = plt.subplots(figsize=(2.2, 1.9))
for _, r in rates.iterrows():
    c = FOC if r.aversive > r.benign else (REV if r.aversive < r.benign else GREY)
    ax.plot([0, 1], [r.benign, r.aversive], color=c, lw=1.0, marker="o", ms=3,
            mfc=c if c == FOC else "white", mec=c, zorder=2)
ax.plot([0, 1], [pool.loc[0, "rate"], pool.loc[1, "rate"]], color="black", lw=2.2, marker="s", ms=4.5, zorder=5)
ax.set_xticks([0, 1]); ax.set_xticklabels(["benign\narm", "self-implicating\narm"])
ax.set_xlim(-0.25, 1.25); ax.set_ylim(-0.03, 0.55)
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
ax.set_ylabel("refusal rate (Sonnet 4.5)")
ax.legend(handles=[Line2D([], [], color=FOC, lw=1, marker="o", ms=3, label=f"higher, aversive ({up})"),
                   Line2D([], [], color=REV, lw=1, marker="o", ms=3, mfc="white", label=f"lower ({dn})"),
                   Line2D([], [], color=GREY, lw=1, marker="o", ms=3, mfc="white", label=f"identical ({tie})"),
                   Line2D([], [], color="black", lw=2.2, marker="s", ms=4.5, label="pooled")],
          loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, handlelength=1.4, columnspacing=0.8)
save(fig, "s1_b")

# ---------------- Figure 4b: cells with no computable effect ----------------
cells = pd.concat([pd.read_csv(f"{AT}/P2_cells_g{i}.csv")[["effect", "variant", "model", "persona", "r_sim"]] for i in range(3)]
                  + [pd.read_csv(f"{AT}/{f}")[["effect", "variant", "model", "persona", "r_sim"]]
                     for f in ["A0_cell_effects.csv", "A1_cell_effects.csv"]])
keys = ["effect", "model", "variant", "persona"]
g = u.groupby(keys + ["wave"]).agg(asym=("refusal_rate", lambda x: x.max() - x.min()), n_cond=("cond", "nunique")).reset_index()
g = g[g.n_cond >= 2].merge(cells.drop_duplicates(keys), on=keys, how="left")
g["nocomp"] = g.r_sim.isna()
g["bin"] = pd.cut(g.asym, [-0.001, 0.0, 0.15, 0.50, 1.0], labels=["0", "0-15%", "15-50%", ">50%"])
t = g.groupby(["model", "bin"], observed=False).agg(n=("nocomp", "size"), share=("nocomp", "mean")).reset_index()
fig, ax = plt.subplots(figsize=(2.25, 1.9))
x = np.arange(4); w = 0.38
s45 = t[t.model == "sonnet45"].set_index("bin"); s5 = t[t.model == "sonnet5"].set_index("bin")
ax.bar(x - w / 2, s45.share.fillna(0).values, w, color=FOC, label="Sonnet 4.5")
ax.bar(x + w / 2, s5.share.fillna(0).values, w, facecolor="white", edgecolor=REF, lw=0.9, label="Sonnet 5")
ax.text(3 - w / 2, s45.share.iloc[3] + 0.03, f"{s45.share.iloc[3]:.0%}", ha="center", fontsize=FT, color=FOC)
ax.set_xticks(x); ax.set_xticklabels([f"{b}\n(n={int(n)})" for b, n in zip(s45.index, s45.n)])
ax.set_ylim(0, 1.0); ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
ax.set_ylabel("cells with no effect size")
ax.set_xlabel("refusal gap between arms")
ax.legend(loc="upper left", handlelength=1.2)
save(fig, "s1_c")
print("panels written to", OUT, "| Sonnet 4.5 bins:", s45.n.tolist(), [round(v, 3) for v in s45.share.tolist()])
