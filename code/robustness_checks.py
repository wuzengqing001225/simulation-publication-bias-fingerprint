"""Scale, transcription, paradigm, subset-interaction, neutral-prompt, conventional-
validation, and item-similarity checks (paper appendix "Scale, transcription, prompt,
and similarity checks").

Usage (repository root):  python code/robustness_checks.py
Output: analysis_tables/robustness_checks.json
Construct embeddings are precomputed by code/embed_constructs.py
(analysis_tables/construct_embeddings.json).
"""
import os, sys, json, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import nm_models as nmm

AT = "analysis_tables"
R = {}
ud = pd.read_csv(f"{AT}/P2_analysis_units.csv")
c = ud.dropna(subset=["z_sim", "z_bare", "z_o", "z_r", "var_o", "var_r"]).reset_index(drop=True)
c["D_pub"] = c.z_o - c.z_r
rng = np.random.default_rng(3)
S = lambda mc, names=("b_bare", "b_rep", "b_pub"), xn=None: nmm.summarize(mc, names=names, xnames=xn)

# 1. level calibration and rank-based partial correlations (effect level)
e = c.groupby("effect").agg(z_sim=("z_sim", "mean"), z_bare=("z_bare", "mean"),
                            z_r=("z_r", "first"), z_o=("z_o", "first")).reset_index()
e["D"] = e.z_o - e.z_r
def pcor(d, x, y, ctrl):
    rk = d[[x, y] + ctrl].rank(); Z = np.column_stack([np.ones(len(rk))] + [rk[k] for k in ctrl])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return np.corrcoef(res(rk[x]), res(rk[y]))[0, 1]
fns = {"sim_rep_marg": lambda d: stats.spearmanr(d.z_sim, d.z_r)[0],
       "sim_orig_marg": lambda d: stats.spearmanr(d.z_sim, d.z_o)[0],
       "sim_bare_marg": lambda d: stats.spearmanr(d.z_sim, d.z_bare)[0],
       "sim_rep_partial": lambda d: pcor(d, "z_sim", "z_r", ["z_bare", "D"]),
       "sim_bare_partial": lambda d: pcor(d, "z_sim", "z_bare", ["z_r", "D"]),
       "sim_D_partial": lambda d: pcor(d, "z_sim", "D", ["z_bare", "z_r"])}
R["rank_effect_level"] = {}
for k, f in fns.items():
    b = [f(e.iloc[rng.integers(0, len(e), len(e))]) for _ in range(4000)]
    R["rank_effect_level"][k] = {"est": float(f(e)), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
bb = [(c.z_sim - c.z_r).iloc[rng.integers(0, len(c), len(c))].mean() for _ in range(4000)]
R["level"] = {"mean_zsim": float(c.z_sim.mean()), "mean_zr": float(c.z_r.mean()), "mean_zo": float(c.z_o.mean()),
              "bias_vs_rep": float((c.z_sim - c.z_r).mean()), "bias_vs_rep_ci": [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))],
              "bias_vs_orig": float((c.z_sim - c.z_o).mean()), "mean_abs_zsim": float(c.z_sim.abs().mean()),
              "mean_abs_zr": float(c.z_r.abs().mean()), "share_sim_gt_rep": float((c.z_sim.abs() > c.z_r.abs()).mean())}
R["main_b0"] = S(nmm.fit(c), names=("b0",))["b0"]

# cell tables (per transcription variant, minimal framing unless noted)
cols = ["effect", "variant", "model", "persona", "r_sim"]
allc = pd.concat([pd.read_csv(f"{AT}/P2_cells_g{i}.csv")[cols] for i in range(3)]
                 + [pd.read_csv(f"{AT}/{f}")[cols] for f in ["A0_cell_effects.csv", "A1_cell_effects.csv"]])
allc = allc[allc.variant.isin(["A", "B"])]

# 2. personas and variance
est_eff = set(c.effect)
pv = allc[allc.effect.isin(est_eff)].dropna(subset=["r_sim"]).pivot_table(
    index=["effect", "variant", "model"], columns="persona", values="r_sim").dropna()
R["persona_variance"] = {"n_cells": len(pv), "median_abs_none": float(pv["none"].abs().median()),
                         "median_abs_demo": float(pv["demographic"].abs().median()),
                         "ceiling_none": float((pv["none"].abs() > .95).mean()),
                         "ceiling_demo": float((pv["demographic"].abs() > .95).mean()),
                         "wilcoxon_p": float(stats.wilcoxon(pv["none"].abs(), pv["demographic"].abs()).pvalue)}

# 3. cross-paired transcriptions (bare bank was built from variant A)
cv = allc[allc.persona == "none"].drop_duplicates(["effect", "variant", "model"]).copy()
cv["z"] = np.arctanh(cv.r_sim.clip(-0.99, 0.99))
za = cv[cv.variant == "A"][["effect", "model", "z"]].rename(columns={"z": "zA"})
zb = cv[cv.variant == "B"][["effect", "model", "z"]].rename(columns={"z": "zB"})
cc = c.merge(za, on=["effect", "model"]).merge(zb, on=["effect", "model"]).dropna(subset=["zA", "zB"])
for v in ["A", "B"]:
    d = cc.drop(columns=["z_sim"]).rename(columns={f"z{v}": "z_sim"})
    R[f"sim_variant_{v}_common"] = {"n_units": len(d), **S(nmm.fit(d))}

# 4. paradigm class
R["D_by_class"] = c.drop_duplicates("effect").groupby("paradigm_class").D_pub.agg(
    ["count", "median", "mean", "min", "max"]).round(2).reset_index().to_dict("records")
R["zsim_by_class"] = c.groupby("paradigm_class").z_sim.mean().round(2).to_dict()
classes = [k for k in c.paradigm_class.value_counts().index if k != "jdm_bias"]
Xc = np.column_stack([(c.paradigm_class == k).astype(float) for k in classes])
R["paradigm_FE"] = S(nmm.fit(c, X=Xc), names=("b_bare", "b_rep", "b_pub", "sd_eff"))
cj = c[c.paradigm_class == "jdm_bias"]
R["jdm_only"] = {"n_units": len(cj), "n_eff": int(cj.effect.nunique()), **S(nmm.fit(cj))}

# 5. subset differences as interaction terms
Dm = (c.z_o - c.z_r).values
for nm, ind in [("ceiling", (c.r_sim.abs() > 0.95).astype(float).values),
                ("uniform_wave", (c.wave != "A0A1").astype(float).values)]:
    R[f"interaction_{nm}"] = S(nmm.fit(c, X=np.column_stack([ind, ind * Dm])), names=("b_pub",),
                               xn=[f"{nm}_main", f"{nm}_x_D"])

# 6. neutral system prompt (cells scored by code/xm_analyze.py from the neutral rerun)
nt = []
for mk in ["sonnet5", "sonnet45"]:
    for part in ["new", "old"]:
        d = pd.read_csv(f"{AT}/NT_{mk}_cells_{part}.csv"); d["model"] = mk
        nt.append(d[["effect", "variant", "model", "r_sim"]])
nt = pd.concat(nt).dropna(subset=["r_sim"]); nt["z"] = np.arctanh(nt.r_sim.clip(-0.99, 0.99))
zn = nt.groupby(["effect", "model"]).z.mean().rename("z_sim").reset_index()
zo = cv.dropna(subset=["z"]).groupby(["effect", "model"]).z.mean().rename("z_sim").reset_index()
base = c.drop(columns=["z_sim"])
common = base.merge(zn, on=["effect", "model"])[["effect", "model"]].merge(base.merge(zo, on=["effect", "model"])[["effect", "model"]])
cn = base.merge(zn, on=["effect", "model"]).merge(common); co = base.merge(zo, on=["effect", "model"]).merge(common)
mm = co[["effect", "model", "z_sim", "z_r"]].merge(cn[["effect", "model", "z_sim"]], on=["effect", "model"], suffixes=("_orig", "_neut"))
R["neutral_prompt"] = {"n_units": len(common), "original": S(nmm.fit(co)), "neutral": S(nmm.fit(cn)),
                       "corr_orig_neutral": float(np.corrcoef(mm.z_sim_orig, mm.z_sim_neut)[0, 1]),
                       "mean_shift": float((mm.z_sim_neut - mm.z_sim_orig).mean()),
                       "dist_to_rep_orig": float((mm.z_sim_orig - mm.z_r).abs().mean()),
                       "dist_to_rep_neut": float((mm.z_sim_neut - mm.z_r).abs().mean())}

# 7. conventional validation
grp = pd.factorize(c.effect)[0]
conv = {"corr_sim_orig": float(np.corrcoef(c.z_sim, c.z_o)[0, 1]), "corr_sim_rep": float(np.corrcoef(c.z_sim, c.z_r)[0, 1]),
        "sign_agree_orig": float((np.sign(c.z_sim) == np.sign(c.z_o)).mean()),
        "sign_agree_rep": float((np.sign(c.z_sim) == np.sign(c.z_r)).mean())}
for ref in ["z_o", "z_r"]:
    f = smf.ols(f"z_sim ~ {ref}", c).fit(cov_type="cluster", cov_kwds={"groups": grp})
    conv[f"slope_on_{ref}"] = [float(f.params[ref])] + [float(x) for x in f.conf_int().loc[ref]]
R["conventional"] = conv

# 8. item similarity in the construct battery
spec = importlib.util.spec_from_file_location("a2x2", os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_ceil2x2.py"))
a2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a2)
_, pair = a2.full_design(pd.read_csv(f"{AT}/ceil2x2_results.csv"))
emb = json.load(open(f"{AT}/construct_embeddings.json"))
texts = json.load(open("protocols/construct_item_texts.json"))
from sklearn.feature_extraction.text import TfidfVectorizer
names = sorted(texts); tf = TfidfVectorizer(stop_words="english").fit([" ".join(texts[k]) for k in names])
TV = {k: tf.transform([" ".join(texts[k])]) for k in names}
def sims(pid):
    x, y = pid.split("|")
    out = {mn: float(np.dot(emb[mn][x], emb[mn][y])) for mn in emb}
    out["tfidf"] = float((TV[x] @ TV[y].T).toarray()[0, 0]); return out
pr = pd.concat([pair.reset_index(drop=True), pd.DataFrame([sims(p) for p in pair.pair_id])], axis=1)
pr.to_csv(f"{AT}/pair_similarity.csv", index=False)
sem = {}
for cov in [None, "all-mpnet-base-v2", "all-MiniLM-L6-v2", "tfidf"]:
    d = pr.rename(columns={cov: "sim"}) if cov else pr
    m = smf.ols("excess ~ fame + pol" + (" + sim" if cov else ""), d).fit()
    row = {"fame": float(m.params["fame"]), "fame_se": float(m.bse["fame"]), "fame_p": float(m.pvalues["fame"])}
    if cov: row.update({"sim": float(m.params["sim"]), "sim_p": float(m.pvalues["sim"])})
    sem[cov or "baseline"] = row
sem["group_means"] = {str(k): v for k, v in pr.groupby("fame")[["all-mpnet-base-v2", "all-MiniLM-L6-v2", "tfidf"]].mean().round(4).to_dict().items()}
sem["mpnet_group_diff_p"] = float(stats.mannwhitneyu(pr[pr.fame == 1]["all-mpnet-base-v2"], pr[pr.fame == 0]["all-mpnet-base-v2"]).pvalue)
R["semantic"] = sem

json.dump(R, open(f"{AT}/robustness_checks.json", "w"), indent=1, default=float)
print("saved", f"{AT}/robustness_checks.json")
