"""Positive controls, transcription cross-pairing (all three families), direction
subset, and structure-experiment missingness bounds.

Usage (repository root):  python code/positive_controls_and_checks.py
Inputs:  analysis_tables/P2_analysis_units.csv, the cell tables, the xmodel_results
         tables, analysis_tables/PC_*_cells_*.csv (scored positive-control runs),
         analysis_tables/ceil2x2_results.csv and protocols/ceil2x2_pairs.csv.
Output:  analysis_tables/positive_controls_and_checks.json
The positive-control prompts are in protocols/positive_control_findings.json
(one-sentence finding per effect); raw responses are available on request.
"""
import os, sys, json, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import nm_models as nmm

AT, XR = "analysis_tables", "xmodel_results"
SM = lambda mc: nmm.summarize(mc, names=("b_bare", "b_rep", "b_pub"))
Z = lambda r: np.arctanh(np.clip(r, -0.99, 0.99))
out = {}

ud = pd.read_csv(f"{AT}/P2_analysis_units.csv")
conf = ud.dropna(subset=["z_sim", "z_bare", "z_o", "z_r", "var_o", "var_r"]).reset_index(drop=True)

# 1. transcription cross-pairing, Claude
cells = pd.concat([pd.read_csv(f"{AT}/P2_cells_g{i}.csv") for i in range(3)]
                  + [pd.read_csv(f"{AT}/{f}") for f in ["A0_cell_effects.csv", "A1_cell_effects.csv"]])
cells = cells[(cells.persona == "none") & cells.variant.isin(["A", "B"])].drop_duplicates(["effect", "variant", "model"])
cells["z"] = Z(cells.r_sim)
def crosspair(units, cl, keys):
    a = cl[cl.variant == "A"][keys + ["z"]].rename(columns={"z": "zA"})
    b = cl[cl.variant == "B"][keys + ["z"]].rename(columns={"z": "zB"})
    m = units.merge(a, on=keys).merge(b, on=keys).dropna(subset=["zA", "zB"]).copy()
    m["z_pool"] = (m.zA + m.zB) / 2
    res = {"n_units": len(m), "corr_AB": float(np.corrcoef(m.zA, m.zB)[0, 1])}
    for lab, col in [("average", "z_pool"), ("first_shared", "zA"), ("second_independent", "zB")]:
        res[lab] = SM(nmm.fit(m.drop(columns=["z_sim"]).rename(columns={col: "z_sim"})))
    return res
out["crosspair_claude"] = crosspair(conf, cells, ["effect", "model"])

# 2. transcription cross-pairing, GPT and DeepSeek
for fam, pre in [("GPT", f"{XR}/XM"), ("DeepSeek", f"{XR}/deepseek/DS")]:
    u = pd.read_csv(f"{pre}_analysis_units.csv")
    for v, n in [("var_r", "n_r"), ("var_o", "n_o")]:
        if v not in u: u[v] = 1 / (u[n] - 3)
    if "model" not in u: u["model"] = fam
    uc = u.dropna(subset=["z_sim", "z_bare", "z_o", "z_r"])
    cl = pd.concat([pd.read_csv(f"{pre}_cells_new.csv"), pd.read_csv(f"{pre}_cells_old.csv")]).dropna(subset=["r_sim"])
    cl["z"] = Z(cl.r_sim)
    out[f"crosspair_{fam}"] = crosspair(uc, cl, ["effect"])

# 3. direction where the model shows no bias
rows = []
fams = [("Claude", conf)]
for fam, pre in [("GPT", f"{XR}/XM"), ("DeepSeek", f"{XR}/deepseek/DS")]:
    u = pd.read_csv(f"{pre}_analysis_units.csv")
    if "var_r" not in u: u["var_r"] = 1 / (u.n_r - 3)
    fams.append((fam, u))
for fam, u in fams:
    u = u.dropna(subset=["z_sim", "z_bare", "z_r"]).copy()
    u["rep_real"] = (u.z_r.abs() / np.sqrt(u.var_r)) > 1.96
    for thr in [0.10, 0.20]:
        null = np.tanh(u.z_bare).abs() < thr
        for lab, m in [("no_bias", null & u.rep_real), ("bias", ~null & u.rep_real)]:
            s = u[m]; agree = np.sign(s.z_sim) == np.sign(s.z_r)
            rows.append({"family": fam, "bare_thr": thr, "subset": lab, "n_units": len(s), "k_agree": int(agree.sum())})
out["direction_subset"] = rows

# 4. positive controls
pcz = {}
for arm in ["orig", "rep"]:
    cl = []
    for mk in ["sonnet5", "sonnet45"]:
        for part in ["new", "old"]:
            fn = f"{AT}/PC_{arm}_{mk}_cells_{part}.csv"
            if os.path.exists(fn):
                d = pd.read_csv(fn); d["model"] = mk; cl.append(d[["effect", "variant", "model", "r_sim"]])
    cl = pd.concat(cl).dropna(subset=["r_sim"]); cl["z"] = Z(cl.r_sim)
    pcz[arm] = cl.groupby(["effect", "model"]).z.mean().rename("z_sim").reset_index()
base = conf.drop(columns=["z_sim"])
for arm in ["orig", "rep"]:
    out[f"pc_{arm}"] = SM(nmm.fit(base.merge(pcz[arm], on=["effect", "model"])))
pr = base.merge(pcz["orig"].rename(columns={"z_sim": "zo_arm"}), on=["effect", "model"]).merge(
     pcz["rep"].rename(columns={"z_sim": "zr_arm"}), on=["effect", "model"])
pr["diff"] = pr.zo_arm - pr.zr_arm; pr["D"] = pr.z_o - pr.z_r
m = smf.ols("diff ~ D", pr).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(pr.effect)[0]})
out["pc_paired"] = {"n_units": len(pr), "slope": float(m.params.D), "ci": [float(x) for x in m.conf_int().loc["D"]], "p": float(m.pvalues.D)}

# 5. structure-experiment missingness bounds
spec = importlib.util.spec_from_file_location("a2", "code/analyze_ceil2x2.py")
a2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a2)
obs, _ = a2.full_design(pd.read_csv(f"{AT}/ceil2x2_results.csv"))
tw = pd.read_csv(f"{AT}/B2_threeway.csv"); cp = pd.read_csv("protocols/ceil2x2_pairs.csv")
grid = pd.concat([tw.drop_duplicates("pair_id").assign(pair_id=lambda d: d.construct_x + "|" + d.construct_y)[["group", "pair_id"]],
                  cp.assign(pair_id=lambda d: d.construct_x + "|" + d.construct_y)[["group", "pair_id"]]]).drop_duplicates()
full = grid.merge(pd.DataFrame({"model": ["sonnet5", "sonnet45"]}), how="cross").merge(
       obs[["group", "pair_id", "model", "excess"]], on=["group", "pair_id", "model"], how="left")
full = full.merge(obs.drop_duplicates(["group", "pair_id"]).set_index(["group", "pair_id"])[["fame", "pol"]],
                  left_on=["group", "pair_id"], right_index=True, how="left")
full.loc[full.fame.isna(), ["fame", "pol"]] = [1, 0]   # fully uncomputable pairs are famous nonpolitical ability pairs
def pfit(d):
    p = d.dropna(subset=["excess"]).groupby(["group", "pair_id", "fame", "pol"]).excess.mean().reset_index()
    f = smf.ols("excess ~ fame + pol", p).fit()
    return {"n_pairs": len(p), "fame": float(f.params.fame), "fame_p": float(f.pvalues.fame), "pol": float(f.params.pol)}
lo = full.excess.min()
f2 = full.copy(); f2.loc[f2.excess.isna() & (f2.fame == 1), "excess"] = lo
f3 = full.copy(); f3.loc[f3.excess.isna(), "excess"] = 0.0
f4 = full.dropna(subset=["excess"]).groupby(["group", "pair_id"]).filter(lambda g: len(g) == 2)
out["c2x2_missingness"] = {"baseline": pfit(full), "worst_case_min": pfit(f2), "missing_zero": pfit(f3),
                           "both_models_only": pfit(f4), "min_observed_excess": float(lo)}

json.dump(out, open(f"{AT}/positive_controls_and_checks.json", "w"), indent=1)
print("saved", f"{AT}/positive_controls_and_checks.json")
