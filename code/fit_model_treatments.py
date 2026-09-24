"""Reproduce the three additional model treatments of the main regression.

  A. narrative-stratified fingerprint: the D_pub slope gets an extra term for
     effects whose original finding still dominates the corpus narrative
     (analysis_tables/narrative_map.json maps effect -> consensus label).
  B. censored likelihood: units with |r_sim| > 0.95 enter as censored
     observations (z_sim known only to exceed arctanh(0.95) in sign
     direction) instead of point observations.
  C. transcription measurement error: z_sim gets a per-unit measurement
     variance (z_A - z_B)^2 / 4 estimated from the disagreement between the
     two transcription variants (imputed at the median where a variant is
     missing), added to the residual variance.

All three share the main two-model specification (effect random intercept,
model indicator, measurement-error latents for both human anchors).

Usage (from the repository root; about 30-60 minutes per fit):

    python code/fit_model_treatments.py

Outputs analysis_tables/model_treatments_posteriors.csv.
"""
import json
import os

import numpy as np
import pandas as pd

AT = "analysis_tables"
CLIP = float(np.arctanh(0.95))


def load_units():
    ud = pd.read_csv(f"{AT}/P2_analysis_units.csv")
    conf = ud.dropna(subset=["z_sim", "z_bare", "z_o", "z_r", "var_o", "var_r"]).reset_index(drop=True)
    # per-unit transcription variance from the two variants' cell tables
    cells = pd.concat([pd.read_csv(f"{AT}/P2_cells_g{i}.csv") for i in range(3)])
    cells = cells[cells.persona == "none"]
    a0 = pd.read_csv(f"{AT}/A0_cell_effects.csv")
    a0 = a0[a0.persona == "none"]
    a1 = pd.read_csv(f"{AT}/A1_cell_effects.csv")
    a1 = a1[a1.persona == "none"]
    allc = pd.concat([c[["effect", "variant", "model", "r_sim"]] for c in (cells, a0, a1)])
    allc["z"] = np.arctanh(allc.r_sim.clip(-0.99, 0.99))
    piv = allc.pivot_table(index=["effect", "model"], columns="variant", values="z", aggfunc="mean")
    dz2 = (piv.get("A") - piv.get("B")) ** 2 / 4.0
    conf = conf.set_index(["effect", "model"])
    conf["var_meas"] = dz2
    conf["var_meas"] = conf.var_meas.fillna(float(conf.var_meas.dropna().median()))
    conf = conf.reset_index()
    narr = json.load(open(f"{AT}/narrative_map.json"))
    conf["is_orig"] = (conf.effect.map(narr) == "original_dominates").astype(float)
    return conf


def fit(conf, treatment):
    import arviz as az
    import pymc as pm
    eff_idx, eff_names = pd.factorize(conf.effect)
    mod_idx, _ = pd.factorize(conf.model)
    em = conf.drop_duplicates("effect").set_index("effect").loc[eff_names]
    with pm.Model():
        z_o_true = pm.Normal("z_o_true", mu=em.z_o.values, sigma=np.sqrt(em.var_o.values), shape=len(eff_names))
        z_r_true = pm.Normal("z_r_true", mu=em.z_r.values, sigma=np.sqrt(em.var_r.values), shape=len(eff_names))
        D = z_o_true - z_r_true
        b0 = pm.Normal("b0", 0, 1)
        b_bare = pm.Normal("b_bare", 0, 1)
        b_rep = pm.Normal("b_rep", 0, 1)
        b_pub = pm.Normal("b_pub", 0, 1)
        b_model = pm.Normal("b_model", 0, 1)
        sd_eff = pm.HalfNormal("sd_eff", 0.5)
        u = pm.Normal("u_eff", 0, sd_eff, shape=len(eff_names))
        sigma = pm.HalfNormal("sigma", 1)
        mu = (b0 + b_bare * conf.z_bare.values + b_rep * z_r_true[eff_idx]
              + b_pub * D[eff_idx] + b_model * mod_idx + u[eff_idx])
        extra = []
        if treatment == "narrative_stratified":
            d_orig = pm.Normal("d_orig", 0, 1)
            mu = mu + d_orig * conf.is_orig.values * D[eff_idx]
            extra = ["d_orig"]
            pm.Normal("y", mu, sigma, observed=conf.z_sim.values)
        elif treatment == "censored":
            hi = (conf.r_sim > 0.95).values
            lo = (conf.r_sim < -0.95).values
            y = np.where(hi, CLIP, np.where(lo, -CLIP, conf.z_sim.values))
            pm.Censored("y", pm.Normal.dist(mu, sigma),
                        lower=np.where(lo, -CLIP, -1e6), upper=np.where(hi, CLIP, 1e6), observed=y)
        elif treatment == "transcription_error":
            pm.Normal("y", mu, pm.math.sqrt(sigma**2 + conf.var_meas.values), observed=conf.z_sim.values)
        idata = pm.sample(1500, tune=1000, chains=4, cores=1,
                          target_accept=0.95, random_seed=11, progressbar=False)
    s = az.summary(idata, var_names=["b_bare", "b_rep", "b_pub", "sigma", "sd_eff"] + extra, hdi_prob=0.95)
    rows = [{"analysis": treatment, "param": v, "mean": s.loc[v, "mean"],
             "hdi_2.5": s.loc[v, "hdi_2.5%"], "hdi_97.5": s.loc[v, "hdi_97.5%"]} for v in s.index]
    bp = idata.posterior.b_pub.values.ravel()
    rows.append({"analysis": treatment, "param": "P_pub_gt0", "mean": float((bp > 0).mean())})
    if treatment == "narrative_stratified":
        bo = bp + idata.posterior.d_orig.values.ravel()
        rows.append({"analysis": treatment, "param": "b_pub_orig_group", "mean": float(bo.mean()),
                     "hdi_2.5": float(np.percentile(bo, 2.5)), "hdi_97.5": float(np.percentile(bo, 97.5)),
                     "P_gt0": float((bo > 0).mean())})
    return rows


if __name__ == "__main__":
    conf = load_units()
    rows = []
    for t in ["narrative_stratified", "censored", "transcription_error"]:
        rows += fit(conf, t)
        print(t, "done")
        pd.DataFrame(rows).to_csv(f"{AT}/model_treatments_posteriors.csv", index=False)
    print("saved", f"{AT}/model_treatments_posteriors.csv")
