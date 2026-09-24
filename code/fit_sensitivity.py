"""Reproduce the sensitivity-subset fits of the main regression.

Fits the same three-coefficient measurement-error model as
code/fit_main_model.py on four pre-specified subsets of the estimation
sample, plus the persona-inclusive outcome:

  full             all 71 units (matches the main fit)
  excl_ceiling     units with |r_sim| <= 0.95
  high_precision   units whose replication has n_r >= 500
  uniform_wave     units from the uniform-protocol final wave only
  persona_inclusive same units, outcome includes persona runs (z from r_sim_all)

Usage (from the repository root):

    python code/fit_sensitivity.py

Output: analysis_tables/sensitivity_posteriors.csv (posterior mean,
95% HDI, and P(beta_pub > 0) per subset and coefficient).
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

UNITS_CSV = "analysis_tables/P2_analysis_units.csv"

ud = pd.read_csv(UNITS_CSV)
conf = ud.dropna(subset=["z_sim", "z_bare", "z_o", "z_r", "var_o", "var_r"]).reset_index(drop=True)


def fit(sub, outcome="z_sim"):
    sub = sub.reset_index(drop=True)
    eff_idx, eff_names = pd.factorize(sub.effect)
    mod_idx, _ = pd.factorize(sub.model)
    em = sub.drop_duplicates("effect").set_index("effect").loc[eff_names]
    with pm.Model():
        z_o_true = pm.Normal("z_o_true", mu=em.z_o.values, sigma=np.sqrt(em.var_o.values), shape=len(eff_names))
        z_r_true = pm.Normal("z_r_true", mu=em.z_r.values, sigma=np.sqrt(em.var_r.values), shape=len(eff_names))
        D_pub = z_o_true - z_r_true
        b0 = pm.Normal("b0", 0, 1)
        b_bare = pm.Normal("b_bare", 0, 1)
        b_rep = pm.Normal("b_rep", 0, 1)
        b_pub = pm.Normal("b_pub", 0, 1)
        b_model = pm.Normal("b_model", 0, 1)
        sd_eff = pm.HalfNormal("sd_eff", 0.5)
        u_eff = pm.Normal("u_eff", 0, sd_eff, shape=len(eff_names))
        sigma = pm.HalfNormal("sigma", 1)
        mu = (b0 + b_bare * sub.z_bare.values + b_rep * z_r_true[eff_idx]
              + b_pub * D_pub[eff_idx] + b_model * mod_idx + u_eff[eff_idx])
        pm.Normal("y", mu, sigma, observed=sub[outcome].values)
        idata = pm.sample(1500, tune=1000, chains=4, cores=1, target_accept=0.95,
                          random_seed=11, progressbar=False)
    rows = []
    for v in ["b_bare", "b_rep", "b_pub", "b_model", "sd_eff", "sigma"]:
        x = idata.posterior[v].values.ravel()
        h = az.hdi(idata, var_names=[v], hdi_prob=0.95)[v].values.ravel()
        rows.append({"param": v, "mean": round(float(x.mean()), 4),
                     "hdi_2.5": round(float(h[0]), 4), "hdi_97.5": round(float(h[1]), 4)})
    p = float((idata.posterior.b_pub.values.ravel() > 0).mean())
    rows.append({"param": "P_pub_gt0", "mean": round(p, 4), "hdi_2.5": np.nan, "hdi_97.5": np.nan})
    return rows


if __name__ == "__main__":
    conf2 = conf.copy()
    conf2["z_all"] = np.arctanh(np.clip(conf2.r_sim_all, -0.99, 0.99))
    jobs = [
        ("full", conf, "z_sim"),
        ("excl_ceiling", conf[conf.r_sim.abs() <= 0.95], "z_sim"),
        ("high_precision", conf[conf.n_r >= 500], "z_sim"),
        ("uniform_wave", conf[conf.wave != "A0A1"], "z_sim"),
        ("persona_inclusive", conf2, "z_all"),
    ]
    out = []
    for name, sub, oc in jobs:
        print(f"{name}: {len(sub)} units", flush=True)
        for r in fit(sub, outcome=oc):
            out.append({"analysis": name, **r})
        pd.DataFrame(out).to_csv("analysis_tables/sensitivity_posteriors.csv", index=False)
        print(f"{name} done", flush=True)
    print("saved analysis_tables/sensitivity_posteriors.csv")
