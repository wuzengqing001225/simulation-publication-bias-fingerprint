"""Fit the main measurement-error model of the paper.

Reads analysis_tables/P2_analysis_units.csv (one row per effect x model),
drops rows without a defined bare task (the estimation sample, n = 71),
and fits the three-coefficient Bayesian measurement-error regression
described in the paper's method section and Appendix G.

Usage (from the repository root):

    python code/fit_main_model.py             # main three-coefficient model
    python code/fit_main_model.py --no-bare   # same 71 units, bare-task control dropped
                                              # (reproduces analysis_tables/nobare_posteriors.csv)

Outputs: posterior summary to stdout and P2_main_model_idata.nc
(ArviZ InferenceData) in the working directory. The posterior shipped
with the repository (analysis_tables/P2_main_model_idata.nc) was produced
by this model with random_seed=11.
"""

import sys
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

NO_BARE = "--no-bare" in sys.argv
UNITS_CSV = "analysis_tables/P2_analysis_units.csv"

ud = pd.read_csv(UNITS_CSV)
conf = ud.dropna(subset=["z_sim", "z_bare", "z_o", "z_r", "var_o", "var_r"]).reset_index(drop=True)
print(f"estimation sample: {len(conf)} units, {conf.effect.nunique()} effects")

eff_idx, eff_names = pd.factorize(conf.effect)
mod_idx, _ = pd.factorize(conf.model)
# one latent anchor pair per EFFECT (shared across the effect's model rows),
# mapped to rows via eff_idx
eff_meta = conf.drop_duplicates("effect").set_index("effect").loc[eff_names]

with pm.Model() as m:
    # human anchors as latent true values with known sampling variance 1/(n-3)
    z_o_true = pm.Normal("z_o_true", mu=eff_meta.z_o.values, sigma=np.sqrt(eff_meta.var_o.values), shape=len(eff_names))
    z_r_true = pm.Normal("z_r_true", mu=eff_meta.z_r.values, sigma=np.sqrt(eff_meta.var_r.values), shape=len(eff_names))
    D_pub = z_o_true - z_r_true

    b0 = pm.Normal("b0", 0, 1)
    if not NO_BARE:
        b_bare = pm.Normal("b_bare", 0, 1)  # model's own bias
    b_rep = pm.Normal("b_rep", 0, 1)       # replicable human effect
    b_pub = pm.Normal("b_pub", 0, 1)       # original-only inflation (the fingerprint)
    b_model = pm.Normal("b_model", 0, 1)   # model indicator (Sonnet 4.5 vs Sonnet 5)
    sd_eff = pm.HalfNormal("sd_eff", 0.5)
    u_eff = pm.Normal("u_eff", 0, sd_eff, shape=len(eff_names))
    sigma = pm.HalfNormal("sigma", 1)

    bare_term = 0.0 if NO_BARE else b_bare * conf.z_bare.values
    mu = (b0 + bare_term + b_rep * z_r_true[eff_idx]
          + b_pub * D_pub[eff_idx] + b_model * mod_idx + u_eff[eff_idx])
    pm.Normal("y", mu, sigma, observed=conf.z_sim.values)

    idata = pm.sample(1500, tune=1000, chains=4, cores=1, target_accept=0.95,
                      random_seed=11, progressbar=True)

varnames = ["b_rep", "b_pub", "b_model", "b0", "sd_eff", "sigma"]
if not NO_BARE:
    varnames.insert(0, "b_bare")
summ = az.summary(idata, var_names=varnames, hdi_prob=0.95)
print(summ.round(3).to_string())
print("\ndivergences:", int(idata.sample_stats.diverging.sum().values))
out_nc = "nobare_model_idata.nc" if NO_BARE else "P2_main_model_idata.nc"
idata.to_netcdf(out_nc)
print("saved", out_nc)
