"""Hierarchical-prior refit, simulation-based calibration (SBC), and a Bayesian power
simulation that uses the paper's actual measurement-error model (code/nm_models.py).

Usage (repository root):  python code/calibration_and_power.py
Outputs: analysis_tables/hierarchical_prior_posteriors.json, sbc_ranks.csv, power_bayes.csv
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, jax
import nm_models as nmm
from numpyro.infer import Predictive

AT = "analysis_tables"
ud = pd.read_csv(f"{AT}/P2_analysis_units.csv")
conf = ud.dropna(subset=["z_sim", "z_bare", "z_o", "z_r", "var_o", "var_r"]).reset_index(drop=True)

# 1. hierarchical (structural) prior on the latent anchors
mc = nmm.fit(conf, prior="hierarchical")
hs = nmm.summarize(mc, names=("b_bare", "b_rep", "b_pub", "sd_eff", "sigma", "mu_D", "tau_D"))
json.dump(hs, open(f"{AT}/hierarchical_prior_posteriors.json", "w"), indent=1)
print("hierarchical:", {k: round(v["mean"], 3) for k, v in hs.items() if isinstance(v, dict)}, flush=True)

# 2. SBC: draw parameters from the prior, simulate z_sim, refit, rank the truth
data, _ = nmm.prep(conf)
prior_pred = Predictive(nmm.model, num_samples=200)
pp = prior_pred(jax.random.PRNGKey(7), **data, y=None)
rows = []
L = 199
for s in range(200):
    y = np.asarray(pp["y"][s])
    if not np.all(np.isfinite(y)) or np.abs(y).max() > 20:
        continue
    m = nmm.fit(conf, y=y, seed=100 + s, warmup=500, samples=L, chains=1)
    post = m.get_samples()
    row = {"sim": s}
    for k in ["b_bare", "b_rep", "b_pub"]:
        row[f"rank_{k}"] = int((np.asarray(post[k]) < float(pp[k][s])).sum())
    rows.append(row)
    if s % 20 == 0:
        print("sbc", s, flush=True)
sbc = pd.DataFrame(rows)
sbc.to_csv(f"{AT}/sbc_ranks.csv", index=False)

# 3. Bayesian power: truth = posterior means of the main fit with b_pub set to a grid value
mc0 = nmm.fit(conf)
p0 = {k: float(np.asarray(v).mean()) for k, v in mc0.get_samples().items() if np.asarray(v).ndim == 1}
eff_idx, eff_names = pd.factorize(conf.effect)
em = conf.drop_duplicates("effect").set_index("effect").loc[eff_names]
mods = sorted(conf.model.unique()); mod_idx = (conf.model == mods[-1]).astype(float).values
rng = np.random.default_rng(11)
out = []
K = 100
for bt in [0.0, 0.5, 1.0]:
    det = eqv = 0
    for k in range(K):
        zo_t = rng.normal(em.z_o.values, np.sqrt(em.var_o.values)); zr_t = rng.normal(em.z_r.values, np.sqrt(em.var_r.values))
        u = rng.normal(0, p0["sd_eff"], len(eff_names))
        mu = (p0["b0"] + p0["b_bare"] * conf.z_bare.values + p0["b_rep"] * zr_t[eff_idx] + bt * (zo_t - zr_t)[eff_idx]
              + p0["b_model"] * mod_idx + u[eff_idx])
        y = rng.normal(mu, p0["sigma"])
        m = nmm.fit(conf, y=y, seed=1000 + k, warmup=500, samples=500, chains=2)
        v = np.asarray(m.get_samples()["b_pub"]); lo, hi = np.percentile(v, [2.5, 97.5])
        det += (lo > 0) or (hi < 0); eqv += (lo > -0.10) and (hi < 0.10)
    out.append({"b_pub_true": bt, "p_detect": det / K, "p_equiv": eqv / K, "n_sims": K})
    print("power", out[-1], flush=True)
pd.DataFrame(out).to_csv(f"{AT}/power_bayes.csv", index=False)
print("ALL DONE", flush=True)
