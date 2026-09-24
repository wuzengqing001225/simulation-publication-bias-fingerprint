"""Design power simulation for the publication-fingerprint coefficient.

Reproduces analysis_tables/power_simulation.csv (the observed design,
71 units, 38 effects) and analysis_tables/power_design_curve.csv (the
design-scaling curve used in the appendix on design precision).

Method. Data are simulated from the fitted main model at the observed
design: the observed z_bare, z_r, and D_pub = z_o - z_r values, the
effect-level intercept spread (sd_eff = 0.756) and residual scale
(sigma = 0.593) fixed at the posterior means of the corrected main fit,
beta_bare = 0.37, beta_rep = 1.0, beta_model = -0.065, intercept 0.393,
while the true beta_pub varies. Each simulated dataset is refit with a
linear mixed model (random intercept per effect):

    y ~ z_bare + z_r + D_pub + mod_i,  groups = effect

which on the real data gives a slightly WIDER interval for the D_pub
coefficient than the Bayesian measurement-error model, so the power
estimates below are mildly conservative. Detection = the 95% CI for
D_pub excludes zero. Equivalence = the CI lies entirely inside
(-0.10, +0.10), the pre-specified equivalence region. 500 simulations
per condition, random seed 11.

For the design-scaling curve, designs with more effects are built by
resampling the observed 38 effects' (z_bare, z_r, D_pub) triples with
replacement and giving each resampled effect two model rows, so the
covariate distribution matches the observed design at every size.

Usage (from the repository root):

    python code/power_simulation.py
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

UNITS_CSV = "analysis_tables/P2_analysis_units.csv"
SD_EFF, SIGMA, B0 = 0.756, 0.593, 0.393
B_BARE, B_REP, B_MODEL = 0.37, 1.0, -0.065
K = 500
rng = np.random.default_rng(11)

ud = pd.read_csv(UNITS_CSV)
conf = ud.dropna(subset=["z_sim", "z_bare", "z_o", "z_r", "var_o", "var_r"]).reset_index(drop=True)
conf["D_pub"] = conf.z_o - conf.z_r
conf["mod_i"] = (conf.model == conf.model.unique()[1]).astype(int)
eff_tab = conf.groupby("effect").agg(z_bare=("z_bare", "mean"), z_r=("z_r", "first"),
                                     D_pub=("D_pub", "first")).reset_index()


def fit_once(df):
    f = smf.mixedlm("y ~ z_bare + z_r + D_pub + mod_i", df, groups=df.effect).fit(reml=True)
    lo, hi = f.conf_int().loc["D_pub"]
    return (lo > 0 or hi < 0), (lo > -0.10 and hi < 0.10)


def sim_observed(b_pub):
    effs, codes = np.unique(conf.effect, return_inverse=True)
    u = rng.normal(0, SD_EFF, len(effs))[codes]
    y = (B0 + B_BARE * conf.z_bare + B_REP * conf.z_r + b_pub * conf.D_pub
         + B_MODEL * conf.mod_i + u + rng.normal(0, SIGMA, len(conf)))
    try:
        return fit_once(conf.assign(y=y))
    except Exception:
        return None


def make_design(n_eff):
    idx = rng.integers(0, len(eff_tab), n_eff)
    rows = []
    for j, i in enumerate(idx):
        for m in (0, 1):
            rows.append({"effect": f"e{j}", "z_bare": eff_tab.z_bare[i],
                         "z_r": eff_tab.z_r[i], "D_pub": eff_tab.D_pub[i], "mod_i": m})
    return pd.DataFrame(rows)


def sim_scaled(n_eff, b_pub):
    des = make_design(n_eff)
    effs = des.effect.unique()
    u = dict(zip(effs, rng.normal(0, SD_EFF, len(effs))))
    y = (B0 + B_BARE * des.z_bare + B_REP * des.z_r + b_pub * des.D_pub
         + B_MODEL * des.mod_i + des.effect.map(u) + rng.normal(0, SIGMA, len(des)))
    try:
        return fit_once(des.assign(y=y))
    except Exception:
        return None


if __name__ == "__main__":
    rows = []
    for bt in [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]:
        res = [r for r in (sim_observed(bt) for _ in range(K)) if r]
        rows.append({"b_pub_true": bt, "p_detect": np.mean([r[0] for r in res]),
                     "p_equiv": np.mean([r[1] for r in res]), "n_sims": len(res)})
        print(rows[-1])
    pd.DataFrame(rows).to_csv("power_simulation.csv", index=False)

    rows = []
    for n_eff in [38, 75, 150, 300]:
        for bt in [0.0, 0.25, 0.50, 1.00]:
            res = [r for r in (sim_scaled(n_eff, bt) for _ in range(K)) if r]
            rows.append({"n_effects": n_eff, "b_pub_true": bt,
                         "p_detect": np.mean([r[0] for r in res]),
                         "p_equiv": np.mean([r[1] for r in res]), "n_sims": len(res)})
            print(rows[-1])
    pd.DataFrame(rows).to_csv("power_design_curve.csv", index=False)
    print("saved power_simulation.csv, power_design_curve.csv")
