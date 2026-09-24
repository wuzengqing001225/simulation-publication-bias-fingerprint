"""Reproduce the ceiling-remediation refits (paper appendix on the ceiling rerun).

Pipeline, per model family:
  1. Fisher-z aggregate the two transcription variants of each rerun effect
     (CEIL_*_cells.csv) into one effect-level (or effect-model-level)
     simulated value, and take the rerun bare value from CEIL_*_bare.csv.
  2. Build a hybrid analysis table: start from the family's analysis units
     and replace z_sim and z_bare for the rerun effects (27 effect-model
     units for the two Claude models, 14 effect units each for GPT and
     DeepSeek). The hybrid tables are written as CEIL_<family>_units.csv
     with a `replaced` flag.
  3. Refit the measurement-error model on the hybrid table: the Claude fit
     is the two-model specification of the main paper (model indicator and
     effect random intercept); the GPT and DeepSeek fits are the
     single-model specification used in the cross-family section.
  4. Write posterior summaries (mean, 95% HDI, P(beta_pub > 0), residual
     scales) to CEIL_refit_posteriors.csv.

Optionally, --responses <main_resp.json> [...] computes the arm-level
variance table (CEIL_arm_variance.csv) behind the within-arm SD statistics
quoted in the appendix. Raw response files are available from the authors
on request; the four cell/bare CSVs shipped in analysis_tables/ are
sufficient for steps 1-4.

Usage (from the repository root; about 30-50 minutes for the three fits):

    python ceiling_kit/ceiling_refit.py
    python ceiling_kit/ceiling_refit.py --responses ceil_main_resp_*.json
"""

import argparse, glob, json, os, re
import numpy as np
import pandas as pd

AT = "analysis_tables"
CEIL_EFFECTS = None  # derived from the cells tables

FAMILIES = {
    "claude": {"units": f"{AT}/P2_analysis_units.csv", "two_model": True,
               "cells": [f"{AT}/CEIL_sonnet5_cells.csv", f"{AT}/CEIL_sonnet45_cells.csv"],
               "bare": [f"{AT}/CEIL_sonnet5_bare.csv", f"{AT}/CEIL_sonnet45_bare.csv"]},
    "gpt": {"units": "xmodel_results/XM_analysis_units.csv", "two_model": False,
            "cells": [f"{AT}/CEIL_gpt_cells.csv"], "bare": [f"{AT}/CEIL_gpt_bare.csv"]},
    "ds": {"units": "xmodel_results/deepseek/DS_analysis_units.csv", "two_model": False,
           "cells": [f"{AT}/CEIL_ds_cells.csv"], "bare": [f"{AT}/CEIL_ds_bare.csv"]},
}

MODEL_KEY = {"sonnet5": "sonnet5", "sonnet45": "sonnet45"}  # cells 'model' column values (Claude)


def z(r):
    return np.arctanh(np.clip(r, -0.99, 0.99))


def build_hybrid(fam, cfg):
    ud = pd.read_csv(cfg["units"])
    conf = ud.dropna(subset=["z_sim", "z_bare", "z_o", "z_r", "var_o", "var_r"]).reset_index(drop=True)
    cells = pd.concat([pd.read_csv(f) for f in cfg["cells"]])
    cells["zv"] = z(cells.r_sim)
    bare = pd.concat([pd.read_csv(f) for f in cfg["bare"]])
    bare["zb"] = z(bare.r_bare)
    if cfg["two_model"]:
        # variant aggregation within effect x model, replacement keyed on both
        newz = cells.groupby(["effect", "model"]).zv.mean()
        newb = bare.set_index(["effect", "model"]).zb
        key = list(zip(conf.effect, conf.model))
        conf["z_sim_r"] = [newz.get(k, conf.z_sim.iloc[i]) for i, k in enumerate(key)]
        conf["z_bare_r"] = [newb.get(k, conf.z_bare.iloc[i]) for i, k in enumerate(key)]
        conf["replaced"] = [k in newz.index for k in key]
    else:
        newz = cells.groupby("effect").zv.mean()
        newb = dict(zip(bare.effect, bare.zb))
        conf["z_sim_r"] = conf.apply(lambda r: newz.get(r.effect, r.z_sim), axis=1)
        conf["z_bare_r"] = conf.apply(lambda r: newb.get(r.effect, r.z_bare), axis=1)
        conf["replaced"] = conf.effect.isin(newz.index)
    conf.to_csv(f"{AT}/CEIL_{fam}_units.csv", index=False)
    print(f"{fam}: {len(conf)} units, {int(conf.replaced.sum())} replaced -> {AT}/CEIL_{fam}_units.csv")
    return conf


def fit(fam, conf, two_model):
    import pymc as pm
    import arviz as az
    eff_idx, eff_names = pd.factorize(conf.effect)
    em = conf.drop_duplicates("effect").set_index("effect").loc[eff_names]
    with pm.Model():
        z_o_true = pm.Normal("z_o_true", mu=em.z_o.values, sigma=np.sqrt(em.var_o.values), shape=len(eff_names))
        z_r_true = pm.Normal("z_r_true", mu=em.z_r.values, sigma=np.sqrt(em.var_r.values), shape=len(eff_names))
        D = z_o_true - z_r_true
        b0 = pm.Normal("b0", 0, 1)
        b_bare = pm.Normal("b_bare", 0, 1)
        b_rep = pm.Normal("b_rep", 0, 1)
        b_pub = pm.Normal("b_pub", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        mu = b0 + b_bare * conf.z_bare_r.values + b_rep * z_r_true[eff_idx] + b_pub * D[eff_idx]
        if two_model:
            b_model = pm.Normal("b_model", 0, 1)
            sd_eff = pm.HalfNormal("sd_eff", 0.5)
            u = pm.Normal("u_eff", 0, sd_eff, shape=len(eff_names))
            mod_idx, _ = pd.factorize(conf.model)
            mu = mu + b_model * mod_idx + u[eff_idx]
        pm.Normal("y", mu, sigma, observed=conf.z_sim_r.values)
        idata = pm.sample(1500, tune=1000, chains=4, cores=1,
                          target_accept=0.95, random_seed=11, progressbar=False)
    import arviz as az
    vars_ = ["b_bare", "b_rep", "b_pub", "sigma"] + (["sd_eff", "b_model"] if two_model else [])
    s = az.summary(idata, var_names=vars_, hdi_prob=0.95)
    rows = [{"family": fam, "param": v, "mean": s.loc[v, "mean"],
             "hdi_2.5": s.loc[v, "hdi_2.5%"], "hdi_97.5": s.loc[v, "hdi_97.5%"]} for v in s.index]
    p_gt0 = float((idata.posterior.b_pub.values.ravel() > 0).mean())
    rows.append({"family": fam, "param": "P_pub_gt0", "mean": p_gt0, "hdi_2.5": np.nan, "hdi_97.5": np.nan})
    idata.to_netcdf(f"{AT}/CEIL_{fam}_idata.nc")
    print(fam, "fit done; P(b_pub>0) =", round(p_gt0, 3))
    return rows


RATING = re.compile(r"RATING\s*[:=]\s*(\d{1,3})", re.I)
ITEM = re.compile(r"ITEM\s*(\d+)\s*[:=]\s*(\d{1,3})", re.I)


def arm_variance(resp_files):
    rows = []
    for fn in resp_files:
        raw = pd.DataFrame(json.load(open(fn)))
        for (eff, var, cond, mod), grp in raw.groupby(["effect", "variant", "cond", "model"], dropna=False):
            vals = []
            for t in grp.text:
                its = ITEM.findall(t or "")
                if its:
                    vals.append(np.mean([int(v) for _, v in its]))
                else:
                    m = RATING.findall(t or "")
                    if m:
                        vals.append(int(m[-1]))
            if len(vals) >= 5:
                rows.append({"file": os.path.basename(fn), "model": mod, "effect": eff,
                             "variant": var, "cond": cond, "n": len(vals),
                             "arm_mean": np.mean(vals), "arm_sd": np.std(vals)})
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{AT}/CEIL_arm_variance.csv", index=False)
    for mod, g in tab.groupby("model"):
        print(f"{mod}: median within-arm SD {g.arm_sd.median():.1f} | share < 3: {(g.arm_sd < 3).mean():.2f}")
    return tab


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", nargs="*", default=None,
                    help="raw ceil_main_resp.json files (optional; for the arm variance table)")
    ap.add_argument("--skip-fit", action="store_true", help="only build hybrid tables (and variance table)")
    args = ap.parse_args()

    all_rows = []
    for fam, cfg in FAMILIES.items():
        conf = build_hybrid(fam, cfg)
        if not args.skip_fit:
            all_rows += fit(fam, conf, cfg["two_model"])
    if all_rows:
        pd.DataFrame(all_rows).to_csv(f"{AT}/CEIL_refit_posteriors.csv", index=False)
        print(f"posteriors -> {AT}/CEIL_refit_posteriors.csv")
    if args.responses:
        arm_variance([f for pat in args.responses for f in sorted(glob.glob(pat))])
