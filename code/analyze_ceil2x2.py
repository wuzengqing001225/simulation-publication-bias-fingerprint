"""Reproduce the fame-by-content 2x2 completion analysis (structure experiment).

Pipeline (the pair is the statistical unit throughout):
  1. subject scores -> per-pair simulated correlations, per model, for the 16
     new pairs (protocols/ceil2x2_pairs.csv), with computability rules
     (n >= 20 scored subjects and nonzero variance on both constructs).
     Writes analysis_tables/ceil2x2_results.csv.
  2. Merge with the original battery's pair table
     (analysis_tables/B2_threeway.csv, the two Claude models) to form the
     full 2x2: famous/undiscussed x political/nonpolitical.
  3. Excess exaggeration per pair-model observation:
     excess = |r_sim| - |r_human|. Pair-level values average the two models.
  4. Cell means, both weightings (equal-weight cell means and pair-level).
  5. Main effects from a pair-level OLS regression
     excess ~ fame + political (n = 33 computable pairs), the estimand
     reported in the paper, plus the equal-weight cell-mean contrasts.
  6. Mann-Whitney tests: fame effect over all pairs (one-sided), and the
     new-cells-only contrast (famous nonpolitical vs undiscussed political).
  7. Winner's-curse simulation for the undiscussed-pair selection at the
     panel's precision (n = 2,058, SE_r ~ 0.022): magnitude-matched and
     worst-case largest-first selection.
  8. Computable-pair counts and the ability-variance-collapse diagnostic.

All numbers quoted in the paper's structure section and Appendix on the
completed design are emitted to analysis_tables/ceil2x2_stats.json.

Usage (from the repository root):

    python code/analyze_ceil2x2.py
"""

import json

import numpy as np
import pandas as pd
from scipy import stats as st

AT = "analysis_tables"
RNG = np.random.default_rng(11)

GROUP_FLAGS = {  # group -> (fame, political)
    "famous": (1, 1),        # original battery famous cell (political-heavy)
    "undiscussed": (0, 0),   # original battery undiscussed cell
    "famous_nonpol": (1, 0),
    "undisc_pol": (0, 1),
}


def new_pair_correlations():
    """Step 1: subject scores -> per-pair r_sim per model for the 16 new pairs."""
    pairs = pd.read_csv("protocols/ceil2x2_pairs.csv")
    sdf = pd.read_csv(f"{AT}/ceil2x2_subject_scores.csv")
    out = []
    for _, pr in pairs.iterrows():
        for mk, g in sdf.groupby("model"):
            x, y = g[pr.construct_x], g[pr.construct_y]
            m = x.notna() & y.notna()
            computable = (m.sum() >= 20) and (x[m].std() > 0) and (y[m].std() > 0)
            r_sim = float(np.corrcoef(x[m], y[m])[0, 1]) if computable else np.nan
            out.append({"group": pr.group, "construct_x": pr.construct_x,
                        "construct_y": pr.construct_y, "r_human": pr.r_human,
                        "model": mk, "r_sim": r_sim, "n_sim": int(m.sum())})
    res = pd.DataFrame(out)
    res["excess"] = res.r_sim.abs() - res.r_human.abs()
    res.to_csv(f"{AT}/ceil2x2_results.csv", index=False)
    return res


def full_design(res):
    """Step 2-3: merge with the original battery cells; pair-level table."""
    tw = pd.read_csv(f"{AT}/B2_threeway.csv")
    tw = tw[tw.model.isin(["sonnet5", "sonnet45"])].dropna(subset=["r_human", "r_sim"]).copy()
    tw["excess"] = tw.r_sim.abs() - tw.r_human.abs()
    obs = pd.concat([
        tw[["group", "pair", "model", "excess"]].rename(columns={"pair": "pair_id"})
        if "pair" in tw.columns else
        tw.assign(pair_id=tw.construct_x + "|" + tw.construct_y)[["group", "pair_id", "model", "excess"]],
        res.dropna(subset=["excess"]).assign(pair_id=res.construct_x + "|" + res.construct_y)[
            ["group", "pair_id", "model", "excess"]],
    ], ignore_index=True)
    obs["fame"] = obs.group.map(lambda g: GROUP_FLAGS[g][0])
    obs["pol"] = obs.group.map(lambda g: GROUP_FLAGS[g][1])
    pair = obs.groupby(["pair_id", "group", "fame", "pol"], as_index=False).excess.mean()
    return obs, pair


def main_effects(obs, pair):
    """Step 4-6: cell means, regression main effects, Mann-Whitney tests."""
    cm = obs.groupby(["fame", "pol"]).excess.mean()
    stats = {"cell_means_obs": {f"fame{f}_pol{p}": round(float(v), 4) for (f, p), v in cm.items()},
             "cell_n_obs": {f"fame{f}_pol{p}": int(n) for (f, p), n in obs.groupby(["fame", "pol"]).size().items()}}
    # equal-weight cell-mean contrasts
    stats["fame_equal_weight"] = round(float((cm[1, 1] + cm[1, 0]) / 2 - (cm[0, 1] + cm[0, 0]) / 2), 4)
    stats["political_equal_weight"] = round(float((cm[1, 1] + cm[0, 1]) / 2 - (cm[1, 0] + cm[0, 0]) / 2), 4)
    # pair-level OLS: excess ~ fame + political (the paper's estimand)
    X = np.column_stack([np.ones(len(pair)), pair.fame, pair.pol])
    b, *_ = np.linalg.lstsq(X, pair.excess.values, rcond=None)
    resid = pair.excess.values - X @ b
    cov = (resid @ resid / (len(pair) - 3)) * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    pv = 2 * st.t.sf(np.abs(b / se), len(pair) - 3)
    stats["pair_regression"] = {"n_pairs": int(len(pair)),
                                "fame": {"coef": round(float(b[1]), 4), "se": round(float(se[1]), 4), "p": round(float(pv[1]), 4)},
                                "political": {"coef": round(float(b[2]), 4), "se": round(float(se[2]), 4), "p": round(float(pv[2]), 4)}}
    # Mann-Whitney, pair as unit
    f_, u_ = pair[pair.fame == 1].excess, pair[pair.fame == 0].excess
    stats["fame_mw_onesided_p"] = round(float(st.mannwhitneyu(f_, u_, alternative="greater").pvalue), 4)
    fn = pair[(pair.fame == 1) & (pair.pol == 0)].excess
    up = pair[(pair.fame == 0) & (pair.pol == 1)].excess
    stats["new_cells"] = {"famous_nonpol_mean": round(float(fn.mean()), 4), "n_famous_nonpol": int(len(fn)),
                          "undisc_pol_mean": round(float(up.mean()), 4), "n_undisc_pol": int(len(up)),
                          "mw_onesided_p": round(float(st.mannwhitneyu(fn, up, alternative="greater").pvalue), 4)}
    return stats


def winners_curse(n_panel=2058, K=2926, n_sims=2000, k_top=10):
    """Step 7: selection-bias bound for magnitude-based undiscussed-pair selection."""
    se = 1.0 / np.sqrt(n_panel - 3)
    match_bias, worst_bias = [], []
    for _ in range(n_sims):
        true_r = RNG.uniform(0.0, 0.5, K) * RNG.choice([-1, 1], K)
        obs_r = true_r + RNG.normal(0, se, K)
        targets = RNG.uniform(0.08, 0.42, k_top)  # magnitude-matched selection
        idx = [int(np.argmin(np.abs(np.abs(obs_r) - t))) for t in targets]
        match_bias.append(np.mean(np.abs(obs_r[idx]) - np.abs(true_r[idx])))
        top = np.argsort(-np.abs(obs_r))[:k_top]   # worst case: largest first
        worst_bias.append(np.mean(np.abs(obs_r[top]) - np.abs(true_r[top])))
    return {"se_r": round(float(se), 4), "matched_bias": round(float(np.mean(match_bias)), 4),
            "worst_case_bias": round(float(np.mean(worst_bias)), 4)}


def computability(res):
    """Step 8: computable-pair counts and the ability-collapse diagnostic."""
    per_pair = res.groupby(["group", "construct_x", "construct_y"]).r_sim.apply(lambda s: s.notna().any())
    return {"new_pairs_total": int(len(per_pair)),
            "new_pairs_computable": int(per_pair.sum()),
            "uncomputable_pairs": ["%s|%s" % (x, y) for (_, x, y), ok in per_pair.items() if not ok]}


if __name__ == "__main__":
    res = new_pair_correlations()
    obs, pair = full_design(res)
    stats = main_effects(obs, pair)
    stats["winners_curse"] = winners_curse()
    stats["computability"] = computability(res)
    json.dump(stats, open(f"{AT}/ceil2x2_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))
    print(f"wrote {AT}/ceil2x2_results.csv and {AT}/ceil2x2_stats.json")
