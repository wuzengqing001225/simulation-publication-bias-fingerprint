"""B2 analysis: construct matrices, three-way comparison, two-component decomposition."""
import numpy as np
import pandas as pd
from scipy import stats


def corr_matrix(df, constructs, min_n=30, min_sd=1e-9):
    """Pairwise-complete Pearson matrix over `constructs`; returns (R, N, sd)."""
    k = len(constructs)
    R = pd.DataFrame(np.nan, index=constructs, columns=constructs, dtype=float)
    N = pd.DataFrame(0, index=constructs, columns=constructs, dtype=int)
    for i, a in enumerate(constructs):
        R.loc[a, a] = 1.0
        for b in constructs[i + 1:]:
            m = df[[a, b]].dropna()
            n = len(m)
            N.loc[a, b] = N.loc[b, a] = n
            if n >= min_n and m[a].std(ddof=1) > min_sd and m[b].std(ddof=1) > min_sd:
                r = float(np.corrcoef(m[a], m[b])[0, 1])
                R.loc[a, b] = R.loc[b, a] = r
    sd = df[constructs].std(ddof=1)
    return R, N, sd


def fisher_z(r, eps=1e-9):
    r = np.clip(r, -1 + eps, 1 - eps)
    return np.arctanh(r)


def boot_ci_r(x, y, B=2000, seed=11, alpha=0.05):
    """Bootstrap CI for Pearson r on a paired sample."""
    m = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(m) < 10:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    xs, ys = m.x.to_numpy(), m.y.to_numpy()
    n = len(xs)
    out = np.empty(B)
    out[:] = np.nan
    for b in range(B):
        idx = rng.integers(0, n, n)
        xb, yb = xs[idx], ys[idx]
        if xb.std() > 1e-12 and yb.std() > 1e-12:
            out[b] = np.corrcoef(xb, yb)[0, 1]
    out = out[~np.isnan(out)]
    if len(out) < 100:
        return (np.nan, np.nan)
    return tuple(np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def build_threeway(pairs, sim_mats, sim_dfs, panel_lookup, icc_lookup):
    """One row per (pair, model). panel_lookup: (x,y)->dict(r,n,ci). """
    rows = []
    for grp, key in [("famous", "famous_pairs"), ("undiscussed", "undiscussed_pairs")]:
        for p in pairs[key]:
            x, y = p["construct_x"], p["construct_y"]
            pl = panel_lookup.get((x, y), {})
            for mk, R in sim_mats.items():
                rs = R.loc[x, y] if (x in R.index and y in R.columns) else np.nan
                dfm = sim_dfs[mk]
                sdx = dfm[x].std(ddof=1) if x in dfm else np.nan
                sdy = dfm[y].std(ddof=1) if y in dfm else np.nan
                nn = int(dfm[[x, y]].dropna().shape[0]) if (x in dfm and y in dfm) else 0
                ci = boot_ci_r(dfm[x], dfm[y]) if nn >= 10 else (np.nan, np.nan)
                rh = pl.get("r", np.nan)
                rows.append({
                    "pair_id": p["pair_id"], "group": grp, "construct_x": x, "construct_y": y,
                    "label": p.get("label"), "model": mk,
                    "lit_r": p.get("lit_r"), "lit_direction": p.get("lit_direction"),
                    "needs_citation_check": p.get("needs_citation_check", False),
                    "r_human": rh, "n_human": pl.get("n"),
                    "icc_x": icc_lookup.get(x), "icc_y": icc_lookup.get(y),
                    "r_sim": rs, "n_sim": nn, "sd_x_sim": sdx, "sd_y_sim": sdy,
                    "r_sim_ci_lo": ci[0], "r_sim_ci_hi": ci[1],
                    "r_gpt41mini_b0": p.get("sim_r"),
                    "abs_diff": abs(rs - rh) if pd.notna(rs) and pd.notna(rh) else np.nan,
                    "signed_diff": (rs - rh) if pd.notna(rs) and pd.notna(rh) else np.nan,
                    "abs_r_human": abs(rh) if pd.notna(rh) else np.nan,
                    "abs_r_sim": abs(rs) if pd.notna(rs) else np.nan,
                    "sign_flip": (bool(np.sign(rs) != np.sign(rh))
                                  if pd.notna(rs) and pd.notna(rh) else None),
                })
    return pd.DataFrame(rows)


def decompose(tw, min_abs_human=0.05, min_sd_ratio=None):
    """Global inflation index + selectivity index, per model.

    global_inflation_index : median over ALL usable pairs of |r_sim| / |r_human|
    excess_inflation (per pair) : |r_sim| - |r_human|   (additive, sd-free)
    selectivity_index : median excess among famous - median excess among undiscussed
                        (also reported on the log-ratio scale)
    """
    out = {}
    for mk, g in tw.groupby("model"):
        u = g[g.abs_r_human.notna() & g.abs_r_sim.notna() & (g.abs_r_human >= min_abs_human)].copy()
        u["ratio"] = u.abs_r_sim / u.abs_r_human
        u["log_ratio"] = np.log(np.clip(u.ratio, 1e-6, None))
        u["excess"] = u.abs_r_sim - u.abs_r_human
        fam = u[u.group == "famous"]
        und = u[u.group == "undiscussed"]
        gi = float(np.median(u.ratio))
        sel_add = float(np.median(fam.excess) - np.median(und.excess))
        sel_log = float(np.median(fam.log_ratio) - np.median(und.log_ratio))
        mw = stats.mannwhitneyu(fam.excess, und.excess, alternative="two-sided") \
            if len(fam) >= 3 and len(und) >= 3 else None
        mwa = stats.mannwhitneyu(fam.abs_diff, und.abs_diff, alternative="two-sided") \
            if len(fam) >= 3 and len(und) >= 3 else None
        out[mk] = {
            "n_pairs_used": int(len(u)), "n_famous": int(len(fam)), "n_undiscussed": int(len(und)),
            "global_inflation_index": round(gi, 4),
            "global_inflation_log_median": round(float(np.median(u.log_ratio)), 4),
            "median_excess_all": round(float(np.median(u.excess)), 4),
            "median_excess_famous": round(float(np.median(fam.excess)), 4),
            "median_excess_undiscussed": round(float(np.median(und.excess)), 4),
            "selectivity_index_additive": round(sel_add, 4),
            "selectivity_index_logratio": round(sel_log, 4),
            "median_absdiff_famous": round(float(np.median(fam.abs_diff)), 4),
            "median_absdiff_undiscussed": round(float(np.median(und.abs_diff)), 4),
            "mw_excess_U": (float(mw.statistic) if mw else None),
            "mw_excess_p": (float(mw.pvalue) if mw else None),
            "mw_absdiff_U": (float(mwa.statistic) if mwa else None),
            "mw_absdiff_p": (float(mwa.pvalue) if mwa else None),
            "n_sign_flips": int(g.sign_flip.fillna(False).sum()),
        }
    return out


def boot_selectivity(tw, model, B=4000, seed=5, min_abs_human=0.05):
    """Bootstrap CI for the selectivity index by resampling PAIRS within group."""
    g = tw[(tw.model == model) & tw.abs_r_human.notna() & tw.abs_r_sim.notna()
           & (tw.abs_r_human >= min_abs_human)].copy()
    g["excess"] = g.abs_r_sim - g.abs_r_human
    f = g[g.group == "famous"].excess.to_numpy()
    u = g[g.group == "undiscussed"].excess.to_numpy()
    if len(f) < 3 or len(u) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    d = np.empty(B)
    for b in range(B):
        d[b] = (np.median(rng.choice(f, len(f), replace=True))
                - np.median(rng.choice(u, len(u), replace=True)))
    return tuple(np.round(np.percentile(d, [2.5, 97.5]), 4))


def tidiness(R, N, min_n=30):
    """Significant-correlation share, mean |r|, eigenvalue concentration."""
    cs = list(R.index)
    vals, ps = [], []
    for i, a in enumerate(cs):
        for b in cs[i + 1:]:
            r, n = R.loc[a, b], N.loc[a, b]
            if pd.notna(r) and n >= min_n:
                vals.append(r)
                t = r * np.sqrt((n - 2) / max(1e-12, 1 - r ** 2))
                ps.append(2 * stats.t.sf(abs(t), n - 2))
    vals, ps = np.array(vals), np.array(ps)
    Rc = R.dropna(how="all").dropna(axis=1, how="all")
    Rc = Rc.loc[Rc.index, Rc.index].fillna(0.0).to_numpy()
    np.fill_diagonal(Rc, 1.0)
    ev = np.sort(np.linalg.eigvalsh((Rc + Rc.T) / 2))[::-1]
    ev = np.clip(ev, 0, None)
    return {"n_pairs": int(len(vals)), "sig_share": float((ps < .05).mean()),
            "mean_abs_r": float(np.abs(vals).mean()),
            "ev1_share": float(ev[0] / ev.sum()),
            "ev5_share": float(ev[:5].sum() / ev.sum())}
