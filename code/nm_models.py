"""NumPyro implementation of the paper's measurement-error regression and its extensions.

The default configuration is the same model as code/fit_main_model.py (PyMC); it is
used for the reviewer-requested reanalyses because it is fast enough for simulation-
based calibration and a Bayesian power simulation.

    z_sim[i] = b0 + b_bare * z_bare[i] + b_rep * zr_true[e] + b_pub * (zo_true[e] - zr_true[e])
               + b_model * model[i] + X[i] @ gamma + u[e] + eps[i]

Options
  prior="functional"   zo_true ~ N(z_o, se_o), zr_true ~ N(z_r, se_r)            (paper's main model)
  prior="hierarchical" zr_true ~ N(mu_r, tau_r), D_true ~ N(mu_D, tau_D) as population
                        distributions, with z_o, z_r entering as observations       (structural version)
  X                    extra unit-level covariates (paradigm dummies, interactions)
  xnames               names for gamma entries
"""
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(4)


def model(eff_idx, mod_idx, z_bare, z_o, se_o, z_r, se_r, X, y=None, prior="functional",
          use_bare=True):
    n_eff = z_o.shape[0]
    if prior == "functional":
        zo_t = numpyro.sample("z_o_true", dist.Normal(z_o, se_o))
        zr_t = numpyro.sample("z_r_true", dist.Normal(z_r, se_r))
    else:
        mu_r = numpyro.sample("mu_r", dist.Normal(0, 1))
        tau_r = numpyro.sample("tau_r", dist.HalfNormal(1))
        mu_D = numpyro.sample("mu_D", dist.Normal(0, 1))
        tau_D = numpyro.sample("tau_D", dist.HalfNormal(1))
        zr_t = numpyro.sample("z_r_true", dist.Normal(mu_r, tau_r).expand([n_eff]))
        D_t = numpyro.sample("D_true", dist.Normal(mu_D, tau_D).expand([n_eff]))
        zo_t = numpyro.deterministic("z_o_true", zr_t + D_t)
        numpyro.sample("z_r_obs", dist.Normal(zr_t, se_r), obs=z_r)
        numpyro.sample("z_o_obs", dist.Normal(zo_t, se_o), obs=z_o)
    D = zo_t - zr_t
    b0 = numpyro.sample("b0", dist.Normal(0, 1))
    b_rep = numpyro.sample("b_rep", dist.Normal(0, 1))
    b_pub = numpyro.sample("b_pub", dist.Normal(0, 1))
    b_model = numpyro.sample("b_model", dist.Normal(0, 1))
    sd_eff = numpyro.sample("sd_eff", dist.HalfNormal(0.5))
    u = numpyro.sample("u_eff", dist.Normal(0, sd_eff).expand([n_eff]))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    mu = b0 + b_rep * zr_t[eff_idx] + b_pub * D[eff_idx] + b_model * mod_idx + u[eff_idx]
    if use_bare:
        b_bare = numpyro.sample("b_bare", dist.Normal(0, 1))
        mu = mu + b_bare * z_bare
    if X.shape[1] > 0:
        gamma = numpyro.sample("gamma", dist.Normal(0, 1).expand([X.shape[1]]))
        mu = mu + X @ gamma
    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)


def prep(df, X=None):
    """df: unit rows with effect, model, z_sim, z_bare, z_o, z_r, var_o, var_r."""
    import pandas as pd
    df = df.reset_index(drop=True)
    eff_idx, eff_names = pd.factorize(df.effect)
    mods = sorted(df.model.unique())
    mod_idx = (df.model == mods[-1]).astype(float).values if len(mods) > 1 else np.zeros(len(df))
    em = df.drop_duplicates("effect").set_index("effect").loc[eff_names]
    if X is None:
        X = np.zeros((len(df), 0))
    zb = df.z_bare.values if "z_bare" in df else np.zeros(len(df))
    zb = np.nan_to_num(zb.astype(float))
    return dict(eff_idx=jnp.array(eff_idx), mod_idx=jnp.array(mod_idx), z_bare=jnp.array(zb),
                z_o=jnp.array(em.z_o.values), se_o=jnp.array(np.sqrt(em.var_o.values)),
                z_r=jnp.array(em.z_r.values), se_r=jnp.array(np.sqrt(em.var_r.values)),
                X=jnp.array(np.asarray(X, float))), df.z_sim.values


def fit(df, X=None, prior="functional", use_bare=True, seed=11, warmup=1000, samples=1500,
        chains=4, target_accept=0.95, y=None, progress=False):
    data, y_obs = prep(df, X)
    y_use = jnp.array(y_obs if y is None else y)
    kern = NUTS(model, target_accept_prob=target_accept)
    mc = MCMC(kern, num_warmup=warmup, num_samples=samples, num_chains=chains,
              chain_method="parallel" if chains > 1 else "sequential", progress_bar=progress)
    mc.run(jax.random.PRNGKey(seed), **data, y=y_use, prior=prior, use_bare=use_bare)
    return mc


def summarize(mc, names=("b_bare", "b_rep", "b_pub", "sd_eff", "sigma"), xnames=None):
    s = mc.get_samples()
    out = {}
    for k in names:
        if k in s:
            v = np.asarray(s[k])
            out[k] = {"mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)),
                      "hi": float(np.percentile(v, 97.5)), "P_gt0": float((v > 0).mean())}
    if xnames is not None and "gamma" in s:
        g = np.asarray(s["gamma"])
        for j, nm in enumerate(xnames):
            v = g[:, j]
            out[nm] = {"mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)),
                       "hi": float(np.percentile(v, 97.5)), "P_gt0": float((v > 0).mean())}
    try:
        div = int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
    except Exception:
        div = -1
    out["_divergences"] = div
    return out
