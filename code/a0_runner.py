"""A0 simulation runner: build per-subject requests from a protocol bank,
parse responses, compute per-cell effect sizes (r scale) with bootstrap CIs.

Model policy (user-specified):
- claude-sonnet-5      : no temperature parameter passed
- claude-sonnet-4-5-20250929 : temperature=1.0
Request construction and scoring utilities for the A0 wave. Requests are
built as JSON lists and dispatched to the model API by a separate runner;
responses are read back from JSON files with the same index fields.
"""
import json, re, random
import numpy as np

MODELS = {
    "sonnet5": {"model": "claude-sonnet-5"},                       # no temperature
    "sonnet45": {"model": "claude-sonnet-4-5-20250929", "temperature": 1.0},
}

FIRST_NAMES = ["Alex","Jordan","Sam","Taylor","Morgan","Casey","Riley","Jamie","Quinn","Avery",
               "Dana","Robin","Blake","Drew","Reese","Skyler","Cameron","Emerson","Finley","Hayden"]

def make_persona(rng, tier):
    """tier: 'none' -> empty persona; 'demographic' -> sampled WEIRD student persona."""
    if tier == "none":
        return ""
    age = rng.randint(18, 24)
    gender = rng.choice(["woman", "man", "woman", "man", "non-binary person"])
    major = rng.choice(["psychology","biology","economics","communications","engineering",
                        "English literature","political science","computer science","nursing","history"])
    year = rng.choice(["first-year","second-year","third-year","fourth-year"])
    return (f"You are a {age}-year-old {gender}, a {year} undergraduate studying {major} "
            f"at a large public university in the United States, participating for course credit. "
            f"Answer as this person would, naturally and honestly.")

def build_requests(bank, n_per_cell, seed=7, models=("sonnet5","sonnet45"),
                   personas=("none","demographic"), max_tokens=200):
    """Yield request dicts + index rows. Between-subjects conditions are assigned
    round-robin within each cell so groups are balanced."""
    rng = random.Random(seed)
    requests, index = [], []
    for proto in bank:
        conds = list(proto.get("conditions") or {"single": None})
        for mk in models:
            for pt in personas:
                for i in range(n_per_cell):
                    cond = conds[i % len(conds)]
                    persona = make_persona(rng, pt)
                    prompt = proto["subject_prompt_template"].replace("{persona}", persona)
                    if proto.get("conditions"):
                        prompt = prompt.replace("{condition}", proto["conditions"][cond])
                    req = {"prompt": prompt, "max_tokens": max_tokens, **MODELS[mk]}
                    requests.append(req)
                    index.append({"effect": proto["effect"], "variant": proto["variant"],
                                  "model": mk, "persona": pt, "cond": cond, "subj": i})
    return requests, index

NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')

def parse_numeric(text, lo=None, hi=None):
    """First number in response, range-checked."""
    if text is None: return None
    m = NUM_RE.search(str(text))
    if not m: return None
    v = float(m.group())
    if lo is not None and v < lo: return None
    if hi is not None and v > hi: return None
    return v

def parse_choice(text, options):
    """Match one of the options (case-insensitive, word-boundary)."""
    if text is None: return None
    t = str(text).lower()
    hits = [o for o in options if re.search(r'\b'+re.escape(o.lower())+r'\b', t)]
    return hits[0] if len(hits) == 1 else (hits[0] if hits else None)

def d_to_r(d, n1, n2):
    a = ((n1+n2)**2) / (n1*n2) if n1 and n2 else 4.0
    return d / np.sqrt(d*d + a)

def two_group_r(vals_a, vals_b):
    """Cohen's d -> r for two independent groups (matching t-test style analyses)."""
    a, b = np.asarray(vals_a, float), np.asarray(vals_b, float)
    if len(a) < 3 or len(b) < 3: return np.nan
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    if sp == 0: return 0.0
    d = (a.mean() - b.mean()) / sp
    return d_to_r(d, len(a), len(b))

def proportion_diff_r(k1, n1, k2, n2):
    """Two-proportion effect as phi over the 2x2 table."""
    if min(n1, n2) < 3: return np.nan
    p1, p2 = k1/n1, k2/n2
    p = (k1+k2)/(n1+n2)
    if p in (0,1): return 0.0
    # phi from chi-square of 2x2
    import math
    chi2 = (n1*n2/(n1+n2)) * (p1-p2)**2 / (p*(1-p))
    return math.copysign(math.sqrt(chi2/(n1+n2)), p1-p2)

def paired_r(vals_a, vals_b):
    """Within-subject two-condition: d_z -> r."""
    a, b = np.asarray(vals_a, float), np.asarray(vals_b, float)
    n = min(len(a), len(b))
    if n < 3: return np.nan
    diff = a[:n]-b[:n]
    if diff.std(ddof=1) == 0: return 0.0
    dz = diff.mean()/diff.std(ddof=1)
    return dz/np.sqrt(dz*dz+1)

def pearson_r(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4 or x.std() == 0 or y.std() == 0: return np.nan
    return float(np.corrcoef(x, y)[0,1])

def bootstrap_ci(stat_fn, data_tuple, n_boot=2000, seed=11):
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        resampled = tuple(np.asarray(d)[rng.integers(0, len(d), len(d))] if len(d) else d
                          for d in data_tuple)
        s = stat_fn(*resampled)
        if not np.isnan(s): stats.append(s)
    if not stats: return (np.nan, np.nan)
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))
