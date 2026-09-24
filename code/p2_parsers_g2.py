"""
Phase-2 group-2 parsers + effect-size estimators.

Scope: the 14 effects at positions 28-41 (0-based, alphabetical) of the 42
"new" phase-2 effects (= P2_master_protocol_bank.json protocols minus the 19
legacy prefixes).

    Kupor2015_god_risk
    McCullough2002_gratitude_agreeableness
    Mellers2001_conjunction_frequency
    Park2019_scarcity_price_quality
    Paunonen2003_attractiveness_extraversion
    Peters2006_numeracy_framing
    Reinhard2013_unconscious_lie_detection
    Scopelliti2015_bragging_emotion
    Shnabel2008_needs_reconciliation
    Stevenson2000_web_brand_attitude
    Thoresen2003_commitment_negative_affect
    Weinstein1984_comparative_optimism
    WilliamsBargh2008_distance_priming
    Zarkadi2013_visual_contrast

Design
------
Each effect exposes  parse(text, variant, cond) -> dict | None.
The dict holds the participant-level scored quantities named by the
protocol's `measure` field; None means unparseable.  Parsing runs in two
passes: STRICT (exact response_format compliance) and, for cells whose
strict parse rate falls below 0.60, LENIENT (label-order-free number
scavenging with a minimum-value-count floor).  Lenient mode never invents a
value: it only relaxes *where* a number may appear and *how many* of the
requested numbers must be present.

Refusals (text is None / empty) are counted separately and excluded from the
parse-rate denominator.

Sign conventions follow each protocol's own `effect_statistic` text, which is
written so that the reported r is directly comparable to the FReD anchor
(r_o).  For two effects the protocol's claim-consistent direction is a
NEGATIVE r and the anchor is likewise negative:
  * Thoresen2003  (r_o = -0.19)  commitment x negative emotionality
  * Weinstein1984 (r_o = -0.3511) mean comparative-risk rating below midpoint

Documented scoring deviations
-----------------------------
Peters2006, variant B: the protocol's `measure` block repeats variant A's
objective-numeracy key (500, 0.1, 10, 20), but variant B's own item text asks
four different questions whose correct answers are 50, 0.001, 100 and 10.
Scoring B against A's key would mark every response wrong and collapse the
objective-numeracy component to zero variance.  The key is therefore taken
from each variant's own item wording (A: 500/0.1/10/20, B: 50/0.001/100/10),
which is what "number of correct answers" means.  Flagged as
PETERS_B_KEY_DEVIATION below.

Mellers2001: the protocol names 'and_are' as the FReD target arm; the unit
level r_sim is computed on that arm only ('who_are' is the paper's
contrasting phrasing and is parsed but not used for the headline r).
"""

import math
import re

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------
# generic tolerant field extraction
# --------------------------------------------------------------------------

NUM = r"[-+]?\d+(?:\.\d+)?"


def _clean(text):
    if text is None:
        return None
    t = str(text).replace("\u2014", "-").replace("\u2013", "-")
    t = re.sub(r"[*`#]", "", t)  # markdown emphasis; '_' kept (PRICE_HIGH etc.)
    return t.strip()


def label_values(text, label):
    """All numbers attached to `label`, in order of appearance."""
    pat = re.compile(rf"\b{re.escape(label)}\b\s*[:=\-]?\s*({NUM})", re.I)
    return [float(m.group(1)) for m in pat.finditer(text)]


def label_one(text, label):
    v = label_values(text, label)
    return v[0] if v else None


def numbered_values(text, n):
    """Lines shaped 'k: value' / 'k. value' for k = 1..n."""
    out = {}
    for m in re.finditer(rf"(?m)^\s*(\d{{1,2}})\s*[:.\)]\s*({NUM})\s*$", text):
        k = int(m.group(1))
        if 1 <= k <= n and k not in out:
            out[k] = float(m.group(2))
    return out


def all_numbers(text):
    return [float(x) for x in re.findall(NUM, text)]


def in_range(vals, lo, hi):
    return [v for v in vals if lo <= v <= hi]


def _mean(xs):
    return float(np.mean(xs))


def rev(v, const=8):
    return const - v


# --------------------------------------------------------------------------
# per-effect parsers
# --------------------------------------------------------------------------
# Each parser signature: (text, variant, cond, lenient) -> dict | None


def p_kupor(text, variant, cond, lenient=False):
    """6 'SENT:' lines then 8 'RATING: 1-7'.  Score = mean of the ratings."""
    sents = [
        s.strip()
        for s in re.findall(r"(?mi)^\s*SENT\s*[:=\-]?\s*(.+?)\s*$", text)
        if len(s.strip().split()) >= 2
    ]
    ratings = in_range([v for v in label_values(text, "RATING")], 1, 7)
    ratings = [r for r in ratings if float(r).is_integer()]
    need_sent, need_rating = (5, 8) if not lenient else (4, 6)
    if len(sents) < need_sent or len(ratings) < need_rating:
        return None
    ratings = ratings[:8]
    return {"risk": _mean(ratings), "n_sent": len(sents)}


_MCC_KEYS = {
    # variant -> (gratitude direct, gratitude reversed, agree direct, agree reversed)
    "A": ((1, 2, 4, 5), (3, 6), (7, 8, 10, 11), (9, 12)),
    "B": ((1, 2, 5), (3, 4, 6), (7, 8, 10, 12), (9, 11)),
}


def p_mccullough(text, variant, cond, lenient=False):
    vals = _items(text, 12, 1, 7, lenient)
    if vals is None:
        return None
    gd, gr, ad, ar = _MCC_KEYS[variant]
    if any(vals.get(i) is None for i in gd + gr + ad + ar):
        return None
    grat = _mean([vals[i] for i in gd] + [rev(vals[i]) for i in gr])
    agree = _mean([vals[i] for i in ad] + [rev(vals[i]) for i in ar])
    return {"gratitude": grat, "agreeableness": agree}


def _items(text, n, lo, hi, lenient, prefix="ITEM"):
    """ITEM1..ITEMn (or NUM/OC/NA style prefixes) -> {1..n: value}."""
    out = {}
    for i in range(1, n + 1):
        v = label_one(text, f"{prefix}{i}")
        if v is not None and lo <= v <= hi:
            out[i] = v
    if len(out) == n:
        return out
    if not lenient:
        return None
    cand = in_range(all_numbers(text), lo, hi)
    # drop leading item indices if the model echoed them
    if len(cand) >= n:
        return {i + 1: cand[i] for i in range(n)}
    return None


def p_mellers(text, variant, cond, lenient=False):
    """6 numbered frequency estimates 0-1000; violation = line4 - line2."""
    vals = numbered_values(text, 6)
    if not (2 in vals and 4 in vals):
        if not lenient:
            return None
        cand = in_range(all_numbers(text), 0, 1000)
        if len(cand) < 6:
            return None
        vals = {i + 1: cand[i] for i in range(6)}
    if not (0 <= vals[2] <= 1000 and 0 <= vals[4] <= 1000):
        return None
    return {
        "conj": vals[4],
        "constituent": vals[2],
        "violation": vals[4] - vals[2],
    }


def p_park(text, variant, cond, lenient=False):
    hi = label_one(text, "PRICE_HIGH")
    lo = label_one(text, "PRICE_LOW")
    if hi is None or lo is None or not (1 <= hi <= 9 and 1 <= lo <= 9):
        if not lenient:
            return None
        cand = in_range(all_numbers(text), 1, 9)
        if len(cand) < 2:
            return None
        hi, lo = cand[0], cand[1]
    return {"pq": hi - lo, "price_high": hi, "price_low": lo}


_PAU_KEYS = {
    "A": ((1, 2, 4, 5), (3, 6), (7, 8, 9, 10), ()),
    "B": ((1, 3, 6), (2, 4, 8), (5, 7, 9, 10), ()),
}


def p_paunonen(text, variant, cond, lenient=False):
    vals = _items(text, 10, 1, 7, lenient)
    if vals is None:
        return None
    ed, er, ad, ar = _PAU_KEYS[variant]
    extra = _mean([vals[i] for i in ed] + [rev(vals[i]) for i in er])
    attract = _mean([vals[i] for i in ad] + [rev(vals[i]) for i in ar])
    return {"extraversion": extra, "attractiveness": attract}


# PETERS_B_KEY_DEVIATION: key read off each variant's own item wording.
_PETERS_KEY = {
    "A": (500.0, 0.1, 10.0, 20.0),
    "B": (50.0, 0.001, 100.0, 10.0),
}


def _num_correct(given, key):
    if given is None:
        return False
    for target in (key, key * 100.0, key / 100.0):
        if abs(given - target) <= max(1e-9, abs(target) * 1e-6):
            return True
    return False


def p_peters(text, variant, cond, lenient=False):
    got = {}
    for i in range(1, 9):
        got[i] = label_one(text, f"NUM{i}")
    rating = label_one(text, "RATING")
    if rating is None or any(got[i] is None for i in range(1, 9)):
        if not lenient:
            return None
        cand = all_numbers(text)
        if len(cand) < 9:
            return None
        for i in range(1, 9):
            got[i] = cand[i - 1]
        rating = cand[8]
    subj = [got[i] for i in range(5, 9)]
    if not all(1 <= s <= 6 for s in subj) or not (1 <= rating <= 9):
        return None
    key = _PETERS_KEY[variant]
    obj = sum(1 for i in range(1, 5) if _num_correct(got[i], key[i - 1]))
    return {"obj": float(obj), "subj": _mean(subj), "rating": float(rating)}


_REINHARD_KEY = {1: "T", 2: "L", 3: "T", 4: "L", 5: "T", 6: "L"}


def p_reinhard(text, variant, cond, lenient=False):
    out = {}
    for m in re.finditer(r"(?m)^\s*([1-6])\s*[:.\)]\s*([TL])\b", text, re.I):
        out[int(m.group(1))] = m.group(2).upper()
    if len(out) != 6:
        if not lenient:
            return None
        toks = re.findall(r"\b([TL])\b", text.upper())
        if len(toks) != 6:
            return None
        out = {i + 1: toks[i] for i in range(6)}
    acc = sum(1 for k in range(1, 7) if out[k] == _REINHARD_KEY[k])
    truth_bias = sum(1 for k in range(1, 7) if out[k] == "T")
    return {"accuracy": float(acc), "truth_bias": float(truth_bias)}


def p_scopelliti(text, variant, cond, lenient=False):
    proud = label_one(text, "PROUD")
    happy = label_one(text, "HAPPY")
    annoyed = label_one(text, "ANNOYED")
    vals = [proud, happy, annoyed]
    if any(v is None or not (1 <= v <= 7) for v in vals):
        if not lenient:
            return None
        cand = in_range(all_numbers(text), 1, 7)
        if len(cand) < 3:
            return None
        proud, happy, annoyed = cand[0], cand[1], cand[2]
    return {"pos": _mean([proud, happy]), "annoyed": float(annoyed)}


def _numbered_scale(text, n, lo, hi, lenient):
    vals = numbered_values(text, n)
    if len(vals) == n and all(lo <= vals[i] <= hi for i in vals):
        return [vals[i] for i in range(1, n + 1)]
    if not lenient:
        return None
    cand = in_range(all_numbers(text), lo, hi)
    if len(cand) < n:
        return None
    return cand[:n]


def p_shnabel(text, variant, cond, lenient=False):
    vals = _numbered_scale(text, 5, 1, 7, lenient)
    if vals is None:
        return None
    return {"reconcile": _mean(vals)}


def p_stevenson(text, variant, cond, lenient=False):
    aw = [label_one(text, f"AW{i}") for i in (1, 2, 3)]
    ab = [label_one(text, f"AB{i}") for i in (1, 2, 3)]
    if any(v is None or not (1 <= v <= 7) for v in aw + ab):
        if not lenient:
            return None
        cand = in_range(all_numbers(text), 1, 7)
        if len(cand) < 6:
            return None
        aw, ab = cand[:3], cand[3:6]
    return {"aw": _mean(aw), "ab": _mean(ab)}


def p_thoresen(text, variant, cond, lenient=False):
    na = [label_one(text, f"NA{i}") for i in range(1, 11)]
    oc = [label_one(text, f"OC{i}") for i in range(1, 7)]
    ok = all(v is not None for v in na + oc)
    ok = ok and all(1 <= v <= 5 for v in na) and all(1 <= v <= 7 for v in oc)
    if not ok:
        if not lenient:
            return None
        na2 = [v for v in (label_one(text, f"NA{i}") for i in range(1, 11)) if v is not None]
        oc2 = [v for v in (label_one(text, f"OC{i}") for i in range(1, 7)) if v is not None]
        if len(na2) < 8 or len(oc2) != 6:
            return None
        na, oc = na2, oc2
        if not (all(1 <= v <= 5 for v in na) and all(1 <= v <= 7 for v in oc)):
            return None
    commit = _mean(list(oc[:4]) + [rev(oc[4]), rev(oc[5])])
    return {"neg_affect": _mean(na), "commitment": commit}


def p_weinstein(text, variant, cond, lenient=False):
    vals = in_range([v for v in label_values(text, "RATING")], 1, 7)
    if len(vals) < 10:
        if not lenient:
            return None
        vals = in_range(all_numbers(text), 1, 7)
        if len(vals) < 8:
            return None
    return {"risk": _mean(vals[:10])}


def p_williams(text, variant, cond, lenient=False):
    sib = label_one(text, "SIBLINGS")
    par = label_one(text, "PARENTS")
    hom = label_one(text, "HOMETOWN")
    vals = [sib, par, hom]
    if any(v is None or not (1 <= v <= 7) for v in vals):
        if not lenient:
            return None
        cand = in_range(all_numbers(text), 1, 7)
        if len(cand) < 3:
            return None
        vals = cand[-3:]
    return {"attachment": _mean(vals)}


def p_zarkadi(text, variant, cond, lenient=False):
    v = label_one(text, "RATING")
    if v is None or not (1 <= v <= 7):
        if not lenient:
            return None
        cand = in_range(all_numbers(text), 1, 7)
        if not cand:
            return None
        v = cand[0]
    return {"dev": abs(v - 4.0), "raw": float(v)}


PARSERS = {
    "Kupor2015_god_risk": p_kupor,
    "McCullough2002_gratitude_agreeableness": p_mccullough,
    "Mellers2001_conjunction_frequency": p_mellers,
    "Park2019_scarcity_price_quality": p_park,
    "Paunonen2003_attractiveness_extraversion": p_paunonen,
    "Peters2006_numeracy_framing": p_peters,
    "Reinhard2013_unconscious_lie_detection": p_reinhard,
    "Scopelliti2015_bragging_emotion": p_scopelliti,
    "Shnabel2008_needs_reconciliation": p_shnabel,
    "Stevenson2000_web_brand_attitude": p_stevenson,
    "Thoresen2003_commitment_negative_affect": p_thoresen,
    "Weinstein1984_comparative_optimism": p_weinstein,
    "WilliamsBargh2008_distance_priming": p_williams,
    "Zarkadi2013_visual_contrast": p_zarkadi,
}


def looks_like_prose_refusal(text):
    """Meta-commentary / refusal that nonetheless has text.  These are real
    format failures (they count in the parse-rate denominator), but the
    lenient pass must not scavenge digits out of their prose."""
    t = _clean(text) or ""
    if not t:
        return False
    if re.search(r"(?mi)^\s*(?:ITEM|NUM|NA|OC|AW|AB|SENT|RATING|PROUD|HAPPY|"
                 r"ANNOYED|PRICE_HIGH|PRICE_LOW|SIBLINGS|PARENTS|HOMETOWN|P[123])"
                 r"\s*\d*\s*[:=]", t):
        return False
    if re.search(r"(?m)^\s*\d{1,2}\s*[:.\)]\s*\S", t):
        return False
    words = re.findall(r"[A-Za-z']{2,}", t)
    return len(words) >= 12


def parse_row(effect, text, variant, cond, lenient=False):
    t = _clean(text)
    if not t:
        return None  # refusal / empty -> handled by the caller as non-denominator
    if lenient and looks_like_prose_refusal(t):
        return None
    try:
        return PARSERS[effect](t, variant, cond, lenient=lenient)
    except Exception:
        return None


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


# DEGENERACY CONVENTIONS (applied uniformly; every affected cell also carries
# a `degenerate` flag in the output table so downstream analysis can exclude
# it rather than having to re-derive it):
#   (a) all scored values identical within AND across groups -> the observed
#       group difference / omnibus effect is exactly zero, so r = 0.0.
#   (b) one variable of a correlational pair is constant -> the sample
#       covariance is exactly zero, so r = 0.0 (r = 0/0 formally; 0.0 is the
#       substantive reading "no covariation was produced").
#   (c) zero within-group variance but DIFFERENT group means -> t is infinite
#       and r is not estimable; reported as NaN, never as +-1.
#   (d) a rank-deficient regression design (e.g. a moderator that is constant
#       or perfectly collinear with the manipulation) -> NaN, not 0.


# Composite means are computed in floating point, so an "identical for every
# participant" composite can still carry ~1e-15 of residue.  Constancy is
# therefore tested with a relative tolerance rather than == 0.
_CONST_TOL = 1e-9


def is_const(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return True
    scale = max(1.0, float(np.max(np.abs(v))))
    return bool(np.std(v) <= _CONST_TOL * scale)


def r_two_group(a, b):
    """Point-biserial r from an independent-samples t test; positive when
    mean(a) > mean(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    if is_const(a) and is_const(b):
        return 0.0 if np.isclose(np.mean(a), np.mean(b)) else np.nan  # (a)/(c)
    t, _ = stats.ttest_ind(a, b, equal_var=True)
    if not np.isfinite(t):
        return 0.0 if np.isclose(np.mean(a), np.mean(b)) else np.nan
    df = len(a) + len(b) - 2
    return float(t / math.sqrt(t * t + df))


def r_paired(x, y):
    """r = sqrt(t^2/(t^2+df)) from a paired t test, signed by mean(x-y)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    if len(d) < 2:
        return np.nan
    if is_const(d):
        # (a)/(c): a constant zero difference is exactly no effect; a
        # constant NON-zero difference gives infinite t -> not estimable.
        return 0.0 if np.isclose(d[0], 0.0) else np.nan
    t, _ = stats.ttest_rel(np.asarray(x, float), np.asarray(y, float))
    if not np.isfinite(t):
        return np.nan
    df = len(d) - 1
    return float(math.copysign(math.sqrt(t * t / (t * t + df)), np.mean(d)))


def r_one_sample(x, mu):
    """r = t/sqrt(t^2+df) against `mu`; negative when mean < mu."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan
    sd = np.std(x, ddof=1)
    if is_const(x):
        # (a)/(c): identical to mu -> exactly zero effect; otherwise t is
        # infinite and r is not estimable.
        return 0.0 if np.isclose(np.mean(x), mu) else np.nan
    t = (np.mean(x) - mu) / (sd / math.sqrt(len(x)))
    df = len(x) - 1
    return float(t / math.sqrt(t * t + df))


def r_pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return np.nan
    if is_const(x) or is_const(y):
        return 0.0  # (b) constant variable -> covariance exactly zero
    return float(np.corrcoef(x, y)[0, 1])


def r_interaction(rating, frame, numeracy):
    """OLS rating ~ frame + numeracy_c + frame*numeracy_c.
    r = -t_int/sqrt(t_int^2+df_resid): positive when the gain-minus-loss gap
    is LARGER at LOW numeracy (b_int < 0), i.e. the FReD claim."""
    y = np.asarray(rating, float)
    f = np.asarray(frame, float)
    z = np.asarray(numeracy, float)
    n = len(y)
    if n < 8:
        return np.nan
    zc = z - z.mean()
    X = np.column_stack([np.ones(n), f, zc, f * zc])
    if is_const(zc) or np.linalg.matrix_rank(X) < 4:
        return np.nan  # (d)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = n - 4
    if df <= 0:
        return np.nan
    s2 = resid @ resid / df
    if s2 <= 0:
        return np.nan
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = math.sqrt(cov[3, 3])
    if se == 0:
        return np.nan
    t = beta[3] / se
    return float(-t / math.sqrt(t * t + df))


def r_anova_oneway(groups, top_key):
    """r = sqrt(eta^2) of the omnibus effect, signed positive when the group
    named by `top_key` has the largest mean."""
    keys = [k for k in groups if len(groups[k]) >= 2]
    if len(keys) < 2 or top_key not in keys:
        return np.nan
    allv = np.concatenate([np.asarray(groups[k], float) for k in keys])
    gm = allv.mean()
    ssb = sum(len(groups[k]) * (np.mean(groups[k]) - gm) ** 2 for k in keys)
    ssw = sum(((np.asarray(groups[k], float) - np.mean(groups[k])) ** 2).sum() for k in keys)
    if np.isclose(ssw, 0.0, atol=1e-12):
        # (a)/(c): no within-group variance at all
        return 0.0 if np.isclose(ssb, 0.0) else np.nan
    if ssb + ssw <= 0:
        return np.nan
    eta2 = ssb / (ssb + ssw)
    means = {k: float(np.mean(groups[k])) for k in keys}
    sign = 1.0 if means[top_key] >= max(means.values()) else -1.0
    return float(sign * math.sqrt(eta2))


def r_interaction_2x2(cells, plus, minus):
    """r = sqrt(F_int/(F_int+df_error)) for a 2x2 between-subjects design.
    `cells` maps the four condition names to value lists; the crossover
    contrast is sum(mean of `plus` cells) - sum(mean of `minus` cells) and
    supplies the sign."""
    keys = list(cells)
    if len(keys) != 4 or any(len(cells[k]) < 2 for k in keys):
        return np.nan
    means = {k: float(np.mean(cells[k])) for k in keys}
    ns = {k: len(cells[k]) for k in keys}
    n_h = 4.0 / sum(1.0 / ns[k] for k in keys)  # harmonic mean cell n
    contrast = sum(means[k] for k in plus) - sum(means[k] for k in minus)
    # SS for the interaction contrast with coefficients +1/+1/-1/-1
    ss_int = (contrast ** 2) * n_h / 4.0
    sse = sum(((np.asarray(cells[k], float) - means[k]) ** 2).sum() for k in keys)
    df_e = sum(ns[k] for k in keys) - 4
    if df_e <= 0:
        return np.nan
    if np.isclose(sse, 0.0, atol=1e-12):
        # (a)/(c)
        return 0.0 if np.isclose(contrast, 0.0) else np.nan
    f = ss_int / (sse / df_e)
    return float(math.copysign(math.sqrt(f / (f + df_e)), contrast))


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------


def boot_ci(fn, blocks, n_boot=2000, seed=0, alpha=0.05):
    """Percentile CI. `blocks` is a list of index-resamplable units; `fn`
    takes a resampled block list and returns r (or nan)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        rs = [[b[i] for i in rng.integers(0, len(b), len(b))] if len(b) else b for b in blocks]
        v = fn(rs)
        if v is not None and np.isfinite(v):
            out.append(v)
    if len(out) < max(50, 0.25 * n_boot):
        return (np.nan, np.nan)
    return (
        float(np.quantile(out, alpha / 2)),
        float(np.quantile(out, 1 - alpha / 2)),
    )
