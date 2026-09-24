"""
Phase-2 group-1 parsers + effect-size estimators.

Scope: the 14 effects at positions 14-27 (0-based) of the 42 new Phase-2 effects,
alphabetically ordered:
    Finucane2000_affect_heuristic ... Kozak2006_action_identification

Design principles
-----------------
* Parsers are written per effect from the protocol bank's `response_format` /
  `measure` fields, and are tolerant of A/B label differences: every parser
  accepts both variants' label spellings, and one effect (Galinsky2006) has a
  genuinely variant-specific DV polarity, handled explicitly.
* Two tiers. `parse(..., lenient=False)` requires the declared format; the
  lenient tier additionally tolerates markdown emphasis, code fences, alternate
  separators (`.`/`)`/`=`/whitespace), unlabeled numbered lines, extra prose,
  and (for composites) missing NON-target items. It never invents values: a row
  is unparsed unless every item the DV needs is present and in range.
* Sign convention: each effect's sign follows its own protocol
  `effect_statistic` verbatim, so r_sim is directly comparable with that
  effect's FReD anchor. For two effects the ORIGINAL CLAIM direction is a
  NEGATIVE r under that convention and this is intentional:
    - Finucane2000: claim = inverse risk-benefit relation (r < 0).
    - Huang2014:    claim = cleanliness prime -> less harsh judgement, and
                    FReD records r_o = -0.1435 with cleanliness coded 1.
  All other effects: positive r = claim direction.
"""

import re
import numpy as np
from scipy import stats

# ----------------------------------------------------------------------------
# text normalisation
# ----------------------------------------------------------------------------

_FENCE = re.compile(r"^\s*```[^\n]*\n|\n?```\s*$")


def _norm(text, lenient):
    """Normalise a raw response body."""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = _FENCE.sub("", t)
    if lenient:
        t = t.replace("**", "").replace("*", "").replace("__", "")
        # bullet prefixes
        t = re.sub(r"(?m)^\s*[-\u2022]\s+", "", t)
    return t.strip()


def _sep(lenient):
    return r"\s*[:.)=]\s*" if not lenient else r"\s*[:.)=\-]?\s+|\s*[:.)=]\s*"


def _find_labeled(t, label, lenient, n_expected=None):
    """All numeric values for a repeated label such as POWER1 / 'RISK 3' / RATING.

    `label` is a regex fragment for the label WITHOUT its index.
    Returns list of (index_or_None, value_str) in document order.
    """
    idx = r"\s*(\d+)?" if True else ""
    if lenient:
        pat = re.compile(
            rf"(?im)^[^\S\n]*{label}{idx}[^\S\n]*[:.)=]?[^\S\n]*(-?\d+(?:\.\d+)?)[^\S\n]*$"
        )
    else:
        pat = re.compile(rf"(?m)^\s*{label}{idx}\s*:\s*(-?\d+(?:\.\d+)?)\s*$")
    return [(m.group(1), m.group(2)) for m in pat.finditer(t)]


def _find_numbered(t, lenient):
    """Lines of the form 'N: value' (Gervais, Glikson, John2016)."""
    if lenient:
        # tolerates 'N: 3: 40' / 'Item 3 = 40' / '3) 40' as well as '3: 40'
        pat = re.compile(r"(?im)^[^\S\n]*(?:item|q|n)?[^\S\n]*[:.)=]?[^\S\n]*(\d{1,2})[^\S\n]*[:.)=][^\S\n]*(-?\d+(?:\.\d+)?)[^\S\n]*$")
    else:
        pat = re.compile(r"(?m)^\s*(\d{1,2})\s*:\s*(-?\d+(?:\.\d+)?)\s*$")
    return {int(m.group(1)): float(m.group(2)) for m in pat.finditer(t)}


def _bare_ints(t):
    """All standalone integers, document order (lenient positional fallback)."""
    return [float(x) for x in re.findall(r"(?m)(?<![\w.])(-?\d+)(?![\w.])", t)]


def _rng(v, lo, hi):
    return v if (v is not None and lo <= v <= hi) else None


def _mean(vals):
    return float(np.mean(vals)) if vals else None


# ----------------------------------------------------------------------------
# per-effect parsers.  Each returns dict of DV components, or None if the row
# cannot be scored without inventing data.
# ----------------------------------------------------------------------------

def p_finucane(t, variant, lenient):
    """24 lines: 'RISK N: 1-7' x12 then 'BEN N: 1-7' x12."""
    risk, ben = {}, {}
    for lab, dest in (("RISK", risk), ("BEN(?:EFIT)?", ben)):
        for i, v in _find_labeled(t, lab, lenient):
            if i is None:
                continue
            val = _rng(float(v), 1, 7)
            if val is not None:
                dest[int(i)] = val
    if lenient and (len(risk) < 12 or len(ben) < 12):
        ints = _bare_ints(t)
        # positional fallback only if the response is exactly 24 in-range values
        if len(ints) == 24 and all(1 <= x <= 7 for x in ints):
            risk = {i + 1: ints[i] for i in range(12)}
            ben = {i + 1: ints[i + 12] for i in range(12)}
    if len(risk) < 12 or len(ben) < 12:
        return None
    return {"risk": [risk[i] for i in range(1, 13)],
            "ben": [ben[i] for i in range(1, 13)]}


def p_galinsky(t, variant, lenient):
    """'INTERP: A|B' + 'CONF: 1-7'.  DV polarity differs between variants:
    variant A codes INTERP=A as the egocentric error, variant B codes INTERP=B."""
    if lenient:
        m = re.search(r"(?i)INTERP\w*\s*[:.)=]?\s*(?:option\s*)?([AB])\b", t)
    else:
        m = re.search(r"(?m)^\s*INTERP\s*:\s*([AB])\s*$", t)
    if not m:
        return None
    letter = m.group(1).upper()
    ego = 1 if ((variant == "A" and letter == "A") or (variant == "B" and letter == "B")) else 0
    conf = None
    c = _find_labeled(t, "CONF", lenient)
    if c:
        conf = _rng(float(c[0][1]), 1, 7)
    return {"ego": ego, "conf": conf, "interp": letter}


def p_gervais(t, variant, lenient):
    """3 lines 'N: 0-100'; DV = mean belief."""
    d = _find_numbered(t, lenient)
    vals = [_rng(d.get(i), 0, 100) for i in (1, 2, 3)]
    if any(v is None for v in vals) and lenient:
        ints = _bare_ints(t)
        cand = [x for x in ints if 0 <= x <= 100]
        if len(cand) == 3:
            vals = cand
    if any(v is None for v in vals):
        return None
    return {"belief": _mean(vals)}


def p_giessner(t, variant, lenient):
    """'POWER1..3', 1-9; DV = mean."""
    got = {}
    for i, v in _find_labeled(t, "POWER", lenient):
        if i is None:
            continue
        val = _rng(float(v), 1, 9)
        if val is not None:
            got[int(i)] = val
    if len(got) < 3 and lenient:
        ints = [x for x in _bare_ints(t) if 1 <= x <= 9]
        if len(ints) == 3:
            got = {1: ints[0], 2: ints[1], 3: ints[2]}
    if len(got) < 3:
        return None
    return {"power": _mean([got[i] for i in (1, 2, 3)])}


def p_glikson(t, variant, lenient):
    """11 lines 'N: 1-7'; competence = mean(6..11), warmth = mean(1..5)."""
    d = {k: _rng(v, 1, 7) for k, v in _find_numbered(t, lenient).items()}
    comp = [d.get(i) for i in range(6, 12)]
    if any(v is None for v in comp) and lenient:
        ints = [x for x in _bare_ints(t) if 1 <= x <= 7]
        if len(ints) == 11:
            d = {i + 1: ints[i] for i in range(11)}
            comp = [d[i] for i in range(6, 12)]
    if any(v is None for v in comp):
        return None
    warm = [d.get(i) for i in range(1, 6)]
    warm = [v for v in warm if v is not None]
    return {"competence": _mean(comp),
            "warmth": _mean(warm) if len(warm) == 5 else None}


def p_helzer(t, variant, lenient):
    """'SELF/SOCIAL/ECONOMIC: 1-7'; target = SELF."""
    def one(lab):
        r = _find_labeled(t, lab, lenient)
        return _rng(float(r[0][1]), 1, 7) if r else None
    self_ = one("SELF")
    if self_ is None and lenient:
        ints = [x for x in _bare_ints(t) if 1 <= x <= 7]
        if len(ints) == 3:
            self_ = ints[0]
    if self_ is None:
        return None
    return {"self": self_, "social": one("SOCIAL"), "economic": one("ECONOMIC")}


def p_hoorens(t, variant, lenient):
    """6 lines 'RATING: 1-7'; DV = mean agreement."""
    vals = [_rng(float(v), 1, 7) for _, v in _find_labeled(t, "RATING", lenient)]
    vals = [v for v in vals if v is not None]
    if len(vals) != 6 and lenient:
        ints = [x for x in _bare_ints(t) if 1 <= x <= 7]
        if len(ints) == 6:
            vals = ints
    if len(vals) != 6:
        return None
    return {"agree": _mean(vals)}


def p_huang2014(t, variant, lenient):
    """6 'SENT:' free-text lines + 6 'RATING: 0-9'; DV = mean rating.
    Strict tier applies the protocol's manipulation-execution check
    (>=5 well-formed sentences); lenient tier drops it but still requires all
    six ratings."""
    if lenient:
        spat = re.compile(r"(?im)^[^\S\n]*SENT\w*[^\S\n]*[:.)=]?[^\S\n]*(\S.*)$")
    else:
        spat = re.compile(r"(?m)^\s*SENT\s*:\s*(\S.*?)\s*$")
    sents = [s for s in spat.findall(t) if len(s.split()) >= 2]
    vals = [_rng(float(v), 0, 9) for _, v in _find_labeled(t, "RATING", lenient)]
    vals = [v for v in vals if v is not None]
    if len(vals) != 6:
        return None
    if not lenient and len(sents) < 5:
        return None
    return {"harsh": _mean(vals), "n_sent": len(sents)}


def p_huang2019(t, variant, lenient):
    """'CHOICE: A|B' + 3 'RATING: 1-7'; DV = 1 if CHOICE == A (holistic option)."""
    if lenient:
        m = re.search(r"(?i)CHOICE\w*\s*[:.)=]?\s*(?:option\s*)?([AB])\b", t)
    else:
        m = re.search(r"(?m)^\s*CHOICE\s*:\s*([AB])\s*$", t)
    if not m:
        return None
    rat = [_rng(float(v), 1, 7) for _, v in _find_labeled(t, "RATING", lenient)]
    rat = [v for v in rat if v is not None]
    return {"holistic": 1 if m.group(1).upper() == "A" else 0,
            "strength": rat[0] if rat else None}


def p_john2016(t, variant, lenient):
    """5 lines 'N: 1-7'; DV = mean impression."""
    d = {k: _rng(v, 1, 7) for k, v in _find_numbered(t, lenient).items()}
    vals = [d.get(i) for i in range(1, 6)]
    if any(v is None for v in vals) and lenient:
        ints = [x for x in _bare_ints(t) if 1 <= x <= 7]
        if len(ints) == 5:
            vals = ints
    if any(v is None for v in vals):
        return None
    return {"impression": _mean(vals)}


def p_jostmann(t, variant, lenient):
    """'VALUE: 1-1000' + 'IMP2..4: 1-7'; primary DV = log(VALUE)."""
    r = _find_labeled(t, "VALUE", lenient)
    val = _rng(float(r[0][1]), 1, 1000) if r else None
    if val is None:
        return None
    imp = {}
    for i, v in _find_labeled(t, "IMP", lenient):
        if i is None:
            continue
        x = _rng(float(v), 1, 7)
        if x is not None:
            imp[int(i)] = x
    return {"log_value": float(np.log(val)),
            "importance": _mean([imp[i] for i in (2, 3, 4)]) if len(imp) >= 3 else None}


def p_kay2014(t, variant, lenient):
    """'G1..G5: 1-7'; DV = mean willingness."""
    got = {}
    for i, v in _find_labeled(t, "G", lenient):
        if i is None:
            continue
        x = _rng(float(v), 1, 7)
        if x is not None:
            got[int(i)] = x
    if len(got) < 5 and lenient:
        ints = [x for x in _bare_ints(t) if 1 <= x <= 7]
        if len(ints) == 5:
            got = {i + 1: ints[i] for i in range(5)}
    if len(got) < 5:
        return None
    return {"willing": _mean([got[i] for i in range(1, 6)])}


def p_kimmarkus(t, variant, lenient):
    """'LIKING: 1-9' + 'OWN: 1-9'; target = LIKING."""
    def one(lab):
        r = _find_labeled(t, lab, lenient)
        return _rng(float(r[0][1]), 1, 9) if r else None
    lik, own = one("LIKING"), one("OWN")
    if lik is None and lenient:
        ints = [x for x in _bare_ints(t) if 1 <= x <= 9]
        if len(ints) == 2:
            lik, own = ints
    if lik is None:
        return None
    return {"liking": lik, "own": own}


def p_kozak(t, variant, lenient):
    """20 lines alternating 'ID N: A|B' and 'MIND N: 1-7'.
    ID level = count of B choices (0-10); mind = mean of the ten ratings."""
    if lenient:
        ipat = re.compile(r"(?im)^[^\S\n]*ID[^\S\n]*(\d+)?[^\S\n]*[:.)=]?[^\S\n]*([AB])\b")
    else:
        ipat = re.compile(r"(?m)^\s*ID\s*(\d+)\s*:\s*([AB])\s*$")
    ids = {}
    for m in ipat.finditer(t):
        if m.group(1) is None:
            continue
        ids[int(m.group(1))] = m.group(2).upper()
    mind = {}
    for i, v in _find_labeled(t, "MIND", lenient):
        if i is None:
            continue
        x = _rng(float(v), 1, 7)
        if x is not None:
            mind[int(i)] = x
    if len(ids) < 10 or len(mind) < 10:
        return None
    return {"id_level": float(sum(1 for i in range(1, 11) if ids[i] == "B")),
            "mind": _mean([mind[i] for i in range(1, 11)])}


PARSERS = {
    "Finucane2000_affect_heuristic": p_finucane,
    "Galinsky2006_power_perspective": p_galinsky,
    "Gervais2012_analytic_priming": p_gervais,
    "Giessner2007_vertical_power": p_giessner,
    "Glikson2017_smiley_competence": p_glikson,
    "Helzer2011_cleansing_conservatism": p_helzer,
    "Hoorens2015_more_less_asymmetry": p_hoorens,
    "Huang2014_cleanliness_effort": p_huang2014,
    "Huang2019_anthropomorphism_holistic": p_huang2019,
    "John2016_hiding_information": p_john2016,
    "Jostmann2009_weight_importance": p_jostmann,
    "Kay2014_structure_goal_pursuit": p_kay2014,
    "KimMarkus1999_uniqueness_preference": p_kimmarkus,
    "Kozak2006_action_identification": p_kozak,
}


def parse(effect, variant, text, lenient=False):
    """Parse one response body. Returns dict of DV components or None."""
    if text is None or not str(text).strip():
        return None
    return PARSERS[effect](_norm(str(text), lenient), variant, lenient)


# ----------------------------------------------------------------------------
# effect-size specifications
# ----------------------------------------------------------------------------
# kind:
#   'two_group'  point-biserial r = t/sqrt(t^2+df), sign from hi-minus-lo means
#   'phi'        signed phi on a 2x2 table (= corr of the two binary variables)
#   'corr_within' Fisher-z mean of within-subject item correlations
#   'corr_between' Pearson r across subjects between two subject-level scores
SPECS = {
    "Finucane2000_affect_heuristic": dict(
        kind="corr_within", x="risk", y="ben",
        claim_sign=-1,
        note="inverse risk-benefit relation; claim direction is NEGATIVE r"),
    "Galinsky2006_power_perspective": dict(
        kind="phi", dv="ego", hi="high_power", lo="low_power", claim_sign=1),
    "Gervais2012_analytic_priming": dict(
        kind="two_group", dv="belief", hi="control", lo="analytic_prime", claim_sign=1),
    "Giessner2007_vertical_power": dict(
        kind="two_group", dv="power", hi="long_line", lo="short_line", claim_sign=1),
    "Glikson2017_smiley_competence": dict(
        kind="two_group", dv="competence", hi="text_no_smiley", lo="text_with_smiley",
        claim_sign=1),
    "Helzer2011_cleansing_conservatism": dict(
        kind="two_group", dv="self", hi="cleansing_reminder", lo="control", claim_sign=1),
    "Hoorens2015_more_less_asymmetry": dict(
        kind="two_group", dv="agree", hi="more", lo="less", claim_sign=1),
    "Huang2014_cleanliness_effort": dict(
        kind="two_group", dv="harsh",
        hi="cleanliness_low_effort", lo="neutral_low_effort", claim_sign=-1,
        note="condition coded cleanliness=1; claim (less harsh) is NEGATIVE r, "
             "matching FReD r_o = -0.1435. High-effort cells excluded."),
    "Huang2019_anthropomorphism_holistic": dict(
        kind="phi", dv="holistic", hi="anthropomorphized", lo="nonanthropomorphized",
        claim_sign=1),
    "John2016_hiding_information": dict(
        kind="two_group", dv="impression", hi="reveal", lo="hide", claim_sign=1),
    "Jostmann2009_weight_importance": dict(
        kind="two_group", dv="log_value", hi="heavy", lo="light", claim_sign=1),
    "Kay2014_structure_goal_pursuit": dict(
        kind="two_group", dv="willing", hi="structure", lo="random", claim_sign=1),
    "KimMarkus1999_uniqueness_preference": dict(
        kind="two_group", dv="liking", hi="minority_target", lo="majority_target",
        claim_sign=1),
    "Kozak2006_action_identification": dict(
        kind="corr_between", x="id_level", y="mind", claim_sign=1),
}


# ----------------------------------------------------------------------------
# estimators
# ----------------------------------------------------------------------------

def _z(r):
    r = float(np.clip(r, -0.999999, 0.999999))
    return np.arctanh(r)


def r_two_group(hi_vals, lo_vals):
    """Point-biserial r from an equal-variance independent-samples t test.

    Degenerate cases (documented convention, not imputation):
      * both groups constant AND equal -> r = 0.0 exactly (zero mean difference,
        zero within-group variance: the simulated cell produced one response for
        everyone, so the condition explains nothing).
      * both groups constant but UNEQUAL -> r = +/-1.0 (perfect separation).
      * fewer than 2 usable observations in a group -> nan (not estimable).
    """
    if len(hi_vals) < 2 or len(lo_vals) < 2:
        return np.nan
    if np.std(hi_vals) == 0 and np.std(lo_vals) == 0:
        dm = float(np.mean(hi_vals) - np.mean(lo_vals))
        return 0.0 if dm == 0 else float(np.sign(dm))
    t, _ = stats.ttest_ind(hi_vals, lo_vals, equal_var=True)
    if not np.isfinite(t):
        return np.nan
    df = len(hi_vals) + len(lo_vals) - 2
    r = t / np.sqrt(t ** 2 + df)
    return float(r)


def r_phi(hi_bin, lo_bin):
    """Signed phi: positive when P(dv=1) is higher in the `hi` condition."""
    if len(hi_bin) < 2 or len(lo_bin) < 2:
        return np.nan
    a = float(sum(hi_bin)); b = len(hi_bin) - a
    c = float(sum(lo_bin)); dd = len(lo_bin) - c
    den = np.sqrt((a + b) * (c + dd) * (a + c) * (b + dd))
    if den == 0:
        # a zero margin means every observation shares the same DV value, so the
        # two conditions have identical proportions: phi = 0 exactly.
        return 0.0
    return float((a * dd - b * c) / den)


def r_corr_within(pairs):
    """Fisher-z mean of within-subject item correlations."""
    zs = []
    for x, y in pairs:
        x = np.asarray(x, float); y = np.asarray(y, float)
        if x.std() == 0 or y.std() == 0:
            continue
        zs.append(_z(np.corrcoef(x, y)[0, 1]))
    if not zs:
        return np.nan
    return float(np.tanh(np.mean(zs)))


def r_corr_between(xs, ys):
    """Pearson r across subjects.

    A constant score on either axis leaves the correlation genuinely undefined
    (no covariance information), so this returns nan rather than 0 -- unlike the
    two-group / phi degenerate cases, where a zero difference IS informative.
    """
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if len(xs) < 3 or xs.std() == 0 or ys.std() == 0:
        return np.nan
    return float(np.corrcoef(xs, ys)[0, 1])


def boot_ci(point_fn, groups, n_boot=2000, seed=0, alpha=0.05):
    """Stratified bootstrap percentile CI.

    `groups` is a list of index-able sequences; each is resampled with
    replacement independently (stratified by condition / group).
    """
    rng = np.random.default_rng(seed)
    reps = []
    ns = [len(g) for g in groups]
    if any(n == 0 for n in ns):
        return (np.nan, np.nan)
    for _ in range(n_boot):
        res = [[g[i] for i in rng.integers(0, n, n)] for g, n in zip(groups, ns)]
        v = point_fn(*res)
        if v is not None and np.isfinite(v):
            reps.append(v)
    if len(reps) < 100:
        return (np.nan, np.nan)
    return (float(np.percentile(reps, 100 * alpha / 2)),
            float(np.percentile(reps, 100 * (1 - alpha / 2))))


# ----------------------------------------------------------------------------
# driver: cell-level and bare-arm estimation
# ----------------------------------------------------------------------------
NB = 2000


def _pull(rows, spec, key):
    """Return the value lists an estimator needs, plus a flag."""
    k = spec["kind"]
    if k in ("two_group", "phi"):
        dv = spec["dv"]
        A = [r["dv"][dv] for r in rows if r["cond"] == spec["hi"] and r["dv"].get(dv) is not None]
        B = [r["dv"][dv] for r in rows if r["cond"] == spec["lo"] and r["dv"].get(dv) is not None]
        return [A, B], len(A) + len(B)
    if k == "corr_within":
        pairs = [(r["dv"][spec["x"]], r["dv"][spec["y"]]) for r in rows]
        return [pairs], len(pairs)
    if k == "corr_between":
        pairs = [(r["dv"][spec["x"]], r["dv"][spec["y"]]) for r in rows]
        return [pairs], len(pairs)
    raise ValueError(k)


def _fn(kind):
    if kind == "two_group":
        return lambda A, B: r_two_group(A, B)
    if kind == "phi":
        return lambda A, B: r_phi(A, B)
    if kind == "corr_within":
        return lambda P: r_corr_within(P)
    if kind == "corr_between":
        return lambda P: r_corr_between([a for a, _ in P], [b for _, b in P])


def seed_aggregate(rows, effect):
    """Collapse the bare arm to one unit per (seed, condition).

    The bare arm is 5 seeds x n = 6 per effect x model. Within-seed subject
    variance is near zero (the six rollouts at one seed are near-duplicates), so
    the seed is the real unit of analysis: each seed x condition cell is reduced
    to the mean of its parsed subject-level DVs before the effect's own statistic
    is applied across the full set of seed-level units.

    For the binary-DV (phi) effects the within-cell responses are unanimous, so a
    seed-level mean is still 0/1 and phi applies unchanged. For 'corr_within'
    the per-subject within-person correlation is Fisher-z averaged inside the
    seed; for 'corr_between' the two subject-level scores are averaged inside
    the seed.
    """
    spec = SPECS[effect]
    k = spec["kind"]
    buckets = {}
    for r in rows:
        buckets.setdefault((r["seed"], r["cond"]), []).append(r["dv"])
    out = []
    for (sd, cond), dvs in sorted(buckets.items()):
        if k in ("two_group", "phi"):
            vals = [x[spec["dv"]] for x in dvs if x.get(spec["dv"]) is not None]
            if not vals:
                continue
            out.append({"cond": cond, "dv": {spec["dv"]: float(np.mean(vals))}})
        elif k == "corr_within":
            zs = []
            for x in dvs:
                a = np.asarray(x[spec["x"]], float); bb = np.asarray(x[spec["y"]], float)
                if a.std() == 0 or bb.std() == 0:
                    continue
                zs.append(_z(np.corrcoef(a, bb)[0, 1]))
            if not zs:
                continue
            out.append({"cond": cond, "_z": float(np.mean(zs))})
        elif k == "corr_between":
            xs = [x[spec["x"]] for x in dvs]; ys = [x[spec["y"]] for x in dvs]
            out.append({"cond": cond,
                        "dv": {spec["x"]: float(np.mean(xs)), spec["y"]: float(np.mean(ys))}})
    return out


def estimate_bare(rows, effect, seed, n_boot=NB):
    """r_bare from seed-aggregated units, same statistic as r_sim."""
    spec = SPECS[effect]
    units = seed_aggregate(rows, effect)
    if spec["kind"] == "corr_within":
        zs = [u["_z"] for u in units]
        if not zs:
            return np.nan, np.nan, np.nan, 0, "no_variance"
        r = float(np.tanh(np.mean(zs)))
        lo, hi = boot_ci(lambda Z: float(np.tanh(np.mean(Z))), [zs],
                           n_boot=n_boot, seed=seed)
        return r, lo, hi, len(zs), ""
    return estimate(units, effect, seed, n_boot=n_boot)


def estimate(rows, effect, seed, n_boot=NB):
    """rows: list of {'cond':..., 'dv': {...}}. Returns (r, lo, hi, n_used, flag)."""
    spec = SPECS[effect]
    groups, n_used = _pull(rows, spec, effect)
    fn = _fn(spec["kind"])
    minn = 2 if spec["kind"] in ("two_group", "phi") else (2 if spec["kind"] == "corr_within" else 3)
    if any(len(g) < minn for g in groups):
        return np.nan, np.nan, np.nan, n_used, "insufficient_n"
    r = fn(*groups)
    if not np.isfinite(r):
        return np.nan, np.nan, np.nan, n_used, "no_variance"
    lo, hi = boot_ci(fn, groups, n_boot=n_boot, seed=seed)
    return float(r), lo, hi, n_used, ""
