"""
p2_parsers_g0.py -- Phase-2 group 0 parsers + effect-size estimation.

Scope: the first 14 of the 42 new effects (alphabetical, 0-indexed 0..13):
    Ackerman2010_haptic_weight, Aryani2020_word_arousal, Bauer2012_consumer_cues,
    BlackBarnes2015_transportation, Blank2022_double_misinformation,
    Cao2018_bayesian_pilot, CarterGilovich2012_experiential_self,
    Caruso2013_money_priming, Correll2007_motherhood_penalty,
    DeNeve1998_swb_neuroticism, DogerliogluDemir2014_incidental_anchor,
    Elliot2010_red_attraction, FathKay2018_hierarchy_corruption,
    Fetherstonhaugh1997_psychophysical_numbing

Every SPEC entry is transcribed from P2_master_protocol_bank.json fields
`response_format` (parsing), `measure` (scoring) and `effect_statistic`
(statistic + sign). Nothing is invented; deviations are flagged in
ORCHESTRATION_NOTES below.

Parsing runs in three tiers (tier 3 is the "tolerant retry" applied only to
cells whose tier-1/2 parse rate falls below 0.60):
  tier 1 STRICT   -- every non-blank line is a well-formed 'LABEL: value' line,
                     the line count matches exactly, values are in range.
  tier 2 LOOSE    -- label:value pairs harvested from anywhere in the text
                     (tolerates a preamble/trailing line); count must match.
  tier 3 TOLERANT -- label-free positional fallback: the in-range values (or
                     A-D letters) are read off in text order; count must match.
Refusals (text is None) are counted separately and are NOT in the parse-rate
denominator. A non-null response that yields no scoreable values (e.g. refusal
prose returned as text) IS a parse failure and stays in the denominator.
"""

import math
import re
import numpy as np

# --------------------------------------------------------------------------
# response-format / scoring specifications
# --------------------------------------------------------------------------
# label kinds:
#   'repeat'  -- n lines carrying the same literal label (RATING:, ANSWER:)
#   'numbered'-- n lines carrying an ordinal label (1:, Item 3:, Q2:, S5:, ITEM7:, PREF2:)
#   'named'   -- n lines carrying distinct named labels, matched by name
# value kinds: 'int', 'num', 'letter'

SPEC = {
    # ---- 0. Ackerman2010_haptic_weight -------------------------------------
    # RF: exactly 6 lines 'RATING: <integer 1-7>'
    # M : composite = mean(item1 overall quality, item2 qualification)
    # S : point-biserial r, positive when heavy > light
    ("Ackerman2010_haptic_weight", "A"): dict(
        label="repeat", pat=r"RATING", n=6, value="int", lo=1, hi=7,
        score="mean_items", items=[1, 2],
        stat="between", pos_cond="heavy", neg_cond="light",
    ),
    ("Ackerman2010_haptic_weight", "B"): dict(
        label="repeat", pat=r"RATING", n=6, value="int", lo=1, hi=7,
        score="mean_items", items=[1, 2],
        stat="between", pos_cond="heavy", neg_cond="light",
    ),

    # ---- 1. Aryani2020_word_arousal ----------------------------------------
    # RF: exactly 12 lines 'N: <1-5>'  (variant B responses often use 'Item N: v')
    # M : kiki items = odd positions 1,3,5,7,9,11; bouba = even 2,4,...,12
    #     target = kiki-mean minus bouba-mean
    # S : paired t vs zero, positive when kiki more arousing
    ("Aryani2020_word_arousal", "A"): dict(
        label="numbered", pat=r"(?:item\s*)?", n=12, value="int", lo=1, hi=5,
        score="mean_diff", items=[1, 3, 5, 7, 9, 11], items2=[2, 4, 6, 8, 10, 12],
        stat="paired_zero",
    ),
    ("Aryani2020_word_arousal", "B"): dict(
        label="numbered", pat=r"(?:item\s*)?", n=12, value="int", lo=1, hi=5,
        score="mean_diff", items=[1, 3, 5, 7, 9, 11], items2=[2, 4, 6, 8, 10, 12],
        stat="paired_zero",
    ),

    # ---- 2. Bauer2012_consumer_cues ----------------------------------------
    # RF: exactly 4 lines 'Q1:'..'Q4:', bare integer 1-7
    # M : Q2 = partner-vs-rival rating, the FReD target in variant A and named
    #     in variant B as "the additional primary target item requested by the
    #     FReD claim". Q2 used for BOTH variants (see ORCHESTRATION_NOTES).
    # S : positive when the individual (control) group rates others as more
    #     partner-like, i.e. consumer cue reduces partner perception.
    ("Bauer2012_consumer_cues", "A"): dict(
        label="numbered", pat=r"Q", n=4, value="int", lo=1, hi=7,
        score="mean_items", items=[2],
        stat="between", pos_cond="individual", neg_cond="consumer",
    ),
    ("Bauer2012_consumer_cues", "B"): dict(
        label="numbered", pat=r"Q", n=4, value="int", lo=1, hi=7,
        score="mean_items", items=[2],
        stat="between", pos_cond="individual", neg_cond="consumer",
    ),

    # ---- 3. BlackBarnes2015_transportation ---------------------------------
    # RF: exactly 7 lines 'RATING: <integer 1-7>'
    # M : A reverses items 2 and 5; B reverses items 6 and 7 (reverse = 8 - x)
    # S : positive when fiction more transported
    ("BlackBarnes2015_transportation", "A"): dict(
        label="repeat", pat=r"RATING", n=7, value="int", lo=1, hi=7,
        score="mean_items", items=[1, 2, 3, 4, 5, 6, 7], reverse=[2, 5], rev_const=8,
        stat="between", pos_cond="fiction", neg_cond="nonfiction",
    ),
    ("BlackBarnes2015_transportation", "B"): dict(
        label="repeat", pat=r"RATING", n=7, value="int", lo=1, hi=7,
        score="mean_items", items=[1, 2, 3, 4, 5, 6, 7], reverse=[6, 7], rev_const=8,
        stat="between", pos_cond="fiction", neg_cond="nonfiction",
    ),

    # ---- 4. Blank2022_double_misinformation --------------------------------
    # RF: exactly 10 lines 'ANSWER: <letter>'
    # M : endorsement (items 1-4) = count of misinformation answers;
    #     availability (items 6-9) = count of correctly reproduced report content.
    #     Item 5 / item 10 are control details, scored but not used.
    # S : Pearson r(availability, endorsement). Sign as worded in the claim
    #     (positive = more availability, more endorsement); FReD records r_o
    #     negative, so the comparison basis for this effect is |r|.
    ("Blank2022_double_misinformation", "A"): dict(
        label="repeat", pat=r"ANSWER", n=10, value="letter", allowed="ABCD",
        score="misinfo_pair",
        endorse={1: "BC", 2: "BC", 3: "B", 4: "B"},
        avail={6: "A", 7: "A", 8: "B", 9: "B"},
        stat="pearson", abs_primary=True,
    ),
    ("Blank2022_double_misinformation", "B"): dict(
        label="repeat", pat=r"ANSWER", n=10, value="letter", allowed="ABCD",
        score="misinfo_pair",
        endorse={1: "BC", 2: "BC", 3: "B", 4: "B"},
        avail={6: "C", 7: "C", 8: "B", 9: "B"},
        stat="pearson", abs_primary=True,
    ),

    # ---- 5. Cao2018_bayesian_pilot -----------------------------------------
    # RF: 'MAN: <0-100>', 'WOMAN: <0-100>'
    # M : target = MAN minus WOMAN
    # S : paired t vs zero, positive when the man is judged more likely
    ("Cao2018_bayesian_pilot", "A"): dict(
        label="named", names=["MAN", "WOMAN"], value="num", lo=0, hi=100,
        score="item_diff", items=[1], items2=[2],
        stat="paired_zero",
    ),
    ("Cao2018_bayesian_pilot", "B"): dict(
        label="named", names=["MAN", "WOMAN"], value="num", lo=0, hi=100,
        score="item_diff", items=[1], items2=[2],
        stat="paired_zero",
    ),

    # ---- 6. CarterGilovich2012_experiential_self ---------------------------
    # RF: exactly 5 lines 'RATING: <integer 1-9>'
    # M : self-insight composite = mean of all 5 items, no reversals
    # S : positive when experiential > material
    ("CarterGilovich2012_experiential_self", "A"): dict(
        label="repeat", pat=r"RATING", n=5, value="int", lo=1, hi=9,
        score="mean_items", items=[1, 2, 3, 4, 5],
        stat="between", pos_cond="experiential", neg_cond="material",
    ),
    ("CarterGilovich2012_experiential_self", "B"): dict(
        label="repeat", pat=r"RATING", n=5, value="int", lo=1, hi=9,
        score="mean_items", items=[1, 2, 3, 4, 5],
        stat="between", pos_cond="experiential", neg_cond="material",
    ),

    # ---- 7. Caruso2013_money_priming ---------------------------------------
    # RF: exactly 8 lines 'S1:'..'S8:', bare integer 1-7
    # M : A reverses S6; B reverses S6 and S7; composite = 8-item mean
    # S : positive when the money group endorses the system more strongly
    ("Caruso2013_money_priming", "A"): dict(
        label="numbered", pat=r"S", n=8, value="int", lo=1, hi=7,
        score="mean_items", items=[1, 2, 3, 4, 5, 6, 7, 8], reverse=[6], rev_const=8,
        stat="between", pos_cond="money", neg_cond="control",
    ),
    ("Caruso2013_money_priming", "B"): dict(
        label="numbered", pat=r"S", n=8, value="int", lo=1, hi=7,
        score="mean_items", items=[1, 2, 3, 4, 5, 6, 7, 8], reverse=[6, 7], rev_const=8,
        stat="between", pos_cond="money", neg_cond="control",
    ),

    # ---- 8. Correll2007_motherhood_penalty ---------------------------------
    # RF: exactly 7 lines 'RATING: <integer>'; lines 1-6 in 1-7, line 7 in 20-80
    # M : competence = mean(item1 competence, item3 reliability)
    # S : positive when the NON-mother is rated more competent (= penalty
    #     present). Variant B's own text used the opposite sign; the master bank
    #     records that sign as harmonised to variant A's, which is applied here.
    ("Correll2007_motherhood_penalty", "A"): dict(
        label="repeat", pat=r"RATING", n=7, value="int",
        lo=[1, 1, 1, 1, 1, 1, 20], hi=[7, 7, 7, 7, 7, 7, 80],
        score="mean_items", items=[1, 3],
        stat="between", pos_cond="nonmother", neg_cond="mother",
    ),
    ("Correll2007_motherhood_penalty", "B"): dict(
        label="repeat", pat=r"RATING", n=7, value="int",
        lo=[1, 1, 1, 1, 1, 1, 20], hi=[7, 7, 7, 7, 7, 7, 80],
        score="mean_items", items=[1, 3],
        stat="between", pos_cond="nonmother", neg_cond="mother",
    ),

    # ---- 9. DeNeve1998_swb_neuroticism -------------------------------------
    # RF: exactly 10 lines 'ITEM1:'..'ITEM10:', bare integer 1-7
    # M : SWB = mean(ITEM1..5). Negative emotionality:
    #     A = mean(ITEM6..10); B = mean(ITEM6,7,8, 8-ITEM9, 8-ITEM10)
    # S : Pearson r(SWB, NE), reported as-is; predicted sign NEGATIVE
    ("DeNeve1998_swb_neuroticism", "A"): dict(
        label="numbered", pat=r"ITEM", n=10, value="int", lo=1, hi=7,
        score="two_composites",
        comp1=[1, 2, 3, 4, 5], comp2=[6, 7, 8, 9, 10], comp2_reverse=[], rev_const=8,
        stat="pearson",
    ),
    ("DeNeve1998_swb_neuroticism", "B"): dict(
        label="numbered", pat=r"ITEM", n=10, value="int", lo=1, hi=7,
        score="two_composites",
        comp1=[1, 2, 3, 4, 5], comp2=[6, 7, 8, 9, 10], comp2_reverse=[9, 10], rev_const=8,
        stat="pearson",
    ),

    # ---- 10. DogerliogluDemir2014_incidental_anchor -------------------------
    # RF: 'WTP: <number>', 'FAIR: <number>'
    # M : target = WTP (FAIR is secondary). Decimals accepted and coerced.
    # S : positive when the high-anchor group states higher WTP
    ("DogerliogluDemir2014_incidental_anchor", "A"): dict(
        label="named", names=["WTP", "FAIR"], value="num", lo=0, hi=10000,
        score="mean_items", items=[1],
        stat="between", pos_cond="high_anchor", neg_cond="low_anchor",
    ),
    ("DogerliogluDemir2014_incidental_anchor", "B"): dict(
        label="named", names=["WTP", "FAIR"], value="num", lo=0, hi=10000,
        score="mean_items", items=[1],
        stat="between", pos_cond="high_anchor", neg_cond="low_anchor",
    ),

    # ---- 11. Elliot2010_red_attraction --------------------------------------
    # RF: 'ATTRACTIVE:', 'DESIRABLE:', 'LIKABLE:', integers 1-9
    # M : attraction composite = mean(ATTRACTIVE, DESIRABLE); LIKABLE discriminant
    # S : positive when the red background yields higher attraction
    ("Elliot2010_red_attraction", "A"): dict(
        label="named", names=["ATTRACTIVE", "DESIRABLE", "LIKABLE"],
        value="int", lo=1, hi=9,
        score="mean_items", items=[1, 2],
        stat="between", pos_cond="red", neg_cond="white",
    ),
    ("Elliot2010_red_attraction", "B"): dict(
        label="named", names=["ATTRACTIVE", "DESIRABLE", "LIKABLE"],
        value="int", lo=1, hi=9,
        score="mean_items", items=[1, 2],
        stat="between", pos_cond="red", neg_cond="white",
    ),

    # ---- 12. FathKay2018_hierarchy_corruption -------------------------------
    # RF: exactly 7 lines 'RATING: <integer 1-7>'
    # M : expected-corruption composite = mean(items 1-5); item 6 mediator,
    #     item 7 trust, neither in the composite
    # S : positive when hierarchical yields more expected corruption
    ("FathKay2018_hierarchy_corruption", "A"): dict(
        label="repeat", pat=r"RATING", n=7, value="int", lo=1, hi=7,
        score="mean_items", items=[1, 2, 3, 4, 5],
        stat="between", pos_cond="hierarchical", neg_cond="flat",
    ),
    ("FathKay2018_hierarchy_corruption", "B"): dict(
        label="repeat", pat=r"RATING", n=7, value="int", lo=1, hi=7,
        score="mean_items", items=[1, 2, 3, 4, 5],
        stat="between", pos_cond="hierarchical", neg_cond="flat",
    ),

    # ---- 13. Fetherstonhaugh1997_psychophysical_numbing --------------------
    # RF: exactly 4 lines 'PREF1:'..'PREF4:', bare integer -6..6 (leading + ok)
    # M : preference score = mean(PREF1..PREF4), no reversals
    # S : positive when the SMALL-camp group prefers the camp programme more
    ("Fetherstonhaugh1997_psychophysical_numbing", "A"): dict(
        label="numbered", pat=r"PREF", n=4, value="int", lo=-6, hi=6,
        score="mean_items", items=[1, 2, 3, 4],
        stat="between", pos_cond="small_camp", neg_cond="large_camp",
    ),
    ("Fetherstonhaugh1997_psychophysical_numbing", "B"): dict(
        label="numbered", pat=r"PREF", n=4, value="int", lo=-6, hi=6,
        score="mean_items", items=[1, 2, 3, 4],
        stat="between", pos_cond="small_camp", neg_cond="large_camp",
    ),
}

ORCHESTRATION_NOTES = """
1. Bauer2012_consumer_cues -- variant A's primary DV is Q2 (partner vs rival),
   the item the FReD claim names ("viewing other people as partners"). Variant
   B's `measure` calls Q1 (trust) the primary published DV but also records Q2
   as "the additional primary target item requested by the FReD claim". Q2 is
   used for both variants so the A/B contrast is on one construct; the sign is
   variant A's (positive = individual/control rated others as more partner-like).
2. Correll2007_motherhood_penalty -- variant B's `effect_statistic` text signs
   the penalty negative; the master bank flags this and harmonises to variant
   A's convention (positive = non-mother rated more competent). A's sign is used.
3. Blank2022_double_misinformation -- FReD records r_o negative while the claim
   as worded is a positive relation. The signed Pearson r is stored in r_sim /
   r_bare; per the bank's instruction the comparison basis for this one effect
   is |r| (abs_primary=True in SPEC).
4. Aryani2020 variant B responses are frequently labelled 'Item N: v' rather
   than 'N: v'; the numbered-label pattern accepts both.
5. Elliot2010 variant B and DogerliogluDemir2014 variant B produced a minority
   of refusal/clarification-request texts that are non-null but carry no
   scoreable values. These count as parse failures, not refusals.
"""

# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
_VALUE_RE = {
    "int": r"[-+]?\d+",
    "num": r"[-+]?\d+(?:\.\d+)?",
    "letter": r"[A-Za-z]",
}


def _rng(spec, i):
    """Per-item (lo, hi); i is 1-based."""
    lo, hi = spec.get("lo"), spec.get("hi")
    if isinstance(lo, list):
        return lo[i - 1], hi[i - 1]
    return lo, hi


def _ok(spec, i, v):
    if spec["value"] == "letter":
        return v.upper() in spec.get("allowed", "ABCD")
    lo, hi = _rng(spec, i)
    if lo is None:
        return True
    return lo <= v <= hi


def _cast(spec, s):
    if spec["value"] == "letter":
        return s.upper()
    s = s.lstrip("+")
    return float(s)


def _label_regex(spec, strict):
    v = _VALUE_RE[spec["value"]]
    kind = spec["label"]
    if kind == "repeat":
        core = rf"{spec['pat']}\s*[:\uff1a]\s*({v})"
    elif kind == "numbered":
        core = rf"{spec['pat']}\s*(\d+)\s*[:\uff1a.]\s*({v})"
    else:  # named
        core = rf"({'|'.join(spec['names'])})\s*[:\uff1a]\s*({v})"
    return re.compile((r"^\s*" + core + r"\s*$") if strict else core,
                      re.IGNORECASE)


def parse_response(text, spec, tier=1):
    """Return (values dict {1-based item -> value}, tier_used) or (None, None).

    tier 1 strict lines, tier 2 loose in-line harvest, tier 3 label-free
    positional fallback.
    """
    if text is None:
        return None, None
    n = spec.get("n", len(spec.get("names", [])))

    for t in range(1, min(tier, 3) + 1):
        vals = _parse_tier(text, spec, n, t)
        if vals is not None:
            return vals, t
    return None, None


def _parse_tier(text, spec, n, t):
    kind = spec["label"]

    if t in (1, 2):
        rx = _label_regex(spec, strict=(t == 1))
        lines = [ln for ln in text.strip().split("\n") if ln.strip()]
        if t == 1 and len(lines) != n:
            return None
        hits = []
        for ln in lines:
            for m in rx.finditer(ln):
                hits.append(m)
                if t == 1:
                    break
        if t == 1 and len(hits) != n:
            return None
        if len(hits) < n:
            return None
        hits = hits[:n]

        vals = {}
        if kind == "repeat":
            for i, m in enumerate(hits, start=1):
                vals[i] = _cast(spec, m.group(1))
        elif kind == "numbered":
            for m in hits:
                idx = int(m.group(1))
                if not 1 <= idx <= n or idx in vals:
                    return None
                vals[idx] = _cast(spec, m.group(2))
            if len(vals) != n:
                return None
        else:  # named
            want = [x.upper() for x in spec["names"]]
            for m in hits:
                nm = m.group(1).upper()
                if nm not in want:
                    return None
                i = want.index(nm) + 1
                if i in vals:
                    return None
                vals[i] = _cast(spec, m.group(2))
            if len(vals) != n:
                return None

    else:  # t == 3, label-free positional
        if spec["value"] == "letter":
            cand = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", text.upper())
        else:
            cand = re.findall(_VALUE_RE[spec["value"]], text)
        keep = []
        for s in cand:
            try:
                v = _cast(spec, s)
            except ValueError:
                continue
            keep.append(v)
        if len(keep) != n:
            return None
        vals = {i: v for i, v in enumerate(keep, start=1)}

    for i, v in vals.items():
        if not _ok(spec, i, v):
            return None
    return vals


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def score_response(vals, spec):
    """Return a float score, or a (x, y) tuple for pearson designs, else None."""
    s = spec["score"]
    rc = spec.get("rev_const", 8)

    def get(i):
        v = vals[i]
        if i in spec.get("reverse", []):
            return rc - v
        return v

    if s == "mean_items":
        return float(np.mean([get(i) for i in spec["items"]]))
    if s == "mean_diff":
        return float(np.mean([vals[i] for i in spec["items"]]) -
                     np.mean([vals[i] for i in spec["items2"]]))
    if s == "item_diff":
        return float(sum(vals[i] for i in spec["items"]) -
                     sum(vals[i] for i in spec["items2"]))
    if s == "two_composites":
        c1 = float(np.mean([vals[i] for i in spec["comp1"]]))
        c2 = float(np.mean([(rc - vals[i]) if i in spec["comp2_reverse"] else vals[i]
                            for i in spec["comp2"]]))
        return (c1, c2)
    if s == "misinfo_pair":
        endorse = sum(1 for i, allowed in spec["endorse"].items()
                      if vals[i] in set(allowed))
        avail = sum(1 for i, correct in spec["avail"].items()
                    if vals[i] in set(correct))
        return (float(avail), float(endorse))
    raise ValueError(f"unknown score kind {s}")


# --------------------------------------------------------------------------
# effect sizes
# --------------------------------------------------------------------------
def _r_between(a, b):
    """a = positive-direction group, b = reference group. r = t/sqrt(t^2+df)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return None
    df = n1 + n2 - 2
    sp2 = ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / df
    if sp2 <= 0:
        d = a.mean() - b.mean()
        return 0.0 if d == 0 else math.copysign(1.0, d)
    t = (a.mean() - b.mean()) / math.sqrt(sp2 * (1 / n1 + 1 / n2))
    return t / math.sqrt(t * t + df)


def _r_paired(d):
    """One-sample t on d vs 0. r = t/sqrt(t^2+df), df = N-1."""
    d = np.asarray(d, float)
    n = len(d)
    if n < 2:
        return None
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0 if d.mean() == 0 else math.copysign(1.0, d.mean())
    t = d.mean() / (sd / math.sqrt(n))
    df = n - 1
    return t / math.sqrt(t * t + df)


def _r_pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def effect_size(spec, data, rng=None):
    """data: dict produced by collect(); returns r or None."""
    st = spec["stat"]
    if st == "between":
        return _r_between(data["pos"], data["neg"])
    if st == "paired_zero":
        return _r_paired(data["d"])
    if st == "pearson":
        return _r_pearson(data["x"], data["y"])
    raise ValueError(st)


def bootstrap_ci(spec, data, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI. Between designs resample within group."""
    rng = np.random.default_rng(seed)
    st = spec["stat"]
    reps = []
    for _ in range(n_boot):
        if st == "between":
            a, b = np.asarray(data["pos"]), np.asarray(data["neg"])
            r = _r_between(rng.choice(a, len(a), replace=True),
                           rng.choice(b, len(b), replace=True))
        elif st == "paired_zero":
            d = np.asarray(data["d"])
            r = _r_paired(rng.choice(d, len(d), replace=True))
        else:
            x, y = np.asarray(data["x"]), np.asarray(data["y"])
            idx = rng.integers(0, len(x), len(x))
            r = _r_pearson(x[idx], y[idx])
        if r is not None and np.isfinite(r):
            reps.append(r)
    if len(reps) < 100:
        return (None, None)
    return (float(np.percentile(reps, 100 * alpha / 2)),
            float(np.percentile(reps, 100 * (1 - alpha / 2))))


# --------------------------------------------------------------------------
# per-cell driver
# --------------------------------------------------------------------------
def collect(rows, spec, tier=1):
    """Parse+score a set of response rows.

    Returns dict with the statistic's input arrays plus counters:
      n_rows, n_refused (text is None), n_denom, n_ok, parse_rate, tiers
    """
    out = dict(pos=[], neg=[], d=[], x=[], y=[],
               n_rows=len(rows), n_refused=0, n_ok=0, tiers={})
    st = spec["stat"]
    for r in rows:
        if r.get("text") is None:
            out["n_refused"] += 1
            continue
        vals, t = parse_response(r["text"], spec, tier=tier)
        if vals is None:
            continue
        sc = score_response(vals, spec)
        out["tiers"][t] = out["tiers"].get(t, 0) + 1
        if st == "between":
            if r.get("cond") == spec["pos_cond"]:
                out["pos"].append(sc)
            elif r.get("cond") == spec["neg_cond"]:
                out["neg"].append(sc)
            else:
                continue
        elif st == "paired_zero":
            out["d"].append(sc)
        else:
            out["x"].append(sc[0])
            out["y"].append(sc[1])
        out["n_ok"] += 1
    out["n_denom"] = out["n_rows"] - out["n_refused"]
    out["parse_rate"] = (out["n_ok"] / out["n_denom"]) if out["n_denom"] else None
    return out
