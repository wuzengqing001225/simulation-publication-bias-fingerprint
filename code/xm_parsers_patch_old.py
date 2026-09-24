"""Cross-model (GPT) parsers + cell/effect-level r estimators for the 19 old A0/A1 effects.

Why this module exists
----------------------
The Claude waves were scored by ``code/a0_scoring.py`` (PARSERS) and
``code/a1_scoring.py`` (PARSERS1).  Those parsers were written against the exact
line formats Claude produced and several of them are brittle in ways that matter
for a different model family:

* ``a0_scoring._ab_lines`` demands *all* N items and silently returns None
  otherwise; it also cannot express per-item value types.
* ``a0_scoring.parse_cooney`` falls back to "first digit anywhere in the text",
  which happily mis-parses a reasoning preamble.
* ``a1_scoring.parse_tversky`` requires a single combined ``N: P, R`` regex and
  cannot read the variant-B ``K_position``/``K_ratio`` layout at all.
* Neither module carries the *variant-B* label vocabularies (RATING, DONATE,
  INTENTIONAL, ATTITUDE, ANSWER, V1..V3, Q1..Q6, S1_I1.., ANGLE, ...), and
  neither carries the scoring keys for Klink / Tamir / Wakslak.

This patch re-implements parsing as a declarative field spec with a three-tier
label search (strict line-anchored -> loose anywhere-in-text -> label-free
positional), so a GPT response with extra whitespace, lower-cased labels, or a
reasoning preamble before the answer block still parses.  Nothing is invented:
a response that does not yield every required field is a parse failure, and
``text is None`` is a refusal, counted separately.

Scoring keys are re-derived from the protocol bank rather than hardcoded:

* Klink back/front vowel key -- from the first vowel of each candidate name in
  the variant's own prompt (u/o = back, i/e = front, a = back per Klink's /ɑ/
  usage).  The rule reproduces the independently-stated variant-B key in the
  protocol ``measure`` field on 12/12 trials, which is what licenses its use on
  variant A.
* Tamir offsets -- regex-extracted pairwise from the 'listed at N cents' lines
  of each variant's own prompt (both variants yield 0,2,4,1,3,0,3,1,4,2,
  matching the ``measure`` text).
* Wakslak BIF abstract-option key -- variant B states A=concrete/B=abstract for
  all 25 items.  For variant A the key is derived by matching each behaviour to
  its variant-B twin and asking which option is the abstract one; 23/25 resolve
  that way.  The 2 residual items (3 'Joining the Army', 25 'Pushing a
  doorbell') are keyed by the purpose/action rule (the option naming a goal,
  outcome or meaning is the abstract one) and are tagged ``route='purpose_rule'``
  in ``WAKSLAK_A_KEY_ROUTE`` so the decision is auditable.

Statistic vocabulary (``stat`` column), matching the protocol ``effect_statistic``
of each effect:
  between       independent-samples t -> r = t/sqrt(t^2+df)
  phi           two-proportion phi = sqrt(chi2/N), signed by p_pos - p_neg
  pearson       Pearson r between two per-participant scores
  paired_zero   one-sample t of a per-participant score vs 0 -> r = t/sqrt(t^2+df)
  prop_vs_half  one-sample proportion vs .5 -> r = 2p - 1

Sign convention throughout: positive r = the direction of the original claim.
"""
import math
import re

import numpy as np

# ==========================================================================
# 1. value parsing primitives
# ==========================================================================

_VALUE_RE = {
    "int": r"[-+]?\d+",
    "num": r"[-+]?\d+(?:\.\d+)?",
    "letter": r"[A-Za-z]",
    "word": r"[A-Za-z]+",
}

# Optional junk that GPT-family models put in front of a label line: bullets,
# markdown emphasis, quotes, brackets.
_PREFIX = r"[\s>*_`\-\u2022\[\(\"']*"
_SUFFIX = r"[\s*_`\]\)\"'.,;]*"
# Accept ASCII colon, full-width colon, en/em dash, equals, or a bare space.
_SEP = r"\s*[:\uff1a=\-\u2013\u2014]\s*|\s+"


def _field_regex(field, strict):
    """Regex matching one field's label + value.

    strict=True anchors the match to a whole line (^...$ with re.M) so a stray
    number elsewhere in the response cannot satisfy the field.
    """
    val = _VALUE_RE[field["value"]]
    names = "|".join(re.escape(n) for n in field["names"])
    # (?<![\d.]) keeps a numeric label from matching inside another number: on
    # the Tversky variant-A line '3: 3, 1.4' the trailing '.4' must not satisfy
    # the label for item 4 (which would then read the next line's value).
    core = rf"(?<![\d.])(?:{names})(?:{_SEP})({val})"
    if strict:
        return re.compile(rf"^{_PREFIX}{core}{_SUFFIX}$", re.I | re.M)
    return re.compile(core, re.I)


def _cast(field, s):
    if field["value"] in ("letter", "word"):
        return s.upper()
    return float(s.lstrip("+"))


def _in_range(field, v):
    if field["value"] in ("letter", "word"):
        return v in [a.upper() for a in field["allowed"]]
    lo, hi = field.get("lo"), field.get("hi")
    if lo is not None and v < lo:
        return False
    if hi is not None and v > hi:
        return False
    return True


def parse_fields(text, spec, max_tier=3):
    """Parse a response into {field_key: value}.

    Returns (values, tier) or (None, None).  Tiers:
      1  every field matched on its own line (label-anchored)
      2  every field matched anywhere in the text (first occurrence wins)
      3  label-free positional: N values of the expected type, in order
         (only offered when every field shares one value type and range)
    """
    if text is None:
        return None, None
    fields = spec["fields"]

    for tier in (1, 2):
        if tier > max_tier:
            break
        vals = {}
        for f in fields:
            m = _field_regex(f, strict=(tier == 1)).search(text)
            if not m:
                vals = None
                break
            v = _cast(f, m.group(1))
            if not _in_range(f, v):
                vals = None
                break
            vals[f["key"]] = v
        if vals is not None and len(vals) == len(fields):
            return vals, tier

    if max_tier < 3:
        return None, None

    kinds = {f["value"] for f in fields}
    if len(kinds) == 1:
        # Homogeneous block: harvest exactly N values of the shared type, in
        # order.  Requiring an exact count (not >= N) keeps a stray number from
        # shifting the whole assignment.
        kind = kinds.pop()
        if kind in ("letter", "word"):
            allowed = "".join(sorted({a.upper() for f in fields for a in f["allowed"]}))
            cand = re.findall(rf"(?<![A-Za-z])([{allowed}])(?![A-Za-z])", text.upper())
        else:
            cand = re.findall(_VALUE_RE[kind], text)
        if len(cand) != len(fields):
            return None, None
        vals = {}
        for f, s in zip(fields, cand):
            v = _cast(f, s)
            if not _in_range(f, v):
                return None, None
            vals[f["key"]] = v
        return vals, 3

    # Heterogeneous fields, label-free: walk the text once, taking for each
    # field in order the next token that matches its type AND range.  Needed for
    # responses that drop the labels but keep the order (GPT's bare-battery
    # Chao answer 'B\n0' for the 'letter, int' pair).
    cursor, vals = 0, {}
    for f in fields:
        if f["value"] in ("letter", "word"):
            allowed = "|".join(re.escape(a) for a in f["allowed"])
            rx = re.compile(rf"(?<![A-Za-z])({allowed})(?![A-Za-z])", re.I)
        else:
            rx = re.compile(_VALUE_RE[f["value"]])
        while True:
            m = rx.search(text, cursor)
            if not m:
                return None, None
            v = _cast(f, m.group(1) if m.lastindex else m.group(0))
            cursor = m.end()
            if _in_range(f, v):
                vals[f["key"]] = v
                break
    return vals, 3


# --- field constructors ---------------------------------------------------

def _f(key, names, value, lo=None, hi=None, allowed="AB"):
    return {"key": key, "names": list(names), "value": value,
            "lo": lo, "hi": hi, "allowed": allowed}


def numbered_letters(n, allowed="AB", prefix=""):
    """N items labelled '1'..'N' (optionally 'Trial 1'..) holding a letter."""
    out = []
    for i in range(1, n + 1):
        names = [f"{prefix}{i}"] if prefix else [str(i)]
        if prefix:
            names.append(str(i))
        out.append(_f(str(i), names, "letter", allowed=allowed))
    return out


def numbered_ints(n, lo, hi, prefix=""):
    out = []
    for i in range(1, n + 1):
        names = [f"{prefix}{i}"] if prefix else [str(i)]
        if prefix:
            names.append(str(i))
        out.append(_f(str(i), names, "int", lo=lo, hi=hi))
    return out


def named_ints(names, lo, hi):
    return [_f(nm, [nm], "int", lo=lo, hi=hi) for nm in names]


# ==========================================================================
# 2. scoring keys re-derived from the protocol bank
# ==========================================================================

# Klink's contrast is the vowel of the stressed first syllable.  The task uses
# the FIRST VOWEL CLUSTER, not the first vowel letter: 'vaylo' and 'zaymo' carry
# the front diphthong /eɪ/ spelled 'ay', so a first-letter rule would mis-key
# them as back.  Digraphs are therefore matched before bare vowels.
_VOWEL_CLUSTER = re.compile(r"(ay|ai|ey|ee|ea|oo|ou|oa|ie|ei|[aeiou])", re.I)
# a = back, per Klink's /ɑ/ usage for the low back vowel.
_BACK_CLUSTERS = {"oo", "ou", "oa", "o", "u", "a"}
_FRONT_CLUSTERS = {"ay", "ai", "ey", "ee", "ea", "ie", "ei", "i", "e"}


def _first_vowel(word):
    """First vowel cluster of a word, lower-cased ('vaylo' -> 'ay')."""
    m = _VOWEL_CLUSTER.search(word)
    return m.group(1).lower() if m else None


def klink_back_key(prompt_template):
    """{trial -> 'A'|'B'} letter of the BACK-vowel name, read off the prompt.

    Handles both variant layouts:
      A:  '1. A. nillen    B. nullen'
      B:  'Trial 1: A = "beelo" | B = "boolo"'
    """
    pairs = re.findall(r"^\s*(\d+)\.\s*A\.\s*(\S+)\s+B\.\s*(\S+)\s*$",
                       prompt_template, re.M)
    if not pairs:
        pairs = re.findall(r"Trial\s*(\d+)\s*:\s*A\s*=\s*\"?(\w+)\"?\s*\|\s*B\s*=\s*\"?(\w+)\"?",
                           prompt_template)
    key = {}
    for n, a, b in pairs:
        va, vb = _first_vowel(a), _first_vowel(b)
        a_back = va in _BACK_CLUSTERS
        b_back = vb in _BACK_CLUSTERS
        a_front = va in _FRONT_CLUSTERS
        b_front = vb in _FRONT_CLUSTERS
        if a_back and b_front:
            key[int(n)] = "A"
        elif b_back and a_front:
            key[int(n)] = "B"
        else:
            key[int(n)] = None          # not a clean back/front contrast
    return key


def klink_key_from_measure(measure_text):
    """The variant-B key as stated verbatim in the protocol 'measure' field."""
    return {int(a): b for a, b in re.findall(r"Trial(\d+)\s*back=([AB])", measure_text)}


def tamir_offsets(prompt_template):
    """{trial -> offset in cents} = OTHER amount minus SELF amount, floored at 0."""
    blocks = re.findall(
        r"Trial\s*(\d+)\s*\n\s*A \(self\):.*?listed at (\d+) cents?\s*\n"
        r"\s*B \(other\):.*?listed at (\d+) cents?",
        prompt_template, re.S)
    if not blocks:
        blocks = re.findall(
            r"Trial\s*(\d+):\s*\[SELF,\s*listed at (\d+) cents?\].*?"
            r"\[OTHER,\s*listed at (\d+) cents?\]",
            prompt_template)
    return {int(n): max(0, int(o) - int(s)) for n, s, o in blocks}


# --- Wakslak BIF -----------------------------------------------------------

_PURPOSE_LEX = re.compile(
    r"\b(getting|gaining|showing|revealing|preventing|protecting|maintaining|"
    r"securing|influencing|helping|serving|teaching|making|removing|seeing|"
    r"knowledge|defense|organized|nutrition|courage|friendliness|remodel|"
    r"cleanliness|fresh|nice|decay|view|clean)\b", re.I)
_ACTION_LEX = re.compile(
    r"\b(writing|following|signing|putting|pulling|wielding|using|vacuuming|"
    r"applying|watering|marking|holding|answering|moving|saying|planting|"
    r"chewing|swallowing|going|washing|brush strokes|lines of print|"
    r"yardstick|tape measure|key in the lock|finger|branches|axe|machine|"
    r"check|ballot|novocain)\b", re.I)


def _bif_items(prompt_template):
    """[(n, behaviour, option_A, option_B)] from either variant's BIF block."""
    items = re.findall(r"^\s*(\d+)\.\s*(.+?)\n\s*A\.\s*(.+?)\n\s*B\.\s*(.+?)$",
                       prompt_template, re.M)
    return [(int(n), b.strip(), a.strip(), c.strip()) for n, b, a, c in items]


def _toks(s):
    stop = {"a", "an", "the", "of", "one", "s", "to", "in", "into", "on", "off",
            "up", "out", "something", "someone", "your", "yours", "you"}
    return set(re.findall(r"[a-z]+", s.lower())) - stop


def _jac(x, y):
    x, y = _toks(x), _toks(y)
    return len(x & y) / max(1, len(x | y))


def _purpose_score(option):
    """+1 leaning abstract (states a goal/outcome), -1 leaning concrete."""
    return len(_PURPOSE_LEX.findall(option)) - len(_ACTION_LEX.findall(option))


def wakslak_abstract_key(prompt_a, prompt_b, min_sim=0.6, min_margin=0.05):
    """{item -> 'A'|'B'} letter of the ABSTRACT (high-level) option in variant A.

    Variant B fixes A=concrete, B=abstract for all 25 items (stated in its
    'measure' field), so each variant-A item is matched to its variant-B twin by
    behaviour text and the option closer to B's abstract option wins.  Items the
    match cannot resolve fall through to the purpose/action rule.  The route
    taken for every item is recorded in the returned ``route`` dict.
    """
    a_items, b_items = _bif_items(prompt_a), _bif_items(prompt_b)
    key, route = {}, {}
    for n, beh, oa, ob in a_items:
        twin, sim = None, -1.0
        for n2, beh2, oa2, ob2 in b_items:
            s = _jac(beh, beh2)
            if s > sim:
                sim, twin = s, (n2, beh2, oa2, ob2)
        decided = None
        if twin and sim >= min_sim:
            _, _, concrete_b, abstract_b = twin
            d_a = _jac(oa, abstract_b) - _jac(oa, concrete_b)
            d_b = _jac(ob, abstract_b) - _jac(ob, concrete_b)
            if abs(d_a - d_b) > min_margin:
                decided = "A" if d_a > d_b else "B"
                route[n] = f"cross_variant(sim={sim:.2f})"
        if decided is None:
            p_a, p_b = _purpose_score(oa), _purpose_score(ob)
            if p_a != p_b:
                decided = "A" if p_a > p_b else "B"
                route[n] = f"purpose_rule(pA={p_a:+d},pB={p_b:+d})"
            else:
                route[n] = "UNRESOLVED"
        key[n] = decided
    return key, route


# ==========================================================================
# 3. effect-size estimators (r on the FReD scale, positive = original claim)
# ==========================================================================

def r_between(pos, neg):
    """Independent-samples t -> r = t/sqrt(t^2+df). pos = claim-direction group."""
    a, b = np.asarray(pos, float), np.asarray(neg, float)
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


def r_phi(pos, neg):
    """Two-proportion phi = sqrt(chi2/N) over the 2x2, signed by p_pos - p_neg."""
    a, b = np.asarray(pos, float), np.asarray(neg, float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return None
    p1, p2 = a.mean(), b.mean()
    p = (a.sum() + b.sum()) / (n1 + n2)
    if p in (0.0, 1.0):
        return 0.0
    chi2 = (n1 * n2 / (n1 + n2)) * (p1 - p2) ** 2 / (p * (1 - p))
    return math.copysign(math.sqrt(chi2 / (n1 + n2)), p1 - p2)


_SD_TOL = 1e-12   # a "constant" column can carry float dust from mean-of-items


def r_pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() <= _SD_TOL or y.std() <= _SD_TOL:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def r_paired_zero(d):
    """One-sample t of per-participant scores vs 0 -> r = t/sqrt(t^2+df)."""
    d = np.asarray(d, float)
    n = len(d)
    if n < 2:
        return None
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0 if d.mean() == 0 else math.copysign(1.0, d.mean())
    t = d.mean() / (sd / math.sqrt(n))
    return t / math.sqrt(t * t + (n - 1))


def r_prop_vs_half(binary):
    """One-sample proportion vs .5; r = 2p - 1 (the protocol's stated reduction)."""
    b = np.asarray(binary, float)
    if len(b) < 2:
        return None
    return float(2 * b.mean() - 1)


_ESTIMATOR = {
    "between": lambda d: r_between(d["pos"], d["neg"]),
    "phi": lambda d: r_phi(d["pos"], d["neg"]),
    "pearson": lambda d: r_pearson(d["x"], d["y"]),
    "paired_zero": lambda d: r_paired_zero(d["d"]),
    "prop_vs_half": lambda d: r_prop_vs_half(d["d"]),
}


def effect_size(stat, data):
    return _ESTIMATOR[stat](data)


def bootstrap_ci(stat, data, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap; two-group designs resample within group."""
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        if stat in ("between", "phi"):
            a, b = np.asarray(data["pos"]), np.asarray(data["neg"])
            if len(a) == 0 or len(b) == 0:
                return (None, None)
            boot = {"pos": rng.choice(a, len(a), replace=True),
                    "neg": rng.choice(b, len(b), replace=True)}
        elif stat == "pearson":
            x, y = np.asarray(data["x"]), np.asarray(data["y"])
            if len(x) == 0:
                return (None, None)
            idx = rng.integers(0, len(x), len(x))
            boot = {"x": x[idx], "y": y[idx]}
        else:
            d = np.asarray(data["d"])
            if len(d) == 0:
                return (None, None)
            boot = {"d": rng.choice(d, len(d), replace=True)}
        r = effect_size(stat, boot)
        if r is not None and np.isfinite(r):
            reps.append(r)
    if len(reps) < 100:
        return (None, None)
    return (float(np.percentile(reps, 100 * alpha / 2)),
            float(np.percentile(reps, 100 * (1 - alpha / 2))))


# ==========================================================================
# 4. per-effect specs: fields, scoring, statistic, condition mapping
# ==========================================================================
# Each entry is keyed (effect, variant) -- and (effect, 'bare') resolves to the
# variant-A spec, because run_xmodel.py renders the bare battery from the
# variant-A templates (confirmed against the observed bare label vocabulary and
# the variant-A condition names in the bare file).
#
#   fields   : list of field dicts consumed by parse_fields()
#   score    : (vals, cond) -> float | (x, y) tuple | None
#   stat     : one of the statistic keys above
#   pos/neg  : condition-name tuples defining the claim-positive / reference
#              group for two-group designs (multiple names allowed: Valdesolo
#              pools two conditions per side)
#   custom   : optional (text -> vals | None) parser replacing parse_fields
#   sample_z : True if the per-participant score needs sample standardisation
#              before the statistic (Slepian)

SPECS = {}


def _spec(effect, variant, **kw):
    SPECS[(effect, variant)] = kw


# ---- Asch 1946: item-1 generous choice, phi across warm/cold -------------
for _v in ("A", "B"):
    _spec("Asch1946_warm_cold", _v,
          fields=numbered_letters(18),
          score=lambda v, c: 1.0 if v["1"] == "A" else 0.0,
          stat="phi", pos=("warm",), neg=("cold",))

# ---- Banerjee 2012: room brightness rating ------------------------------
_spec("Banerjee2012_brightness", "A",
      fields=[_f("bright", ["BRIGHTNESS"], "int", lo=1, hi=7)],
      score=lambda v, c: v["bright"],
      stat="between", pos=("ethical",), neg=("unethical",))
_spec("Banerjee2012_brightness", "B",
      fields=[_f("bright", ["RATING", "BRIGHTNESS"], "int", lo=1, hi=7)],
      score=lambda v, c: v["bright"],
      stat="between", pos=("ethical",), neg=("unethical",))

# ---- Bargh 2012: r(physical-warmth-extraction index, loneliness) ---------
# Variant A: S1.5/6/7 = bathing frequency/temperature/duration; S2.1-10 lonely.
_spec("Bargh2012_warmth", "A",
      fields=([_f(f"S1.{i}", [f"S1.{i}"], "int", lo=1, hi=9) for i in range(1, 8)] +
              [_f(f"S2.{i}", [f"S2.{i}"], "int", lo=1, hi=4) for i in range(1, 11)]),
      score=lambda v, c: ((v["S1.5"] + v["S1.6"] + v["S1.7"]) / 3.0,
                          float(np.mean([v[f"S2.{i}"] for i in range(1, 11)]))),
      stat="pearson")
# Variant B: B1/B2/B3 = frequency/temperature/duration; L1-L10 lonely (sum).
_spec("Bargh2012_warmth", "B",
      fields=([_f("F1", ["F1"], "int", lo=1, hi=5),
               _f("F2", ["F2"], "int", lo=0, hi=10000),
               _f("B1", ["B1"], "int", lo=1, hi=8),
               _f("B2", ["B2"], "int", lo=1, hi=6),
               _f("B3", ["B3"], "int", lo=1, hi=7)] +
              [_f(f"L{i}", [f"L{i}"], "int", lo=1, hi=4) for i in range(1, 11)]),
      score=lambda v, c: ((v["B1"] + v["B2"] + v["B3"]) / 3.0,
                          float(sum(v[f"L{i}"] for i in range(1, 11)))),
      stat="pearson")

# ---- Chao 2017: donation rate, crowding out ------------------------------
_spec("Chao2017_thankyou_gift", "A",
      fields=[_f("1", ["1"], "letter", allowed="AB"),
              _f("2", ["2"], "int", lo=0, hi=10)],
      score=lambda v, c: 1.0 if v["1"] == "A" else 0.0,
      stat="phi", pos=("no_gift",), neg=("gift_salient",))


def _chao_b(text):
    """Variant-B tolerant parser.

    The protocol asks for 'DONATE: 1|0' + 'AMOUNT: 0-5'.  GPT answered with a
    free-form dollar figure only ('Donation amount: $2.00', '[Donation: $2]',
    'I will donate $1.').  The amount is recoverable, so DONATE is taken as
    amount > 0 -- which is what the labelled form would have encoded -- and the
    cell is flagged, because a sample in which every response names a positive
    amount carries no variance on the binary primary DV.
    """
    if text is None:
        return None
    m = re.search(r"DONATE(?:\s*[:\uff1a=]\s*)([01])\b", text, re.I)
    if m:
        donate = float(m.group(1))
        a = re.search(r"AMOUNT(?:\s*[:\uff1a=]\s*)\$?\s*(\d+(?:\.\d+)?)", text, re.I)
        return {"donate": donate, "amount": float(a.group(1)) if a else None}
    m = re.search(r"\$\s*(\d+(?:\.\d+)?)", text)
    if not m:
        m = re.search(r"(?:donat\w*|amount)\D{0,20}?(\d+(?:\.\d+)?)", text, re.I)
    if not m:
        return None
    amt = float(m.group(1))
    if not (0 <= amt <= 5):
        return None
    return {"donate": 1.0 if amt > 0 else 0.0, "amount": amt}


_spec("Chao2017_thankyou_gift", "B",
      fields=[], custom=_chao_b,
      score=lambda v, c: v["donate"],
      stat="phi", pos=("no_gift",), neg=("salient_gift",))

# ---- Cooney 2016: predicted happiness under fair vs unfair procedure -----
_spec("Cooney2016_fairness", "A",
      fields=[_f("h", ["HAPPINESS"], "int", lo=1, hi=9)],
      score=lambda v, c: v["h"],
      stat="between", pos=("fair_procedure",), neg=("unfair_procedure",))
_spec("Cooney2016_fairness", "B",
      fields=[_f("h", ["HAPPINESS"], "int", lo=0, hi=10)],
      score=lambda v, c: v["h"],
      stat="between", pos=("fair_procedure",), neg=("unfair_procedure",))

# ---- Eyal 2008: judged wrongness, near vs distant future ----------------
_spec("Eyal2008_moral_distance", "A",
      fields=numbered_ints(3, -5, 5),
      score=lambda v, c: -float(np.mean([v["1"], v["2"], v["3"]])),
      stat="between", pos=("distant_future",), neg=("near_future",))
_spec("Eyal2008_moral_distance", "B",
      fields=[_f(f"V{i}", [f"V{i}"], "int", lo=-5, hi=5) for i in (1, 2, 3)],
      score=lambda v, c: -float(np.mean([v["V1"], v["V2"], v["V3"]])),
      stat="between", pos=("distant",), neg=("near",))

# ---- Genschow 2017: correspondence bias, control vs anti-free-will -------
_spec("Genschow2017_free_will_attribution", "A",
      fields=([_f(f"S{s}.{L}", [f"S{s}.{L}"], "int", lo=1, hi=7)
               for s in range(1, 5) for L in "abcd"] +
              [_f(f"F{i}", [f"F{i}"], "int", lo=1, hi=7) for i in range(1, 12)]),
      score=lambda v, c: float(
          np.mean([v[f"S{s}.{L}"] for s in range(1, 5) for L in "ab"]) -
          np.mean([v[f"S{s}.{L}"] for s in range(1, 5) for L in "cd"])),
      stat="between", pos=("control",), neg=("anti_free_will",))
_spec("Genschow2017_free_will_attribution", "B",
      fields=([_f(f"S{s}_{k}", [f"S{s}_{k}"], "int", lo=1, hi=7)
               for s in range(1, 5) for k in ("I1", "I2", "E1", "E2")] +
              named_ints([f"FW{i}" for i in (1, 2, 3)], 1, 7) +
              named_ints([f"D{i}" for i in (1, 2, 3)], 1, 7) +
              named_ints([f"DET{i}" for i in (1, 2, 3)], 1, 7)),
      score=lambda v, c: float(np.mean([
          (v[f"S{s}_I1"] + v[f"S{s}_I2"]) / 2.0 - (v[f"S{s}_E1"] + v[f"S{s}_E2"]) / 2.0
          for s in range(1, 5)])),
      stat="between", pos=("control",), neg=("anti_free_will",))

# ---- Husnu 2010: future contact intentions ------------------------------
_spec("Husnu2010_imagined_contact", "A",
      fields=numbered_ints(4, 1, 9),
      score=lambda v, c: float(np.mean([v[str(i)] for i in range(1, 5)])),
      stat="between", pos=("imagined_contact",), neg=("control",))
_spec("Husnu2010_imagined_contact", "B",
      fields=named_ints([f"Q{i}" for i in range(1, 7)], 1, 7),
      score=lambda v, c: float(np.mean([v[f"Q{i}"] for i in range(1, 6)])),
      stat="between", pos=("imagined_contact",), neg=("control",))

# ---- Klink 2000: back-vowel 'richer' preference vs chance ---------------
# score = back-vowel choice proportion minus .5, tested against 0 (one-sample t),
# exactly the variant-B protocol statistic; the key is injected at build time.
def _klink_score(key):
    items = [i for i, letter in key.items() if letter in ("A", "B")]

    def f(v, c):
        hits = [1.0 if v[str(i)] == key[i] else 0.0 for i in items]
        return float(np.mean(hits)) - 0.5
    return f


_spec("Klink2000_soundsymbolism", "A",
      fields=numbered_letters(12), score=None, stat="paired_zero",
      needs_key="klink")
_spec("Klink2000_soundsymbolism", "B",
      fields=numbered_letters(12, prefix="Trial "), score=None,
      stat="paired_zero", needs_key="klink")

# ---- Knobe 2003: intentionality, harm vs help ---------------------------
_spec("Knobe2003_side_effect", "A",
      fields=[_f("1", ["1"], "letter", allowed="AB"),
              _f("2", ["2"], "int", lo=1, hi=7)],
      score=lambda v, c: 1.0 if v["1"] == "A" else 0.0,
      stat="phi", pos=("harm",), neg=("help",))
_spec("Knobe2003_side_effect", "B",
      fields=[_f("intentional", ["INTENTIONAL"], "int", lo=0, hi=1),
              _f("moral", ["MORAL"], "int", lo=0, hi=6)],
      score=lambda v, c: float(v["intentional"]),
      stat="phi", pos=("harm",), neg=("help",))

# ---- Miyamoto 2002: attributed attitude, pro vs anti essay --------------
_spec("Miyamoto2002_correspondence_bias", "A",
      fields=[_f("1", ["1"], "int", lo=1, hi=9), _f("2", ["2"], "int", lo=1, hi=5)],
      score=lambda v, c: v["1"],
      stat="between", pos=("pro_essay",), neg=("anti_essay",))
_spec("Miyamoto2002_correspondence_bias", "B",
      fields=[_f("att", ["ATTITUDE"], "int", lo=1, hi=9),
              _f("conf", ["CONFIDENCE"], "int", lo=1, hi=9)],
      score=lambda v, c: v["att"],
      stat="between", pos=("pro_essay",), neg=("anti_essay",))

# ---- Oppenheimer 2009: estimated prior rolls ---------------------------
_spec("Oppenheimer2009_retrospective_gambler", "A",
      fields=[_f("rolls", ["ROLLS"], "int", lo=0, hi=1000000)],
      score=lambda v, c: v["rolls"],
      stat="between", pos=("rare_three_sixes",), neg=("two_dice_two_sixes",))
_spec("Oppenheimer2009_retrospective_gambler", "B",
      fields=[_f("rolls", ["ANSWER", "ROLLS"], "int", lo=0, hi=1000000)],
      score=lambda v, c: v["rolls"],
      stat="between", pos=("three_sixes_rare",), neg=("two_sixes_streak",))

# ---- Rottenstreich 2001: affect-rich prize choice ----------------------
_spec("Rottenstreich2001_affective_lottery", "A",
      fields=[_f("choice", ["CHOICE"], "letter", allowed="AB"),
              _f("strength", ["STRENGTH"], "int", lo=1, hi=5)],
      score=lambda v, c: 1.0 if v["choice"] == "A" else 0.0,
      stat="phi", pos=("low_probability",), neg=("certainty",))
_spec("Rottenstreich2001_affective_lottery", "B",
      fields=[_f("choice", ["CHOICE"], "letter", allowed="AB")],
      score=lambda v, c: 1.0 if v["choice"] == "A" else 0.0,
      stat="phi", pos=("low_probability",), neg=("certainty",))

# ---- Slepian 2012: z(slant) - z(distance) contrast ---------------------
# The per-participant index needs sample standardisation, so score returns the
# raw pair and the driver z-scores each component within the cell.
_spec("Slepian2012_secrets", "A",
      fields=[_f("slant", ["SLANT", "ANGLE"], "num", lo=0, hi=90),
              _f("dist", ["DISTANCE"], "num", lo=0, hi=100000)],
      score=lambda v, c: (v["slant"], v["dist"]),
      stat="between", pos=("big_secret",), neg=("small_secret",), sample_z=True)
_spec("Slepian2012_secrets", "B",
      fields=[_f("slant", ["ANGLE", "SLANT"], "num", lo=0, hi=90),
              _f("dist", ["DISTANCE"], "num", lo=0, hi=100000)],
      score=lambda v, c: (v["slant"], v["dist"]),
      stat="between", pos=("big_secret",), neg=("small_secret",), sample_z=True)

# ---- Tamir 2012: money forgone to answer the SELF question -------------
def _tamir_score(offsets, self_token):
    def f(v, c):
        return float(np.mean([offsets[i] if v[str(i)] == self_token else 0.0
                              for i in range(1, 11)]))
    return f


_spec("Tamir2012_selfdisclosure", "A",
      fields=numbered_letters(10), score=None, stat="paired_zero",
      needs_key="tamir", self_token="A")
_spec("Tamir2012_selfdisclosure", "B",
      fields=[_f(str(i), [f"Trial {i}", str(i)], "word", allowed=["SELF", "OTHER"])
              for i in range(1, 11)],
      score=None, stat="paired_zero", needs_key="tamir", self_token="SELF")

# ---- Tversky 1973: first-position majority vs chance -------------------
_spec("Tversky1973_availability_letters", "A",
      fields=[_f(str(i), [str(i)], "int", lo=1, hi=3) for i in range(1, 6)],
      score=lambda v, c: 1.0 if sum(1 for i in range(1, 6) if v[str(i)] == 1) >= 3 else 0.0,
      stat="prop_vs_half")
_spec("Tversky1973_availability_letters", "B",
      fields=([_f(f"{L}_position", [f"{L}_position"], "letter", allowed="FT")
               for L in "KLNRV"] +
              [_f(f"{L}_ratio", [f"{L}_ratio"], "num", lo=0, hi=100000)
               for L in "KLNRV"]),
      score=lambda v, c: 1.0 if sum(1 for L in "KLNRV" if v[f"{L}_position"] == "F") >= 3 else 0.0,
      stat="prop_vs_half")

# ---- Valdesolo 2007: +1 -1 +1 -1 contrast on fairness ratings ----------
_spec("Valdesolo2007_hypocrisy", "A",
      fields=[_f("fair", ["FAIRNESS", "Fairness rating"], "int", lo=1, hi=7)],
      score=lambda v, c: v["fair"], stat="between",
      pos=("self_actor", "ingroup_actor"), neg=("other_actor", "outgroup_actor"))
_spec("Valdesolo2007_hypocrisy", "B",
      fields=[_f("fair", ["Fairness rating", "FAIRNESS"], "int", lo=1, hi=7)],
      score=lambda v, c: v["fair"], stat="between",
      pos=("self_actor", "ingroup_actor"), neg=("other_actor", "outgroup_actor"))

# ---- Van Boven 2007: anticipation minus retrospection -----------------
# Variant A's Q1/Q2 are position-coded, so the order condition maps them back.
def _vb_a(v, c):
    if c == "order_future_first":
        return float(v["1"] - v["2"])
    if c == "order_past_first":
        return float(v["2"] - v["1"])
    return None


_spec("VanBoven2007_anticipation", "A",
      fields=numbered_ints(2, 1, 9, prefix="Q"),
      score=_vb_a, stat="paired_zero")
_spec("VanBoven2007_anticipation", "B",
      fields=[_f("ant", ["ANTICIPATION"], "int", lo=1, hi=9),
              _f("ret", ["RETROSPECTION"], "int", lo=1, hi=9)],
      score=lambda v, c: float(v["ant"] - v["ret"]), stat="paired_zero")

# ---- Wakslak 2006: BIF abstractness, low vs high likelihood -----------
def _wakslak_score(key):
    items = [i for i, letter in key.items() if letter in ("A", "B")]

    def f(v, c):
        return float(sum(1 for i in items if v[str(i)] == key[i]))
    return f


_spec("Wakslak2006_construal", "A",
      fields=numbered_letters(25), score=None, stat="between",
      pos=("low_likelihood",), neg=("high_likelihood",), needs_key="wakslak")
_spec("Wakslak2006_construal", "B",
      fields=numbered_letters(25),
      score=lambda v, c: float(sum(1 for i in range(1, 26) if v[str(i)] == "B")),
      stat="between", pos=("low_likelihood",), neg=("high_likelihood",))


# ==========================================================================
# 5. bind the bank-derived scoring keys into the specs
# ==========================================================================
KEY_AUDIT = {}          # human-readable record of every derived key
WAKSLAK_A_KEY_ROUTE = {}


def bind_keys(bank_protocols):
    """Fill in the specs whose scoring key must be read off the protocol bank.

    bank_protocols: the list under P2_master_protocol_bank.json['protocols'].
    Returns KEY_AUDIT.  Raises if a required protocol is missing -- a missing
    key must fail loudly rather than score silently against a guess.
    """
    P = {(p["effect"], p["variant"]): p for p in bank_protocols}

    # -- Klink: derive from each variant's own name pairs, validate against the
    #    variant-B key stated in its measure field.
    stated = klink_key_from_measure(P[("Klink2000_soundsymbolism", "B")]["measure"])
    for v in ("A", "B"):
        derived = klink_back_key(P[("Klink2000_soundsymbolism", v)]["subject_prompt_template"])
        if not derived or any(x is None for x in derived.values()):
            raise ValueError(f"Klink variant {v}: vowel rule left items unkeyed: {derived}")
        SPECS[("Klink2000_soundsymbolism", v)]["score"] = _klink_score(derived)
        KEY_AUDIT[f"Klink2000_soundsymbolism/{v}"] = derived
    agree = sum(1 for k, val in stated.items() if derived.get(k) == val)
    KEY_AUDIT["Klink2000_soundsymbolism/validation"] = (
        f"vowel rule vs protocol-stated variant-B key: {agree}/{len(stated)} agree")
    if agree != len(stated):
        raise ValueError("Klink vowel rule disagrees with the protocol-stated key")

    # -- Tamir: offsets from each variant's own payment table.
    for v in ("A", "B"):
        offs = tamir_offsets(P[("Tamir2012_selfdisclosure", v)]["subject_prompt_template"])
        if sorted(offs) != list(range(1, 11)):
            raise ValueError(f"Tamir variant {v}: payment table incomplete: {offs}")
        sp = SPECS[("Tamir2012_selfdisclosure", v)]
        sp["score"] = _tamir_score(offs, sp["self_token"])
        KEY_AUDIT[f"Tamir2012_selfdisclosure/{v}"] = [offs[i] for i in range(1, 11)]

    # -- Wakslak: variant B states A=concrete/B=abstract; variant A is derived.
    key_a, route = wakslak_abstract_key(
        P[("Wakslak2006_construal", "A")]["subject_prompt_template"],
        P[("Wakslak2006_construal", "B")]["subject_prompt_template"])
    unresolved = [i for i, x in key_a.items() if x is None]
    if unresolved:
        raise ValueError(f"Wakslak variant A: items unkeyed {unresolved}")
    SPECS[("Wakslak2006_construal", "A")]["score"] = _wakslak_score(key_a)
    WAKSLAK_A_KEY_ROUTE.update(route)
    KEY_AUDIT["Wakslak2006_construal/A"] = key_a
    KEY_AUDIT["Wakslak2006_construal/A_route"] = route
    KEY_AUDIT["Wakslak2006_construal/B"] = {i: "B" for i in range(1, 26)}
    return KEY_AUDIT


def get_spec(effect, variant):
    """Variant lookup; 'bare' resolves to the variant-A spec (run_xmodel.py
    renders the bare battery from the variant-A templates)."""
    if variant == "bare":
        variant = "A"
    return SPECS.get((effect, variant))


# ==========================================================================
# 6. drivers: cell-level (main) and effect-level (bare)
# ==========================================================================

# GPT declines out of role in prose rather than returning text=None, which the
# Claude-era pipeline used as its refusal signal.  These are recorded separately
# (n_refusal_text) so refusals are not silently booked as format failures; the
# parse-rate denominator convention is left unchanged.
_REFUSAL_TEXT = re.compile(
    r"\b(?:I\s+(?:can'?t|cannot|can\u2019t|won'?t|am\s+unable\s+to|do\s+not|don'?t)\b"
    r"[^.\n]{0,80}?(?:provide|answer|respond|complete|give|simulate|pretend|have)"
    r"|as\s+an\s+AI\b|I'?m\s+an\s+AI\b|I\s+don\u2019t\s+have\s+personal)", re.I)


def looks_like_refusal(text):
    return bool(text) and bool(_REFUSAL_TEXT.search(text))


def parse_rows(rows, spec, max_tier=3):
    """Parse+score a set of response rows.

    Returns dict with:
      recs      [{cond, score, tier}]  score is float or (x, y)
      n_rows / n_refused / n_parse_fail / n_ok / n_denom / parse_rate
      tiers     {tier -> count}
    A refusal is text is None (or an error with no text); everything else that
    fails to yield all required fields is a parse failure.
    """
    out = {"recs": [], "n_rows": len(rows), "n_refused": 0, "n_refusal_text": 0,
           "n_parse_fail": 0, "tiers": {}}
    custom = spec.get("custom")
    for r in rows:
        text = r.get("text")
        if text is None or (isinstance(text, str) and not text.strip()):
            out["n_refused"] += 1
            continue
        if looks_like_refusal(text):
            out["n_refusal_text"] += 1
            out["n_parse_fail"] += 1
            continue
        if custom is not None:
            vals, tier = custom(text), 1
            if vals is None:
                tier = None
        else:
            vals, tier = parse_fields(text, spec, max_tier=max_tier)
        if vals is None:
            out["n_parse_fail"] += 1
            continue
        sc = spec["score"](vals, r.get("cond"))
        if sc is None:
            out["n_parse_fail"] += 1
            continue
        out["tiers"][tier] = out["tiers"].get(tier, 0) + 1
        out["recs"].append({"cond": r.get("cond"), "score": sc, "tier": tier})
    out["n_ok"] = len(out["recs"])
    out["n_denom"] = out["n_rows"] - out["n_refused"]
    out["parse_rate"] = (out["n_ok"] / out["n_denom"]) if out["n_denom"] else None
    return out


def _zs(vals):
    a = np.asarray(vals, float)
    sd = a.std(ddof=0)
    return np.zeros_like(a) if sd == 0 else (a - a.mean()) / sd


def build_data(parsed, spec):
    """Turn parse_rows() output into the estimator's input arrays.

    Applies sample standardisation for sample_z specs (Slepian: the individual
    index is z(slant) - z(distance), standardised within the cell) and splits
    two-group designs on the spec's pos/neg condition names.
    """
    recs = parsed["recs"]
    stat = spec["stat"]

    if spec.get("sample_z"):
        if not recs:
            return {"pos": [], "neg": []}, 0
        zx = _zs([r["score"][0] for r in recs])
        zy = _zs([r["score"][1] for r in recs])
        for r, a, b in zip(recs, zx, zy):
            r["score"] = float(a - b)

    if stat == "pearson":
        return ({"x": [r["score"][0] for r in recs],
                 "y": [r["score"][1] for r in recs]}, len(recs))
    if stat in ("paired_zero", "prop_vs_half"):
        return {"d": [r["score"] for r in recs]}, len(recs)

    pos = [r["score"] for r in recs if r["cond"] in spec["pos"]]
    neg = [r["score"] for r in recs if r["cond"] in spec["neg"]]
    return {"pos": pos, "neg": neg}, len(pos) + len(neg)


def score_cell(rows, effect, variant, n_boot=2000, seed=0, max_tier=3):
    """One (effect, variant) cell -> a row for XM_cells_old.csv."""
    spec = get_spec(effect, variant)
    if spec is None:
        return None
    parsed = parse_rows(rows, spec, max_tier=max_tier)
    data, n_used = build_data(parsed, spec)
    r = effect_size(spec["stat"], data) if n_used else None
    ci_lo, ci_hi = ((None, None) if r is None
                    else bootstrap_ci(spec["stat"], data, n_boot=n_boot, seed=seed))
    # What a strict label-anchored read alone would have recovered, so the gain
    # from the tolerant tiers (and from any custom parser) is auditable.  A spec
    # that is custom-only (no declarative fields) has no strict baseline.
    strict = ({"parse_rate": None} if (spec.get("custom") and not spec["fields"])
              else parse_rows(rows, dict(spec, custom=None), max_tier=1))

    flags = []
    if parsed["parse_rate"] is not None and parsed["parse_rate"] < 0.8:
        flags.append("low_parse")
    if parsed["n_refused"]:
        flags.append(f"refusals={parsed['n_refused']}")
    if parsed["n_refusal_text"]:
        flags.append(f"prose_refusals={parsed['n_refusal_text']}")
    if spec["stat"] in ("between", "phi") and (
            len(data["pos"]) < 2 or len(data["neg"]) < 2):
        flags.append("no_contrast")
    if spec.get("custom"):
        flags.append("custom_parser")
    if n_used < parsed["n_ok"]:
        # 3-condition designs contribute only the protocol's primary contrast.
        flags.append(f"n_offcontrast={parsed['n_ok'] - n_used}")
    zero_var = 0
    if spec["stat"] in ("between", "phi"):
        for side in ("pos", "neg"):
            a = np.asarray(data[side], float)
            if len(a) >= 2 and a.std(ddof=1) <= _SD_TOL:
                zero_var += 1
    elif spec["stat"] in ("paired_zero", "prop_vs_half"):
        a = np.asarray(data["d"], float)
        if len(a) >= 2 and a.std(ddof=1) <= _SD_TOL:
            zero_var = 1
    elif spec["stat"] == "pearson":
        for side in ("x", "y"):
            a = np.asarray(data[side], float)
            if len(a) >= 2 and a.std(ddof=1) <= _SD_TOL:
                zero_var += 1
    if zero_var:
        flags.append("zero_variance")

    return {
        "effect": effect, "variant": variant,
        "r_sim": r, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "n_ok": parsed["n_ok"], "parse_rate": parsed["parse_rate"],
        "parse_rate_strict": strict["parse_rate"],
        "n_used": n_used,
        "n_denom": parsed["n_denom"], "n_refused": parsed["n_refused"],
        "n_parse_fail": parsed["n_parse_fail"],
        "n_refusal_text": parsed["n_refusal_text"],
        "parse_tier_max": (max(parsed["tiers"]) if parsed["tiers"] else None),
        "stat": spec["stat"], "zero_var": zero_var,
        "flag": ";".join(flags),
    }


def score_bare(rows, effect, n_boot=2000, seed=0, max_tier=3):
    """All bare-task rows for one effect -> a row for XM_bare_old.csv.

    Per the protocol, seed is the repetition unit but the whole battery pools to
    a single effect-level r_bare (GPT contributes one model).
    """
    spec = get_spec(effect, "bare")
    if spec is None:
        return None
    parsed = parse_rows(rows, spec, max_tier=max_tier)
    data, n_used = build_data(parsed, spec)
    r = effect_size(spec["stat"], data) if n_used else None
    ci_lo, ci_hi = ((None, None) if r is None
                    else bootstrap_ci(spec["stat"], data, n_boot=n_boot, seed=seed))
    flags = []
    if parsed["parse_rate"] is not None and parsed["parse_rate"] < 0.8:
        flags.append("low_parse")
    if parsed["n_refused"]:
        flags.append(f"refusals={parsed['n_refused']}")
    if parsed["n_refusal_text"]:
        flags.append(f"prose_refusals={parsed['n_refusal_text']}")
    if spec["stat"] in ("between", "phi") and (
            len(data["pos"]) < 2 or len(data["neg"]) < 2):
        flags.append("no_contrast")
    return {
        "effect": effect, "r_bare": r, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "n_ok": parsed["n_ok"], "parse_rate": parsed["parse_rate"],
        "n_denom": parsed["n_denom"], "n_refused": parsed["n_refused"],
        "n_parse_fail": parsed["n_parse_fail"],
        "n_refusal_text": parsed["n_refusal_text"],
        "n_seeds": len({r_.get("seed") for r_ in rows}),
        "stat": spec["stat"], "flag": ";".join(flags),
    }


EFFECTS_OLD19 = sorted({e for e, _ in SPECS})


# ==========================================================================
# 7. self-test
# ==========================================================================
if __name__ == "__main__":
    import json
    import sys

    bank_path, main_path, bare_path = sys.argv[1], sys.argv[2], sys.argv[3]
    protocols = json.load(open(bank_path))["protocols"]
    for k, v in bind_keys(protocols).items():
        print(f"{k}: {v}")

    main_rows = [r for r in json.load(open(main_path)) if r["effect"] in set(EFFECTS_OLD19)]
    bare_rows = [r for r in json.load(open(bare_path)) if r["effect"] in set(EFFECTS_OLD19)]

    print("\n--- cells ---")
    for effect in EFFECTS_OLD19:
        for variant in ("A", "B"):
            rows = [r for r in main_rows
                    if r["effect"] == effect and r["variant"] == variant]
            row = score_cell(rows, effect, variant, n_boot=200)
            print(f"{effect:40s} {variant}  r={row['r_sim']}  "
                  f"parse={row['parse_rate']}  {row['flag']}")

    print("\n--- bare ---")
    for effect in EFFECTS_OLD19:
        rows = [r for r in bare_rows if r["effect"] == effect]
        if not rows:
            print(f"{effect:40s}  (no bare rows -- bare_eligible=False)")
            continue
        row = score_bare(rows, effect, n_boot=200)
        print(f"{effect:40s}  r_bare={row['r_bare']}  "
              f"parse={row['parse_rate']}  {row['flag']}")
