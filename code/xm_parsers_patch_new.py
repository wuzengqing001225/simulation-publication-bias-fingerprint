"""
xm_parsers_patch_new.py -- GPT (gpt-5.6-terra) wording patches for the 42
Phase-2 "new" effects.

The three parser groups (p2_parsers_g0/g1/g2.py) were written against Claude
responses.  Re-running them unchanged on the GPT arm reproduces every one of
the 336 Claude reference cells in P2_cells_g0/g1/g2.csv exactly (see
VALIDATION below), so the estimators are untouched here.  Only two GPT-specific
FORMAT deviations needed new code, both in group 0's generic label machinery.
Group 1 and group 2 parsers required no patch.

--------------------------------------------------------------------------
PATCH 1 -- MIXED_LABEL pass for 'repeat'-label specs  (tier 2.5)
--------------------------------------------------------------------------
Symptom (Correll2007_motherhood_penalty variant B, 7/30 GPT responses):

    RATING: 6
    RATING: 6
    RATING: 6
    RATING: 5
    RATING: 6
    RATING: 6
    58                <- label dropped on the last line only

The protocol asks for 7 'RATING: <int>' lines whose 7th line is an age on a
20-80 scale.  GPT emits the label on lines 1-6 and a bare number on line 7.
The stock tiers cannot recover this: tier 1/2 need n labelled hits, and tier 3
(label-free positional) is only reached when a cell's parse rate falls below
0.60 -- this cell sits at 0.767, so the seven responses would be silently
dropped rather than scored.

Fix: a MIXED pass that reads one value per non-blank line, accepting either a
labelled 'PAT: v' line or a bare in-range value line, requiring exactly n lines
and n values.  It is inserted between tier 2 and tier 3, so a response is only
read this way after the strict and loose labelled passes have both failed.
It never invents a value and never reorders: line k supplies item k, which is
what makes the per-item ranges (lo/hi as lists) still enforceable.

--------------------------------------------------------------------------
PATCH 2 -- MIXED pass for 'numbered'-label specs
--------------------------------------------------------------------------
Symptom (Aryani2020_word_arousal variant B, 30/30 GPT responses):

    4
    2
    5
    2
    ...              <- 12 bare integers, the 'N:' ordinal prefix dropped

Same MIXED pass handles it: 12 non-blank lines, each a bare in-range value, so
line k is item k.  This cell would in fact also have been rescued by the stock
tier-3 escalation (its labelled parse rate is 0.00, below the 0.60 trigger),
and MIXED returns the identical values there; the patch simply makes the
recovery explicit and range-checked per item instead of scavenging every number
in the text.

--------------------------------------------------------------------------
NOT patched (genuine failures, left in the parse-rate denominator)
--------------------------------------------------------------------------
* FathKay2018_hierarchy_corruption B, 1 response: the body is the single line
  'RATING: 2' -- six of seven required values are absent.  Truncation, not a
  wording difference; unrecoverable and correctly counted as a parse failure.
* 2 responses across the GPT main arm have text=None (true refusals) and are
  excluded from the parse-rate denominator, per the group conventions.
* Bare arm: 11 responses are explicit first-person refusals with prose bodies
  ("I can't honestly provide numeric ratings because I do not have a life,
  moods, or feelings.").  These carry no scoreable values; they are format
  failures inside the denominator, not label-wording problems.

--------------------------------------------------------------------------
VALIDATION (regression against the Claude arm)
--------------------------------------------------------------------------
UNPATCHED, this code reproduces the Claude reference tables as follows
(r_sim / r_bare agreement to <5e-4 with identical n_ok):

  main cells   P2_cells_g0.csv  112/112     P2_cells_g1.csv 112/112
               P2_cells_g2.csv  111/112
  bare arm     P2_bare_g0.csv    20/20      P2_bare_g1.csv   14/14
               P2_bare_g2.csv    18/18  (both r_bare and r_bare_seedagg)

The single main-cell mismatch is a cell the reference table itself marks
degenerate=1: Peters2006_numeracy_framing A / sonnet45 / demographic.
Objective numeracy is constant there, so the fitted frame x numeracy
coefficient is floating-point residue (beta_int ~ 2e-15) and its t ratio is
numerically arbitrary; the reference records -0.0937, this code -0.2512.
Neither number is meaningful and the cell is excluded downstream by its
degenerate flag.  All other Peters cells match exactly.

Two conventions were pinned by that regression and are load-bearing:
  * group 0's per-cell driver defaults to the LOOSE labelled pass (tier=2),
    not the strict pass, before considering escalation to tier 3.
    P2_cells_g0.csv records parse_tier=2 for two cells (both
    CarterGilovich2012_experiential_self B / sonnet45) and parse_tier=3 for
    two more (Aryani2020_word_arousal B / sonnet45), so a labelled pass looser
    than strict is part of the reference ladder; defaulting to tier 1 loses one
    CarterGilovich2012 B response and shifts that cell's r_sim by 0.006.
  * for group-0/group-1 BETWEEN designs the bare arm's r_bare is computed on
    seed-level condition means, and a seed present in only ONE condition is
    dropped rather than contributing an unpaired mean.  This is what
    reconciles FathKay2018 / sonnet45 (0.8853 with the drop, 0.8946 without).
    Group 2's published r_bare column is instead individual-level, with the
    seed-aggregated value carried separately as r_bare_seedagg; each group's
    own column semantics are preserved so the GPT numbers are column-for-column
    comparable with the Claude tables.

CAVEAT -- the patch is not Claude-neutral.  Re-running the Claude arm WITH the
patch changes 3 of the 112 group-0 main cells, because the MIXED pass also
recovers Claude responses that the stock ladder dropped:
    Aryani2020_word_arousal        B / sonnet5 / demographic
        n_ok 38 -> 40,  r_sim 0.9992  -> 0.9978  (dr 0.001)
    Correll2007_motherhood_penalty B / sonnet5 / demographic
        n_ok 59 -> 60,  r_sim -0.1744 -> -0.1759 (dr 0.001)
    Correll2007_motherhood_penalty B / sonnet5 / none
        n_ok 49 -> 60,  r_sim -0.2981 -> -0.2046 (dr 0.094)
The first two are immaterial.  The third is not: the patch recovers 11 further
Claude responses and moves r_sim by 0.09.  The published Claude table was built
WITHOUT the patch, so for these two effects the GPT and Claude cells are not
parsed under an identical ladder.  Any GPT-vs-Claude contrast on
Correll2007_motherhood_penalty B (and, marginally, Aryani2020_word_arousal B)
should re-run the Claude arm through this patch first; the other 40 effects are
unaffected.
"""

import re

import numpy as np


# --------------------------------------------------------------------------
# PATCH 1 + 2: mixed labelled/bare per-line pass for group 0 specs
# --------------------------------------------------------------------------

def _mixed_line_values(text, spec, g0):
    """One value per non-blank line, label optional.  Returns {1..n: v} or None.

    `g0` is the executed p2_parsers_g0 namespace (supplies _VALUE_RE, _cast,
    _ok).  Accepts, per line:
        '<PAT>: v'  /  '<PAT> N: v'  /  '<NAME>: v'   (labelled), or
        'v'                                            (bare value only)
    Requires exactly spec['n'] non-blank lines and every value in its own
    item's declared range, so nothing is invented or reordered.
    """
    if text is None:
        return None
    n = spec.get("n", len(spec.get("names", [])))
    vre = g0["_VALUE_RE"][spec["value"]]
    lines = [ln.strip() for ln in str(text).strip().split("\n") if ln.strip()]
    if len(lines) != n:
        return None

    # a labelled line: anything non-numeric up front, then ':' or '=' , then v
    lab = re.compile(rf"^\s*[A-Za-z][A-Za-z0-9 _\-]*\s*[:=\uff1a]\s*({vre})\s*$")
    bare = re.compile(rf"^\s*({vre})\s*$")

    vals = {}
    for i, ln in enumerate(lines, start=1):
        m = lab.match(ln) or bare.match(ln)
        if not m:
            return None
        try:
            v = g0["_cast"](spec, m.group(1))
        except ValueError:
            return None
        if not g0["_ok"](spec, i, v):
            return None
        vals[i] = v
    return vals if len(vals) == n else None


def parse_response_patched(text, spec, g0, tier=2):
    """Group-0 parse with the MIXED pass inserted between tiers 2 and 3.

    tier<=2 -> strict, loose, mixed
    tier>=3 -> strict, loose, mixed, label-free positional
    Returns (values, tier_label) with tier_label in {1, 2, 2.5, 3}.
    """
    if text is None:
        return None, None
    n = spec.get("n", len(spec.get("names", [])))
    for t in (1, 2):
        if t > tier:
            break
        v = g0["_parse_tier"](text, spec, n, t)
        if v is not None:
            return v, t
    v = _mixed_line_values(text, spec, g0)
    if v is not None:
        return v, 2.5
    if tier >= 3:
        v = g0["_parse_tier"](text, spec, n, 3)
        if v is not None:
            return v, 3
    return None, None


def install(g0):
    """Monkey-patch the executed p2_parsers_g0 namespace in place.

    After this call g0['parse_response'] and therefore g0['collect'] use the
    patched tier ladder.  Idempotent.
    """
    if g0.get("_XM_PATCHED"):
        return g0
    g0["_xm_orig_parse_response"] = g0["parse_response"]
    g0["parse_response"] = lambda text, spec, tier=2: parse_response_patched(
        text, spec, g0, tier=tier)
    g0["_XM_PATCHED"] = True
    return g0
