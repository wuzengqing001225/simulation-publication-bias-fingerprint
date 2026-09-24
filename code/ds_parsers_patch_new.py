"""
ds_parsers_patch_new.py -- DeepSeek (deepseek-v4-flash) wording patch for the
42 Phase-2 "new" effects.

Baseline: the three stock parser groups (p2_parsers_g0/g1/g2.py) plus the
GPT-round patch xm_parsers_patch_new.py (group-0 MIXED tier-2.5 pass), applied
unchanged.  On the DeepSeek main arm that baseline already reaches 30/30 on
82 of the 84 effect x variant cells, and 30/30 on all 26 bare-arm effects, so
no estimator and no group-0/group-2 parser is touched here.

Exactly ONE DeepSeek-specific deviation needed new code.

--------------------------------------------------------------------------
PATCH -- group 1 `p_giessner`: ordinal-agnostic POWER item recovery
--------------------------------------------------------------------------
Symptom (Giessner2007_vertical_power variant A, 2/30 DeepSeek responses):

    POWER7: 7          POWER: 7
    POWER8: 8    and   POWER: 8
    POWER9: 7          POWER: 7

The protocol asks for three lines 'POWER1:'..'POWER3:'.  DeepSeek emits three
well-formed labelled lines but numbers them 7-9 in one response (continuing the
item numbering of the preceding stimulus block) and drops the ordinal entirely
in another.  The stock parser reads the ordinal into a dict and then indexes
got[1], got[2], got[3], so the first response raises KeyError: 1 -- an
uncaught exception inside the per-cell driver, not a scored parse failure --
and the second yields an empty dict (all indices None are skipped) and fails
strict; its lenient fallback does fire, but only via the label-free
`_bare_ints` scavenge.

Fix: `p_giessner_patched` collects the numeric values of POWER-labelled lines
in DOCUMENT ORDER and accepts the response iff there are exactly three of them,
each in 1-9, and their ordinals are either all absent or a strictly increasing
run of three (any offset).  Mixed present/absent ordinals, duplicated ordinals,
a non-consecutive set such as {1,2,5}, and any count other than three are all
rejected as before.

This is order- and offset-agnostic on purpose and is safe here specifically
because the protocol makes it so: both variants state the participant-level DV
is the arithmetic mean of POWER1-POWER3, that all three items are primary,
score identically on the same 1-9 scale, and that none is reverse-scored.  The
mean is therefore invariant to which item supplied which value, so no item
identity is being guessed.  The patch is NOT generalised to other repeated-label
effects in group 1 (Finucane RISK/BEN, Kozak, KimMarkus, ...) where items enter
the DV asymmetrically and ordinals are load-bearing.

--------------------------------------------------------------------------
NOT patched (left in the parse-rate denominator)
--------------------------------------------------------------------------
* Mellers2001_conjunction_frequency A: strict parse rate 0.567.  The stock
  group-2 ladder escalates to the lenient pass on its own (trigger < 0.60) and
  recovers 30/30, so the cell is scored at parse_rate 1.000 with
  parse_mode='lenient'.  No new code.
* Main arm: no refusals inside the 42 new effects.  All 2,520 rows carry a
  non-empty body (the single text=None row in xm_main_resp.json belongs to
  Genschow2017_free_will_attribution, an old-19 effect).  Every one of the
  84 cells therefore has n_denom = 30 and n_refusal = 0.
* Bare arm: 6 of the 780 rows across the 26 new effects with bare data have
  an empty-string body -- Bauer2012_consumer_cues (1),
  DogerliogluDemir2014_incidental_anchor (2),
  Huang2019_anthropomorphism_holistic (1),
  Park2019_scarcity_price_quality (2).  These are counted in n_refusal and
  excluded from the parse-rate denominator, per the group conventions; all
  four effects parse 100% of their remaining rows.  No response in either arm
  is a prose refusal, so unlike the GPT bare arm there are no format failures
  sitting inside the denominator.

--------------------------------------------------------------------------
VALIDATION (regression)
--------------------------------------------------------------------------
Applied to the GPT arm (xm_main_resp.json / xm_bare_resp.json), this patch on
top of xm_parsers_patch_new.py reproduces XM_cells_new.csv and XM_bare_new.csv
exactly -- no GPT response numbers its POWER items off-baseline, so the patch is
inert there.  Applied to the Claude arm it is likewise inert on
P2_cells_g1.csv / P2_bare_g1.csv.  Unlike the GPT MIXED pass, this patch is
therefore arm-neutral: the DeepSeek, GPT and Claude Giessner cells are parsed
under an identical ladder.  See the regression block at the bottom of this file.
"""

import re


def p_giessner_patched(t, variant, lenient, g1):
    """POWER1..3 on 1-9, DV = mean.  Ordinal-agnostic; never invents a value.

    `g1` is the executed p2_parsers_g1 namespace (supplies _find_labeled,
    _bare_ints, _rng, _mean).
    """
    hits = g1["_find_labeled"](t, "POWER", lenient)
    idx = [h[0] for h in hits]
    vals = [g1["_rng"](float(h[1]), 1, 9) for h in hits]

    ok = len(vals) == 3 and all(v is not None for v in vals)
    if ok:
        if all(i is None for i in idx):
            pass                                  # unnumbered: doc order
        elif all(i is not None for i in idx):
            ii = [int(i) for i in idx]
            ok = ii[1] == ii[0] + 1 and ii[2] == ii[1] + 1   # consecutive run
        else:
            ok = False                            # mixed numbered/unnumbered
    if ok:
        return {"power": g1["_mean"](vals)}

    if lenient:
        ints = [x for x in g1["_bare_ints"](t) if 1 <= x <= 9]
        if len(ints) == 3:
            return {"power": g1["_mean"](ints)}
    return None


def install(g1):
    """Monkey-patch the executed p2_parsers_g1 namespace in place.  Idempotent.

    Rebinds both the module-level function and its PARSERS entry, so
    g1['parse'] uses the patched parser.
    """
    if g1.get("_DS_PATCHED"):
        return g1
    g1["_ds_orig_p_giessner"] = g1["p_giessner"]
    patched = lambda t, variant, lenient: p_giessner_patched(t, variant, lenient, g1)
    g1["p_giessner"] = patched
    g1["PARSERS"]["Giessner2007_vertical_power"] = patched
    g1["_DS_PATCHED"] = True
    return g1


# --------------------------------------------------------------------------
# REGRESSION -- run this file directly to reproduce the validation claims.
# --------------------------------------------------------------------------
# Requires, in the unpacked publication_bias_fingerprint_package:
#   code/p2_parsers_g1.py
#   raw_responses/p2_main_responses.json      (Claude arm)
#   xmodel_results/xm_main_resp.json          (GPT arm)
#
# For every Giessner2007_vertical_power response in the Claude and GPT arms,
# under BOTH the strict and lenient normalisations, the patched parser must
# return exactly what the stock parser returns.  Any nonzero count below means
# the patch is no longer arm-neutral and the affected cells must be re-parsed
# under one common ladder before cross-model contrasts.
#
# Observed: 0 / 474 Claude responses and 0 / 60 GPT responses change.
if __name__ == "__main__":
    import json, os, sys

    PKG = sys.argv[1] if len(sys.argv) > 1 else "publication_bias_fingerprint_package"
    g1 = {"__name__": "p2g1"}
    exec(open(os.path.join(PKG, "code", "p2_parsers_g1.py")).read(), g1)
    stock = g1["p_giessner"]

    for label, path in [("claude", "raw_responses/p2_main_responses.json"),
                        ("gpt",    "xmodel_results/xm_main_resp.json")]:
        rows = [r for r in json.load(open(os.path.join(PKG, path)))
                if r.get("effect") == "Giessner2007_vertical_power" and r.get("text")]
        changed = 0
        for r in rows:
            for lenient in (False, True):
                t = g1["_norm"](str(r["text"]), lenient)
                a = p_giessner_patched(t, r.get("variant"), lenient, g1)
                b = stock(t, r.get("variant"), lenient)
                if (a is None) != (b is None) or (a and b and abs(a["power"] - b["power"]) > 1e-12):
                    changed += 1
        print(f"{label:7s} Giessner responses whose parse changes: {changed} / {len(rows)}")
