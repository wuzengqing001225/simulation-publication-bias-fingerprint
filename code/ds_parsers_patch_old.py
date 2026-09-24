"""DeepSeek (deepseek-v4-flash) delta patch over ``xm_parsers_patch_old.py``.

Scope of this module
--------------------
The GPT-round patch ``xm_parsers_patch_old.py`` was applied unchanged to the
DeepSeek main-simulation and bare-battery responses for the 19 old A0/A1
effects.  It already reaches a **1.000 parse rate on 37/38 main cells and on
all 17 bare effects** (median 1.000, no effect below the 0.80 incremental-patch
threshold), so essentially nothing needed re-deriving: the declarative field
spec, the three-tier label search, the Klink / Tamir / Wakslak key
re-derivations and all condition-name mappings carry over to this model family
as they stand.

Exactly one DeepSeek-specific wording was not covered, and this module is that
single delta:

* **JSON-object answer blocks.**  DeepSeek occasionally answers a labelled
  field spec as a JSON object instead of ``LABEL: value`` lines --
  ``{"V1": -5, "V2": -4, "V3": 0}`` (Eyal2008 variant B, 1 row) and
  ``{"donation_amount": 2}`` (Chao2017 variant B, 1 row; already recovered by
  that spec's custom amount parser).  The upstream label/value separator
  ``_SEP`` accepts a colon, full-width colon, dash, equals or bare space, but
  not the closing double-quote that JSON puts *between* the label and the
  colon, so the Eyal row failed on a purely typographic difference while every
  required field was present and in range.

The fix is deliberately typographic and nothing more: allow an optional
closing quote / bracket character between a field label and its separator.  It
cannot invent a value -- a response still has to carry every required field,
in range, or it remains a parse failure, and ``text`` that is None or blank is
still counted as a refusal rather than a format failure.

The patch is verified by ``verify_delta()``: over all 1,140 main rows and 510
bare rows it must change the parsed result of **exactly** the rows it is meant
to fix and leave every other row's values and tier untouched.  Import this
module instead of the GPT patch to score the DeepSeek wave; the public surface
(``bind_keys``, ``get_spec``, ``score_cell``, ``score_bare``,
``EFFECTS_OLD19``, the estimators) is re-exported unchanged.
"""
import re

import xm_parsers_patch_old as _base

# Re-export the untouched public surface.
from xm_parsers_patch_old import (  # noqa: F401
    EFFECTS_OLD19, KEY_AUDIT, SPECS, WAKSLAK_A_KEY_ROUTE, bind_keys,
    bootstrap_ci, build_data, effect_size, get_spec, klink_back_key,
    klink_key_from_measure, looks_like_refusal, parse_fields, parse_rows,
    r_between, r_paired_zero, r_pearson, r_phi, r_prop_vs_half, score_bare,
    score_cell, tamir_offsets, wakslak_abstract_key,
)

# --------------------------------------------------------------------------
# the one delta: tolerate a JSON-quoted / bracketed label
# --------------------------------------------------------------------------
# Upstream: r"\s*[:\uff1a=\-\u2013\u2014]\s*|\s+"
# Here: an optional closing quote or bracket may sit between the label and the
# separator, so '"V1": -5' and '[V1]: -5' read the same as 'V1: -5'.  The
# alternation is kept in the same order and the bare-whitespace branch is
# preserved, so any string the upstream separator matched still matches.
_SEP_DS = r"[\"'\]\)]?\s*[:\uff1a=\-\u2013\u2014]\s*|[\"'\]\)]?\s+"

_ORIG_SEP = _base._SEP
_base._SEP = _SEP_DS


def restore_base_separator():
    """Undo the monkey-patch (used by the self-test to A/B the two behaviours)."""
    _base._SEP = _ORIG_SEP


def apply_delta():
    _base._SEP = _SEP_DS


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------

def verify_delta(main_rows, bare_rows):
    """A/B the delta over every row; return the rows whose parse changed.

    Each entry is (effect, variant, cond, before, after) where ``before`` and
    ``after`` are ``(values, tier)`` under the base and patched separators.
    A clean delta changes only rows that previously failed (before[0] is None)
    and never alters an already-parsed row's values.
    """
    changed, altered = [], []
    todo = [(r, r.get("variant", "bare")) for r in main_rows] + \
           [(r, "bare") for r in bare_rows]
    for r, variant in todo:
        spec = get_spec(r["effect"], variant)
        if spec is None or not spec.get("fields"):
            continue
        text = r.get("text")
        if text is None or not str(text).strip():
            continue
        restore_base_separator()
        before = parse_fields(text, spec)
        apply_delta()
        after = parse_fields(text, spec)
        if before != after:
            changed.append((r["effect"], variant, r.get("cond"), before, after))
            if before[0] is not None:
                altered.append((r["effect"], variant, r.get("cond")))
    apply_delta()
    return changed, altered


if __name__ == "__main__":
    import json
    import sys

    bank_path, main_path, bare_path = sys.argv[1], sys.argv[2], sys.argv[3]
    bind_keys(json.load(open(bank_path))["protocols"])
    keep = set(EFFECTS_OLD19)
    main_rows = [r for r in json.load(open(main_path)) if r["effect"] in keep]
    bare_rows = [r for r in json.load(open(bare_path)) if r["effect"] in keep]

    changed, altered = verify_delta(main_rows, bare_rows)
    print(f"rows whose parse changed: {len(changed)}")
    for c in changed:
        print("  ", c[0], c[1], c[2], "->", c[4][0], f"tier={c[4][1]}")
    print(f"already-parsed rows altered (must be 0): {len(altered)}")
    assert not altered, altered
