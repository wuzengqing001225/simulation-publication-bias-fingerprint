# DS Old 19-Effects Parsing Notes (DeepSeek deepseek-v4-flash)

Parser: `ds_parsers_patch_old.py` = GPT-round `xm_parsers_patch_old.py` + one typography-level increment (independently self-checkable).
Data: xm_main_resp.json contains 19×2 cells×30 = 1,140 rows; xm_bare_resp.json contains 17 effects×30 = 510 rows.

## 1. Parse rate: GPT patch works nearly out of the box

37 of 38 cells in the main simulation are 30/30 right out of the box; **median parse rate 1.000, no low-parse effects** (<0.8).
The only unmatched row is the JSON answer block for the Eyal2008 variant B (see §2).
All 17 bare-task effects are 30/30.

The GPT patch's declarative field spectrum, three-tier label retrieval, Klink/Tamir/Wakslak scoring-key redirection,
and the variant condition-name mappings (Eyal near/distant, Chao salient_gift, Oppenheimer
three_sixes_rare) required **no redirection at all** on DeepSeek — condition-name coverage was checked cell by cell:
only Chao (gift_nonsalient / non_salient_gift) and Oppenheimer
(common_two_sixes_and_a_three / representative_mix) have a third condition excluded from the main comparison,
consistent with the plan, recorded as `n_offcontrast=10`.

## 2. The single additional patch: JSON object answer blocks

DeepSeek occasionally answers a labeled field spectrum as a JSON object:
`{"V1": -5, "V2": -4, "V3": 0}` (Eyal2008 variant B, 1 row),
`{"donation_amount": 2}` (Chao2017 variant B, 1 row — already caught by that cell's custom amount parser).
The upstream `_SEP` accepts colon/full-width colon/dash/equals/bare space, but does not accept a closing
quote mark sandwiched between the label and the colon inside JSON, so that row failed purely due to a
typography difference despite **all required fields being present and in-domain**.

The patch does exactly one thing: it allows a closing quote mark or closing parenthesis to appear between
the label and the separator. `verify_delta()` performs an A/B comparison across all 1,650 rows: **exactly
1 row's parse result changed (Eyal B near → tier 2), 0 rows of existing parses were altered**. No value is
fabricated — missing fields still count as failures.

## 3. Refusals (empty text, listed honestly and separately)

13 rows, all with `error=None` and text null: 1 row in the main simulation (Genschow A control);
12 rows in the bare tasks (Valdesolo 3, Cooney 3, VanBoven 3, Tversky 2, Genschow 1).
`n_refusal_text` (prose refusal) is **entirely 0** on DeepSeek — unlike the GPT round,
DeepSeek does not break character with a prose refusal; it simply returns empty.

## 4. Cells for which r could not be produced

- **Bargh2012_warmth main simulation A**: parsed 30/30, but the statistic is a pearson r
  (warmth index × loneliness), and across the 30 participants the three warmth items are
  constantly S1.5/6/7 = 4/4/3, giving zero variance on the x side, so the correlation is undefined →
  `r_sim` left blank, `zero_var=1`. This is a model zero-variance output, not a parsing issue.
  Variant B has variance, r=+0.177. The bare task for the same effect has variance, r_bare=+0.224.
  (For comparison: in the GPT round, both cells A and B and the bare task were all zero-variance/unscoreable.)
- **Banerjee2012_brightness, Slepian2012_secrets bare tasks**: the bare file simply has no rows for these
  two effects, consistent with `bare_eligible=False` in `P2_paradigm_class.csv`, marked `bare_ineligible_no_rows`.

## 5. Interpretation notes

- **10 zero-variance cells in total** (18 in the GPT round); the `zero_var` column indicates which side
  has zero variance. In these cases r is determined by the sign of the mean difference (±1 or 0), the CI
  degenerates, and it should not be interpreted as a continuous effect size:
  Asch A/B (r=1.0) and Knobe A/B (r=1.0) fall under "both groups each constant and directionally
  consistent" saturation; Chao A/B and Rottenstreich A/B's r=0 reflect **no contrast** (both groups have
  the same value), not "no effect" — Chao's 30 rows of donation amounts are all positive, and
  Rottenstreich's 30 rows all chose B (money), so the binary main DV has no variance.
- **Chao2017 variant B still goes through the custom parser**: all 30 rows are free-format amounts
  ('Donation: $2', 'I would donate $2.00', 'DONATION: $1.00'), none using the
  'DONATE: 1|0' label, so DONATE is inferred from amount>0, the cell is marked `custom_parser`.
- **Cells carried by the leniency tier** (strict line-anchored baseline vs. final): Tversky A 0.000→1.000
  (tier 2), Chao A 0.300→1.000, Knobe A 0.600→1.000, Eyal B 0.467→0.967→1.000,
  Eyal A 0.833→1.000. These tier 2/3 readings have been checked line by line against the original text:
  DeepSeek often gives bare-value blocks ('A\n5', 'B\n4') or combined lines like '1: 3, 2.0', and the
  readings are correct.
- **Bare-task Tversky r_bare = -1.000**: all 28 rows answered '3' for all five letters (the third position
  being more frequent), i.e., not a single respondent met the availability-error criterion of "answering
  the first position more often," p=0 → r=2p-1=-1. The direction is clearly reversed.
- **Tamir bare task r_bare = 0**: all 30 rows chose OTHER, individual scores constantly 0 (zero variance).

## 6. Direction check

All r signs are defined according to the direction of the original claim. Cells with negative direction
in the main simulation: Banerjee B (-0.134), Eyal B (-0.087), Genschow A (-0.271),
Wakslak A (-0.170), Wakslak B (-0.014). The only negative value in the bare tasks is Tversky (-1.000).