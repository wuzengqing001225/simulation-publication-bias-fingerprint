# XM Old 19 Effects Parsing Notes (GPT gpt-5.6-terra)

Parser/Estimator: `xm_parsers_patch_old.py` (can run its own self-check independently).
Data: xm_main_resp.json (19×2 cells ×30 = 1,140 rows), xm_bare_resp.json (17 effects ×30 = 510 rows).

## 1. Parse Rate

All 38 cells in the main simulation are 30/30, **median parse rate 1.000, no low-parse cells** (<0.8).
Refusals: all 1,140 rows have non-empty text, `n_refused = 0`.
Only 1 cell required the tolerance layer: Tversky variant A (strict line-anchoring 0/30 → with tolerance 30/30, tier 2).

Of the 17 bare-task effects, 16 are 30/30; Bargh2012_warmth is 0/30 (see §4).

## 2. Necessary Patches Relative to the Existing Parsers

The existing `a0_scoring.PARSERS` / `a1_scoring.PARSERS1` on the GPT data:
18 of 19 cells for variant A are usable, but **only 7 of 19 cells for variant B are usable** — because they never included
variant B's label vocabulary (RATING / DONATE / INTENTIONAL / ATTITUDE / ANSWER /
V1..V3 / Q1..Q6 / S1_I1.. / ANGLE / Trial N: SELF / K_position ...).
Two other spots are genuine bugs, unrelated to the model:

- `a1_scoring.parse_genschow` requires ≥20 hits of the pattern `S\d+\.[a-z]`, but variant A only has
  16 story items (F1-F11 don't match this pattern), so both A and B cells fail (0/30).
- `a0_scoring.parse_cooney`'s fallback is "the first 1-9 digit in the full text," which mistakes
  a number in the reasoning preamble for the rating; this patch does not use such a fallback — if the label can't be found, it's logged as a parse failure.

The patch changes parsing to a declarative field schema + three-tier label lookup (strict full-line → anywhere in the text →
positional order if unlabeled), so extra whitespace, lowercase labels, and reasoning text before the answer block don't affect parsing,
but any missing required field still counts as a failure — no backfilling is done.

## 3. Scoring Key Re-derivation (all extracted from the protocol library regex, none hardcoded)

- **Klink post-vowel key**: determined by the **initial vowel cluster** of the candidate names in each variant's own prompt
  (u/o/a = back, i/e = front; a, per Klink's /ɑ/ usage, counts as back).
  Note that the `ay` in `vaylo` / `zaymo` is the front diphthong /eɪ/; looking only at the first letter would misclassify it as back,
  so double-letter clusters are matched before single vowels. This rule is **fully consistent (12/12)** with variant B's key
  independently specified in the protocol library's measure field, which is what justifies using it for variant A.
- **Tamir payment table**: pairwise extraction of 'listed at N cents', both variants yield
  offsets = [0,2,4,1,3,0,3,1,4,2], consistent with the measure text.
- **Wakslak BIF abstraction-item key**: for variant B, the measure explicitly specifies all 25 items as A=concrete/B=abstract.
  Variant A's key is inferred by matching behavioral text across variants; **23/25 are determined by matching**; the remaining 2 items
  (3 'Joining the Army', 25 'Pushing a doorbell') have no counterpart item in variant B,
  so a rule (an option stating purpose/outcome/significance is the abstract item) is used instead, and these are flagged as
  `purpose_rule` in `WAKSLAK_A_KEY_ROUTE` for review.

## 4. Cells for Which No r Could Be Produced (not guessed, recorded honestly)

- **Bargh2012_warmth main simulation, cells A and B**: parsed 30/30, but the statistic is
  Pearson r (warmth index × loneliness), and GPT gave **identical
  warmth item triples** across all 30 participants (cell A: S1.5/6/7 constantly 4/4/3; cell B: B1/B2/B3 constantly 5/4/3),
  so x-side variance is 0, the correlation is undefined → `r_sim` left blank, `zero_var=1`.
  This is zero-variance output from GPT, not a parsing issue.
- **Bargh2012_warmth bare task**: all 30 rows are unscoreable — 8 rows filled all 17 items with 'N/A',
  18 rows filled everything with 0 or 1 (outside the 1-4 / 1-9 scale range), 4 rows explicitly refused in prose
  ("I cannot provide a numerical value," "I don't have personal habits"). GPT refused via prose rather than returning text=null,
  so the patch counts `n_refusal_text` separately, not lumping refusals in with format failures;
  but to be consistent with the denominator convention of the existing P2 table, refusal text is still counted into `n_parse_fail`.
- **Banerjee2012_brightness, Slepian2012_secrets bare tasks**: the bare-task file has no rows at all for these two
  effects, consistent with `bare_eligible=False` in `P2_paradigm_class.csv`
  ("requires recalling a personal moral/secret experience"), flagged as `bare_ineligible_no_rows`.

## 5. Other Interpretive Notes

- **Chao2017 variant B used a custom parser**. The protocol calls for 'DONATE: 1|0' + 'AMOUNT: 0-5',
  but GPT gave all 30 rows in free-form amounts only ('Donation amount: $2.00', '[Donation: $2]',
  'I will donate $1.'). The amount can be recovered, so DONATE is inferred from amount>0
  (the labeling itself encodes this meaning), and the cell is flagged `custom_parser`. But since all 30 rows have positive amounts,
  the binary primary DV has no variance, r=0 and it's flagged `zero_variance` — this 0 is "no contrast," not "no effect."
- **Condition name mapping** (an A1-era lesson, now written into the estimator per each variant's actual cond values):
  Eyal near/distant(B) vs near_future/distant_future(A);
  Chao salient_gift/non_salient_gift(B) vs gift_salient/gift_nonsalient(A);
  Oppenheimer three_sixes_rare/two_sixes_streak(B) vs
  rare_three_sixes/two_dice_two_sixes(A).
- **Three-condition designs use only the protocol-specified primary contrast**; participants in the third condition are excluded from the statistics,
  recorded as `n_offcontrast=10` (Chao's non_salient, Oppenheimer's representative).
- **Valdesolo** implemented as the equivalent simplified form of the planned +1,-1,+1,-1 contrast:
  independent-samples t of (self+ingroup) vs (other+outgroup).
- **VanBoven variant A**'s Q1/Q2 are positionally coded; the reverse lookup of which item is "anticipation" is done based on order_future_first /
  order_past_first, and then anticipation minus retrospection is taken.
- **Slepian**'s individual index z(slant)−z(distance) is standardized within-cell before running the between-group t-test.
- **15 cells total have zero variance** (GPT produced highly repetitive output in most cells); the `zero_var` column indicates
  which side has zero variance; in that case r is determined by the sign of the mean difference (±1 or 0), and the CI degenerates.
  Asch (r=1.0) and Klink A (r=1.0) fall into this category, and these values should not be interpreted as continuous effect sizes.

## 6. Direction Check

The sign of every r has been defined per the original claim direction, and cross-checked cell by cell against printed group means.
Cells with the opposite sign from the original direction (negative r): Eyal A (-0.409, but B is +0.366),
Genschow A/B (-0.566/-0.243), Husnu A (-0.028), Slepian A/B (-0.159/-0.370),
Tversky A/B (-0.067/-0.600); negative values for the bare tasks are shown in XM_bare_old.csv.