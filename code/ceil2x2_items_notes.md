# ceil2x2_items.py — construct notes

14 new constructs, 50 items. Companion to `b2_items.py` (format authority); the other
9 constructs of the ceil2x2 pair design come from the B2 bank.

All batteries are **parallel forms**: facet coverage, response format and keying balance
follow the benchmark instrument used for the same-named Twin-2K-500 panel construct, with
wording composed here rather than copied.

| construct | benchmark instrument (style target) | items | fmt | scoring direction |
|---|---|---|---|---|
| P_RFS | Altemeyer & Hunsberger Revised Religious Fundamentalism, 12-item short form (balanced 6/6) | 12 | likert7 | high = more fundamentalist; 6 items reverse-keyed |
| P_agreeableness | BFI-2 agreeableness domain (compassion / respectfulness / trust) | 6 | likert7 | high = more agreeable; 3 reverse |
| P_neuroticism | BFI-2 negative-emotionality domain (anxiety / depression / volatility) | 6 | likert7 | high = more neurotic; 2 reverse |
| P_maximization | Schwartz et al. Maximization Scale (alternative search / decision difficulty / high standards) | 6 | likert7 | high = more maximizing; 1 reverse |
| P_fluid | Cattell/Raven-style numeric induction (number series; no figural items — text-only channel) | 6 | num (key, tol=0.001) | high = more correct |
| P_finliteracy | Lusardi & Mitchell "Big Three": compounding, real interest/inflation, diversification | 3 | mc (key) | high = more correct |
| P_wason | Wason selection task, text-rendered: one abstract letter/number, one thematic (book price) | 2 | mc (key) | high = more correct (falsifying pair chosen) |
| P_dictator_sender | one-shot anonymous dictator game, $10 endowment (Forsythe/Hoffman paradigm) | 1 | num (no key, raw) | high = gave more (0–10 dollars) |
| W_base_rate_30eng | Kahneman & Tversky (1973) lawyer–engineer sketch, 30 engineers / 70 lawyers | 1 | pct (no key) | raw judged % engineer; normative = 30, so high = stronger base-rate neglect |
| W_less_is_more_A | Hsee (1998) evaluability / less-is-more, dinnerware Set A judged in isolation (24 pieces, all intact) | 1 | num (no key, raw) | high = larger willingness to pay for Set A |
| P_mentalaccounting | Thaler topical-account scenarios: lost theatre ticket, earmarked $500 windfall, prepaid ski pass (sunk cost) | 3 | likert7 | high = stronger mental accounting; ticket item reverse-keyed (refusing to rebuy = bias) |
| P_age_bracket | standard panel demographic, 6 brackets (18–24 … 65+) | 1 | cat | score = bracket ordinal, monotone in age |
| P_education | standard panel demographic, 7 levels (< high school … doctoral/professional) | 1 | cat | score = attainment ordinal |
| P_income | standard panel demographic, 7 brackets (< $25k … $200k+) | 1 | cat | score = income-bracket ordinal |

## Format additions over b2_items.py

`cat` is the one new `fmt`: a self-reported category answered as the bracket number listed
in `anchors`, scored as that ordinal. It carries no `key` — demographics are not ability
items. `likert7`, `pct`, `num` and `mc` behave exactly as in `b2_items.py`.

Two `num` items (`P_dictator_sender`, `W_less_is_more_A`) intentionally carry `key=None`:
the response *is* the measure (dollars given, dollars offered), not a correctness check.
The 11 keyed items are the 6 fluid series (`tol=0.001`), the 3 financial-literacy items and
the 2 Wason items.

## Note on the ceil2x2 pairs

`W_base_rate_30eng` is keyed so that *higher = more neglect*; the design table's predicted
`W_policy_support_liberal × W_base_rate_30eng` correlation (−0.1146) is stated on the panel's
own coding of that variable and should be sign-checked against it before scoring.
