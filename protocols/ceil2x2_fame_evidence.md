# Literature status of the 16 construct pairs in the 2x2 supplementary-angle design

## Verification method

All 16 pairs were checked on **2026-08-31** by web search against the open scholarly record (publisher pages, PubMed/PMC, Psychological Bulletin / Journal of Economic Literature / Judgment and Decision Making article pages, and author publication lists). No subscription database, no citation-index API and no full-text corpus search was used, so the two claim types carry asymmetric evidential weight. For the eight `famous_nonpol` pairs the target was a **review, meta-analysis or textbook-level source that asserts the relationship**; a positive hit is strong evidence and is recorded with its APA citation. For the eight `undisc_pol` pairs the target was the **absence** of a direct literature assertion; a null result from public web search is weaker evidence than a hit, because it cannot exclude a direct finding buried in a paywalled correlation table or an unindexed supplement. Each null is therefore reported together with the adjacent literature the search *did* return, so a reader can judge how close the nearest published claim comes. In the accompanying `ceil2x2_pairs_verified.csv`, `citation_verified = true` means *the literature-status claim asserted for that row was checked and upheld* - a review-level source was found for a `famous_nonpol` row, or no direct source was found for an `undisc_pol` row. Search strings are recorded verbatim in the `search_query` column.

Summary: **8 / 8** famous pairs have review-level or better support; **8 / 8** undiscussed pairs returned no direct literature. Three famous pairs and three undiscussed pairs carry caveats that a reviewer will notice; they are stated in full below rather than smoothed over.

---

## Part 1 - `famous_nonpol`: review-level support for the asserted relationship

### F1. P_numeracy x P_finliteracy  (r = +0.4057, n = 2058, predicted direction: positive)

**Evidence type:** review  
**Citation:** Stolper, O. A., & Walter, A. (2017). Financial literacy, financial advice, and financial behavior. Journal of Business Economics, 87(5), 581-643. [reviewing Hastings, J. S., Madrian, B. C., & Skimmyhorn, W. L. (2013). Financial literacy, financial education, and economic outcomes. Annual Review of Economics, 5, 347-373]; see also Lusardi, A., & Mitchell, O. S. (2014). The economic importance of financial literacy: Theory and evidence. Journal of Economic Literature, 52(1), 5-44.  
**Search query:** `review numeracy and financial literacy relationship measurement household finance survey`  
**Search date:** 2026-08-31  
**Verified:** yes

Stolper & Walter's review of the financial-literacy literature states that Hastings et al. (2013) document that respondents with higher cognitive ability and greater comfort with numerical calculation exhibit higher financial literacy on average. The Lusardi-Mitchell review class treats numeracy items as constitutive of the basic financial-literacy index, which is also the caveat flagged in the original rationale: part of the observed association is item overlap rather than a substantive link.

### F2. P_age_bracket x P_neuroticism  (r = -0.2901, n = 2058, predicted direction: negative)

**Evidence type:** meta-analysis  
**Citation:** Roberts, B. W., Walton, K. E., & Viechtbauer, W. (2006). Patterns of mean-level change in personality traits across the life course: A meta-analysis of longitudinal studies. Psychological Bulletin, 132(1), 1-25.  
**Search query:** `meta-analysis age differences neuroticism decline maturity principle Roberts Walton Viechtbauer`  
**Search date:** 2026-08-31  
**Verified:** yes

Meta-analysis of 92 longitudinal samples; emotional stability (i.e. inverse neuroticism) increases with age, most steeply in young adulthood (20-40). This is the canonical 'maturity principle' result and is restated at textbook level in introductory-psychology texts. Caveat: later coordinated longitudinal analyses (Graham et al., 2020; Atherton et al., 2021) find the neuroticism decline modest and largely complete by the mid-twenties, so the sign is secure but the magnitude is contested.

### F3. P_conscientiousness x P_education  (r = +0.1595, n = 2058, predicted direction: positive)

**Evidence type:** meta-analysis  
**Citation:** Poropat, A. E. (2009). A meta-analysis of the five-factor model of personality and academic performance. Psychological Bulletin, 135(2), 322-338.  
**Search query:** `Poropat meta-analysis Big Five conscientiousness academic performance`  
**Search date:** 2026-08-31  
**Verified:** yes

Cumulative N > 70,000; academic performance correlates significantly with conscientiousness, and the association is largely independent of intelligence. Three further meta-analyses (Poropat, 2009; Richardson et al., 2012; Trapmann et al., 2007) place the conscientiousness-performance correlation at .19-.27. Caveat: the meta-analytic staple is academic PERFORMANCE (grades/GPA); the present pair uses educational ATTAINMENT, a related but not identical outcome.

### F4. P_neuroticism x P_maximization  (r = +0.1437, n = 2058, predicted direction: positive)

**Evidence type:** review  
**Citation:** Cheek, N. N., & Schwartz, B. (2016). On the meaning and measurement of maximization. Judgment and Decision Making, 11(2), 126-146.  
**Search query:** `maximization scale regret negative affect neuroticism Schwartz 2002`  
**Search date:** 2026-08-31  
**Verified:** yes

This review of the maximization-measurement literature states that, as measured by the original Maximization Scale, maximizers are more prone to regret, more perfectionistic and more neurotic than satisficers, citing Purvis et al. (2011) and Schwartz et al. (2002). Caveat: the review's own point is that this pattern is partly a property of the MS rather than of maximizing as a construct - Diab et al. (2008) and Nenkov et al. (2008) argue the maladaptive-trait correlations are measurement artifacts, and theory-based alternative scales do not reproduce them.

### F5. P_income x P_finliteracy  (r = +0.1295, n = 2058, predicted direction: positive)

**Evidence type:** review  
**Citation:** Lusardi, A., & Mitchell, O. S. (2014). The economic importance of financial literacy: Theory and evidence. Journal of Economic Literature, 52(1), 5-44; Lusardi, A., & Mitchell, O. S. (2023). The importance of financial literacy: Opening a new field. Journal of Economic Perspectives, 37(4), 137-154.  
**Search query:** `financial literacy income wealth gradient review evidence Lusardi Mitchell 2023 Journal of Economic Perspectives`  
**Search date:** 2026-08-31  
**Verified:** yes

Both review articles identify the least financially savvy population subgroups and report that more financially literate people plan, save, invest in stocks and accumulate more wealth. The literacy-by-income/wealth gradient is a headline stylized fact of this review class rather than an incidental result.

### F6. P_openness x P_fluid  (r = +0.1217, n = 2058, predicted direction: positive)

**Evidence type:** meta-analysis  
**Citation:** Anglim, J., Dunlop, P. D., Wee, S., Horwood, S., Wood, J. K., & Marty, A. (2022). Personality and intelligence: A meta-analysis. Psychological Bulletin; Ackerman, P. L., & Heggestad, E. D. (1997). Intelligence, personality, and interests: Evidence for overlapping traits. Psychological Bulletin, 121(2), 219-245.  
**Search query:** `Ackerman Heggestad 1997 meta-analysis openness fluid intelligence trait complexes`  
**Search date:** 2026-08-31  
**Verified:** yes

Openness is the Big Five trait with the largest correlation with intelligence; Anglim et al.'s meta-analysis (>200 studies) reports openness-general cognition rho = .20, and Ackerman & Heggestad's 135-study meta-analysis reports openness-fluid intelligence r = .08 with a stronger link to crystallized intelligence. Caveat: the observed r = .12 for this pair sits inside the meta-analytic range, but the SPECIFICALLY FLUID association is the weak end of this literature - the famous version of the claim is openness-Gc.

### F7. P_agreeableness x P_dictator_sender  (r = +0.1155, n = 2058, predicted direction: positive)

**Evidence type:** meta-analysis  
**Citation:** Thielmann, I., Spadaro, G., & Balliet, D. (2020). Personality and prosocial behavior: A theoretical framework and meta-analysis. Psychological Bulletin, 146(1), 30-90.  
**Search query:** `Thielmann Spadaro Balliet 2020 meta-analysis personality prosocial behavior economic games agreeableness`  
**Search date:** 2026-08-31  
**Verified:** yes

Meta-analysis of 770 studies / 3,523 effects covering 8 broad and 43 narrow traits across six economic games including the Dictator Game; meta-analytic correlations range -.18 <= rho <= .26, with traits conceptually linked to unconditional concern for others (FFM agreeableness, honesty-humility, social value orientation) among the significant predictors. As the original rationale notes, the fame here is meta-analytic rather than textbook.

### F8. P_crt2_score x P_wason  (r = +0.1006, n = 2058, predicted direction: positive)

**Evidence type:** monograph + primary empirical  
**Citation:** Toplak, M. E., West, R. F., & Stanovich, K. E. (2014). Assessing miserly information processing: An expansion of the Cognitive Reflection Test. Thinking & Reasoning; Stanovich, K. E., West, R. F., & Toplak, M. E. (2016). The rationality quotient: Toward a test of rational thinking. MIT Press.  
**Search query:** `cognitive reflection test Wason selection task correlation rational thinking Toplak Stanovich West`  
**Search date:** 2026-08-31  
**Verified:** yes

Toplak et al. (2014) administered both deontic and non-deontic Wason selection tasks inside a rational-thinking battery alongside the CRT, and review-style syntheses of CRT correlates list 'Wason selection task performance (Toplak et al., 2014a)' among the established associations. The Rationality Quotient monograph embeds both the CRT and the selection task in one assessment framework. WEAKEST OF THE EIGHT: no meta-analysis targets this specific pair; the CR meta-analytic literature (Otero, Salgado, & Moscoso, 2022, Intelligence) covers CR against cognitive abilities and bias-avoidance generally, not the selection task specifically.

---

## Part 2 - `undisc_pol`: verification that no direct literature exists

### U1. W_policy_support_liberal x W_proportion_dom_1A  (r = +0.4116, n = 735)

**Search query:** `"proportion dominance" political ideology policy support correlation`  
**Search date:** 2026-08-31  
**Conclusion:** no direct literature found

No study was found relating proportion dominance to liberal policy support. The individual-differences literature on proportion dominance runs through thinking style, not ideology: Bartels (2006, Organizational Behavior and Human Decision Processes) reports that participants scored as 'rational' thinkers exhibited less proportion dominance than 'experiential' thinkers, and Erlandsson et al. (2015) analyse mediators (perceived impact) rather than attitudinal correlates. Searches on ideology and cognitive bias return illusory correlation, negativity bias and social-dominance orientation - a different construct that shares only the word 'dominance'.

### U2. P_political_liberalism x W_proportion_dom_1A  (r = +0.2406, n = 735)

**Search query:** `political liberalism conservatism "proportion dominance" scope insensitivity individual differences`  
**Search date:** 2026-08-31  
**Conclusion:** no direct literature found

No direct literature found. Proportion dominance is treated as a numerical-evaluability phenomenon and is studied against numeracy and analytic-vs-experiential thinking style (Bartels, 2006; Villanova & Pandelaere, 2024, on numeracy and absolute-vs-relative sensitivity). Ideology-and-judgment work exists for moral foundations and utilitarian judgment, but nothing was found pairing an ideology measure with proportion dominance. Adjacency risk to disclose: both proportion dominance and ideology have documented links to analytic thinking style, so an indirect chain is conceivable even though no paper asserts the direct association.

### U3. P_religious_nonattendance x P_mentalaccounting  (r = +0.1559, n = 2058)

**Search query:** `religiosity religious attendance mental accounting Thaler money categorization`  
**Search date:** 2026-08-31  
**Conclusion:** no direct literature found

No direct literature found. Searches return two disjoint bodies: the mental-accounting literature (Thaler, 1999; the 2025 registered-report replication of its 17 classic problems) which has no religiosity variable, and the religion-and-money literature which concerns borrowing, debt repayment, financial self-efficacy and generosity rather than Thaler-style non-fungibility bias. No paper was found reporting a religious-attendance x mental-accounting association.

### U4. P_religious_nonattendance x W_prob_maximizing_cards  (r = +0.1501, n = 1032)

**Search query:** `probability matching maximizing individual differences religiosity correlates`  
**Search date:** 2026-08-31  
**Conclusion:** no direct literature found

No direct literature found. Probability matching is studied as a choice-learning and financial-decision phenomenon, with individual differences reported against wealth, statistics training and self-reported pattern detection (e.g. the PLOS ONE experimental study of probability matching in financial decision making), not against religiosity. Adjacency risk to disclose: religiosity has a documented negative association with analytic/reflective cognitive style (Gervais & Norenzayan; Pennycook et al.), and cognitive reflection has been linked to endorsing maximizing strategies on probabilistic prediction tasks (Koehler & James, 2010), so an indirect two-step chain exists in the literature - but no publication asserts the attendance x probability-matching link directly.

### U5. P_political_liberalism x P_mentalaccounting  (r = +0.1371, n = 2058)

**Search query:** `political ideology liberalism mental accounting bias correlation`  
**Search date:** 2026-08-31  
**Conclusion:** no direct literature found

No direct literature found. Ideology-and-cognition searches return illusory correlation, negativity bias, risk perception and cultural-theory work; none of the retrieved items measures mental accounting. Consistent with the original rationale that ideology has been tied to risk and loss aversion but not to money categorization.

### U6. W_policy_support_liberal x W_base_rate_30eng  (r = -0.1146, n = 1043)

**Search query:** `base rate neglect engineer lawyer problem political attitudes policy support individual differences`  
**Search date:** 2026-08-31  
**Conclusion:** no direct literature found

No direct literature found. The individual-differences literature on the engineer/lawyer problem is uniformly cognitive: cognitive reflection and short-term memory (Vartanian et al., 2018, Journal of Cognitive Neuroscience), numeracy and Bayesian integration (Mangiavacchi et al., 2023), and causal-structure manipulations (Psychonomic Bulletin & Review, 2020). No retrieved study relates base-rate neglect to policy-attitude scales.

### U7. P_RFS x W_less_is_more_A  (r = +0.1068, n = 735)

**Search query:** `less-is-more effect joint separate evaluation individual differences correlates scale`  
**Search date:** 2026-08-31  
**Conclusion:** no direct literature found

No direct literature found. The joint-vs-separate evaluation literature is experimental and effect-focused (Hsee's evaluability hypothesis; Hsee & Zhang's distinction bias; the Collabra close replication of Hsee 1998). Retrieved work reports no attitudinal or religious individual-difference correlates at all, and nothing linking the effect to the RFS under either reading of that abbreviation.

### U8. P_RFS x W_prob_maximizing_cards  (r = -0.0995, n = 1032)

**Search query:** `religious fundamentalism scale RFS decision-making bias probability matching correlation`  
**Search date:** 2026-08-31  
**Conclusion:** no direct literature found

No direct literature found. RFS validity work reports correlates in the authoritarianism/dogmatism/hope space (RFS-RWA r = .66-.75; RFS-dogmatism r = .47-.78) and, in the cognitive domain, executive-function measures such as card sorting (Zhong et al., 2017, Neuropsychologia). Nothing links the RFS to probability matching. Same indirect-chain caveat as pair 11 applies; also note that P_RFS remains ambiguous in the source table, and this check was run against the Religious Fundamentalism Scale reading, which is the reading most likely to have an attitudinal correlates literature.

---

## Residual risks a reviewer may still raise

1. **Construct mismatch on F3.** The meta-analytic claim concerns conscientiousness and academic *performance* (grades, GPA); the design measures educational *attainment*. The pair is famous in the adjacent sense, not the literal one.
2. **F6 is the weak-fluid end of a strong claim.** Openness-intelligence is famous; openness-*fluid* intelligence is the small half of it (meta-analytic r = .08 to rho = .20, against a stronger crystallized association). The observed r = .12 is consistent, but the fame belongs to openness-Gc.
3. **F8 has no meta-analysis.** CRT x Wason rests on a primary empirical battery study plus an MIT Press monograph and review-style restatements of it, not on a quantitative synthesis of that specific pair.
4. **F4 is contested by design.** The review that supports neuroticism-maximization is the same literature that argues the association is a property of the original Maximization Scale rather than of maximizing.
5. **F1 has item overlap.** Numeracy items are constitutive of the basic financial-literacy index, so part of the correlation is measurement overlap - already flagged in the source rationale and not resolved by the citation.
6. **U2, U4 and U8 have a two-step indirect path.** Proportion dominance and probability matching both have documented links to analytic-vs-experiential thinking style, and both ideology and religiosity have documented links to analytic thinking style. No publication asserts the direct pairing, which is what 'undiscussed' claims, but the pairs are not causally insulated from the literature.
7. **P_RFS remains ambiguous.** U7 and U8 were checked against the Religious Fundamentalism Scale reading of `P_RFS`, the reading most likely to possess an attitudinal-correlates literature and therefore the most conservative one for a null claim. If `P_RFS` denotes something else in the source instrument, those two nulls should be re-run.
8. **Null-search asymmetry.** Every `undisc_pol` conclusion is 'no direct literature found by public web search on 2026-08-31', not 'no such finding exists'. A full-text or citation-index sweep would strengthen these eight rows and is the obvious next step if a reviewer presses.
