# What do llm-simulated participants track?

Code, protocols, and derived data for the paper. Raw model responses are not
redistributed here; they are available from the authors on request.

## Where each claim in the paper lives

| Paper claim | File(s) |
|---|---|
| The 61 effects: anchor effect sizes, sample sizes, paradigm class, transcription provenance | `protocols/effect_summary_61.csv` |
| The 61 effects: original references and DOIs | `protocols/effect_DOI_list_61.csv` |
| No-bare-control refit on the estimation sample (Appendix on robustness) | `analysis_tables/nobare_posteriors.csv` |
| Sensitivity-subset fits (Fig. 2b, Appendix on robustness) | `code/fit_sensitivity.py`, `analysis_tables/sensitivity_posteriors.csv` |
| All 122 simulation prompt templates (61 effects x 2 transcription variants) | `protocols/P2_master_protocol_bank.json` |
| Bare-task battery, five wording seeds per task | `protocols/bare_task_bank.json` |
| Per-effect transcription fidelity notes and evidence tier | `protocols/P2_master_protocol_bank.json` (fields `evidence_tier`, `notes`) |
| Per-effect bare effect sizes | `analysis_tables/P2_bare_g0.csv`, `P2_bare_g1.csv`, `P2_bare_g2.csv`, `P2_bare_old19.csv` |
| Analysis units for the estimation model | `analysis_tables/P2_analysis_units.csv` |
| Design power simulation and scaling curve (Appendix on precision) | `analysis_tables/power_simulation.csv`, `analysis_tables/power_design_curve.csv`, `code/power_simulation.py` |
| Positive controls, transcription cross-pairing (three families), direction subset, structure missingness bounds | `code/positive_controls_and_checks.py`, `analysis_tables/positive_controls_and_checks.json`, `analysis_tables/PC_*_cells_*.csv`, `protocols/positive_control_findings.json` |
| Print-size main-text panels (Figures 2 and 4) | `code/make_main_figures.py` |
| Scale, transcription, paradigm, interaction, neutral-prompt, conventional-validation, item-similarity checks | `code/robustness_checks.py`, `analysis_tables/robustness_checks.json` |
| NumPyro implementation of the main model | `code/nm_models.py` |
| Hierarchical prior, simulation-based calibration, Bayesian power | `code/calibration_and_power.py`, `analysis_tables/hierarchical_prior_posteriors.json`, `sbc_ranks.csv`, `power_bayes.csv` |
| Neutral-system-prompt rerun, scored cells | `analysis_tables/NT_*_cells_*.csv` |
| Construct item texts and embeddings | `protocols/construct_item_texts.json`, `code/embed_constructs.py`, `analysis_tables/construct_embeddings.json` |
| Posterior draws of the main measurement-error model | `analysis_tables/P2_main_model_idata.nc` |
| Narrative labels, all four rounds and the consensus | `analysis_tables/narrative_final_fourround.csv` |
| Refusal condition-type mapping / per-cell refusal rates | `analysis_tables/S1_condition_map.csv`, `analysis_tables/S1_refusal_units.csv` |
| Construct-battery (structure) results | `analysis_tables/B2_threeway.csv`, `B2_construct_matrix_*.csv` |
| Main model fitting script | `code/fit_main_model.py` |
| Parsers and scoring code | `code/` (`a0_scoring.py`, `a1_scoring.py`, `p2_parsers_g*.py`) |
| Cross-family parser adaptations | `code/xm_parsers_patch_*.py`, `code/ds_parsers_patch_*.py` |
| Ceiling-fix rerun: kit, per-cell results (Appendix on ceiling rerun) | `ceiling_kit/`, `analysis_tables/CEIL_*_cells.csv`, `CEIL_*_bare.csv` (Claude, GPT, DeepSeek) |
| Likelihood treatments: narrative-stratified, censored, transcription-error fits | `code/fit_model_treatments.py`, `analysis_tables/model_treatments_posteriors.csv`, `analysis_tables/narrative_map.json` |
| Fame-by-content design completion (structure experiment) | `protocols/ceil2x2_pairs.csv` (with per-pair citation evidence), `protocols/ceil2x2_fame_evidence.md`, `code/ceil2x2_items.py`, `code/ceil2x2_items_notes.md`, `analysis_tables/ceil2x2_subject_scores.csv` |
| Fame-by-content analysis (pair correlations, cell means, main effects, tests, winner's-curse simulation) | `code/analyze_ceil2x2.py`, `analysis_tables/ceil2x2_results.csv`, `analysis_tables/ceil2x2_stats.json` |
| Ceiling-remediation refits: hybrid unit tables, posteriors, arm variance | `ceiling_kit/ceiling_refit.py`, `analysis_tables/CEIL_{claude,gpt,ds}_units.csv`, `CEIL_{claude,gpt,ds}_idata.nc`, `CEIL_refit_posteriors.csv`, `CEIL_arm_variance.csv` |
| Cross-family end-to-end scorer (raw responses -> cells -> units -> fit) | `code/xm_analyze.py` |
| Cross-family runner (OpenAI-compatible APIs) | `xmodel_kit/` |
| Cross-family derived results (GPT, DeepSeek) | `xmodel_results/` |

## Layout

- `protocols/` — effect lists, prompt template banks, paradigm-class coding,
  shortlists used for sampling.
- `code/` — runners, per-effect response parsers, scoring, and analysis
  scripts for the main (Claude) waves.
- `analysis_tables/` — derived per-cell effect sizes, analysis units, refusal
  tables, narrative labels, and posterior draws (`.nc`, ArviZ InferenceData).
- `xmodel_kit/` — self-contained kit to rerun the full protocol against any
  OpenAI-compatible API (see its `README.md`).
- `ceiling_kit/` — kit for the exploratory headroom-scale rerun of the 14
  ceiling effects (see its `README.md`).
- `xmodel_results/` — derived results of the GPT and DeepSeek reruns.

## Reproducing the main result

Environment: Python 3.11+, `pip install -r requirements.txt`.

1. Effect selection and anchors: `protocols/effect_summary_61.csv` (references and DOIs in `protocols/effect_DOI_list_61.csv`).
2. Simulation prompts: `protocols/P2_master_protocol_bank.json`; bare-task
   prompts: `protocols/bare_task_bank.json`.
3. Parse responses with `code/p2_parsers_g*.py`; score to per-cell effect
   sizes (`analysis_tables/P2_cells_g*.csv`).
4. Fit the measurement-error model: `python code/fit_main_model.py`
   (reads `analysis_tables/P2_analysis_units.csv`; the reported posterior
   is `analysis_tables/P2_main_model_idata.nc`, random seed 11).
5. Cross-family replication: run the kit (`xmodel_kit/README.md`), then score
   end to end with
   `python code/xm_analyze.py --main xm_main_resp.json --bare xm_bare_resp.json --prefix XM --fit`
   (validated to reproduce the shipped GPT tables exactly).
