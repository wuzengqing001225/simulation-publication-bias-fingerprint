# Ceiling-fix rerun kit

Reruns the 14 effects whose simulated effect sizes sat at |r| > .95 in the
main waves, with the response format changed to a 0-100 continuous scale
(everything else in the protocols is unchanged). Exploratory supplement to
the main paper; also measures how much scale headroom reduces the
effect-level residual.

## Run (per model, ~2,400 calls; 16 workers ~20-40 min)

    OPENAI_API_KEY=sk-...  python run_ceiling.py \
        --model deepseek-v4-flash --base-url https://api.deepseek.com/v1 \
        --out deepseek_ceiling --workers 16

    OPENAI_API_KEY=sk-...  python run_ceiling.py \
        --model gpt-5.6-terra --out gpt_ceiling --workers 16

Interrupting is safe: rerunning the same command resumes (completed rows are
kept; output is written atomically every 100 calls). On 429 rate-limit
errors lower --workers to 8.

## Scoring

Score the two response files with the bundled analyzer:

    python ceiling_analyze.py --main deepseek_ceiling/ceil_main_resp.json \
        --bare deepseek_ceiling/ceil_bare_resp.json --prefix CEIL_ds

## Files

- ceiling_bank.json — 28 rewritten protocols (14 effects x 2 variants) with
  machine-readable scoring specs and per-condition bare-task prompts.
- run_ceiling.py — multithreaded resumable runner (OpenAI-compatible APIs).
- ceiling_analyze.py — generic 0-100 parser + per-effect scorer
  (validated end to end on synthetic data with known effect directions).

## Reproducing the paper's remediated refits

From the repository root (about 30-50 minutes for the three Bayesian fits):

    python ceiling_kit/ceiling_refit.py

Outputs, written to `analysis_tables/`:

- `CEIL_{claude,gpt,ds}_units.csv` — hybrid analysis tables (rerun values
  substituted for the ceiling effects; `replaced` flag per unit).
- `CEIL_{claude,gpt,ds}_idata.nc` — full posteriors of the three refits.
- `CEIL_refit_posteriors.csv` — coefficient means, 95% HDIs, P(beta_pub > 0).
- `CEIL_arm_variance.csv` — arm-level rating means and SDs (pass
  `--responses <raw ceil_main_resp.json files>`; raw responses are
  available from the authors on request).
