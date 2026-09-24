# Cross-model replication kit

## Contents
- run_xmodel.py — executor (OpenAI-compatible API; works with GPT, DeepSeek, local vLLM, etc.)
- P2_master_protocol_bank.json — full experimental protocol for 61 effects × 2 variants
- P2_paradigm_class.csv — paradigm categories and bare-state measurable markers
- P2_analysis_units_claude.csv — analysis-ready data for the two Claude models (for comparison)

## Run (full replication; at `--n 30` about 3,660 main + 1,290 bare = ~5,000 calls, ~6,200 at the default `--n 40`)
    pip install openai
    OPENAI_API_KEY=sk-... python run_xmodel.py --model gpt-5.2 --n 40 --out gpt52

## Quick version (first run the 9 A1 effects to validate the pipeline, ~800 calls)
    OPENAI_API_KEY=sk-... python run_xmodel.py --model gpt-5.2 --n 40 --out gpt52_pilot \
      --effects Knobe2003_side_effect,Tversky1973_availability_letters,Husnu2010_imagined_contact,Eyal2008_moral_distance,Rottenstreich2001_affective_lottery,Oppenheimer2009_retrospective_gambler,Miyamoto2002_correspondence_bias,Genschow2017_free_will_attribution,Chao2017_thankyou_gift

## Local open-source models (OpenAI-compatible port of vLLM/ollama)
    python run_xmodel.py --model llama-3.3-70b --base-url http://localhost:8000/v1 --n 40 --out llama70b

## Afterward
Parse xm_main_resp.json and xm_bare_resp.json from the --out directory with the
parsers in ../code/ (see xm_analyze.py; the *_parsers_patch_*.py files show the
per-model adaptations used for GPT and DeepSeek), then fit the three-coefficient
model with ../code/fit_main_model.py on the resulting analysis units.

## Notes
- For reasoning models (thinking tokens count against the budget), max_completion_tokens=2000 has been set; if you still see empty responses, increase this value;
- Checkpoint/resume: main saves every 5 protocols; after an interruption, delete the completed portion and rerun, or simply accept the partial results;
- Please keep refusals as-is (text=null) — they are data for the S1 analysis.