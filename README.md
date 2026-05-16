# Psychological Coherence in LLMs — Code and Data

> **Anonymous submission.** Author information will be added upon de-anonymization.

This repository contains the code and pre-collected results used in our study of psychological coherence in large language models. We examine whether LLMs produce self-reports (via psychometric surveys) that are *coherent* with their revealed behavioral tendencies (via established behavioral tasks), across four behavioral domains and 11 frontier models.

---

## Repository Structure

```
.
├── configs/                   # Experiment configuration files
│   ├── behavior/              # Behavioral task configs (CCT, Sycophancy, Honesty, IAT)
│   ├── big5/                  # Big Five self-report sweep configs
│   ├── tpb/                   # TPB self-report sweep configs
│   ├── merge/                 # Merge configs (linking SR to behavior results)
│   ├── openrouter_models.json # Model registry (keys → OpenRouter model IDs)
│   └── selected_diverse_personas.json
│
├── src/                       # Core library
│   ├── core/                  # Shared types
│   ├── llms/                  # OpenAI-compatible LLM client (OpenRouter)
│   ├── perturbations/         # Parameter grid & persona steering utilities
│   ├── runner/                # Task runners (CCT, Sycophancy, Honesty, IAT, TPB)
│   ├── surveys/               # TPB Likert survey implementation
│   └── tasks/                 # Behavioral task environments (CCT, Sycophancy, Honesty, IAT)
│
├── scripts/
│   ├── config_based_sweeps/   # Entry-point sweep scripts (reproduce data collection)
│   ├── merging/               # Scripts to join self-report and behavioral results
│   ├── analysis/              # RQ analysis scripts (reproduce all paper figures)
│   └── helper/                # Diagnostic and visualization helpers
│
└── results/                   # Pre-collected results (included in this repo)
    ├── between/               # Between-session results
    │   ├── grid/              # Parameter-grid induction
    │   │   ├── session_sr/    # Self-report runs
    │   │   └── session_beh/   # Behavioral task runs
    │   └── personas/          # Persona induction
    │       ├── session_sr/
    │       └── session_beh/
    ├── within/                # Within-session results (SR + behavior in same context)
    │   ├── grid/
    │   └── personas/
    └── merged/                # Merged SR × behavior CSVs (ready for analysis)
        ├── between/
        └── personas/
```

---

## Experimental Design

### Behavioral Tasks

| Task | Abbreviation | What it measures |
|------|-------------|------------------|
| Columbia Card Task | CCT | Risk-taking |
| Sycophancy probe | Sycophancy | Social conformity / opinion-shifting |
| Honesty calibration | Honesty | Confidence calibration under social pressure |
| Implicit Association Test | IAT | Implicit bias (response-order sensitivity) |

### Self-Report Instruments

- **TPB (Theory of Planned Behavior):** Attitude, subjective norms, perceived behavioral control, and intention — administered per task and behavioral policy.
- **Big Five (Big5):** Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism.

### Induction Conditions

- **Parameter grid:** 3 system prompts × 3 temperatures × 3 seeds = 27 conditions per model.
- **Persona induction:** Diverse persona descriptions drawn from `configs/selected_diverse_personas.json`.

### Session Designs

- **Between-session:** Self-reports and behavioral tasks run in independent sessions (no shared context).
- **Within-session:** Self-reports and behavioral tasks run in the same conversation context.

---

## Setup

### Requirements

- Python 3.10+
- An [OpenRouter](https://openrouter.ai) API key


### Environment

Create a `.env` file in the root directory:

```
OPENROUTER_API_KEY=your_key_here
```

Load it before running any script:

```bash
set -a; source .env; set +a
```

---

## Reproducing the Results

> **Note:** Pre-collected results are already included under `results/`. The steps below are for full reproduction from scratch. Re-running the sweeps will take substantial time and API cost. For analysis only, skip to [Step 3](#step-3-merge-self-reports-with-behavior) after confirming the `results/` CSVs are present.

### Step 1 — Self-Report Sweeps (TPB and Big Five)

**Between-session, parameter grid:**
```bash
# TPB — Sycophancy task
python scripts/config_based_sweeps/sweep_tpb_variants.py \
  --config configs/tpb/tpb_sycophancy_psycohere_grid.json \
  --out_root results/between/grid/session_sr --resume

# TPB — Honesty task
python scripts/config_based_sweeps/sweep_tpb_variants.py \
  --config configs/tpb/tpb_honesty_psycohere_grid.json \
  --out_root results/between/grid/session_sr --resume

# TPB — CCT task
python scripts/config_based_sweeps/sweep_tpb_variants.py \
  --config configs/tpb/tpb_cct_psycohere_grid.json \
  --out_root results/between/grid/session_sr --resume

# TPB — IAT task
python scripts/config_based_sweeps/sweep_tpb_variants.py \
  --config configs/tpb/tpb_iat_psycohere_grid.json \
  --out_root results/between/grid/session_sr --resume

# Big Five
python scripts/config_based_sweeps/sweep_tpb_variants.py \
  --config configs/big5/big5_psycohere_grid.json \
  --out_root results/between/grid/session_sr --resume
```

**Between-session, persona induction** — repeat with `configs/tpb/*_personas.json` and `configs/big5/big5_psycohere_personas.json`, writing to `results/between/personas/session_sr`.

### Step 2 — Behavioral Task Sweeps

**Between-session, parameter grid:**
```bash
# CCT
python scripts/config_based_sweeps/sweep_cct_variants.py \
  --config configs/behavior/cct_psycohere_grid.json \
  --out_root results/between/grid/session_beh --resume

# Sycophancy
python scripts/config_based_sweeps/sweep_sycophancy_variants.py \
  --config configs/behavior/sycophancy_psycohere_grid.json \
  --out_root results/between/grid/session_beh --resume

# Honesty
python scripts/config_based_sweeps/sweep_honesty_variants.py \
  --config configs/behavior/honesty_psycohere_grid.json \
  --out_root results/between/grid/session_beh --resume

# IAT
python scripts/config_based_sweeps/sweep_iat_variants.py \
  --config configs/behavior/iat_psycohere_grid.json \
  --out_root results/between/grid/session_beh --resume
```

Repeat with `configs/behavior/*_personas.json` → `results/between/personas/session_beh` for persona induction.

**Within-session** (TPB + behavior in the same context):
```bash
# Grid — Honesty
python scripts/config_based_sweeps/sweep_combined_variants.py \
  --sr_config  configs/tpb/tpb_honesty_psycohere_grid.json \
  --beh_config configs/behavior/honesty_psycohere_grid.json \
  --out_root   results/within/grid --resume

# Grid — CCT
python scripts/config_based_sweeps/sweep_combined_variants.py \
  --sr_config  configs/tpb/tpb_cct_psycohere_grid.json \
  --beh_config configs/behavior/cct_psycohere_grid.json \
  --out_root   results/within/grid --resume
```

Repeat with persona configs → `results/within/personas`.

### Step 3 — Merge Self-Reports with Behavior

**CCT:**
```bash
# Big5 × CCT (grid)
python scripts/merging/merge_selfreport_cct.py \
  --selfreport_csv results/between/grid/session_sr/big5_psycohere_grid/big5/tpb_likert_runs.csv \
  --instrument big5 \
  --cct_runs_csv results/between/grid/session_beh/cct-psycohere-grid/cct-neutral/cct_runs.csv \
  --out_csv results/merged/between/grid/big5_x_cct.csv

# TPB × CCT (grid)
python scripts/merging/merge_tpb_with_behavior.py \
  --config configs/merge/merge_cct_tpb_psycohere.json \
  --tpb_root results/between/grid/session_sr/tpb_cct_psycohere_grid \
  --behavior_runs_csv results/between/grid/session_beh/cct-psycohere-grid/cct-neutral/cct_runs.csv \
  --out_prefix results/merged/between/grid/tpb_x_cct \
  --tpb_runs_filename tpb_likert_runs.csv
```

Replace `grid` with `personas` and adjust paths accordingly for persona-induction variants. Analogous scripts exist for Sycophancy (`merge_selfreport_sycophancy.py`), Honesty (`merge_selfreport_honesty.py`), and IAT (`merge_selfreport_iat.py`).

### Step 4 — Reproduce Paper Analyses and Figures

All four core research questions have dedicated analysis scripts:

```bash
# RQ1 — Coherence between self-reports and behavior
python scripts/analysis/rq1_alignment_analysis.py \
  --cct_master     results/analysis/cct/cct_master.csv \
  --syc_master     results/analysis/sycophancy/sycophancy_master.csv \
  --honesty_master results/analysis/honesty/honesty_master.csv \
  --iat_master     results/analysis/iat/iat_master.csv \
  --out_dir        results/analysis/rq1_alignment

# RQ2 — Framework comparison (Big Five vs TPB)
python scripts/analysis/rq2_framework_comparison.py \
  --cct_master     results/analysis/cct/cct_master.csv \
  --syc_master     results/analysis/sycophancy/sycophancy_master.csv \
  --honesty_master results/analysis/honesty/honesty_master.csv \
  --iat_master     results/analysis/iat/iat_master.csv \
  --out_dir        results/analysis/rq2_framework

# RQ3 — Context separation (within- vs between-session)
python scripts/analysis/rq3_context_separation.py \
  --cct_master     results/analysis/cct/cct_master.csv \
  --syc_master     results/analysis/sycophancy/sycophancy_master.csv \
  --honesty_master results/analysis/honesty/honesty_master.csv \
  --iat_master     results/analysis/iat/iat_master.csv \
  --out_dir        results/analysis/rq3_context \
  --n_boot         2000

# RQ4 — Induction comparison (grid vs persona)
python scripts/analysis/rq4_induction_comparison.py \
  --cct_master     results/analysis/cct/cct_master.csv \
  --syc_master     results/analysis/sycophancy/sycophancy_master.csv \
  --honesty_master results/analysis/honesty/honesty_master.csv \
  --iat_master     results/analysis/iat/iat_master.csv \
  --out_dir        results/analysis/rq4_induction \
  --n_boot         2000
```

Additional diagnostic scripts (variance, floor/ceiling, instruction sensitivity) are in `scripts/analysis/` and `scripts/helper/`.

---

## Models

All models are accessed via [OpenRouter](https://openrouter.ai). The model registry is in `configs/openrouter_models.json`. The 11 models used in the study span multiple model families and parameter scales; exact model IDs are listed in the registry file.

---

## License

To be added upon de-anonymization.
