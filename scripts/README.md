# Scripts Reference

This directory contains all entry-point scripts for data collection, merging, analysis, and visualization. The four subdirectories map to the four stages of the pipeline:

```
scripts/
├── config_based_sweeps/   # Stage 1 — run LLM data collection
├── merging/               # Stage 2 — join SR and behavior results
├── analysis/              # Stage 3 — reproduce RQ figures and stats
└── helper/                # Diagnostics and per-model visualizations
```

Pre-collected results are already under `results/`. Re-running collection requires an OpenRouter API key and significant cost/time. **For analysis only, skip directly to [Stage 2](#stage-2-merging) after verifying your `results/` CSVs are present.**

---

## Prerequisites

```bash
# Load your API key
set -a; source .env; set +a

# Confirm it is set
echo $OPENROUTER_API_KEY
```

All sweep scripts read model IDs from `configs/openrouter_models.json` via the `models_config` field in each JSON config.

---

## Stage 1 — Data Collection (`config_based_sweeps/`)

### Common flags (all sweep scripts)

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | *(required)* | Path to the JSON experiment config |
| `--out_root` | *(required)* | Root directory for outputs |
| `--resume` | off | Skip runs already recorded in the output CSV — **always use this** |
| `--fail_fast` | off | Stop on first error (default: log and continue) |
| `--model_keys` | all in config | Run a subset of models only (honesty and TPB scripts only) |

---

### 1a. TPB Self-Report Sweep — `sweep_tpb_variants.py`

Runs the Theory of Planned Behavior (TPB) Likert battery for a single task and induction condition.

**Scale:** 27 conditions/model (grid) or 30 conditions/model (personas) × 11 models × 2 variants per task.

**Outputs** written to `<out_root>/<experiment_name>/<variant_id>/`:
- `tpb_likert_runs.csv` — one row per completed run (used by merging scripts)
- `tpb_likert.csv` — item-level long-format responses

#### Between-session, parameter grid

```bash
# Sycophancy task
python scripts/config_based_sweeps/sweep_tpb_variants.py \
  --config configs/tpb/tpb_sycophancy_psycohere_grid.json \
  --out_root results/between/grid/session_sr --resume

# Honesty task
python scripts/config_based_sweeps/sweep_tpb_variants.py \
  --config configs/tpb/tpb_honesty_psycohere_grid.json \
  --out_root results/between/grid/session_sr --resume

# CCT (risk) task
python scripts/config_based_sweeps/sweep_tpb_variants.py \
  --config configs/tpb/tpb_cct_psycohere_grid.json \
  --out_root results/between/grid/session_sr --resume

# IAT (implicit bias) task
python scripts/config_based_sweeps/sweep_tpb_variants.py \
  --config configs/tpb/tpb_iat_psycohere_grid.json \
  --out_root results/between/grid/session_sr --resume
```

#### Between-session, persona induction

```bash
python scripts/config_based_sweeps/sweep_tpb_variants.py \
  --config configs/tpb/tpb_sycophancy_psycohere_personas.json \
  --out_root results/between/personas/session_sr --resume

python scripts/config_based_sweeps/sweep_tpb_variants.py \
  --config configs/tpb/tpb_honesty_psycohere_personas.json \
  --out_root results/between/personas/session_sr --resume

python scripts/config_based_sweeps/sweep_tpb_variants.py \
  --config configs/tpb/tpb_cct_psycohere_personas.json \
  --out_root results/between/personas/session_sr --resume

python scripts/config_based_sweeps/sweep_tpb_variants.py \
  --config configs/tpb/tpb_iat_psycohere_personas.json \
  --out_root results/between/personas/session_sr --resume
```

---

### 1b. Big Five Self-Report Sweep — `sweep_tpb_variants.py`

The Big Five sweep uses the same script as TPB; the config switches the item set to BFI-44.

**Scale:** 27 conditions/model (grid) or 30 conditions/model (personas) × 11 models × 1 variant.

**Outputs** written to `<out_root>/<experiment_name>/big5/`:
- `tpb_likert_runs.csv`
- `tpb_likert.csv`

```bash
# Grid
python scripts/config_based_sweeps/sweep_tpb_variants.py \
  --config configs/big5/big5_psycohere_grid.json \
  --out_root results/between/grid/session_sr --resume

# Personas
python scripts/config_based_sweeps/sweep_tpb_variants.py \
  --config configs/big5/big5_psycohere_personas.json \
  --out_root results/between/personas/session_sr --resume
```

---

### 1c. CCT Behavioral Sweep — `sweep_cct_variants.py`

Runs the Columbia Card Task (risk-taking). Each run is a 20-round card game.

**Scale:** 297 runs/variant (27 × 11 grid) or 330 runs/variant (30 × 11 personas).

**Outputs** written to `<out_root>/<experiment_name>/<variant_id>/`:
- `cct_runs.csv` — one row per completed game (primary outcome: mean cards flipped)
- `cct_steps.csv` — round-level choices across all games

```bash
# Grid
python scripts/config_based_sweeps/sweep_cct_variants.py \
  --config configs/behavior/cct_psycohere_grid.json \
  --out_root results/between/grid/session_beh --resume

# Personas
python scripts/config_based_sweeps/sweep_cct_variants.py \
  --config configs/behavior/cct_psycohere_personas.json \
  --out_root results/between/personas/session_beh --resume
```

---

### 1d. Sycophancy Behavioral Sweep — `sweep_sycophancy_variants.py`

Runs the Asch-style sycophancy probe. The model answers a question, receives a confederate challenge, and re-answers.

**Scale:** 297 runs (grid) or 330 runs (personas).

**Outputs** written to `<out_root>/<experiment_name>/<variant_id>/`:
- `sycophancy_runs.csv` — one row per episode (primary outcome: opinion shift)
- `sycophancy_steps.csv` — turn-level responses

```bash
# Grid
python scripts/config_based_sweeps/sweep_sycophancy_variants.py \
  --config configs/behavior/sycophancy_psycohere_grid.json \
  --out_root results/between/grid/session_beh --resume

# Personas
python scripts/config_based_sweeps/sweep_sycophancy_variants.py \
  --config configs/behavior/sycophancy_psycohere_personas.json \
  --out_root results/between/personas/session_beh --resume
```

**Smoke test** (no API calls):
```bash
python scripts/config_based_sweeps/sweep_sycophancy_variants.py \
  --config configs/behavior/sycophancy_psycohere_grid.json \
  --out_root /tmp/test_syc \
  --provider_override mock --mock_mode agree_user --max_runs 5
```

---

### 1e. Honesty Behavioral Sweep — `sweep_honesty_variants.py`

Runs the confidence-calibration honesty task. The model answers trivia questions, sees social pressure to update, and reports a final confidence.

**Scale:** 297 runs (grid) or 330 runs (personas).

**Outputs** written to `<out_root>/<experiment_name>/<variant_id>/`:
- `honesty_runs.csv` — one row per episode (primary outcome: confidence delta)
- `honesty_steps.csv` — turn-level responses

```bash
# Grid
python scripts/config_based_sweeps/sweep_honesty_variants.py \
  --config configs/behavior/honesty_psycohere_grid.json \
  --out_root results/between/grid/session_beh --resume

# Personas
python scripts/config_based_sweeps/sweep_honesty_variants.py \
  --config configs/behavior/honesty_psycohere_personas.json \
  --out_root results/between/personas/session_beh --resume
```

**Run a single model** (useful for spot-checks):
```bash
python scripts/config_based_sweeps/sweep_honesty_variants.py \
  --config configs/behavior/honesty_psycohere_grid.json \
  --out_root results/between/grid/session_beh \
  --model_keys claude37_sonnet --resume
```

---

### 1f. IAT Behavioral Sweep — `sweep_iat_variants.py`

Runs the Implicit Association Test across 6 domain pairs. Each episode measures response-order sensitivity as a proxy for implicit bias.

**Scale:** 297 runs (grid) or 330 runs (personas).

**Outputs** written to `<out_root>/<experiment_name>/<variant_id>/`:
- `iat_runs.csv` — one row per completed test (primary outcome: D-score / bias index)
- `iat_steps.csv` — trial-level responses

```bash
# Grid
python scripts/config_based_sweeps/sweep_iat_variants.py \
  --config configs/behavior/iat_psycohere_grid.json \
  --out_root results/between/grid/session_beh --resume

# Personas
python scripts/config_based_sweeps/sweep_iat_variants.py \
  --config configs/behavior/iat_psycohere_personas.json \
  --out_root results/between/personas/session_beh --resume
```

**Dry run** (mock LLM, no API calls):
```bash
python scripts/config_based_sweeps/sweep_iat_variants.py \
  --config configs/behavior/iat_psycohere_grid.json \
  --out_root /tmp/test_iat \
  --mock_llm random
```

---

### 1g. Within-Session Combined Sweep — `sweep_combined_variants.py`

Runs SR and behavior **in the same message thread** — the SR phase primes the context before the behavioral task begins. Requires a matched pair of SR and behavior configs (identical grid axes).

**Outputs** written to `<out_root>/<sr_experiment_name>/<sr_variant_id>/`:
- `combined_runs.csv` — one row per run with all SR subscale means + behavioral outcome columns

#### Grid induction

```bash
# CCT
python scripts/config_based_sweeps/sweep_combined_variants.py \
  --sr_config  configs/tpb/tpb_cct_psycohere_grid.json \
  --beh_config configs/behavior/cct_psycohere_grid.json \
  --out_root   results/within/grid --resume

# Sycophancy
python scripts/config_based_sweeps/sweep_combined_variants.py \
  --sr_config  configs/tpb/tpb_sycophancy_psycohere_grid.json \
  --beh_config configs/behavior/sycophancy_psycohere_grid.json \
  --out_root   results/within/grid --resume

# Honesty
python scripts/config_based_sweeps/sweep_combined_variants.py \
  --sr_config  configs/tpb/tpb_honesty_psycohere_grid.json \
  --beh_config configs/behavior/honesty_psycohere_grid.json \
  --out_root   results/within/grid --resume

# IAT
python scripts/config_based_sweeps/sweep_combined_variants.py \
  --sr_config  configs/tpb/tpb_iat_psycohere_grid.json \
  --beh_config configs/behavior/iat_psycohere_grid.json \
  --out_root   results/within/grid --resume
```

#### Persona induction

```bash
# CCT
python scripts/config_based_sweeps/sweep_combined_variants.py \
  --sr_config  configs/tpb/tpb_cct_psycohere_personas.json \
  --beh_config configs/behavior/cct_psycohere_personas.json \
  --out_root   results/within/personas --resume

# Sycophancy
python scripts/config_based_sweeps/sweep_combined_variants.py \
  --sr_config  configs/tpb/tpb_sycophancy_psycohere_personas.json \
  --beh_config configs/behavior/sycophancy_psycohere_personas.json \
  --out_root   results/within/personas --resume

# Honesty
python scripts/config_based_sweeps/sweep_combined_variants.py \
  --sr_config  configs/tpb/tpb_honesty_psycohere_personas.json \
  --beh_config configs/behavior/honesty_psycohere_personas.json \
  --out_root   results/within/personas --resume

# IAT
python scripts/config_based_sweeps/sweep_combined_variants.py \
  --sr_config  configs/tpb/tpb_iat_psycohere_personas.json \
  --beh_config configs/behavior/iat_psycohere_personas.json \
  --out_root   results/within/personas --resume
```

> **Note:** The combined runner validates that both configs use identical grid axes (same seeds, temperatures, persona mode). It will raise a `GridMismatch` error if they differ.

---

## Monitoring Collection Progress

### Quick row counts

The fastest way to check how many runs are complete is to count rows in each output CSV:

```bash
# Count completed runs across all between-session grid SR sweeps
wc -l results/between/grid/session_sr/**/tpb_likert_runs.csv

# Count completed behavioral runs (grid)
wc -l results/between/grid/session_beh/**/cct_runs.csv
wc -l results/between/grid/session_beh/**/sycophancy_runs.csv
wc -l results/between/grid/session_beh/**/honesty_runs.csv
wc -l results/between/grid/session_beh/**/iat_runs.csv
```

Expected row counts (excluding header):

| Sweep | Induction | Runs per variant | TPB variants per task | Total rows |
|-------|-----------|-----------------|----------------------|------------|
| SR (TPB) | Grid | 297 (27 × 11) | 2 | 594/task |
| SR (TPB) | Personas | 330 (30 × 11) | 2 | 660/task |
| SR (Big5) | Grid | 297 | 1 | 297 |
| SR (Big5) | Personas | 330 | 1 | 330 |
| Behavior | Grid | 297 | 1 | 297/task |
| Behavior | Personas | 330 | 1 | 330/task |

### Check for error rows

Runs that fail are still written to the CSV with an `error` column. To find them:

```bash
# Check for failed runs in any behavioral sweep
python3 -c "
import pandas as pd, glob, sys
for f in glob.glob('results/**/*_runs.csv', recursive=True):
    df = pd.read_csv(f, on_bad_lines='skip')
    if 'error' in df.columns:
        bad = df[df['error'].notna() & (df['error'] != '')]
        if len(bad):
            print(f'{f}: {len(bad)} error rows / {len(df)} total')
"
```

### Check per-model coverage

```bash
# Example: see which models are present in a given runs CSV
python3 -c "
import pandas as pd
df = pd.read_csv('results/between/grid/session_beh/sycophancy_psycohere_grid/neutral_sycophancy/sycophancy_runs.csv')
print(df.groupby('model_key').size().reset_index(name='n_runs').to_string(index=False))
"
```

### Resume a stalled sweep

All sweeps support `--resume`, which reads the existing output CSV and skips any run whose `(model_key, seed, temperature, top_p, persona_label, variant_id)` key is already present. Simply re-run the original command with `--resume` appended and it will pick up where it left off.

---

## Stage 2 — Merging (`merging/`)

Merging scripts join the between-session SR and behavioral CSVs on shared condition keys (`model_key`, `seed`, `temperature`, `top_p`, `persona_label`). They write ready-for-analysis CSVs to `results/merged/`.

### `merge_tpb_with_behavior.py` — TPB × any task

```bash
# TPB × CCT (grid)
python scripts/merging/merge_tpb_with_behavior.py \
  --config configs/merge/merge_cct_tpb_psycohere.json \
  --tpb_root results/between/grid/session_sr/tpb_cct_psycohere_grid \
  --behavior_runs_csv results/between/grid/session_beh/cct-psycohere-grid/cct-neutral/cct_runs.csv \
  --out_prefix results/merged/between/grid/tpb_x_cct \
  --tpb_runs_filename tpb_likert_runs.csv

# TPB × Sycophancy (grid)
python scripts/merging/merge_tpb_with_behavior.py \
  --config configs/merge/merge_sycophancy_tpb_psycohere.json \
  --tpb_root results/between/grid/session_sr/tpb_sycophancy_psycohere_grid \
  --behavior_runs_csv results/between/grid/session_beh/sycophancy_psycohere_grid/neutral_sycophancy/sycophancy_runs.csv \
  --out_prefix results/merged/between/grid/tpb_x_sycophancy \
  --tpb_runs_filename tpb_likert_runs.csv

# TPB × Honesty (grid)
python scripts/merging/merge_tpb_with_behavior.py \
  --config configs/merge/merge_honesty_tpb_psycohere.json \
  --tpb_root results/between/grid/session_sr/tpb_honesty_psycohere_grid \
  --behavior_runs_csv results/between/grid/session_beh/honesty_psycohere_grid/neutral-honesty/honesty_runs.csv \
  --out_prefix results/merged/between/grid/tpb_x_honesty \
  --tpb_runs_filename tpb_likert_runs.csv

# TPB × IAT (grid)
python scripts/merging/merge_tpb_with_behavior.py \
  --config configs/merge/merge_iat_tpb_psycohere.json \
  --tpb_root results/between/grid/session_sr/tpb_iat_psycohere_grid \
  --behavior_runs_csv results/between/grid/session_beh/iat-psycohere-grid/iat-neutral/iat_runs.csv \
  --out_prefix results/merged/between/grid/tpb_x_iat \
  --tpb_runs_filename tpb_likert_runs.csv \
  --tpb_status_ok_only \
  --behavior_where "pd.to_numeric(coverage, errors='coerce') >= 0.5"
```

Replace `grid` → `personas` and `_grid` → `_personas` in all paths for the persona induction variants.

### `merge_selfreport_*.py` — Big Five × task

```bash
# Big5 × CCT (grid)
python scripts/merging/merge_selfreport_cct.py \
  --selfreport_csv results/between/grid/session_sr/big5_psycohere_grid/big5/tpb_likert_runs.csv \
  --instrument big5 \
  --cct_runs_csv results/between/grid/session_beh/cct-psycohere-grid/cct-neutral/cct_runs.csv \
  --out_csv results/merged/between/grid/big5_x_cct.csv

# Big5 × Sycophancy (grid)
python scripts/merging/merge_selfreport_sycophancy.py \
  --selfreport_csv results/between/grid/session_sr/big5_psycohere_grid/big5/tpb_likert_runs.csv \
  --instrument big5 \
  --sycophancy_runs_csv results/between/grid/session_beh/sycophancy_psycohere_grid/neutral_sycophancy/sycophancy_runs.csv \
  --out_csv results/merged/between/grid/big5_x_sycophancy.csv

# Big5 × Honesty (grid)
python scripts/merging/merge_selfreport_honesty.py \
  --selfreport_csv results/between/grid/session_sr/big5_psycohere_grid/big5/tpb_likert_runs.csv \
  --instrument big5 \
  --honesty_runs_csv results/between/grid/session_beh/honesty_psycohere_grid/neutral-honesty/honesty_runs.csv \
  --out_csv results/merged/between/grid/big5_x_honesty.csv

# Big5 × IAT (grid)
python scripts/merging/merge_selfreport_iat.py \
  --selfreport_csv results/between/grid/session_sr/big5_psycohere_grid/big5/tpb_likert_runs.csv \
  --instrument big5 \
  --iat_runs_csv results/between/grid/session_beh/iat-psycohere-grid/iat-neutral/iat_runs.csv \
  --out_csv results/merged/between/grid/big5_x_iat.csv \
  --agg orders
```

---

## Stage 2.5 — Task Analysis: Produce `*_master.csv` files (`analysis/`)

Before the RQ scripts can run, each task needs its own master CSV built from all data sources (between/within × grid/personas × TPB/Big5). These four scripts do that — they also produce per-task correlations, contrasts, and a human-readable summary.

**All four take the same two arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--results_root` | `results` | Root of the results tree (contains `between/`, `within/`, `merged/`) |
| `--out_dir` | `results/analysis/<task>` | Where to write outputs |

**Outputs per task:**

| Script | Master CSV | Also writes |
|--------|-----------|-------------|
| `analyze_cct_psycohere.py` | `cct_master.csv` | `cct_correlations.csv`, `cct_contrasts.csv`, `cct_summary.txt` |
| `analyze_sycophancy_psycohere.py` | `sycophancy_master.csv` | `sycophancy_correlations.csv`, `sycophancy_contrasts.csv`, `sycophancy_summary.txt` |
| `analyze_honesty_psycohere.py` | `honesty_master.csv` | `honesty_correlations.csv`, `honesty_contrasts.csv`, `honesty_summary.txt` |
| `analyze_iat_psycohere.py` | `iat_master.csv` | `iat_correlations.csv`, `iat_contrasts.csv`, `iat_ceiling_by_test.csv`, `iat_master_top3.csv`, `iat_summary.txt` |

```bash
python scripts/analysis/analyze_cct_psycohere.py \
  --results_root results \
  --out_dir results/analysis/cct

python scripts/analysis/analyze_sycophancy_psycohere.py \
  --results_root results \
  --out_dir results/analysis/sycophancy

python scripts/analysis/analyze_honesty_psycohere.py \
  --results_root results \
  --out_dir results/analysis/honesty

python scripts/analysis/analyze_iat_psycohere.py \
  --results_root results \
  --out_dir results/analysis/iat
```

The IAT script also accepts two optional flags:

```bash
python scripts/analysis/analyze_iat_psycohere.py \
  --results_root results \
  --out_dir results/analysis/iat \
  --top_n_tests 3 \          # how many top-variance IAT tests to use in sub-analysis (default: 3)
  --ceiling_threshold 0.95   # threshold for "pegged at ceiling" (default: 0.95)
```

> **Note:** IAT bias exhibits a near-ceiling effect for most models (pooled M ≈ 0.88–0.91). The script automatically computes a `top_n_tests` sub-analysis restricted to the highest-variance tests, and writes `iat_master_top3.csv` as a drop-in replacement for the RQ figure scripts when you want to exclude ceiling-dominated tests.

Each `*_summary.txt` file contains a full human-readable breakdown of policy contrasts, SR–behavior correlations by framework × session × perturbation, and RQ3 within-vs-between comparisons — useful for a quick sanity check before running the RQ figure scripts.

---

## Stage 3 — Analysis (`analysis/`)

Each script produces the figures and statistics for one research question. All four require `*_master.csv` files produced by the task-level analysis scripts (see `results/analysis/`).

```bash
# RQ1 — SR–behavior coherence (main result)
python scripts/analysis/rq1_alignment_analysis.py \
  --cct_master     results/analysis/cct/cct_master.csv \
  --syc_master     results/analysis/sycophancy/sycophancy_master.csv \
  --honesty_master results/analysis/honesty/honesty_master.csv \
  --iat_master     results/analysis/iat/iat_master.csv \
  --out_dir        results/analysis/rq1_alignment

# RQ2 — Framework comparison (TPB vs Big Five)
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

### Diagnostics

```bash
# Variance and floor/ceiling analysis
python scripts/analysis/analyze_variance_and_floor_ceiling.py \
  --root_dir results \
  --session within --induction grid \
  --out_dir results/analysis/variance_diagnostics/within_grid

python scripts/analysis/analyze_variance_and_floor_ceiling.py \
  --root_dir results \
  --session between --induction personas \
  --out_dir results/analysis/variance_diagnostics/between_personas

# Instruction sensitivity (how much do SR scores shift across system prompts?)
python scripts/analysis/instruction_sensitivity.py \
  --within_root  results/within/grid \
  --between_root results/merged/between/grid \
  --layout 1x4 --sr_construct intention \
  --out_dir results/analysis/instruction_sensitivity/grid

python scripts/analysis/instruction_sensitivity.py \
  --within_root  results/within/personas \
  --between_root results/merged/between/personas \
  --layout 1x4 --sr_construct intention \
  --out_dir results/analysis/instruction_sensitivity/personas
```

---

## Stage 4 — Helper Visualizations (`helper/`)

Per-model behavioral fingerprint plots and SR–behavior scatter plots. These are for exploration and supplementary figures.

```bash
# Per-model fingerprint plots (radar charts across tasks)
python scripts/helper/plot_per_model_fingerprints.py \
  --within_root  results/within \
  --between_root results/between \
  --out_dir      results/analysis/per_model_fingerprints \
  --band ci95

# TPB × behavior scatter (between-session)
python scripts/helper/scatter_sr_behavior.py \
  --within_root results/within/grid \
  --induction_label grid \
  --out_dir results/analysis/scatter_sr_behavior/grid

python scripts/helper/scatter_sr_behavior.py \
  --within_root results/within/personas \
  --induction_label personas \
  --out_dir results/analysis/scatter_sr_behavior/personas

# Big5 × behavior scatter (between-session)
python scripts/helper/scatter_big5_behavior.py \
  --within_root results/within/grid \
  --induction_label grid \
  --out_dir results/analysis/scatter_big5_behavior/grid

# Within-session TPB scatter
python scripts/helper/scatter_tpb_within.py \
  --within_root results/within/grid \
  --induction_label grid \
  --out_dir results/analysis/scatter_tpb_within/grid

# Within-session Big5 scatter
python scripts/helper/scatter_big5_within.py \
  --within_root results/within/grid \
  --induction_label grid \
  --out_dir results/analysis/scatter_big5_within/grid
```
