#!/usr/bin/env python3
"""
rq1_alignment_analysis.py
=========================
RQ1: Shared-context SR-behavior coherence under maximally favorable conditions.

Best-case setup:
  - Session type: WITHIN-session (SR and behavior share the same context window)
  - Framework:    TPB (fine-grained, task-specific)
  - Perturbation: GRID (parameter-only induction — isolates measurement
                  coupling from identity-induction confounds)

Headline question: under conditions that most favor coherence emergence, how
often does TPB SR actually predict behavior in the theoretically expected
direction? How often does it reach significance?

Computation
-----------
For each of 11 models × 4 tasks × 2 constructs (Intention + task-specific
theoretically-strongest), compute the within-model Pearson r between the TPB
construct and align_score (the per-policy sign-corrected behavioral outcome).

Outputs 88 cells for the main headline + 176 cells (all 4 TPB constructs) for
the appendix.

Usage
-----
python scripts/analysis_scripts/rq1_alignment_analysis.py \\
    --cct_master     results/psycohere_v1/analysis/cct/cct_master.csv \\
    --syc_master     results/psycohere_v1/analysis/sycophancy/sycophancy_master.csv \\
    --honesty_master results/psycohere_v1/analysis/honesty/honesty_master.csv \\
    --iat_master     results/psycohere_v1/analysis/iat/iat_master.csv \\
    --out_dir        results/psycohere_v1/analysis/rq1_alignment
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patheffects as path_effects

# Local config
import sys
sys.path.insert(0, str(Path(__file__).parent))
from rq_config import (
    MODEL_ORDER, MODEL_LABELS, COLORS,
    TASK_ORDER, TASK_LABELS,
    TPB_CONSTRUCTS, INTENTION_COL, PRIMARY_CONSTRUCT,
    load_task_tpb, pearson_ci, safe_pearsonr, stars, wilson_ci,
    fisher_z_mean_ci, mundlak_within_between, policy_contrast_r,
)

from psycohere_style import (
    apply_style, C, FS, BAR, HEAT,
    style_ax, style_heatmap_ax, add_zero_line,
    panel_title, suptitle as fig_suptitle,
    annotate_bar, heatmap_cell_text,
)
apply_style()


# ── Cell-level computation ─────────────────────────────────────────────────

def compute_cells(data: dict, construct_set: str = "headline") -> pd.DataFrame:
    """Compute per-cell alignment statistics.

    data: dict with keys in TASK_ORDER mapping to DataFrames filtered to
          within/tpb/grid (one row per condition × policy).
    construct_set:
      "headline" → Intention + task-specific primary construct (88 cells)
      "all"      → all 4 TPB constructs (176 cells)
    """
    rows = []
    for task in TASK_ORDER:
        df = data[task]
        # Which constructs to compute for this task?
        if construct_set == "headline":
            primary_col, primary_label = PRIMARY_CONSTRUCT[task]
            # De-dupe when primary == Intention (IAT case)
            constructs = [(INTENTION_COL, "Intention")]
            if primary_col != INTENTION_COL:
                constructs.append((primary_col, primary_label))
        else:  # "all"
            constructs = TPB_CONSTRUCTS

        for construct_col, construct_label in constructs:
            is_primary = (construct_col == PRIMARY_CONSTRUCT[task][0])
            for m in MODEL_ORDER:
                sub = df[df.model_key == m]
                r, p, n = safe_pearsonr(sub[construct_col], sub["align_score"])
                lo, hi = pearson_ci(r, n)
                # Theoretical direction: positive r = aligned for volitional tasks,
                # negative r = aligned for IAT (compensatory-effort prediction:
                # explicit anti-bias intention co-occurs with persistent implicit bias).
                expected_sign = -1 if task == "iat" else +1
                aligned_with_theory = (
                    (np.sign(r) == expected_sign) if not np.isnan(r) else np.nan
                )
                rows.append({
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "construct": construct_col,
                    "construct_label": construct_label,
                    "is_primary_construct": is_primary,
                    "is_intention": construct_col == INTENTION_COL,
                    "model": m,
                    "model_label": MODEL_LABELS[m],
                    "r": r, "p": p, "ci_lo": lo, "ci_hi": hi, "n": n,
                    "direction_correct": aligned_with_theory,
                    "significant": (p < 0.05) if not np.isnan(p) else np.nan,
                    "alignment_hit": (aligned_with_theory and (p < 0.05))
                                     if not (np.isnan(r) or np.isnan(p)) else np.nan,
                })
    return pd.DataFrame(rows)


def compute_between_model(data: dict) -> pd.DataFrame:
    """Between-model r: correlate model-mean SR with model-mean align_score.
    Computed per (task, construct) for headline construct set.
    """
    rows = []
    for task in TASK_ORDER:
        df = data[task]
        primary_col, primary_label = PRIMARY_CONSTRUCT[task]
        constructs = [(INTENTION_COL, "Intention")]
        if primary_col != INTENTION_COL:
            constructs.append((primary_col, primary_label))

        for construct_col, construct_label in constructs:
            model_means = df.groupby("model_key").agg(
                x=(construct_col, "mean"),
                y=("align_score", "mean"),
            ).dropna()
            r, p, n = safe_pearsonr(model_means["x"], model_means["y"])
            lo, hi = pearson_ci(r, n)
            rows.append({
                "task": task, "task_label": TASK_LABELS[task],
                "construct": construct_col, "construct_label": construct_label,
                "is_primary_construct": (construct_col == primary_col),
                "r": r, "p": p, "ci_lo": lo, "ci_hi": hi, "n_models": n,
                "direction_correct": (r > 0) if not np.isnan(r) else np.nan,
                "significant": (p < 0.05) if not np.isnan(p) else np.nan,
            })
    return pd.DataFrame(rows)


def compute_mundlak(data: dict) -> pd.DataFrame:
    """Pooled OLS with Mundlak within/between decomposition + cluster-robust SEs.

    Complementary to the per-model Fisher-z aggregation: where Fisher-z gives
    you the meta-analytic mean of 11 separately-estimated within-model slopes,
    Mundlak estimates a SINGLE pooled within-model slope across the full
    dataset, assuming homogeneity. The two should give similar β_within
    values for cells where the per-model heterogeneity is not extreme.

    Runs per (task, construct) — both Intention + theoretically-primary.
    Outputs are standardized (z-scored) so β is comparable to Pearson r.
    """
    rows = []
    for task in TASK_ORDER:
        df = data[task]
        primary_col, primary_label = PRIMARY_CONSTRUCT[task]
        constructs = [(INTENTION_COL, "Intention")]
        if primary_col != INTENTION_COL:
            constructs.append((primary_col, primary_label))

        for construct_col, construct_label in constructs:
            m = mundlak_within_between(df, construct_col, "align_score",
                                         group_col="model_key",
                                         z_normalize=True)
            rows.append({
                "task": task, "task_label": TASK_LABELS[task],
                "construct": construct_col, "construct_label": construct_label,
                "is_primary_construct": (construct_col == primary_col),
                **m,
            })
    return pd.DataFrame(rows)


def compute_policy_contrast(data: dict) -> pd.DataFrame:
    """Policy-contrast specification for each task's Intention + primary
    construct. For tasks with two opposing policies, the difference-score
    approach removes individual response-style variance (a model that always
    says "5" on Likert has its additive offset cancelled).

    See Armitage & Conner (2001) for the rationale.
    """
    rows = []
    for task in TASK_ORDER:
        df = data[task]
        primary_col, primary_label = PRIMARY_CONSTRUCT[task]
        constructs = [(INTENTION_COL, "Intention")]
        if primary_col != INTENTION_COL:
            constructs.append((primary_col, primary_label))

        for construct_col, construct_label in constructs:
            # Use raw behavioral outcome per-task — NOT align_score, because
            # align_score is itself already per-policy sign-corrected (which
            # defeats the purpose of the contrast). We want the raw outcome
            # so the difference score preserves the intended theoretical
            # direction.
            beh_col_map = {
                "cct": "mean_k_norm",
                "syc": "sycophancy_rate",
                "honesty": "align_score",  # Honesty lacks a universal raw y
                "iat": "mean_bias_score",
            }
            beh_col = beh_col_map[task]
            if beh_col not in df.columns:
                # Fallback to align_score if raw missing
                beh_col = "align_score"
            res = policy_contrast_r(df, construct_col, beh_col,
                                      policy_col="policy_id",
                                      group_col="model_key")
            lo, hi = pearson_ci(res["r"], res["n_pairs"])
            rows.append({
                "task": task, "task_label": TASK_LABELS[task],
                "construct": construct_col, "construct_label": construct_label,
                "is_primary_construct": (construct_col == primary_col),
                "y_col": beh_col,
                "r_contrast": res["r"], "p_contrast": res["p"],
                "ci_lo": lo, "ci_hi": hi, "n_pairs": res["n_pairs"],
                "policies": " vs ".join(res["policies"]) if res["policies"] else "",
            })
    return pd.DataFrame(rows)


# ── Aggregations ──────────────────────────────────────────────────────────

def aggregate_by_task(cells: pd.DataFrame) -> pd.DataFrame:
    """Per-task × construct summary with:
    - Fisher-z-aggregated mean r with 95% CI (effect-size metric, uses all
      underlying observations)
    - Wilson 95% CI on alignment proportion (cell-count metric)
    """
    # Cast bool columns to int so groupby sum returns integer counts
    cells = cells.copy()
    for col in ("direction_correct", "significant", "alignment_hit"):
        cells[col] = cells[col].astype("Int64")
    out = (cells.groupby(["task_label", "construct_label",
                          "is_primary_construct", "is_intention"])
                .agg(n_cells=("r", "size"),
                     n_direction=("direction_correct", "sum"),
                     n_significant=("significant", "sum"),
                     n_alignment=("alignment_hit", "sum"),
                     mean_r=("r", "mean"),  # simple mean, for reference
                     median_r=("r", "median"))
                .reset_index())
    out["pct_direction"] = 100 * out["n_direction"] / out["n_cells"]
    out["pct_alignment"] = 100 * out["n_alignment"] / out["n_cells"]

    # Fisher-z weighted mean r with CI per cell-group
    fz_stats = []
    for _, row in out.iterrows():
        grp = cells[(cells["task_label"] == row["task_label"]) &
                    (cells["construct_label"] == row["construct_label"])]
        mr, lo, hi, k = fisher_z_mean_ci(grp["r"].values, grp["n"].values)
        fz_stats.append((mr, lo, hi, k))
    out["fz_mean_r"] = [s[0] for s in fz_stats]
    out["fz_ci_lo"] = [s[1] for s in fz_stats]
    out["fz_ci_hi"] = [s[2] for s in fz_stats]
    out["fz_n_cells_used"] = [s[3] for s in fz_stats]

    # Wilson CIs on proportions
    ci_dir = out.apply(lambda r: wilson_ci(int(r["n_direction"]),
                                            int(r["n_cells"])), axis=1)
    ci_align = out.apply(lambda r: wilson_ci(int(r["n_alignment"]),
                                              int(r["n_cells"])), axis=1)
    out["dir_ci_lo"] = [100 * c[0] for c in ci_dir]
    out["dir_ci_hi"] = [100 * c[1] for c in ci_dir]
    out["align_ci_lo"] = [100 * c[0] for c in ci_align]
    out["align_ci_hi"] = [100 * c[1] for c in ci_align]
    return out


def aggregate_by_model(cells: pd.DataFrame) -> pd.DataFrame:
    """Per-model summary with Fisher-z mean r + 95% CI AND Wilson CIs on
    proportions. Sorted by Fisher-z mean r descending."""
    # Cast bool columns to int so groupby sum returns integer counts (not bool dtype)
    cells = cells.copy()
    for col in ("direction_correct", "significant", "alignment_hit"):
        cells[col] = cells[col].astype("Int64")
    out = (cells.groupby(["model", "model_label"])
                .agg(n_cells=("r", "size"),
                     n_direction=("direction_correct", "sum"),
                     n_significant=("significant", "sum"),
                     n_alignment=("alignment_hit", "sum"),
                     mean_r=("r", "mean"))
                .reset_index())
    out["pct_direction"] = 100 * out["n_direction"] / out["n_cells"]
    out["pct_alignment"] = 100 * out["n_alignment"] / out["n_cells"]

    # Fisher-z weighted mean r with 95% CI, per model
    fz_stats = []
    for _, row in out.iterrows():
        grp = cells[cells["model"] == row["model"]]
        mr, lo, hi, k = fisher_z_mean_ci(grp["r"].values, grp["n"].values)
        fz_stats.append((mr, lo, hi, k))
    out["fz_mean_r"] = [s[0] for s in fz_stats]
    out["fz_ci_lo"] = [s[1] for s in fz_stats]
    out["fz_ci_hi"] = [s[2] for s in fz_stats]
    out["fz_n_cells_used"] = [s[3] for s in fz_stats]

    # Wilson CIs on proportions (for reference alongside Fisher-z CIs)
    ci_dir = out.apply(lambda r: wilson_ci(int(r["n_direction"]),
                                            int(r["n_cells"])), axis=1)
    ci_align = out.apply(lambda r: wilson_ci(int(r["n_alignment"]),
                                              int(r["n_cells"])), axis=1)
    out["dir_ci_lo"] = [100 * c[0] for c in ci_dir]
    out["dir_ci_hi"] = [100 * c[1] for c in ci_dir]
    out["align_ci_lo"] = [100 * c[0] for c in ci_align]
    out["align_ci_hi"] = [100 * c[1] for c in ci_align]

    # Sort by Fisher-z mean r (effect-size ranking, most informative)
    out = out.sort_values(["fz_mean_r", "n_alignment"],
                          ascending=[False, False]).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out


# ── Printout ──────────────────────────────────────────────────────────────

def print_headline(cells: pd.DataFrame,
                   by_task: pd.DataFrame,
                   by_model: pd.DataFrame,
                   btwn: pd.DataFrame) -> dict:
    """Print a human-readable headline + return structured JSON.

    Leads with Fisher-z-aggregated mean r (effect size, primary evidence)
    then reports proportion-of-cells metrics with Wilson CIs and the null
    baseline (under r=0 at each cell, expected hit rate ≈ 2.5%).
    """
    total_cells = len(cells)
    # Overall Fisher-z aggregated mean r across all cells
    overall_mr, overall_lo, overall_hi, n_used = fisher_z_mean_ci(
        cells["r"].values, cells["n"].values)
    # Excluding IAT (theoretical boundary)
    non_iat = cells[cells["task"] != "iat"]
    noniat_mr, noniat_lo, noniat_hi, _ = fisher_z_mean_ci(
        non_iat["r"].values, non_iat["n"].values)

    n_dir = int(cells["direction_correct"].sum())
    n_sig = int(cells["significant"].sum())
    n_align = int(cells["alignment_hit"].sum())
    dir_lo, dir_hi = wilson_ci(n_dir, total_cells)
    sig_lo, sig_hi = wilson_ci(n_sig, total_cells)
    align_lo, align_hi = wilson_ci(n_align, total_cells)

    # Null baseline: under r=0 population for every cell, each cell has
    # P(r>0 AND p<.05) = alpha/2 = 2.5% (positive tail). With 77 cells,
    # expected hits ≈ 77 × 0.025 ≈ 1.9 cells.
    null_rate = 0.025
    expected_null = total_cells * null_rate
    # Binomial z-test for observed vs null
    null_se = np.sqrt(total_cells * null_rate * (1 - null_rate))
    null_z = (n_align - expected_null) / null_se
    fold_over_null = n_align / expected_null if expected_null > 0 else np.nan

    print("\n" + "=" * 78)
    print("RQ1 HEADLINE — Within-session · TPB · grid perturbation")
    print("Best-case SR-behavior coherence test (priming condition)")
    print("=" * 78)
    print(f"\nTotal cells: {total_cells}  "
          f"(11 models × 4 tasks × 2 constructs, de-duped when primary == Intention)")

    print(f"\n  ━━ PRIMARY EFFECT-SIZE EVIDENCE (Fisher-z aggregated) ━━")
    print(f"    Mean within-model r, ALL tasks:      "
          f"{overall_mr:+.3f}  95% CI [{overall_lo:+.3f}, {overall_hi:+.3f}]")
    print(f"    Mean within-model r, ex-IAT:         "
          f"{noniat_mr:+.3f}  95% CI [{noniat_lo:+.3f}, {noniat_hi:+.3f}]")
    print(f"    ▸ Excluding IAT (implicit, outside TPB's theoretical scope:")
    print(f"      TPB predicts deliberative/volitional behavior only)")

    print(f"\n  ━━ PROPORTION-OF-CELLS METRICS (Wilson 95% CIs) ━━")
    print(f"    Direction-correct (r > 0):           "
          f"{n_dir}/{total_cells} ({100*n_dir/total_cells:.1f}%)  "
          f"[{100*dir_lo:.1f}%, {100*dir_hi:.1f}%]")
    print(f"    Significant (p < .05, either sign):  "
          f"{n_sig}/{total_cells} ({100*n_sig/total_cells:.1f}%)  "
          f"[{100*sig_lo:.1f}%, {100*sig_hi:.1f}%]")
    print(f"    Alignment hit (correct AND sig):     "
          f"{n_align}/{total_cells} ({100*n_align/total_cells:.1f}%)  "
          f"[{100*align_lo:.1f}%, {100*align_hi:.1f}%]")
    print(f"\n  ━━ NULL BASELINE (Expected if r=0 at every cell) ━━")
    print(f"    Under pure null, each cell's P(r>0 AND p<.05) = α/2 = 2.5%")
    print(f"    Expected alignment hits: {expected_null:.1f}/{total_cells}  "
          f"Observed: {n_align}/{total_cells}  "
          f"Fold-over-null: {fold_over_null:.1f}×  z = {null_z:.1f}")

    print("\n" + "-" * 78)
    print("PER-TASK BREAKDOWN (Fisher-z mean r with CI, then alignment rate)")
    print("-" * 78)
    print(f"{'Task':<12}  {'Construct':<18}  {'Fisher-z r [95% CI]':<30}  "
          f"{'align %':<12}")
    for _, row in by_task.iterrows():
        marker = "★" if row["is_primary_construct"] else " "
        fz_str = f"{row['fz_mean_r']:+.2f} [{row['fz_ci_lo']:+.2f}, {row['fz_ci_hi']:+.2f}]"
        ci = f"({row['align_ci_lo']:.0f}–{row['align_ci_hi']:.0f}%)"
        print(f"{marker} {row['task_label']:<10}  {row['construct_label']:<18}  "
              f"{fz_str:<30}  "
              f"{int(row['n_alignment'])}/{int(row['n_cells'])} "
              f"({row['pct_alignment']:.0f}%) {ci}")
    print("  ★ = theoretically-motivated primary construct")

    print("\n" + "-" * 78)
    print("PER-MODEL RANKING (sorted by Fisher-z mean r)")
    print("-" * 78)
    for _, row in by_model.iterrows():
        bar = "█" * int(row["n_alignment"]) + "·" * (int(row["n_direction"]) - int(row["n_alignment"]))
        bar = bar.ljust(int(row["n_cells"]), " ")
        fz_str = f"{row['fz_mean_r']:+.2f} [{row['fz_ci_lo']:+.2f}, {row['fz_ci_hi']:+.2f}]"
        print(f"  #{row['rank']:>2}  {row['model_label']:<22}  "
              f"r = {fz_str:<24}  "
              f"align {int(row['n_alignment'])}/{int(row['n_cells'])}  {bar}")
    print("  █ = alignment hit (correct direction + significant)")
    print("  · = correct direction but not significant")

    print("\n" + "-" * 78)
    print("BETWEEN-MODEL COHERENCE (n=11 model means; Fisher-z 95% CIs)")
    print("-" * 78)
    print(f"{'Task':<12}  {'Construct':<18}  {'btwn r':<8}  {'95% CI':<22}  {'p':<8}")
    for _, row in btwn.iterrows():
        marker = "★" if row["is_primary_construct"] else " "
        ci = f"[{row['ci_lo']:+.2f}, {row['ci_hi']:+.2f}]" \
             if not np.isnan(row["ci_lo"]) else "[nan, nan]"
        print(f"{marker} {row['task_label']:<10}  {row['construct_label']:<18}  "
              f"{row['r']:+.3f}   {ci:<22}  p={row['p']:.3f}{stars(row['p'])}")

    # Build headline JSON
    primary_only = by_task[by_task["is_primary_construct"]]
    strongest_row = primary_only.loc[primary_only["fz_mean_r"].idxmax()]
    weakest_row = primary_only.loc[primary_only["fz_mean_r"].idxmin()]

    return {
        "n_cells_total": total_cells,
        "fisher_z_mean_r_all": round(overall_mr, 4),
        "fisher_z_mean_r_all_ci_95": [round(overall_lo, 4), round(overall_hi, 4)],
        "fisher_z_mean_r_ex_iat": round(noniat_mr, 4),
        "fisher_z_mean_r_ex_iat_ci_95": [round(noniat_lo, 4), round(noniat_hi, 4)],
        "n_direction_correct": n_dir,
        "pct_direction_correct": round(100 * n_dir / total_cells, 2),
        "direction_ci_95": [round(100 * dir_lo, 2), round(100 * dir_hi, 2)],
        "n_significant": n_sig,
        "pct_significant": round(100 * n_sig / total_cells, 2),
        "significant_ci_95": [round(100 * sig_lo, 2), round(100 * sig_hi, 2)],
        "n_alignment_hit": n_align,
        "pct_alignment_hit": round(100 * n_align / total_cells, 2),
        "alignment_ci_95": [round(100 * align_lo, 2), round(100 * align_hi, 2)],
        "null_baseline_expected_hits": round(expected_null, 2),
        "fold_over_null": round(fold_over_null, 1),
        "null_z": round(null_z, 2),
        "strongest_task": strongest_row["task_label"],
        "strongest_task_construct": strongest_row["construct_label"],
        "strongest_task_fz_r": round(float(strongest_row["fz_mean_r"]), 3),
        "strongest_task_fz_ci": [round(float(strongest_row["fz_ci_lo"]), 3),
                                  round(float(strongest_row["fz_ci_hi"]), 3)],
        "strongest_task_alignment_pct": round(float(strongest_row["pct_alignment"]), 1),
        "weakest_task": weakest_row["task_label"],
        "weakest_task_construct": weakest_row["construct_label"],
        "weakest_task_fz_r": round(float(weakest_row["fz_mean_r"]), 3),
        "weakest_task_fz_ci": [round(float(weakest_row["fz_ci_lo"]), 3),
                                round(float(weakest_row["fz_ci_hi"]), 3)],
        "weakest_task_alignment_pct": round(float(weakest_row["pct_alignment"]), 1),
        "most_aligned_model": by_model.iloc[0]["model_label"],
        "most_aligned_model_fz_r": round(float(by_model.iloc[0]["fz_mean_r"]), 3),
        "most_aligned_model_fz_ci": [round(float(by_model.iloc[0]["fz_ci_lo"]), 3),
                                      round(float(by_model.iloc[0]["fz_ci_hi"]), 3)],
        "most_aligned_model_align": int(by_model.iloc[0]["n_alignment"]),
        "least_aligned_model": by_model.iloc[-1]["model_label"],
        "least_aligned_model_fz_r": round(float(by_model.iloc[-1]["fz_mean_r"]), 3),
        "least_aligned_model_align": int(by_model.iloc[-1]["n_alignment"]),
    }


def print_mundlak_and_contrast(mundlak: pd.DataFrame, contrast: pd.DataFrame):
    """Print Mundlak OLS + policy-contrast tables. Separate from main
    headline since these are robustness/supplementary analyses."""
    print("\n" + "=" * 78)
    print("ROBUSTNESS — MUNDLAK POOLED OLS + CLUSTER-ROBUST SEs")
    print("β standardized (z-scored x, y) — directly comparable to Pearson r.")
    print("Cluster = model_key.")
    print("=" * 78)
    print(f"\n{'Task':<12}  {'Construct':<18}  "
          f"{'β_within [95% CI]':<26}  {'β_between [95% CI]':<26}")
    for _, row in mundlak.iterrows():
        marker = "★" if row["is_primary_construct"] else " "
        bw = (f"{row['beta_within']:+.3f} [{row['ci_within_lo']:+.2f},"
              f"{row['ci_within_hi']:+.2f}]{stars(row['p_within'])}")
        bb = (f"{row['beta_between']:+.3f} [{row['ci_between_lo']:+.2f},"
              f"{row['ci_between_hi']:+.2f}]{stars(row['p_between'])}")
        print(f"{marker} {row['task_label']:<10}  {row['construct_label']:<18}  "
              f"{bw:<26}  {bb:<26}")
    print("  ★ = theoretically-motivated primary construct")
    print(f"  N_obs range: {int(mundlak['n_obs'].min())}-{int(mundlak['n_obs'].max())}; "
          f"N_groups = {int(mundlak['n_groups'].iloc[0])} models")

    print("\n" + "=" * 78)
    print("ROBUSTNESS — POLICY-CONTRAST SPECIFICATION")
    print("r(Δx, Δy) where Δ = difference between opposing policies within")
    print("matched conditions. Removes response-style variance.")
    print("=" * 78)
    print(f"\n{'Task':<12}  {'Construct':<18}  {'r_contrast':<12}  "
          f"{'95% CI':<22}  {'n_pairs':<8}")
    for _, row in contrast.iterrows():
        marker = "★" if row["is_primary_construct"] else " "
        ci = (f"[{row['ci_lo']:+.2f},{row['ci_hi']:+.2f}]"
              if not np.isnan(row["ci_lo"]) else "[nan, nan]")
        r_str = (f"{row['r_contrast']:+.3f}{stars(row['p_contrast'])}"
                 if not np.isnan(row['r_contrast']) else "  nan")
        print(f"{marker} {row['task_label']:<10}  {row['construct_label']:<18}  "
              f"{r_str:<12}  {ci:<22}  n={int(row['n_pairs'])}")
    print("  ★ = theoretically-motivated primary construct")


# ── Figure (3-panel RQ1 summary) ──────────────────────────────────────────

def figure_rq1(cells: pd.DataFrame, by_model: pd.DataFrame,
               by_task: pd.DataFrame, out_dir: Path):
    """Three-panel RQ1 figure (coarse to fine).
    Panel A: per-task × construct Fisher-z mean r with 95% CI
    Panel B: per-model Fisher-z mean r with 95% CI (effect-size ranking)
    Panel C: per-cell heatmap of within-model r (all 77 cells)
    """
    fig = plt.figure(figsize=(22, 9.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.5],
                          wspace=0.30, left=0.06, right=0.98, top=0.90, bottom=0.12)
    # Panel B shares the same axes height as the figure, but the bar chart
    # needs its ylim clamped tightly — done after ax_b is built below.

    # Common ordering of models — sorted by per-model Fisher-z mean r (descending)
    models_sorted = by_model["model"].tolist()
    n_models = len(by_model)

    # ── Panel A: per-task Fisher-z mean r with CI (coarsest aggregation) ──
    ax_a = fig.add_subplot(gs[0, 0])
    task_rows = []
    for task in TASK_ORDER:
        label = TASK_LABELS[task]
        int_row = by_task[(by_task["task_label"] == label) &
                          (by_task["is_intention"])].iloc[0]
        pri_row = by_task[(by_task["task_label"] == label) &
                          (by_task["is_primary_construct"])].iloc[0]
        task_rows.append((task, label, int_row, pri_row))
    task_rows.sort(key=lambda t: t[3]["fz_mean_r"], reverse=True)

    y_pos_a = np.arange(len(task_rows))[::-1] * 1.3
    bar_h = BAR["height"]
    ytick_pos_a = []   # correct label position per task (midpoint of drawn bars)
    for i, (task, label, int_row, pri_row) in enumerate(task_rows):
        yp = y_pos_a[i]
        pr = pri_row["fz_mean_r"]; plo = pri_row["fz_ci_lo"]; phi = pri_row["fz_ci_hi"]
        ax_a.barh(yp + bar_h/2 + 0.06, pr, height=bar_h,
                   color=C["warm"], alpha=BAR["alpha"], edgecolor=C["bar_edge"], linewidth=BAR["edge_lw"], zorder=3)
        ax_a.errorbar(pr, yp + bar_h/2 + 0.06,
                       xerr=[[max(0, pr - plo)], [max(0, phi - pr)]],
                       fmt="none", ecolor=BAR["ecolor"], capsize=BAR["capsize"], lw=BAR["elinewidth"], zorder=4)
        pri_label_text = pri_row["construct_label"]
        pri_is_int = (pri_row["is_intention"] == True)
        if not pri_is_int:
            ir = int_row["fz_mean_r"]; ilo = int_row["fz_ci_lo"]; ihi = int_row["fz_ci_hi"]
            ax_a.barh(yp - bar_h/2 - 0.06, ir, height=bar_h,
                       color=C["cool"], alpha=BAR["alpha"], edgecolor=C["bar_edge"], linewidth=BAR["edge_lw"], zorder=3)
            ax_a.errorbar(ir, yp - bar_h/2 - 0.06,
                           xerr=[[max(0, ir - ilo)], [max(0, ihi - ir)]],
                           fmt="none", ecolor=BAR["ecolor"], capsize=BAR["capsize"], lw=BAR["elinewidth"], zorder=4)
            t1 = ax_a.text(0.01, yp + bar_h/2 + 0.06, f"  {pri_label_text}",
                      transform=ax_a.get_yaxis_transform(),
                      va="center", fontsize=FS["construct_tag"], color=C["warm_label"], fontweight="bold",
                      zorder=6)
            t1.set_path_effects([path_effects.withStroke(linewidth=4, foreground="white")])
            t2 = ax_a.text(0.01, yp - bar_h/2 - 0.06, "  Intention",
                      transform=ax_a.get_yaxis_transform(),
                      va="center", fontsize=FS["construct_tag"], color=C["cool_label"], fontweight="bold",
                      zorder=6)
            t2.set_path_effects([path_effects.withStroke(linewidth=4, foreground="white")])
            ytick_pos_a.append(yp)                    # midpoint between two bars
        else:
            t = ax_a.text(0.01, yp + bar_h/2 + 0.06, f"  {pri_label_text}",
                      transform=ax_a.get_yaxis_transform(),
                      va="center", fontsize=FS["construct_tag"], color=C["warm_label"], fontweight="bold",
                      zorder=6)
            t.set_path_effects([path_effects.withStroke(linewidth=4, foreground="white")])
            ytick_pos_a.append(yp + bar_h/2 + 0.06)  # single bar — label at bar centre
    # Full task names for Panel A y-axis (override rq_config short labels)
    TASK_FULL_LABELS = {
        "cct":     "Risk-taking\n(CCT)",
        "syc":     "Sycophancy",
        "honesty": "Honesty",
        "iat":     "Implicit bias\n(IAT)",
    }

    add_zero_line(ax_a, "v")
    ax_a.set_yticks(ytick_pos_a)
    ax_a.set_yticklabels([TASK_FULL_LABELS.get(t[0], t[1]) for t in task_rows],
                          fontsize=FS["tick"], fontweight="bold")
    ax_a.set_xlabel("Fisher-z aggregated mean r (95% CI)", fontsize=FS["axis_label"])
    ax_a.tick_params(axis="x", labelsize=FS["tick"])
    x_ext_a = max(0.75, float(np.nanmax([
        by_task["fz_ci_hi"].max(), -by_task["fz_ci_lo"].min()
    ])) + 0.1)
    ax_a.set_xlim(-x_ext_a, x_ext_a)
    panel_title(ax_a, "A. Per-task coherence (TPB)")
    style_ax(ax_a, grid_axis="x")

    # IAT inversion annotation — point to start (right-edge) of the IAT bar
    iat_idx = next((i for i, (t, *_) in enumerate(task_rows) if t == "iat"), None)
    if iat_idx is not None:
        iat_y = y_pos_a[iat_idx] + bar_h / 2 + 0.06  # vertical position of the Intention bar
        # Arrow points to (x=0, y=iat_y) — the start (right edge) of the leftward IAT bar
        # Text sits to the right of the bar, in the empty space above the y=0 line
        ax_a.annotate(
            "Inverted as\ntheoretically\nexpected",
            xy=(0.0, iat_y),
            xytext=(0.32, iat_y + 0.05),
            fontsize=FS["annotation"], color=C["annotation"], style="italic",
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color="#37474F",
                              lw=1.5, connectionstyle="arc3,rad=0.15"),
            zorder=10,
        )

    # ── Panel B: per-model Fisher-z mean r with CI ──
    ax_b = fig.add_subplot(gs[0, 1])
    y_pos = np.arange(n_models)[::-1]
    fz_r = by_model["fz_mean_r"].values
    fz_lo = by_model["fz_ci_lo"].values
    fz_hi = by_model["fz_ci_hi"].values
    n_cells_per_model = int(by_model["n_cells"].iloc[0])
    # Uniform colour — warm for positive bars, muted grey-taupe for negative
    # (avoids confusion with the cool/blue secondary-construct colour in Panel A)
    bar_color_b = [C["warm"] if r >= 0 else C["warm_neg"] for r in fz_r]

    err_lo = np.maximum(0, fz_r - fz_lo)
    err_hi = np.maximum(0, fz_hi - fz_r)
    ax_b.barh(y_pos, fz_r, height=BAR["height"],
              color=bar_color_b, alpha=BAR["alpha"], edgecolor=C["bar_edge"], linewidth=BAR["edge_lw"], zorder=2)
    ax_b.errorbar(fz_r, y_pos, xerr=[err_lo, err_hi],
                   fmt="none", ecolor=BAR["ecolor"], capsize=BAR["capsize"], lw=BAR["elinewidth"], zorder=4)

    # Mean Fisher-z r annotation — for positive-r bars place to LEFT of zero (so they don't intrude
    # into the next panel); for negative-r bars place to RIGHT of zero
    for i, (_, row) in enumerate(by_model.iterrows()):
        yp = y_pos[i]
        if fz_r[i] >= 0:
            x_text = -0.02
            ha = "right"
        else:
            x_text = 0.02
            ha = "left"
        ax_b.text(x_text, yp,
                  f"r = {fz_r[i]:+.2f} ",
                  va="center", ha=ha, fontsize=FS["bar_label"], color=C["annotation"],
                  fontweight="bold")

    add_zero_line(ax_b, "v")
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([MODEL_LABELS[m] for m in models_sorted], fontsize=FS["tick"])
    ax_b.set_xlabel("Fisher-z aggregated mean r (95% CI)", fontsize=FS["axis_label"])
    ax_b.tick_params(axis="x", labelsize=FS["tick"])
    x_ext = max(0.75, float(np.nanmax(np.abs([fz_lo.min(), fz_hi.max()]))) + 0.1)
    ax_b.set_xlim(-x_ext, x_ext)
    ax_b.set_ylim(-0.5, n_models - 0.5)   # match imshow row extent in Panel C
    panel_title(ax_b, "B. Per-model coherence (TPB)")
    style_ax(ax_b, grid_axis="x")

    # ── Panel C: per-cell heatmap (finest aggregation) ──
    ax_c = fig.add_subplot(gs[0, 2])
    # Abbreviated task / construct names for compact column headers
    TASK_ABBREV = {
        "cct":     "CCT",
        "syc":     "Syc.",
        "honesty": "Hon.",
        "iat":     "IAT",
    }
    CONSTRUCT_ABBREV = {
        "Intention":       "Intent.",
        "Attitude":        "Attitude",
        "Subjective Norm": "Subj. Norm",
        "PBC":             "PBC",
    }
    col_order = []
    col_labels = []
    for task in TASK_ORDER:
        primary_col, primary_label = PRIMARY_CONSTRUCT[task]
        task_abbr = TASK_ABBREV.get(task, TASK_LABELS[task])
        col_order.append((task, INTENTION_COL))
        col_labels.append(f"{task_abbr}\n{CONSTRUCT_ABBREV.get('Intention', 'Intention')}")
        if primary_col != INTENTION_COL:
            col_order.append((task, primary_col))
            col_labels.append(f"{task_abbr}\n{CONSTRUCT_ABBREV.get(primary_label, primary_label)}")

    n_cols = len(col_order)
    heat = np.full((n_models, n_cols), np.nan)
    sigmask = np.zeros((n_models, n_cols), dtype=bool)
    for j, (task, construct) in enumerate(col_order):
        for i, m in enumerate(models_sorted):
            row = cells[(cells["task"] == task) &
                        (cells["construct"] == construct) &
                        (cells["model"] == m)]
            if len(row):
                heat[i, j] = row["r"].iloc[0]
                p = row["p"].iloc[0]
                sigmask[i, j] = (p < 0.05) if not np.isnan(p) else False

    vmax = max(0.05, float(np.nanmax(np.abs(heat))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax_c.imshow(heat, aspect="auto", cmap=HEAT["cmap"], norm=norm)

    for i in range(n_models):
        for j in range(n_cols):
            val = heat[i, j]
            if np.isnan(val):
                continue
            heatmap_cell_text(ax_c, j, i, val, sig=sigmask[i, j])

    for sep in range(2, n_cols, 2):
        ax_c.axvline(sep - 0.5, color=C["spine"], lw=HEAT["task_sep_lw"], alpha=HEAT["task_sep_alpha"])

    ax_c.set_xticks(range(n_cols))
    ax_c.set_xticklabels(col_labels, fontsize=FS["tick"])
    ax_c.set_yticks(range(n_models))
    ax_c.set_yticklabels([MODEL_LABELS[m] for m in models_sorted], fontsize=FS["tick"])
    panel_title(ax_c, "C. Within-model Pearson r — TPB  (* = p < .05)")
    cbar = fig.colorbar(im, ax=ax_c, shrink=HEAT["cbar_shrink"], pad=HEAT["cbar_pad"])
    cbar.ax.tick_params(labelsize=FS["colorbar"])
    ax_c.tick_params(length=0)
    style_heatmap_ax(ax_c)

    # Sync Panel B vertical extent to Panel C so model-name rows align exactly.
    # We do this after both panels are fully built by matching their bbox heights.
    fig.canvas.draw()
    pos_b = ax_b.get_position()
    pos_c = ax_c.get_position()
    ax_b.set_position([pos_b.x0, pos_c.y0, pos_b.width, pos_c.height])

    fig_suptitle(fig, "RQ1 — Same-session: Does Self-Report↔Behavior coherence exist under best-case conditions?")

    for ext in ["pdf", "png"]:
        p = out_dir / f"rq1_alignment_summary.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  saved: {p}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cct_master",
                    default="results/psycohere_v1/analysis/cct/cct_master.csv")
    ap.add_argument("--syc_master",
                    default="results/psycohere_v1/analysis/sycophancy/sycophancy_master.csv")
    ap.add_argument("--honesty_master",
                    default="results/psycohere_v1/analysis/honesty/honesty_master.csv")
    ap.add_argument("--iat_master",
                    default="results/psycohere_v1/analysis/iat/iat_master.csv")
    ap.add_argument("--out_dir",
                    default="results/psycohere_v1/analysis/rq1_alignment")
    ap.add_argument("--out_suffix", default="",
                    help="Optional suffix on output filenames (e.g. '_top3' for "
                         "ceiling-corrected IAT paired with iat_master_top3.csv).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading TPB masters (filtered: within-session, grid perturbation)...")
    data = {
        "cct":     load_task_tpb(args.cct_master,     "cct",     "within", "grid"),
        "syc":     load_task_tpb(args.syc_master,     "syc",     "within", "grid"),
        "honesty": load_task_tpb(args.honesty_master, "honesty", "within", "grid"),
        "iat":     load_task_tpb(args.iat_master,     "iat",     "within", "grid"),
    }
    print("\nData coverage (after filtering):")
    for task, df in data.items():
        n_models = df.model_key.nunique()
        print(f"  {task:<10}  {len(df):>5} rows  ({n_models} models, "
              f"~{len(df)//max(1,n_models)}/model)")

    # Headline (88 cells: Intention + primary per task)
    cells_headline = compute_cells(data, construct_set="headline")
    by_task = aggregate_by_task(cells_headline)
    by_model = aggregate_by_model(cells_headline)
    btwn = compute_between_model(data)

    # Appendix (176 cells: all 4 TPB constructs per task)
    cells_all = compute_cells(data, construct_set="all")
    by_task_all = aggregate_by_task(cells_all)

    # Save CSVs
    suffix = args.out_suffix
    cells_headline.to_csv(out_dir / f"rq1_alignment_cells{suffix}.csv", index=False)
    by_task.to_csv(out_dir / f"rq1_alignment_by_task{suffix}.csv", index=False)
    by_model.to_csv(out_dir / f"rq1_alignment_by_model{suffix}.csv", index=False)
    btwn.to_csv(out_dir / f"rq1_alignment_between_model{suffix}.csv", index=False)
    cells_all.to_csv(out_dir / f"rq1_alignment_cells_all_constructs{suffix}.csv",
                      index=False)
    by_task_all.to_csv(out_dir / f"rq1_alignment_by_task_all_constructs{suffix}.csv",
                        index=False)

    # Robustness analyses: Mundlak pooled OLS + policy-contrast specification
    print("\nComputing Mundlak OLS with cluster-robust SEs (robustness)...")
    mundlak = compute_mundlak(data)
    mundlak.to_csv(out_dir / f"rq1_mundlak{suffix}.csv", index=False)
    print("Computing policy-contrast specification (robustness)...")
    contrast = compute_policy_contrast(data)
    contrast.to_csv(out_dir / f"rq1_policy_contrast{suffix}.csv", index=False)

    print(f"\nSaved CSVs in {out_dir}")

    # Printout + headline JSON
    headline = print_headline(cells_headline, by_task, by_model, btwn)
    with open(out_dir / f"rq1_headline{suffix}.json", "w") as f:
        json.dump(headline, f, indent=2)
    print(f"\nHeadline JSON saved: {out_dir / f'rq1_headline{suffix}.json'}")

    # Mundlak + contrast printout (robustness section)
    print_mundlak_and_contrast(mundlak, contrast)

    # Figure
    print("\nGenerating RQ1 summary figure...")
    figure_rq1(cells_headline, by_model, by_task, out_dir)

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()