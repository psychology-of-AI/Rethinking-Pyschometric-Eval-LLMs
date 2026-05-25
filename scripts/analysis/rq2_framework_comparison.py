#!/usr/bin/env python3
"""
rq2_framework_comparison.py
===========================
RQ2: Framework specificity. Does fine-grained, causally-grounded, task-specific
TPB outperform coarse-grained Big Five personality traits in predicting
behaviour, under the same within-session shared-context conditions?

Setup
-----
Same filter as RQ1: session_type='within', perturbation='grid'. Adds Big5
framework alongside TPB. Per task we compute:
  - TPB: Intention + theoretically-primary construct
  - Big5: 2 theoretically-motivated traits
Both are aggregated via within-model Pearson r → Fisher-z meta-analytic mean.
Big5 r's are sign-flipped to align with theoretical direction, so higher
r_aligned = more theory-consistent in both frameworks.

Headline comparison: per task, does max_TPB_r > max_Big5_r?

Outputs
-------
  rq2_framework_cells.csv — all 11 models × 4 tasks × 4 constructs (2 TPB + 2 Big5)
  rq2_by_task_framework.csv — per-task × framework Fisher-z summary
  rq2_framework_comparison.csv — per-task Δ (TPB − Big5)
  rq2_framework_summary.{pdf,png} — 2-panel figure
  rq2_headline.json — structured headline numbers

Usage
-----
python scripts/analysis_scripts/rq2_framework_comparison.py \\
    --cct_master     results/psycohere_v1/analysis/cct/cct_master.csv \\
    --syc_master     results/psycohere_v1/analysis/sycophancy/sycophancy_master.csv \\
    --honesty_master results/psycohere_v1/analysis/honesty/honesty_master.csv \\
    --iat_master     results/psycohere_v1/analysis/iat/iat_master.csv \\
    --out_dir        results/psycohere_v1/analysis/rq2_framework
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from rq_config import (
    MODEL_ORDER, MODEL_LABELS, COLORS,
    TASK_ORDER, TASK_LABELS,
    INTENTION_COL, PRIMARY_CONSTRUCT, BIG5_THEORY_PAIRS,
    load_task_tpb, load_task_big5,
    pearson_ci, safe_pearsonr, stars, wilson_ci, fisher_z_mean_ci,
    mundlak_within_between,
)

from psycohere_style import (
    apply_style, C, FS, BAR, HEAT,
    style_ax, style_heatmap_ax, add_zero_line,
    panel_title, suptitle as fig_suptitle,
    annotate_bar, heatmap_cell_text,
)
apply_style()


# ── Cell-level computation ─────────────────────────────────────────────────

def compute_cells(tpb_data: dict, big5_data: dict) -> pd.DataFrame:
    """Compute per-model within-r cells for both TPB (2 constructs per task)
    and Big5 (2 traits per task, sign-flipped to theoretical direction).

    Expected cells: 11 models × 4 tasks × 4 constructs = 176 rows, but IAT
    TPB de-dupes to 1 construct (primary = Intention), so 11 × (4 tasks ×
    4 constructs − 11 × 1) = 165 rows.
    """
    rows = []

    # TPB side
    for task in TASK_ORDER:
        df = tpb_data[task]
        primary_col, primary_label = PRIMARY_CONSTRUCT[task]
        constructs = [(INTENTION_COL, "Intention", +1)]
        if primary_col != INTENTION_COL:
            constructs.append((primary_col, primary_label, +1))

        for construct_col, construct_label, expected_sign in constructs:
            for m in MODEL_ORDER:
                sub = df[df.model_key == m]
                r, p, n = safe_pearsonr(sub[construct_col], sub["align_score"])
                lo, hi = pearson_ci(r, n)
                # For TPB with align_score, expected sign is always +1
                r_aligned = r * expected_sign if not np.isnan(r) else np.nan
                rows.append({
                    "framework":       "TPB",
                    "task":            task,
                    "task_label":      TASK_LABELS[task],
                    "construct":       construct_col,
                    "construct_label": construct_label,
                    "expected_sign":   expected_sign,
                    "outcome_col":     "align_score",
                    "model":           m,
                    "model_label":     MODEL_LABELS[m],
                    "r":               r,
                    "r_aligned":       r_aligned,
                    "p":               p,
                    "ci_lo":           lo,
                    "ci_hi":           hi,
                    "n":               n,
                    "direction_correct": (r_aligned > 0)
                                          if not np.isnan(r_aligned) else np.nan,
                    "significant":       (p < 0.05)
                                          if not np.isnan(p) else np.nan,
                    "alignment_hit":     ((r_aligned > 0) and (p < 0.05))
                                          if not (np.isnan(r_aligned) or np.isnan(p))
                                          else np.nan,
                })

    # Big5 side — uses task-specific raw behavioural outcome (not align_score)
    # because Big5 has no per-policy framing in our experimental design
    # (single unified 'big5' policy). Expected sign applied to get r_aligned.
    for task in TASK_ORDER:
        df = big5_data[task]
        for trait_col, trait_label, expected_sign, outcome_col in BIG5_THEORY_PAIRS[task]:
            for m in MODEL_ORDER:
                sub = df[df.model_key == m]
                if trait_col not in sub.columns or outcome_col not in sub.columns:
                    continue
                r, p, n = safe_pearsonr(sub[trait_col], sub[outcome_col])
                lo, hi = pearson_ci(r, n)
                r_aligned = r * expected_sign if not np.isnan(r) else np.nan
                rows.append({
                    "framework":       "Big5",
                    "task":            task,
                    "task_label":      TASK_LABELS[task],
                    "construct":       trait_col,
                    "construct_label": trait_label,
                    "expected_sign":   expected_sign,
                    "outcome_col":     outcome_col,
                    "model":           m,
                    "model_label":     MODEL_LABELS[m],
                    "r":               r,
                    "r_aligned":       r_aligned,
                    "p":               p,
                    "ci_lo":           lo,
                    "ci_hi":           hi,
                    "n":               n,
                    "direction_correct": (r_aligned > 0)
                                          if not np.isnan(r_aligned) else np.nan,
                    "significant":       (p < 0.05)
                                          if not np.isnan(p) else np.nan,
                    "alignment_hit":     ((r_aligned > 0) and (p < 0.05))
                                          if not (np.isnan(r_aligned) or np.isnan(p))
                                          else np.nan,
                })

    return pd.DataFrame(rows)


# ── Aggregations ──────────────────────────────────────────────────────────

def aggregate_by_task_framework(cells: pd.DataFrame) -> pd.DataFrame:
    """Per (task × framework × construct) Fisher-z mean r_aligned.

    Uses r_aligned (sign-flipped to theoretical expectation) so positive =
    theory-consistent for both frameworks.
    """
    out = (cells.groupby(["framework", "task", "task_label",
                           "construct", "construct_label"])
                .agg(n_cells=("r_aligned", "size"),
                     n_direction=("direction_correct", "sum"),
                     n_significant=("significant", "sum"),
                     n_alignment=("alignment_hit", "sum"),
                     mean_r=("r_aligned", "mean"),
                     median_r=("r_aligned", "median"))
                .reset_index())

    fz_stats = []
    for _, row in out.iterrows():
        grp = cells[(cells["framework"] == row["framework"]) &
                    (cells["task"] == row["task"]) &
                    (cells["construct"] == row["construct"])]
        mr, lo, hi, k = fisher_z_mean_ci(grp["r_aligned"].values,
                                          grp["n"].values)
        fz_stats.append((mr, lo, hi, k))
    out["fz_mean_r"] = [s[0] for s in fz_stats]
    out["fz_ci_lo"] = [s[1] for s in fz_stats]
    out["fz_ci_hi"] = [s[2] for s in fz_stats]
    out["fz_n_cells_used"] = [s[3] for s in fz_stats]

    out["pct_direction"] = 100 * out["n_direction"] / out["n_cells"]
    out["pct_alignment"] = 100 * out["n_alignment"] / out["n_cells"]
    return out


def framework_head_to_head(by_task_framework: pd.DataFrame) -> pd.DataFrame:
    """Per-task comparison: best of TPB vs best of Big5.

    For each task, picks the construct (within each framework) with the
    highest Fisher-z fz_mean_r and compares TPB-best vs Big5-best.
    """
    rows = []
    for task in TASK_ORDER:
        for framework in ["TPB", "Big5"]:
            sub = by_task_framework[
                (by_task_framework["task"] == task) &
                (by_task_framework["framework"] == framework)
            ]
            if sub.empty:
                continue
            best = sub.loc[sub["fz_mean_r"].idxmax()]
            rows.append({
                "task":            task,
                "task_label":      TASK_LABELS[task],
                "framework":       framework,
                "best_construct":  best["construct_label"],
                "fz_mean_r":       best["fz_mean_r"],
                "fz_ci_lo":        best["fz_ci_lo"],
                "fz_ci_hi":        best["fz_ci_hi"],
                "pct_alignment":   best["pct_alignment"],
                "n_alignment":     best["n_alignment"],
                "n_cells":         best["n_cells"],
            })
    wide = pd.DataFrame(rows)

    # Compute per-task Δ (TPB − Big5) on fz_mean_r
    delta_rows = []
    for task in TASK_ORDER:
        tpb_row = wide[(wide["task"] == task) & (wide["framework"] == "TPB")]
        b5_row  = wide[(wide["task"] == task) & (wide["framework"] == "Big5")]
        if tpb_row.empty or b5_row.empty:
            continue
        delta_rows.append({
            "task":          task,
            "task_label":    TASK_LABELS[task],
            "tpb_best":      tpb_row.iloc[0]["best_construct"],
            "tpb_r":         tpb_row.iloc[0]["fz_mean_r"],
            "tpb_ci_lo":     tpb_row.iloc[0]["fz_ci_lo"],
            "tpb_ci_hi":     tpb_row.iloc[0]["fz_ci_hi"],
            "big5_best":     b5_row.iloc[0]["best_construct"],
            "big5_r":        b5_row.iloc[0]["fz_mean_r"],
            "big5_ci_lo":    b5_row.iloc[0]["fz_ci_lo"],
            "big5_ci_hi":    b5_row.iloc[0]["fz_ci_hi"],
            "delta_r":       tpb_row.iloc[0]["fz_mean_r"] - b5_row.iloc[0]["fz_mean_r"],
            "tpb_wins":      tpb_row.iloc[0]["fz_mean_r"] > b5_row.iloc[0]["fz_mean_r"],
        })
    return wide, pd.DataFrame(delta_rows)


def compute_mundlak_frameworks(tpb_data: dict, big5_data: dict) -> pd.DataFrame:
    """Mundlak pooled OLS for both frameworks, per-task-per-construct.
    Robustness complement to the Fisher-z per-model-aggregated analysis."""
    rows = []

    # TPB
    for task in TASK_ORDER:
        df = tpb_data[task]
        primary_col, primary_label = PRIMARY_CONSTRUCT[task]
        constructs = [(INTENTION_COL, "Intention")]
        if primary_col != INTENTION_COL:
            constructs.append((primary_col, primary_label))
        for construct_col, construct_label in constructs:
            m = mundlak_within_between(df, construct_col, "align_score",
                                         group_col="model_key",
                                         z_normalize=True)
            rows.append({
                "framework": "TPB", "task": task,
                "task_label": TASK_LABELS[task],
                "construct": construct_col,
                "construct_label": construct_label,
                "expected_sign": +1,
                **m,
            })

    # Big5 (uses task-specific raw outcome; sign-flip the β by expected_sign
    # so positive = theory-consistent)
    for task in TASK_ORDER:
        df = big5_data[task]
        for trait_col, trait_label, expected_sign, outcome_col in BIG5_THEORY_PAIRS[task]:
            if trait_col not in df.columns or outcome_col not in df.columns:
                continue
            m = mundlak_within_between(df, trait_col, outcome_col,
                                         group_col="model_key",
                                         z_normalize=True)
            # Flip β by expected_sign so positive = theory-consistent
            for k in ["beta_within", "beta_between"]:
                if not np.isnan(m[k]):
                    m[k] = m[k] * expected_sign
            # Flip CI bounds too
            if expected_sign < 0:
                m["ci_within_lo"], m["ci_within_hi"] = (
                    -m["ci_within_hi"], -m["ci_within_lo"])
                m["ci_between_lo"], m["ci_between_hi"] = (
                    -m["ci_between_hi"], -m["ci_between_lo"])
            rows.append({
                "framework": "Big5", "task": task,
                "task_label": TASK_LABELS[task],
                "construct": trait_col,
                "construct_label": trait_label,
                "expected_sign": expected_sign,
                "outcome_col": outcome_col,
                **m,
            })

    return pd.DataFrame(rows)


# ── Printout ──────────────────────────────────────────────────────────────

def print_headline(by_task_framework: pd.DataFrame,
                   head_to_head: pd.DataFrame,
                   delta_table: pd.DataFrame,
                   mundlak: pd.DataFrame) -> dict:
    """Print RQ2 headline + return structured JSON."""
    print("\n" + "=" * 78)
    print("RQ2 HEADLINE — Within-session · grid · TPB vs Big5")
    print("Does fine-grained TPB outperform coarse Big5?")
    print("=" * 78)

    # Overall Fisher-z by framework (across all 4 tasks, averaging constructs)
    tpb_all = by_task_framework[by_task_framework["framework"] == "TPB"]
    b5_all = by_task_framework[by_task_framework["framework"] == "Big5"]
    tpb_mr = tpb_all["fz_mean_r"].mean()
    b5_mr = b5_all["fz_mean_r"].mean()

    print(f"\n  ━━ OVERALL FRAMEWORK MEANS ━━")
    print(f"    TPB  mean r_aligned (avg across task×construct):  {tpb_mr:+.3f}")
    print(f"    Big5 mean r_aligned (avg across task×construct):  {b5_mr:+.3f}")
    print(f"    Δ (TPB − Big5):                                   {tpb_mr - b5_mr:+.3f}")

    print("\n" + "-" * 78)
    print("PER-TASK × FRAMEWORK (Fisher-z mean r_aligned with 95% CI)")
    print("-" * 78)
    print(f"{'Task':<12}  {'Framework':<10}  {'Construct':<20}  "
          f"{'Fisher-z r [95% CI]':<28}")
    for _, row in by_task_framework.iterrows():
        fz_str = f"{row['fz_mean_r']:+.2f} [{row['fz_ci_lo']:+.2f}, {row['fz_ci_hi']:+.2f}]"
        print(f"  {row['task_label']:<10}  {row['framework']:<10}  "
              f"{row['construct_label']:<20}  {fz_str:<28}")

    print("\n" + "-" * 78)
    print("HEAD-TO-HEAD: Best TPB vs Best Big5 per task")
    print("-" * 78)
    print(f"{'Task':<12}  {'TPB-best':<20}  {'TPB r':<8}  "
          f"{'Big5-best':<16}  {'Big5 r':<8}  {'Δ':<8}  Winner")
    for _, row in delta_table.iterrows():
        winner = "TPB ✓" if row["tpb_wins"] else "Big5 ✗"
        print(f"  {row['task_label']:<10}  {row['tpb_best']:<20}  "
              f"{row['tpb_r']:+.3f}   {row['big5_best']:<16}  "
              f"{row['big5_r']:+.3f}   {row['delta_r']:+.3f}  {winner}")

    n_tasks_tpb_wins = int(delta_table["tpb_wins"].sum())
    n_tasks_total = len(delta_table)
    print(f"\n  TPB wins on {n_tasks_tpb_wins}/{n_tasks_total} tasks "
          f"(headline: fine-grained framework superiority)")

    print("\n" + "-" * 78)
    print("ROBUSTNESS — Mundlak β_within (standardized, cluster-robust)")
    print("-" * 78)
    print(f"{'Task':<12}  {'Framework':<10}  {'Construct':<20}  "
          f"{'β_within [95% CI]':<28}")
    for _, row in mundlak.iterrows():
        bw = (f"{row['beta_within']:+.2f} [{row['ci_within_lo']:+.2f},"
              f"{row['ci_within_hi']:+.2f}]{stars(row['p_within'])}")
        print(f"  {row['task_label']:<10}  {row['framework']:<10}  "
              f"{row['construct_label']:<20}  {bw:<28}")

    return {
        "overall_tpb_mean_r":   round(float(tpb_mr), 4),
        "overall_big5_mean_r":  round(float(b5_mr), 4),
        "overall_delta_r":      round(float(tpb_mr - b5_mr), 4),
        "n_tasks_tpb_wins":     n_tasks_tpb_wins,
        "n_tasks_total":        n_tasks_total,
        "per_task_deltas": [
            {
                "task": r["task_label"],
                "tpb_construct": r["tpb_best"],
                "tpb_r": round(float(r["tpb_r"]), 3),
                "tpb_ci": [round(float(r["tpb_ci_lo"]), 3),
                           round(float(r["tpb_ci_hi"]), 3)],
                "big5_construct": r["big5_best"],
                "big5_r": round(float(r["big5_r"]), 3),
                "big5_ci": [round(float(r["big5_ci_lo"]), 3),
                            round(float(r["big5_ci_hi"]), 3)],
                "delta_r": round(float(r["delta_r"]), 3),
                "tpb_wins": bool(r["tpb_wins"]),
            }
            for _, r in delta_table.iterrows()
        ],
    }


# ── Figure (2-panel RQ2 summary) ──────────────────────────────────────────

def figure_rq2(by_task_framework: pd.DataFrame,
               delta_table: pd.DataFrame,
               out_dir: Path,
               cells: pd.DataFrame = None):
    """Three-panel RQ2 figure (coarse to fine), matching RQ1 visual conventions.
    Panel A: per-task Fisher-z r_aligned (top: TPB, bottom: Big5)
    Panel B: per-model Fisher-z mean r_aligned, two bars per model (TPB orange, Big5 blue)
    Panel C: Big5 per-cell within-model r_aligned heatmap (mirror of RQ1 panel C)
    """
    import matplotlib.patheffects as path_effects
    from matplotlib.colors import TwoSlopeNorm

    fig = plt.figure(figsize=(24, 9.0))
    gs_outer = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.5],
                                wspace=0.35, left=0.05, right=0.97,
                                top=0.88, bottom=0.10)

    # Panel A is split vertically into TPB (top) + Big5 (bottom)
    gs_a = gs_outer[0, 0].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.30)
    ax_a_tpb = fig.add_subplot(gs_a[0, 0])
    ax_a_b5  = fig.add_subplot(gs_a[1, 0])

    ax_b = fig.add_subplot(gs_outer[0, 1])
    ax_c = fig.add_subplot(gs_outer[0, 2])

    # ── Panel A subpanels: per-task coherence (top: TPB, bottom: Big5) ──
    # Match RQ1 task ordering: Honesty / Sycophancy / CCT / IAT (TPB-best descending)
    tpb_only = by_task_framework[by_task_framework["framework"] == "TPB"].copy()
    task_max_tpb = tpb_only.groupby("task")["fz_mean_r"].max().to_dict()
    tasks_sorted = sorted(TASK_ORDER, key=lambda t: task_max_tpb.get(t, 0), reverse=True)

    bar_h_a = BAR["height"]

    TASK_FULL_LABELS = {
        "cct":     "Risk-taking\n(CCT)",
        "syc":     "Sycophancy",
        "honesty": "Honesty",
        "iat":     "Implicit bias\n(IAT)",
    }

    def _plot_subpanel(ax, framework, color_bar, color_label, title_letter):
        sub = by_task_framework[by_task_framework["framework"] == framework].copy()
        # For each task in task_sorted, draw N bars (constructs) stacked vertically
        # within that task's "slot."
        y_pos_a = []
        y_labels = []
        cur_y = 0
        task_centers = []
        for task in tasks_sorted:
            rows = sub[sub["task"] == task].copy()
            n_rows = len(rows)
            if n_rows == 0:
                cur_y -= 1.3
                continue
            # Place each construct's bar
            slot_centre = cur_y - (n_rows - 1) * 0.35  # center of the task slot
            for k, (_, row) in enumerate(rows.iterrows()):
                yp = cur_y - k * 0.7
                ax.barh(yp, row["fz_mean_r"], height=bar_h_a,
                        color=color_bar, alpha=BAR["alpha"], edgecolor=C["bar_edge"], linewidth=BAR["edge_lw"], zorder=3)
                err_lo = max(0, row["fz_mean_r"] - row["fz_ci_lo"])
                err_hi = max(0, row["fz_ci_hi"] - row["fz_mean_r"])
                ax.errorbar(row["fz_mean_r"], yp,
                            xerr=[[err_lo], [err_hi]],
                            fmt="none", ecolor=BAR["ecolor"],
                            capsize=BAR["capsize"], lw=BAR["elinewidth"], zorder=4)
                # white-stroked construct label inside/over the bar
                t = ax.text(0.01, yp, f"  {row['construct_label']}",
                            transform=ax.get_yaxis_transform(),
                            va="center", fontsize=FS["construct_tag"], color=color_label,
                            fontweight="bold", zorder=6)
                t.set_path_effects([
                    path_effects.withStroke(linewidth=4, foreground="white")
                ])
            # Task y-tick label position = midpoint of the rows in this slot
            task_mid = cur_y - (n_rows - 1) * 0.7 / 2
            y_pos_a.append(task_mid)
            y_labels.append(TASK_FULL_LABELS.get(task, TASK_LABELS[task]))
            cur_y -= n_rows * 0.7 + 0.5  # gap between tasks

        add_zero_line(ax, "v")
        ax.set_yticks(y_pos_a)
        ax.set_yticklabels(y_labels, fontsize=FS["tick"], fontweight="bold")
        ax.set_xlim(-0.8, 0.9)
        ax.set_xlabel("Fisher-z mean r_aligned (95% CI)", fontsize=FS["axis_label"])
        ax.tick_params(axis="x", labelsize=FS["tick"])
        ax.set_title(f"{title_letter}. Per-task — {framework}",
                     fontsize=FS["panel_title"], fontweight="bold", loc="left", pad=8)
        style_ax(ax, grid_axis="x")

    _plot_subpanel(ax_a_tpb, "TPB",  C["warm"], C["warm_label"], "A1")
    _plot_subpanel(ax_a_b5,  "Big5", C["cool"], C["cool_label"], "A2")

    # ── Panel B: per-model two-bar comparison, sorted by TPB mean r descending ──
    # Aggregate per-model per-framework
    if cells is None:
        raise ValueError("cells DataFrame required for Panel B")

    per_model_fw = []
    for m in MODEL_ORDER:
        for fw in ("TPB", "Big5"):
            sub = cells[(cells["model"] == m) & (cells["framework"] == fw)]
            if len(sub) == 0:
                continue
            mr, lo, hi, k = fisher_z_mean_ci(sub["r_aligned"].values, sub["n"].values)
            per_model_fw.append({
                "model": m,
                "model_label": MODEL_LABELS[m],
                "framework": fw,
                "fz_mean_r": mr, "fz_ci_lo": lo, "fz_ci_hi": hi,
            })
    pm = pd.DataFrame(per_model_fw)

    # Order models by RQ1 ordering — sort by TPB Fisher-z r descending
    tpb_sort = (pm[pm["framework"] == "TPB"]
                  .sort_values("fz_mean_r", ascending=False)["model"].tolist())
    n_models = len(tpb_sort)

    bar_h_b = BAR["height_b"]
    y_offsets = {"TPB": +bar_h_b/2 + 0.04, "Big5": -bar_h_b/2 - 0.04}
    fw_color = {"TPB": C["warm"], "Big5": C["cool"]}
    fw_alpha = {"TPB": 0.92, "Big5": 0.85}

    y_pos = np.arange(n_models)[::-1]
    for i, m in enumerate(tpb_sort):
        for fw in ("TPB", "Big5"):
            row = pm[(pm["model"] == m) & (pm["framework"] == fw)]
            if len(row) == 0:
                continue
            r = row["fz_mean_r"].iloc[0]
            lo = row["fz_ci_lo"].iloc[0]
            hi = row["fz_ci_hi"].iloc[0]
            ax_b.barh(y_pos[i] + y_offsets[fw], r, height=bar_h_b,
                      color=fw_color[fw], alpha=BAR["alpha"],
                      edgecolor=C["bar_edge"], linewidth=BAR["edge_lw"], zorder=3,
                      label=fw if i == 0 else None)
            err_lo = max(0, r - lo)
            err_hi = max(0, hi - r)
            ax_b.errorbar(r, y_pos[i] + y_offsets[fw],
                          xerr=[[err_lo], [err_hi]],
                          fmt="none", ecolor=BAR["ecolor"], capsize=BAR["capsize"], lw=BAR["elinewidth"],
                          zorder=4)

    add_zero_line(ax_b, "v")
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([MODEL_LABELS[m] for m in tpb_sort], fontsize=FS["tick"])
    ax_b.set_xlabel("Fisher-z mean r_aligned (95% CI)", fontsize=FS["axis_label"])
    ax_b.tick_params(axis="x", labelsize=FS["tick"])
    x_ext = max(0.85, pm[["fz_ci_lo", "fz_ci_hi"]].abs().max().max() + 0.1)
    ax_b.set_xlim(-x_ext, x_ext)
    ax_b.set_ylim(-0.5, n_models - 0.5)   # match imshow row extent in Panel C
    ax_b.set_title("B. Per-model: TPB (orange) vs Big5 (blue)",
               fontsize=FS["panel_title"], fontweight="bold", loc="left", pad=8, x=-0.2)
    style_ax(ax_b, grid_axis="x")
    # Legend in lower-right corner of Panel B
    from matplotlib.patches import Patch
    ax_b.legend(
        handles=[
            Patch(facecolor=C["warm"], alpha=BAR["alpha"], label="TPB"),
            Patch(facecolor=C["cool"], alpha=BAR["alpha"], label="Big5"),
        ],
        loc="lower right", fontsize=FS["legend"], frameon=True, framealpha=0.9,
    )

    # ── Panel C: Big5 per-cell heatmap (mirror of RQ1 panel C) ──
    TASK_ABBREV = {"cct": "CCT", "syc": "Syc.", "honesty": "Hon.", "iat": "IAT"}

    # Ordered Big5 columns: per task, the Big5 traits in BIG5_THEORY_PAIRS order
    col_order = []
    col_labels = []
    for task in TASK_ORDER:
        task_abbr = TASK_ABBREV.get(task, TASK_LABELS[task])
        for trait_col, trait_label, _exp_sign, _outcome in BIG5_THEORY_PAIRS[task]:
            col_order.append((task, trait_col))
            # Abbreviate trait label
            trait_short = {
                "Conscientiousness": "Consc.",
                "Agreeableness": "Agree.",
                "Neuroticism": "Neurot.",
                "Openness": "Open.",
            }.get(trait_label, trait_label)
            col_labels.append(f"{task_abbr}\n{trait_short}")
    n_cols = len(col_order)

    big5_cells = cells[cells["framework"] == "Big5"]
    heat = np.full((n_models, n_cols), np.nan)
    sigmask = np.zeros((n_models, n_cols), dtype=bool)
    for j, (task, trait_col) in enumerate(col_order):
        for i, m in enumerate(tpb_sort):
            row = big5_cells[(big5_cells["task"] == task) &
                             (big5_cells["construct"] == trait_col) &
                             (big5_cells["model"] == m)]
            if len(row):
                heat[i, j] = row["r_aligned"].iloc[0]
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
            txt_color = "white" if abs(val) > 0.35 else "black"
            sig_marker = "*" if sigmask[i, j] else ""
            ax_c.text(j, i, f"{val:+.2f}{sig_marker}",
                      ha="center", va="center", fontsize=FS["heatmap_cell"],
                      color=txt_color, fontweight="bold")

    # Vertical separators between tasks (every 2 columns)
    for sep in range(2, n_cols, 2):
        ax_c.axvline(sep - 0.5, color=C["spine"], lw=HEAT["task_sep_lw"], alpha=HEAT["task_sep_alpha"])

    ax_c.set_xticks(range(n_cols))
    ax_c.set_xticklabels(col_labels, fontsize=FS["tick"])
    ax_c.set_yticks(range(n_models))
    ax_c.set_yticklabels([MODEL_LABELS[m] for m in tpb_sort], fontsize=FS["tick"])
    panel_title(ax_c, "C. Big5 within-model r_aligned  (* = p < .05)")
    cbar = fig.colorbar(im, ax=ax_c, shrink=0.7, pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    ax_c.tick_params(length=0)
    style_heatmap_ax(ax_c)

    # Sync Panel B vertical extent to Panel C so model-name rows align exactly.
    fig.canvas.draw()
    pos_b = ax_b.get_position()
    pos_c = ax_c.get_position()
    ax_b.set_position([pos_b.x0, pos_c.y0, pos_b.width, pos_c.height])

    fig.suptitle(
        "RQ2 — Same-session: Does TPB granularity outperform Big5 in predicting task behavior?",
        fontsize=FS["suptitle"], fontweight="bold", y=0.97,
    )

    for ext in ["pdf", "png"]:
        p = out_dir / f"rq2_framework_summary.{ext}"
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
                    default="results/psycohere_v1/analysis/rq2_framework")
    ap.add_argument("--out_suffix", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading TPB + Big5 masters (within-session, grid perturbation)...")
    tpb_data = {
        "cct":     load_task_tpb(args.cct_master,     "cct",     "within", "grid"),
        "syc":     load_task_tpb(args.syc_master,     "syc",     "within", "grid"),
        "honesty": load_task_tpb(args.honesty_master, "honesty", "within", "grid"),
        "iat":     load_task_tpb(args.iat_master,     "iat",     "within", "grid"),
    }
    big5_data = {
        "cct":     load_task_big5(args.cct_master,     "cct",     "within", "grid"),
        "syc":     load_task_big5(args.syc_master,     "syc",     "within", "grid"),
        "honesty": load_task_big5(args.honesty_master, "honesty", "within", "grid"),
        "iat":     load_task_big5(args.iat_master,     "iat",     "within", "grid"),
    }
    print("\nData coverage (after filtering):")
    for task in TASK_ORDER:
        tpb_n = len(tpb_data[task]); b5_n = len(big5_data[task])
        print(f"  {task:<10}  TPB={tpb_n:>5} rows  Big5={b5_n:>5} rows")

    cells = compute_cells(tpb_data, big5_data)
    by_task_framework = aggregate_by_task_framework(cells)
    head_to_head, delta_table = framework_head_to_head(by_task_framework)
    print("\nComputing Mundlak OLS for both frameworks (robustness)...")
    mundlak = compute_mundlak_frameworks(tpb_data, big5_data)

    suffix = args.out_suffix
    cells.to_csv(out_dir / f"rq2_framework_cells{suffix}.csv", index=False)
    by_task_framework.to_csv(out_dir / f"rq2_by_task_framework{suffix}.csv", index=False)
    delta_table.to_csv(out_dir / f"rq2_framework_comparison{suffix}.csv", index=False)
    mundlak.to_csv(out_dir / f"rq2_mundlak{suffix}.csv", index=False)
    print(f"\nSaved CSVs in {out_dir}")

    headline = print_headline(by_task_framework, head_to_head, delta_table, mundlak)
    with open(out_dir / f"rq2_headline{suffix}.json", "w") as f:
        json.dump(headline, f, indent=2)

    print("\nGenerating RQ2 summary figure...")
    figure_rq2(by_task_framework, delta_table, out_dir, cells=cells)

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()