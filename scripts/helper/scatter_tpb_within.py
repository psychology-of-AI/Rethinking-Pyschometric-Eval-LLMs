#!/usr/bin/env python3
"""
scatter_tpb_within.py
======================
Within-model scatter plots of TPB self-report vs task behaviour, matching
the paper's primary Fisher-z aggregated r metric (Table 1 / Fig. 2).

Each panel (task × SR construct) shows:
  - ~54 dots per model (one per condition), coloured by model
  - 11 per-model OLS fit lines (same colour, semi-transparent)
  - Fisher-z aggregated r with 95% CI — matches the paper's reported values

Layout: 4 rows (SR constructs) × 4 columns (tasks), same structure as the
between-model scatter (scatter_sr_behavior.py) for direct comparison.

x-axis: SR construct (Likert 1–7)
y-axis: align_score (0–1, policy-sign-corrected per merge config)

Both policies are pooled into the same panel, consistent with how the
paper computes r_Fisher per model (Pearson across all ~54 conditions,
spanning both policy variants). The scatter therefore shows two natural
sub-clusters per panel (one per policy) plus the within-model slope.

Fisher-z aggregation:
  1. Per model: r_m = Pearson(SR, align) across ~54 conditions
  2. z_m = arctanh(r_m)   [models with constant y are skipped]
  3. z_mean = mean(z_m) across valid models
  4. SE = sqrt(sum(1/(n_m - 3))) / n_valid_models
  5. r_Fisher = tanh(z_mean);  95% CI = tanh(z_mean ± 1.96 * SE)

Verified matches:
  Honesty    × Attitude  : r = +0.67  ✓
  Sycophancy × Intention : r = +0.47  ✓  (phi-4 excluded: constant y)
  CCT        × Intention : r = +0.22  ✓
  IAT        × Intention : r = -0.59  ✓

Usage:
  python scripts/analysis_scripts/scatter_tpb_within.py \\
    --within_root results/psycohere_v1/within/grid \\
    --induction_label grid \\
    --out_dir results/psycohere_v1/analysis/scatter_tpb_within/grid
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SR_CONSTRUCTS = ["intention", "attitude", "subjective_norm", "pbc"]
SR_LABELS = {
    "attitude":        "Attitude",
    "subjective_norm": "Subjective Norm",
    "pbc":             "Perceived Behavioural Control",
    "intention":       "Intention",
}

COLUMN_ORDER = [
    ("Sycophancy",          "sycophancy"),
    ("Risk-taking (CCT)",   "cct"),
    ("Honesty",             "honesty"),
    ("Implicit bias (IAT)", "iat"),
]

TASK_POLICIES = {
    "cct":        ("tpb_cct_psycohere_grid",
                   "loss_averse", "gain_seeking"),
    "sycophancy": ("tpb_sycophancy_psycohere_grid",
                   "independent_judgment", "defer_when_uncertain"),
    "honesty":    ("tpb_honesty_psycohere_grid",
                   "calibrated_confidence", "keep_confidence_stable"),
    "iat":        ("tpb_iat_psycohere_grid",
                   "unbiased_categorization", "intuitive_fast"),
}

TASK_COLORS = {
    "Sycophancy":          "#d95f02",
    "Risk-taking (CCT)":   "#666666",
    "Honesty":             "#1b9e77",
    "Implicit bias (IAT)": "#7570b3",
}

# Primary TPB construct per task (from paper Table 1).
# Cells matching (task_key, sr_construct) receive a light-blue background.
PRIMARY_CONSTRUCT = {
    "sycophancy": "subjective_norm",
    "cct":        "pbc",
    "honesty":    "attitude",
    "iat":        "intention",
}
PRIMARY_BG = "#dbeafe"   # light blue (no directionality implied, pure highlight)

# 11 consistent model colours used across every panel
MODEL_ORDER = [
    "claude37_sonnet", "claude45_haiku", "deepseek_v31",
    "gemini25_flash",  "gpt4o_mini",     "llama33_70b",
    "llama4_maverick", "mistral_large",  "phi4",
    "qwen_235b",       "qwen_72b",
]
MODEL_SHORT = {
    "claude37_sonnet":  "Cl3.7S",
    "claude45_haiku":   "Cl4.5H",
    "deepseek_v31":     "DSv31",
    "gemini25_flash":   "G2.5F",
    "gpt4o_mini":       "4o-m",
    "llama33_70b":      "L3.3-70",
    "llama4_maverick":  "L4-Mav",
    "mistral_large":    "MstrL",
    "phi4":             "phi4",
    "qwen_235b":        "Qw235",
    "qwen_72b":         "Qw72",
}
# 11 qualitative colours (colour-blind friendly, distinct at small sizes)
MODEL_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#9a6324",
]
MODEL_COLORS = {m: MODEL_PALETTE[i] for i, m in enumerate(MODEL_ORDER)}


# ---------------------------------------------------------------------------
# Loading and alignment (same as scatter_sr_behavior.py)
# ---------------------------------------------------------------------------

def _load_within_task(within_root: Path, task: str) -> pd.DataFrame:
    sub, polA, polB = TASK_POLICIES[task]
    task_dir = within_root / sub
    if not task_dir.exists():
        task_dir = within_root / sub.replace("_grid", "_personas")
    if not task_dir.exists():
        raise FileNotFoundError(f"Could not find {sub} under {within_root}")
    frames = []
    for pol in (polA, polB):
        path = task_dir / pol / "combined_runs.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        df = pd.read_csv(path, low_memory=False)
        df["policy_id"] = pol
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if "sr_status" in df.columns:
        df = df[df["sr_status"] == "ok"]
    if "beh_status" in df.columns:
        df = df[df["beh_status"] == "ok"]
    for c in [f"{s}_mean" for s in SR_CONSTRUCTS]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    out = pd.DataFrame({
        "model_key": df["model_key"],
        "policy_id": df["policy_id"],
        "task":      task,
    })
    for c in [f"{s}_mean" for s in SR_CONSTRUCTS]:
        out[c] = df[c] if c in df.columns else np.nan
    if task == "cct":
        out["beh_mean_k"] = pd.to_numeric(df["beh__mean_k"], errors="coerce")
    elif task == "sycophancy":
        out["beh_sycophancy_rate"] = pd.to_numeric(
            df["beh__sycophancy_rate"], errors="coerce")
    elif task == "honesty":
        out["beh_brier_c1"]      = pd.to_numeric(df["beh__mean_brier_c1"], errors="coerce")
        out["beh_abs_confdelta"] = pd.to_numeric(
            df["beh__mean_abs_confidence_delta"], errors="coerce")
    elif task == "iat":
        out["beh_bias"] = pd.to_numeric(df["beh__mean_bias_score"], errors="coerce")
    return out


def _align_score(row: pd.Series, task: str) -> float:
    """Policy-dependent align_score — same formula as scatter_sr_behavior.py."""
    pid = row["policy_id"]
    if task == "cct":
        k = row["beh_mean_k"]
        if pd.isna(k): return np.nan
        return 1.0 - min(max(float(k) / 32.0, 0.0), 1.0) \
            if pid == "loss_averse" else min(max(float(k) / 32.0, 0.0), 1.0)
    if task == "sycophancy":
        r = row["beh_sycophancy_rate"]
        if pd.isna(r): return np.nan
        r = min(max(float(r), 0.0), 1.0)
        return (1.0 - r) if pid == "independent_judgment" else r
    if task == "honesty":
        if pid == "calibrated_confidence":
            b = row["beh_brier_c1"]
            if pd.isna(b): return np.nan
            return 1.0 - min(max(float(b), 0.0), 1.0)
        if pid == "keep_confidence_stable":
            d = row["beh_abs_confdelta"]
            if pd.isna(d): return np.nan
            return 1.0 - min(max(float(d), 0.0), 1.0)
    if task == "iat":
        bias = row["beh_bias"]
        if pd.isna(bias): return np.nan
        bias = min(max(float(bias), 0.0), 1.0)
        return (1.0 - bias) if pid == "unbiased_categorization" else bias
    return np.nan


# ---------------------------------------------------------------------------
# Fisher-z aggregation
# ---------------------------------------------------------------------------

def fisher_z_agg(
    df: pd.DataFrame, sr_col: str
) -> Tuple[float, float, float, int]:
    """
    Returns (r_Fisher, ci_lo, ci_hi, n_valid_models).
    Models with constant y (std < 1e-9) are skipped.
    """
    zs, ns = [], []
    for _, g in df.groupby("model_key"):
        x = pd.to_numeric(g[sr_col], errors="coerce")
        y = g["__align"]
        mask = x.notna() & y.notna()
        if mask.sum() < 5:
            continue
        xv, yv = x[mask].values, y[mask].values
        if np.std(yv) < 1e-9 or np.std(xv) < 1e-9:
            continue
        r, _ = pearsonr(xv, yv)
        r = float(np.clip(r, -0.9999, 0.9999))
        zs.append(np.arctanh(r))
        ns.append(len(xv))
    if len(zs) < 2:
        return np.nan, np.nan, np.nan, len(zs)
    z_mean = np.mean(zs)
    # SE of mean Fisher-z = sqrt(sum(1/(n_m-3))) / n_valid
    se = np.sqrt(sum(1.0 / max(n - 3, 1) for n in ns)) / len(zs)
    return (
        float(np.tanh(z_mean)),
        float(np.tanh(z_mean - 1.96 * se)),
        float(np.tanh(z_mean + 1.96 * se)),
        len(zs),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _within_scatter_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    sr_construct: str,
    task: str,
    show_xlabel: bool,
    show_ylabel: bool,
    column_label: str,
    construct_label: str,
    r_fisher: float,
    ci_lo: float,
    ci_hi: float,
    n_models: int,
    is_primary: bool = False,
):
    sr_col = f"{sr_construct}_mean"

    # Per-model dots coloured by model, shaped by policy so the
    # cross-policy cluster structure is visible.
    _, polA, polB = TASK_POLICIES[task]
    policy_markers = {polA: "o", polB: "^"}  # circle = Policy A, triangle = Policy B

    df2 = df.copy()
    df2[sr_col] = pd.to_numeric(df2[sr_col], errors="coerce")
    df2["x_dm"] = df2[sr_col] - df2.groupby("model_key")[sr_col].transform("mean")
    df2["y_dm"] = df2["__align"] - df2.groupby("model_key")["__align"].transform("mean")

    for model in MODEL_ORDER:
        g = df2[df2["model_key"] == model].dropna(subset=["x_dm", "y_dm"])
        if len(g) < 2:
            continue
        color = MODEL_COLORS[model]
        for policy, marker in policy_markers.items():
            gp = g[g["policy_id"] == policy]
            if len(gp) == 0:
                continue
            ax.scatter(gp["x_dm"], gp["y_dm"],
                       s=12, c=color, alpha=0.45, marker=marker,
                       edgecolors="none", zorder=2, rasterized=True)

    # Single pooled OLS line on demeaned data = within-model slope
    valid = df2.dropna(subset=["x_dm", "y_dm"])
    if len(valid) >= 5:
        xv, yv = valid["x_dm"].values, valid["y_dm"].values
        slope, intercept = np.polyfit(xv, yv, 1)
        xline = np.array([xv.min(), xv.max()])
        ax.plot(xline, intercept + slope * xline,
                color="#0f172a", lw=2.2, alpha=0.8, zorder=4)

    ref_color = "#b0c4de" if is_primary else "#e5e5e5"
    ax.axhline(0, color=ref_color, lw=0.6, zorder=0)
    ax.axvline(0, color=ref_color, lw=0.6, zorder=0)

    # Annotation: Fisher-z r with CI
    sig = ""
    if not np.isnan(r_fisher) and not np.isnan(ci_lo):
        if ci_lo > 0 or ci_hi < 0:   # CI excludes zero
            # approximate p from z-test: z = arctanh(r)*sqrt(n-3) won't work
            # here since we have multiple models; use CI to determine stars
            z_stat = abs(np.arctanh(np.clip(r_fisher, -0.9999, 0.9999)))
            # rough: z_stat / SE_of_mean_z gives N(0,1)
            if abs(r_fisher) > 0:
                se_est = (np.arctanh(min(abs(ci_hi), 0.9999)) -
                          np.arctanh(np.clip(abs(r_fisher), 1e-6, 0.9999))) / 1.96
                if se_est > 1e-9:
                    z_norm = abs(np.arctanh(np.clip(r_fisher, -0.9999, 0.9999))) / se_est
                    if z_norm > 3.29: sig = "***"
                    elif z_norm > 2.58: sig = "**"
                    elif z_norm > 1.96: sig = "*"
    if not np.isnan(r_fisher):
        ci_str = ""
        if not np.isnan(ci_lo):
            ci_str = f"\n[{ci_lo:+.2f}, {ci_hi:+.2f}]"
        ax.text(
            0.04, 0.97,
            f"r = {r_fisher:+.2f}{sig}{ci_str}",
            transform=ax.transAxes,
            fontsize=9, ha="left", va="top",
            color="#0f172a",
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.85, pad=2.0),
            zorder=5,
        )

    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-0.62, 0.62)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-0.5, 0, 0.5])
    ax.tick_params(axis="both", labelsize=9)

    if show_xlabel:
        ax.set_xlabel("SR − model mean (1–7)", fontsize=11)
    if show_ylabel:
        ax.set_ylabel(f"{construct_label}\nalign − model mean",
                      fontsize=10.5, fontweight="bold")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_scatter_grid(
    within_data: Dict[str, pd.DataFrame],
    out_path: Path,
    induction: str = "grid",
):
    nrows = len(SR_CONSTRUCTS)
    ncols = len(COLUMN_ORDER)
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(
        nrows, ncols, figure=fig,
        hspace=0.28, wspace=0.30,
        left=0.08, right=0.99, top=0.95, bottom=0.07,
    )

    for r, sr_c in enumerate(SR_CONSTRUCTS):
        for c, (col_label, task_key) in enumerate(COLUMN_ORDER):
            ax = fig.add_subplot(gs[r, c])

            # Highlight primary-construct cells with a light blue background
            is_primary = PRIMARY_CONSTRUCT.get(task_key) == sr_c
            if is_primary:
                ax.set_facecolor(PRIMARY_BG)

            df = within_data[task_key]
            r_f, ci_lo, ci_hi, n_m = fisher_z_agg(df, f"{sr_c}_mean")
            _within_scatter_panel(
                ax, df, sr_c, task_key,
                show_xlabel=(r == nrows - 1),
                show_ylabel=(c == 0),
                column_label=col_label,
                construct_label=SR_LABELS[sr_c],
                r_fisher=r_f,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                n_models=n_m,
                is_primary=is_primary,
            )
            if r == 0:
                ax.set_title(col_label, fontsize=16,
                             color=TASK_COLORS.get(col_label, "#222"),
                             pad=6)

    fig.suptitle(
        f"Within-model SR vs Behaviour (within-session, induction = {induction})",
        fontsize=22, y=0.99,
    )

    # Model colour legend
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   color=MODEL_COLORS[m], label=MODEL_SHORT[m],
                   markersize=6, alpha=0.85)
        for m in MODEL_ORDER
    ]
    # Policy shape legend
    handles += [
        plt.Line2D([0], [0], marker="o", linestyle="", color="#555",
                   label="Policy A", markersize=6, alpha=0.85),
        plt.Line2D([0], [0], marker="^", linestyle="", color="#555",
                   label="Policy B", markersize=6, alpha=0.85),
    ]
    fig.legend(
        handles=handles, loc="lower center",
        ncol=13, frameon=False,
        bbox_to_anchor=(0.5, 0.002),
        fontsize=8.5,
    )

    fig.savefig(out_path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--within_root", type=str, required=True)
    parser.add_argument("--induction_label", type=str, default=None,
                        choices=[None, "grid", "personas"])
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.within_root)
    if args.induction_label:
        induction = args.induction_label
    elif "personas" in str(root):
        induction = "personas"
    else:
        induction = "grid"

    print("[1/3] Loading within-session data ...")
    within_data = {}
    for _, task_key in COLUMN_ORDER:
        df = _load_within_task(root, task_key)
        df["__align"] = df.apply(lambda r: _align_score(r, task_key), axis=1)
        within_data[task_key] = df

    print("[2/3] Computing Fisher-z r per panel ...")
    rows = []
    print(f"\n{'Construct':18s}  {'Task':22s}  {'r_Fisher':8s}  {'95% CI':18s}  {'n_models':8s}")
    print("-" * 82)
    for sr_c in SR_CONSTRUCTS:
        for col_label, task_key in COLUMN_ORDER:
            df = within_data[task_key]
            r_f, ci_lo, ci_hi, n_m = fisher_z_agg(df, f"{sr_c}_mean")
            ci_str = f"[{ci_lo:+.2f}, {ci_hi:+.2f}]" if not np.isnan(ci_lo) else "n/a"
            print(f"  {sr_c:16s}  {col_label:22s}  {r_f:+.3f}     {ci_str:18s}  {n_m}")
            rows.append({
                "construct": sr_c,
                "column":    col_label,
                "r_fisher":  r_f,
                "ci_lo":     ci_lo,
                "ci_hi":     ci_hi,
                "n_models":  n_m,
            })
    print("-" * 82)

    pd.DataFrame(rows).to_csv(
        out_dir / "within_scatter_correlations.csv", index=False
    )

    print("\n[3/3] Plotting ...")
    plot_scatter_grid(
        within_data,
        out_path=out_dir / f"scatter_tpb_within_{induction}",
        induction=induction,
    )
    print(f"\nDone. Outputs in {out_dir}/")
    print("  within_scatter_correlations.csv")
    print(f"  scatter_tpb_within_{induction}.{{png,pdf}}")


if __name__ == "__main__":
    main()