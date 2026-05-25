#!/usr/bin/env python3
"""
scatter_big5_behavior.py
========================
Between-model scatter plots of Big Five self-report against task behaviour,
using the same Mundlak β_between approach as scatter_sr_behavior.py.

Each panel shows 11 dots (one per model). For a given (task, trait) cell:
  x = per-model mean of the Big Five trait (POOLED across ~27 conditions)
  y = per-model mean of align_score (Policy-A-direction alignment)

Big Five SR is task-agnostic (no policy framing), so there is no policy
split to pool across. All conditions for a given model/task are used
directly. The behaviour y-axis uses the Policy A alignment formula as the
reference direction (same sign convention as scatter_sr_behavior.py), so
the two figures can be read side-by-side:
  Sycophancy   : align = 1 - sycophancy_rate   (high = independent)
  CCT          : align = 1 - mean_k / 32        (high = loss-averse)
  Honesty      : align = 1 - brier_c1           (high = calibrated;
                   for Big5, the paper uses brier_c1 as the single metric
                   since there is no policy context, matching Table 13)
  IAT          : align = 1 - mean_bias_score     (high = unbiased)

Behaviour x SR trait axes are both on their native scales:
  x: Big Five trait (Likert 1–5)
  y: align_score (0–1)

Layout: rows = 5 Big Five traits (Extraversion / Agreeableness /
Conscientiousness / Neuroticism / Openness), columns = 4 tasks.

β_between is the Mundlak between-model coefficient from pooled OLS with
z-standardised x and y, cluster-robust SEs at the model level.

Usage:
  python scripts/analysis_scripts/scatter_big5_behavior.py \\
    --within_root results/psycohere_v1/within/grid \\
    --induction_label grid \\
    --out_dir results/psycohere_v1/analysis/scatter_big5_behavior/grid
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import pearsonr

try:
    import statsmodels.formula.api as smf
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BIG5_TRAITS = [
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "neuroticism",
    "openness",
]
BIG5_LABELS = {
    "extraversion":      "Extraversion",
    "agreeableness":     "Agreeableness",
    "conscientiousness": "Conscientiousness",
    "neuroticism":       "Neuroticism",
    "openness":          "Openness",
}

# Columns in combined_runs.csv
BIG5_SR_COLS = {t: f"{t}_mean" for t in BIG5_TRAITS}

# Task column order — 4 columns, matching scatter_sr_behavior.py
COLUMN_ORDER = [
    ("Sycophancy",         "sycophancy"),
    ("Risk-taking (CCT)",  "cct"),
    ("Honesty",            "honesty"),
    ("Implicit bias (IAT)","iat"),
]

# Within-session data dirs (Big5 layout: big5_psycohere_grid/{task}/big5/)
TASK_BIG5_DIRS = {
    "sycophancy": "big5_psycohere_grid/sycophancy/big5",
    "cct":        "big5_psycohere_grid/cct/big5",
    "honesty":    "big5_psycohere_grid/honesty/big5",
    "iat":        "big5_psycohere_grid/iat/big5",
}

TASK_COLORS = {
    "Sycophancy":          "#d95f02",
    "Risk-taking (CCT)":   "#666666",
    "Honesty":             "#1b9e77",
    "Implicit bias (IAT)": "#7570b3",
}

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


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_big5_task(within_root: Path, task: str) -> pd.DataFrame:
    """
    Load Big5 combined_runs for a task. Returns a DataFrame with
    model_key, Big5 trait means, and the task behaviour columns.
    """
    sub = TASK_BIG5_DIRS[task]
    task_dir = within_root / sub
    if not task_dir.exists():
        # tolerate persona naming
        task_dir = within_root / sub.replace("_grid", "_personas")
    if not task_dir.exists():
        raise FileNotFoundError(
            f"Could not find {sub} (or _personas variant) under {within_root}"
        )
    path = task_dir / "combined_runs.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    df = pd.read_csv(path, low_memory=False)
    if "sr_status" in df.columns:
        df = df[df["sr_status"] == "ok"]
    if "beh_status" in df.columns:
        df = df[df["beh_status"] == "ok"]

    for t in BIG5_TRAITS:
        col = BIG5_SR_COLS[t]
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    out = pd.DataFrame({"model_key": df["model_key"], "task": task})
    for t in BIG5_TRAITS:
        col = BIG5_SR_COLS[t]
        out[col] = df[col] if col in df.columns else np.nan

    # Behaviour columns (same task-specific set as scatter_sr_behavior.py)
    if task == "cct":
        out["beh_mean_k"] = pd.to_numeric(df["beh__mean_k"], errors="coerce")
    elif task == "sycophancy":
        out["beh_sycophancy_rate"] = pd.to_numeric(
            df["beh__sycophancy_rate"], errors="coerce"
        )
    elif task == "honesty":
        out["beh_brier_c1"]      = pd.to_numeric(df["beh__mean_brier_c1"], errors="coerce")
        out["beh_abs_confdelta"] = pd.to_numeric(df["beh__mean_abs_confidence_delta"], errors="coerce")
    elif task == "iat":
        out["beh_bias"] = pd.to_numeric(
            df["beh__mean_bias_score"], errors="coerce"
        )
    return out


def _align_score(row: pd.Series, task: str) -> float:
    """
    Per-condition alignment score in [0, 1]. Uses the Policy A direction
    as the reference (high = aligned with loss-averse / independent /
    calibrated-or-stable / unbiased), matching scatter_sr_behavior.py.
    """
    if task == "cct":
        k = row["beh_mean_k"]
        if pd.isna(k):
            return np.nan
        return 1.0 - min(max(float(k) / 32.0, 0.0), 1.0)
    if task == "sycophancy":
        r = row["beh_sycophancy_rate"]
        if pd.isna(r):
            return np.nan
        return 1.0 - min(max(float(r), 0.0), 1.0)
    if task == "honesty":
        # For Big5 (no policy context), the paper uses 1 - brier_c1 as the
        # single calibration-alignment metric, matching the Big5 robustness
        # table (tab:rq2_robustness). This gives Cons β_between=+0.68 and
        # Open β_between=+0.54, matching the paper exactly.
        b = row["beh_brier_c1"]
        if pd.isna(b):
            return np.nan
        return 1.0 - min(max(float(b), 0.0), 1.0)
    if task == "iat":
        b = row["beh_bias"]
        if pd.isna(b):
            return np.nan
        return 1.0 - min(max(float(b), 0.0), 1.0)
    raise ValueError(f"Unknown task {task!r}")


def per_model_data(big5_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Per (model × column) row: model_key, column, task, trait means
    (per-model mean across ~27 conditions), align_beh (per-model mean),
    n_conditions.
    """
    rows = []
    for col_label, task_key in COLUMN_ORDER:
        df = big5_data[task_key].copy()
        df["__align"] = df.apply(lambda r: _align_score(r, task_key), axis=1)
        for model, g in df.groupby("model_key", dropna=False):
            valid = g.dropna(subset=["__align"])
            if len(valid) == 0:
                continue
            row = {
                "model_key":    model,
                "column":       col_label,
                "task":         task_key,
                "n_conditions": len(valid),
                "align_beh":    float(valid["__align"].mean()),
            }
            for t in BIG5_TRAITS:
                col = BIG5_SR_COLS[t]
                s = pd.to_numeric(g[col], errors="coerce").dropna()
                row[col] = float(s.mean()) if len(s) > 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mundlak β_between
# ---------------------------------------------------------------------------

def mundlak_between(
    big5_data: Dict[str, pd.DataFrame],
    trait: str,
    task_key: str,
) -> Tuple[float, float, float]:
    """
    Mundlak pooled OLS with z-standardised x (trait) and y (align_score),
    cluster-robust SEs at the model level.
    Returns (β_within, β_between, p_β_between).
    """
    if not _HAS_STATSMODELS:
        return np.nan, np.nan, np.nan
    df = big5_data[task_key].copy()
    df["__align"] = df.apply(lambda r: _align_score(r, task_key), axis=1)
    sr_col = BIG5_SR_COLS[trait]
    df = df.dropna(subset=[sr_col, "__align"])
    if len(df) < 15:
        return np.nan, np.nan, np.nan
    x = pd.to_numeric(df[sr_col], errors="coerce")
    y = pd.to_numeric(df["__align"], errors="coerce")
    if x.std() < 1e-9 or y.std() < 1e-9:
        return np.nan, np.nan, np.nan
    df = df.copy()
    df["x_z"] = (x - x.mean()) / x.std()
    df["y_z"] = (y - y.mean()) / y.std()
    df["x_dm"] = df.groupby("model_key")["x_z"].transform(lambda s: s - s.mean())
    df["x_bar"] = df.groupby("model_key")["x_z"].transform("mean")
    try:
        model = smf.ols("y_z ~ x_dm + x_bar", data=df)
        res = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": df["model_key"]},
        )
        return (
            float(res.params["x_dm"]),
            float(res.params["x_bar"]),
            float(res.pvalues["x_bar"]),
        )
    except Exception:
        return np.nan, np.nan, np.nan


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan, int(mask.sum())
    xc, yc = x[mask], y[mask]
    if np.std(xc) < 1e-9 or np.std(yc) < 1e-9:
        return np.nan, np.nan, int(mask.sum())
    r, p = pearsonr(xc, yc)
    return float(r), float(p), int(mask.sum())


def _scatter_panel(
    ax: plt.Axes,
    sub: pd.DataFrame,
    trait: str,
    show_xlabel: bool,
    show_ylabel: bool,
    column_label: str,
    trait_label: str,
    beta_between: float,
    beta_between_p: float,
):
    sr_col = BIG5_SR_COLS[trait]
    x = pd.to_numeric(sub[sr_col], errors="coerce").values
    y = pd.to_numeric(sub["align_beh"], errors="coerce").values
    models = sub["model_key"].astype(str).values

    ax.axhline(0.5, color="#d4d4d4", lw=0.6, zorder=0)

    color = TASK_COLORS.get(column_label, "#444444")
    ax.scatter(x, y, s=90, c=color, alpha=0.85,
               edgecolors="white", linewidths=0.9, zorder=3)

    # Greedy label placement (most isolated first, skip overlaps)
    if len(x) > 0:
        cx, cy = np.nanmean(x), np.nanmean(y)
        dist = np.sqrt(((x - cx) / 4) ** 2 + ((y - cy) / 1) ** 2)
        order = np.argsort(-dist)
        placed = []
        bw_x, bw_y = 0.22, 0.055  # x on 1-5 scale, narrower than 1-7
        for idx in order:
            xi, yi = x[idx], y[idx]
            if not (np.isfinite(xi) and np.isfinite(yi)):
                continue
            lx = xi + 0.06
            ly = yi + 0.035
            box = (lx, lx + bw_x * 2, ly, ly + bw_y * 2)
            if any(box[0] < p[1] and box[1] > p[0] and
                   box[2] < p[3] and box[3] > p[2] for p in placed):
                continue
            placed.append(box)
            short = MODEL_SHORT.get(models[idx], models[idx][:6])
            ax.annotate(short, (xi, yi),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=7.5, color="#374151", zorder=4)

    r_pm, p_pm, n = _pearson(x, y)

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 3 and not np.isnan(r_pm):
        xc, yc = x[mask], y[mask]
        slope, intercept = np.polyfit(xc, yc, 1)
        xline = np.array([xc.min() - 0.03, xc.max() + 0.03])
        ax.plot(xline, intercept + slope * xline,
                color="#0f172a", lw=1.4, alpha=0.6, zorder=2)

    sig = lambda pp: ("***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else "")
    lines = []
    if not np.isnan(beta_between):
        lines.append(f"β = {beta_between:+.2f}{sig(beta_between_p)}")
    if not np.isnan(r_pm):
        lines.append(f"r = {r_pm:+.2f}")
    ax.text(0.04, 0.97, "\n".join(lines),
            transform=ax.transAxes, fontsize=9.5, ha="left", va="top",
            color="#0f172a",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.0),
            zorder=5)

    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="both", labelsize=9)

    if show_xlabel:
        ax.set_xlabel("Big Five trait (1–5)", fontsize=11)
    if show_ylabel:
        ax.set_ylabel(f"{trait_label}\nalign (0–1)",
                      fontsize=10.5, fontweight="bold")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_scatter_grid(
    cell: pd.DataFrame,
    big5_data: Dict[str, pd.DataFrame],
    out_path: Path,
    induction: str = "grid",
):
    nrows = len(BIG5_TRAITS)
    ncols = len(COLUMN_ORDER)
    fig = plt.figure(figsize=(16, 17))
    gs = gridspec.GridSpec(
        nrows, ncols, figure=fig,
        hspace=0.28, wspace=0.30,
        left=0.08, right=0.99, top=0.95, bottom=0.06,
    )

    for r, trait in enumerate(BIG5_TRAITS):
        for c, (col_label, task_key) in enumerate(COLUMN_ORDER):
            ax = fig.add_subplot(gs[r, c])
            sel = cell[cell["column"] == col_label]
            _, beta_b, beta_b_p = mundlak_between(big5_data, trait, task_key)
            _scatter_panel(
                ax, sel, trait,
                show_xlabel=(r == nrows - 1),
                show_ylabel=(c == 0),
                column_label=col_label,
                trait_label=BIG5_LABELS[trait],
                beta_between=beta_b,
                beta_between_p=beta_b_p,
            )
            if r == 0:
                ax.set_title(col_label, fontsize=16,
                             color=TASK_COLORS.get(col_label, "#222"),
                             pad=6)

    fig.suptitle(
        f"Between-model Big Five SR vs Behaviour (within-session, induction = {induction})",
        fontsize=22, y=0.99,
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
    parser.add_argument(
        "--within_root", type=str, required=True,
        help="Path to within/grid (or within/personas) folder",
    )
    parser.add_argument(
        "--induction_label", type=str, default=None,
        choices=[None, "grid", "personas"],
    )
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.within_root)
    big5_data = {t: _load_big5_task(root, t) for _, t in COLUMN_ORDER}

    if args.induction_label:
        induction = args.induction_label
    elif "personas" in str(root):
        induction = "personas"
    else:
        induction = "grid"

    if not _HAS_STATSMODELS:
        print("[warn] statsmodels not installed; β_between will be NaN.")

    print("[1/3] Aggregating per-model means ...")
    cell = per_model_data(big5_data)
    cell.to_csv(out_dir / "per_model_scatter_data_big5.csv", index=False)
    print(f"      {len(cell)} (model × column) rows written")

    print("[2/3] Computing Mundlak β_between per panel ...")
    rows = []
    for trait in BIG5_TRAITS:
        for col_label, task_key in COLUMN_ORDER:
            sel = cell[cell["column"] == col_label]
            sr_col = BIG5_SR_COLS[trait]
            x = pd.to_numeric(sel[sr_col], errors="coerce").values
            y = pd.to_numeric(sel["align_beh"], errors="coerce").values
            r_pm, p_pm, n_pm = _pearson(x, y)
            _, beta_b, beta_b_p = mundlak_between(big5_data, trait, task_key)
            rows.append({
                "trait":             trait,
                "column":            col_label,
                "n_models":          n_pm,
                "r_pearson_means":   r_pm,
                "p_pearson_means":   p_pm,
                "beta_between":      beta_b,
                "p_beta_between":    beta_b_p,
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "scatter_panel_correlations_big5.csv", index=False)

    print("\nPer-panel β_between:")
    print("-" * 90)
    for _, r_ in summary.iterrows():
        print(f"  {r_['trait']:20s} {r_['column']:22s}  "
              f"β={r_['beta_between']:+.2f}  "
              f"p={r_['p_beta_between']:.3f}  "
              f"r(means)={r_['r_pearson_means']:+.2f}")
    print("-" * 90)

    print("[3/3] Plotting scatter grid ...")
    plot_scatter_grid(
        cell, big5_data,
        out_path=out_dir / f"scatter_big5_vs_behavior_{induction}",
        induction=induction,
    )
    print(f"\nDone. Outputs written to {out_dir}/")
    print("  per_model_scatter_data_big5.csv")
    print("  scatter_panel_correlations_big5.csv")
    print(f"  scatter_big5_vs_behavior_{induction}.{{png,pdf}}")


if __name__ == "__main__":
    main()
