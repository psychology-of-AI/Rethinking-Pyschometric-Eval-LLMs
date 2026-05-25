#!/usr/bin/env python3
"""
scatter_big5_within.py
=======================
Within-model scatter plots of Big Five self-report vs task behaviour,
matching the paper's RQ2 Fisher-z aggregated r_aligned metric (Fig. 2).

All 5 Big Five traits are shown as rows (OCEAN order). Panels that
correspond to a theoretically-motivated trait-task pair (per the paper's
Table tab:tasks_appendix) are shaded:
  - Light GREEN : theory predicts a POSITIVE relationship (sign = +1)
  - Light RED   : theory predicts a NEGATIVE relationship (sign = −1)
  - White       : no theoretical prediction for this trait-task pair

Theoretically motivated pairs:
  CCT:        Neuroticism(−), Openness(+)
  Sycophancy: Agreeableness(+), Neuroticism(+)
  Honesty:    Conscientiousness(+), Openness(+)
  IAT:        Agreeableness(−), Openness(−)

For shaded panels, r is sign-corrected (r_aligned = sign × raw_r).
For unshaded panels, r is shown as raw Pearson r with no sign correction.

Layout: 5 rows (OCEAN) × 4 columns (tasks).

Usage:
  python scripts/helper_scripts/scatter_big5_within.py \\
    --within_root results/psycohere_v1/within/grid \\
    --induction_label grid \\
    --out_dir results/psycohere_v1/analysis/scatter_big5_within/grid
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

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

# All 5 traits in OCEAN order (rows)
BIG5_TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]
BIG5_LABELS = {
    "extraversion":      "Extraversion",
    "agreeableness":     "Agreeableness",
    "conscientiousness": "Conscientiousness",
    "neuroticism":       "Neuroticism",
    "openness":          "Openness",
}
# Raw behaviour columns used for r_aligned computation (matching paper's
# "correlate each trait with the task's primary behavioural outcome").
# NOT align_score — those are different. The paper explicitly confirms this
# with: CCT-Neuroticism "expected −, actual r_aligned = +0.02" which matches
# sign × r(trait, raw_beh), not sign × r(trait, align).
TASK_RAW_BEH_COL = {
    "cct":        "beh_mean_k",
    "sycophancy": "beh_sycophancy_rate",
    "honesty":    "beh_brier_c1",
    "iat":        "beh_bias",
}

BIG5_SR_COLS = {t: f"{t}_mean" for t in BIG5_TRAITS}

# Theoretically-motivated pairs per task, with the sign being the
# EXPECTED direction of r_aligned as reported in the paper's Table
# tab:tasks_appendix. Positive = green (theory-consistent positive),
# negative = red (theory-consistent negative).
#
# Paper table row:
#   CCT:        Neur(−); Open(+)   → Neur: red,   Open: green
#   Sycophancy: Agree(+); Neur(+)  → both: green
#   Honesty:    Cons(+); Open(+)   → both: green
#   IAT:        Agree(−); Open(−)  → both: red
#
# Note: the paper's r_aligned for CCT-Neuroticism is explicitly cited as
# "expected −" (paper §RQ2), confirming this table-direct interpretation.
TASK_THEORY = {
    "cct":        {"neuroticism": -1, "openness": +1},
    "sycophancy": {"agreeableness": +1, "neuroticism": +1},
    "honesty":    {"conscientiousness": +1, "openness": +1},
    "iat":        {"agreeableness": -1, "openness": -1},
}

# Panel background colours
BG_POSITIVE = "#d4edda"   # light green
BG_NEGATIVE = "#f8d7da"   # light red
BG_NEUTRAL  = "#ffffff"   # white

TASK_BIG5_DIRS = {
    "sycophancy": "big5_psycohere_grid/sycophancy/big5",
    "cct":        "big5_psycohere_grid/cct/big5",
    "honesty":    "big5_psycohere_grid/honesty/big5",
    "iat":        "big5_psycohere_grid/iat/big5",
}

COLUMN_ORDER = [
    ("Sycophancy",          "sycophancy"),
    ("Risk-taking (CCT)",   "cct"),
    ("Honesty",             "honesty"),
    ("Implicit bias (IAT)", "iat"),
]

TASK_COLORS = {
    "Sycophancy":          "#d95f02",
    "Risk-taking (CCT)":   "#666666",
    "Honesty":             "#1b9e77",
    "Implicit bias (IAT)": "#7570b3",
}

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
MODEL_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#9a6324",
]
MODEL_COLORS = {m: MODEL_PALETTE[i] for i, m in enumerate(MODEL_ORDER)}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_big5_task(within_root: Path, task: str) -> pd.DataFrame:
    sub = TASK_BIG5_DIRS[task]
    task_dir = within_root / sub
    if not task_dir.exists():
        task_dir = within_root / sub.replace("_grid", "_personas")
    if not task_dir.exists():
        raise FileNotFoundError(
            f"Could not find {sub} (or _personas variant) under {within_root}"
        )
    df = pd.read_csv(task_dir / "combined_runs.csv", low_memory=False)
    if "sr_status" in df.columns:
        df = df[df["sr_status"] == "ok"]
    if "beh_status" in df.columns:
        df = df[df["beh_status"] == "ok"]
    for col in BIG5_SR_COLS.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    out = pd.DataFrame({"model_key": df["model_key"], "task": task})
    for col in BIG5_SR_COLS.values():
        out[col] = df[col] if col in df.columns else np.nan
    if task == "cct":
        out["beh_mean_k"] = pd.to_numeric(df["beh__mean_k"], errors="coerce")
    elif task == "sycophancy":
        out["beh_sycophancy_rate"] = pd.to_numeric(
            df["beh__sycophancy_rate"], errors="coerce")
    elif task == "honesty":
        out["beh_brier_c1"] = pd.to_numeric(
            df["beh__mean_brier_c1"], errors="coerce")
    elif task == "iat":
        out["beh_bias"] = pd.to_numeric(
            df["beh__mean_bias_score"], errors="coerce")
    return out


def _align_score(row: pd.Series, task: str) -> float:
    if task == "cct":
        k = row["beh_mean_k"]
        if pd.isna(k): return np.nan
        return 1.0 - min(max(float(k) / 32.0, 0.0), 1.0)
    if task == "sycophancy":
        r = row["beh_sycophancy_rate"]
        if pd.isna(r): return np.nan
        return 1.0 - min(max(float(r), 0.0), 1.0)
    if task == "honesty":
        b = row["beh_brier_c1"]
        if pd.isna(b): return np.nan
        return 1.0 - min(max(float(b), 0.0), 1.0)
    if task == "iat":
        b = row["beh_bias"]
        if pd.isna(b): return np.nan
        return 1.0 - min(max(float(b), 0.0), 1.0)
    return np.nan


# ---------------------------------------------------------------------------
# Fisher-z aggregation
# ---------------------------------------------------------------------------

def fisher_z_agg(
    df: pd.DataFrame, sr_col: str, sign: int = 1,
    beh_col: str = "__align",
) -> Tuple[float, float, float, int]:
    """
    Fisher-z r_aligned = sign × r(trait, beh_col) per model.
    For theory panels use beh_col = raw behavior column (matching paper).
    For unshaded panels beh_col defaults to __align.
    """
    zs, ns = [], []
    for _, g in df.groupby("model_key"):
        x = pd.to_numeric(g[sr_col], errors="coerce")
        y = pd.to_numeric(g[beh_col], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 5: continue
        xv, yv = x[mask].values, y[mask].values
        if np.std(xv) < 1e-9 or np.std(yv) < 1e-9: continue
        r, _ = pearsonr(xv, yv)
        r_signed = float(sign) * float(np.clip(r, -0.9999, 0.9999))
        zs.append(np.arctanh(np.clip(r_signed, -0.9999, 0.9999)))
        ns.append(len(xv))
    if len(zs) < 2:
        return np.nan, np.nan, np.nan, len(zs)
    z_mean = np.mean(zs)
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

def _panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    sr_col: str,
    sign: int,
    bg_color: str,
    r_val: float,
    ci_lo: float,
    ci_hi: float,
    show_xlabel: bool,
    show_ylabel: bool,
    ylabel_text: str,
    is_theory: bool,
):
    # Background shading
    ax.set_facecolor(bg_color)

    # Demean x (raw trait, NOT sign-flipped) and y by per-model mean.
    # Rightward on x-axis = more of the trait. The sign of r_aligned tells
    # you whether the slope is theory-consistent (+) or not (−).
    df2 = df.copy()
    df2[sr_col] = pd.to_numeric(df2[sr_col], errors="coerce")
    df2["x_dm"] = df2[sr_col] - \
        df2.groupby("model_key")[sr_col].transform("mean")
    df2["y_dm"] = df2["__align"] - \
        df2.groupby("model_key")["__align"].transform("mean")

    ax.axhline(0, color="#cccccc", lw=0.6, zorder=0)
    ax.axvline(0, color="#cccccc", lw=0.6, zorder=0)

    for model in MODEL_ORDER:
        g = df2[df2["model_key"] == model].dropna(subset=["x_dm", "y_dm"])
        if len(g) < 2: continue
        ax.scatter(g["x_dm"], g["y_dm"], s=14, c=MODEL_COLORS[model],
                   alpha=0.50, edgecolors="none", zorder=2, rasterized=True)

    valid = df2.dropna(subset=["x_dm", "y_dm"])
    if len(valid) >= 5:
        xv, yv = valid["x_dm"].values, valid["y_dm"].values
        slope, intercept = np.polyfit(xv, yv, 1)
        xline = np.array([xv.min(), xv.max()])
        ax.plot(xline, intercept + slope * xline,
                color="#0f172a", lw=2.0, alpha=0.8, zorder=4)

    # Annotation
    if not np.isnan(r_val):
        sig = ""
        if not np.isnan(ci_lo) and (ci_lo > 0 or ci_hi < 0):
            z_abs = abs(np.arctanh(np.clip(r_val, -0.9999, 0.9999)))
            se_est = abs(
                np.arctanh(np.clip(abs(ci_hi if r_val >= 0 else ci_lo), 1e-6, 0.9999)) -
                np.arctanh(np.clip(abs(r_val), 1e-6, 0.9999))
            ) / 1.96
            if se_est > 1e-9:
                z_norm = z_abs / se_est
                if z_norm > 3.29:   sig = "***"
                elif z_norm > 2.58: sig = "**"
                elif z_norm > 1.96: sig = "*"
        prefix = "r = " if is_theory else "r = "
        ci_str = f"\n[{ci_lo:+.2f}, {ci_hi:+.2f}]" if not np.isnan(ci_lo) else ""
        ax.text(0.04, 0.97, f"{prefix}{r_val:+.2f}{sig}{ci_str}",
                transform=ax.transAxes, fontsize=8.5, ha="left", va="top",
                color="#0f172a",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.75, pad=1.5),
                zorder=5)

    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-0.55, 0.55)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-0.5, 0, 0.5])
    ax.tick_params(axis="both", labelsize=8.5)

    if show_xlabel:
        ax.set_xlabel("trait − model mean", fontsize=10)
    if show_ylabel:
        ax.set_ylabel(f"{ylabel_text}\nalign − model mean",
                      fontsize=10.5, fontweight="bold")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#aaaaaa")


def plot_grid(
    big5_data: Dict[str, pd.DataFrame],
    out_path: Path,
    induction: str = "grid",
):
    nrows = len(BIG5_TRAITS)
    ncols = len(COLUMN_ORDER)
    fig = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(
        nrows, ncols, figure=fig,
        hspace=0.22, wspace=0.28,
        left=0.10, right=0.99, top=0.94, bottom=0.08,
    )

    for c, (col_label, task_key) in enumerate(COLUMN_ORDER):
        df = big5_data[task_key]
        theory = TASK_THEORY[task_key]  # {trait: sign}

        for r, trait in enumerate(BIG5_TRAITS):
            ax = fig.add_subplot(gs[r, c])
            sr_col = BIG5_SR_COLS[trait]
            sign = theory.get(trait, None)
            is_theory = sign is not None

            # Background
            if not is_theory:
                bg = BG_NEUTRAL
            elif sign == 1:
                bg = BG_POSITIVE
            else:
                bg = BG_NEGATIVE

            # For theoretically-motivated panels: use raw behavior column
            # with sign-correction to match the paper's r_aligned exactly.
            # Paper formula: r_aligned = sign × r(trait, raw_beh).
            # For unshaded panels: use align with sign=1 (no theory prediction).
            if is_theory:
                raw_beh = TASK_RAW_BEH_COL[task_key]
                r_val, ci_lo, ci_hi, n_m = fisher_z_agg(
                    df, sr_col, sign, beh_col=raw_beh)
            else:
                r_val, ci_lo, ci_hi, n_m = fisher_z_agg(df, sr_col, 1)

            _panel(
                ax, df, sr_col, 1, bg,
                r_val, ci_lo, ci_hi,
                show_xlabel=(r == nrows - 1),
                show_ylabel=(c == 0),
                ylabel_text=BIG5_LABELS[trait],
                is_theory=is_theory,
            )

            if r == 0:
                ax.set_title(col_label, fontsize=16,
                             color=TASK_COLORS.get(col_label, "#222"),
                             pad=6)

    fig.suptitle(
        f"Within-model Big Five SR vs Behaviour "
        f"(within-session, induction = {induction})",
        fontsize=22, y=0.99,
    )

    # Model colour legend
    model_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=MODEL_COLORS[m],
                   label=MODEL_SHORT[m], markersize=6, alpha=0.85)
        for m in MODEL_ORDER
    ]
    # Shading legend
    shade_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=BG_POSITIVE, ec="#aaaaaa",
                       label="Theory: positive"),
        plt.Rectangle((0, 0), 1, 1, fc=BG_NEGATIVE, ec="#aaaaaa",
                       label="Theory: negative"),
        plt.Rectangle((0, 0), 1, 1, fc=BG_NEUTRAL, ec="#aaaaaa",
                       label="No prediction"),
    ]
    fig.legend(
        handles=model_handles + shade_handles,
        loc="lower center", ncol=14,
        frameon=False, bbox_to_anchor=(0.5, 0.002), fontsize=8.5,
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
    induction = args.induction_label or ("personas" if "personas" in str(root) else "grid")

    print("[1/3] Loading ...")
    big5_data = {}
    for _, task_key in COLUMN_ORDER:
        df = _load_big5_task(root, task_key)
        df["__align"] = df.apply(lambda r: _align_score(r, task_key), axis=1)
        big5_data[task_key] = df

    print("[2/3] Computing r per panel ...")
    rows = []
    print(f"\n{'Trait':20s}  {'Task':12s}  {'theory_sign':12s}  "
          f"{'r':7s}  {'95% CI':18s}")
    print("-" * 75)
    for _, task_key in COLUMN_ORDER:
        df = big5_data[task_key]
        theory = TASK_THEORY[task_key]
        raw_beh = TASK_RAW_BEH_COL[task_key]
        for trait in BIG5_TRAITS:
            sign = theory.get(trait, None)
            is_theory = sign is not None
            if is_theory:
                r_val, ci_lo, ci_hi, n_m = fisher_z_agg(
                    df, BIG5_SR_COLS[trait], sign, beh_col=raw_beh)
            else:
                r_val, ci_lo, ci_hi, n_m = fisher_z_agg(df, BIG5_SR_COLS[trait], 1)
            ci_str = f"[{ci_lo:+.2f},{ci_hi:+.2f}]" if not np.isnan(ci_lo) else "n/a"
            sign_str = f"{sign:+d}" if sign is not None else "—"
            print(f"  {trait:18s}  {task_key:12s}  {sign_str:12s}  "
                  f"{r_val:+.3f}  {ci_str}")
            rows.append({"trait": trait, "task": task_key, "theory_sign": sign,
                         "r_aligned": r_val, "ci_lo": ci_lo, "ci_hi": ci_hi, "n_models": n_m})
    print("-" * 75)
    pd.DataFrame(rows).to_csv(
        out_dir / "within_scatter_correlations_big5.csv", index=False)

    print("\n[3/3] Plotting ...")
    plot_grid(big5_data,
              out_path=out_dir / f"scatter_big5_within_{induction}",
              induction=induction)
    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
