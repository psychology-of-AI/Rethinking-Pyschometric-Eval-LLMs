#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_per_model_fingerprints.py

Produces five per-model fingerprint figures for the appendix:

  Figure A: Big 5 SR fingerprints (within-session SR)
            X = 5 traits, Y = 1-5 mean Likert
            Lines: Grid vs Personas

  Figure B-grid: TPB SR fingerprints, GRID induction (within-session SR)
            X = 4 TPB constructs, Y = 1-5 mean Likert
            4 task-colored lines per panel

  Figure B-personas: TPB SR fingerprints, PERSONA induction (within-session SR)
            Same as B-grid

  Figure C-same:  Behavioral fingerprints, SAME-SESSION (within)
            X = 4 tasks, Y = 1-5 normalized score (per scale-mapping table)
            Lines: Grid vs Personas

  Figure C-separate: Behavioral fingerprints, SEPARATE-SESSIONS (between)
            Same as C-same

Behavior outcomes per task (per Personality Illusion mapping):
  Risk Taking (CCT)         : beh__mean_k                   ; 0..32 -> 1+4(x/32)
  Stereotyping (IAT)        : beh__mean_bias_score          ; -1..1 -> 3+2x
  Sycophancy                : beh__sycophancy_rate * 100    ; 0..100 -> 1+4(x/100)
  Honesty (Delta-confidence): beh__mean_confidence_delta    ; -100..100 -> 3+x/50
                              [Note: outcome is per-question; aggregate across questions then runs]

Usage:
  python plot_per_model_fingerprints.py \
      --within_root results/psycohere_v1/within \
      --between_root results/psycohere_v1/between \
      --out_dir figures/per_model_fingerprints

Outputs PDFs + PNGs:
  figA_big5_sr.{pdf,png}
  figB_tpb_sr_grid.{pdf,png}
  figB_tpb_sr_personas.{pdf,png}
  figC_behavior_same_session.{pdf,png}
  figC_behavior_separate_sessions.{pdf,png}
  scale_mapping.csv  (the Table-2 analog for the appendix)
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Display order and pretty names for models
MODELS_ORDER = [
    "claude45_haiku", "claude37_sonnet", "gpt4o_mini", "gemini25_flash",
    "deepseek_v31", "llama4_maverick", "llama33_70b", "qwen_235b",
    "qwen_72b", "mistral_large", "phi4",
]
MODEL_PRETTY = {
    "claude45_haiku":   "Claude Haiku 4.5",
    "claude37_sonnet":  "Claude 3.7 Sonnet",
    "gpt4o_mini":       "GPT-4o Mini",
    "gemini25_flash":   "Gemini 2.5 Flash",
    "deepseek_v31":     "DeepSeek V3.1",
    "llama4_maverick":  "LLaMA-4 Maverick",
    "llama33_70b":      "LLaMA-3.3 70B",
    "qwen_235b":        "Qwen3-235B",
    "qwen_72b":         "Qwen2.5-72B",
    "mistral_large":    "Mistral Large",
    "phi4":             "Phi-4",
}

# Big 5 trait order
BIG5_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
BIG5_LABELS = ["Open.", "Consc.", "Extra.", "Agree.", "Neuro."]

# TPB constructs
TPB_CONSTRUCTS = ["attitude", "subjective_norm", "pbc", "intention"]
TPB_LABELS = ["Att.", "SN", "PBC", "Int."]

# Tasks (5 for behavior figures - Honesty splits into overconfidence + consistency)
# For SR figures, the underlying TPB tasks are still 4: cct, sycophancy, honesty, iat
TASKS_SR = ["cct", "sycophancy", "honesty", "iat"]
TASKS_BEH = ["cct", "sycophancy", "honesty_overconf", "honesty_consist", "iat"]

TASK_LABELS = {
    "cct":              "Risk Taking",
    "sycophancy":       "Sycophancy",
    "honesty":          "Honesty",
    "honesty_overconf": "Epistemic Honesty",
    "honesty_consist":  "Self-Reflective Honesty",
    "iat":              "Stereotyping",
}

# Short abbreviations used on x-axes (full labels go in the legend in empty cell)
TASK_LABELS_SHORT = {
    "cct":              "Risk",
    "sycophancy":       "Syco.",
    "honesty":          "Honesty",
    "honesty_overconf": "Epist.",
    "honesty_consist":  "Self-Refl.",
    "iat":              "Stereo.",
}

# Visual styling
COLOR_GRID     = "#1f77b4"   # blue
COLOR_PERSONAS = "#d62728"   # red
TASK_COLORS = {
    "cct":              "#1f77b4",
    "sycophancy":       "#ff7f0e",
    "honesty":          "#2ca02c",
    "honesty_overconf": "#2ca02c",
    "honesty_consist":  "#8c564b",
    "iat":              "#9467bd",
}

GRID_LAYOUT = (2, 6)   # rows x cols for 11 panels (one cell empty)


# ---------------------------------------------------------------------------
# Behavioral mapping table (the appendix Table 2 analog)
# ---------------------------------------------------------------------------

SCALE_MAPPING_TABLE = pd.DataFrame([
    {
        "Task": "Risk Taking",
        "Raw range": "0...32 cards",
        "Mapping to 1-5": "1 + 4(x/32)",
        "Neutral/Mid -> Mapped": "16 -> 3.0 (moderate risk)",
        "High value means": "More risk-seeking",
    },
    {
        "Task": "Stereotyping",
        "Raw range": "-1...1 d-score",
        "Mapping to 1-5": "3 + 2x",
        "Neutral/Mid -> Mapped": "0 -> 3.0 (no implicit pref.)",
        "High value means": "Stronger implicit association; sign gives direction",
    },
    {
        "Task": "Sycophancy",
        "Raw range": "0...100% flip rate",
        "Mapping to 1-5": "1 + 4(x/100)",
        "Neutral/Mid -> Mapped": "50% -> 3.0 (half the time)",
        "High value means": "More frequent overriding",
    },
    {
        "Task": "Epistemic Honesty",
        "Raw range": "-100...100 pp (mean Delta-confidence, signed)",
        "Mapping to 1-5": "3 + x/50",
        "Neutral/Mid -> Mapped": "0 -> 3.0 (perfect calibration on avg.)",
        "High value means": "Positive: overconfident; negative: under-confident",
    },
    {
        "Task": "Self-Reflective Honesty",
        "Raw range": "0...100% C1=C2 consistency rate",
        "Mapping to 1-5": "1 + 4(x/100)",
        "Neutral/Mid -> Mapped": "50% -> 3.0 (half consistent)",
        "High value means": "More C1-C2 consistency",
    },
])


# ---------------------------------------------------------------------------
# Behavioral score -> 1-5 normalization
# ---------------------------------------------------------------------------

def map_behavior_to_15(task: str, x: pd.Series) -> pd.Series:
    """Apply task-specific mapping from raw behavior to 1-5 scale, clipping inputs to raw range."""
    x = pd.to_numeric(x, errors="coerce")
    if task == "cct":
        # 0..32 -> 1 + 4(x/32)
        x = x.clip(0, 32)
        return 1 + 4 * (x / 32.0)
    elif task == "sycophancy":
        # input is sycophancy_rate in [0,1]; convert to percent then map
        rate_pct = (x * 100.0).clip(0, 100)
        return 1 + 4 * (rate_pct / 100.0)
    elif task == "honesty_overconf":
        # x is mean_confidence_delta in raw units (~-2.5..+2.5 on 0-10 confidence scale).
        # Multiply by 10 to convert to percentage-points (-100..+100), then map per
        # Personality Illusion: 3 + x/50, where 0 -> 3.0 (perfect calibration).
        pp = (x * 10.0).clip(-100, 100)
        return 3 + (pp / 50.0)
    elif task == "honesty_consist":
        # x is mean_abs_confidence_delta in raw units (0..10 on confidence scale).
        # Convert to consistency rate: rate% = (10 - abs_delta) / 10 * 100.
        # Then map per Personality Illusion's Self-Reflective Honesty: 1 + 4(rate/100).
        # Simplifies to: 5 - 0.4 * abs_delta (clipped).
        abs_delta = x.clip(0, 10)
        return 5 - 0.4 * abs_delta
    elif task == "iat":
        # -1..1 d-score -> 3 + 2x
        x = x.clip(-1, 1)
        return 3 + 2 * x
    else:
        raise ValueError(f"Unknown task: {task}")


def behavior_column_for_task(task: str, columns: Optional[list] = None) -> Optional[str]:
    """Return the raw column name in the runs CSV that contains the per-task behavior.
    Tries both 'beh__<col>' (within-session combined_runs) and '<col>' (between-session beh-only runs).
    If `columns` is provided, returns the actual matching column or None if not found."""
    raw_name = {
        "cct":              "mean_k",
        "sycophancy":       "sycophancy_rate",
        "honesty_overconf": "mean_confidence_delta",
        "honesty_consist":  "mean_abs_confidence_delta",
        "iat":              "mean_bias_score",
    }[task]
    if columns is None:
        return f"beh__{raw_name}"
    for cand in (f"beh__{raw_name}", raw_name):
        if cand in columns:
            return cand
    return None


# ---------------------------------------------------------------------------
# Mean + 95% CI helper
# ---------------------------------------------------------------------------

def mean_band(values: pd.Series, method: str = "sd") -> tuple[float, float, float]:
    """Return (mean, low, high) where the band is method-dependent.

    method='sd'    : +/-1 SD around the mean (variability across conditions; default)
    method='ci95'  : 95% CI of the mean (analytic, +/- 1.96 * SE)
    method='ci99'  : 99% CI of the mean (matches Personality Illusion's choice)
    """
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    m = v.mean()
    if len(v) == 1:
        return (m, m, m)
    if method == "sd":
        s = v.std(ddof=1)
        return (m, m - s, m + s)
    elif method == "ci99":
        se = v.std(ddof=1) / np.sqrt(len(v))
        return (m, m - 2.576 * se, m + 2.576 * se)
    else:  # ci95
        se = v.std(ddof=1) / np.sqrt(len(v))
        return (m, m - 1.96 * se, m + 1.96 * se)


# Backwards-compatible alias
def mean_ci(values: pd.Series) -> tuple[float, float, float]:
    return mean_band(values, method="ci95")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_big5_within(within_root: Path, induction: str) -> pd.DataFrame:
    """Load Big5 within-session SR. induction in {grid, personas}."""
    parts = []
    for task in TASKS_SR:
        f = within_root / induction / f"big5_psycohere_{induction}" / task / "big5" / "combined_runs.csv"
        if f.exists():
            df = pd.read_csv(f, on_bad_lines="skip", engine="python")
            df["task"] = task
            df["induction"] = induction
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def load_tpb_within(within_root: Path, induction: str) -> pd.DataFrame:
    """Load TPB within-session SR + behavior, all 4 tasks both policies.

    For Honesty, emits TWO copies of each row (one tagged honesty_overconf, one
    honesty_consist) so the downstream behavior loop can plot both Honesty
    dimensions separately. SR columns are identical between the two copies."""
    parts = []
    for task in TASKS_SR:
        task_root = within_root / induction / f"tpb_{task}_psycohere_{induction}"
        if not task_root.exists():
            continue
        for policy_dir in task_root.iterdir():
            if not policy_dir.is_dir():
                continue
            f = policy_dir / "combined_runs.csv"
            if f.exists():
                df = pd.read_csv(f, on_bad_lines="skip", engine="python")
                df["policy"] = policy_dir.name
                df["induction"] = induction
                if task == "honesty":
                    # Split honesty rows into two task-dimension copies for Figure C
                    df_oc = df.copy(); df_oc["task"] = "honesty_overconf"
                    df_co = df.copy(); df_co["task"] = "honesty_consist"
                    parts.extend([df_oc, df_co])
                    # Also keep original-task copy for Figure B (TPB SR uses task='honesty')
                    df_orig = df.copy(); df_orig["task"] = "honesty"
                    parts.append(df_orig)
                else:
                    df["task"] = task
                    parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def load_tpb_between_behavior(between_root: Path, induction: str) -> pd.DataFrame:
    """Load between-session BEHAVIOR (session_beh side), all 4 tasks.

    Between-session runs are TRIAL-level (one row per dilemma/question/test),
    so we aggregate to per-(model, condition) outcomes first.

    Returns a long-form DataFrame with columns including:
      model_key, task, induction, and a task-specific outcome column matching
      the within-session naming (beh__mean_k, beh__sycophancy_rate,
      beh__mean_confidence_delta, beh__mean_bias_score).
    """
    parts = []
    sb_root = between_root / induction / "session_beh"
    if not sb_root.exists():
        return pd.DataFrame()

    task_specs = {
        "cct": {
            "dirs":      ["cct-psycohere-grid", "cct-psycohere-personas",
                          "cct_psycohere_grid", "cct_psycohere_personas"],
            "fname":     "cct_runs.csv",
            "trial_col": None,  # CCT runs file may already be at run-level with mean_k
            "out_col":   "beh__mean_k",
        },
        "sycophancy": {
            "dirs":      ["sycophancy-psycohere-grid", "sycophancy-psycohere-personas",
                          "sycophancy_psycohere_grid", "sycophancy_psycohere_personas"],
            "fname":     "sycophancy_runs.csv",
            "trial_col": "sycophancy",
            "out_col":   "beh__sycophancy_rate",
        },
        "honesty": {
            "dirs":      ["honesty-psycohere-grid", "honesty-psycohere-personas",
                          "honesty_psycohere_grid", "honesty_psycohere_personas"],
            "fname":     "honesty_runs.csv",
            "trial_col": "confidence_delta",  # for overconf: mean signed; for consist: mean abs
            "out_col":   "beh__mean_confidence_delta",
        },
        "iat": {
            "dirs":      ["iat-psycohere-grid", "iat-psycohere-personas",
                          "iat_psycohere_grid", "iat_psycohere_personas"],
            "fname":     "iat_runs.csv",
            "trial_col": "bias",
            "out_col":   "beh__mean_bias_score",
        },
    }

    for task, spec in task_specs.items():
        for dn in spec["dirs"]:
            task_root = sb_root / dn
            if not task_root.exists():
                continue
            for sub in task_root.rglob(spec["fname"]):
                df = pd.read_csv(sub, on_bad_lines="skip", engine="python")
                if "model_key" not in df.columns:
                    continue

                # Group identifies a single condition (run)
                group_cols = [c for c in [
                    "model_key", "model_id", "seed", "temperature", "top_p",
                    "system_prompt", "persona_label", "prompt_variant", "condition_id",
                ] if c in df.columns]
                if not group_cols or "model_key" not in group_cols:
                    group_cols = ["model_key"]

                if spec["trial_col"] and spec["trial_col"] in df.columns:
                    df[spec["trial_col"]] = pd.to_numeric(df[spec["trial_col"]], errors="coerce")

                    if task == "honesty":
                        # Compute BOTH signed mean (overconfidence) and mean of abs (consistency)
                        agg_signed = (df.groupby(group_cols, dropna=False)[spec["trial_col"]]
                                        .mean().reset_index()
                                        .rename(columns={spec["trial_col"]: "beh__mean_confidence_delta"}))
                        df["_abs"] = df[spec["trial_col"]].abs()
                        agg_abs = (df.groupby(group_cols, dropna=False)["_abs"]
                                     .mean().reset_index()
                                     .rename(columns={"_abs": "beh__mean_abs_confidence_delta"}))
                        # Emit one DataFrame per honesty dimension
                        agg_oc = agg_signed.copy(); agg_oc["task"] = "honesty_overconf"
                        agg_co = agg_abs.copy();    agg_co["task"] = "honesty_consist"
                        agg_oc["induction"] = induction
                        agg_co["induction"] = induction
                        parts.extend([agg_oc, agg_co])
                        continue
                    else:
                        agg = (df.groupby(group_cols, dropna=False)[spec["trial_col"]]
                                 .mean().reset_index()
                                 .rename(columns={spec["trial_col"]: spec["out_col"]}))
                else:
                    # CCT or any task already at run-level
                    if "mean_k" in df.columns:
                        agg = (df.drop_duplicates(subset=group_cols)
                                 [group_cols + ["mean_k"]]
                                 .rename(columns={"mean_k": spec["out_col"]}))
                    else:
                        rounds_col = next((c for c in ["n_flips", "k", "cards_flipped"]
                                           if c in df.columns), None)
                        if rounds_col is None:
                            continue
                        df[rounds_col] = pd.to_numeric(df[rounds_col], errors="coerce")
                        agg = (df.groupby(group_cols, dropna=False)[rounds_col]
                                 .mean().reset_index()
                                 .rename(columns={rounds_col: spec["out_col"]}))

                agg["task"] = task
                agg["induction"] = induction
                parts.append(agg)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def setup_panel_grid(title: str, ylabel: str, ymin: float = 1.0, ymax: float = 5.0):
    """Create the panel grid with one empty cell."""
    fig, axes = plt.subplots(*GRID_LAYOUT, figsize=(18, 7.0), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    # Hide the 12th panel (we have 11 models)
    axes_flat[-1].set_visible(False)
    for ax in axes_flat[:-1]:
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.2, linestyle=":")
        ax.tick_params(axis="x", labelsize=12, rotation=30)
        ax.tick_params(axis="y", labelsize=12)
    fig.suptitle(title, fontsize=18, y=0.98, fontweight="bold")
    fig.supylabel(ylabel, fontsize=14, x=0.005)
    return fig, axes_flat


def render_corner_legend(ax, header: str, items: list[tuple[str, str, str]]):
    """Render a header + abbreviation legend in the empty 12th panel.
    items: list of (color_or_None, abbrev, full_name) tuples.
    If color_or_None is None, the row has no color swatch (pure abbrev = full).
    If a color is supplied, a small filled marker of that color precedes the abbrev.
    Equals signs are aligned to the right edge of the longest abbreviation across rows.
    """
    ax.set_visible(True)
    ax.axis("off")
    ax.set_title(header, fontsize=13, fontweight="bold", loc="left", pad=4)

    n = len(items)
    if n == 0:
        return

    fig = ax.figure
    fig.canvas.draw()  # ensure renderer exists for bbox queries

    top_y = 0.85
    line_step = 0.85 / max(n, 1)

    # Pass 1: render the bold abbreviations and measure their widths.
    abbrev_artists = []  # (item_index, text_artist)
    for i, (color, abbrev, full) in enumerate(items):
        y = top_y - i * line_step
        text_x = 0.10 if color is not None else 0.0
        if color is not None:
            ax.plot([0.04], [y], marker="o", color=color, markersize=10,
                    transform=ax.transAxes, clip_on=False)
        if abbrev and abbrev != full:
            t = ax.text(text_x, y, abbrev,
                        transform=ax.transAxes, fontsize=12, fontweight="bold", va="center")
            abbrev_artists.append((i, text_x, y, t, full))
        else:
            ax.text(text_x, y, full, transform=ax.transAxes, fontsize=12, va="center")

    if not abbrev_artists:
        return

    # Pass 2: find the rightmost edge of any abbreviation, in axes coordinates.
    inv = ax.transAxes.inverted()
    max_right_x = 0.0
    for (_, text_x, _, t, _) in abbrev_artists:
        bbox = t.get_window_extent()
        right_axes = inv.transform((bbox.x1, bbox.y0))[0]
        if right_axes > max_right_x:
            max_right_x = right_axes

    # Pass 3: place the "= full" text at a consistent column for all rows.
    eq_x = max_right_x + 0.04  # small horizontal gap after the longest abbreviation
    for (i, text_x, y, t, full) in abbrev_artists:
        ax.text(eq_x, y, "=  " + full,
                transform=ax.transAxes, fontsize=12, va="center")


def save_figure(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    # Tight title-to-plot gap (rect top close to 1.0); leave a little space for suptitle
    fig.tight_layout(rect=[0.01, 0, 1, 0.94])
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {pdf} + {png}")


# ---------------------------------------------------------------------------
# Figure A: Big 5 SR fingerprints
# ---------------------------------------------------------------------------

def make_fig_A(within_root: Path, out_dir: Path, band: str = "sd"):
    print("Building Figure A: Big 5 SR fingerprints (within-session)")
    df_grid = load_big5_within(within_root, "grid")
    df_per  = load_big5_within(within_root, "personas")
    if df_grid.empty and df_per.empty:
        print("  No Big5 data found; skipping.")
        return

    # Big5 doesn't depend on task -- aggregate across all 4 tasks per (model, induction)
    fig, axes = setup_panel_grid("Big Five self-report fingerprints (within-session)", "Mean Likert (1-5)", 1, 5)

    for i, model in enumerate(MODELS_ORDER):
        ax = axes[i]
        # Plot grid line
        for df, color, label in [(df_grid, COLOR_GRID, "Grid"),
                                  (df_per, COLOR_PERSONAS, "Personas")]:
            if df.empty:
                continue
            sub = df[df["model_key"] == model]
            ys, lo, hi = [], [], []
            for trait in BIG5_TRAITS:
                col = f"{trait}_mean"
                if col not in sub.columns:
                    ys.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                    continue
                m, l, h = mean_band(sub[col], method=band)
                ys.append(m); lo.append(l); hi.append(h)
            ys = np.array(ys); lo = np.array(lo); hi = np.array(hi)
            xs = np.arange(len(BIG5_TRAITS))
            ax.plot(xs, ys, color=color, linewidth=1.6, label=label, marker="o", markersize=4)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.18)
        ax.set_title(MODEL_PRETTY.get(model, model), fontsize=14, fontweight="bold")
        ax.set_xticks(np.arange(len(BIG5_LABELS)))
        ax.set_xticklabels(BIG5_LABELS, rotation=30, ha="right", fontsize=12)

    # Empty-cell legend: induction colors + Big5 trait abbreviations
    legend_items = [
        (COLOR_GRID, "", "Grid (parameter perturbation)"),
        (COLOR_PERSONAS, "", "Personas (persona prompting)"),
        (None, "", ""),  # spacer
        (None, "Open.",  "Openness"),
        (None, "Consc.", "Conscientiousness"),
        (None, "Extra.", "Extraversion"),
        (None, "Agree.", "Agreeableness"),
        (None, "Neuro.", "Neuroticism"),
    ]
    render_corner_legend(axes[-1], "Legend", legend_items)

    save_figure(fig, out_dir, "figA_big5_sr")


# ---------------------------------------------------------------------------
# Figure B: TPB SR fingerprints (one per induction)
# ---------------------------------------------------------------------------

def make_fig_B(within_root: Path, induction: str, out_dir: Path, band: str = "sd"):
    print(f"Building Figure B-{induction}: TPB SR fingerprints (within-session, {induction})")
    df = load_tpb_within(within_root, induction)
    if df.empty:
        print(f"  No TPB {induction} data; skipping.")
        return

    fig, axes = setup_panel_grid(
        f"TPB self-report fingerprints (within-session, {induction} induction)",
        "Mean Likert (1-7)", 1, 7
    )

    for i, model in enumerate(MODELS_ORDER):
        ax = axes[i]
        sub_model = df[df["model_key"] == model]
        for task in TASKS_SR:
            sub_task = sub_model[sub_model["task"] == task]
            if sub_task.empty:
                continue
            ys, lo, hi = [], [], []
            for c in TPB_CONSTRUCTS:
                col = f"{c}_mean"
                if col not in sub_task.columns:
                    ys.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                    continue
                m, l, h = mean_band(sub_task[col], method=band)
                ys.append(m); lo.append(l); hi.append(h)
            ys = np.array(ys); lo = np.array(lo); hi = np.array(hi)
            xs = np.arange(len(TPB_CONSTRUCTS))
            color = TASK_COLORS[task]
            ax.plot(xs, ys, color=color, linewidth=1.4, label=TASK_LABELS[task],
                    marker="o", markersize=3.5)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.13)
        ax.set_title(MODEL_PRETTY.get(model, model), fontsize=14, fontweight="bold")
        ax.set_xticks(np.arange(len(TPB_LABELS)))
        ax.set_xticklabels(TPB_LABELS, rotation=30, ha="right", fontsize=12)

    # Empty-cell legend: task colors + TPB construct abbreviations
    legend_items = [
        (TASK_COLORS["cct"],         "", "Risk Taking (CCT)"),
        (TASK_COLORS["sycophancy"],  "", "Sycophancy"),
        (TASK_COLORS["honesty"],     "", "Honesty"),
        (TASK_COLORS["iat"],         "", "Stereotyping (IAT)"),
        (None, "", ""),  # spacer
        (None, "Att.", "Attitude"),
        (None, "SN",   "Subjective Norm"),
        (None, "PBC",  "Perceived Behav. Control"),
        (None, "Int.", "Intention"),
    ]
    render_corner_legend(axes[-1], "Legend", legend_items)

    save_figure(fig, out_dir, f"figB_tpb_sr_{induction}")


# ---------------------------------------------------------------------------
# Figure C: Behavioral fingerprints (one per session type)
# ---------------------------------------------------------------------------

def make_fig_C(within_root: Path, between_root: Path, session: str, out_dir: Path, band: str = "sd"):
    """session in {'same', 'separate'}"""
    print(f"Building Figure C-{session}: Behavioral fingerprints ({session}-session)")

    if session == "same":
        df_grid = load_tpb_within(within_root, "grid")
        df_per  = load_tpb_within(within_root, "personas")
        title = "Behavioral fingerprints (same-session)"
    else:
        df_grid = load_tpb_between_behavior(between_root, "grid")
        df_per  = load_tpb_between_behavior(between_root, "personas")
        title = "Behavioral fingerprints (separate-sessions)"

    if df_grid.empty and df_per.empty:
        print(f"  No data for C-{session}; skipping.")
        return

    fig, axes = setup_panel_grid(title, "Mapped score (1-5)", 1, 5)

    for i, model in enumerate(MODELS_ORDER):
        ax = axes[i]
        for df, color, label in [(df_grid, COLOR_GRID, "Grid"),
                                  (df_per, COLOR_PERSONAS, "Personas")]:
            if df.empty:
                continue
            sub_model = df[df["model_key"] == model]
            ys, lo, hi = [], [], []
            for task in TASKS_BEH:
                sub_task = sub_model[sub_model["task"] == task]
                col = behavior_column_for_task(task, sub_task.columns.tolist())
                if col is None or sub_task.empty:
                    ys.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                    continue
                # Map to 1-5 scale
                mapped = map_behavior_to_15(task, sub_task[col])
                m, l, h = mean_band(mapped, method=band)
                ys.append(m); lo.append(l); hi.append(h)
            ys = np.array(ys); lo = np.array(lo); hi = np.array(hi)
            xs = np.arange(len(TASKS_BEH))
            ax.plot(xs, ys, color=color, linewidth=1.6, label=label, marker="o", markersize=4)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.18)
        ax.set_title(MODEL_PRETTY.get(model, model), fontsize=14, fontweight="bold")
        ax.set_xticks(np.arange(len(TASKS_BEH)))
        ax.set_xticklabels([TASK_LABELS_SHORT[t] for t in TASKS_BEH],
                           rotation=30, ha="right", fontsize=12)
        ax.axhline(3.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    # Empty-cell legend: induction colors + behavior task abbreviations
    legend_items = [
        (COLOR_GRID,     "", "Grid (parameter perturbation)"),
        (COLOR_PERSONAS, "", "Personas (persona prompting)"),
        (None, "", ""),  # spacer
        (None, "Risk",       "Risk Taking"),
        (None, "Syco.",      "Sycophancy"),
        (None, "Epist.",     "Epistemic Honesty"),
        (None, "Self-Refl.", "Self-Reflective Honesty"),
        (None, "Stereo.",    "Stereotyping"),
    ]
    render_corner_legend(axes[-1], "Legend", legend_items)

    save_figure(fig, out_dir,
                f"figC_behavior_{'same_session' if session == 'same' else 'separate_sessions'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--within_root", required=True,
                    help="Path to .../within (containing grid/ and personas/ subdirs)")
    ap.add_argument("--between_root", required=False, default=None,
                    help="Path to .../between (only needed for separate-session figures)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--skip_separate", action="store_true",
                    help="Skip the separate-session behavior figure (if between data unavailable)")
    ap.add_argument("--band", choices=["sd", "ci95", "ci99"], default="sd",
                    help="What the shaded band represents (default: sd, +/-1 SD across "
                         "conditions; alternatives: ci95 or ci99 for CI of the mean)")
    args = ap.parse_args()

    within_root = Path(args.within_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the scale-mapping table (Table 2 analog) for the appendix
    SCALE_MAPPING_TABLE.to_csv(out_dir / "scale_mapping.csv", index=False)
    print(f"Saved scale mapping table: {out_dir / 'scale_mapping.csv'}")
    band_label = {"sd": "+/-1 SD across conditions",
                  "ci95": "95% CI of the mean",
                  "ci99": "99% CI of the mean"}[args.band]
    print(f"Shaded bands represent: {band_label}")
    print()

    # Figure A
    make_fig_A(within_root, out_dir, band=args.band)

    # Figure B (one per induction)
    make_fig_B(within_root, "grid", out_dir, band=args.band)
    make_fig_B(within_root, "personas", out_dir, band=args.band)

    # Figure C (one per session type)
    make_fig_C(within_root, None, "same", out_dir, band=args.band)

    if args.skip_separate:
        print("Skipped Figure C-separate (--skip_separate set).")
    elif args.between_root is None:
        print("Skipped Figure C-separate (no --between_root provided).")
    else:
        between_root = Path(args.between_root)
        make_fig_C(within_root, between_root, "separate", out_dir, band=args.band)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
