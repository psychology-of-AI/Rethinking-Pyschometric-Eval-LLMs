#!/usr/bin/env python3
"""
instruction_sensitivity.py
==========================
Decomposes within-session SR-Behaviour coupling into its two sources:

  (1) Policy-driven coupled shift: SR and Behaviour both shift between
      Policy A and Policy B (priming-style coherence; carried by the
      policy framing being in the prompt window).
  (2) Stable dispositional structure: independent of policy, models that
      score higher on SR also produce more policy-aligned behaviour
      (common-cause coherence; what survives session separation).

The within-session "instruction-sensitivity ratio"
    R = |d|_SR  /  |d|_Beh
indexes which source dominates.  R < 1 means behaviour is more policy-
sensitive than SR (Sycophancy-like, Source 1 dominates → cross-session
coherence collapses).  R > 1 means SR is more policy-sensitive than
behaviour (Honesty/IAT-like, Source 2 component is large → cross-session
coherence survives, possibly as inversion).

Outputs
-------
  per_cell_shifts.csv            per-(model × task × construct) Cohen's d
                                 for SR and Behaviour, and the ratio
  task_summary.csv               task-level pooled |d|_SR, |d|_Beh, ratio,
                                 paired Wilcoxon p across models
  ridgeline_distributions.{pdf,png}
                                 four-panel ridgeline figure (1 panel per
                                 task) showing SR and Behaviour kernel
                                 densities split by Policy A vs Policy B,
                                 z-scored within (model × measure) so that
                                 SR and Behaviour share an x-axis
  scatter_taxonomy.{pdf,png}     |d|_SR vs |d|_Beh scatter, log-log,
                                 with diagonal and quadrant labels

Usage
-----
Two input modes are supported:

  Mode A — master CSVs (matches the rest of the rq*.py pipeline):
    python instruction_sensitivity.py \
        --cct_master     results/.../cct_master.csv \
        --syc_master     results/.../sycophancy_master.csv \
        --honesty_master results/.../honesty_master.csv \
        --iat_master     results/.../iat_master.csv \
        --out_dir        results/.../analysis/instruction_sensitivity

  Mode B — within-session combined_runs CSVs (one folder per task):
    python instruction_sensitivity.py \
        --within_root    results/psycohere_v1/within/grid \
        --out_dir        results/.../analysis/instruction_sensitivity

If both are supplied, master CSVs take precedence.

Notes
-----
- Within-session, parameter-grid only.  Persona induction can be added by
  pointing --within_root at .../within/personas (mode B) — the script
  auto-detects which one is provided and labels outputs accordingly.
- SR construct used for the Cohen's d is Attitude by default (--sr_construct
  attitude); pass --sr_construct intention to use Intention instead.  All four
  constructs are written to per_cell_shifts.csv regardless.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import gaussian_kde, wilcoxon


# ---------------------------------------------------------------------------
# Task / policy / behaviour-column registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict] = {
    "cct": dict(
        label="Risk-taking (CCT)",
        policies=("loss_averse", "gain_seeking"),
        beh_col="beh__mean_k",
        # Trial-level column in merged/between/grid/tpb_x_<task>_long.csv used to
        # aggregate to per-condition between-session behaviour. May differ in
        # scale from beh_col above; the panel renderer min-max normalises per
        # task across all sources before joint-z-scoring per model.
        between_trial_col="mean_k",
        beh_label="cards flipped",
        beh_higher_aligns_with="B",
        within_dir="tpb_cct_psycohere_grid",  # under within/grid/...
    ),
    "sycophancy": dict(
        label="Sycophancy",
        policies=("independent_judgment", "defer_when_uncertain"),
        beh_col="beh__sycophancy_rate",
        between_trial_col="sycophancy",
        beh_label="sycophancy rate",
        beh_higher_aligns_with="B",
        within_dir="tpb_sycophancy_psycohere_grid",
    ),
    "honesty": dict(
        label="Honesty",
        policies=("calibrated_confidence", "keep_confidence_stable"),
        beh_col="beh__mean_abs_confidence_delta",
        between_trial_col="inconsistency_abs",
        beh_label="|Δconfidence|",
        beh_higher_aligns_with="A",
        within_dir="tpb_honesty_psycohere_grid",
    ),
    "iat": dict(
        label="Implicit bias (IAT)",
        policies=("unbiased_categorization", "intuitive_fast"),
        beh_col="beh__mean_bias_score",
        between_trial_col="bias",
        beh_label="IAT bias",
        beh_higher_aligns_with="B",
        within_dir="tpb_iat_psycohere_grid",
    ),
}

TASK_ORDER = ["sycophancy", "cct", "honesty", "iat"]  # by ratio

SR_CONSTRUCTS = ["attitude_mean", "subjective_norm_mean", "pbc_mean", "intention_mean"]

# Colours roughly matching the paper figures.
TASK_COLORS = {
    "cct": "#666666",
    "sycophancy": "#d95f02",
    "honesty": "#1b9e77",
    "iat": "#7570b3",
}
POLICY_COLORS = ("#3b82f6", "#ef4444")   # A=blue, B=red
ROW_BEH = "#0f172a"
ROW_SR = "#475569"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_within_combined(within_root: Path, task: str) -> pd.DataFrame:
    """
    Load combined_runs.csv for both policies of a task under within_root and
    return a long DataFrame with columns:
        model_key, temperature, seed, persona_label, policy_id,
        attitude_mean, subjective_norm_mean, pbc_mean, intention_mean,
        beh_value
    """
    info = TASKS[task]
    polA, polB = info["policies"]
    task_dir = within_root / info["within_dir"]
    if not task_dir.exists():
        # tolerate persona naming
        alt = info["within_dir"].replace("_grid", "_personas")
        task_dir = within_root / alt
    if not task_dir.exists():
        raise FileNotFoundError(
            f"Could not find {info['within_dir']} (or _personas variant) under {within_root}"
        )

    frames = []
    for pol in (polA, polB):
        path = task_dir / pol / "combined_runs.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        df = pd.read_csv(path, low_memory=False)
        df["policy_id"] = pol
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # status filter
    if "sr_status" in df.columns:
        df = df[df["sr_status"] == "ok"]
    if "beh_status" in df.columns:
        df = df[df["beh_status"] == "ok"]

    # numeric coerce on relevant columns
    num_cols = SR_CONSTRUCTS + [info["beh_col"]]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # rename behaviour column to a canonical name
    df = df.rename(columns={info["beh_col"]: "beh_value"})

    keep = [
        "model_key", "temperature", "seed", "persona_label", "policy_id",
        *SR_CONSTRUCTS, "beh_value",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def _load_between_aggregated(between_root: Path, task: str) -> pd.DataFrame:
    """
    Load merged/between/grid/tpb_x_<task>_long.csv and aggregate per-condition
    behaviour means. Returns a DataFrame with columns:
        model_key, temperature, seed, persona_label, policy_id, beh_value
    The 'policy_id' is preserved (the policy that was used in the SR session)
    but the behaviour itself was generated without it in context.
    """
    info = TASKS[task]
    trial_col = info.get("between_trial_col")
    if trial_col is None:
        raise ValueError(f"No 'between_trial_col' configured for task {task!r}")
    candidates = [
        between_root / f"tpb_x_{task}_long.csv",
        # tolerate alt naming in case sycophancy etc. use different file names
        between_root / f"tpb_x_{task[:3]}_long.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"Could not find tpb_x_{task}_long.csv under {between_root}"
        )
    df = pd.read_csv(path, low_memory=False)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    df[trial_col] = pd.to_numeric(df[trial_col], errors="coerce")
    cond = ["model_key", "temperature", "seed", "persona_label", "policy_id"]
    cond = [c for c in cond if c in df.columns]
    agg = df.groupby(cond)[trial_col].mean().reset_index().rename(
        columns={trial_col: "beh_value"}
    )
    return agg


def _load_master(master_csv: Path, task: str) -> pd.DataFrame:
    """
    Load a master CSV produced by the analyze_<task>_psycohere.py pipeline.
    Master CSVs are long (one row per behavioural trial with SR fields
    repeated). We deduplicate to one row per condition for SR; behavioural
    aggregation is done by mean.
    """
    info = TASKS[task]
    df = pd.read_csv(master_csv, low_memory=False)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    # within-session only, grid only, between-session is excluded for this analysis
    if "session_type" in df.columns:
        df = df[df["session_type"].isin(["within", "within_session"])]
    if "perturbation" in df.columns:
        df = df[df["perturbation"] == "grid"]

    beh_col = info["beh_col"].replace("beh__", "")  # master CSVs sometimes drop the prefix
    # Try to detect the right behaviour column name
    candidates = [info["beh_col"], beh_col]
    for col in candidates:
        if col in df.columns:
            beh_actual = col
            break
    else:
        # final fallback: align_score as a normalised proxy (less interpretable
        # but avoids hard failure)
        beh_actual = "align_score" if "align_score" in df.columns else None
    if beh_actual is None:
        raise ValueError(
            f"Could not find behaviour column for {task} in {master_csv}; "
            f"looked for {candidates}"
        )

    for c in SR_CONSTRUCTS + [beh_actual]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # condition key
    cond_key = ["model_key", "temperature", "seed", "persona_label", "policy_id"]
    cond_key = [c for c in cond_key if c in df.columns]

    # SR is constant per condition (one TPB run); take first
    sr = df.groupby(cond_key)[SR_CONSTRUCTS].first().reset_index()
    # Behaviour is per-trial; aggregate to mean
    beh = df.groupby(cond_key)[beh_actual].mean().reset_index().rename(columns={beh_actual: "beh_value"})
    out = sr.merge(beh, on=cond_key, how="inner")
    return out


def load_all(args: argparse.Namespace) -> Tuple[Dict[str, pd.DataFrame], str, Optional[Dict[str, pd.DataFrame]]]:
    """
    Returns (within_data, induction_label, between_data_or_None).

    within_data: dict {task: long_df} with SR + behaviour from within-session
    induction_label: 'grid' or 'personas'
    between_data: optional dict {task: long_df} with between-session behaviour only
                   (no SR fields), aggregated per condition
    """
    masters = dict(
        cct=args.cct_master, sycophancy=args.syc_master,
        honesty=args.honesty_master, iat=args.iat_master,
    )
    if all(masters.values()):
        out = {t: _load_master(Path(p), t) for t, p in masters.items()}
        induction = args.induction_label or "grid"
        between = None  # master-CSV mode does not provide between data; user can
                        # still pass --between_root to overlay it
        if args.between_root:
            between = {
                t: _load_between_aggregated(Path(args.between_root), t)
                for t in TASKS
            }
        return out, induction, between
    if args.within_root:
        root = Path(args.within_root)
        out = {t: _load_within_combined(root, t) for t in TASKS}
        if args.induction_label:
            induction = args.induction_label
        elif "personas" in str(root):
            induction = "personas"
        else:
            induction = "grid"
        between = None
        if args.between_root:
            between = {
                t: _load_between_aggregated(Path(args.between_root), t)
                for t in TASKS
            }
        return out, induction, between
    raise SystemExit(
        "Must provide either all four --*_master CSVs OR --within_root "
        "(pointing at within/grid or within/personas)."
    )


# ---------------------------------------------------------------------------
# Core stat: Cohen's d between two policies
# ---------------------------------------------------------------------------

def cohen_d_paired_groups(xa: np.ndarray, xb: np.ndarray) -> float:
    """Standardized mean difference (a - b) using pooled SD; signed."""
    xa = pd.to_numeric(pd.Series(xa), errors="coerce").dropna().values
    xb = pd.to_numeric(pd.Series(xb), errors="coerce").dropna().values
    if len(xa) < 2 or len(xb) < 2:
        return np.nan
    pooled = np.sqrt((np.var(xa, ddof=1) + np.var(xb, ddof=1)) / 2)
    if pooled < 1e-9:
        return np.nan
    return (np.mean(xa) - np.mean(xb)) / pooled


def _safe_mean(arr) -> float:
    arr = pd.to_numeric(pd.Series(arr), errors="coerce").dropna().values
    return float(np.mean(arr)) if len(arr) else np.nan


def per_cell_shifts(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    For each (task, model): compute |d| for behaviour and for each SR construct
    (Policy A vs Policy B). Also record the raw policy-conditional means so the
    plotting layer can show the absolute Likert level alongside the standardised
    shift. Returns a long DataFrame.
    """
    rows = []
    for task, df in data.items():
        polA, polB = TASKS[task]["policies"]
        for model in df["model_key"].dropna().unique():
            sub = df[df["model_key"] == model]
            bA = sub[sub["policy_id"] == polA]["beh_value"].values
            bB = sub[sub["policy_id"] == polB]["beh_value"].values
            d_beh = cohen_d_paired_groups(bA, bB)
            row = dict(
                task=task, model=model,
                d_beh=d_beh,
                mean_beh_A=_safe_mean(bA),
                mean_beh_B=_safe_mean(bB),
                n_A=len(bA), n_B=len(bB),
            )
            for c in SR_CONSTRUCTS:
                if c not in sub.columns:
                    row[f"d_sr_{c.replace('_mean','')}"] = np.nan
                    row[f"mean_sr_{c.replace('_mean','')}_A"] = np.nan
                    row[f"mean_sr_{c.replace('_mean','')}_B"] = np.nan
                    continue
                sA = sub[sub["policy_id"] == polA][c].values
                sB = sub[sub["policy_id"] == polB][c].values
                row[f"d_sr_{c.replace('_mean','')}"] = cohen_d_paired_groups(sA, sB)
                row[f"mean_sr_{c.replace('_mean','')}_A"] = _safe_mean(sA)
                row[f"mean_sr_{c.replace('_mean','')}_B"] = _safe_mean(sB)
            rows.append(row)
    return pd.DataFrame(rows)


def task_summary(per_cell: pd.DataFrame, sr_construct: str = "attitude") -> pd.DataFrame:
    """
    Aggregate per-cell shifts to task level: |d|_Beh, |d|_SR_<construct>, ratio,
    paired Wilcoxon p across models on absolute values, simple counts. Also
    carries forward the cross-model mean of the raw policy-conditional means
    (Likert for SR; native units for Behaviour) for use in plot annotations.
    """
    sr_col = f"d_sr_{sr_construct}"
    sr_mean_A = f"mean_sr_{sr_construct}_A"
    sr_mean_B = f"mean_sr_{sr_construct}_B"
    rows = []
    for task in TASK_ORDER:
        sub = per_cell[per_cell["task"] == task].copy()
        sub["abs_d_beh"] = sub["d_beh"].abs()
        sub["abs_d_sr"] = sub[sr_col].abs()
        n = len(sub.dropna(subset=["abs_d_beh", "abs_d_sr"]))
        if n < 3:
            rows.append(dict(task=task, n_models=n))
            continue
        paired = sub.dropna(subset=["abs_d_beh", "abs_d_sr"])
        try:
            stat = wilcoxon(paired["abs_d_sr"], paired["abs_d_beh"])
            p = float(stat.pvalue)
        except ValueError:
            p = np.nan
        d_beh = float(np.nanmean(sub["abs_d_beh"]))
        d_sr = float(np.nanmean(sub["abs_d_sr"]))
        rows.append(dict(
            task=task,
            n_models=n,
            mean_abs_d_beh=d_beh,
            mean_abs_d_sr=d_sr,
            ratio_SR_to_Beh=d_sr / d_beh if d_beh > 0 else np.nan,
            wilcoxon_p_SR_vs_Beh=p,
            sign=("SR>Beh" if d_sr > d_beh else "Beh>SR"),
            mean_sr_A=float(np.nanmean(sub[sr_mean_A])),
            mean_sr_B=float(np.nanmean(sub[sr_mean_B])),
            mean_beh_A=float(np.nanmean(sub["mean_beh_A"])),
            mean_beh_B=float(np.nanmean(sub["mean_beh_B"])),
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting: ridgeline distributions
# ---------------------------------------------------------------------------

def _zscore_within(df: pd.DataFrame, value_col: str) -> pd.Series:
    """
    z-score the value column within each model. This puts SR and Beh
    on a comparable scale for the ridgeline x-axis, controlling for
    cross-model differences in absolute level.
    """
    return df.groupby("model_key")[value_col].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 1e-9 else 0.0
    )


def _minmax_normalise_per_model(*series_list: pd.Series) -> Tuple[pd.Series, ...]:
    """
    Min-max normalise each series to [0,1] using a SHARED min and max computed
    per model across all input series. This puts behaviour from different
    sources (within-session, between-session) on the same axis when their raw
    scales differ but their values are linearly comparable in meaning. Returns
    new series in the same order. NaN preserved.
    """
    # Tag every value with its model and source for joint per-model normalisation.
    pieces = []
    for i, s in enumerate(series_list):
        if not isinstance(s.index, pd.MultiIndex):
            piece = s.to_frame("value").copy()
        else:
            piece = s.to_frame("value").copy()
        piece["__src"] = i
        pieces.append(piece.reset_index())
    pooled = pd.concat(pieces, ignore_index=True)
    if "model_key" not in pooled.columns:
        # nothing to do
        return series_list
    grouped = pooled.groupby("model_key")["value"]
    pooled["__min"] = grouped.transform("min")
    pooled["__max"] = grouped.transform("max")
    rng = (pooled["__max"] - pooled["__min"]).replace(0, np.nan)
    pooled["__norm"] = (pooled["value"] - pooled["__min"]) / rng
    out = []
    for i, s in enumerate(series_list):
        sl = pooled[pooled["__src"] == i]["__norm"].reset_index(drop=True)
        sl.index = s.index
        out.append(sl)
    return tuple(out)


def _zscore_joint_per_model(
    *frames_and_cols: Tuple[pd.DataFrame, str],
) -> Tuple[pd.Series, ...]:
    """
    Joint per-model z-score across multiple DataFrames. Computes one
    (mean, std) pair per model by pooling values from every (frame, col)
    pair, then applies that same standardisation to each frame's column.

    This preserves the displacement between sources: if within-session
    behaviour values are systematically higher than between-session
    values for a given model, that shift survives standardisation. By
    contrast, z-scoring each frame independently per model would centre
    each source at z=0 by construction and erase the displacement.

    Returns one pd.Series per (frame, col) tuple, in input order, with
    the same index as the source frame's column.
    """
    pieces = []
    for i, (frame, col) in enumerate(frames_and_cols):
        s = pd.to_numeric(frame[col], errors="coerce")
        piece = pd.DataFrame({
            "__src": i,
            "model_key": frame["model_key"].values,
            "value": s.values,
        })
        pieces.append(piece)
    pooled = pd.concat(pieces, ignore_index=True)
    grouped = pooled.groupby("model_key")["value"]
    means = grouped.transform("mean")
    stds = grouped.transform(lambda x: x.std(ddof=0))
    z = (pooled["value"] - means) / stds.replace(0, np.nan)
    out = []
    cursor = 0
    for i, (frame, _col) in enumerate(frames_and_cols):
        n = len(frame)
        zsl = z.iloc[cursor:cursor + n].reset_index(drop=True)
        zsl.index = frame.index
        out.append(zsl)
        cursor += n
    return tuple(out)


def ridgeline_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    task: str,
    sr_construct: str,
    summary_row: Optional[pd.Series] = None,
    between_df: Optional[pd.DataFrame] = None,
    show_mean_lines: bool = False,
):
    """
    Two- or three-row task ridgeline.

      Row order (bottom to top):
        - SR construct (z-scored within model), split by Policy A/B
        - Behaviour, within-session (z-scored within model), split by Policy A/B
        - Behaviour, between-session (joint-normalised; pooled grey distribution)
          [optional: only rendered when `between_df` is provided]

    For interpretability we orient ALL behaviour rows so that "rightward = more
    aligned with Policy A". When `between_df` is provided, behaviour from both
    sources is min-max-normalised per model BEFORE z-scoring, so within and
    between live on a comparable axis even if their raw scales differ
    (e.g. Honesty's `beh__mean_abs_confidence_delta` vs `inconsistency_abs`).
    """
    info = TASKS[task]
    polA, polB = info["policies"]
    sr_col = f"{sr_construct}_mean"

    plot_df = df.copy()

    # If a between-session frame is provided, pool behaviour and renormalise.
    if between_df is not None and len(between_df) > 0:
        # Min-max normalise per model across (within ∪ between) so the two
        # sources sit on the same [0,1] scale before z-scoring.
        within_beh = plot_df.set_index(
            ["model_key", "temperature", "seed", "persona_label", "policy_id"]
        )["beh_value"]
        between_beh = between_df.set_index(
            ["model_key", "temperature", "seed", "persona_label", "policy_id"]
        )["beh_value"]
        w_norm, b_norm = _minmax_normalise_per_model(within_beh, between_beh)
        plot_df = plot_df.copy()
        plot_df["beh_value_norm"] = plot_df.set_index(
            ["model_key", "temperature", "seed", "persona_label", "policy_id"]
        ).index.map(w_norm.to_dict())
        between_plot = between_df.copy()
        between_plot["beh_value_norm"] = between_plot.set_index(
            ["model_key", "temperature", "seed", "persona_label", "policy_id"]
        ).index.map(b_norm.to_dict())
        # Joint per-model z-score across within ∪ between on the joint-
        # normalised values. This preserves the within-vs-between displacement
        # on the visual axis (per-source z-scoring would have re-centred each
        # source at z=0 by construction, erasing the priming effect).
        w_z, b_z = _zscore_joint_per_model(
            (plot_df, "beh_value_norm"),
            (between_plot, "beh_value_norm"),
        )
        plot_df["beh_z"] = w_z
        between_plot["beh_z"] = b_z
    else:
        plot_df["beh_z"] = _zscore_within(plot_df, "beh_value")
        between_plot = None

    plot_df["sr_z"] = _zscore_within(plot_df, sr_col)
    plot_df = plot_df.dropna(subset=["beh_z", "sr_z"])

    # Reorient Behaviour so the visual axis matches "more aligned with Policy A".
    beh_flipped = info.get("beh_higher_aligns_with", "A") == "B"
    if beh_flipped:
        plot_df["beh_z"] = -plot_df["beh_z"]
        if between_plot is not None:
            between_plot["beh_z"] = -between_plot["beh_z"]

    # Row layout depends on whether we're showing between-session behaviour.
    if between_plot is not None:
        rows = [
            ("Behaviour\n(between-session)", "beh_z", 2.0, "between"),
            ("Behaviour\n(within-session)", "beh_z", 1.0, "within"),
            (f"SR ({sr_construct.title()})\n(within-session)", "sr_z", 0.0, "within"),
        ]
        ax_ymax = 3.25
    else:
        rows = [
            ("Behaviour", "beh_z", 1.0, "within"),
            (f"SR ({sr_construct.title()})", "sr_z", 0.0, "within"),
        ]
        ax_ymax = 2.25

    grid = np.linspace(-3.5, 3.5, 400)
    BANDWIDTH = 0.35

    def _kde(values: np.ndarray) -> np.ndarray:
        if len(values) < 3:
            return np.zeros_like(grid)
        try:
            kde = gaussian_kde(values, bw_method=BANDWIDTH)
            return kde(grid)
        except np.linalg.LinAlgError:
            return np.zeros_like(grid)

    GREY = "#6b7280"

    # Helper for drawing a vertical dotted mean line at z=mean(values),
    # constrained to the KDE's vertical span at that row.
    def _draw_mean_line(values, y0, color):
        v = pd.to_numeric(pd.Series(values), errors="coerce").dropna().values
        if len(v) < 2:
            return
        m = float(np.mean(v))
        if m < grid.min() or m > grid.max():
            return
        # Look up KDE height at the mean (already peak-normalised to 0.85)
        density = _kde(v)
        if density.max() > 0:
            density = density / density.max() * 0.85
        # interpolate height at m
        h = float(np.interp(m, grid, density))
        ax.plot([m, m], [y0, y0 + h], linestyle=":", color=color,
                lw=1.4, alpha=0.95)

    for label, col, y0, source in rows:
        if source == "between":
            # Single pooled distribution, no policy split.
            v = between_plot[col].dropna().values
            density = _kde(v)
            if density.max() > 0:
                density = density / density.max() * 0.85
            ax.fill_between(grid, y0, y0 + density,
                            color=GREY, alpha=0.45, lw=0)
            ax.plot(grid, y0 + density, color=GREY, lw=1.4, alpha=0.9)
            if show_mean_lines:
                _draw_mean_line(v, y0, GREY)
        else:
            for pol, color in zip((polA, polB), POLICY_COLORS):
                v = plot_df.loc[plot_df["policy_id"] == pol, col].values
                density = _kde(v)
                if density.max() > 0:
                    density = density / density.max() * 0.85
                ax.fill_between(grid, y0, y0 + density,
                                color=color, alpha=0.35, lw=0)
                ax.plot(grid, y0 + density, color=color, lw=1.3, alpha=0.9)
                if show_mean_lines:
                    _draw_mean_line(v, y0, color)

        # axis hint at row baseline
        ax.axhline(y0, color="black", lw=0.6, alpha=0.5, xmax=0.65)

        # Row label on the left
        display_label = label
        if "Behaviour" in label and beh_flipped and source != "between":
            display_label = f"{label}\n(axis inverted)"
        elif "Behaviour" in label and beh_flipped and source == "between":
            display_label = f"{label}\n(axis inverted)"
        if source == "between":
            label_color = GREY
        elif "Behaviour" in label:
            label_color = ROW_BEH
        else:
            label_color = ROW_SR
        ax.text(-3.85, y0 + 0.45, display_label, ha="right", va="center",
                fontsize=9.5, color=label_color)

    # |d| and raw-mean annotations on the right gutter.
    if summary_row is not None:
        bd = summary_row.get("mean_abs_d_beh", np.nan)
        sd = summary_row.get("mean_abs_d_sr", np.nan)
        ratio = summary_row.get("ratio_SR_to_Beh", np.nan)
        p = summary_row.get("wilcoxon_p_SR_vs_Beh", np.nan)

        sr_A = summary_row.get("mean_sr_A", np.nan)
        sr_B = summary_row.get("mean_sr_B", np.nan)
        beh_A = summary_row.get("mean_beh_A", np.nan)
        beh_B = summary_row.get("mean_beh_B", np.nan)

        beh_unit = info.get("beh_label", "")
        gutter_x = 3.85

        # Within-session Behaviour annotations (always at y=1.0)
        ax.text(gutter_x, 1.55, f"|d|={bd:.2f}", ha="left", fontsize=8.5,
                color=ROW_BEH, fontweight="bold")
        if not (np.isnan(beh_A) or np.isnan(beh_B)):
            ax.text(gutter_x, 1.30,
                    f"raw: {beh_A:.2f} / {beh_B:.2f}",
                    ha="left", fontsize=7.3, color=ROW_BEH, alpha=0.9)
            if beh_unit:
                ax.text(gutter_x, 1.10, f"({beh_unit})",
                        ha="left", fontsize=6.8, color=ROW_BEH, alpha=0.7,
                        fontstyle="italic")

        # SR row (always at y=0.0)
        ax.text(gutter_x, 0.55, f"|d|={sd:.2f}", ha="left", fontsize=8.5,
                color=ROW_SR, fontweight="bold")
        if not (np.isnan(sr_A) or np.isnan(sr_B)):
            ax.text(gutter_x, 0.30,
                    f"raw: {sr_A:.2f} / {sr_B:.2f}",
                    ha="left", fontsize=7.3, color=ROW_SR, alpha=0.9)
            ax.text(gutter_x, 0.10, "(Likert 1–7)",
                    ha="left", fontsize=6.8, color=ROW_SR, alpha=0.7,
                    fontstyle="italic")

        # Between-session row annotation (only if rendered)
        if between_plot is not None:
            mean_b = float(between_plot["beh_value"].mean()) if "beh_value" in between_plot.columns else np.nan
            ax.text(gutter_x, 2.55, "no policy primed",
                    ha="left", fontsize=8.0, color=GREY, fontweight="bold")
            if not np.isnan(mean_b):
                ax.text(gutter_x, 2.30,
                        f"raw: {mean_b:.2f}",
                        ha="left", fontsize=7.3, color=GREY, alpha=0.9)
            if beh_unit:
                ax.text(gutter_x, 2.10, f"({beh_unit})",
                        ha="left", fontsize=6.8, color=GREY, alpha=0.7,
                        fontstyle="italic")

        # SR/Beh ratio annotation goes at the very top
        if not np.isnan(ratio):
            star = ""
            if not np.isnan(p):
                if p < 0.001: star = "***"
                elif p < 0.01: star = "**"
                elif p < 0.05: star = "*"
            ratio_y = 2.97 if between_plot is not None else 1.97
            ax.text(gutter_x, ratio_y,
                    f"SR/Beh = {ratio:.2f}{star}",
                    ha="left", fontsize=8.5, fontstyle="italic",
                    color="black", fontweight="bold")

    ax.set_xlim(-3.7, 5.6)
    ax.set_ylim(-0.1, ax_ymax)
    ax.set_yticks([])
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xlabel("z-score (within model)", fontsize=9)
    ax.set_title(info["label"], fontsize=12, color=TASK_COLORS[task], pad=6)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def _zscore_within_named(df: pd.DataFrame, value_col: str) -> pd.Series:
    """Same as _zscore_within but operates on an existing column (no rename)."""
    return df.groupby("model_key")[value_col].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 1e-9 else 0.0
    )


def plot_ridgelines(
    data: Dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    out_path: Path,
    sr_construct: str = "attitude",
    induction: str = "grid",
    layout: str = "2x2",
    between_data: Optional[Dict[str, pd.DataFrame]] = None,
    show_mean_lines: bool = False,
):
    """
    Four-panel ridgeline figure (one per task) plus a shared legend.

    Parameters
    ----------
    layout : '2x2' (default), '1x4', or '4x1'
        Panel arrangement. '1x4' is a single row (wide); '4x1' a single
        column (tall); '2x2' a square grid.
    between_data : optional dict {task: long_df}
        If provided, each panel gains a third row at the top showing
        between-session behaviour as a single grey distribution. Useful for
        seeing whether between-session behaviour aligns with the within-
        session Policy A or Policy B distribution.
    """
    has_between = between_data is not None
    # Three-row panels need more vertical room; bump figsize for those layouts.
    if layout == "2x2":
        nrows, ncols = 2, 2
        figsize = (12, 9.0 if has_between else 7.2)
        hspace, wspace = 0.55, 0.32
        margins = dict(left=0.10, right=0.97, top=0.90, bottom=0.13)
        legend_y, notes_y = 0.04, 0.01
    elif layout == "1x4":
        nrows, ncols = 1, 4
        figsize = (22, 5.8 if has_between else 4.6)
        hspace, wspace = 0.40, 0.30
        margins = dict(left=0.05, right=0.99, top=0.82, bottom=0.22)
        legend_y, notes_y = 0.06, 0.015
    elif layout == "4x1":
        nrows, ncols = 4, 1
        figsize = (8.5, 18.0 if has_between else 14.5)
        hspace, wspace = 0.55, 0.30
        margins = dict(left=0.14, right=0.97, top=0.94, bottom=0.07)
        legend_y, notes_y = 0.035, 0.012
    else:
        raise ValueError(f"Unknown layout {layout!r}; choose 2x2 / 1x4 / 4x1")

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig,
                           hspace=hspace, wspace=wspace, **margins)

    for i, task in enumerate(TASK_ORDER):
        if layout == "2x2":
            ax = fig.add_subplot(gs[i // 2, i % 2])
        elif layout == "1x4":
            ax = fig.add_subplot(gs[0, i])
        else:  # 4x1
            ax = fig.add_subplot(gs[i, 0])
        srow = summary[summary["task"] == task]
        srow = srow.iloc[0] if len(srow) else None
        between_df_for_task = between_data.get(task) if has_between else None
        ridgeline_panel(ax, data[task], task, sr_construct, srow,
                        between_df=between_df_for_task,
                        show_mean_lines=show_mean_lines)

    # shared title and legend
    if has_between:
        title = (
            f"Policy shift in SR vs Behaviour  ·  induction = {induction}\n"
            "Bottom: SR (within). Middle: Behaviour (within, primed). "
            "Top: Behaviour (between, no policy in context). "
            "Rightward = more aligned with Policy A."
        )
    else:
        title = (
            f"Within-session policy shift in SR vs Behaviour  ·  induction = {induction}\n"
            "Both rows oriented so rightward = more aligned with Policy A; "
            "tighter overlap of red/blue = less policy-driven shift"
        )
    fig.suptitle(title, fontsize=13, y=0.99)

    leg_handles = [
        plt.Rectangle((0,0),1,1, color=POLICY_COLORS[0], alpha=0.5, label="Policy A"),
        plt.Rectangle((0,0),1,1, color=POLICY_COLORS[1], alpha=0.5, label="Policy B"),
    ]
    if has_between:
        leg_handles.append(
            plt.Rectangle((0,0),1,1, color="#6b7280", alpha=0.5,
                          label="Between-session (no policy in context)")
        )
    fig.legend(handles=leg_handles, loc="lower center",
               ncol=3 if has_between else 2,
               frameon=False,
               bbox_to_anchor=(0.5, legend_y), fontsize=10.5)

    # task-policy mapping note
    notes = "  ·  ".join(
        f"{TASKS[t]['label']}: A={TASKS[t]['policies'][0]}, B={TASKS[t]['policies'][1]}"
        for t in TASK_ORDER
    )
    fig.text(0.5, notes_y, notes, ha="center", fontsize=7.5, color="#444")

    fig.savefig(out_path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plotting: scatter taxonomy
# ---------------------------------------------------------------------------

def plot_scatter_taxonomy(
    per_cell: pd.DataFrame,
    summary: pd.DataFrame,
    out_path: Path,
    sr_construct: str = "attitude",
    induction: str = "grid",
):
    sr_col = f"d_sr_{sr_construct}"
    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    # per-model points
    for task in TASK_ORDER:
        sub = per_cell[per_cell["task"] == task].copy()
        x = sub["d_beh"].abs().values
        y = sub[sr_col].abs().values
        ax.scatter(x, y, color=TASK_COLORS[task], alpha=0.35, s=40,
                   edgecolor="white", linewidth=0.5)

    # task means with error bars (SE across models)
    for _, row in summary.iterrows():
        if "mean_abs_d_beh" not in row:
            continue
        sub = per_cell[per_cell["task"] == row["task"]]
        x_mean = sub["d_beh"].abs().mean()
        y_mean = sub[sr_col].abs().mean()
        x_se = sub["d_beh"].abs().std() / np.sqrt(sub["d_beh"].abs().count())
        y_se = sub[sr_col].abs().std() / np.sqrt(sub[sr_col].abs().count())
        ax.errorbar(x_mean, y_mean, xerr=x_se, yerr=y_se,
                    fmt="o", markersize=14, color=TASK_COLORS[row["task"]],
                    label=TASKS[row["task"]]["label"], capsize=4,
                    mec="black", mew=1.2, ecolor=TASK_COLORS[row["task"]])

    # diagonal
    xs = np.array([0.05, 30])
    ax.plot(xs, xs, "--", color="gray", alpha=0.45, lw=1)
    ax.text(15, 16.5, "y = x", color="gray", fontsize=10, alpha=0.7, rotation=45)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.05, 30)
    ax.set_ylim(0.05, 30)
    ax.set_xlabel("|Cohen d|  Behaviour shift  (Policy A vs B)", fontsize=11)
    ax.set_ylabel(f"|Cohen d|  Self-Report shift\n(Policy A vs B, {sr_construct.title()})",
                  fontsize=11)
    ax.set_title(
        f"Instruction-sensitivity taxonomy  ·  induction = {induction}\n"
        "Above diagonal: SR moves more than Behaviour (decoupling regime)\n"
        "Below diagonal: Behaviour moves more than SR (priming regime)",
        fontsize=11
    )
    ax.legend(loc="lower right", fontsize=10, framealpha=0.92)
    ax.grid(alpha=0.25, which="both")

    # quadrant labels
    ax.text(0.08, 18, "SR > Beh\n(decoupling /\ncommon-cause)",
            fontsize=9.5, color="#333",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0efe8", alpha=0.85))
    ax.text(8, 0.08, "Beh > SR\n(priming /\ncontext-driven)",
            fontsize=9.5, color="#333",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0efe8", alpha=0.85))

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Big Five ridgelines
#
# Big5 SR is task-agnostic (no policy referent), and the behavioural runs that
# pair with Big5 are also unconditioned by any task policy. So each row of the
# Big5 ridgeline figure is a single pooled distribution per source, except the
# SR row which overlays two theoretically-motivated traits per task (matching
# the paper's Table 1 mappings).
#
# Layout per panel (top to bottom, mirroring the TPB version):
#   Top    — Behaviour, between-session (single grey distribution)
#   Middle — Behaviour, within-session  (single dark distribution)
#   Bottom — SR (two trait KDEs overlaid; theory-flipped so rightward
#            equals "trait pushes behaviour rightward according to theory")
#
# Behaviour rows are NOT flipped — they follow each task's natural direction
# (higher mean_k = more risk; higher sycophancy_rate = more deferring; etc.).
# Trait SR z-scores are flipped per (task, trait) when the trait's theoretical
# sign on behaviour is negative, so high-trait-aligned-direction is rightward
# on both rows. This way, visual alignment of the SR peak with the behaviour
# peak indicates trait-behaviour coherence.
# ---------------------------------------------------------------------------

BIG5_TRAIT_COLS = {
    "Extraversion":      "extraversion_mean",
    "Agreeableness":     "agreeableness_mean",
    "Conscientiousness": "conscientiousness_mean",
    "Neuroticism":       "neuroticism_mean",
    "Openness":          "openness_mean",
}

# Default theoretical Big Five trait pair per task with expected sign on
# behaviour (sign meaning higher-trait → higher behaviour in raw direction).
# Matches Table 1 of the paper.
BIG5_DEFAULT_PAIRS = {
    "cct":        [("Neuroticism", "-"),       ("Openness", "+")],
    "sycophancy": [("Agreeableness", "+"),     ("Neuroticism", "+")],
    "honesty":    [("Conscientiousness", "+"), ("Openness", "+")],
    "iat":        [("Agreeableness", "-"),     ("Openness", "-")],
}

# Within-session and between-session behaviour columns for each task.
BIG5_BEH_COLS = {
    "cct":        dict(within="beh__mean_k",                    between="mean_k"),
    "sycophancy": dict(within="beh__sycophancy_rate",           between="sycophancy_rate"),
    "honesty":    dict(within="beh__mean_abs_confidence_delta", between="mean_inconsistency_abs"),
    "iat":        dict(within="beh__mean_bias_score",           between="bias"),
}


def _load_big5_within(within_root: Path, task: str) -> pd.DataFrame:
    """Load Big5 within-session combined_runs.csv for a task."""
    path = within_root / "big5_psycohere_grid" / task / "big5" / "combined_runs.csv"
    if not path.exists():
        # tolerate persona naming
        path = within_root / "big5_psycohere_personas" / task / "big5" / "combined_runs.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find Big5 within combined_runs for {task} under {within_root}"
        )
    df = pd.read_csv(path, low_memory=False)
    if "sr_status" in df.columns:
        df = df[df["sr_status"] == "ok"]
    if "beh_status" in df.columns:
        df = df[df["beh_status"] == "ok"]
    beh_col = BIG5_BEH_COLS[task]["within"]
    for c in list(BIG5_TRAIT_COLS.values()) + [beh_col]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = ["model_key", "temperature", "seed", "persona_label",
            *BIG5_TRAIT_COLS.values(), beh_col]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    return df.rename(columns={beh_col: "beh_value"})


def _load_big5_between(between_root: Path, task: str) -> pd.DataFrame:
    """Load Big5 between-session merged CSV (already per-condition aggregated)."""
    path = between_root / f"big5_x_{task}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find Big5 between merged CSV at {path}"
        )
    df = pd.read_csv(path, low_memory=False)
    beh_col = BIG5_BEH_COLS[task]["between"]
    for c in list(BIG5_TRAIT_COLS.values()) + [beh_col]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = ["model_key", "temperature", "seed", "persona_label",
            *BIG5_TRAIT_COLS.values(), beh_col]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    # Aggregate to per-condition if the source has multiple rows per condition
    # (IAT has ~6 rows per condition: one per stereotype domain).
    cond_keys = ["model_key", "temperature", "seed", "persona_label"]
    cond_keys = [c for c in cond_keys if c in df.columns]
    rows_per_cond = len(df) / max(df.groupby(cond_keys).ngroups, 1)
    if rows_per_cond > 1.5:
        # average behaviour, take first SR (constant per condition by design)
        sr_first = df.groupby(cond_keys)[list(BIG5_TRAIT_COLS.values())].first().reset_index()
        beh_mean = df.groupby(cond_keys)[beh_col].mean().reset_index()
        df = sr_first.merge(beh_mean, on=cond_keys, how="inner")
    return df.rename(columns={beh_col: "beh_value"})


def _parse_big5_trait_arg(arg: Optional[str]) -> Dict[str, list]:
    """
    Parse --big5_traits like "cct=Neuroticism:-,Openness:+ sycophancy=..."
    Tokens are space-separated; per-task spec is "task=Trait1:Sign,Trait2:Sign".
    Sign is '+' or '-' indicating expected behavioural direction. If omitted,
    defaults from BIG5_DEFAULT_PAIRS are used.
    """
    if not arg:
        return dict(BIG5_DEFAULT_PAIRS)
    out = dict(BIG5_DEFAULT_PAIRS)
    for tok in arg.split():
        if "=" not in tok:
            continue
        task, traits_str = tok.split("=", 1)
        task = task.strip().lower()
        if task not in BIG5_DEFAULT_PAIRS:
            continue
        pairs = []
        for ts in traits_str.split(","):
            ts = ts.strip()
            if ":" in ts:
                name, sign = ts.split(":", 1)
                name = name.strip().title()
                sign = sign.strip()
            else:
                name = ts.strip().title()
                sign = "+"
            if name not in BIG5_TRAIT_COLS:
                # try fuzzy match
                for canonical in BIG5_TRAIT_COLS:
                    if canonical.lower().startswith(name.lower()[:4]):
                        name = canonical
                        break
            if name in BIG5_TRAIT_COLS:
                pairs.append((name, sign))
        if pairs:
            out[task] = pairs
    return out


def big5_ridgeline_panel(
    ax: plt.Axes,
    within_df: pd.DataFrame,
    between_df: Optional[pd.DataFrame],
    task: str,
    traits: list,
    show_mean_lines: bool = False,
):
    """
    Single-task three-row Big5 ridgeline.
    """
    info = TASKS[task]

    # Joint min-max normalise behaviour per model across within ∪ between, so
    # within and between rows live on a comparable axis even when raw scales
    # differ (e.g. honesty's 0-2.3 vs 0-1).
    if between_df is not None and len(between_df) > 0:
        cond_keys = ["model_key", "temperature", "seed", "persona_label"]
        within_indexed = within_df.set_index(cond_keys)["beh_value"]
        between_indexed = between_df.set_index(cond_keys)["beh_value"]
        w_norm, b_norm = _minmax_normalise_per_model(within_indexed, between_indexed)
        within_plot = within_df.copy()
        within_plot["beh_value_norm"] = within_plot.set_index(cond_keys).index.map(w_norm.to_dict())
        between_plot = between_df.copy()
        between_plot["beh_value_norm"] = between_plot.set_index(cond_keys).index.map(b_norm.to_dict())
        # Joint per-model z-score across within ∪ between (see TPB panel for rationale).
        w_z, b_z = _zscore_joint_per_model(
            (within_plot, "beh_value_norm"),
            (between_plot, "beh_value_norm"),
        )
        within_plot["beh_z"] = w_z
        between_plot["beh_z"] = b_z
    else:
        within_plot = within_df.copy()
        within_plot["beh_z"] = _zscore_within(within_plot, "beh_value")
        between_plot = None

    # SR z-scores per trait (z-score per model on each trait column).
    for trait_name, _sign in traits:
        col = BIG5_TRAIT_COLS[trait_name]
        within_plot[f"sr_z_{trait_name}"] = _zscore_within(within_plot, col)

    # Apply the SAME behaviour-axis convention as the TPB panel: when the
    # task's raw behaviour aligns with Policy B, flip the behaviour z-axis
    # so that "rightward = Policy A direction" on every panel. This keeps
    # task-level visual orientation consistent across the TPB and Big5
    # figures, allowing side-by-side reading of, e.g., the IAT panel in
    # one figure against the IAT panel in the other.
    beh_flipped = info.get("beh_higher_aligns_with", "A") == "B"
    if beh_flipped:
        within_plot["beh_z"] = -within_plot["beh_z"]
        if between_plot is not None:
            between_plot["beh_z"] = -between_plot["beh_z"]

    # Sign-flip each trait's z-score so rightward = "Policy A direction".
    # For an A-aligned task (beh axis NOT flipped, e.g. Honesty), rightward
    # = high raw behaviour, so a trait with sign '+' (high-trait predicts
    # high-raw) keeps its orientation (no flip), while sign '-' flips.
    # For a B-aligned task (beh axis flipped, e.g. CCT/Syc/IAT), rightward
    # = low raw behaviour, so sign '+' must flip and sign '-' keeps its
    # orientation. Combined rule: flip iff (sign == '-') XOR beh_flipped.
    for trait_name, sign in traits:
        flip_trait = (sign == "-") ^ beh_flipped
        if flip_trait:
            within_plot[f"sr_z_{trait_name}"] = -within_plot[f"sr_z_{trait_name}"]

    rows = [
        ("Behaviour\n(between-session)", "between"),
        ("Behaviour\n(within-session)",  "within"),
        ("SR (Big Five)",                 "sr"),
    ]
    if between_plot is None:
        # collapse to 2 rows when no between data
        rows = [r for r in rows if r[1] != "between"]

    grid = np.linspace(-3.5, 3.5, 400)
    BANDWIDTH = 0.35
    GREY = "#6b7280"
    DARK_BEH = "#0f172a"

    def _kde(values):
        v = pd.to_numeric(pd.Series(values), errors="coerce").dropna().values
        if len(v) < 3:
            return np.zeros_like(grid)
        try:
            kde = gaussian_kde(v, bw_method=BANDWIDTH)
            return kde(grid)
        except np.linalg.LinAlgError:
            return np.zeros_like(grid)

    def _draw_mean_line(values, y0, color):
        v = pd.to_numeric(pd.Series(values), errors="coerce").dropna().values
        if len(v) < 2:
            return
        m = float(np.mean(v))
        if m < grid.min() or m > grid.max():
            return
        density = _kde(v)
        if density.max() > 0:
            density = density / density.max() * 0.85
        h = float(np.interp(m, grid, density))
        ax.plot([m, m], [y0, y0 + h], linestyle=":", color=color,
                lw=1.4, alpha=0.95)

    # vertical positions: top is between-row, then within, then SR at bottom
    n_rows = len(rows)
    y0_for_idx = {0: float(n_rows - 1) * 1.0,
                  1: float(n_rows - 2) * 1.0,
                  2: 0.0}
    # invert so first row in `rows` is at the TOP of the panel
    rows_with_y = [(label, src, n_rows - 1 - i) for i, (label, src) in enumerate(rows)]

    for label, src, y0 in rows_with_y:
        if src == "between":
            v = between_plot["beh_z"].dropna().values
            density = _kde(v)
            if density.max() > 0:
                density = density / density.max() * 0.85
            ax.fill_between(grid, y0, y0 + density, color=GREY, alpha=0.45, lw=0)
            ax.plot(grid, y0 + density, color=GREY, lw=1.4, alpha=0.9)
            if show_mean_lines:
                _draw_mean_line(v, y0, GREY)
            label_color = GREY
        elif src == "within":
            v = within_plot["beh_z"].dropna().values
            density = _kde(v)
            if density.max() > 0:
                density = density / density.max() * 0.85
            ax.fill_between(grid, y0, y0 + density, color=DARK_BEH, alpha=0.35, lw=0)
            ax.plot(grid, y0 + density, color=DARK_BEH, lw=1.3, alpha=0.9)
            if show_mean_lines:
                _draw_mean_line(v, y0, DARK_BEH)
            label_color = DARK_BEH
        else:  # sr — overlay two trait distributions in policy colours
            for (trait_name, _sign), color in zip(traits, POLICY_COLORS):
                v = within_plot[f"sr_z_{trait_name}"].dropna().values
                density = _kde(v)
                if density.max() > 0:
                    density = density / density.max() * 0.85
                ax.fill_between(grid, y0, y0 + density, color=color, alpha=0.35, lw=0)
                ax.plot(grid, y0 + density, color=color, lw=1.3, alpha=0.9)
                if show_mean_lines:
                    _draw_mean_line(v, y0, color)
            label_color = ROW_SR

        ax.axhline(y0, color="black", lw=0.6, alpha=0.5, xmax=0.65)
        display_label = label
        if "Behaviour" in label and beh_flipped:
            display_label = f"{label}\n(axis inverted)"
        ax.text(-3.85, y0 + 0.45, display_label, ha="right", va="center",
                fontsize=9.5, color=label_color)

    # Right-gutter raw means
    gutter_x = 3.85
    if between_plot is not None:
        bm = float(between_plot["beh_value"].mean())
        y_top = float(n_rows - 1)
        ax.text(gutter_x, y_top + 0.55, "no SR primed", ha="left", fontsize=8.0,
                color=GREY, fontweight="bold")
        ax.text(gutter_x, y_top + 0.30, f"raw: {bm:.2f}",
                ha="left", fontsize=7.3, color=GREY, alpha=0.9)
        if info.get("beh_label"):
            ax.text(gutter_x, y_top + 0.10, f"({info['beh_label']})",
                    ha="left", fontsize=6.8, color=GREY, alpha=0.7,
                    fontstyle="italic")

    wm = float(within_plot["beh_value"].mean())
    y_within = float(n_rows - 2) if between_plot is not None else float(n_rows - 1)
    ax.text(gutter_x, y_within + 0.55, "SR primed", ha="left", fontsize=8.0,
            color=DARK_BEH, fontweight="bold")
    ax.text(gutter_x, y_within + 0.30, f"raw: {wm:.2f}",
            ha="left", fontsize=7.3, color=DARK_BEH, alpha=0.9)
    if info.get("beh_label"):
        ax.text(gutter_x, y_within + 0.10, f"({info['beh_label']})",
                ha="left", fontsize=6.8, color=DARK_BEH, alpha=0.7,
                fontstyle="italic")

    # SR row gutter: list trait names with their sign
    sr_lines = []
    for trait_name, sign in traits:
        col = BIG5_TRAIT_COLS[trait_name]
        m = float(within_plot[col].mean())
        sr_lines.append(f"{trait_name[:5]} ({sign}): {m:.2f}")
    ax.text(gutter_x, 0.55, sr_lines[0],
            ha="left", fontsize=7.6, color=POLICY_COLORS[0], fontweight="bold")
    if len(sr_lines) > 1:
        ax.text(gutter_x, 0.30, sr_lines[1],
                ha="left", fontsize=7.6, color=POLICY_COLORS[1], fontweight="bold")
    ax.text(gutter_x, 0.10, "(Likert 1–5, mean)",
            ha="left", fontsize=6.8, color=ROW_SR, alpha=0.7, fontstyle="italic")

    # Axes setup
    ax.set_xlim(-3.7, 5.6)
    ax.set_ylim(-0.1, float(n_rows) + 0.25)
    ax.set_yticks([])
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xlabel("z-score (within model)", fontsize=9)
    ax.set_title(info["label"], fontsize=12, color=TASK_COLORS[task], pad=6)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def plot_big5_ridgelines(
    within_data: Dict[str, pd.DataFrame],
    between_data: Optional[Dict[str, pd.DataFrame]],
    traits_per_task: Dict[str, list],
    out_path: Path,
    induction: str = "grid",
    layout: str = "2x2",
    show_mean_lines: bool = False,
):
    """
    Build the four-panel Big Five ridgeline figure.
    Same layout flags as plot_ridgelines (TPB version).
    """
    has_between = between_data is not None
    if layout == "2x2":
        nrows, ncols = 2, 2
        figsize = (12, 9.0 if has_between else 7.2)
        hspace, wspace = 0.55, 0.32
        margins = dict(left=0.10, right=0.97, top=0.90, bottom=0.13)
        legend_y, notes_y = 0.04, 0.01
    elif layout == "1x4":
        nrows, ncols = 1, 4
        figsize = (22, 5.8 if has_between else 4.6)
        hspace, wspace = 0.40, 0.30
        margins = dict(left=0.05, right=0.99, top=0.82, bottom=0.22)
        legend_y, notes_y = 0.06, 0.015
    elif layout == "4x1":
        nrows, ncols = 4, 1
        figsize = (8.5, 18.0 if has_between else 14.5)
        hspace, wspace = 0.55, 0.30
        margins = dict(left=0.14, right=0.97, top=0.94, bottom=0.07)
        legend_y, notes_y = 0.035, 0.012
    else:
        raise ValueError(f"Unknown layout {layout!r}")

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig,
                           hspace=hspace, wspace=wspace, **margins)

    for i, task in enumerate(TASK_ORDER):
        if layout == "2x2":
            ax = fig.add_subplot(gs[i // 2, i % 2])
        elif layout == "1x4":
            ax = fig.add_subplot(gs[0, i])
        else:
            ax = fig.add_subplot(gs[i, 0])
        between_for_task = between_data.get(task) if has_between else None
        big5_ridgeline_panel(ax, within_data[task], between_for_task,
                             task, traits_per_task[task],
                             show_mean_lines=show_mean_lines)

    title = (
        f"Big Five SR vs Behaviour  ·  induction = {induction}\n"
        "Top: Behaviour (between-session, no SR primed). "
        "Middle: Behaviour (within-session, SR primed). "
        "Bottom: SR (two theoretical traits). "
        "Rightward = aligned with Policy A direction (matching TPB figure)."
    )
    fig.suptitle(title, fontsize=12.5, y=0.99)

    # Legend: show one swatch per trait used in any panel + the two beh sources
    seen = []
    leg_handles = []
    for task in TASK_ORDER:
        for (trait_name, sign), col in zip(traits_per_task[task], POLICY_COLORS):
            key = (trait_name, sign)
            if key in seen:
                continue
            seen.append(key)
    # Generic policy-coloured legend that names the two slot positions
    leg_handles = [
        plt.Rectangle((0,0),1,1, color=POLICY_COLORS[0], alpha=0.5,
                      label="Trait 1 (per-task)"),
        plt.Rectangle((0,0),1,1, color=POLICY_COLORS[1], alpha=0.5,
                      label="Trait 2 (per-task)"),
        plt.Rectangle((0,0),1,1, color="#0f172a", alpha=0.5,
                      label="Behaviour (within-session)"),
    ]
    if has_between:
        leg_handles.append(
            plt.Rectangle((0,0),1,1, color="#6b7280", alpha=0.5,
                          label="Behaviour (between-session)")
        )
    fig.legend(handles=leg_handles, loc="lower center",
               ncol=4 if has_between else 3,
               frameon=False,
               bbox_to_anchor=(0.5, legend_y), fontsize=10)

    notes = "  ·  ".join(
        f"{TASKS[t]['label']}: T1={traits_per_task[t][0][0]}({traits_per_task[t][0][1]}), "
        f"T2={traits_per_task[t][1][0]}({traits_per_task[t][1][1]})"
        for t in TASK_ORDER
    )
    fig.text(0.5, notes_y, notes, ha="center", fontsize=7.3, color="#444")

    fig.savefig(out_path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cct_master", type=str, default=None,
                        help="Master CSV for CCT (mode A)")
    parser.add_argument("--syc_master", type=str, default=None)
    parser.add_argument("--honesty_master", type=str, default=None)
    parser.add_argument("--iat_master", type=str, default=None)
    parser.add_argument("--within_root", type=str, default=None,
                        help="Path to within/grid (or within/personas) folder (mode B)")
    parser.add_argument("--between_root", type=str, default=None,
                        help="Optional. Path to merged/between/grid (or "
                             "merged/between/personas) folder. When provided, "
                             "ridgeline panels gain a 3rd row at the top showing "
                             "between-session behaviour as a single grey "
                             "distribution (no policy primed).")
    parser.add_argument("--induction_label", type=str, default=None,
                        choices=[None, "grid", "personas"],
                        help="Override induction label used in titles/filenames")
    parser.add_argument("--sr_construct", type=str, default="attitude",
                        choices=["attitude", "subjective_norm", "pbc", "intention"],
                        help="Which TPB construct to use for the headline ratio")
    parser.add_argument("--layout", type=str, default="2x2",
                        choices=["2x2", "1x4", "4x1"],
                        help="Ridgeline panel arrangement: 2x2 (default), "
                             "1x4 (single wide row), or 4x1 (single tall column)")
    parser.add_argument("--big5_traits", type=str, default=None,
                        help="Optional Big5 trait override, format: "
                             "'cct=Neuroticism:-,Openness:+ "
                             "sycophancy=Agreeableness:+,Neuroticism:+ "
                             "honesty=Conscientiousness:+,Openness:+ "
                             "iat=Agreeableness:-,Openness:-'. Sign indicates "
                             "the trait's expected direction on behaviour. If "
                             "omitted, defaults from Table 1 are used. Big5 "
                             "ridgelines are auto-generated when both within "
                             "and between Big5 data are detected.")
    parser.add_argument("--mean_lines", action="store_true",
                        help="Draw dotted vertical mean lines on every KDE in "
                             "the ridgeline panels (TPB and Big5).")
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading TPB data …")
    data, induction, between_data = load_all(args)
    for t, df in data.items():
        n_pol = df.groupby("policy_id").size().to_dict()
        print(f"      {t:12s}  within rows={len(df):5d}  per_policy={n_pol}")
    if between_data is not None:
        for t, df in between_data.items():
            print(f"      {t:12s}  between rows={len(df):5d}")

    print("[2/5] Computing per-(model × task) Cohen's d …")
    per_cell = per_cell_shifts(data)
    per_cell.to_csv(out_dir / "per_cell_shifts.csv", index=False)

    print("[3/5] Aggregating to task summary (Wilcoxon paired tests) …")
    summary = task_summary(per_cell, sr_construct=args.sr_construct)
    summary.to_csv(out_dir / "task_summary.csv", index=False)

    print("\nTask-level summary:")
    print("-" * 78)
    print(summary.round(3).to_string(index=False))
    print("-" * 78)

    print("[4/5] Plotting TPB figures …")
    plot_ridgelines(
        data, summary,
        out_path=out_dir / f"ridgeline_distributions_{induction}",
        sr_construct=args.sr_construct,
        induction=induction,
        layout=args.layout,
        between_data=between_data,
        show_mean_lines=args.mean_lines,
    )
    plot_scatter_taxonomy(
        per_cell, summary,
        out_path=out_dir / f"scatter_taxonomy_{induction}",
        sr_construct=args.sr_construct,
        induction=induction,
    )

    # ---------------------------------------------------------------
    # Big5 ridgelines: auto-trigger if Big5 data is available.
    # ---------------------------------------------------------------
    big5_within_data = None
    big5_between_data = None
    if args.within_root:
        try:
            big5_within_data = {
                t: _load_big5_within(Path(args.within_root), t) for t in TASKS
            }
        except FileNotFoundError as e:
            print(f"[5/5] Big5 within data not found ({e}); skipping Big5 figure.")
    if args.between_root:
        try:
            big5_between_data = {
                t: _load_big5_between(Path(args.between_root), t) for t in TASKS
            }
        except FileNotFoundError as e:
            print(f"      Big5 between data not found ({e}); proceeding without between row.")

    if big5_within_data is not None:
        print("[5/5] Plotting Big5 ridgeline figure …")
        traits_per_task = _parse_big5_trait_arg(args.big5_traits)
        for t, df in big5_within_data.items():
            print(f"      {t:12s}  big5 within rows={len(df):5d}")
        if big5_between_data is not None:
            for t, df in big5_between_data.items():
                print(f"      {t:12s}  big5 between rows={len(df):5d}")
        plot_big5_ridgelines(
            big5_within_data, big5_between_data, traits_per_task,
            out_path=out_dir / f"big5_ridgeline_distributions_{induction}",
            induction=induction,
            layout=args.layout,
            show_mean_lines=args.mean_lines,
        )

    print(f"\nDone. Outputs written to {out_dir}/")
    print("  per_cell_shifts.csv")
    print("  task_summary.csv")
    print(f"  ridgeline_distributions_{induction}.{{png,pdf}}")
    print(f"  scatter_taxonomy_{induction}.{{png,pdf}}")
    if big5_within_data is not None:
        print(f"  big5_ridgeline_distributions_{induction}.{{png,pdf}}")


if __name__ == "__main__":
    main()
