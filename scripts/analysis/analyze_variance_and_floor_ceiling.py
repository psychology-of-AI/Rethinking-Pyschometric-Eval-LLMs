#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_variance_and_floor_ceiling.py

Computes descriptive-statistics tables that defend SR-behaviour correlation
analyses against reviewer concerns about variance and floor/ceiling effects,
modelled after the Personality Illusion paper (Han et al. 2025) Appendix H.4.

Per task / construct (and optionally per policy), reports:
  - Scale range (theoretical)
  - Mean (SD) across (model x condition) cells
  - % of theoretical range used
  - % at floor (within epsilon of theoretical min)
  - % at ceiling (within epsilon of theoretical max)
  - ICC: between-model agreement on rank ordering across conditions
  - Kruskal-Wallis eta-squared: between-model variance fraction
  - KW p-value and significance stars

Three tables produced per slice:
  Table 1 (behaviour):  per task, pooled across all policies; 5 rows in our design
                        (Risk Taking, Sycophancy, Epistemic Honesty,
                         Self-Reflective Honesty, Stereotyping).
                        Policies are experimental treatments designed to span the
                        behavioural range; variance diagnostics evaluate the task as
                        a whole.  Individual policies may collapse to floor/ceiling
                        by design (e.g. independent_judgment drives sycophancy to
                        floor); this is absorbed into the task-level pool rather
                        than treated as a measurement artefact.
  Table 2 (Big 5 SR):   per (trait x task) for within-session;
                        per trait only for between-session
                        (Big 5 SR is task-agnostic but collected per task in within)
  Table 3 (TPB SR):     per (task x construct), aggregated across policies; 16 rows

Slices (use --session and --induction):
  within-grid       : RQ1/RQ2 defense
  within-personas   : RQ4 same-session diagnostics
  between-grid      : RQ3 cross-session test
  between-personas  : RQ4 main rescue test (the BIG ONE for RQ4 defense)

Usage:
  # RQ1/RQ2 defense (paper subsection 1)
  python analyze_variance_and_floor_ceiling.py \
      --root_dir results/psycohere_v1 \
      --session within --induction grid \
      --out_dir results/psycohere_v1/analysis/variance_diagnostics/within_grid

  # RQ4 defense (paper subsection 2)
  python analyze_variance_and_floor_ceiling.py \
      --root_dir results/psycohere_v1 \
      --session between --induction personas \
      --out_dir results/psycohere_v1/analysis/variance_diagnostics/between_personas

Outputs (in --out_dir):
  variance_behavior_per_task_policy.csv
  variance_big5_sr.csv
  variance_tpb_sr_per_task_construct.csv
  *.tex versions for direct LaTeX inclusion
  variance_summary.txt
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BIG5_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
BIG5_PRETTY = {
    "openness":          "Openness",
    "conscientiousness": "Conscientiousness",
    "extraversion":      "Extraversion",
    "agreeableness":     "Agreeableness",
    "neuroticism":       "Neuroticism",
}
BIG5_RANGE = (1, 5)

TPB_CONSTRUCTS = ["attitude", "subjective_norm", "pbc", "intention"]
TPB_PRETTY = {
    "attitude":        "Attitude",
    "subjective_norm": "Subjective Norm",
    "pbc":             "PBC",
    "intention":       "Intention",
}
TPB_RANGE = (1, 7)

TASK_PRETTY = {
    "cct":              "Risk Taking",
    "sycophancy":       "Sycophancy",
    "honesty":          "Honesty",
    "honesty_overconf": "Epistemic Honesty",
    "honesty_consist":  "Self-Reflective Honesty",
    "iat":              "Stereotyping",
}

# Behavioural raw-scale ranges (after task-specific unit conversion)
BEH_RAW_RANGE = {
    "cct":              (0, 32),       # mean cards flipped
    "sycophancy":       (0, 100),      # flip rate %
    "honesty_overconf": (-100, 100),   # mean confidence delta in pp
    "honesty_consist":  (0, 100),      # consistency rate %
    "iat":              (-1, 1),       # d-score
}


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _safe_kruskal(groups):
    valid = [np.asarray(g, dtype=float) for g in groups]
    valid = [g[np.isfinite(g)] for g in valid]
    valid = [g for g in valid if len(g) >= 1 and (np.std(g) > 0 if len(g) > 1 else True)]
    if len(valid) < 2:
        return (np.nan, np.nan)
    try:
        H, p = stats.kruskal(*valid)
        return (H, p)
    except Exception:
        return (np.nan, np.nan)


def _kw_eta_squared(groups):
    valid = [np.asarray(g, dtype=float) for g in groups]
    valid = [g[np.isfinite(g)] for g in valid]
    valid = [g for g in valid if len(g) >= 1]
    if len(valid) < 2:
        return np.nan
    n = sum(len(g) for g in valid)
    k = len(valid)
    if n - k <= 0:
        return np.nan
    H, _ = _safe_kruskal(valid)
    if not np.isfinite(H):
        return np.nan
    return max((H - k + 1) / (n - k), 0.0)


def _icc_2k(values_by_model):
    vals_per_model = []
    for v in values_by_model.values():
        a = np.asarray(v, dtype=float)
        a = a[np.isfinite(a)]
        if len(a) > 0:
            vals_per_model.append(a)
    if len(vals_per_model) < 2:
        return np.nan
    flat = np.concatenate(vals_per_model)
    grand = flat.mean()
    n_total = len(flat)
    k = len(vals_per_model)
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in vals_per_model)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in vals_per_model)
    df_between = k - 1
    df_within = n_total - k
    if df_between <= 0 or df_within <= 0:
        return np.nan
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    if ms_between <= 0:
        return np.nan
    icc = (ms_between - ms_within) / ms_between
    return max(min(icc, 1.0), 0.0)


def descriptive_stats(values, scale_range, eps_frac: float = 0.005):
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    n = len(v)
    if n == 0:
        return {"n_cells": 0, "mean": np.nan, "sd": np.nan,
                "obs_min": np.nan, "obs_max": np.nan,
                "pct_range_used": np.nan, "pct_at_floor": np.nan, "pct_at_ceiling": np.nan}
    lo, hi = scale_range
    rng = hi - lo
    eps = rng * eps_frac
    return {
        "n_cells":        n,
        "mean":           float(np.mean(v)),
        "sd":             float(np.std(v, ddof=1)) if n > 1 else 0.0,
        "obs_min":        float(np.min(v)),
        "obs_max":        float(np.max(v)),
        "pct_range_used": float((np.max(v) - np.min(v)) / rng * 100) if rng > 0 else np.nan,
        "pct_at_floor":   float((v <= (lo + eps)).mean() * 100),
        "pct_at_ceiling": float((v >= (hi - eps)).mean() * 100),
    }


def between_model_stats(values_by_model):
    groups = [np.asarray(v, dtype=float) for v in values_by_model.values()]
    groups = [g[np.isfinite(g)] for g in groups]
    H, p = _safe_kruskal(groups)
    eta2 = _kw_eta_squared(groups)
    icc = _icc_2k(values_by_model)
    return {"icc": icc, "kw_eta2": eta2, "kw_p": p}


def stars(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_within_tpb(root: Path, induction: str) -> pd.DataFrame:
    parts = []
    slice_root = root / "within" / induction
    for task in ["cct", "sycophancy", "honesty", "iat"]:
        task_root = slice_root / f"tpb_{task}_psycohere_{induction}"
        if not task_root.exists():
            continue
        for policy_dir in task_root.iterdir():
            if not policy_dir.is_dir():
                continue
            f = policy_dir / "combined_runs.csv"
            if f.exists():
                df = pd.read_csv(f, on_bad_lines="skip", engine="python")
                df["task"] = task
                df["policy"] = policy_dir.name
                parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_within_big5(root: Path, induction: str) -> pd.DataFrame:
    parts = []
    slice_root = root / "within" / induction
    for task in ["cct", "sycophancy", "honesty", "iat"]:
        f = slice_root / f"big5_psycohere_{induction}" / task / "big5" / "combined_runs.csv"
        if f.exists():
            df = pd.read_csv(f, on_bad_lines="skip", engine="python")
            df["task"] = task
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_between_tpb_sr(root: Path, induction: str) -> pd.DataFrame:
    parts = []
    slice_root = root / "between" / induction / "session_sr"
    for task in ["cct", "sycophancy", "honesty", "iat"]:
        task_root = slice_root / f"tpb_{task}_psycohere_{induction}"
        if not task_root.exists():
            continue
        for policy_dir in task_root.iterdir():
            if not policy_dir.is_dir():
                continue
            f = policy_dir / "tpb_likert_runs.csv"
            if f.exists():
                df = pd.read_csv(f, on_bad_lines="skip", engine="python")
                df["task"] = task
                df["policy"] = policy_dir.name
                parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_between_big5_sr(root: Path, induction: str) -> pd.DataFrame:
    """Load between-session Big Five SR per task, mirroring load_between_tpb_sr.

    Tries two directory patterns for each task:
      1. session_sr / big5_{task}_psycohere_{induction} / <policy_dir> / tpb_likert_runs.csv
         (parallel structure to TPB)
      2. session_sr / big5_psycohere_{induction} / {task} / big5 / tpb_likert_runs.csv
         (within-session-style sub-directory)
    Falls back to the original single-file path if neither per-task pattern is found,
    but in that case logs a warning because the resulting 5-cell table cannot
    distinguish persona-induced trait shifts across tasks.
    """
    parts = []
    slice_root = root / "between" / induction / "session_sr"

    for task in ["cct", "sycophancy", "honesty", "iat"]:
        # Pattern 1: big5_{task}_psycohere_{induction} / <policy_dir> / tpb_likert_runs.csv
        task_root_1 = slice_root / f"big5_{task}_psycohere_{induction}"
        if task_root_1.exists():
            for policy_dir in task_root_1.iterdir():
                if not policy_dir.is_dir():
                    continue
                f = policy_dir / "tpb_likert_runs.csv"
                if f.exists():
                    df = pd.read_csv(f, on_bad_lines="skip", engine="python")
                    df["task"] = task
                    df["policy"] = policy_dir.name
                    parts.append(df)
            continue  # found pattern 1 for this task, skip pattern 2

        # Pattern 2: big5_psycohere_{induction} / {task} / big5 / tpb_likert_runs.csv
        f2 = (slice_root / f"big5_psycohere_{induction}" / task / "big5"
              / "tpb_likert_runs.csv")
        if f2.exists():
            df = pd.read_csv(f2, on_bad_lines="skip", engine="python")
            df["task"] = task
            parts.append(df)

    if parts:
        return pd.concat(parts, ignore_index=True)

    # Fallback: original single-file path (no per-task split)
    f_fallback = (root / "between" / induction / "session_sr"
                  / f"big5_psycohere_{induction}" / "big5" / "tpb_likert_runs.csv")
    if f_fallback.exists():
        print("  WARNING: no per-task Big Five SR found for between-session; "
              "falling back to single-file (5 cells only). "
              "Check directory structure if per-task data is expected.")
        return pd.read_csv(f_fallback, on_bad_lines="skip", engine="python")

    return pd.DataFrame()


def load_between_behavior(root: Path, induction: str) -> pd.DataFrame:
    """Aggregate trial-level CSVs to per-(model, condition) outcomes."""
    parts = []
    slice_root = root / "between" / induction / "session_beh"
    if not slice_root.exists():
        return pd.DataFrame()

    task_specs = {
        "cct": {
            "dirs": [f"cct-psycohere-{induction}", f"cct_psycohere_{induction}"],
            "fname": "cct_runs.csv",
            "trial_col": None,
            "out_col": "beh__mean_k",
        },
        "sycophancy": {
            "dirs": [f"sycophancy-psycohere-{induction}", f"sycophancy_psycohere_{induction}"],
            "fname": "sycophancy_runs.csv",
            "trial_col": "sycophancy",
            "out_col": "beh__sycophancy_rate",
        },
        "honesty": {
            "dirs": [f"honesty-psycohere-{induction}", f"honesty_psycohere_{induction}"],
            "fname": "honesty_runs.csv",
            "trial_col": "confidence_delta",
            "out_col": None,
        },
        "iat": {
            "dirs": [f"iat-psycohere-{induction}", f"iat_psycohere_{induction}"],
            "fname": "iat_runs.csv",
            "trial_col": "bias",
            "out_col": "beh__mean_bias_score",
        },
    }

    for task, spec in task_specs.items():
        for dn in spec["dirs"]:
            task_root = slice_root / dn
            if not task_root.exists():
                continue
            for policy_dir in task_root.iterdir():
                if not policy_dir.is_dir():
                    continue
                f = policy_dir / spec["fname"]
                if not f.exists():
                    continue
                df = pd.read_csv(f, on_bad_lines="skip", engine="python")
                if "model_key" not in df.columns:
                    continue
                # Group at condition level (one row per model x condition), NOT trial level.
                # seed/temperature/top_p are run-level noise; prompt_variant and
                # question_id are trial-level identifiers — including them produces
                # one row per item rather than per condition, inflating n_cells
                # (observed as 20 782 for honesty vs 330 for CCT).
                gk = [c for c in ["model_key", "model_id",
                                  "persona_label", "condition_id", "system_prompt"]
                      if c in df.columns]
                if not gk or "model_key" not in gk:
                    gk = ["model_key"]

                if task == "honesty" and spec["trial_col"] in df.columns:
                    df[spec["trial_col"]] = pd.to_numeric(df[spec["trial_col"]], errors="coerce")
                    agg_signed = (df.groupby(gk, dropna=False)[spec["trial_col"]]
                                    .mean().reset_index()
                                    .rename(columns={spec["trial_col"]: "beh__mean_confidence_delta"}))
                    df["_abs"] = df[spec["trial_col"]].abs()
                    agg_abs = (df.groupby(gk, dropna=False)["_abs"]
                                 .mean().reset_index()
                                 .rename(columns={"_abs": "beh__mean_abs_confidence_delta"}))
                    merged = pd.merge(agg_signed, agg_abs, on=gk, how="outer")
                    merged["task"] = "honesty"
                    merged["policy"] = policy_dir.name
                    parts.append(merged)
                    continue

                if spec["trial_col"] and spec["trial_col"] in df.columns:
                    df[spec["trial_col"]] = pd.to_numeric(df[spec["trial_col"]], errors="coerce")
                    agg = (df.groupby(gk, dropna=False)[spec["trial_col"]]
                             .mean().reset_index()
                             .rename(columns={spec["trial_col"]: spec["out_col"]}))
                else:
                    if "mean_k" in df.columns:
                        agg = (df.drop_duplicates(subset=gk)[gk + ["mean_k"]]
                                 .rename(columns={"mean_k": spec["out_col"]}))
                    else:
                        continue

                agg["task"] = task
                agg["policy"] = policy_dir.name
                parts.append(agg)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------
# Behaviour table builder
# ---------------------------------------------------------------------------

def _values_by_model(sub: pd.DataFrame, col: str, scale_factor: float = 1.0,
                      transform=None):
    raw = pd.to_numeric(sub[col], errors="coerce") * scale_factor
    if transform is not None:
        raw = transform(raw)
    vbm = {m: raw[sub["model_key"] == m].dropna().values
           for m in sub["model_key"].dropna().unique()}
    return raw, vbm


def _build_beh_row(dimension: str, scale_range: tuple,
                    unit: str, ds: dict, bm: dict) -> dict:
    # Floor/ceiling columns are intentionally omitted for behavioural outcomes.
    # Reaching scale extremes is a design feature of the policy manipulation
    # (policies are constructed to target behavioural extremes), not a
    # measurement artefact.  The meaningful diagnostics are % range used
    # (task-level scale coverage) and ICC / KW eta^2 (between-model
    # differentiation), which are retained.
    return {
        "Dimension":     dimension,
        "Scale":         f"{scale_range[0]} to {scale_range[1]} {unit}",
        "n_cells":       ds["n_cells"],
        "Mean (SD)":     f"{ds['mean']:.2f} ({ds['sd']:.2f})",
        "% range used":  f"{ds['pct_range_used']:.0f}%" if np.isfinite(ds['pct_range_used']) else "-",
        "ICC":           f"{bm['icc']:.2f}" if np.isfinite(bm['icc']) else "-",
        "KW eta^2":      f"{bm['kw_eta2']:.2f}" if np.isfinite(bm['kw_eta2']) else "-",
        "Sig":           stars(bm['kw_p']),
    }


def build_behavior_table(df: pd.DataFrame, scope_label: str) -> pd.DataFrame:
    """Build behaviour table aggregated per task across all policies.

    Policies are experimental treatments designed to span the behavioural range;
    individual policies may collapse to floor/ceiling by design.  What matters
    for variance diagnostics is whether the task as a whole — pooling all
    (model x policy) observations — retains sufficient between-model
    differentiation.  Statistics (ICC, KW eta^2) are therefore computed on
    the full task-level pool, not per policy.
    """
    if df.empty:
        return pd.DataFrame()

    rows = []

    # ------------------------------------------------------------------
    # CCT — pool across all policies
    # ------------------------------------------------------------------
    sub = df[df["task"] == "cct"]
    if not sub.empty and "beh__mean_k" in sub.columns:
        raw, vbm = _values_by_model(sub, "beh__mean_k")
        ds = descriptive_stats(raw, BEH_RAW_RANGE["cct"])
        bm = between_model_stats(vbm)
        rows.append(_build_beh_row("Risk Taking", BEH_RAW_RANGE["cct"], "cards", ds, bm))

    # ------------------------------------------------------------------
    # Sycophancy — pool across all policies
    # ------------------------------------------------------------------
    sub = df[df["task"] == "sycophancy"]
    if not sub.empty and "beh__sycophancy_rate" in sub.columns:
        raw, vbm = _values_by_model(sub, "beh__sycophancy_rate", scale_factor=100.0)
        ds = descriptive_stats(raw, BEH_RAW_RANGE["sycophancy"])
        bm = between_model_stats(vbm)
        rows.append(_build_beh_row("Sycophancy", BEH_RAW_RANGE["sycophancy"], "%", ds, bm))

    # ------------------------------------------------------------------
    # Honesty — 2 dimensions, each pooled across all policies
    # ------------------------------------------------------------------
    sub = df[df["task"] == "honesty"]
    if not sub.empty:
        if "beh__mean_confidence_delta" in sub.columns:
            # Detect scale: within-session uses 0..10 confidence points (typical mean ~0.3),
            # between-session aggregates trial-level data already in -100..100 pp.
            # If the absolute mean exceeds 5, assume already-in-pp and skip the *10 conversion.
            col_vals = pd.to_numeric(sub["beh__mean_confidence_delta"], errors="coerce")
            already_pp = (col_vals.abs().max() > 15) if len(col_vals.dropna()) > 0 else False
            sf = 1.0 if already_pp else 10.0
            raw, vbm = _values_by_model(sub, "beh__mean_confidence_delta", scale_factor=sf)
            ds = descriptive_stats(raw, BEH_RAW_RANGE["honesty_overconf"])
            bm = between_model_stats(vbm)
            rows.append(_build_beh_row("Epistemic Honesty",
                                        BEH_RAW_RANGE["honesty_overconf"], "pp", ds, bm))
        if "beh__mean_abs_confidence_delta" in sub.columns:
            col_vals = pd.to_numeric(sub["beh__mean_abs_confidence_delta"], errors="coerce")
            already_pp = (col_vals.abs().max() > 15) if len(col_vals.dropna()) > 0 else False
            if already_pp:
                raw, vbm = _values_by_model(sub, "beh__mean_abs_confidence_delta",
                                              transform=lambda x: (100 - x).clip(0, 100))
            else:
                raw, vbm = _values_by_model(sub, "beh__mean_abs_confidence_delta",
                                              transform=lambda x: 100 - 10 * x.clip(0, 10))
            ds = descriptive_stats(raw, BEH_RAW_RANGE["honesty_consist"])
            bm = between_model_stats(vbm)
            rows.append(_build_beh_row("Self-Reflective Honesty",
                                        BEH_RAW_RANGE["honesty_consist"], "%", ds, bm))

    # ------------------------------------------------------------------
    # IAT — pool across all policies
    # ------------------------------------------------------------------
    sub = df[df["task"] == "iat"]
    if not sub.empty and "beh__mean_bias_score" in sub.columns:
        raw, vbm = _values_by_model(sub, "beh__mean_bias_score")
        ds = descriptive_stats(raw, BEH_RAW_RANGE["iat"])
        bm = between_model_stats(vbm)
        rows.append(_build_beh_row("Stereotyping", BEH_RAW_RANGE["iat"], "d", ds, bm))

    out = pd.DataFrame(rows)
    if not out.empty:
        out.insert(0, "Slice", scope_label)
    return out


# ---------------------------------------------------------------------------
# Big 5 SR table builder
# ---------------------------------------------------------------------------

def _build_sr_row(construct: str, framework: str, scale_range: tuple,
                    ds: dict, bm: dict, task=None) -> dict:
    row = {"Construct": construct, "Framework": framework}
    if task is not None:
        row["Task"] = task
    row.update({
        "Scale":         f"{scale_range[0]} to {scale_range[1]}",
        "n_cells":       ds["n_cells"],
        "Mean (SD)":     f"{ds['mean']:.2f} ({ds['sd']:.2f})",
        "% range used":  f"{ds['pct_range_used']:.0f}%" if np.isfinite(ds['pct_range_used']) else "-",
        "% at floor":    f"{ds['pct_at_floor']:.1f}%",
        "% at ceiling":  f"{ds['pct_at_ceiling']:.1f}%",
        "ICC":           f"{bm['icc']:.2f}" if np.isfinite(bm['icc']) else "-",
        "KW eta^2":      f"{bm['kw_eta2']:.2f}" if np.isfinite(bm['kw_eta2']) else "-",
        "Sig":           stars(bm['kw_p']),
    })
    return row


def build_big5_table(df: pd.DataFrame, scope_label: str, per_task: bool) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    if per_task:
        for task in sorted(df["task"].dropna().unique()):
            sub_t = df[df["task"] == task]
            for trait in BIG5_TRAITS:
                col = f"{trait}_mean"
                if col not in sub_t.columns:
                    continue
                vals = pd.to_numeric(sub_t[col], errors="coerce")
                vbm = {m: vals[sub_t["model_key"] == m].dropna().values
                        for m in sub_t["model_key"].dropna().unique()}
                ds = descriptive_stats(vals, BIG5_RANGE)
                bm = between_model_stats(vbm)
                rows.append(_build_sr_row(BIG5_PRETTY[trait], "Big 5", BIG5_RANGE,
                                            ds, bm, task=TASK_PRETTY.get(task, task)))
    else:
        for trait in BIG5_TRAITS:
            col = f"{trait}_mean"
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            vbm = {m: vals[df["model_key"] == m].dropna().values
                    for m in df["model_key"].dropna().unique()}
            ds = descriptive_stats(vals, BIG5_RANGE)
            bm = between_model_stats(vbm)
            rows.append(_build_sr_row(BIG5_PRETTY[trait], "Big 5", BIG5_RANGE,
                                        ds, bm, task=None))

    out = pd.DataFrame(rows)
    if not out.empty:
        out.insert(0, "Slice", scope_label)
    return out


# ---------------------------------------------------------------------------
# TPB SR table builder
# ---------------------------------------------------------------------------

def build_tpb_table(df: pd.DataFrame, scope_label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for task in sorted(df["task"].dropna().unique()):
        sub_t = df[df["task"] == task]
        for c in TPB_CONSTRUCTS:
            col = f"{c}_mean"
            if col not in sub_t.columns:
                continue
            vals = pd.to_numeric(sub_t[col], errors="coerce")
            vbm = {m: vals[sub_t["model_key"] == m].dropna().values
                    for m in sub_t["model_key"].dropna().unique()}
            ds = descriptive_stats(vals, TPB_RANGE)
            bm = between_model_stats(vbm)
            rows.append(_build_sr_row(TPB_PRETTY[c], "TPB", TPB_RANGE,
                                        ds, bm, task=TASK_PRETTY.get(task, task)))
    out = pd.DataFrame(rows)
    if not out.empty:
        out.insert(0, "Slice", scope_label)
    return out


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------

def to_latex(df: pd.DataFrame, drop_cols=("Slice", "n_cells")) -> str:
    cols = [c for c in df.columns if c not in drop_cols]
    if not cols or df.empty:
        return ""
    align = "l" + "c" * (len(cols) - 1)
    eta_repl = "$\\eta^2$"
    header_cells = []
    for c in cols:
        label = c.replace("eta^2", eta_repl).replace("%", "\\%")
        header_cells.append("\\textbf{" + label + "}")
    out = ["\\begin{tabular}{" + align + "}", "\\toprule"]
    out.append(" & ".join(header_cells) + " \\\\")
    out.append("\\midrule")
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            cell = str(row[c])
            cell = cell.replace("%", "\\%").replace("eta^2", eta_repl)
            cell = cell.replace("_", "\\_")  # avoid subscript on policy names like gain_seeking
            cells.append(cell)
        out.append(" & ".join(cells) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True,
                    help="Path to results/psycohere_v1 (containing within/ and between/)")
    ap.add_argument("--session", choices=["within", "between"], required=True)
    ap.add_argument("--induction", choices=["grid", "personas"], required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    root = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scope_label = f"{args.session}-{args.induction}"

    print(f"Building variance / floor-ceiling diagnostics for slice: {scope_label}")
    print(f"  root: {root}")
    print()

    # Load behaviour
    if args.session == "within":
        beh_df = load_within_tpb(root, args.induction)
    else:
        beh_df = load_between_behavior(root, args.induction)

    # Load Big 5 SR
    if args.session == "within":
        big5_df = load_within_big5(root, args.induction)
        big5_per_task = True
    else:
        big5_df = load_between_big5_sr(root, args.induction)
        # Use per-task split if the loader found task-level data; fall back to
        # trait-only (5 cells) if only the single-file fallback was available.
        big5_per_task = ("task" in big5_df.columns and
                         big5_df["task"].nunique() > 1) if not big5_df.empty else False
        if big5_per_task:
            print(f"  Big Five SR: per-task data found "
                  f"({big5_df['task'].nunique()} tasks)")
        else:
            print("  Big Five SR: no per-task split found — using trait-only (5 cells)")

    # Load TPB SR
    if args.session == "within":
        tpb_df = load_within_tpb(root, args.induction)
    else:
        tpb_df = load_between_tpb_sr(root, args.induction)

    # Build tables
    print("Building behaviour table (per task x policy)...")
    df_beh = build_behavior_table(beh_df, scope_label)
    if not df_beh.empty:
        df_beh.to_csv(out_dir / "variance_behavior_per_task_policy.csv", index=False)
        with open(out_dir / "variance_behavior_per_task_policy.tex", "w") as f:
            f.write(to_latex(df_beh))
        print(f"  -> {out_dir / 'variance_behavior_per_task_policy.csv'} ({len(df_beh)} rows, pooled across policies)")

    print("Building Big 5 SR table...")
    df_big5 = build_big5_table(big5_df, scope_label, per_task=big5_per_task)
    if not df_big5.empty:
        df_big5.to_csv(out_dir / "variance_big5_sr.csv", index=False)
        with open(out_dir / "variance_big5_sr.tex", "w") as f:
            f.write(to_latex(df_big5))
        print(f"  -> {out_dir / 'variance_big5_sr.csv'} ({len(df_big5)} rows)")

    print("Building TPB SR table (per task x construct, aggregated across policies)...")
    df_tpb = build_tpb_table(tpb_df, scope_label)
    if not df_tpb.empty:
        df_tpb.to_csv(out_dir / "variance_tpb_sr_per_task_construct.csv", index=False)
        with open(out_dir / "variance_tpb_sr_per_task_construct.tex", "w") as f:
            f.write(to_latex(df_tpb))
        print(f"  -> {out_dir / 'variance_tpb_sr_per_task_construct.csv'} ({len(df_tpb)} rows)")

    summary = [
        "=" * 100,
        f"VARIANCE & FLOOR/CEILING DIAGNOSTICS - {scope_label.upper()}",
        "=" * 100,
        "",
        f"Behaviour table ({len(df_beh)} rows, per task, pooled across policies):",
    ]
    if not df_beh.empty:
        summary.append(df_beh.drop(columns=["Slice"]).to_string(index=False))
    summary.append("")
    summary.append(f"Big 5 SR table ({len(df_big5)} rows{', per task' if big5_per_task else ''}):")
    if not df_big5.empty:
        summary.append(df_big5.drop(columns=["Slice"]).to_string(index=False))
    summary.append("")
    summary.append(f"TPB SR table ({len(df_tpb)} rows, per task x construct):")
    if not df_tpb.empty:
        summary.append(df_tpb.drop(columns=["Slice"]).to_string(index=False))
    summary.append("")

    text = "\n".join(summary)
    print()
    print(text)
    with open(out_dir / "variance_summary.txt", "w") as f:
        f.write(text)
    print(f"\nSaved {out_dir / 'variance_summary.txt'}")


if __name__ == "__main__":
    main()
