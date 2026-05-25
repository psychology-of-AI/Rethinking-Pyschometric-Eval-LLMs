#!/usr/bin/env python3
"""
rq3_context_separation.py
=========================
RQ3: Context separation. Does the within-session SR-behavior coherence from
RQ1 survive when SR and behavior are conducted in SEPARATE conversation
sessions, or does it collapse --- indicating that RQ1's coherence was
shared-context priming rather than a stable disposition?

Primary comparison
------------------
For every (task, framework, construct, model) cell, we compute the Fisher-z
aggregated Pearson r under TWO session conditions:
  - session_type = 'within'  (SR + behavior in same thread, RQ1 setup)
  - session_type = 'between' (SR and behavior in separate threads)
All else equal (perturbation='grid', same matched keys). We then examine:
  Δr = r_within − r_between

A large positive Δr ⇒ coherence was shared-context only, collapses between
sessions. Near-zero Δr ⇒ stable disposition.

Frameworks
----------
Both TPB and Big5, inheriting the framework-asymmetric outcome convention
from RQ2 (TPB → align_score, Big5 → raw outcome × expected_sign).

Robustness
----------
  - Mundlak pooled OLS with cluster-robust SEs, run separately per session
  - Bootstrap Δr 95% CIs (resampling models) as supplementary to Fisher-z
    pooled SE

Outputs
-------
  rq3_context_cells.csv          — per-cell r_within, r_between, Δr
  rq3_by_task_framework.csv      — per (task × framework × construct × session) Fisher-z
  rq3_delta_table.csv            — per-task best-construct Δr with CI
  rq3_mundlak.csv                — Mundlak β for both sessions
  rq3_bootstrap.csv              — bootstrap Δr CIs
  rq3_headline.json              — structured headline
  rq3_context_summary.{pdf,png}  — 2-panel figure

Usage
-----
python scripts/analysis_scripts/rq3_context_separation.py \\
    --cct_master     results/psycohere_v1/analysis/cct/cct_master.csv \\
    --syc_master     results/psycohere_v1/analysis/sycophancy/sycophancy_master.csv \\
    --honesty_master results/psycohere_v1/analysis/honesty/honesty_master.csv \\
    --iat_master     results/psycohere_v1/analysis/iat/iat_master.csv \\
    --out_dir        results/psycohere_v1/analysis/rq3_context
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
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from rq_config import (
    MODEL_ORDER, MODEL_LABELS, COLORS,
    TASK_ORDER, TASK_LABELS,
    INTENTION_COL, PRIMARY_CONSTRUCT, BIG5_THEORY_PAIRS,
    load_task_tpb, load_task_big5,
    pearson_ci, safe_pearsonr, stars, fisher_z_mean_ci,
    mundlak_within_between,
)


from psycohere_style import (
    apply_style, C, FS, BAR, HEAT,
    style_ax, style_heatmap_ax, add_zero_line,
    panel_title, suptitle as fig_suptitle,
    annotate_bar, heatmap_cell_text,
)
apply_style()

# ── Cell-level computation ────────────────────────────────────────────────

def compute_cells(tpb_within, tpb_between, big5_within, big5_between):
    """Per-model within-r cells for BOTH session conditions and BOTH frameworks.

    Returns a flat DataFrame with columns: framework, task, construct, model,
    session, r, r_aligned, p, ci_lo, ci_hi, n. Each (framework, task,
    construct, model) appears twice — once per session.
    """
    rows = []
    for session_label, data_dict in [("within", tpb_within),
                                      ("between", tpb_between)]:
        for task in TASK_ORDER:
            df = data_dict[task]
            primary_col, primary_label = PRIMARY_CONSTRUCT[task]
            constructs = [(INTENTION_COL, "Intention", +1)]
            if primary_col != INTENTION_COL:
                constructs.append((primary_col, primary_label, +1))

            for construct_col, construct_label, expected_sign in constructs:
                for m in MODEL_ORDER:
                    sub = df[df.model_key == m]
                    r, p, n = safe_pearsonr(sub[construct_col], sub["align_score"])
                    lo, hi = pearson_ci(r, n)
                    r_aligned = r * expected_sign if not np.isnan(r) else np.nan
                    rows.append({
                        "framework":       "TPB",
                        "session":         session_label,
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
                    })

    for session_label, data_dict in [("within", big5_within),
                                      ("between", big5_between)]:
        for task in TASK_ORDER:
            df = data_dict[task]
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
                        "session":         session_label,
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
                    })

    return pd.DataFrame(rows)


# ── Aggregation ───────────────────────────────────────────────────────────

def aggregate_by_task_framework(cells: pd.DataFrame) -> pd.DataFrame:
    """Per (framework × task × construct × session) Fisher-z mean r_aligned."""
    out = (cells.groupby(["framework", "session", "task", "task_label",
                           "construct", "construct_label"])
                .agg(n_cells=("r_aligned", "size"),
                     mean_r=("r_aligned", "mean"))
                .reset_index())
    fz_stats = []
    for _, row in out.iterrows():
        grp = cells[(cells["framework"] == row["framework"]) &
                    (cells["session"] == row["session"]) &
                    (cells["task"] == row["task"]) &
                    (cells["construct"] == row["construct"])]
        mr, lo, hi, k = fisher_z_mean_ci(grp["r_aligned"].values,
                                          grp["n"].values)
        fz_stats.append((mr, lo, hi, k))
    out["fz_mean_r"] = [s[0] for s in fz_stats]
    out["fz_ci_lo"]  = [s[1] for s in fz_stats]
    out["fz_ci_hi"]  = [s[2] for s in fz_stats]
    out["fz_n_used"] = [s[3] for s in fz_stats]
    return out


def compute_delta_r(by_tfs: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """For each (framework × task × construct), compute Δr = within − between
    with a pooled 95% CI on the Fisher-z scale.

    SE(Δz) = sqrt(SE_within² + SE_between²) assuming session independence,
    then back-transform endpoints via tanh.
    """
    rows = []
    keys = by_tfs[["framework", "task", "construct"]].drop_duplicates()
    for _, k in keys.iterrows():
        w = by_tfs[(by_tfs["framework"] == k["framework"]) &
                    (by_tfs["task"] == k["task"]) &
                    (by_tfs["construct"] == k["construct"]) &
                    (by_tfs["session"] == "within")]
        b = by_tfs[(by_tfs["framework"] == k["framework"]) &
                    (by_tfs["task"] == k["task"]) &
                    (by_tfs["construct"] == k["construct"]) &
                    (by_tfs["session"] == "between")]
        if w.empty or b.empty:
            continue
        w = w.iloc[0]; b = b.iloc[0]

        # Recompute z-scale SEs from the cell-level data for this slice
        def _se_z(cells_slice):
            cs = cells_slice.dropna(subset=["r_aligned", "n"])
            cs = cs[cs["n"] > 3]
            if len(cs) == 0: return np.nan, np.nan
            weights = cs["n"].values - 3
            zs = np.arctanh(np.clip(cs["r_aligned"].values, -0.9999, 0.9999))
            z_mean = float(np.sum(weights * zs) / np.sum(weights))
            se_z = float(1.0 / np.sqrt(np.sum(weights)))
            return z_mean, se_z

        cells_w = cells[(cells["framework"] == k["framework"]) &
                         (cells["task"] == k["task"]) &
                         (cells["construct"] == k["construct"]) &
                         (cells["session"] == "within")]
        cells_b = cells[(cells["framework"] == k["framework"]) &
                         (cells["task"] == k["task"]) &
                         (cells["construct"] == k["construct"]) &
                         (cells["session"] == "between")]
        z_w, se_w = _se_z(cells_w)
        z_b, se_b = _se_z(cells_b)
        if np.isnan(z_w) or np.isnan(z_b):
            continue
        dz = z_w - z_b
        dz_se = float(np.sqrt(se_w**2 + se_b**2))
        crit = stats.norm.ppf(0.975)
        dz_lo = dz - crit * dz_se
        dz_hi = dz + crit * dz_se
        dz_z = dz / dz_se if dz_se > 0 else np.nan
        dz_p = 2 * (1 - stats.norm.cdf(abs(dz_z))) if not np.isnan(dz_z) else np.nan

        # Back-transform Δr for reporting (approximation: r_within - r_between
        # on the r scale rather than tanh(Δz); we report both for transparency).
        dr_approx = w["fz_mean_r"] - b["fz_mean_r"]

        rows.append({
            "framework":    k["framework"],
            "task":         k["task"],
            "task_label":   w["task_label"],
            "construct":    k["construct"],
            "construct_label": w["construct_label"],
            "r_within":     w["fz_mean_r"],
            "r_within_lo":  w["fz_ci_lo"],
            "r_within_hi":  w["fz_ci_hi"],
            "r_between":    b["fz_mean_r"],
            "r_between_lo": b["fz_ci_lo"],
            "r_between_hi": b["fz_ci_hi"],
            "delta_r":      dr_approx,
            "delta_z":      dz,
            "delta_z_se":   dz_se,
            "delta_z_lo":   dz_lo,
            "delta_z_hi":   dz_hi,
            # Back-transform z-CI to r-CI for Δr reporting (endpoints approx.)
            "delta_r_lo":   float(np.tanh(z_w - crit*se_w) - np.tanh(z_b + crit*se_b)),
            "delta_r_hi":   float(np.tanh(z_w + crit*se_w) - np.tanh(z_b - crit*se_b)),
            "delta_z_p":    dz_p,
        })
    return pd.DataFrame(rows)


def bootstrap_delta_r(cells: pd.DataFrame, n_boot: int = 2000,
                       seed: int = 42) -> pd.DataFrame:
    """Bootstrap 95% CIs on Δr by resampling models with replacement.
    For each resampled model set, recompute Fisher-z means per session,
    then Δr = r_within − r_between. Percentile CI over bootstrap replicates.

    Implementation: pre-indexes cells into numpy arrays keyed by
    (framework, task, construct, model, session) to avoid pandas filtering
    inside the bootstrap loop, which otherwise makes this O(n_boot × n_cells
    × n_models) slow.
    """
    rng = np.random.default_rng(seed)
    rows = []
    keys = cells[["framework", "task", "construct",
                   "task_label", "construct_label"]].drop_duplicates()
    all_models = sorted(cells["model"].unique())
    n_models = len(all_models)
    model_idx = {m: i for i, m in enumerate(all_models)}

    for _, k in keys.iterrows():
        sub = cells[(cells["framework"] == k["framework"]) &
                    (cells["task"] == k["task"]) &
                    (cells["construct"] == k["construct"])]
        if sub.empty:
            continue

        # Pre-extract arrays indexed by model position for both sessions
        # shape (n_models,) — NaN when session absent for a model
        r_w = np.full(n_models, np.nan)
        n_w = np.full(n_models, np.nan)
        r_b = np.full(n_models, np.nan)
        n_b = np.full(n_models, np.nan)
        for _, row in sub.iterrows():
            mi = model_idx[row["model"]]
            if row["session"] == "within":
                r_w[mi] = row["r_aligned"]; n_w[mi] = row["n"]
            else:
                r_b[mi] = row["r_aligned"]; n_b[mi] = row["n"]

        # Fisher-z transform (vectorized)
        z_w = np.arctanh(np.clip(r_w, -0.9999, 0.9999))
        z_b = np.arctanh(np.clip(r_b, -0.9999, 0.9999))

        def fz_mean(zs, ns, indices):
            """Weighted Fisher-z mean over a sample of model indices.
            Returns tanh(weighted_mean) on the r scale, or NaN."""
            zs_s = zs[indices]; ns_s = ns[indices]
            mask = ~np.isnan(zs_s) & (ns_s > 3)
            if mask.sum() < 2:
                return np.nan
            w = ns_s[mask] - 3
            return float(np.tanh(np.sum(w * zs_s[mask]) / np.sum(w)))

        deltas = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n_models, size=n_models)
            mw = fz_mean(z_w, n_w, idx)
            mb = fz_mean(z_b, n_b, idx)
            deltas[i] = (mw - mb) if not (np.isnan(mw) or np.isnan(mb)) else np.nan

        deltas = deltas[~np.isnan(deltas)]
        if len(deltas) < 10:
            continue
        rows.append({
            "framework":    k["framework"],
            "task":         k["task"],
            "task_label":   k["task_label"],
            "construct":    k["construct"],
            "construct_label": k["construct_label"],
            "boot_delta_r_mean":  float(np.mean(deltas)),
            "boot_delta_r_ci_lo": float(np.quantile(deltas, 0.025)),
            "boot_delta_r_ci_hi": float(np.quantile(deltas, 0.975)),
            "n_boot_valid":       int(len(deltas)),
        })
    return pd.DataFrame(rows)


def compute_mundlak_both_sessions(tpb_within, tpb_between,
                                    big5_within, big5_between) -> pd.DataFrame:
    """Mundlak pooled OLS for every (framework × session × task × construct).
    Big5 β's are sign-flipped to theory-consistent direction."""
    rows = []
    for session_label, tpb_data, big5_data in [
        ("within", tpb_within, big5_within),
        ("between", tpb_between, big5_between),
    ]:
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
                    "framework": "TPB", "session": session_label,
                    "task": task, "task_label": TASK_LABELS[task],
                    "construct": construct_col,
                    "construct_label": construct_label,
                    "expected_sign": +1,
                    **m,
                })
        # Big5
        for task in TASK_ORDER:
            df = big5_data[task]
            for trait_col, trait_label, expected_sign, outcome_col in BIG5_THEORY_PAIRS[task]:
                if trait_col not in df.columns or outcome_col not in df.columns:
                    continue
                m = mundlak_within_between(df, trait_col, outcome_col,
                                             group_col="model_key",
                                             z_normalize=True)
                for k in ["beta_within", "beta_between"]:
                    if not np.isnan(m[k]):
                        m[k] = m[k] * expected_sign
                if expected_sign < 0:
                    m["ci_within_lo"], m["ci_within_hi"] = (
                        -m["ci_within_hi"], -m["ci_within_lo"])
                    m["ci_between_lo"], m["ci_between_hi"] = (
                        -m["ci_between_hi"], -m["ci_between_lo"])
                rows.append({
                    "framework": "Big5", "session": session_label,
                    "task": task, "task_label": TASK_LABELS[task],
                    "construct": trait_col,
                    "construct_label": trait_label,
                    "expected_sign": expected_sign,
                    "outcome_col": outcome_col,
                    **m,
                })
    return pd.DataFrame(rows)


# ── Per-task best-construct Δr ────────────────────────────────────────────

def best_construct_delta(delta_df: pd.DataFrame) -> pd.DataFrame:
    """For each (framework, task), pick the construct with the largest
    within-session r_aligned and report its Δr. This mirrors RQ2's
    head-to-head best-construct framing.
    """
    rows = []
    for (framework, task), grp in delta_df.groupby(["framework", "task"]):
        best = grp.loc[grp["r_within"].idxmax()]
        rows.append({
            "framework":   framework,
            "task":        task,
            "task_label":  best["task_label"],
            "best_construct": best["construct_label"],
            "r_within":    best["r_within"],
            "r_within_lo": best["r_within_lo"],
            "r_within_hi": best["r_within_hi"],
            "r_between":   best["r_between"],
            "r_between_lo": best["r_between_lo"],
            "r_between_hi": best["r_between_hi"],
            "delta_r":     best["delta_r"],
            "delta_r_lo":  best["delta_r_lo"],
            "delta_r_hi":  best["delta_r_hi"],
            "delta_z_p":   best["delta_z_p"],
        })
    return pd.DataFrame(rows)


# ── Per-model aggregation ─────────────────────────────────────────────────

def aggregate_by_model_session(cells: pd.DataFrame,
                                framework: str = "TPB") -> pd.DataFrame:
    """For a given framework, per-model Fisher-z mean r_aligned per session,
    plus per-model Δr with 95% CI and p-value (pooled Fisher-z SE)."""
    sub = cells[cells["framework"] == framework]
    out = []
    for m in MODEL_ORDER:
        for session in ["within", "between"]:
            grp = sub[(sub["model"] == m) & (sub["session"] == session)]
            if grp.empty: continue
            mr, lo, hi, k = fisher_z_mean_ci(grp["r_aligned"].values,
                                              grp["n"].values)
            out.append({
                "framework":   framework,
                "model":       m,
                "model_label": MODEL_LABELS[m],
                "session":     session,
                "fz_mean_r":   mr,
                "fz_ci_lo":    lo,
                "fz_ci_hi":    hi,
                "n_cells":     k,
            })
    out_df = pd.DataFrame(out)

    # Wide format for Δr per model, computed with pooled SE on z-scale
    def _z_and_se(cells_slice):
        cs = cells_slice.dropna(subset=["r_aligned", "n"])
        cs = cs[cs["n"] > 3]
        if len(cs) < 2: return np.nan, np.nan
        weights = cs["n"].values - 3
        zs = np.arctanh(np.clip(cs["r_aligned"].values, -0.9999, 0.9999))
        z_mean = float(np.sum(weights * zs) / np.sum(weights))
        se_z = float(1.0 / np.sqrt(np.sum(weights)))
        return z_mean, se_z

    crit = stats.norm.ppf(0.975)
    wide_rows = []
    for m in MODEL_ORDER:
        cw = sub[(sub["model"] == m) & (sub["session"] == "within")]
        cb = sub[(sub["model"] == m) & (sub["session"] == "between")]
        if cw.empty or cb.empty: continue
        z_w, se_w = _z_and_se(cw)
        z_b, se_b = _z_and_se(cb)
        if np.isnan(z_w) or np.isnan(z_b): continue
        dz = z_w - z_b
        dz_se = float(np.sqrt(se_w**2 + se_b**2))
        dz_lo = dz - crit * dz_se
        dz_hi = dz + crit * dz_se
        dz_p = 2 * (1 - stats.norm.cdf(abs(dz / dz_se))) if dz_se > 0 else np.nan
        # Back-transform for reporting on r scale
        r_w = float(np.tanh(z_w))
        r_b = float(np.tanh(z_b))
        # CI endpoints on r scale via tanh of z-CI endpoints
        r_w_lo = float(np.tanh(z_w - crit * se_w))
        r_w_hi = float(np.tanh(z_w + crit * se_w))
        r_b_lo = float(np.tanh(z_b - crit * se_b))
        r_b_hi = float(np.tanh(z_b + crit * se_b))
        dr = r_w - r_b
        # Approximate Δr CI via tanh endpoints of z-CI (same as per-task version)
        dr_lo = float(np.tanh(z_w - crit*se_w) - np.tanh(z_b + crit*se_b))
        dr_hi = float(np.tanh(z_w + crit*se_w) - np.tanh(z_b - crit*se_b))
        wide_rows.append({
            "model":       m,
            "model_label": MODEL_LABELS[m],
            "r_within":    r_w,
            "r_within_lo": r_w_lo,
            "r_within_hi": r_w_hi,
            "r_between":   r_b,
            "r_between_lo": r_b_lo,
            "r_between_hi": r_b_hi,
            "delta_r":     dr,
            "delta_r_lo":  dr_lo,
            "delta_r_hi":  dr_hi,
            "delta_z":     dz,
            "delta_z_se":  dz_se,
            "delta_z_p":   dz_p,
        })
    wide = pd.DataFrame(wide_rows)
    wide = wide.sort_values("r_within", ascending=False).reset_index(drop=True)
    return out_df, wide


# ── Printout ──────────────────────────────────────────────────────────────

def print_headline(delta_df: pd.DataFrame, best_df: pd.DataFrame,
                   per_model_wide_tpb: pd.DataFrame,
                   boot: pd.DataFrame) -> dict:
    print("\n" + "=" * 78)
    print("RQ3 HEADLINE — Context Separation (within vs between session)")
    print("Does shared-context coherence survive session separation?")
    print("=" * 78)

    print("\n" + "-" * 78)
    print("PER-TASK × FRAMEWORK Δr = r_within − r_between (best construct)")
    print("-" * 78)
    print(f"{'Task':<12}  {'Framework':<8}  {'Best Construct':<18}  "
          f"{'r_within':<10}  {'r_between':<10}  {'Δr [95% CI]':<22}")
    for _, row in best_df.sort_values(["framework", "task"]).iterrows():
        dr_str = (f"{row['delta_r']:+.2f} [{row['delta_r_lo']:+.2f},"
                  f"{row['delta_r_hi']:+.2f}]{stars(row['delta_z_p'])}")
        print(f"  {row['task_label']:<10}  {row['framework']:<8}  "
              f"{row['best_construct']:<18}  "
              f"{row['r_within']:+.2f}      {row['r_between']:+.2f}      {dr_str}")

    print("\n" + "-" * 78)
    print("PER-MODEL Δr (TPB framework, pooled across tasks/constructs)")
    print("-" * 78)
    print(f"{'Model':<22}  {'r_within [CI]':<22}  "
          f"{'r_between [CI]':<22}  {'Δr':<8}")
    for _, row in per_model_wide_tpb.iterrows():
        rw = f"{row['r_within']:+.2f} [{row['r_within_lo']:+.2f},{row['r_within_hi']:+.2f}]"
        rb = f"{row['r_between']:+.2f} [{row['r_between_lo']:+.2f},{row['r_between_hi']:+.2f}]"
        print(f"  {row['model_label']:<22}  {rw:<22}  {rb:<22}  "
              f"{row['delta_r']:+.2f}")

    print("\n" + "-" * 78)
    print("BOOTSTRAP Δr 95% CIs (2000 iterations, resampling models)")
    print("-" * 78)
    print(f"{'Task':<12}  {'Framework':<8}  {'Construct':<20}  "
          f"{'Δr (Fisher-z)':<16}  {'Δr (bootstrap)':<22}")
    # Join best-construct with bootstrap
    for _, row in best_df.sort_values(["framework", "task"]).iterrows():
        boot_row = boot[(boot["framework"] == row["framework"]) &
                         (boot["task"] == row["task"]) &
                         (boot["construct_label"] == row["best_construct"])]
        if boot_row.empty: continue
        br = boot_row.iloc[0]
        fz_str = f"{row['delta_r']:+.2f} [{row['delta_r_lo']:+.2f},{row['delta_r_hi']:+.2f}]"
        bt_str = (f"{br['boot_delta_r_mean']:+.2f} "
                  f"[{br['boot_delta_r_ci_lo']:+.2f},{br['boot_delta_r_ci_hi']:+.2f}]")
        print(f"  {row['task_label']:<10}  {row['framework']:<8}  "
              f"{row['best_construct']:<20}  {fz_str:<16}  {bt_str:<22}")

    # Build headline JSON
    overall = {}
    for framework in ["TPB", "Big5"]:
        rows = best_df[best_df["framework"] == framework]
        overall[framework] = {
            "mean_within":  round(float(rows["r_within"].mean()), 4),
            "mean_between": round(float(rows["r_between"].mean()), 4),
            "mean_delta":   round(float(rows["delta_r"].mean()), 4),
        }
    return {
        "overall_per_framework": overall,
        "per_task_best": [
            {
                "framework":        r["framework"],
                "task":             r["task_label"],
                "construct":        r["best_construct"],
                "r_within":         round(float(r["r_within"]), 3),
                "r_between":        round(float(r["r_between"]), 3),
                "delta_r":          round(float(r["delta_r"]), 3),
                "delta_r_ci":       [round(float(r["delta_r_lo"]), 3),
                                      round(float(r["delta_r_hi"]), 3)],
                "delta_p":          (round(float(r["delta_z_p"]), 4)
                                      if pd.notna(r["delta_z_p"]) else None),
            }
            for _, r in best_df.sort_values(["framework", "task"]).iterrows()
        ],
    }


# ── Figure ────────────────────────────────────────────────────────────────

# ── SR / Behavior consistency between sessions ────────────────────────────

def compute_consistency_per_task(tpb_within: dict, tpb_between: dict) -> pd.DataFrame:
    """Per-task SR-context-stability and Behavior-context-stability.

    For each (task × model), pair the within-session and between-session cells
    on (model × policy × persona × seed × temperature × top_p) and compute:
      - SR consistency  = Pearson r between SR-within and SR-between
      - Beh consistency = Pearson r between Behavior-within and Behavior-between

    Aggregate across models via inverse-variance-weighted Fisher-z meta.

    Interpretation (per Rafal):
      - SR is elicited first in BOTH within and between modes; the model does
        not know during SR whether behavior will follow. So SR consistency
        cleanly tests whether the SR is robust to a context shift (knowing
        that behavior comes next in same conv vs separate conv).
      - Behavior consistency mixes context-stability with priming: in within
        mode, behavior is preceded by SR (potential prime); in between mode,
        no SR was just provided. Low behavior consistency therefore implies
        that behavior is heavily shaped by the immediately-preceding SR,
        i.e. context-window priming.
    """
    # Per-task SR construct (theoretically primary for that task) and
    # behavioral outcome.
    SR_COL = {
        "cct":     "intention_mean",
        "syc":     "intention_mean",
        "honesty": "attitude_mean",
        "iat":     "intention_mean",
    }
    BEH_COL = "align_score"  # task-specific, sign-corrected, populated in both sessions

    rows = []
    for task in TASK_ORDER:
        sr_col = SR_COL[task]
        # Build cell-level paired DataFrame
        w_data = tpb_within[task]
        b_data = tpb_between[task]

        key_cols = [c for c in ["model_key", "policy_id", "persona_label",
                                 "seed", "temperature", "top_p"]
                    if c in w_data.columns and c in b_data.columns]
        # Ensure columns exist
        if sr_col not in w_data.columns or sr_col not in b_data.columns: continue
        if BEH_COL not in w_data.columns or BEH_COL not in b_data.columns: continue

        w_cells = (w_data.groupby(key_cols)
                         .agg(sr_w=(sr_col, "mean"), beh_w=(BEH_COL, "mean"))
                         .reset_index())
        b_cells = (b_data.groupby(key_cols)
                         .agg(sr_b=(sr_col, "mean"), beh_b=(BEH_COL, "mean"))
                         .reset_index())
        merged = w_cells.merge(b_cells, on=key_cols, how="inner")

        # Per-model correlations
        sr_rs, beh_rs, ns = [], [], []
        for m in sorted(merged["model_key"].unique()):
            sub = merged[merged["model_key"] == m]
            r_sr, _, n_sr = safe_pearsonr(sub["sr_w"], sub["sr_b"])
            r_b,  _, n_b  = safe_pearsonr(sub["beh_w"], sub["beh_b"])
            if not np.isnan(r_sr): sr_rs.append((m, r_sr, n_sr))
            if not np.isnan(r_b):  beh_rs.append((m, r_b,  n_b))
            ns.append(n_sr)

        # Fisher-z aggregate across models for each metric
        sr_z, sr_lo, sr_hi, _ = fisher_z_mean_ci(
            np.array([r for _, r, _ in sr_rs]),
            np.array([n for _, _, n in sr_rs]),
        )
        beh_z, beh_lo, beh_hi, _ = fisher_z_mean_ci(
            np.array([r for _, r, _ in beh_rs]),
            np.array([n for _, _, n in beh_rs]),
        )
        rows.append({
            "task":              task,
            "task_label":        TASK_LABELS[task],
            "sr_construct":      sr_col,
            "n_models_sr":       len(sr_rs),
            "n_models_beh":      len(beh_rs),
            "sr_consistency":    sr_z,
            "sr_consistency_lo": sr_lo,
            "sr_consistency_hi": sr_hi,
            "beh_consistency":   beh_z,
            "beh_consistency_lo": beh_lo,
            "beh_consistency_hi": beh_hi,
        })
    return pd.DataFrame(rows)


def compute_consistency_per_model_task(tpb_within: dict,
                                        tpb_between: dict) -> pd.DataFrame:
    """Per-(model × task) SR consistency and Behavior consistency.

    Returns one row per (task, model) with raw r values and p-values.
    Used to render the per-model × task heatmaps in Panel C.
    """
    SR_COL = {
        "cct":     "intention_mean",
        "syc":     "intention_mean",
        "honesty": "attitude_mean",
        "iat":     "intention_mean",
    }
    BEH_COL = "align_score"

    rows = []
    for task in TASK_ORDER:
        sr_col = SR_COL[task]
        w_data = tpb_within[task]
        b_data = tpb_between[task]

        key_cols = [c for c in ["model_key", "policy_id", "persona_label",
                                 "seed", "temperature", "top_p"]
                    if c in w_data.columns and c in b_data.columns]
        if sr_col not in w_data.columns or sr_col not in b_data.columns: continue
        if BEH_COL not in w_data.columns or BEH_COL not in b_data.columns: continue

        w_cells = (w_data.groupby(key_cols)
                         .agg(sr_w=(sr_col, "mean"), beh_w=(BEH_COL, "mean"))
                         .reset_index())
        b_cells = (b_data.groupby(key_cols)
                         .agg(sr_b=(sr_col, "mean"), beh_b=(BEH_COL, "mean"))
                         .reset_index())
        merged = w_cells.merge(b_cells, on=key_cols, how="inner")

        for m in sorted(merged["model_key"].unique()):
            sub = merged[merged["model_key"] == m]
            r_sr, p_sr, n_sr = safe_pearsonr(sub["sr_w"], sub["sr_b"])
            r_b,  p_b,  n_b  = safe_pearsonr(sub["beh_w"], sub["beh_b"])
            rows.append({
                "task":          task,
                "task_label":    TASK_LABELS[task],
                "model":         m,
                "model_label":   MODEL_LABELS.get(m, m),
                "n":             n_sr,
                "r_sr":          r_sr,
                "p_sr":          p_sr,
                "r_beh":         r_b,
                "p_beh":         p_b,
            })
    return pd.DataFrame(rows)


def figure_rq3(by_tfs: pd.DataFrame,
               cells: pd.DataFrame,
               consistency_per_model: pd.DataFrame,
               out_dir: Path):
    """Three-panel RQ3 figure (coarse to fine), matching RQ1/RQ2 conventions.

    Panel A1/A2: per-task within-model coherence in same-session vs
                 separate-sessions; all TPB constructs (Intention + primary)
                 per task. TPB only.
    Panel B:     per-model TPB same-session vs separate-sessions paired bars.
    Panel C:     side-by-side heatmaps — left = SR consistency
                 (rows=models, cols=tasks), right = Behavior consistency.
    """
    import matplotlib.patheffects as path_effects
    from matplotlib.patches import Patch
    from matplotlib.colors import TwoSlopeNorm

    fig = plt.figure(figsize=(24, 9.0))
    gs_outer = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.4],
                                wspace=0.40, left=0.05, right=0.95,
                                top=0.88, bottom=0.10)

    # Panel A is split vertically: A1 (same-session) + A2 (separate-sessions)
    gs_a = gs_outer[0, 0].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.30)
    ax_a_w = fig.add_subplot(gs_a[0, 0])
    ax_a_b = fig.add_subplot(gs_a[1, 0])

    ax_b   = fig.add_subplot(gs_outer[0, 1])

    # Panel C is split horizontally into left (SR) and right (Behavior),
    # with a thin gap and visual separator (matching RQ1/RQ2 task-separator style)
    gs_c = gs_outer[0, 2].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.08)
    ax_c_sr  = fig.add_subplot(gs_c[0, 0])
    ax_c_beh = fig.add_subplot(gs_c[0, 1])

    # ── Task ordering: same-session TPB best descending (matches RQ1/RQ2) ──
    tpb_within_only = by_tfs[(by_tfs["framework"] == "TPB") &
                              (by_tfs["session"] == "within")]
    task_max_tpb = (tpb_within_only.groupby("task")["fz_mean_r"]
                                    .max().to_dict())
    tasks_sorted = sorted(TASK_ORDER, key=lambda t: task_max_tpb.get(t, 0),
                          reverse=True)

    # ── Panel A subpanels: per-task TPB-only, all constructs ──
    bar_h_a = BAR["height"]

    def _plot_a(ax, session_label, title_letter, nice_session,
                primary_color, secondary_color, primary_label_color):
        cur_y = 0
        y_pos_a, y_labels = [], []
        for task in tasks_sorted:
            # All TPB constructs for this task
            rows = by_tfs[(by_tfs["framework"] == "TPB") &
                           (by_tfs["task"] == task) &
                           (by_tfs["session"] == session_label)]
            if rows.empty: continue

            # Sort within-task by primary first, then by other constructs.
            primary_col, primary_label = PRIMARY_CONSTRUCT[task]
            ordered = []
            for _, r in rows.iterrows():
                is_primary = (r["construct"] == primary_col)
                ordered.append((r, is_primary))
            ordered.sort(key=lambda x: (not x[1], x[0]["construct_label"]))

            slot_y_centres = []
            for k, (row, is_primary) in enumerate(ordered):
                yp = cur_y - k * 0.7
                color = primary_color if is_primary else secondary_color
                ax.barh(yp, row["fz_mean_r"], height=bar_h_a,
                        color=color, alpha=BAR["alpha"], edgecolor=C["bar_edge"], linewidth=BAR["edge_lw"], zorder=3)
                err_lo = max(0, row["fz_mean_r"] - row["fz_ci_lo"])
                err_hi = max(0, row["fz_ci_hi"] - row["fz_mean_r"])
                ax.errorbar(row["fz_mean_r"], yp,
                            xerr=[[err_lo], [err_hi]],
                            fmt="none", ecolor=BAR["ecolor"],
                            capsize=BAR["capsize"], lw=BAR["elinewidth"], zorder=4)
                t = ax.text(0.01, yp, f"  {row['construct_label']}",
                            transform=ax.get_yaxis_transform(),
                            va="center", fontsize=FS["construct_tag"], color=primary_label_color,
                            fontweight="bold", zorder=6)
                t.set_path_effects([
                    path_effects.withStroke(linewidth=4, foreground="white")
                ])
                slot_y_centres.append(yp)
            if slot_y_centres:
                y_pos_a.append(np.mean(slot_y_centres))
                y_labels.append(TASK_LABELS[task])
                cur_y -= len(slot_y_centres) * 0.7 + 0.5

        add_zero_line(ax, "v")
        ax.set_yticks(y_pos_a)
        ax.set_yticklabels(y_labels, fontsize=FS["tick"], fontweight="bold")
        ax.set_xlim(-0.85, 0.95)
        ax.set_xlabel("Fisher-z mean r_aligned (95% CI)", fontsize=FS["axis_label"])
        ax.tick_params(axis="x", labelsize=FS["tick"])
        ax.set_title(f"{title_letter}. Per-task TPB — {nice_session}",
                     fontsize=FS["panel_title"], fontweight="bold", loc="left", pad=8,
                     x=-0.18)
        style_ax(ax, grid_axis="x")

    # A1 (same-session) = orange family (matches TPB)
    _plot_a(ax_a_w, "within",  "A1", "same-session",
            primary_color=C["warm"], secondary_color=C["warm_light"],
            primary_label_color=C["warm_label"])
    # A2 (separate-sessions) = teal family
    _plot_a(ax_a_b, "between", "A2", "separate-sessions",
            primary_color=C["teal"], secondary_color=C["teal_light"],
            primary_label_color=C["teal_label"])

    # ── Panel B: per-model TPB same-session vs separate-sessions ──
    tpb_only = cells[cells["framework"] == "TPB"]
    per_model_session = []
    for m in MODEL_ORDER:
        for sess in ("within", "between"):
            csub = tpb_only[(tpb_only["model"] == m) &
                            (tpb_only["session"] == sess)]
            if len(csub) == 0: continue
            mr, lo, hi, k = fisher_z_mean_ci(csub["r_aligned"].values,
                                              csub["n"].values)
            per_model_session.append({"model": m, "session": sess,
                                       "fz_mean_r": mr, "fz_ci_lo": lo,
                                       "fz_ci_hi": hi})
    pm = pd.DataFrame(per_model_session)
    within_sort = (pm[pm["session"] == "within"]
                     .sort_values("fz_mean_r", ascending=False)["model"].tolist())
    n_models = len(within_sort)

    bar_h_b = 0.36
    y_offsets = {"within": +bar_h_b/2 + 0.04, "between": -bar_h_b/2 - 0.04}
    sess_color = {"within": C["warm"], "between": C["teal"]}
    sess_alpha = {"within": 0.92, "between": 0.92}

    y_pos = np.arange(n_models)[::-1]
    for i, m in enumerate(within_sort):
        for sess in ("within", "between"):
            row = pm[(pm["model"] == m) & (pm["session"] == sess)]
            if len(row) == 0: continue
            r  = row["fz_mean_r"].iloc[0]
            lo = row["fz_ci_lo"].iloc[0]; hi = row["fz_ci_hi"].iloc[0]
            ax_b.barh(y_pos[i] + y_offsets[sess], r, height=bar_h_b,
                      color=sess_color[sess], alpha=BAR["alpha"], edgecolor=C["bar_edge"], linewidth=BAR["edge_lw"], zorder=3)
            err_lo = max(0, r - lo); err_hi = max(0, hi - r)
            ax_b.errorbar(r, y_pos[i] + y_offsets[sess],
                          xerr=[[err_lo], [err_hi]],
                          fmt="none", ecolor=BAR["ecolor"], capsize=BAR["capsize"], lw=BAR["elinewidth"],
                          zorder=4)

    add_zero_line(ax_b, "v")
    ax_b.set_yticks(y_pos)

    # Identify retained models: separate-sessions CI strictly above 0
    retained = set()
    for m in within_sort:
        rb = pm[(pm["model"] == m) & (pm["session"] == "between")]
        if len(rb) == 0: continue
        if rb["fz_ci_lo"].iloc[0] > 0:
            retained.add(m)

    # Highlight retained models: yellow tinted background band + green text + checkmark
    for i, m in enumerate(within_sort):
        if m in retained:
            ax_b.axhspan(y_pos[i] - 0.5, y_pos[i] + 0.5,
                         facecolor=C["survivor_bg"], alpha=0.55, zorder=0)

    yticklabels = []
    for m in within_sort:
        if m in retained:
            yticklabels.append(f"{MODEL_LABELS[m]} ✓")
        else:
            yticklabels.append(MODEL_LABELS[m])
    ax_b.set_yticklabels(yticklabels, fontsize=FS["tick"])
    # Color retained model labels green
    for tick_label, m in zip(ax_b.get_yticklabels(), within_sort):
        if m in retained:
            tick_label.set_color("#1B5E20")
            tick_label.set_fontweight("bold")
    ax_b.set_xlabel("Fisher-z mean r_aligned (95% CI)", fontsize=FS["axis_label"])
    ax_b.tick_params(axis="x", labelsize=FS["tick"])
    x_ext = max(0.85, pm[["fz_ci_lo", "fz_ci_hi"]].abs().max().max() + 0.1)
    ax_b.set_xlim(-x_ext, x_ext)
    # Tighten vertical extent to match the heatmap: matplotlib imshow places
    # row i at integer i with cell extent i-0.5 to i+0.5, so matching y-lim
    # gives 1:1 vertical alignment with Panel C heatmaps
    ax_b.set_ylim(-0.5, n_models - 0.5)
    ax_b.set_title("B. Per-model TPB: same vs separate sessions",
               fontsize=FS["panel_title"], fontweight="bold", loc="left", pad=8,
               x=-0.20)
    style_ax(ax_b, grid_axis="x")
    ax_b.legend(handles=[
        Patch(facecolor=C["warm"], alpha=BAR["alpha"], label="same-session"),
        Patch(facecolor=C["teal"], alpha=BAR["alpha"], label="separate-sessions"),
    ], loc="lower right", fontsize=FS["legend"], frameon=True, framealpha=0.9)

    # ── Panel C: side-by-side heatmaps (left = SR, right = Behavior) ──
    TASK_ABBREV = {"cct": "CCT", "syc": "Syc.", "honesty": "Hon.", "iat": "IAT"}

    def _plot_heatmap(ax, value_col, p_col, title, model_order, task_order,
                      title_x=0.0):
        n_m = len(model_order)
        n_t = len(task_order)
        heat = np.full((n_m, n_t), np.nan)
        sigmask = np.zeros((n_m, n_t), dtype=bool)
        for j, t in enumerate(task_order):
            for i, m in enumerate(model_order):
                row = consistency_per_model[
                    (consistency_per_model["task"] == t) &
                    (consistency_per_model["model"] == m)
                ]
                if len(row):
                    v = row[value_col].iloc[0]
                    if not np.isnan(v):
                        heat[i, j] = v
                    p = row[p_col].iloc[0]
                    if not np.isnan(p):
                        sigmask[i, j] = (p < 0.05)

        vmax = max(0.05, float(np.nanmax(np.abs(heat))))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(heat, aspect="auto", cmap=HEAT["cmap"], norm=norm)

        for i in range(n_m):
            for j in range(n_t):
                v = heat[i, j]
                if np.isnan(v): continue
                txt_color = "white" if abs(v) > 0.45 else "black"
                marker = "*" if sigmask[i, j] else ""
                ax.text(j, i, f"{v:+.2f}{marker}",
                        ha="center", va="center", fontsize=FS["heatmap_cell"],
                        color=txt_color, fontweight="bold")

        ax.set_xticks(range(n_t))
        ax.set_xticklabels([TASK_ABBREV.get(t, TASK_LABELS[t])
                            for t in task_order], fontsize=FS["tick"], fontweight="bold")
        ax.set_yticks(range(n_m))
        ax.set_yticklabels([MODEL_LABELS[m] for m in model_order], fontsize=FS["tick"])
        ax.set_title(title, fontsize=FS["panel_title"], fontweight="bold", loc="left", pad=8,
                     x=title_x)
        style_heatmap_ax(ax)
        return im

    # Use the same model order as Panel B (TPB same-session r descending)
    im_sr = _plot_heatmap(ax_c_sr, "r_sr", "p_sr",
                           "C1. Self-report consistency", within_sort, tasks_sorted,
                           title_x=-0.30)
    im_b  = _plot_heatmap(ax_c_beh, "r_beh", "p_beh",
                           "C2. Behavior consistency", within_sort, tasks_sorted)

    # Hide y-tick labels on the right heatmap to save space
    ax_c_beh.set_yticklabels([])

    # Shared colorbar — placed in its own axes via fig.add_axes so neither
    # C1 nor C2 gets shrunk
    pos = ax_c_beh.get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.008, pos.y0, 0.012, pos.height])
    cbar = fig.colorbar(im_b, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Cross-session r  (* = p < .05)", fontsize=FS["colorbar"])

    suptitle = ("RQ3 — Context Separation: which tasks survive when "
                "Self-Report (SR) and behavior are in separate conversations?")
    fig.suptitle(suptitle, fontsize=FS["suptitle"], fontweight="bold", y=0.97)

    for ext in ["pdf", "png"]:
        p = out_dir / f"rq3_context_summary.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  saved: {p}")
    plt.close(fig)


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
                    default="results/psycohere_v1/analysis/rq3_context")
    ap.add_argument("--n_boot", type=int, default=2000,
                    help="Bootstrap iterations for Δr CIs.")
    ap.add_argument("--out_suffix", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    masters = {
        "cct":     args.cct_master,
        "syc":     args.syc_master,
        "honesty": args.honesty_master,
        "iat":     args.iat_master,
    }

    print("Loading TPB + Big5 masters (within AND between session, grid)...")
    tpb_within  = {t: load_task_tpb(p, t, "within",  "grid") for t, p in masters.items()}
    tpb_between = {t: load_task_tpb(p, t, "between", "grid") for t, p in masters.items()}
    big5_within  = {t: load_task_big5(p, t, "within",  "grid") for t, p in masters.items()}
    big5_between = {t: load_task_big5(p, t, "between", "grid") for t, p in masters.items()}

    print("\nData coverage:")
    for t in TASK_ORDER:
        print(f"  {t:<10}  TPB within={len(tpb_within[t]):>5}  "
              f"between={len(tpb_between[t]):>5}    "
              f"Big5 within={len(big5_within[t]):>5}  "
              f"between={len(big5_between[t]):>5}")

    cells = compute_cells(tpb_within, tpb_between, big5_within, big5_between)
    by_tfs = aggregate_by_task_framework(cells)
    delta_df = compute_delta_r(by_tfs, cells)
    best_df = best_construct_delta(delta_df)
    print(f"\nRunning bootstrap Δr CIs ({args.n_boot} iterations)...")
    boot = bootstrap_delta_r(cells, n_boot=args.n_boot)
    print("Running Mundlak OLS for both sessions...")
    mundlak = compute_mundlak_both_sessions(
        tpb_within, tpb_between, big5_within, big5_between)

    per_model_tpb_long, per_model_tpb_wide = aggregate_by_model_session(
        cells, framework="TPB")
    per_model_b5_long, per_model_b5_wide = aggregate_by_model_session(
        cells, framework="Big5")

    # Save CSVs
    suffix = args.out_suffix
    cells.to_csv(out_dir / f"rq3_context_cells{suffix}.csv", index=False)
    by_tfs.to_csv(out_dir / f"rq3_by_task_framework{suffix}.csv", index=False)
    delta_df.to_csv(out_dir / f"rq3_delta_table{suffix}.csv", index=False)
    best_df.to_csv(out_dir / f"rq3_best_construct_delta{suffix}.csv", index=False)
    boot.to_csv(out_dir / f"rq3_bootstrap{suffix}.csv", index=False)
    mundlak.to_csv(out_dir / f"rq3_mundlak{suffix}.csv", index=False)
    per_model_tpb_wide.to_csv(out_dir / f"rq3_per_model_tpb{suffix}.csv", index=False)
    per_model_b5_wide.to_csv(out_dir / f"rq3_per_model_big5{suffix}.csv", index=False)
    print(f"\nSaved CSVs in {out_dir}")

    headline = print_headline(delta_df, best_df, per_model_tpb_wide, boot)
    with open(out_dir / f"rq3_headline{suffix}.json", "w") as f:
        json.dump(headline, f, indent=2)

    print("\nComputing per-task SR/behaviour cross-session consistency (aggregate)...")
    consistency = compute_consistency_per_task(tpb_within, tpb_between)
    consistency.to_csv(out_dir / f"rq3_consistency{suffix}.csv", index=False)
    print(consistency.round(3).to_string(index=False))

    print("\nComputing per-(model x task) cross-session consistency...")
    consistency_per_model = compute_consistency_per_model_task(
        tpb_within, tpb_between)
    consistency_per_model.to_csv(
        out_dir / f"rq3_consistency_per_model{suffix}.csv", index=False)

    print("\nGenerating RQ3 summary figure...")
    figure_rq3(by_tfs, cells, consistency_per_model, out_dir)

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()