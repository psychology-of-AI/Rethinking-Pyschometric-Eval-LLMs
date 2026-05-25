#!/usr/bin/env python3
"""
rq4_induction_comparison.py
===========================
RQ4: Induction Comparison. Does persona-based identity induction produce
different SR-behaviour coherence than parameter-grid induction? Specifically,
can persona induction RESCUE coherence in the between-session condition
where grid collapsed (RQ3)?

Primary comparison
------------------
For every (framework, task, construct, session, model) cell, we compute the
Fisher-z aggregated Pearson r under TWO induction conditions:
  - perturbation = 'grid'      (parameter-level variation at fixed persona)
  - perturbation = 'personas'  (persona-based identity induction)
All else equal (same session_type, same matched keys). We then examine:
  Δr_induction = r_personas − r_grid

Unlike RQ3, positive Δr here means personas HELPS; this is the intuitive
direction (no sign flipping for display). The headline question is whether
between-session Δr_induction is significantly positive — i.e. whether
carrying a persona label across sessions reproduces coherence that parameter
variation alone cannot sustain.

Frameworks
----------
Both TPB and Big5, inheriting the framework-asymmetric outcome convention
from RQ2/RQ3 (TPB -> align_score, Big5 -> raw outcome × expected_sign).

Robustness
----------
  - Mundlak pooled OLS with cluster-robust SEs, run separately per (session,
    induction) cell — 4 Mundlak fits per task × construct
  - Bootstrap Δr_induction 95% CIs (resampling models) as supplementary to
    Fisher-z pooled SE

Outputs
-------
  rq4_cells.csv                      — per-cell r under each (session, induction)
  rq4_by_task_framework_induction.csv — (framework × task × construct × session × induction) Fisher-z
  rq4_delta_table.csv                — per-cell Δr_induction with CI, p
  rq4_best_construct_delta.csv       — per-task best-construct Δr (both sessions)
  rq4_mundlak.csv                    — Mundlak β for all 4 cells (grid/personas × within/between)
  rq4_bootstrap.csv                  — bootstrap Δr_induction CIs
  rq4_per_model_tpb_between.csv      — per-model r_BG, r_BP, Δr for Panel B
  rq4_headline.json                  — structured headline JSON
  rq4_induction_summary.{pdf,png}    — main figure (between-session rescue, 2 panels)
  rq4_induction_appendix.{pdf,png}   — appendix figure (full 2×2 interaction)

Usage
-----
python rq4_induction_comparison.py \\
    --cct_master     cct_master.csv \\
    --syc_master     sycophancy_master.csv \\
    --honesty_master honesty_master.csv \\
    --iat_master     iat_master.csv \\
    --out_dir        results/psycohere_v1/analysis/rq4_induction
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

def compute_cells(data_tables: dict) -> pd.DataFrame:
    """Per-model r cells across all (framework × session × induction × task × construct).

    data_tables is a nested dict:
        data_tables[framework][session][induction][task] -> DataFrame
    where framework ∈ {'tpb','big5'}, session ∈ {'within','between'},
    induction ∈ {'grid','personas'}, task ∈ TASK_ORDER.

    Returns a flat DataFrame with columns: framework, session, induction,
    task, construct, model, r, r_aligned, p, ci_lo, ci_hi, n.
    """
    rows = []
    # TPB cells
    for session in ("within", "between"):
        for induction in ("grid", "personas"):
            for task in TASK_ORDER:
                df = data_tables["tpb"][session][induction][task]
                primary_col, primary_label = PRIMARY_CONSTRUCT[task]
                constructs = [(INTENTION_COL, "Intention", +1)]
                if primary_col != INTENTION_COL:
                    constructs.append((primary_col, primary_label, +1))
                for construct_col, construct_label, expected_sign in constructs:
                    for m in MODEL_ORDER:
                        sub = df[df.model_key == m]
                        r, p, n = safe_pearsonr(sub[construct_col],
                                                sub["align_score"])
                        lo, hi = pearson_ci(r, n)
                        r_al = r * expected_sign if not np.isnan(r) else np.nan
                        rows.append({
                            "framework":       "TPB",
                            "session":         session,
                            "induction":       induction,
                            "task":            task,
                            "task_label":      TASK_LABELS[task],
                            "construct":       construct_col,
                            "construct_label": construct_label,
                            "expected_sign":   expected_sign,
                            "outcome_col":     "align_score",
                            "model":           m,
                            "model_label":     MODEL_LABELS[m],
                            "r":               r,
                            "r_aligned":       r_al,
                            "p":               p,
                            "ci_lo":           lo,
                            "ci_hi":           hi,
                            "n":               n,
                        })
    # Big5 cells
    for session in ("within", "between"):
        for induction in ("grid", "personas"):
            for task in TASK_ORDER:
                df = data_tables["big5"][session][induction][task]
                for trait_col, trait_label, expected_sign, outcome_col in BIG5_THEORY_PAIRS[task]:
                    for m in MODEL_ORDER:
                        sub = df[df.model_key == m]
                        if trait_col not in sub.columns or outcome_col not in sub.columns:
                            continue
                        r, p, n = safe_pearsonr(sub[trait_col], sub[outcome_col])
                        lo, hi = pearson_ci(r, n)
                        r_al = r * expected_sign if not np.isnan(r) else np.nan
                        rows.append({
                            "framework":       "Big5",
                            "session":         session,
                            "induction":       induction,
                            "task":            task,
                            "task_label":      TASK_LABELS[task],
                            "construct":       trait_col,
                            "construct_label": trait_label,
                            "expected_sign":   expected_sign,
                            "outcome_col":     outcome_col,
                            "model":           m,
                            "model_label":     MODEL_LABELS[m],
                            "r":               r,
                            "r_aligned":       r_al,
                            "p":               p,
                            "ci_lo":           lo,
                            "ci_hi":           hi,
                            "n":               n,
                        })
    return pd.DataFrame(rows)


# ── Aggregation ───────────────────────────────────────────────────────────

def aggregate_by_cell(cells: pd.DataFrame) -> pd.DataFrame:
    """Per (framework × session × induction × task × construct) Fisher-z mean r."""
    key_cols = ["framework", "session", "induction", "task", "task_label",
                "construct", "construct_label"]
    out = (cells.groupby(key_cols)
                .agg(n_cells=("r_aligned", "size"),
                     mean_r=("r_aligned", "mean"))
                .reset_index())
    fz_stats = []
    for _, row in out.iterrows():
        grp = cells[(cells["framework"] == row["framework"]) &
                    (cells["session"] == row["session"]) &
                    (cells["induction"] == row["induction"]) &
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


def _se_z_slice(cells_slice: pd.DataFrame) -> tuple[float, float]:
    """Return (z_mean, se_z) for a slice of cells; (nan, nan) if degenerate."""
    cs = cells_slice.dropna(subset=["r_aligned", "n"])
    cs = cs[cs["n"] > 3]
    if len(cs) == 0:
        return np.nan, np.nan
    weights = cs["n"].values - 3
    zs = np.arctanh(np.clip(cs["r_aligned"].values, -0.9999, 0.9999))
    z_mean = float(np.sum(weights * zs) / np.sum(weights))
    se_z = float(1.0 / np.sqrt(np.sum(weights)))
    return z_mean, se_z


def compute_delta_r_induction(by_cell: pd.DataFrame,
                               cells: pd.DataFrame) -> pd.DataFrame:
    """For each (framework × session × task × construct), compute
    Δr_induction = r_personas − r_grid with pooled z-scale CI + p."""
    rows = []
    keys = by_cell[["framework", "session", "task", "construct"]].drop_duplicates()
    for _, k in keys.iterrows():
        g = by_cell[(by_cell["framework"] == k["framework"]) &
                    (by_cell["session"] == k["session"]) &
                    (by_cell["task"] == k["task"]) &
                    (by_cell["construct"] == k["construct"]) &
                    (by_cell["induction"] == "grid")]
        p = by_cell[(by_cell["framework"] == k["framework"]) &
                    (by_cell["session"] == k["session"]) &
                    (by_cell["task"] == k["task"]) &
                    (by_cell["construct"] == k["construct"]) &
                    (by_cell["induction"] == "personas")]
        if g.empty or p.empty:
            continue
        g = g.iloc[0]; p = p.iloc[0]

        cells_g = cells[(cells["framework"] == k["framework"]) &
                        (cells["session"] == k["session"]) &
                        (cells["task"] == k["task"]) &
                        (cells["construct"] == k["construct"]) &
                        (cells["induction"] == "grid")]
        cells_p = cells[(cells["framework"] == k["framework"]) &
                        (cells["session"] == k["session"]) &
                        (cells["task"] == k["task"]) &
                        (cells["construct"] == k["construct"]) &
                        (cells["induction"] == "personas")]
        z_g, se_g = _se_z_slice(cells_g)
        z_p, se_p = _se_z_slice(cells_p)
        if np.isnan(z_g) or np.isnan(z_p):
            continue
        # Δ on z-scale: personas − grid (positive = personas helps)
        dz = z_p - z_g
        dz_se = float(np.sqrt(se_g**2 + se_p**2))
        crit = stats.norm.ppf(0.975)
        dz_lo = dz - crit * dz_se
        dz_hi = dz + crit * dz_se
        dz_z = dz / dz_se if dz_se > 0 else np.nan
        dz_p_val = (2 * (1 - stats.norm.cdf(abs(dz_z)))
                    if not np.isnan(dz_z) else np.nan)

        # Back-transform Δ to r-scale for reporting (endpoint approximation)
        dr = p["fz_mean_r"] - g["fz_mean_r"]
        dr_lo = float(np.tanh(z_p - crit*se_p) - np.tanh(z_g + crit*se_g))
        dr_hi = float(np.tanh(z_p + crit*se_p) - np.tanh(z_g - crit*se_g))

        rows.append({
            "framework":       k["framework"],
            "session":         k["session"],
            "task":            k["task"],
            "task_label":      g["task_label"],
            "construct":       k["construct"],
            "construct_label": g["construct_label"],
            "r_grid":          g["fz_mean_r"],
            "r_grid_lo":       g["fz_ci_lo"],
            "r_grid_hi":       g["fz_ci_hi"],
            "r_personas":      p["fz_mean_r"],
            "r_personas_lo":   p["fz_ci_lo"],
            "r_personas_hi":   p["fz_ci_hi"],
            "delta_r":         dr,
            "delta_r_lo":      dr_lo,
            "delta_r_hi":      dr_hi,
            "delta_z":         dz,
            "delta_z_se":      dz_se,
            "delta_z_lo":      dz_lo,
            "delta_z_hi":      dz_hi,
            "delta_z_p":       dz_p_val,
        })
    return pd.DataFrame(rows)


def best_construct_delta(delta_df: pd.DataFrame) -> pd.DataFrame:
    """For each (framework, session, task), pick the construct with the
    largest r_grid baseline and report Δr_induction on that construct.

    Note: we select on r_grid (not on r_personas or Δr) so the choice is
    consistent across induction conditions and doesn't cherry-pick rescue
    cases. For TPB this typically selects the theoretically-motivated
    primary construct; for Big5, the theory-pair trait with stronger
    baseline coupling.
    """
    rows = []
    for (framework, session, task), grp in delta_df.groupby(
            ["framework", "session", "task"]):
        best = grp.loc[grp["r_grid"].idxmax()]
        rows.append({
            "framework":       framework,
            "session":         session,
            "task":            task,
            "task_label":      best["task_label"],
            "best_construct":  best["construct_label"],
            "r_grid":          best["r_grid"],
            "r_grid_lo":       best["r_grid_lo"],
            "r_grid_hi":       best["r_grid_hi"],
            "r_personas":      best["r_personas"],
            "r_personas_lo":   best["r_personas_lo"],
            "r_personas_hi":   best["r_personas_hi"],
            "delta_r":         best["delta_r"],
            "delta_r_lo":      best["delta_r_lo"],
            "delta_r_hi":      best["delta_r_hi"],
            "delta_z_p":       best["delta_z_p"],
        })
    return pd.DataFrame(rows)


# ── Bootstrap ─────────────────────────────────────────────────────────────

def bootstrap_delta_r_induction(cells: pd.DataFrame,
                                 n_boot: int = 2000,
                                 seed: int = 42) -> pd.DataFrame:
    """Bootstrap 95% CIs on Δr_induction by resampling models with
    replacement. For each resampled model set, recompute Fisher-z means per
    induction, then Δr = r_personas − r_grid. Percentile CI over replicates.
    """
    rng = np.random.default_rng(seed)
    rows = []
    keys = cells[["framework", "session", "task", "construct",
                  "task_label", "construct_label"]].drop_duplicates()
    all_models = sorted(cells["model"].unique())
    n_models = len(all_models)
    model_idx = {m: i for i, m in enumerate(all_models)}

    for _, k in keys.iterrows():
        sub = cells[(cells["framework"] == k["framework"]) &
                    (cells["session"] == k["session"]) &
                    (cells["task"] == k["task"]) &
                    (cells["construct"] == k["construct"])]
        if sub.empty:
            continue
        r_g = np.full(n_models, np.nan); n_g = np.full(n_models, np.nan)
        r_p = np.full(n_models, np.nan); n_p = np.full(n_models, np.nan)
        for _, row in sub.iterrows():
            mi = model_idx[row["model"]]
            if row["induction"] == "grid":
                r_g[mi] = row["r_aligned"]; n_g[mi] = row["n"]
            else:
                r_p[mi] = row["r_aligned"]; n_p[mi] = row["n"]

        z_g = np.arctanh(np.clip(r_g, -0.9999, 0.9999))
        z_p = np.arctanh(np.clip(r_p, -0.9999, 0.9999))

        def fz_mean(zs, ns, indices):
            zs_s = zs[indices]; ns_s = ns[indices]
            mask = ~np.isnan(zs_s) & (ns_s > 3)
            if mask.sum() < 2:
                return np.nan
            w = ns_s[mask] - 3
            return float(np.tanh(np.sum(w * zs_s[mask]) / np.sum(w)))

        deltas = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n_models, size=n_models)
            mg = fz_mean(z_g, n_g, idx)
            mp = fz_mean(z_p, n_p, idx)
            deltas[i] = ((mp - mg) if not (np.isnan(mp) or np.isnan(mg))
                         else np.nan)
        deltas = deltas[~np.isnan(deltas)]
        if len(deltas) < 10:
            continue
        rows.append({
            "framework":           k["framework"],
            "session":             k["session"],
            "task":                k["task"],
            "task_label":          k["task_label"],
            "construct":           k["construct"],
            "construct_label":     k["construct_label"],
            "boot_delta_r_mean":   float(np.mean(deltas)),
            "boot_delta_r_ci_lo":  float(np.quantile(deltas, 0.025)),
            "boot_delta_r_ci_hi":  float(np.quantile(deltas, 0.975)),
            "n_boot_valid":        int(len(deltas)),
        })
    return pd.DataFrame(rows)


# ── Mundlak ───────────────────────────────────────────────────────────────

def compute_mundlak_all(data_tables: dict) -> pd.DataFrame:
    """Mundlak OLS for every (framework × session × induction × task ×
    construct) — 4 Mundlak fits per (task, construct) × framework.
    Big5 β's are sign-flipped to theory-consistent direction."""
    rows = []
    for session in ("within", "between"):
        for induction in ("grid", "personas"):
            # TPB
            for task in TASK_ORDER:
                df = data_tables["tpb"][session][induction][task]
                primary_col, primary_label = PRIMARY_CONSTRUCT[task]
                constructs = [(INTENTION_COL, "Intention")]
                if primary_col != INTENTION_COL:
                    constructs.append((primary_col, primary_label))
                for construct_col, construct_label in constructs:
                    m = mundlak_within_between(df, construct_col, "align_score",
                                                group_col="model_key",
                                                z_normalize=True)
                    rows.append({
                        "framework":       "TPB",
                        "session":         session,
                        "induction":       induction,
                        "task":            task,
                        "task_label":      TASK_LABELS[task],
                        "construct":       construct_col,
                        "construct_label": construct_label,
                        "expected_sign":   +1,
                        **m,
                    })
            # Big5
            for task in TASK_ORDER:
                df = data_tables["big5"][session][induction][task]
                for trait_col, trait_label, expected_sign, outcome_col in BIG5_THEORY_PAIRS[task]:
                    if trait_col not in df.columns or outcome_col not in df.columns:
                        continue
                    m = mundlak_within_between(df, trait_col, outcome_col,
                                                group_col="model_key",
                                                z_normalize=True)
                    for key in ("beta_within", "beta_between"):
                        if not np.isnan(m[key]):
                            m[key] = m[key] * expected_sign
                    if expected_sign < 0:
                        m["ci_within_lo"],  m["ci_within_hi"]  = (
                            -m["ci_within_hi"],  -m["ci_within_lo"])
                        m["ci_between_lo"], m["ci_between_hi"] = (
                            -m["ci_between_hi"], -m["ci_between_lo"])
                    rows.append({
                        "framework":       "Big5",
                        "session":         session,
                        "induction":       induction,
                        "task":            task,
                        "task_label":      TASK_LABELS[task],
                        "construct":       trait_col,
                        "construct_label": trait_label,
                        "expected_sign":   expected_sign,
                        "outcome_col":     outcome_col,
                        **m,
                    })
    return pd.DataFrame(rows)


# ── Formal pooled OLS with SR × induction interaction ────────────────────

def compute_ols_interaction(data_tables: dict,
                              delta_df: pd.DataFrame) -> pd.DataFrame:
    """For each (framework × session × task), fit a pooled OLS with an
    explicit SR × induction interaction term, using the best-baseline
    construct (picked per cell from delta_df by maximum r_grid).

    Model
    -----
        y_z = β_0 + β_SR * SR_z + β_P * I(personas)
              + β_{SR×P} * (SR_z · I(personas))
              + model fixed effects + ε,
    where SR_z and y_z are z-standardised within each
    (model × induction) cell so the coefficients are on a
    correlation-like scale. The interaction β_{SR×P} estimates how much
    the SR → behaviour slope changes when moving from parameter grid to
    persona induction; in expectation it equals Δr_induction on the
    Fisher-z scale but is obtained under the OLS framework that some
    reviewers prefer for interaction tests. Cluster-robust SEs are
    clustered by model_key.

    Parameters
    ----------
    data_tables : dict
        Same nested dict built in main().
    delta_df : pd.DataFrame
        Output of compute_delta_r_induction, used to select the
        best-baseline construct per (framework, session, task) via the
        same rule as best_construct_delta.

    Returns
    -------
    pd.DataFrame with one row per (framework × session × task × best
    construct), containing β_{SR×P}, its 95% CI, p-value, and the
    per-induction simple slopes β_SR (grid) and β_SR + β_{SR×P}
    (personas).
    """
    import statsmodels.formula.api as smf

    # Best construct per (framework, session, task) — max r_grid
    best_keys = (delta_df.sort_values(["framework", "session", "task",
                                         "r_grid"],
                                        ascending=[True, True, True, False])
                          .groupby(["framework", "session", "task"],
                                    as_index=False).first())

    rows = []
    for _, row in best_keys.iterrows():
        fw = row["framework"]; session = row["session"]
        task = row["task"];    construct_col = row["construct"]
        construct_label = row["construct_label"]
        fw_key = "tpb" if fw == "TPB" else "big5"

        # Determine outcome column and sign correction
        if fw == "TPB":
            outcome_col = "align_score"
            expected_sign = +1
        else:  # Big5: find matching theory pair for (task, construct_col)
            pair = next(((t, l, s, o) for t, l, s, o in
                          BIG5_THEORY_PAIRS[task]
                          if t == construct_col), None)
            if pair is None:
                continue
            _, _, expected_sign, outcome_col = pair

        # Pool grid + personas data for this cell
        df_g = data_tables[fw_key][session]["grid"][task].copy()
        df_p = data_tables[fw_key][session]["personas"][task].copy()
        df_g["induction"] = 0
        df_p["induction"] = 1
        pool = pd.concat([df_g, df_p], ignore_index=True)
        # Keep only necessary columns, drop NaNs
        need = [construct_col, outcome_col, "model_key", "induction"]
        pool = pool[need].copy()
        pool = pool.dropna(subset=[construct_col, outcome_col])
        if len(pool) < 30 or pool["model_key"].nunique() < 3:
            continue

        # Apply sign correction: theory-aligned outcome
        pool["y"]  = pd.to_numeric(pool[outcome_col], errors="coerce") \
                        * expected_sign
        pool["SR"] = pd.to_numeric(pool[construct_col], errors="coerce")
        pool = pool.dropna(subset=["y", "SR"])

        # Z-standardise within each (model × induction) cell so the
        # slope is on a within-cell standard-deviation scale. Use
        # transform() rather than apply() to preserve grouping columns.
        sr_mean = pool.groupby(["model_key", "induction"])["SR"].transform(
            "mean")
        sr_sd   = pool.groupby(["model_key", "induction"])["SR"].transform(
            "std").replace(0, np.nan)
        y_mean  = pool.groupby(["model_key", "induction"])["y"].transform(
            "mean")
        y_sd    = pool.groupby(["model_key", "induction"])["y"].transform(
            "std").replace(0, np.nan)
        pool["SR_z"] = (pool["SR"] - sr_mean) / sr_sd
        pool["y_z"]  = (pool["y"]  - y_mean)  / y_sd
        pool = pool.dropna(subset=["SR_z", "y_z"])
        if len(pool) < 30 or pool["model_key"].nunique() < 3:
            continue

        # Fit interaction model with model fixed effects
        try:
            res = smf.ols("y_z ~ SR_z * induction + C(model_key)",
                          data=pool).fit(
                cov_type="cluster",
                cov_kwds={"groups": pool["model_key"].values})
        except Exception as _e:
            import warnings
            warnings.warn(f"OLS fit failed for {fw}/{session}/{task}: {_e}")
            continue

        # Extract interaction coefficient and simple slopes
        inter_key = "SR_z:induction"
        if inter_key not in res.params.index:
            continue
        beta_inter = float(res.params[inter_key])
        se_inter   = float(res.bse[inter_key])
        ci = res.conf_int().loc[inter_key]
        ci_lo, ci_hi = float(ci[0]), float(ci[1])
        p_inter    = float(res.pvalues[inter_key])

        # Simple slopes: grid = β_SR; personas = β_SR + β_{SR:induction}
        beta_sr    = float(res.params.get("SR_z", np.nan))
        beta_grid     = beta_sr
        beta_personas = beta_sr + beta_inter

        rows.append({
            "framework":        fw,
            "session":          session,
            "task":             task,
            "task_label":       row["task_label"],
            "construct":        construct_col,
            "construct_label": construct_label,
            "beta_grid":        beta_grid,
            "beta_personas":    beta_personas,
            "beta_interaction": beta_inter,
            "se_interaction":   se_inter,
            "ci_interaction_lo": ci_lo,
            "ci_interaction_hi": ci_hi,
            "p_interaction":    p_inter,
            "n_obs":            int(res.nobs),
            "n_models":         int(pool["model_key"].nunique()),
        })
    _cols = ["framework", "session", "task", "task_label", "construct",
             "construct_label", "beta_grid", "beta_personas", "beta_interaction",
             "se_interaction", "ci_interaction_lo", "ci_interaction_hi",
             "p_interaction", "n_obs", "n_models"]
    return pd.DataFrame(rows, columns=_cols) if rows else pd.DataFrame(columns=_cols)

def aggregate_by_model_induction(cells: pd.DataFrame,
                                   framework: str = "TPB",
                                   session: str = "between") -> pd.DataFrame:
    """For a given (framework, session), per-model Fisher-z mean r_aligned
    per induction plus Δr_induction with 95% CI and p-value."""
    sub = cells[(cells["framework"] == framework) &
                (cells["session"] == session)]
    crit = stats.norm.ppf(0.975)
    wide_rows = []
    for m in MODEL_ORDER:
        cg = sub[(sub["model"] == m) & (sub["induction"] == "grid")]
        cp = sub[(sub["model"] == m) & (sub["induction"] == "personas")]
        if cg.empty or cp.empty:
            continue
        z_g, se_g = _se_z_slice(cg)
        z_p, se_p = _se_z_slice(cp)
        if np.isnan(z_g) or np.isnan(z_p):
            continue
        r_g = float(np.tanh(z_g));  r_p = float(np.tanh(z_p))
        r_g_lo = float(np.tanh(z_g - crit*se_g))
        r_g_hi = float(np.tanh(z_g + crit*se_g))
        r_p_lo = float(np.tanh(z_p - crit*se_p))
        r_p_hi = float(np.tanh(z_p + crit*se_p))
        dz = z_p - z_g
        dz_se = float(np.sqrt(se_g**2 + se_p**2))
        dz_p_val = (2 * (1 - stats.norm.cdf(abs(dz / dz_se)))
                    if dz_se > 0 else np.nan)
        dr = r_p - r_g
        dr_lo = float(np.tanh(z_p - crit*se_p) - np.tanh(z_g + crit*se_g))
        dr_hi = float(np.tanh(z_p + crit*se_p) - np.tanh(z_g - crit*se_g))
        wide_rows.append({
            "framework":     framework,
            "session":       session,
            "model":         m,
            "model_label":   MODEL_LABELS[m],
            "r_grid":        r_g,
            "r_grid_lo":     r_g_lo,
            "r_grid_hi":     r_g_hi,
            "r_personas":    r_p,
            "r_personas_lo": r_p_lo,
            "r_personas_hi": r_p_hi,
            "delta_r":       dr,
            "delta_r_lo":    dr_lo,
            "delta_r_hi":    dr_hi,
            "delta_z":       dz,
            "delta_z_se":    dz_se,
            "delta_z_p":     dz_p_val,
        })
    wide = pd.DataFrame(wide_rows)
    # Sort by r_grid ascending so weakest baseline (best rescue candidates)
    # are at the top of the plot
    wide = wide.sort_values("r_grid", ascending=False).reset_index(drop=True)
    return wide


def classify_rescue(row: pd.Series) -> str:
    """Classify per-model rescue status from between-session TPB r's.

    Categories:
      'rescued'       : r_grid CI ≤ 0 AND r_personas CI strictly > 0
      'both_retained' : both CIs strictly > 0 (dispositional — no rescue
                         needed)
      'both_collapse' : neither CI > 0 (personas didn't help)
      'personas_hurt' : r_grid CI > 0 but r_personas CI ≤ 0
      'other'         : anything ambiguous (CI touches zero but doesn't
                         strictly exclude it, negative baselines, etc.)
    """
    g_pos = row["r_grid_lo"] > 0
    p_pos = row["r_personas_lo"] > 0
    if not g_pos and p_pos:
        return "rescued"
    if g_pos and p_pos:
        return "both_retained"
    if g_pos and not p_pos:
        return "personas_hurt"
    if not g_pos and not p_pos:
        return "both_collapse"
    return "other"


# ── Printout ──────────────────────────────────────────────────────────────

def print_headline(delta_df: pd.DataFrame, best_df: pd.DataFrame,
                   per_model_wide_tpb: pd.DataFrame,
                   boot: pd.DataFrame) -> dict:
    print("\n" + "=" * 78)
    print("RQ4 HEADLINE — Induction Comparison (grid vs personas)")
    print("Does persona induction rescue between-session coherence collapse?")
    print("=" * 78)

    for session in ("between", "within"):
        print("\n" + "-" * 78)
        print(f"PER-TASK × FRAMEWORK Δr_induction, session = '{session}'")
        print("(Δ > 0 ⇒ personas HELPS relative to grid)")
        print("-" * 78)
        print(f"{'Task':<12}  {'Framework':<8}  {'Best Construct':<18}  "
              f"{'r_grid':<10}  {'r_personas':<12}  {'Δr [95% CI]':<24}")
        sub = best_df[best_df["session"] == session].sort_values(
            ["framework", "task"])
        for _, row in sub.iterrows():
            dr_str = (f"{row['delta_r']:+.2f} [{row['delta_r_lo']:+.2f},"
                      f"{row['delta_r_hi']:+.2f}]{stars(row['delta_z_p'])}")
            print(f"  {row['task_label']:<10}  {row['framework']:<8}  "
                  f"{row['best_construct']:<18}  "
                  f"{row['r_grid']:+.2f}      {row['r_personas']:+.2f}"
                  f"        {dr_str}")

    print("\n" + "-" * 78)
    print("PER-MODEL Δr_induction (TPB, between-session, pooled across "
          "task × construct)")
    print("(this is the RESCUE table — does persona induction recover "
          "coherence lost across sessions?)")
    print("-" * 78)
    print(f"{'Model':<22}  {'r_grid [CI]':<22}  "
          f"{'r_personas [CI]':<22}  {'Δr':<8}  {'Status':<16}")
    for _, row in per_model_wide_tpb.iterrows():
        rg = f"{row['r_grid']:+.2f} [{row['r_grid_lo']:+.2f},{row['r_grid_hi']:+.2f}]"
        rp = f"{row['r_personas']:+.2f} [{row['r_personas_lo']:+.2f},{row['r_personas_hi']:+.2f}]"
        status = classify_rescue(row)
        print(f"  {row['model_label']:<22}  {rg:<22}  {rp:<22}  "
              f"{row['delta_r']:+.2f}   {status}")

    print("\n" + "-" * 78)
    print("BOOTSTRAP Δr_induction 95% CIs (2000 iterations, resampling models)")
    print("-" * 78)
    print(f"{'Session':<10}  {'Task':<12}  {'Framework':<8}  "
          f"{'Construct':<20}  {'Δr (Fisher-z)':<16}  {'Δr (bootstrap)':<22}")
    for session in ("between", "within"):
        for _, row in best_df[best_df["session"] == session].sort_values(
                ["framework", "task"]).iterrows():
            boot_row = boot[(boot["session"] == session) &
                             (boot["framework"] == row["framework"]) &
                             (boot["task"] == row["task"]) &
                             (boot["construct_label"] == row["best_construct"])]
            if boot_row.empty: continue
            br = boot_row.iloc[0]
            fz_str = (f"{row['delta_r']:+.2f} "
                      f"[{row['delta_r_lo']:+.2f},{row['delta_r_hi']:+.2f}]")
            bt_str = (f"{br['boot_delta_r_mean']:+.2f} "
                      f"[{br['boot_delta_r_ci_lo']:+.2f},"
                      f"{br['boot_delta_r_ci_hi']:+.2f}]")
            print(f"  {session:<8}  {row['task_label']:<10}  "
                  f"{row['framework']:<8}  {row['best_construct']:<20}  "
                  f"{fz_str:<16}  {bt_str:<22}")

    # Headline JSON
    overall = {}
    for framework in ("TPB", "Big5"):
        for session in ("within", "between"):
            rows = best_df[(best_df["framework"] == framework) &
                            (best_df["session"] == session)]
            overall[f"{framework}_{session}"] = {
                "mean_r_grid":     round(float(rows["r_grid"].mean()), 4),
                "mean_r_personas": round(float(rows["r_personas"].mean()), 4),
                "mean_delta":      round(float(rows["delta_r"].mean()), 4),
            }
    # Per-model rescue tallies
    per_model_wide_tpb = per_model_wide_tpb.copy()
    per_model_wide_tpb["status"] = per_model_wide_tpb.apply(
        classify_rescue, axis=1)
    status_counts = per_model_wide_tpb["status"].value_counts().to_dict()

    return {
        "overall_per_framework_session": overall,
        "per_task_best": [
            {
                "session":    r["session"],
                "framework":  r["framework"],
                "task":       r["task_label"],
                "construct":  r["best_construct"],
                "r_grid":     round(float(r["r_grid"]), 3),
                "r_personas": round(float(r["r_personas"]), 3),
                "delta_r":    round(float(r["delta_r"]), 3),
                "delta_r_ci": [round(float(r["delta_r_lo"]), 3),
                                round(float(r["delta_r_hi"]), 3)],
                "delta_p":    (round(float(r["delta_z_p"]), 4)
                                if pd.notna(r["delta_z_p"]) else None),
            }
            for _, r in best_df.sort_values(
                ["session", "framework", "task"]).iterrows()
        ],
        "per_model_rescue_tpb_between": {
            "counts": status_counts,
            "models": [
                {
                    "model":       r["model_label"],
                    "r_grid":      round(float(r["r_grid"]), 3),
                    "r_personas":  round(float(r["r_personas"]), 3),
                    "delta_r":     round(float(r["delta_r"]), 3),
                    "status":      r["status"],
                }
                for _, r in per_model_wide_tpb.iterrows()
            ],
        },
    }


# ── Figure: main (separate-sessions rescue, 3 panels) ─────────────────────


def figure_rq4_main(by_cell: pd.DataFrame,
                     cells: pd.DataFrame,
                     per_model_wide: pd.DataFrame,
                     fixed_model_order: list,
                     stability_df: pd.DataFrame,
                     discriminability_df: pd.DataFrame,
                     out_dir: Path):
    """RQ4 main figure (3-panel coarse-to-fine, matching RQ3 conventions).

    Panel A: per-task TPB Intention + primary, separate-sessions only,
             A1 = grid (teal), A2 = personas (purple).
    Panel B: per-model TPB separate-sessions, two paired bars
             (grid teal, personas purple), models in separate-sessions
             grid r descending. Yellow band on rescued models (CI strict
             positive under personas AND CI ≤ 0 under grid).
    Panel C: per-(model × task) Δ heatmaps comparing the two inductions
             on the SR side. C1 = SR-diversity ΔSD (personas SD minus
             grid SD); C2 = SR-stability Δr (personas r minus grid r).
             Both centred at 0; blue = personas more (diverse / stable).
    """
    import matplotlib.patheffects as path_effects
    from matplotlib.patches import Patch
    from matplotlib.colors import TwoSlopeNorm

    # Color scheme
    GRID_COLOR     = C["teal"]   # teal (matches RQ3 separate-sessions)
    GRID_SECOND    = C["teal_light"]   # lighter teal (non-primary construct)
    PERSONA_COLOR  = C["plum"]   # purple (new for RQ4)
    PERSONA_SECOND = C["plum_light"]   # lighter purple (non-primary construct)

    fig = plt.figure(figsize=(24, 9.0))
    gs_outer = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.4],
                                wspace=0.40, left=0.05, right=0.95,
                                top=0.88, bottom=0.10)
    gs_a = gs_outer[0, 0].subgridspec(2, 1, height_ratios=[1.0, 1.0],
                                       hspace=0.30)
    ax_a_g = fig.add_subplot(gs_a[0, 0])  # A1 grid
    ax_a_p = fig.add_subplot(gs_a[1, 0])  # A2 personas
    ax_b   = fig.add_subplot(gs_outer[0, 1])

    gs_c = gs_outer[0, 2].subgridspec(1, 2, width_ratios=[1.0, 1.0],
                                       wspace=0.55)
    ax_c_g = fig.add_subplot(gs_c[0, 0])  # C1 grid stability
    ax_c_p = fig.add_subplot(gs_c[0, 1])  # C2 personas stability

    # ── Task ordering: same as RQ1/RQ2/RQ3 (Honesty / Sycophancy / CCT / IAT)
    # Use TPB-grid separate-sessions r descending to match the inherited order
    by_grid_b = by_cell[(by_cell["framework"] == "TPB") &
                         (by_cell["session"] == "between") &
                         (by_cell["induction"] == "grid")]
    # Task ordering: within-session grid TPB best descending — matches RQ1/2/3
    by_grid_w = by_cell[(by_cell["framework"] == "TPB") &
                         (by_cell["session"] == "within") &
                         (by_cell["induction"] == "grid")]
    task_max = by_grid_w.groupby("task")["fz_mean_r"].max().to_dict()
    tasks_sorted = sorted(TASK_ORDER, key=lambda t: task_max.get(t, 0),
                          reverse=True)

    # ── Panel A subpanels ──
    bar_h_a = BAR["height"]

    def _plot_a(ax, induction_label, title_letter, nice_label,
                primary_color, secondary_color, primary_label_color):
        cur_y = 0
        y_pos_a, y_labels = [], []
        for task in tasks_sorted:
            rows = by_cell[(by_cell["framework"] == "TPB") &
                            (by_cell["session"] == "between") &
                            (by_cell["induction"] == induction_label) &
                            (by_cell["task"] == task)]
            if rows.empty:
                continue
            primary_col, _ = PRIMARY_CONSTRUCT[task]
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
                            va="center", fontsize=FS["construct_tag"],
                            color=primary_label_color,
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
        ax.set_title(f"{title_letter}. Per-task TPB — {nice_label}",
                     fontsize=FS["panel_title"], fontweight="bold", loc="left", pad=8,
                     x=-0.18)
        style_ax(ax, grid_axis="x")

    _plot_a(ax_a_g, "grid",     "A1", "grid",
            GRID_COLOR,    GRID_SECOND,    C["teal_label"])
    _plot_a(ax_a_p, "personas", "A2", "personas",
            PERSONA_COLOR, PERSONA_SECOND, C["plum_label"])

    # ── Panel B: per-model separate-sessions, grid vs personas ──
    # Models in same order as RQ3 Panel B (same-session grid r descending),
    # so readers can scan the same model row across both figures.
    pm = per_model_wide.set_index("model").reindex(fixed_model_order).reset_index()
    pm = pm.dropna(subset=["r_grid"]).reset_index(drop=True)
    n_models = len(pm)

    bar_h_b = 0.36
    y_pos = np.arange(n_models)[::-1]

    # Identify rescued models: grid CI ≤ 0 AND personas CI strictly > 0
    rescued = set()
    for _, row in pm.iterrows():
        grid_collapsed = row["r_grid_lo"] <= 0
        personas_recovered = row["r_personas_lo"] > 0
        if grid_collapsed and personas_recovered:
            rescued.add(row["model"])

    # Highlight rescued model rows (yellow band like RQ3)
    for i, row in pm.iterrows():
        if row["model"] in rescued:
            ax_b.axhspan(y_pos[i] - 0.5, y_pos[i] + 0.5,
                         facecolor=C["highlight_bg"], alpha=0.55, zorder=0)

    for i, row in pm.iterrows():
        # Grid bar (top of pair)
        yp_g = y_pos[i] + bar_h_b/2 + 0.04
        ax_b.barh(yp_g, row["r_grid"], height=bar_h_b,
                  color=GRID_COLOR, alpha=BAR["alpha"], edgecolor=C["bar_edge"], linewidth=BAR["edge_lw"], zorder=3)
        ax_b.errorbar(row["r_grid"], yp_g,
                      xerr=[[max(0, row["r_grid"] - row["r_grid_lo"])],
                            [max(0, row["r_grid_hi"] - row["r_grid"])]],
                      fmt="none", ecolor=BAR["ecolor"], capsize=BAR["capsize"], lw=BAR["elinewidth"], zorder=4)
        # Personas bar (bottom of pair)
        yp_p = y_pos[i] - bar_h_b/2 - 0.04
        ax_b.barh(yp_p, row["r_personas"], height=bar_h_b,
                  color=PERSONA_COLOR, alpha=BAR["alpha"], edgecolor=C["bar_edge"], linewidth=BAR["edge_lw"], zorder=3)
        ax_b.errorbar(row["r_personas"], yp_p,
                      xerr=[[max(0, row["r_personas"] - row["r_personas_lo"])],
                            [max(0, row["r_personas_hi"] - row["r_personas"])]],
                      fmt="none", ecolor=BAR["ecolor"], capsize=BAR["capsize"], lw=BAR["elinewidth"], zorder=4)

    add_zero_line(ax_b, "v")
    ax_b.set_yticks(y_pos)
    yticklabels = []
    for _, row in pm.iterrows():
        if row["model"] in rescued:
            yticklabels.append(f"{row['model_label']} ✓")
        else:
            yticklabels.append(row["model_label"])
    ax_b.set_yticklabels(yticklabels, fontsize=FS["tick"])
    for tick_label, (_, row) in zip(ax_b.get_yticklabels(), pm.iterrows()):
        if row["model"] in rescued:
            tick_label.set_color("#1B5E20")
            tick_label.set_fontweight("bold")
    ax_b.set_xlabel("Fisher-z mean r_aligned (95% CI)", fontsize=FS["axis_label"])
    ax_b.tick_params(axis="x", labelsize=FS["tick"])
    x_ext = max(0.85, max(pm[["r_grid_lo", "r_grid_hi",
                               "r_personas_lo", "r_personas_hi"]].abs().max()) + 0.1)
    ax_b.set_xlim(-x_ext, x_ext)
    ax_b.set_ylim(-0.5, n_models - 0.5)
    ax_b.set_title("B. Per-model TPB: grid vs personas",
               fontsize=FS["panel_title"], fontweight="bold", loc="left", pad=8,
               x=-0.20)
    style_ax(ax_b, grid_axis="x")
    ax_b.legend(handles=[
        Patch(facecolor=GRID_COLOR,    alpha=BAR["alpha"], label="grid"),
        Patch(facecolor=PERSONA_COLOR, alpha=BAR["alpha"], label="personas"),
    ], loc="lower right", fontsize=FS["legend"], frameon=True, framealpha=0.9)

    # ── Panel C: SR stability heatmaps (model × task; grid | personas) ──
    TASK_ABBREV = {"cct": "CCT", "syc": "Syc.", "honesty": "Hon.", "iat": "IAT"}
    PRIMARY_SR_CONSTRUCT = {
        "cct":     "intention_mean",
        "syc":     "intention_mean",
        "honesty": "attitude_mean",
        "iat":     "intention_mean",
    }

    # Same model order as Panel B
    model_order = pm["model"].tolist()
    n_m = len(model_order)
    n_t = len(tasks_sorted)

    # ── Build C1 data: SR-discriminability Δ_SD (personas − grid) ──
    disc_b = discriminability_df[
        (discriminability_df["framework"] == "TPB") &
        (discriminability_df["session"] == "between")
    ]
    disc_delta = np.full((n_m, n_t), np.nan)
    for j, t in enumerate(tasks_sorted):
        primary = PRIMARY_SR_CONSTRUCT[t]
        for i, m in enumerate(model_order):
            g = disc_b[(disc_b["task"] == t) &
                       (disc_b["construct"] == primary) &
                       (disc_b["model"] == m) &
                       (disc_b["induction"] == "grid")]
            p = disc_b[(disc_b["task"] == t) &
                       (disc_b["construct"] == primary) &
                       (disc_b["model"] == m) &
                       (disc_b["induction"] == "personas")]
            if len(g) and len(p):
                sd_g = float(g["sd"].iloc[0])
                sd_p = float(p["sd"].iloc[0])
                if not np.isnan(sd_g) and not np.isnan(sd_p):
                    disc_delta[i, j] = sd_p - sd_g

    # ── Build C2 data: SR-stability Δr (personas − grid) ──
    stab_delta = np.full((n_m, n_t), np.nan)
    for j, t in enumerate(tasks_sorted):
        primary = PRIMARY_SR_CONSTRUCT[t]
        for i, m in enumerate(model_order):
            g = stability_df[(stability_df["framework"] == "TPB") &
                              (stability_df["task"] == t) &
                              (stability_df["construct"] == primary) &
                              (stability_df["model"] == m) &
                              (stability_df["induction"] == "grid")]
            p = stability_df[(stability_df["framework"] == "TPB") &
                              (stability_df["task"] == t) &
                              (stability_df["construct"] == primary) &
                              (stability_df["model"] == m) &
                              (stability_df["induction"] == "personas")]
            if len(g) and len(p):
                rg = float(g["r"].iloc[0])
                rp = float(p["r"].iloc[0])
                if not np.isnan(rg) and not np.isnan(rp):
                    stab_delta[i, j] = rp - rg

    # ── Plot C1: discriminability Δ (personas − grid SR-SD) ──
    # Use vmax of 0.7 to capture the empirical range; positive (blue) = personas more discriminable
    disc_delta_max = 0.7
    norm_disc = TwoSlopeNorm(vmin=-disc_delta_max, vcenter=0, vmax=disc_delta_max)
    im_c1 = ax_c_g.imshow(disc_delta, aspect="auto", cmap=HEAT["cmap"],
                           norm=norm_disc)
    for i in range(n_m):
        for j in range(n_t):
            v = disc_delta[i, j]
            if np.isnan(v): continue
            txt_color = "white" if abs(v) > 0.35 else "black"
            ax_c_g.text(j, i, f"{v:+.2f}",
                        ha="center", va="center", fontsize=FS["heatmap_cell"],
                        color=txt_color, fontweight="bold")
    ax_c_g.set_xticks(range(n_t))
    ax_c_g.set_xticklabels([TASK_ABBREV[t] for t in tasks_sorted],
                            fontsize=FS["tick"], fontweight="bold")
    ax_c_g.set_yticks(range(n_m))
    ax_c_g.set_yticklabels([MODEL_LABELS[m] for m in model_order], fontsize=FS["tick"])
    ax_c_g.set_title("C1. SR diversity (ΔSD)",
                  fontsize=FS["panel_title"], fontweight="bold", loc="left", pad=8,
                  x=-0.30)
    style_heatmap_ax(ax_c_g)

    # ── Plot C2: stability Δr (blue = personas more stable) ──
    delta_max = 1.0
    norm_delta = TwoSlopeNorm(vmin=-delta_max, vcenter=0, vmax=delta_max)
    im_c2 = ax_c_p.imshow(stab_delta, aspect="auto", cmap=HEAT["cmap"],
                           norm=norm_delta)
    for i in range(n_m):
        for j in range(n_t):
            v = stab_delta[i, j]
            if np.isnan(v): continue
            txt_color = "white" if abs(v) > 0.45 else "black"
            ax_c_p.text(j, i, f"{v:+.2f}",
                        ha="center", va="center", fontsize=FS["heatmap_cell"],
                        color=txt_color, fontweight="bold")
    ax_c_p.set_xticks(range(n_t))
    ax_c_p.set_xticklabels([TASK_ABBREV[t] for t in tasks_sorted],
                            fontsize=FS["tick"], fontweight="bold")
    ax_c_p.set_yticks(range(n_m))
    ax_c_p.set_yticklabels([])
    ax_c_p.set_title("C2. SR stability (Δr)",
                  fontsize=FS["panel_title"], fontweight="bold", loc="left", pad=8)
    style_heatmap_ax(ax_c_p)

    # Colorbars: dedicated axes outside gridspec so they don't shrink heatmaps
    pos_c1 = ax_c_g.get_position()
    cbar_c1_ax = fig.add_axes([pos_c1.x1 + 0.005, pos_c1.y0, 0.010, pos_c1.height])
    cbar_c1 = fig.colorbar(im_c1, cax=cbar_c1_ax)
    cbar_c1.ax.tick_params(labelsize=FS["colorbar"])
    cbar_c1.set_label("personas SD − grid SD\n(>0: personas more diverse)",
                       fontsize=FS["colorbar"])

    pos_c2 = ax_c_p.get_position()
    cbar_c2_ax = fig.add_axes([pos_c2.x1 + 0.005, pos_c2.y0, 0.010, pos_c2.height])
    cbar_c2 = fig.colorbar(im_c2, cax=cbar_c2_ax)
    cbar_c2.ax.tick_params(labelsize=FS["colorbar"])
    cbar_c2.set_label("personas r − grid r\n(>0: personas more stable)",
                       fontsize=FS["colorbar"])

    suptitle = ("RQ4 — Identity Induction: does persona grounding rescue "
                "separate-sessions coherence?")
    fig.suptitle(suptitle, fontsize=FS["suptitle"], fontweight="bold", y=0.97)

    for ext in ("pdf", "png"):
        p = out_dir / f"rq4_induction_summary.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  saved: {p}")
    plt.close(fig)


# ── Supplementary figure: SR discriminability heatmap ─────────────────────


def figure_rq4_supp_discriminability(disc_df: pd.DataFrame,
                                       per_model_wide: pd.DataFrame,
                                       fixed_model_order: list,
                                       out_dir: Path):
    """Supplementary figure: per-(model × task) absolute SR diversity
    (SD across conditions) under grid vs personas, for the TPB primary
    construct per task.

    Two panels (the Δ panel is in the main figure, Panel C1):
      A. SD heatmap — grid (sequential viridis colormap)
      B. SD heatmap — personas (same scale as A)
    """
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0],
                          wspace=0.10, left=0.07, right=0.92,
                          top=0.88, bottom=0.10)
    ax_g = fig.add_subplot(gs[0, 0])
    ax_p = fig.add_subplot(gs[0, 1])

    # Task ordering: same as main figure
    tasks_sorted = ["honesty", "syc", "cct", "iat"]
    TASK_ABBREV = {"cct": "CCT", "syc": "Syc.", "honesty": "Hon.", "iat": "IAT"}
    PRIMARY_SR_CONSTRUCT = {
        "cct":     "intention_mean",
        "syc":     "intention_mean",
        "honesty": "attitude_mean",
        "iat":     "intention_mean",
    }

    # Model order matches RQ3 Panel B + main figure (same-session grid r descending)
    model_order = fixed_model_order

    # Filter to TPB, between-session, primary construct only
    sub = disc_df[(disc_df["framework"] == "TPB") &
                   (disc_df["session"] == "between")].copy()

    n_m, n_t = len(model_order), len(tasks_sorted)
    sd_grid = np.full((n_m, n_t), np.nan)
    sd_pers = np.full((n_m, n_t), np.nan)
    for j, t in enumerate(tasks_sorted):
        primary = PRIMARY_SR_CONSTRUCT[t]
        for i, m in enumerate(model_order):
            for ind, mat in [("grid", sd_grid), ("personas", sd_pers)]:
                row = sub[(sub["task"] == t) &
                          (sub["construct"] == primary) &
                          (sub["model"] == m) &
                          (sub["induction"] == ind)]
                if len(row):
                    mat[i, j] = float(row["sd"].iloc[0])

    # Common SD scale across grid and personas heatmaps so they're comparable
    sd_max = float(np.nanmax(np.concatenate([sd_grid.ravel(), sd_pers.ravel()])))
    sd_max = max(sd_max, 0.5)

    def _plot_seq_heatmap(ax, heat, title, vmin, vmax, title_x=0.0):
        im = ax.imshow(heat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        for i in range(n_m):
            for j in range(n_t):
                v = heat[i, j]
                if np.isnan(v): continue
                # Threshold for white text on darker viridis cells
                txt_color = "white" if v < 0.55 * vmax else "black"
                ax.text(j, i, f"{v:.2f}",
                        ha="center", va="center", fontsize=FS["heatmap_cell"],
                        color=txt_color, fontweight="bold")
        ax.set_xticks(range(n_t))
        ax.set_xticklabels([TASK_ABBREV[t] for t in tasks_sorted],
                           fontsize=FS["tick"], fontweight="bold")
        ax.set_yticks(range(n_m))
        ax.set_yticklabels([MODEL_LABELS[m] for m in model_order], fontsize=FS["tick"])
        ax.set_title(title, fontsize=FS["panel_title"], fontweight="bold", loc="left", pad=8,
                     x=title_x)
        style_heatmap_ax(ax)
        return im

    im_g = _plot_seq_heatmap(ax_g, sd_grid, "A. SR-SD — grid",
                              vmin=0, vmax=sd_max, title_x=-0.35)
    im_p = _plot_seq_heatmap(ax_p, sd_pers, "B. SR-SD — personas",
                              vmin=0, vmax=sd_max)
    ax_p.set_yticklabels([])

    pos = ax_p.get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.008, pos.y0, 0.012, pos.height])
    cbar = fig.colorbar(im_p, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=FS["colorbar"])
    cbar.set_label("SR-SD across conditions", fontsize=FS["colorbar"])

    fig.suptitle("RQ4 supplementary: absolute SR diversity (SD across "
                  "conditions) under grid vs personas",
                  fontsize=FS["suptitle"], fontweight="bold", y=0.98)

    for ext in ("pdf", "png"):
        p = out_dir / f"rq4_supp_discriminability.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  saved: {p}")
    plt.close(fig)


# ── Supplementary figure: absolute SR stability heatmap ───────────────────


def figure_rq4_supp_sr_stability(stability_df: pd.DataFrame,
                                    per_model_wide: pd.DataFrame,
                                    fixed_model_order: list,
                                    out_dir: Path):
    """Supplementary figure: per-(model × task) absolute SR stability
    (cross-session r) under grid vs personas. Mirrors the main figure
    Panel C2 (Δr) but shows the absolute r values that the Δr is computed
    from. Useful to ground the Δr in concrete starting and ending values.
    """
    from matplotlib.colors import TwoSlopeNorm

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0],
                          wspace=0.10, left=0.07, right=0.92,
                          top=0.88, bottom=0.10)
    ax_g = fig.add_subplot(gs[0, 0])
    ax_p = fig.add_subplot(gs[0, 1])

    tasks_sorted = ["honesty", "syc", "cct", "iat"]
    TASK_ABBREV = {"cct": "CCT", "syc": "Syc.", "honesty": "Hon.", "iat": "IAT"}
    PRIMARY_SR_CONSTRUCT = {
        "cct":     "intention_mean",
        "syc":     "intention_mean",
        "honesty": "attitude_mean",
        "iat":     "intention_mean",
    }

    pm = per_model_wide.copy()
    model_order = fixed_model_order

    n_m, n_t = len(model_order), len(tasks_sorted)

    def _build(induction):
        heat = np.full((n_m, n_t), np.nan)
        sigmask = np.zeros((n_m, n_t), dtype=bool)
        for j, t in enumerate(tasks_sorted):
            primary = PRIMARY_SR_CONSTRUCT[t]
            for i, m in enumerate(model_order):
                row = stability_df[
                    (stability_df["framework"] == "TPB") &
                    (stability_df["induction"] == induction) &
                    (stability_df["task"] == t) &
                    (stability_df["construct"] == primary) &
                    (stability_df["model"] == m)
                ]
                if len(row):
                    v = row["r"].iloc[0]
                    if not np.isnan(v): heat[i, j] = v
                    p = row["p"].iloc[0]
                    if not np.isnan(p): sigmask[i, j] = (p < 0.05)
        return heat, sigmask

    def _plot(ax, induction, title, title_x=0.0):
        heat, sigmask = _build(induction)
        norm = TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0)
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
        ax.set_xticklabels([TASK_ABBREV[t] for t in tasks_sorted],
                           fontsize=FS["tick"], fontweight="bold")
        ax.set_yticks(range(n_m))
        ax.set_yticklabels([MODEL_LABELS[m] for m in model_order], fontsize=FS["tick"])
        ax.set_title(title, fontsize=FS["panel_title"], fontweight="bold", loc="left", pad=8,
                     x=title_x)
        style_heatmap_ax(ax)
        return im

    im_g = _plot(ax_g, "grid",     "A. SR stability — grid",     title_x=-0.35)
    im_p = _plot(ax_p, "personas", "B. SR stability — personas")
    ax_p.set_yticklabels([])

    pos = ax_p.get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.008, pos.y0, 0.012, pos.height])
    cbar = fig.colorbar(im_p, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=FS["colorbar"])
    cbar.set_label("Cross-session SR r  (* = p < .05)", fontsize=FS["colorbar"])

    fig.suptitle("RQ4 supplementary: absolute SR stability (cross-session r) "
                  "under grid vs personas",
                  fontsize=FS["suptitle"], fontweight="bold", y=0.98)

    for ext in ("pdf", "png"):
        p = out_dir / f"rq4_supp_sr_stability.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  saved: {p}")
    plt.close(fig)


# ── Supplementary figure: behaviour stability heatmap ─────────────────────


def figure_rq4_supp_behavior_stability(beh_stab_df: pd.DataFrame,
                                          per_model_wide: pd.DataFrame,
                                          fixed_model_order: list,
                                          out_dir: Path):
    """Supplementary figure: per-(model × task) Behaviour stability under
    grid vs personas. Mirrors main figure Panel C (SR stability) but for
    behaviour. Useful for diagnosing whether RQ4's coherence non-rescue is
    driven by SR-side or behaviour-side context-sensitivity.
    """
    from matplotlib.colors import TwoSlopeNorm

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0],
                          wspace=0.10, left=0.07, right=0.92,
                          top=0.88, bottom=0.10)
    ax_g = fig.add_subplot(gs[0, 0])
    ax_p = fig.add_subplot(gs[0, 1])

    tasks_sorted = ["honesty", "syc", "cct", "iat"]
    TASK_ABBREV = {"cct": "CCT", "syc": "Syc.", "honesty": "Hon.", "iat": "IAT"}

    pm = per_model_wide.copy()
    model_order = fixed_model_order

    n_m, n_t = len(model_order), len(tasks_sorted)

    def _build_heatmap(induction):
        heat = np.full((n_m, n_t), np.nan)
        sigmask = np.zeros((n_m, n_t), dtype=bool)
        for j, t in enumerate(tasks_sorted):
            for i, m in enumerate(model_order):
                row = beh_stab_df[(beh_stab_df["induction"] == induction) &
                                   (beh_stab_df["task"] == t) &
                                   (beh_stab_df["model"] == m)]
                if len(row):
                    v = row["r"].iloc[0]
                    if not np.isnan(v): heat[i, j] = v
                    p = row["p"].iloc[0]
                    if not np.isnan(p): sigmask[i, j] = (p < 0.05)
        return heat, sigmask

    def _plot(ax, induction, title, title_x=0.0):
        heat, sigmask = _build_heatmap(induction)
        norm = TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0)
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
        ax.set_xticklabels([TASK_ABBREV[t] for t in tasks_sorted],
                           fontsize=FS["tick"], fontweight="bold")
        ax.set_yticks(range(n_m))
        ax.set_yticklabels([MODEL_LABELS[m] for m in model_order], fontsize=FS["tick"])
        ax.set_title(title, fontsize=FS["panel_title"], fontweight="bold", loc="left", pad=8,
                     x=title_x)
        style_heatmap_ax(ax)
        return im

    im_g = _plot(ax_g, "grid",     "A. Behaviour stability — grid", title_x=-0.35)
    im_p = _plot(ax_p, "personas", "B. Behaviour stability — personas")
    ax_p.set_yticklabels([])

    pos = ax_p.get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.008, pos.y0, 0.012, pos.height])
    cbar = fig.colorbar(im_p, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Cross-session behaviour r  (* = p < .05)", fontsize=FS["colorbar"])

    fig.suptitle("RQ4 supplementary: Behaviour stability under grid vs personas "
                  "(separate vs same-session, align_score)",
                  fontsize=FS["suptitle"], fontweight="bold", y=0.98)

    for ext in ("pdf", "png"):
        p = out_dir / f"rq4_supp_behavior_stability.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  saved: {p}")
    plt.close(fig)


# ── Figure: appendix (full 2×2 interaction, TPB primary only) ─────────────

def figure_rq4_appendix(best_df: pd.DataFrame, out_dir: Path):
    """Appendix figure: full session × induction interaction for TPB primary
    construct. Per task, 4 paired bars:
      grid-within, grid-between, personas-within, personas-between.
    Unlike the main figure which focuses on between-session, the appendix
    reveals whether any induction/session interaction is ordinal or
    crossover.
    """
    tpb_best = best_df[best_df["framework"] == "TPB"]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(1, 1, left=0.32, right=0.92, top=0.90, bottom=0.10)
    ax = fig.add_subplot(gs[0, 0])

    tasks = TASK_ORDER
    y_centers = np.arange(len(tasks)) * 2.2
    y_centers = y_centers[::-1]
    bar_h = 0.36

    for ti, task in enumerate(tasks):
        y_base = y_centers[ti]
        # 4 bars, ordered top-to-bottom:
        #   grid-within, personas-within, grid-between, personas-between
        entries = []
        for session in ("within", "between"):
            row = tpb_best[(tpb_best["task"] == task) &
                            (tpb_best["session"] == session)]
            if row.empty: continue
            row = row.iloc[0]
            entries.append({
                "session":   session,
                "induction": "grid",
                "r":         row["r_grid"],
                "r_lo":      row["r_grid_lo"],
                "r_hi":      row["r_grid_hi"],
                "construct": row["best_construct"],
            })
            entries.append({
                "session":   session,
                "induction": "personas",
                "r":         row["r_personas"],
                "r_lo":      row["r_personas_lo"],
                "r_hi":      row["r_personas_hi"],
                "construct": row["best_construct"],
            })
        n = len(entries)
        offsets = (np.arange(n) - (n - 1) / 2) * bar_h * 1.1
        for k, e in enumerate(entries):
            yp = y_base + offsets[::-1][k]
            is_within = e["session"] == "within"
            is_grid = e["induction"] == "grid"
            # Color encodes session (orange=within, teal=between);
            # shade encodes induction (dark=grid, light=personas).
            if is_within:
                color = C["warm"] if is_grid else C["warm_light"]
            else:
                color = "#00695C" if is_grid else "#80CBC4"
            ax.barh(yp, e["r"], height=bar_h, color=color,
                    alpha=0.95, edgecolor=C["bar_edge"], linewidth=BAR["edge_lw"], zorder=3)
            err_lo = max(0, e["r"] - e["r_lo"])
            err_hi = max(0, e["r_hi"] - e["r"])
            ax.errorbar(e["r"], yp, xerr=[[err_lo], [err_hi]],
                         fmt="none", ecolor="black",
                         capsize=3.5, lw=1.0, zorder=4)
            text_color = C["warm_label"] if is_within else C["teal_label"]
            lbl = f"{e['session']}–{e['induction']} ({e['construct']})"
            if e["r"] >= 0:
                ax.text(-0.09, yp, lbl + "  ",
                         va="center", ha="right", fontsize=FS["bar_label"],
                         color=text_color)
            else:
                ax.text(0.09, yp, "  " + lbl,
                         va="center", ha="left", fontsize=FS["bar_label"],
                         color=text_color)

    ax.axvline(0, color="black", lw=0.9, ls="--", alpha=0.5)
    ax.set_yticks(y_centers)
    ax.set_yticklabels([TASK_LABELS[t] for t in tasks],
                        fontsize=FS["panel_title"], fontweight="bold")
    ax.set_xlabel("Fisher-z aggregated mean r_aligned (95% CI)", fontsize=FS["axis_label"])
    ax.tick_params(axis="x", labelsize=14)
    ax.set_xlim(-1.2, 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, lw=0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C["warm"], label="within–grid"),
        Patch(facecolor=C["warm_light"], label="within–personas"),
        Patch(facecolor="#00695C", label="between–grid"),
        Patch(facecolor="#80CBC4", label="between–personas"),
    ]
    # Place legend below the x-axis so it cannot collide with labels
    # placed in the positive half for negative bars (IAT) or in the
    # negative half for positive bars (Honesty, CCT).
    ax.legend(handles=legend_elements, loc="upper center",
               bbox_to_anchor=(0.5, -0.09), ncol=4,
               fontsize=FS["legend"], frameon=True, framealpha=0.9,
               edgecolor="lightgray")

    fig.suptitle(
        "RQ4 appendix — Full session × induction interaction (TPB primary "
        "construct)", fontsize=21, fontweight="bold", y=0.97,
    )

    for ext in ("pdf", "png"):
        p = out_dir / f"rq4_induction_appendix.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  saved: {p}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cct_master",     default="cct_master.csv")
    ap.add_argument("--syc_master",     default="sycophancy_master.csv")
    ap.add_argument("--honesty_master", default="honesty_master.csv")
    ap.add_argument("--iat_master",     default="iat_master.csv")
    ap.add_argument("--out_dir",        default="rq4_induction")
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

    print("Loading TPB + Big5 masters (within/between × grid/personas)...")
    data_tables = {"tpb": {}, "big5": {}}
    for session in ("within", "between"):
        data_tables["tpb"][session]  = {}
        data_tables["big5"][session] = {}
        for induction in ("grid", "personas"):
            data_tables["tpb"][session][induction]  = {
                t: load_task_tpb(p, t, session, induction)
                for t, p in masters.items()
            }
            data_tables["big5"][session][induction] = {
                t: load_task_big5(p, t, session, induction)
                for t, p in masters.items()
            }

    print("\nData coverage (rows per slice):")
    print(f"{'task':<10}  {'fw':<5}  {'session':<8}  "
          f"{'grid':>6}  {'personas':>8}")
    for fw in ("tpb", "big5"):
        for session in ("within", "between"):
            for t in TASK_ORDER:
                g = len(data_tables[fw][session]["grid"][t])
                p = len(data_tables[fw][session]["personas"][t])
                print(f"  {t:<8}  {fw:<5}  {session:<8}  {g:>6}  {p:>8}")

    cells = compute_cells(data_tables)
    by_cell = aggregate_by_cell(cells)
    delta_df = compute_delta_r_induction(by_cell, cells)
    best_df = best_construct_delta(delta_df)
    print(f"\nRunning bootstrap Δr_induction CIs ({args.n_boot} iter)...")
    boot = bootstrap_delta_r_induction(cells, n_boot=args.n_boot)
    print("Running Mundlak OLS for all 4 (session × induction) cells...")
    mundlak = compute_mundlak_all(data_tables)

    per_model_tpb_between = aggregate_by_model_induction(
        cells, framework="TPB", session="between")
    per_model_tpb_within = aggregate_by_model_induction(
        cells, framework="TPB", session="within")

    print("Running formal OLS interaction tests (SR × induction) per cell...")
    ols_int = compute_ols_interaction(data_tables, delta_df)

    # Save CSVs
    suffix = args.out_suffix
    cells.to_csv(    out_dir / f"rq4_cells{suffix}.csv", index=False)
    by_cell.to_csv(  out_dir / f"rq4_by_task_framework_induction{suffix}.csv",
                     index=False)
    delta_df.to_csv( out_dir / f"rq4_delta_table{suffix}.csv", index=False)
    best_df.to_csv(  out_dir / f"rq4_best_construct_delta{suffix}.csv",
                     index=False)
    boot.to_csv(     out_dir / f"rq4_bootstrap{suffix}.csv", index=False)
    mundlak.to_csv(  out_dir / f"rq4_mundlak{suffix}.csv", index=False)
    per_model_tpb_between.to_csv(
        out_dir / f"rq4_per_model_tpb_between{suffix}.csv", index=False)
    per_model_tpb_within.to_csv(
        out_dir / f"rq4_per_model_tpb_within{suffix}.csv", index=False)
    ols_int.to_csv(  out_dir / f"rq4_ols_interaction{suffix}.csv", index=False)
    print(f"\nSaved CSVs in {out_dir}")

    headline = print_headline(delta_df, best_df, per_model_tpb_between, boot)

    # OLS interaction table for direct side-by-side with Fisher-z Δr
    print("\n" + "-" * 78)
    print("FORMAL OLS INTERACTION: β_{SR × induction} with cluster-robust SE")
    print("Cluster: model_key; z-standardised within (model × induction).")
    print("β > 0 ⇒ personas strengthens SR → behaviour slope vs. grid")
    print("-" * 78)
    print(f"{'Session':<9}  {'Task':<12}  {'FW':<5}  {'Construct':<18}  "
          f"{'β grid':>7}  {'β pers.':>7}  "
          f"{'β interaction [95% CI]':<26}  {'p':<6}")
    for session in ("between", "within"):
        sub_ols = ols_int[ols_int["session"] == session] if "session" in ols_int.columns else pd.DataFrame()
        if sub_ols.empty:
            print(f"  {session}: no OLS interaction results (statsmodels fit may have failed — check warnings above)")
            continue
        for _, r in sub_ols.sort_values(["framework", "task"]).iterrows():
            sig = ""
            if pd.notna(r["p_interaction"]):
                if r["p_interaction"] < 0.001:   sig = "***"
                elif r["p_interaction"] < 0.01:  sig = "**"
                elif r["p_interaction"] < 0.05:  sig = "*"
            beta_str = (f"{r['beta_interaction']:+.2f} "
                        f"[{r['ci_interaction_lo']:+.2f},"
                        f"{r['ci_interaction_hi']:+.2f}]{sig}")
            print(f"  {session:<7}  {r['task_label']:<10}  "
                  f"{r['framework']:<5}  {r['construct_label']:<18}  "
                  f"{r['beta_grid']:+.2f}     {r['beta_personas']:+.2f}     "
                  f"{beta_str:<26}  {r['p_interaction']:.3f}")
    with open(out_dir / f"rq4_headline{suffix}.json", "w") as f:
        json.dump(headline, f, indent=2)

    # Compute SR stability for Panel C (calls into rq4_prereqs)
    print("\nComputing SR stability per (framework × induction × task × construct × model)...")
    from rq4_prereqs import (compute_stability, compute_discriminability,
                              compute_behavior_stability)
    stability_df = compute_stability(data_tables)
    stability_df.to_csv(out_dir / f"rq4_stability{suffix}.csv", index=False)

    print("Computing SR discriminability per (framework × induction × task × construct × model)...")
    disc_df = compute_discriminability(data_tables)
    disc_df.to_csv(out_dir / f"rq4_discriminability{suffix}.csv", index=False)

    print("Computing behaviour stability per (induction × task × model)...")
    beh_stab_df = compute_behavior_stability(data_tables)
    beh_stab_df.to_csv(out_dir / f"rq4_behavior_stability{suffix}.csv", index=False)

    # Fixed model order shared by RQ4 main + supp figures.
    # Same as RQ3 Panel B: TPB same-session grid r descending — so readers
    # can scan the same model row across RQ3 and RQ4 figures.
    fixed_model_order = (per_model_tpb_within
                            .sort_values("r_grid", ascending=False)
                            ["model"].tolist())

    print("\nGenerating RQ4 main summary figure (3-panel)...")
    figure_rq4_main(by_cell, cells, per_model_tpb_between, fixed_model_order,
                     stability_df, disc_df, out_dir)
    print("\nGenerating RQ4 supplementary figure: absolute SR stability...")
    figure_rq4_supp_sr_stability(stability_df, per_model_tpb_between,
                                   fixed_model_order, out_dir)
    print("\nGenerating RQ4 supplementary figure: absolute SR discriminability...")
    figure_rq4_supp_discriminability(disc_df, per_model_tpb_between,
                                       fixed_model_order, out_dir)
    print("\nGenerating RQ4 supplementary figure: behaviour stability...")
    figure_rq4_supp_behavior_stability(beh_stab_df, per_model_tpb_between,
                                          fixed_model_order, out_dir)
    print("\nGenerating RQ4 appendix figure...")
    figure_rq4_appendix(best_df, out_dir)

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()