#!/usr/bin/env python3
"""
analyze_iat_psycohere.py
========================
Unified IAT analysis for psycohere_v1.

Loads all IAT data sources (between-session TPB/Big5, within-session TPB/Big5),
standardizes columns, computes align_score per policy, then runs core RQ analyses:

  RQ1: Within-session SR ↔ behavior correlation (shared-context coherence)
  RQ2: TPB vs Big5 predictive power comparison (framework specificity)
  RQ3: Within vs between session SR↔behavior comparison (context separation)
  RQ4: Grid vs personas perturbation discrimination (identity induction)

Plus (v2): ceiling-effect diagnostics per (model × test) and a sub-analysis
restricted to the top-N variance tests (default 3).

Behavioral outcome:
  mean_bias_score (raw, 0-1) — proportion of stereotype-consistent responses.
  unbiased_categorization → align_score = 1 - mean_bias_score
  intuitive_fast          → align_score = mean_bias_score
  big5 (neutral)          → align_score = 1 - mean_bias_score

Important: IAT has a strong ceiling effect (pooled mean bias ≈ 0.88-0.91 across
models). Most of the discriminative variance comes from gender_science and
gender_sexuality tests. The per-test ceiling summary + top-N variance sub-
analysis isolate where the usable signal actually lives.

Per-test analysis is BETWEEN-SESSION ONLY: within-session combined files store
mean_bias_score already averaged across all 6 tests, so per-test detail is
unrecoverable there.

Usage
-----
python scripts/analysis/analyze_iat_psycohere.py \
    --results_root results \
    --out_dir results/analysis/iat

Optional flags:
    --top_n_tests 3            (default 3; picked by pooled SD of bias)
    --ceiling_threshold 0.95   (threshold for "pegged" in pct_at_ceiling)

Outputs
-------
  iat_master.csv                  — all conditions, standardized columns
  iat_master_topN.csv             — between-session slice, top-N variance tests only
                                    (drop-in replacement for between-session figure use)
  iat_contrasts.csv               — per-model unbiased-vs-intuitive contrast
  iat_correlations.csv            — SR↔behavior correlations (all 6 tests)
  iat_correlations_topN.csv       — correlations restricted to top-N variance tests
  iat_ceiling_by_test_model.csv   — per-(model × test) ceiling stats
  iat_ceiling_by_test.csv         — per-test pooled stats, sorted by SD
  iat_summary.txt                 — human-readable summary
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ── Constants ──────────────────────────────────────────────────────────────────

MODELS = {
    "qwen_72b", "qwen_235b", "gpt4o_mini", "gemini25_flash", "llama33_70b",
    "llama4_maverick", "deepseek_v31", "phi4", "mistral_large",
    "claude37_sonnet", "claude45_haiku",
}

TPB_CONSTRUCTS = ["attitude_mean", "subjective_norm_mean", "pbc_mean", "intention_mean"]
BIG5_TRAITS    = ["extraversion_mean", "agreeableness_mean", "conscientiousness_mean",
                   "neuroticism_mean", "openness_mean"]

IAT_BEH_COLS = ["mean_bias_score", "bias", "coverage", "n_tests"]

MATCH_KEY = ["model_key", "seed", "temperature", "top_p", "persona_label"]

POLICIES = ["unbiased_categorization", "intuitive_fast"]


# ── Generic helpers ────────────────────────────────────────────────────────────

def _filter_models(df: pd.DataFrame) -> pd.DataFrame:
    if "model_key" not in df.columns:
        return df
    return df[df.model_key.isin(MODELS)].copy()


def _coerce_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _add_align_score(df: pd.DataFrame) -> pd.DataFrame:
    """Raw bias-based align_score. No clipping: IAT bias range is [-1, 1]
    (negative values = anti-stereotypic responses), so clipping to [0, 1]
    would compress variance at the lower end and attenuate correlations."""
    df = df.copy()
    if "policy_id" not in df.columns or "mean_bias_score" not in df.columns:
        return df

    bias = pd.to_numeric(df["mean_bias_score"], errors="coerce")
    align = pd.Series(np.nan, index=df.index)

    mask_unb = df["policy_id"] == "unbiased_categorization"
    mask_int = df["policy_id"] == "intuitive_fast"
    mask_big5 = df["policy_id"] == "big5"

    align[mask_unb]  = 1.0 - bias[mask_unb]
    align[mask_int]  = bias[mask_int]
    align[mask_big5] = 1.0 - bias[mask_big5]

    df["align_score"] = align
    return df


# ── Condition-level loaders ────────────────────────────────────────────────────

def load_between_tpb(root: Path, perturbation: str) -> Optional[pd.DataFrame]:
    """Between-session TPB×IAT aggregated from LONG file using raw bias."""
    path_long = root / "merged" / "between" / perturbation / "tpb_x_iat_long.csv"
    if not path_long.exists():
        print(f"  [warn] not found: {path_long}")
        return None

    df = pd.read_csv(path_long, on_bad_lines="skip")
    df = _filter_models(df)
    df = _coerce_numeric(df, TPB_CONSTRUCTS + ["bias", "coverage"])

    grp_cols = MATCH_KEY + ["policy_id"] + [c for c in TPB_CONSTRUCTS if c in df.columns]
    grp_cols = [c for c in dict.fromkeys(grp_cols) if c in df.columns]
    agg = (df.groupby(grp_cols, dropna=False)
             .agg(mean_bias_score=("bias", "mean"),
                  mean_coverage=("coverage", "mean"),
                  n_tests=("bias", "count"))
             .reset_index())

    agg["session_type"] = "between"
    agg["framework"]    = "tpb"
    agg["perturbation"] = perturbation
    agg = _add_align_score(agg)
    return agg


def load_between_big5(root: Path, perturbation: str) -> Optional[pd.DataFrame]:
    """Between-session Big5×IAT aggregated from per-test rows to condition level."""
    path = root / "merged" / "between" / perturbation / "big5_x_iat.csv"
    if not path.exists():
        print(f"  [warn] not found: {path}")
        return None

    df = pd.read_csv(path, on_bad_lines="skip")
    df = _filter_models(df)
    df = _coerce_numeric(df, BIG5_TRAITS + ["bias", "coverage"])

    grp_cols = MATCH_KEY + [c for c in BIG5_TRAITS if c in df.columns]
    grp_cols = [c for c in dict.fromkeys(grp_cols) if c in df.columns]
    agg = (df.groupby(grp_cols, dropna=False)
             .agg(mean_bias_score=("bias", "mean"),
                  mean_coverage=("coverage", "mean"),
                  n_tests=("bias", "count"))
             .reset_index())

    agg["session_type"] = "between"
    agg["framework"]    = "big5"
    agg["perturbation"] = perturbation
    agg["policy_id"]    = "big5"
    agg = _add_align_score(agg)
    return agg


def load_within_tpb(root: Path, perturbation: str) -> Optional[pd.DataFrame]:
    """Within-session TPB×IAT combined files."""
    exp_dir_name = f"tpb_iat_psycohere_{perturbation}"
    frames = []
    for policy in POLICIES:
        path = root / "within" / perturbation / exp_dir_name / policy / "combined_runs.csv"
        if not path.exists():
            print(f"  [warn] not found: {path}")
            continue
        df = pd.read_csv(path, on_bad_lines="skip")
        df = _filter_models(df)
        if "sr_status" in df.columns:
            df = df[df.sr_status == "ok"]
        if "beh_status" in df.columns:
            df = df[df.beh_status == "ok"]
        rename = {c: c.replace("beh__", "") for c in df.columns if c.startswith("beh__")}
        df = df.rename(columns=rename)
        df = _coerce_numeric(df, TPB_CONSTRUCTS + IAT_BEH_COLS)
        dedup_cols = MATCH_KEY + (["run_id"] if "run_id" in df.columns else [])
        df = df.drop_duplicates(subset=dedup_cols, keep="last")
        df["session_type"] = "within"
        df["framework"]    = "tpb"
        df["perturbation"] = perturbation
        df["policy_id"]    = policy
        df = _add_align_score(df)
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_within_big5(root: Path, perturbation: str) -> Optional[pd.DataFrame]:
    """Within-session Big5×IAT combined file."""
    exp_dir_name = f"big5_psycohere_{perturbation}"
    candidates = [
        root / "within" / perturbation / exp_dir_name / "iat" / "big5" / "combined_runs.csv",
        root / "within" / perturbation / exp_dir_name / "big5" / "combined_runs.csv",
        root / "within" / perturbation / f"big5_iat_psycohere_{perturbation}" / "big5" / "combined_runs.csv",
    ]
    path = None
    for c in candidates:
        if c.exists():
            path = c
            break
    if path is None:
        print(f"  [warn] big5 within iat {perturbation}: not found (tried {len(candidates)} paths)")
        return None

    df = pd.read_csv(path, on_bad_lines="skip")
    df = _filter_models(df)
    if "sr_status" in df.columns:
        df = df[df.sr_status == "ok"]
    if "beh_status" in df.columns:
        df = df[df.beh_status == "ok"]
    rename = {c: c.replace("beh__", "") for c in df.columns if c.startswith("beh__")}
    df = df.rename(columns=rename)
    df = _coerce_numeric(df, BIG5_TRAITS + IAT_BEH_COLS)
    dedup_cols = MATCH_KEY + (["run_id"] if "run_id" in df.columns else [])
    df = df.drop_duplicates(subset=dedup_cols, keep="last")
    df["session_type"] = "within"
    df["framework"]    = "big5"
    df["perturbation"] = perturbation
    df["policy_id"]    = "big5"
    df = _add_align_score(df)
    return df


# ── Per-test loaders (new; preserve test_id) ───────────────────────────────────

def load_per_test_tpb(root: Path, perturbation: str) -> Optional[pd.DataFrame]:
    """Load TPB long file, aggregate over orders but keep test_id.
    Returns one row per (condition × policy × test_id)."""
    path = root / "merged" / "between" / perturbation / "tpb_x_iat_long.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, on_bad_lines="skip")
    df = _filter_models(df)
    df = _coerce_numeric(df, TPB_CONSTRUCTS + ["bias", "coverage"])
    if "test_id" not in df.columns:
        print(f"  [warn] test_id missing in {path}; per-test analysis will skip TPB")
        return None

    grp = MATCH_KEY + ["policy_id", "test_id"] + [c for c in TPB_CONSTRUCTS if c in df.columns]
    grp = [c for c in dict.fromkeys(grp) if c in df.columns]
    agg = (df.groupby(grp, dropna=False)
             .agg(mean_bias_score=("bias", "mean"),
                  n_orders=("bias", "count"))
             .reset_index())
    agg["framework"] = "tpb"
    agg["perturbation"] = perturbation
    return agg


def load_per_test_big5(root: Path, perturbation: str) -> Optional[pd.DataFrame]:
    """Load Big5 file (already one row per condition × test_id from --agg orders)."""
    path = root / "merged" / "between" / perturbation / "big5_x_iat.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, on_bad_lines="skip")
    df = _filter_models(df)
    df = _coerce_numeric(df, BIG5_TRAITS + ["bias", "coverage"])
    if "test_id" not in df.columns:
        print(f"  [warn] test_id missing in {path}; per-test analysis will skip Big5")
        return None

    df = df.rename(columns={"bias": "mean_bias_score"})
    df["policy_id"] = "big5"
    df["framework"] = "big5"
    df["perturbation"] = perturbation

    keep = (MATCH_KEY + ["policy_id", "test_id", "mean_bias_score", "framework", "perturbation"]
            + [c for c in BIG5_TRAITS if c in df.columns])
    return df[[c for c in keep if c in df.columns]].copy()


def load_all_per_test(root: Path) -> pd.DataFrame:
    """Combine per-test data from both frameworks × both perturbations."""
    frames = []
    for pert in ["grid", "personas"]:
        for fn, label in [(load_per_test_tpb, f"TPB per-test {pert}"),
                          (load_per_test_big5, f"Big5 per-test {pert}")]:
            df = fn(root, pert)
            if df is not None and len(df) > 0:
                frames.append(df)
                print(f"  ✓ {label}: {len(df)} rows")
            else:
                print(f"  ✗ {label}: skipped")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Ceiling diagnostics ────────────────────────────────────────────────────────

def compute_ceiling_by_test_model(df_per_test: pd.DataFrame,
                                   ceiling_threshold: float = 0.95) -> pd.DataFrame:
    """Per (model × test) ceiling stats, pooled across policies/perturbations."""
    rows = []
    for (model, test), grp in df_per_test.groupby(["model_key", "test_id"], dropna=False):
        bias = pd.to_numeric(grp["mean_bias_score"], errors="coerce").dropna()
        if len(bias) == 0:
            continue
        rows.append({
            "model_key": model,
            "test_id": test,
            "n": int(len(bias)),
            "mean_bias":      round(float(bias.mean()), 4),
            "std_bias":       round(float(bias.std(ddof=1)) if len(bias) > 1 else 0.0, 4),
            "min_bias":       round(float(bias.min()), 4),
            "max_bias":       round(float(bias.max()), 4),
            "pct_at_ceiling": round(float((bias >= ceiling_threshold).mean()), 4),
        })
    return pd.DataFrame(rows)


def compute_ceiling_by_test(df_per_test: pd.DataFrame,
                             ceiling_threshold: float = 0.95) -> pd.DataFrame:
    """Per-test pooled stats across all (model × condition × policy) rows."""
    rows = []
    for test, grp in df_per_test.groupby("test_id", dropna=False):
        bias = pd.to_numeric(grp["mean_bias_score"], errors="coerce").dropna()
        if len(bias) == 0:
            continue
        rows.append({
            "test_id": test,
            "n": int(len(bias)),
            "pooled_mean_bias":      round(float(bias.mean()), 4),
            "pooled_std_bias":       round(float(bias.std(ddof=1)), 4),
            "pooled_min_bias":       round(float(bias.min()), 4),
            "pooled_max_bias":       round(float(bias.max()), 4),
            "pct_at_ceiling":        round(float((bias >= ceiling_threshold).mean()), 4),
            "n_models":              int(grp["model_key"].nunique()),
        })
    return (pd.DataFrame(rows)
            .sort_values("pooled_std_bias", ascending=False)
            .reset_index(drop=True))


def select_top_variance_tests(ceiling_by_test: pd.DataFrame, n: int = 3) -> List[str]:
    """Pick top-N tests by pooled SD of bias."""
    if ceiling_by_test.empty:
        return []
    return ceiling_by_test.head(n)["test_id"].tolist()


def aggregate_to_selected_tests(df_per_test: pd.DataFrame,
                                 selected_tests: List[str]) -> pd.DataFrame:
    """Filter to selected tests, aggregate back to condition×policy level."""
    if not selected_tests or df_per_test.empty:
        return pd.DataFrame()

    df = df_per_test[df_per_test.test_id.isin(selected_tests)].copy()
    frames = []

    for fw, sub in df.groupby("framework", dropna=False):
        sr_cols = TPB_CONSTRUCTS if str(fw) == "tpb" else BIG5_TRAITS
        grp = (MATCH_KEY + ["policy_id", "framework", "perturbation"]
               + [c for c in sr_cols if c in sub.columns])
        grp = [c for c in dict.fromkeys(grp) if c in sub.columns]
        agg = (sub.groupby(grp, dropna=False)
                 .agg(mean_bias_score=("mean_bias_score", "mean"),
                      n_tests_used=("mean_bias_score", "count"))
                 .reset_index())
        agg["session_type"] = "between_topN"
        agg = _add_align_score(agg)
        frames.append(agg)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Correlation machinery ──────────────────────────────────────────────────────

def pearson_r(x: pd.Series, y: pd.Series) -> tuple:
    mask = x.notna() & y.notna()
    n = mask.sum()
    if n < 5:
        return (np.nan, np.nan, n)
    if x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return (np.nan, np.nan, n)
    r, p = stats.pearsonr(x[mask], y[mask])
    return (round(float(r), 4), round(float(p), 4), int(n))


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sr_cols = {"tpb": TPB_CONSTRUCTS, "big5": BIG5_TRAITS}

    for (fw, sess, pert, pol), grp in df.groupby(
            ["framework", "session_type", "perturbation", "policy_id"], dropna=False):
        constructs = sr_cols.get(str(fw), [])
        for sr_col in constructs:
            if sr_col not in grp.columns:
                continue
            for beh_col in ["align_score", "mean_bias_score"]:
                if beh_col not in grp.columns:
                    continue
                r, p, n = pearson_r(
                    pd.to_numeric(grp[sr_col], errors="coerce"),
                    pd.to_numeric(grp[beh_col], errors="coerce"),
                )
                rows.append({
                    "framework":    fw,
                    "session_type": sess,
                    "perturbation": pert,
                    "policy_id":    pol,
                    "sr_construct": sr_col,
                    "beh_outcome":  beh_col,
                    "r": r, "p": p, "n": n,
                })
    return pd.DataFrame(rows)


def compute_policy_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model unbiased − intuitive_fast contrast on intention and bias.

    Note: bias_contrast is only meaningful within-session — between-session IAT
    runs are policy-agnostic (same iat_runs.csv joins to both policy SR variants),
    so bias_unb == bias_intu by construction and their difference is always zero.
    Between-session rows retain intention_contrast (since SR IS policy-conditioned)
    but have NaN for all bias-related columns.
    """
    rows = []
    tpb = df[(df.framework == "tpb") & df.policy_id.isin(POLICIES)].copy()

    for (sess, pert, model), grp in tpb.groupby(
            ["session_type", "perturbation", "model_key"], dropna=False):
        unb = grp[grp.policy_id == "unbiased_categorization"]
        intu = grp[grp.policy_id == "intuitive_fast"]
        if unb.empty or intu.empty:
            continue

        def _mean(sub, col):
            if col not in sub.columns:
                return np.nan
            return pd.to_numeric(sub[col], errors="coerce").mean()

        int_unb  = _mean(unb,  "intention_mean")
        int_intu = _mean(intu, "intention_mean")

        # Bias contrast only meaningful within-session (behavior is policy-conditioned)
        if sess == "within":
            bias_unb  = _mean(unb,  "mean_bias_score")
            bias_intu = _mean(intu, "mean_bias_score")
        else:
            bias_unb = np.nan
            bias_intu = np.nan

        def _diff(a, b):
            return round(a - b, 4) if not (np.isnan(a) or np.isnan(b)) else np.nan

        rows.append({
            "session_type":           sess,
            "perturbation":           pert,
            "model_key":              model,
            "intention_unbiased":     round(int_unb,  4) if not np.isnan(int_unb)  else np.nan,
            "intention_intuitive":    round(int_intu, 4) if not np.isnan(int_intu) else np.nan,
            "intention_contrast":     _diff(int_unb, int_intu),
            "mean_bias_unbiased":     round(bias_unb,  4) if not np.isnan(bias_unb)  else np.nan,
            "mean_bias_intuitive":    round(bias_intu, 4) if not np.isnan(bias_intu) else np.nan,
            "bias_contrast":          _diff(bias_unb, bias_intu),
        })
    return pd.DataFrame(rows)


# ── Summary ────────────────────────────────────────────────────────────────────

def _abbrev_test(test_id: str, width: int = 12) -> str:
    return (test_id or "")[:width]


def _stars(p) -> str:
    if pd.isna(p):
        return "   "
    if p < 0.001: return "***"
    if p < 0.01:  return "** "
    if p < 0.05:  return "*  "
    return "   "


def print_summary(df: pd.DataFrame, corr: pd.DataFrame, contrasts: pd.DataFrame,
                  ceiling_by_test_model: pd.DataFrame, ceiling_by_test: pd.DataFrame,
                  corr_topN: pd.DataFrame, selected_tests: List[str], top_n: int,
                  ceiling_threshold: float, out_txt: Path) -> None:
    lines = []
    w = lines.append

    w("=" * 78)
    w("IAT ANALYSIS SUMMARY — psycohere_v1")
    w("=" * 78)

    # Data coverage
    w("\n── Data coverage ────────────────────────────────────────────────────────")
    for (fw, sess, pert), grp in df.groupby(["framework", "session_type", "perturbation"]):
        n_cond = grp[MATCH_KEY].drop_duplicates().shape[0]
        n_models = grp.model_key.nunique()
        policies = sorted(grp.policy_id.dropna().unique())
        w(f"  {fw:<5} {sess:<8} {pert:<9}  {n_cond:>4} conditions | "
          f"{n_models} models | policies: {policies}")

    # Per-test ceiling summary
    w("\n── Per-test ceiling summary (between-session pooled) ────────────────────")
    w(f"  pct_at_ceiling = fraction of runs with bias >= {ceiling_threshold}")
    w("  Sorted by pooled SD of bias (descending = more usable signal)\n")
    if not ceiling_by_test.empty:
        w(f"  {'test_id':<24}  {'n':>5}  {'mean':>6}  {'SD':>6}  {'min':>6}  {'max':>6}  {'pct_peg':>7}")
        for _, r in ceiling_by_test.iterrows():
            w(f"  {r.test_id:<24}  {r.n:>5d}  "
              f"{r.pooled_mean_bias:>6.3f}  {r.pooled_std_bias:>6.3f}  "
              f"{r.pooled_min_bias:>6.3f}  {r.pooled_max_bias:>6.3f}  "
              f"{r.pct_at_ceiling:>7.3f}")
    else:
        w("  (per-test data not available)")

    # Ceiling map (model × test)
    w("\n── Ceiling map: mean_bias by (model × test) ─────────────────────────────")
    w("  ▓ = pct_at_ceiling >= 0.80 ;  ▒ = 0.40–0.80 ;  · = < 0.40")
    w("  (aggregated across policies and perturbations)\n")
    if not ceiling_by_test_model.empty and not ceiling_by_test.empty:
        tests = ceiling_by_test.test_id.tolist()
        models = sorted(ceiling_by_test_model.model_key.unique())
        header = f"  {'model_key':<22}  " + "  ".join(f"{_abbrev_test(t, 11):>11}" for t in tests)
        w(header)
        for m in models:
            cells = []
            for t in tests:
                row = ceiling_by_test_model[
                    (ceiling_by_test_model.model_key == m) &
                    (ceiling_by_test_model.test_id == t)
                ]
                if row.empty:
                    cells.append(f"{'—':>11}")
                else:
                    mb  = float(row.iloc[0].mean_bias)
                    pct = float(row.iloc[0].pct_at_ceiling)
                    mark = "▓" if pct >= 0.80 else "▒" if pct >= 0.40 else "·"
                    cells.append(f"{mb:>6.3f} {mark:>3}")
            w(f"  {m:<22}  " + "  ".join(cells))
    else:
        w("  (per-test data not available)")

    # Top-N variance tests
    w("\n── Top-N variance tests (selected for sub-analysis) ─────────────────────")
    if selected_tests:
        w(f"  top {top_n} tests by pooled SD: {selected_tests}\n")
    else:
        w("  (no tests selected)\n")

    # Behavioral baselines (pooled all tests)
    w("\n── Behavioral baselines (mean_bias_score, all tests, between-session) ───")
    w("  Note: IAT bias is near-ceiling for most models. Restricted range bounds")
    w("  all correlations below — the top-N sub-analysis addresses this.\n")
    btwn = df[df.session_type == "between"]
    if not btwn.empty and "mean_bias_score" in btwn.columns:
        baseline = (btwn.groupby("model_key")["mean_bias_score"]
                    .mean().sort_values().round(4))
        for m, v in baseline.items():
            bar = "▓" * int(v * 20)
            w(f"  {m:<22}  bias={v:.4f}  {bar}")
        pooled = btwn["mean_bias_score"].mean()
        w(f"  {'── pooled ──':<22}  bias={pooled:.4f}")

    # Policy contrast
    w("\n── Behavioral shift: unbiased vs intuitive policy ───────────────────────")
    w("  unbiased = unbiased_categorization (target: low bias)")
    w("  intuitive = intuitive_fast (target: high bias)")
    w("  bias_contrast = bias(unbiased) − bias(intuitive); expect NEGATIVE if policies work")
    w("  Between-session: bias_contrast is structurally zero (IAT runs are policy-")
    w("    agnostic, joined to both SR variants), so only intention_contrast is shown.\n")
    for (sess, pert), grp in contrasts.groupby(["session_type", "perturbation"]):
        if sess == "between":
            w(f"  [{sess}/{pert}]  intention(unb−int)")
            for _, r in grp.sort_values("intention_contrast", ascending=False).iterrows():
                ic = f"{r.intention_contrast:+.3f}" if pd.notna(r.intention_contrast) else "   nan"
                w(f"    {r.model_key:<22}  {ic}")
            ic_mean = grp.intention_contrast.mean()
            w(f"    {'── pooled ──':<22}  {ic_mean:+.3f}")
            w("")
        else:  # within
            w(f"  [{sess}/{pert}]  intention(unb−int)  bias_unb  bias_int   bias_contrast")
            for _, r in grp.sort_values("bias_contrast").iterrows():
                ic = f"{r.intention_contrast:+.3f}" if pd.notna(r.intention_contrast) else "   nan"
                bu = f"{r.mean_bias_unbiased:.4f}"  if pd.notna(r.mean_bias_unbiased)  else "  nan"
                bi = f"{r.mean_bias_intuitive:.4f}" if pd.notna(r.mean_bias_intuitive) else "  nan"
                bc = f"{r.bias_contrast:+.4f}"      if pd.notna(r.bias_contrast)      else "  nan"
                w(f"    {r.model_key:<22}  {ic}             {bu}    {bi}    {bc}")
            bc_mean = grp.bias_contrast.mean()
            ic_mean = grp.intention_contrast.mean()
            w(f"    {'── pooled ──':<22}  {ic_mean:+.3f}                                    {bc_mean:+.4f}")
            w("")

    # Key correlations (all 6 tests)
    w("\n── SR → Behavior (intention → align_score) — ALL 6 tests ───────────────")
    key_corr = corr[
        (corr.sr_construct == "intention_mean") &
        (corr.beh_outcome == "align_score") &
        (corr.framework == "tpb")
    ].copy()
    for _, r in key_corr.sort_values(["session_type", "perturbation", "policy_id"]).iterrows():
        r_str = f"{r.r:+.3f}" if pd.notna(r.r) else "  nan"
        p_str = f"{r.p:.3f}"  if pd.notna(r.p) else " nan"
        w(f"  {r.session_type:<8} {r.perturbation:<9} {r.policy_id:<25}  "
          f"r={r_str}  p={p_str}{_stars(r.p)}  n={r.n}")

    # Key correlations restricted to top-N
    w(f"\n── SR → Behavior (intention → align_score) — TOP-{top_n} tests only ──────────")
    w(f"  tests: {selected_tests}\n")
    if not corr_topN.empty:
        key_corr_top = corr_topN[
            (corr_topN.sr_construct == "intention_mean") &
            (corr_topN.beh_outcome == "align_score") &
            (corr_topN.framework == "tpb")
        ].copy()
        for _, r in key_corr_top.sort_values(["perturbation", "policy_id"]).iterrows():
            r_str = f"{r.r:+.3f}" if pd.notna(r.r) else "  nan"
            p_str = f"{r.p:.3f}"  if pd.notna(r.p) else " nan"
            w(f"  {r.perturbation:<9} {r.policy_id:<25}  "
              f"r={r_str}  p={p_str}{_stars(r.p)}  n={r.n}")
        w("\n  Δr vs all-6-tests (between-session only):")
        for _, r in key_corr_top.iterrows():
            base = key_corr[
                (key_corr.session_type == "between") &
                (key_corr.perturbation == r.perturbation) &
                (key_corr.policy_id == r.policy_id)
            ]
            if base.empty or pd.isna(r.r) or pd.isna(base.iloc[0].r):
                continue
            delta = r.r - base.iloc[0].r
            w(f"    {r.perturbation:<9} {r.policy_id:<25}  "
              f"top{top_n} r={r.r:+.3f}  all6 r={base.iloc[0].r:+.3f}  Δ={delta:+.3f}")

    # All TPB constructs (all 6 tests)
    w("\n── All TPB constructs → align_score (all 6 tests) ───────────────────────")
    all_tpb = corr[
        (corr.framework == "tpb") &
        (corr.beh_outcome == "align_score") &
        (corr.sr_construct.isin(TPB_CONSTRUCTS))
    ].copy()
    for (sess, pert, pol), grp in all_tpb.groupby(
            ["session_type", "perturbation", "policy_id"]):
        w(f"  [{sess}/{pert}/{pol}]")
        for _, r in grp.sort_values("r", key=lambda s: s.abs(),
                                     ascending=False, na_position="last").iterrows():
            r_str = f"{r.r:+.3f}" if pd.notna(r.r) else "  nan"
            p_str = f"{r.p:.3f}"  if pd.notna(r.p) else " nan"
            w(f"    {r.sr_construct:<30}  r={r_str}  p={p_str}{_stars(r.p)}  n={r.n}")

    # Big5 → bias (all 6 tests)
    w("\n── Big5 traits → mean_bias_score (lower=less biased, all 6 tests) ──────")
    big5_corr = corr[
        (corr.framework == "big5") &
        (corr.beh_outcome == "mean_bias_score") &
        (corr.sr_construct.isin(BIG5_TRAITS))
    ].copy()
    for (sess, pert), grp in big5_corr.groupby(["session_type", "perturbation"]):
        w(f"  [{sess}/{pert}]")
        for _, r in grp.sort_values("r", key=lambda s: s.abs(),
                                     ascending=False, na_position="last").iterrows():
            r_str = f"{r.r:+.3f}" if pd.notna(r.r) else "  nan"
            p_str = f"{r.p:.3f}"  if pd.notna(r.p) else " nan"
            w(f"    {r.sr_construct:<30}  r={r_str}  p={p_str}{_stars(r.p)}  n={r.n}")

    # Big5 → bias restricted to top-N
    w(f"\n── Big5 traits → mean_bias_score — TOP-{top_n} tests only ───────────────────")
    if not corr_topN.empty:
        big5_top = corr_topN[
            (corr_topN.framework == "big5") &
            (corr_topN.beh_outcome == "mean_bias_score") &
            (corr_topN.sr_construct.isin(BIG5_TRAITS))
        ].copy()
        for pert, grp in big5_top.groupby("perturbation"):
            w(f"  [top{top_n}/{pert}]")
            for _, r in grp.sort_values("r", key=lambda s: s.abs(),
                                         ascending=False, na_position="last").iterrows():
                r_str = f"{r.r:+.3f}" if pd.notna(r.r) else "  nan"
                p_str = f"{r.p:.3f}"  if pd.notna(r.p) else " nan"
                w(f"    {r.sr_construct:<30}  r={r_str}  p={p_str}{_stars(r.p)}  n={r.n}")

    # RQ3: within vs between
    w("\n── RQ3: Within vs Between (all 6 tests) ────────────────────────────────")
    for pol in POLICIES:
        for pert in ["grid", "personas"]:
            wi = key_corr[(key_corr.session_type == "within") &
                          (key_corr.perturbation == pert) &
                          (key_corr.policy_id == pol)]
            be = key_corr[(key_corr.session_type == "between") &
                          (key_corr.perturbation == pert) &
                          (key_corr.policy_id == pol)]
            if wi.empty or be.empty:
                continue
            r_wi = wi.iloc[0]
            r_be = be.iloc[0]
            if pd.notna(r_wi.r) and pd.notna(r_be.r):
                delta = r_wi.r - r_be.r
                w(f"  {pol:<26} {pert:<9}  "
                  f"within r={r_wi.r:+.3f}  between r={r_be.r:+.3f}  Δ={delta:+.3f}")

    w("\n  Note: Top-N sub-analysis is between-session only (within-session")
    w("        combined files store mean_bias already averaged over all tests).")

    w("\n" + "=" * 78)
    text = "\n".join(lines)
    print(text)
    out_txt.write_text(text, encoding="utf-8")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--out_dir",      default="results/analysis/iat")
    ap.add_argument("--top_n_tests",  type=int,   default=3,
                    help="Number of top-variance tests to use in sub-analysis (default 3)")
    ap.add_argument("--ceiling_threshold", type=float, default=0.95,
                    help="Threshold for pct_at_ceiling (default 0.95)")
    args = ap.parse_args()

    root    = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Condition-level master
    print("Loading IAT data sources (condition-level)...")
    frames = []
    loaders = [
        ("between TPB grid",      load_between_tpb,  root, "grid"),
        ("between TPB personas",  load_between_tpb,  root, "personas"),
        ("between Big5 grid",     load_between_big5, root, "grid"),
        ("between Big5 personas", load_between_big5, root, "personas"),
        ("within TPB grid",       load_within_tpb,   root, "grid"),
        ("within TPB personas",   load_within_tpb,   root, "personas"),
        ("within Big5 grid",      load_within_big5,  root, "grid"),
        ("within Big5 personas",  load_within_big5,  root, "personas"),
    ]
    for label, fn, *fargs in loaders:
        df = fn(*fargs)
        if df is not None:
            print(f"  ✓ {label}: {len(df)} rows")
            frames.append(df)
        else:
            print(f"  ✗ {label}: skipped")

    if not frames:
        print("No data loaded. Exiting.")
        return

    master = pd.concat(frames, ignore_index=True)
    for c in ["align_score", "mean_bias_score"] + TPB_CONSTRUCTS + BIG5_TRAITS:
        if c in master.columns:
            master[c] = pd.to_numeric(master[c], errors="coerce")
    master_path = out_dir / "iat_master.csv"
    master.to_csv(master_path, index=False)
    print(f"\nMaster file: {master_path} ({len(master)} rows)")

    # Per-test (between-session only)
    print("\nLoading per-test data (between-session only)...")
    per_test = load_all_per_test(root)

    ceiling_bm = pd.DataFrame()
    ceiling_t  = pd.DataFrame()
    selected_tests: List[str] = []
    corr_topN  = pd.DataFrame()

    if not per_test.empty:
        ceiling_bm = compute_ceiling_by_test_model(per_test, args.ceiling_threshold)
        ceiling_t  = compute_ceiling_by_test(per_test, args.ceiling_threshold)
        ceiling_bm.to_csv(out_dir / "iat_ceiling_by_test_model.csv", index=False)
        ceiling_t.to_csv(out_dir  / "iat_ceiling_by_test.csv", index=False)
        print(f"  Per-(model×test) ceiling: {len(ceiling_bm)} rows → iat_ceiling_by_test_model.csv")
        print(f"  Per-test ceiling:         {len(ceiling_t)}  rows → iat_ceiling_by_test.csv")

        selected_tests = select_top_variance_tests(ceiling_t, n=args.top_n_tests)
        print(f"  Top-{args.top_n_tests} variance tests: {selected_tests}")

        topN_agg = aggregate_to_selected_tests(per_test, selected_tests)
        if not topN_agg.empty:
            corr_topN = compute_correlations(topN_agg)
            corr_topN.to_csv(out_dir / f"iat_correlations_top{args.top_n_tests}.csv",
                              index=False)
            print(f"  Top-{args.top_n_tests} correlations: {len(corr_topN)} rows "
                  f"→ iat_correlations_top{args.top_n_tests}.csv")

            # Drop-in master file for between-session top-N figures:
            # relabel session_type ("between_topN" → "between") and rename
            # n_tests_used → n_tests so this file is column-compatible with
            # the between-session slice of iat_master.csv. Figure scripts
            # can swap --iat_master iat_master_top3.csv for the top-3 version.
            master_topN = topN_agg.copy()
            master_topN["session_type"] = "between"
            if "n_tests_used" in master_topN.columns:
                master_topN = master_topN.rename(columns={"n_tests_used": "n_tests"})
            master_topN_path = out_dir / f"iat_master_top{args.top_n_tests}.csv"
            master_topN.to_csv(master_topN_path, index=False)
            print(f"  Top-{args.top_n_tests} master:       {len(master_topN)} rows "
                  f"→ {master_topN_path.name}  "
                  f"(drop-in between-session replacement for figure scripts)")
    else:
        print("  (no per-test data found — ceiling analysis skipped)")

    # Main correlations + contrasts
    print("\nComputing correlations (all 6 tests)...")
    corr = compute_correlations(master)
    corr.to_csv(out_dir / "iat_correlations.csv", index=False)
    print(f"  Correlations: {len(corr)} rows → iat_correlations.csv")

    print("Computing policy contrasts...")
    contrasts = compute_policy_contrasts(master)
    contrasts.to_csv(out_dir / "iat_contrasts.csv", index=False)
    print(f"  Contrasts: {len(contrasts)} rows → iat_contrasts.csv")

    # Summary
    print("\nGenerating summary...")
    print_summary(master, corr, contrasts, ceiling_bm, ceiling_t,
                  corr_topN, selected_tests, args.top_n_tests,
                  args.ceiling_threshold, out_dir / "iat_summary.txt")
    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()