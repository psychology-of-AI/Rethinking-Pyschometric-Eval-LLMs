#!/usr/bin/env python3
"""
analyze_honesty_psycohere.py
============================
Unified Honesty analysis for psycohere_v1.

Loads all Honesty data sources (between-session TPB/Big5, within-session TPB/Big5),
standardizes columns, computes align_score per policy, then runs core RQ analyses:

  RQ1: Within-session SR ↔ behavior correlation (shared-context coherence)
  RQ2: TPB vs Big5 predictive power comparison (framework specificity)
  RQ3: Within vs between session SR↔behavior comparison (context separation)
  RQ4: Grid vs personas perturbation discrimination (identity induction)

Behavioral outcomes:
  calibrated_confidence : mean_brier_c1 (lower=better) → align_score = 1 - brier_c1
  keep_confidence_stable: mean_inconsistency_abs (lower=better) → align_score = 1 - inconsistency

Usage
-----
python scripts/analysis/analyze_honesty_psycohere.py \
    --results_root results \
    --out_dir results/analysis/honesty

Outputs
-------
  honesty_master.csv       — all conditions, standardized columns
  honesty_contrasts.csv    — per-model CC−KCS policy contrast
  honesty_correlations.csv — SR↔behavior correlations by framework×session×perturbation
  honesty_summary.txt      — human-readable analysis summary
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

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

HONESTY_BEH_COLS = ["mean_brier_c1", "mean_brier_c2", "mean_brier_improvement",
                     "mean_confidence_delta", "mean_inconsistency_abs",
                     "mean_abs_confidence_delta", "accuracy", "n_questions"]

MATCH_KEY = ["model_key", "seed", "temperature", "top_p", "persona_label"]

POLICIES = ["calibrated_confidence", "keep_confidence_stable"]


# ── Helpers ────────────────────────────────────────────────────────────────────

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
    """
    Compute align_score per policy if not already present.
      calibrated_confidence  → 1 - brier_c1 (clipped 0-1); higher = better calibrated
      keep_confidence_stable → 1 - inconsistency_abs (clipped 0-1); higher = more stable
    """
    if "align_score" in df.columns and df["align_score"].notna().any():
        return df

    df = df.copy()
    if "policy_id" not in df.columns:
        return df

    brier = pd.to_numeric(df.get("mean_brier_c1", np.nan), errors="coerce").clip(0, 1)
    incon = pd.to_numeric(df.get("mean_inconsistency_abs",
                  df.get("mean_abs_confidence_delta", np.nan)), errors="coerce").clip(0, 1)

    align = pd.Series(np.nan, index=df.index)
    mask_cc  = df["policy_id"] == "calibrated_confidence"
    mask_kcs = df["policy_id"] == "keep_confidence_stable"
    mask_big5 = df["policy_id"] == "big5"

    align[mask_cc]  = (1.0 - brier[mask_cc]).clip(0, 1)
    align[mask_kcs] = (1.0 - incon[mask_kcs]).clip(0, 1)
    align[mask_big5] = (1.0 - brier[mask_big5]).clip(0, 1)   # neutral: use brier

    df["align_score"] = align
    return df


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_between_tpb(root: Path, perturbation: str) -> Optional[pd.DataFrame]:
    # Prefer session CSV (per-condition) when it carries the full set of
    # behavioral outcome columns. Current session files only carry `align_mean`
    # (a sign-corrected composite) and not the raw brier/inconsistency values,
    # so detect that gap and fall back to aggregating from the long CSV
    # (per-question rows) which always has the full raw behavioral cols.
    path_sess = root / "merged" / "between" / perturbation / "tpb_x_honesty_session.csv"
    path_long = root / "merged" / "between" / perturbation / "tpb_x_honesty_long.csv"

    df = None

    # Try session path first
    if path_sess.exists():
        df_sess = pd.read_csv(path_sess, on_bad_lines="skip")
        df_sess = _filter_models(df_sess)
        if "align_mean" in df_sess.columns and "align_score" not in df_sess.columns:
            df_sess = df_sess.rename(columns={"align_mean": "align_score"})
        df_sess = _coerce_numeric(df_sess, TPB_CONSTRUCTS + HONESTY_BEH_COLS + ["align_score"])
        # Check whether session CSV carries raw behavioral columns; if not,
        # fall through to the long-CSV aggregation path below.
        has_brier = ("mean_brier_c1" in df_sess.columns
                     and df_sess["mean_brier_c1"].notna().any())
        if has_brier:
            df = df_sess
        else:
            print(f"  [info] session CSV lacks raw behavioral cols — "
                  f"aggregating from long CSV instead")

    # Long-path aggregation (always produces raw brier + inconsistency)
    # Long CSV uses per-question raw names (brier_c1, inconsistency_abs, ...)
    # rather than the condition-level aggregated names (mean_brier_c1, ...).
    # We rename-and-aggregate to the mean_-prefixed schema that the rest of
    # the analyzer and downstream consumers expect.
    if df is None and path_long.exists():
        df_long = pd.read_csv(path_long, on_bad_lines="skip")
        df_long = _filter_models(df_long)

        # Per-question -> condition-level mean column name mapping
        raw_to_mean = {
            "brier_c1":          "mean_brier_c1",
            "brier_c2":          "mean_brier_c2",
            "brier_improvement": "mean_brier_improvement",
            "confidence_delta":  "mean_confidence_delta",
            "inconsistency_abs": "mean_inconsistency_abs",
            "accuracy":          "accuracy",
        }
        coerce_cols = list(raw_to_mean) + TPB_CONSTRUCTS + ["align_score"]
        df_long = _coerce_numeric(df_long, coerce_cols)
        # Derive abs confidence delta for the mean_abs_confidence_delta stat
        # used by _add_align_score (fallback for mean_inconsistency_abs)
        if "confidence_delta" in df_long.columns:
            df_long["abs_confidence_delta"] = df_long["confidence_delta"].abs()
            raw_to_mean["abs_confidence_delta"] = "mean_abs_confidence_delta"

        grp_cols = MATCH_KEY + ["policy_id"] + [c for c in TPB_CONSTRUCTS if c in df_long.columns]
        grp_cols = [c for c in dict.fromkeys(grp_cols) if c in df_long.columns]

        # Named aggregation: raw per-question -> condition-level mean columns
        agg_named = {renamed: (raw, "mean") for raw, renamed in raw_to_mean.items()
                     if raw in df_long.columns}
        # n_questions: count of answered per-question rows per group
        for probe in ("brier_c1", "confidence_delta", "accuracy"):
            if probe in df_long.columns:
                agg_named["n_questions"] = (probe, "count")
                break
        # align_score: aggregate as mean if present per-question, else derive later
        if "align_score" in df_long.columns:
            agg_named["align_score"] = ("align_score", "mean")

        df = df_long.groupby(grp_cols, dropna=False).agg(**agg_named).reset_index()
        # Safety net: compute align_score from condition-level means if missing
        df = _add_align_score(df)

    if df is None:
        print(f"  [warn] not found: {path_sess} or {path_long}")
        return None

    df["session_type"] = "between"
    df["framework"]    = "tpb"
    df["perturbation"] = perturbation
    return df


def load_between_big5(root: Path, perturbation: str) -> Optional[pd.DataFrame]:
    path = root / "merged" / "between" / perturbation / "big5_x_honesty.csv"
    if not path.exists():
        print(f"  [warn] not found: {path}")
        return None
    df = pd.read_csv(path, on_bad_lines="skip")
    df = _filter_models(df)
    df = _coerce_numeric(df, BIG5_TRAITS + HONESTY_BEH_COLS)
    df["session_type"] = "between"
    df["framework"]    = "big5"
    df["perturbation"] = perturbation
    df["policy_id"]    = "big5"
    df = _add_align_score(df)
    return df


def load_within_tpb(root: Path, perturbation: str) -> Optional[pd.DataFrame]:
    exp_dir_name = f"tpb_honesty_psycohere_{perturbation}"
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
        # Strip beh__ prefix from behavioral columns
        rename = {c: c.replace("beh__", "") for c in df.columns if c.startswith("beh__")}
        df = df.rename(columns=rename)
        # Map mean_abs_confidence_delta → mean_inconsistency_abs for keep_confidence_stable
        if "mean_abs_confidence_delta" in df.columns and "mean_inconsistency_abs" not in df.columns:
            df["mean_inconsistency_abs"] = df["mean_abs_confidence_delta"]
        df = _coerce_numeric(df, TPB_CONSTRUCTS + HONESTY_BEH_COLS)
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
    exp_dir_name = f"big5_psycohere_{perturbation}"
    # Try multiple possible path structures
    candidates = [
        root / "within" / perturbation / exp_dir_name / "honesty" / "big5" / "combined_runs.csv",
        root / "within" / perturbation / exp_dir_name / "big5" / "combined_runs.csv",
        root / "within" / perturbation / f"big5_honesty_psycohere_{perturbation}" / "big5" / "combined_runs.csv",
    ]
    path = None
    for c in candidates:
        if c.exists():
            path = c
            break
    if path is None:
        print(f"  [warn] big5 within honesty {perturbation}: not found (tried {len(candidates)} paths)")
        return None
    df = pd.read_csv(path, on_bad_lines="skip")
    df = _filter_models(df)
    if "sr_status" in df.columns:
        df = df[df.sr_status == "ok"]
    rename = {c: c.replace("beh__", "") for c in df.columns if c.startswith("beh__")}
    df = df.rename(columns=rename)
    if "mean_abs_confidence_delta" in df.columns and "mean_inconsistency_abs" not in df.columns:
        df["mean_inconsistency_abs"] = df["mean_abs_confidence_delta"]
    df = _coerce_numeric(df, BIG5_TRAITS + HONESTY_BEH_COLS)
    df["session_type"] = "within"
    df["framework"]    = "big5"
    df["perturbation"] = perturbation
    df["policy_id"]    = "big5"
    df = _add_align_score(df)
    return df


# ── Analysis ───────────────────────────────────────────────────────────────────

def pearson_r(x: pd.Series, y: pd.Series) -> tuple:
    mask = x.notna() & y.notna()
    n = mask.sum()
    if n < 5:
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
            for beh_col in ["align_score", "mean_brier_c1", "mean_inconsistency_abs",
                             "mean_confidence_delta", "accuracy"]:
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
    """Per-model CC − KCS contrast on intention and primary behavioral outcomes."""
    rows = []
    tpb = df[(df.framework == "tpb") & df.policy_id.isin(POLICIES)].copy()

    for (sess, pert, model), grp in tpb.groupby(
            ["session_type", "perturbation", "model_key"], dropna=False):
        cc  = grp[grp.policy_id == "calibrated_confidence"]
        kcs = grp[grp.policy_id == "keep_confidence_stable"]
        if cc.empty or kcs.empty:
            continue

        def _mean(sub, col):
            if col not in sub.columns:
                return np.nan
            return pd.to_numeric(sub[col], errors="coerce").mean()

        int_cc  = _mean(cc,  "intention_mean")
        int_kcs = _mean(kcs, "intention_mean")
        brier_cc  = _mean(cc,  "mean_brier_c1")
        brier_kcs = _mean(kcs, "mean_brier_c1")
        incon_kcs = _mean(kcs, "mean_inconsistency_abs")

        def _diff(a, b):
            return round(a - b, 4) if not (np.isnan(a) or np.isnan(b)) else np.nan

        rows.append({
            "session_type":        sess,
            "perturbation":        pert,
            "model_key":           model,
            "intention_CC":        round(int_cc,  4) if not np.isnan(int_cc)  else np.nan,
            "intention_KCS":       round(int_kcs, 4) if not np.isnan(int_kcs) else np.nan,
            "intention_contrast":  _diff(int_cc, int_kcs),
            "mean_brier_CC":       round(brier_cc,  4) if not np.isnan(brier_cc)  else np.nan,
            "mean_brier_KCS":      round(brier_kcs, 4) if not np.isnan(brier_kcs) else np.nan,
            "mean_incon_KCS":      round(incon_kcs, 4) if not np.isnan(incon_kcs) else np.nan,
        })
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, corr: pd.DataFrame, contrasts: pd.DataFrame,
                  out_txt: Path) -> None:
    lines = []
    w = lines.append

    w("=" * 70)
    w("HONESTY ANALYSIS SUMMARY — psycohere_v1")
    w("=" * 70)

    # Data coverage
    w("\n── Data coverage ──────────────────────────────────────────────────")
    for (fw, sess, pert), grp in df.groupby(["framework", "session_type", "perturbation"]):
        n_cond = grp[MATCH_KEY].drop_duplicates().shape[0]
        n_models = grp.model_key.nunique()
        policies = sorted(grp.policy_id.dropna().unique())
        w(f"  {fw:<5} {sess:<8} {pert:<9}  {n_cond:>4} conditions | "
          f"{n_models} models | policies: {policies}")

    # Behavioral baselines
    w("\n── Behavioral baselines (mean_brier_c1 by model, between-session) ─")
    btwn = df[df.session_type == "between"]
    if not btwn.empty and "mean_brier_c1" in btwn.columns:
        baseline = (btwn.groupby("model_key")["mean_brier_c1"]
                    .mean().sort_values().round(4))
        for m, v in baseline.items():
            bar = "▓" * int((1 - v) * 20)
            w(f"  {m:<22}  brier={v:.4f}  {bar}")

    # Behavioral shift: CC vs KCS
    w("\n── Behavioral shift: calibrated vs stable policy ──────────────────")
    w("  CC = calibrated_confidence (target: low brier_c1)")
    w("  KCS = keep_confidence_stable (target: low inconsistency_abs)\n")
    for (sess, pert), grp in contrasts.groupby(["session_type", "perturbation"]):
        w(f"  [{sess}/{pert}]  intention(CC−KCS)  brier_CC  brier_KCS")
        for _, r in grp.sort_values("intention_contrast", ascending=False).iterrows():
            ic   = f"{r.intention_contrast:+.3f}" if pd.notna(r.intention_contrast) else "   nan"
            bc   = f"{r.mean_brier_CC:.4f}" if pd.notna(r.mean_brier_CC) else "  nan"
            bkcs = f"{r.mean_brier_KCS:.4f}" if pd.notna(r.mean_brier_KCS) else "  nan"
            w(f"    {r.model_key:<22}  {ic}  {bc}  {bkcs}")
        ic_mean = grp.intention_contrast.mean()
        w(f"    {'── pooled ──':<22}  {ic_mean:+.3f}")
        w("")

    # Key correlations: intention → align_score
    w("\n── SR → Behavior (intention → align_score) ─────────────────────────")
    w("  align_score = 1-brier for CC, 1-inconsistency for KCS")
    w("  Positive r: higher SR intention → behavior consistent with policy\n")
    key_corr = corr[
        (corr.sr_construct == "intention_mean") &
        (corr.beh_outcome == "align_score") &
        (corr.framework == "tpb")
    ].copy()
    for _, r in key_corr.sort_values(["session_type", "perturbation", "policy_id"]).iterrows():
        stars = ("***" if r.p < 0.001 else "**" if r.p < 0.01
                 else "*" if r.p < 0.05 else "   ")
        w(f"  {r.session_type:<8} {r.perturbation:<9} {r.policy_id:<25}  "
          f"r={r.r:+.3f}  p={r.p:.3f}{stars}  n={r.n}")

    # All TPB constructs
    w("\n── All TPB constructs → align_score ────────────────────────────────")
    all_tpb = corr[
        (corr.framework == "tpb") &
        (corr.beh_outcome == "align_score") &
        (corr.sr_construct.isin(TPB_CONSTRUCTS))
    ].copy()
    for (sess, pert, pol), grp in all_tpb.groupby(
            ["session_type", "perturbation", "policy_id"]):
        w(f"  [{sess}/{pert}/{pol}]")
        for _, r in grp.sort_values("r", key=abs, ascending=False).iterrows():
            stars = ("***" if r.p < 0.001 else "**" if r.p < 0.01
                     else "*" if r.p < 0.05 else "   ")
            w(f"    {r.sr_construct:<30}  r={r.r:+.3f}  p={r.p:.3f}{stars}  n={r.n}")

    # Big5 → mean_brier_c1
    w("\n── Big5 traits → mean_brier_c1 (lower=better calibrated) ──────────")
    big5_corr = corr[
        (corr.framework == "big5") &
        (corr.beh_outcome == "mean_brier_c1") &
        (corr.sr_construct.isin(BIG5_TRAITS))
    ].copy()
    for (sess, pert), grp in big5_corr.groupby(["session_type", "perturbation"]):
        w(f"  [{sess}/{pert}]")
        for _, r in grp.sort_values("r", key=abs, ascending=False).iterrows():
            stars = ("***" if r.p < 0.001 else "**" if r.p < 0.01
                     else "*" if r.p < 0.05 else "   ")
            w(f"    {r.sr_construct:<30}  r={r.r:+.3f}  p={r.p:.3f}{stars}  n={r.n}")

    # RQ3: within vs between
    w("\n── RQ3: Within vs Between ─────────────────────────────────────────")
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

    w("\n" + "=" * 70)
    text = "\n".join(lines)
    print(text)
    out_txt.write_text(text, encoding="utf-8")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--out_dir",      default="results/analysis/honesty")
    args = ap.parse_args()

    root    = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Honesty data sources...")
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
    for c in ["align_score", "mean_brier_c1", "mean_inconsistency_abs"] + \
              TPB_CONSTRUCTS + BIG5_TRAITS:
        if c in master.columns:
            master[c] = pd.to_numeric(master[c], errors="coerce")

    master_path = out_dir / "honesty_master.csv"
    master.to_csv(master_path, index=False)
    print(f"\nMaster file: {master_path} ({len(master)} rows)")

    print("Computing correlations...")
    corr = compute_correlations(master)
    corr.to_csv(out_dir / "honesty_correlations.csv", index=False)
    print(f"  Correlations: {len(corr)} rows → honesty_correlations.csv")

    print("Computing policy contrasts...")
    contrasts = compute_policy_contrasts(master)
    contrasts.to_csv(out_dir / "honesty_contrasts.csv", index=False)
    print(f"  Contrasts: {len(contrasts)} rows → honesty_contrasts.csv")

    print("\nGenerating summary...")
    print_summary(master, corr, contrasts, out_dir / "honesty_summary.txt")
    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()