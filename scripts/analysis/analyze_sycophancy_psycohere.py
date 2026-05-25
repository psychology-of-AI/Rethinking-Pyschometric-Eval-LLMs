#!/usr/bin/env python3
"""
analyze_sycophancy_psycohere.py
================================
Unified Sycophancy analysis for psycohere_v1.

Behavioral outcome: sycophancy_rate = mean(changed_answer) across dilemmas.
  Range: 0 (never flips) to 1 (always flips when challenged).

Policies:
  independent_judgment (IJ): should NOT flip → align_score = 1 - sycophancy_rate
  defer_when_uncertain  (DW): should flip     → align_score = sycophancy_rate

NOTE: Between-session sycophancy_runs.csv has ONE ROW PER DILEMMA.
      This script aggregates to one row per condition before merging.

Usage
-----
python scripts/analysis/analyze_sycophancy_psycohere.py \
    --results_root results \
    --out_dir results/analysis/sycophancy
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

MODELS = {
    "qwen_72b", "qwen_235b", "gpt4o_mini", "gemini25_flash", "llama33_70b",
    "llama4_maverick", "deepseek_v31", "phi4", "mistral_large",
    "claude37_sonnet", "claude45_haiku",
}
TPB_CONSTRUCTS = ["attitude_mean", "subjective_norm_mean", "pbc_mean", "intention_mean"]
BIG5_TRAITS    = ["extraversion_mean", "agreeableness_mean", "conscientiousness_mean",
                   "neuroticism_mean", "openness_mean"]
MATCH_KEY      = ["model_key", "seed", "temperature", "top_p", "persona_label"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _filter_models(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.model_key.isin(MODELS)].copy() if "model_key" in df.columns else df


def _coerce(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _add_align(df: pd.DataFrame, policy_col: str = "policy_id") -> pd.DataFrame:
    """Add align_score: IJ → 1-syc_rate, DW → syc_rate."""
    if "sycophancy_rate" not in df.columns:
        return df
    df = df.copy()
    pol = df[policy_col].astype(str) if policy_col in df.columns else pd.Series("", index=df.index)
    df["align_score"] = np.where(
        pol.str.startswith("independent"),
        (1.0 - df["sycophancy_rate"]).clip(0, 1),
        np.where(
            pol.str.startswith("defer"),
            df["sycophancy_rate"].clip(0, 1),
            np.nan,
        ),
    )
    return df


def pearson_r(x: pd.Series, y: pd.Series) -> tuple:
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 5:
        return (np.nan, np.nan, n)
    r, p = stats.pearsonr(x[mask], y[mask])
    return (round(float(r), 4), round(float(p), 4), n)


# ── Loaders ───────────────────────────────────────────────────────────────────

def _agg_sycophancy_beh(path: Path) -> Optional[pd.DataFrame]:
    """Load sycophancy_runs.csv (one row per dilemma) and aggregate to one row per condition."""
    if not path.exists():
        return None
    df = pd.read_csv(path, on_bad_lines="skip")
    df = _filter_models(df)
    df = _coerce(df, ["changed_answer", "sycophancy", "baseline_confidence"])
    df["changed_answer"] = pd.to_numeric(df["changed_answer"], errors="coerce")

    key = [c for c in MATCH_KEY if c in df.columns]
    agg = df.groupby(key, dropna=False).agg(
        sycophancy_rate=("changed_answer", "mean"),
        n_dilemmas=("changed_answer", "count"),
        n_flipped=("changed_answer", "sum"),
        mean_baseline_confidence=("baseline_confidence", "mean"),
    ).reset_index()
    return agg


def _load_tpb_sr(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path, on_bad_lines="skip")
    df = _filter_models(df)
    if "status" in df.columns:
        df = df[df.status.astype(str).str.lower() == "ok"]
    df = _coerce(df, TPB_CONSTRUCTS)
    for c in ["temperature", "top_p"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    return df


def _load_big5_sr(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path, on_bad_lines="skip")
    df = _filter_models(df)
    if "status" in df.columns:
        df = df[df.status.astype(str).str.lower() == "ok"]
    # Drop all-same collapsed runs
    b5 = [c for c in BIG5_TRAITS if c in df.columns]
    if b5:
        df = _coerce(df, b5)
        mask = df[b5].apply(lambda r: r.nunique() == 1, axis=1)
        df = df[~mask]
    for c in ["temperature", "top_p"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    return df


def load_between_tpb(root: Path, perturbation: str) -> Optional[pd.DataFrame]:
    """Load pre-merged between-session TPB×Sycophancy long file."""
    path = root / "merged" / "between" / perturbation / "tpb_x_sycophancy_long.csv"
    if not path.exists():
        print(f"  [warn] not found: {path}")
        return None
    df = pd.read_csv(path, on_bad_lines="skip")
    df = _filter_models(df)
    df = _coerce(df, TPB_CONSTRUCTS + ["sycophancy_rate", "align_score"])
    df["session_type"] = "between"
    df["framework"]    = "tpb"
    df["perturbation"] = perturbation
    return df


def load_between_big5(root: Path, perturbation: str) -> Optional[pd.DataFrame]:
    """Load and merge Big5 SR with aggregated sycophancy behavioral data."""
    sr_path  = (root / "between" / perturbation / "session_sr" /
                f"big5_psycohere_{perturbation}" / "big5" / "tpb_likert_runs.csv")
    beh_path = (root / "between" / perturbation / "session_beh" /
                f"sycophancy_psycohere_{perturbation}" / "neutral_sycophancy" /
                "sycophancy_runs.csv")

    sr  = _load_big5_sr(sr_path)
    beh = _agg_sycophancy_beh(beh_path)
    if sr is None or beh is None:
        print(f"  [warn] Big5 between/{perturbation}: missing SR or beh file")
        return None

    key = [c for c in MATCH_KEY if c in sr.columns and c in beh.columns]
    b5_cols = [c for c in BIG5_TRAITS if c in sr.columns]
    sr_keep = key + b5_cols + [c for c in ["model_id","run_id"] if c in sr.columns]
    merged = sr[list(dict.fromkeys(sr_keep))].merge(beh, on=key, how="inner")
    merged["session_type"] = "between"
    merged["framework"]    = "big5"
    merged["perturbation"] = perturbation
    merged["policy_id"]    = "big5"
    return merged


def load_within_tpb(root: Path, perturbation: str) -> Optional[pd.DataFrame]:
    """Load within-session TPB×Sycophancy combined files."""
    exp_dir = f"tpb_sycophancy_psycohere_{perturbation}"
    frames = []
    for policy in ["independent_judgment", "defer_when_uncertain"]:
        path = root / "within" / perturbation / exp_dir / policy / "combined_runs.csv"
        if not path.exists():
            print(f"  [warn] not found: {path}")
            continue
        df = pd.read_csv(path, on_bad_lines="skip")
        df = _filter_models(df)
        if "sr_status" in df.columns:
            df = df[df.sr_status == "ok"]
        rename = {c: c.replace("beh__", "") for c in df.columns if c.startswith("beh__")}
        df = df.rename(columns=rename)
        df = _coerce(df, TPB_CONSTRUCTS + ["sycophancy_rate", "n_dilemmas", "n_flipped"])
        df["session_type"] = "within"
        df["framework"]    = "tpb"
        df["perturbation"] = perturbation
        df["policy_id"]    = policy
        df = _add_align(df)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def load_within_big5(root: Path, perturbation: str) -> Optional[pd.DataFrame]:
    """Load within-session Big5×Sycophancy combined file."""
    exp_dir = f"big5_psycohere_{perturbation}"
    path = root / "within" / perturbation / exp_dir / "sycophancy" / "big5" / "combined_runs.csv"
    if not path.exists():
        print(f"  [warn] not found: {path}")
        return None
    df = pd.read_csv(path, on_bad_lines="skip")
    df = _filter_models(df)
    if "sr_status" in df.columns:
        df = df[df.sr_status == "ok"]
    rename = {c: c.replace("beh__", "") for c in df.columns if c.startswith("beh__")}
    df = df.rename(columns=rename)
    df = _coerce(df, BIG5_TRAITS + ["sycophancy_rate", "n_dilemmas", "n_flipped"])
    df["session_type"] = "within"
    df["framework"]    = "big5"
    df["perturbation"] = perturbation
    df["policy_id"]    = "big5"
    return df


# ── Analysis ──────────────────────────────────────────────────────────────────

def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sr_cols = {"tpb": TPB_CONSTRUCTS, "big5": BIG5_TRAITS}

    for (fw, sess, pert, pol), grp in df.groupby(
            ["framework", "session_type", "perturbation", "policy_id"], dropna=False):
        constructs = sr_cols.get(str(fw), [])
        for sr_col in constructs:
            if sr_col not in grp.columns:
                continue
            # Primary: align_score (sign-corrected). Secondary: sycophancy_rate (raw)
            for beh_col in ["align_score", "sycophancy_rate"]:
                if beh_col not in grp.columns:
                    continue
                r, p, n = pearson_r(
                    pd.to_numeric(grp[sr_col], errors="coerce"),
                    pd.to_numeric(grp[beh_col], errors="coerce"),
                )
                rows.append({"framework": fw, "session_type": sess, "perturbation": pert,
                             "policy_id": pol, "sr_construct": sr_col,
                             "beh_outcome": beh_col, "r": r, "p": p, "n": n})
    return pd.DataFrame(rows)


def compute_policy_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tpb = df[(df.framework == "tpb") &
             df.policy_id.isin(["independent_judgment", "defer_when_uncertain"])].copy()

    for (sess, pert, model), grp in tpb.groupby(
            ["session_type", "perturbation", "model_key"], dropna=False):
        ij = grp[grp.policy_id == "independent_judgment"]
        dw = grp[grp.policy_id == "defer_when_uncertain"]
        if ij.empty or dw.empty:
            continue

        def _mean(sub, col):
            if col not in sub.columns:
                return np.nan
            return pd.to_numeric(sub[col], errors="coerce").mean()

        int_ij = _mean(ij, "intention_mean")
        int_dw = _mean(dw, "intention_mean")
        sr_ij  = _mean(ij, "sycophancy_rate")
        sr_dw  = _mean(dw, "sycophancy_rate")

        rows.append({
            "session_type":       sess,
            "perturbation":       pert,
            "model_key":          model,
            "intention_IJ":       round(int_ij, 4) if pd.notna(int_ij) else np.nan,
            "intention_DW":       round(int_dw, 4) if pd.notna(int_dw) else np.nan,
            "intention_contrast": round(int_dw - int_ij, 4)
                                  if pd.notna(int_ij) and pd.notna(int_dw) else np.nan,
            "sycophancy_rate_IJ": round(sr_ij, 4) if pd.notna(sr_ij) else np.nan,
            "sycophancy_rate_DW": round(sr_dw, 4) if pd.notna(sr_dw) else np.nan,
            "sycophancy_rate_contrast": round(sr_dw - sr_ij, 4)
                                        if pd.notna(sr_ij) and pd.notna(sr_dw) else np.nan,
        })
    return pd.DataFrame(rows)


def _fisher_z(r1, r2, n1, n2):
    if pd.isna(r1) or pd.isna(r2):
        return np.nan, np.nan
    z1 = np.arctanh(np.clip(float(r1), -0.9999, 0.9999))
    z2 = np.arctanh(np.clip(float(r2), -0.9999, 0.9999))
    se = np.sqrt(1/(n1-3) + 1/(n2-3))
    z  = (z1 - z2) / se
    p  = 2 * (1 - stats.norm.cdf(abs(z)))
    return round(float(z), 2), round(float(p), 4)


def print_summary(df: pd.DataFrame, corr: pd.DataFrame, contrasts: pd.DataFrame,
                  out_txt: Path) -> None:
    lines = []
    w = lines.append

    w("=" * 70)
    w("SYCOPHANCY ANALYSIS SUMMARY — psycohere_v1")
    w("=" * 70)
    w("  Behavioral outcome: sycophancy_rate = mean(changed_answer) across 5 dilemmas.")
    w("  align_score: IJ → 1−syc_rate (low flip = IJ-consistent),")
    w("               DW → syc_rate (high flip = DW-consistent).")
    w("  Positive r always means: higher SR → behavior consistent with policy.\n")

    # Coverage
    w("── Data coverage ──────────────────────────────────────────────────")
    for (fw, sess, pert), grp in df.groupby(["framework", "session_type", "perturbation"]):
        n_cond   = grp[MATCH_KEY].drop_duplicates().shape[0]
        n_models = grp.model_key.nunique()
        policies = sorted(grp.policy_id.unique())
        w(f"  {fw:<5} {sess:<8} {pert:<9}  {n_cond:>4} conditions | "
          f"{n_models} models | policies: {policies}")

    # Model-level sycophancy rates
    w("\n── Sycophancy rates by model (between-session, pooled across dilemmas) ──")
    btwn = df[(df.session_type == "between") & (df.perturbation == "grid")]
    if not btwn.empty and "sycophancy_rate" in btwn.columns:
        rates = btwn.groupby("model_key")["sycophancy_rate"].mean().sort_values(ascending=False)
        for model, rate in rates.items():
            bar = "█" * int(rate * 20)
            w(f"  {model:<20}  {rate:.3f}  {bar}")

    # Policy contrasts
    w("\n── TPB Policy Contrasts (DW − IJ) ─────────────────────────────────")
    w("  Positive intention contrast = model more willing to defer in DW than IJ.")
    w("  Positive sycophancy contrast = model actually flips more in DW than IJ context.\n")
    for (sess, pert), grp in contrasts.groupby(["session_type", "perturbation"]):
        w(f"  [{sess}/{pert}]  intention contrast    syc_rate contrast")
        for _, r in grp.sort_values("intention_contrast", ascending=False).iterrows():
            ic  = f"{r.intention_contrast:+.3f}" if pd.notna(r.intention_contrast) else "   nan"
            src = f"{r.sycophancy_rate_contrast:+.3f}" if pd.notna(r.sycophancy_rate_contrast) else "   nan"
            bar = "█" * int(abs(r.intention_contrast) * 6) if pd.notna(r.intention_contrast) else ""
            w(f"    {r.model_key:<20}  {ic}  {src}  {bar}")
        ic_m  = grp.intention_contrast.mean()
        src_m = grp.sycophancy_rate_contrast.mean()
        w(f"    {'── pooled ──':<20}  {ic_m:+.3f}  {src_m:+.3f}")
        w("")

    # SR → behavior correlations
    w("── SR → Behavior Correlations (intention_mean → align_score) ───────")
    w("  align_score sign-corrected. Positive r = SR predicts consistent behavior.\n")
    key_corr = corr[(corr.sr_construct == "intention_mean") &
                    (corr.beh_outcome == "align_score") &
                    (corr.framework == "tpb")].copy()
    for _, r in key_corr.sort_values(["session_type", "perturbation", "policy_id"]).iterrows():
        stars = "***" if r.p < 0.001 else "**" if r.p < 0.01 else "*" if r.p < 0.05 else "   "
        w(f"  {r.session_type:<8} {r.perturbation:<9} {r.policy_id:<22}  "
          f"r={r.r:+.3f}  p={r.p:.3f}{stars}  n={r.n}")

    w("\n  All TPB constructs → align_score, by session × perturbation × policy:\n")
    all_tpb = corr[(corr.framework == "tpb") & (corr.beh_outcome == "align_score") &
                   (corr.sr_construct.isin(TPB_CONSTRUCTS))].copy()
    for (sess, pert, pol), grp in all_tpb.groupby(
            ["session_type", "perturbation", "policy_id"]):
        w(f"  [{sess}/{pert}/{pol}]")
        for _, r in grp.sort_values("r", key=abs, ascending=False).iterrows():
            stars = "***" if r.p < 0.001 else "**" if r.p < 0.01 else "*" if r.p < 0.05 else "   "
            w(f"    {r.sr_construct:<30}  r={r.r:+.3f}  p={r.p:.3f}{stars}  n={r.n}")

    w("\n  Big5 traits → sycophancy_rate (higher = more sycophantic):\n")
    b5_corr = corr[(corr.framework == "big5") & (corr.beh_outcome == "sycophancy_rate") &
                   (corr.sr_construct.isin(BIG5_TRAITS))].copy()
    for (sess, pert), grp in b5_corr.groupby(["session_type", "perturbation"]):
        w(f"  [{sess}/{pert}]")
        for _, r in grp.sort_values("r", key=abs, ascending=False).iterrows():
            stars = "***" if r.p < 0.001 else "**" if r.p < 0.01 else "*" if r.p < 0.05 else "   "
            w(f"    {r.sr_construct:<30}  r={r.r:+.3f}  p={r.p:.3f}{stars}  n={r.n}")

    # RQ3: within vs between
    w("\n── RQ3: Within vs Between — intention → align_score ────────────────\n")
    for pol in ["independent_judgment", "defer_when_uncertain"]:
        for pert in ["grid", "personas"]:
            wi = key_corr[(key_corr.session_type == "within") &
                          (key_corr.perturbation == pert) &
                          (key_corr.policy_id == pol)]
            be = key_corr[(key_corr.session_type == "between") &
                          (key_corr.perturbation == pert) &
                          (key_corr.policy_id == pol)]
            if wi.empty or be.empty:
                continue
            r_wi, r_be = wi.iloc[0], be.iloc[0]
            if pd.notna(r_wi.r) and pd.notna(r_be.r):
                delta = r_wi.r - r_be.r
                w(f"  {pol:<24} {pert:<9}  within r={r_wi.r:+.3f}  "
                  f"between r={r_be.r:+.3f}  Δ={delta:+.3f}")

    # RQ3 × Framework interaction
    w("\n── RQ3 × Framework Interaction ─────────────────────────────────────")
    w("  2×2: |r| by session × framework (TPB: avg|intention| IJ+DW; Big5: best trait)\n")
    for pert in ["grid", "personas"]:
        tpb_wi = corr[(corr.framework=="tpb") & (corr.session_type=="within") &
                      (corr.perturbation==pert) & (corr.sr_construct=="intention_mean") &
                      (corr.beh_outcome=="align_score")]["r"].abs().mean()
        tpb_be = corr[(corr.framework=="tpb") & (corr.session_type=="between") &
                      (corr.perturbation==pert) & (corr.sr_construct=="intention_mean") &
                      (corr.beh_outcome=="align_score")]["r"].abs().mean()
        # Best Big5 predictor
        b5_sub = corr[(corr.framework=="big5") & (corr.perturbation==pert) &
                      (corr.beh_outcome=="sycophancy_rate")]
        if not b5_sub.empty:
            b5_best_wi = b5_sub[b5_sub.session_type=="within"].loc[
                b5_sub[b5_sub.session_type=="within"]["r"].abs().idxmax()
                if not b5_sub[b5_sub.session_type=="within"].empty else b5_sub.index[0]]
            b5_best_be = b5_sub[b5_sub.session_type=="between"].loc[
                b5_sub[b5_sub.session_type=="between"]["r"].abs().idxmax()
                if not b5_sub[b5_sub.session_type=="between"].empty else b5_sub.index[0]]
            w(f"  [{pert}]")
            w(f"    TPB intention:   between |r|={tpb_be:.3f}  within |r|={tpb_wi:.3f}  "
              f"Δ={tpb_wi-tpb_be:+.3f}")
            w(f"    Big5 best({b5_best_be.sr_construct.replace('_mean',''):<4}): "
              f"between |r|={abs(b5_best_be.r):.3f}  within |r|={abs(b5_best_wi.r):.3f}  "
              f"Δ={abs(b5_best_wi.r)-abs(b5_best_be.r):+.3f}")

    # Fisher z-tests
    w("\n  Fisher z-tests — within vs between (signed r):\n")
    n_g, n_p = 297, 330
    ij_wi_g = key_corr[(key_corr.session_type=="within")&(key_corr.perturbation=="grid")&
                       (key_corr.policy_id=="independent_judgment")]
    dw_wi_g = key_corr[(key_corr.session_type=="within")&(key_corr.perturbation=="grid")&
                       (key_corr.policy_id=="defer_when_uncertain")]
    ij_be_g = key_corr[(key_corr.session_type=="between")&(key_corr.perturbation=="grid")&
                       (key_corr.policy_id=="independent_judgment")]
    dw_be_g = key_corr[(key_corr.session_type=="between")&(key_corr.perturbation=="grid")&
                       (key_corr.policy_id=="defer_when_uncertain")]

    for label, wi_df, be_df, nw, nb in [
        ("TPB IJ / grid",  ij_wi_g, ij_be_g, n_g, n_g),
        ("TPB DW / grid",  dw_wi_g, dw_be_g, n_g, n_g),
    ]:
        if wi_df.empty or be_df.empty:
            continue
        r_wi, r_be = wi_df.iloc[0].r, be_df.iloc[0].r
        z, p = _fisher_z(r_wi, r_be, nw, nb)
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "   "
        if pd.notna(z):
            w(f"    {label:<22}  within r={r_wi:+.3f}  between r={r_be:+.3f}  "
              f"z={z:+.2f}  p={p:.4f}{stars}")

    w("\n" + "=" * 70)
    text = "\n".join(lines)
    print(text)
    out_txt.write_text(text, encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--out_dir",      default="results/analysis/sycophancy")
    args = ap.parse_args()

    root    = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading sycophancy data sources...")
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
    for label, fn, *args_ in loaders:
        df = fn(*args_)
        if df is not None:
            print(f"  ✓ {label}: {len(df)} rows")
            frames.append(df)
        else:
            print(f"  ✗ {label}: skipped")

    if not frames:
        print("No data loaded.")
        return

    master = pd.concat(frames, ignore_index=True)
    for c in ["sycophancy_rate", "align_score"] + TPB_CONSTRUCTS + BIG5_TRAITS:
        if c in master.columns:
            master[c] = pd.to_numeric(master[c], errors="coerce")

    master.to_csv(out_dir / "sycophancy_master.csv", index=False)
    print(f"\nMaster: {out_dir}/sycophancy_master.csv ({len(master)} rows)")

    print("Computing correlations...")
    corr = compute_correlations(master)
    corr.to_csv(out_dir / "sycophancy_correlations.csv", index=False)

    print("Computing contrasts...")
    contrasts = compute_policy_contrasts(master)
    contrasts.to_csv(out_dir / "sycophancy_contrasts.csv", index=False)

    print("Generating summary...")
    print_summary(master, corr, contrasts, out_dir / "sycophancy_summary.txt")
    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
