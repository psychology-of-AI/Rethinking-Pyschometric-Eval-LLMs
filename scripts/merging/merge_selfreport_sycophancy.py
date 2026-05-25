#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_selfreport_sycophancy.py — Instrument-agnostic self-report × Sycophancy merger (v1)
==========================================================================================
Merges a self-report likert_runs CSV (Big5, TPB, …) with a sycophancy_runs.csv
on the grid match key:
    (model_key, seed, temperature, top_p, persona_label)

sycophancy_runs.csv has one row per (condition × dilemma_id). This script
aggregates per-question rows to condition level before joining, mirroring the
aggregation logic in merge_selfreport_honesty.py.

Output columns (condition level):
  n_questions              — number of scored dilemmas
  sycophancy_rate          — proportion of dilemmas where the model flipped (mean of changed_answer)
  mean_sycophancy          — mean continuous sycophancy score (if column present)
  mean_baseline_confidence — mean pre-pushback confidence

Usage
-----
# Big5 × Sycophancy (grid)
python scripts/merging_scripts/merge_selfreport_sycophancy.py \\
    --selfreport_csv results/psycohere_v1/between/grid/session_sr/big5_psycohere_grid/big5/tpb_likert_runs.csv \\
    --instrument big5 \\
    --sycophancy_runs_csv results/psycohere_v1/between/grid/session_beh/sycophancy_psycohere_grid/neutral_sycophancy/sycophancy_runs.csv \\
    --out_csv results/psycohere_v1/merged/between/grid/big5_x_sycophancy.csv

# Big5 × Sycophancy (personas)
python scripts/merging_scripts/merge_selfreport_sycophancy.py \\
    --selfreport_csv results/psycohere_v1/between/personas/session_sr/big5_psycohere_personas/big5/tpb_likert_runs.csv \\
    --instrument big5 \\
    --sycophancy_runs_csv results/psycohere_v1/between/personas/session_beh/sycophancy_psycohere_personas/neutral_sycophancy/sycophancy_runs.csv \\
    --out_csv results/psycohere_v1/merged/between/personas/big5_x_sycophancy.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MATCH_KEY = ["model_key", "seed", "temperature", "top_p", "persona_label"]

INSTRUMENT_COL = "gen_extra__prompt_variant"
STATUS_COL = "status"

_SR_EXCLUDE = {
    "mean_k", "loss_rate", "total_payoff", "total_expected_payoff",
    "prop_max_flips", "n_rounds", "max_flips",
}

# Sycophancy per-question columns → condition-level aggregations
# (output_name, source_col, agg_fn)
SYCOPHANCY_AGG_SPEC = [
    ("n_questions",              "dilemma_id",         "count"),
    ("sycophancy_rate",          "changed_answer",     "mean"),
    ("mean_sycophancy",          "sycophancy",         "mean"),
    ("mean_baseline_confidence", "baseline_confidence", "mean"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_subscale_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.endswith("_mean") and c not in _SR_EXCLUDE]


def _is_allsame(row: pd.Series) -> bool:
    return bool(row.nunique() == 1)


def _coerce_key_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    for c in ["temperature", "top_p"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(6)
    return df


def _infer_from_path(path: str, kind: str) -> Optional[str]:
    """Infer session_type or perturbation from a file path.
    kind: 'session_type' ('between' | 'within') or 'perturbation' ('grid' | 'personas').
    Returns None if no unambiguous match.
    """
    p = str(path).replace("\\", "/").lower()
    if kind == "session_type":
        for token in ["/between/", "/within/"]:
            if token in p:
                return token.strip("/")
    elif kind == "perturbation":
        for token in ["/grid/", "/personas/"]:
            if token in p:
                return token.strip("/")
    return None


def _load_selfreport(
    path: str,
    instrument: Optional[str],
    exclude_models: List[str],
    drop_allsame: bool,
    status_ok_only: bool,
) -> pd.DataFrame:
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    n_raw = len(df)

    if instrument and INSTRUMENT_COL in df.columns:
        available = sorted(df[INSTRUMENT_COL].dropna().astype(str).unique().tolist())
        df = df[df[INSTRUMENT_COL].astype(str).str.strip() == instrument].copy()
        print(f"[selfreport] instrument={instrument!r}: {len(df)}/{n_raw} rows")
        if len(df) == 0:
            print(f"[selfreport] ERROR: no rows matched instrument={instrument!r}")
            print(f"[selfreport] Available variants: {available}")
            sys.exit(1)
    elif instrument:
        print(f"[selfreport] WARNING: column '{INSTRUMENT_COL}' not found")

    if status_ok_only and STATUS_COL in df.columns:
        before = len(df)
        df = df[df[STATUS_COL].astype(str).str.lower() == "ok"].copy()
        print(f"[selfreport] status=ok: {len(df)}/{before} rows kept")
        if len(df) == 0:
            print("[selfreport] ERROR: no ok rows remain")
            sys.exit(1)

    subscale_cols = _detect_subscale_cols(df)
    if subscale_cols:
        print(f"[selfreport] subscale cols: {subscale_cols}")
        for c in subscale_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if drop_allsame and len(df) > 0:
            mask = df[subscale_cols].apply(_is_allsame, axis=1)
            n_drop = int(mask.sum())
            df = df[~mask].copy()
            if n_drop:
                print(f"[selfreport] dropped {n_drop} all-same collapsed runs")
    else:
        print("[selfreport] WARNING: no *_mean subscale columns detected")

    if exclude_models and "model_key" in df.columns:
        before = len(df)
        df = df[~df["model_key"].isin(exclude_models)].copy()
        print(f"[selfreport] excluded {exclude_models}: {len(df)}/{before} rows kept")

    return _coerce_key_types(df)


def _load_and_aggregate_sycophancy(
    path: str,
    exclude_models: List[str],
    status_ok_only: bool,
) -> pd.DataFrame:
    """Load sycophancy_runs.csv (per-question rows) and aggregate to condition level."""
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    n_raw = len(df)
    print(f"[sycophancy] loaded {n_raw} per-question rows from {path}")

    if status_ok_only and STATUS_COL in df.columns:
        before = len(df)
        df = df[df[STATUS_COL].astype(str).str.lower() == "ok"].copy()
        print(f"[sycophancy] status=ok: {len(df)}/{before} rows kept")

    if exclude_models and "model_key" in df.columns:
        before = len(df)
        df = df[~df["model_key"].isin(exclude_models)].copy()
        print(f"[sycophancy] excluded {exclude_models}: {before - len(df)} rows dropped")

    df = _coerce_key_types(df)

    # Coerce numeric outcome cols (skip dilemma_id — used only for counting)
    for _, src_col, _ in SYCOPHANCY_AGG_SPEC:
        if src_col in df.columns and src_col != "dilemma_id":
            df[src_col] = pd.to_numeric(df[src_col], errors="coerce")

    # Aggregate per-question → per-condition
    grp_key = [k for k in MATCH_KEY if k in df.columns]
    agg_dict = {}
    rename_map = {}
    for out_name, src_col, agg_fn in SYCOPHANCY_AGG_SPEC:
        if src_col in df.columns:
            agg_dict[src_col] = agg_fn
            rename_map[src_col] = out_name
        else:
            print(f"[sycophancy] note: source col '{src_col}' not present — skipping '{out_name}'")

    if not agg_dict:
        print("[sycophancy] ERROR: none of the expected outcome columns found")
        sys.exit(1)

    agg = df.groupby(grp_key, dropna=False).agg(agg_dict).reset_index()
    agg = agg.rename(columns=rename_map)

    print(f"[sycophancy] aggregated to {len(agg)} condition rows")
    print(f"[sycophancy] models: {sorted(agg.model_key.unique())}")
    if "sycophancy_rate" in agg.columns:
        print("[sycophancy] sycophancy_rate by model:")
        print(agg.groupby("model_key")["sycophancy_rate"].mean().round(4)
              .sort_values().to_string())

    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge self-report subscale means with Sycophancy behavioral runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--selfreport_csv", required=True,
                    help="Path to tpb_likert_runs.csv (Big5 or other instrument).")
    ap.add_argument("--instrument", default=None,
                    help="gen_extra__prompt_variant value to filter (e.g. 'big5').")
    ap.add_argument("--sycophancy_runs_csv", required=True,
                    help="Path to sycophancy_runs.csv (per-question rows).")
    ap.add_argument("--out_csv", required=True,
                    help="Output path for merged condition-level CSV.")
    ap.add_argument("--exclude_models", default="llama31_8b",
                    help="Comma-separated model_key values to exclude (default: llama31_8b).")
    ap.add_argument("--drop_allsame", action="store_true", default=True)
    ap.add_argument("--no_drop_allsame", dest="drop_allsame", action="store_false")
    ap.add_argument("--status_ok_only", action="store_true", default=True)
    ap.add_argument("--how", default="inner", choices=["inner", "left", "outer"])

    # ── Schema-completing columns for downstream figure/analysis scripts ──
    # Without these, downstream filters like `df[df.framework == 'big5']` will
    # drop every row from this merger's output.
    ap.add_argument("--session_type", default=None,
                    choices=["between", "within"],
                    help="Session type tag written to every output row. "
                         "Auto-inferred from --selfreport_csv path if omitted.")
    ap.add_argument("--perturbation", default=None,
                    choices=["grid", "personas"],
                    help="Perturbation tag written to every output row. "
                         "Auto-inferred from --selfreport_csv path if omitted.")
    ap.add_argument("--framework", default=None,
                    help="Framework tag written to every output row. "
                         "Defaults to --instrument value (e.g. 'big5').")
    ap.add_argument("--policy_id", default=None,
                    help="policy_id tag written to every output row. "
                         "Defaults to --framework value (Big5/HEXACO have no "
                         "policy axis, so policy_id = framework name is standard).")
    args = ap.parse_args()

    exclude_models = [m.strip() for m in (args.exclude_models or "").split(",") if m.strip()]

    sr = _load_selfreport(
        path=args.selfreport_csv,
        instrument=args.instrument,
        exclude_models=exclude_models,
        drop_allsame=args.drop_allsame,
        status_ok_only=args.status_ok_only,
    )

    syc = _load_and_aggregate_sycophancy(
        path=args.sycophancy_runs_csv,
        exclude_models=exclude_models,
        status_ok_only=args.status_ok_only,
    )

    merge_keys = [k for k in MATCH_KEY if k in sr.columns and k in syc.columns]
    missing = [k for k in MATCH_KEY if k not in merge_keys]
    if missing:
        print(f"[merge] WARNING: match keys not in both files: {missing}")
    print(f"[merge] merging on: {merge_keys}")

    subscale_cols = _detect_subscale_cols(sr)
    sr_carry = list(dict.fromkeys(
        [c for c in merge_keys + subscale_cols + [INSTRUMENT_COL, "model_id", "run_id"]
         if c in sr.columns]
    ))

    syc_outcome_cols = [out for out, _, _ in SYCOPHANCY_AGG_SPEC if out in syc.columns]
    syc_carry = list(dict.fromkeys(
        [c for c in merge_keys + syc_outcome_cols if c in syc.columns]
    ))

    merged = sr[sr_carry].merge(
        syc[syc_carry],
        on=merge_keys,
        how=args.how,
        suffixes=("__sr", "__syc"),
    )

    # ── Resolve and stamp schema-completing columns ─────────────────────
    framework = args.framework or args.instrument
    if not framework:
        print("[merge] ERROR: --framework not set and --instrument not provided. "
              "Pass --framework explicitly.")
        sys.exit(1)

    session_type = args.session_type or _infer_from_path(args.selfreport_csv, "session_type")
    if not session_type:
        print("[merge] ERROR: could not infer --session_type from path "
              f"({args.selfreport_csv}). Pass --session_type explicitly "
              "('between' or 'within').")
        sys.exit(1)

    perturbation = args.perturbation or _infer_from_path(args.selfreport_csv, "perturbation")
    if not perturbation:
        print("[merge] ERROR: could not infer --perturbation from path "
              f"({args.selfreport_csv}). Pass --perturbation explicitly "
              "('grid' or 'personas').")
        sys.exit(1)

    policy_id = args.policy_id or framework

    merged["framework"]    = framework
    merged["session_type"] = session_type
    merged["perturbation"] = perturbation
    merged["policy_id"]    = policy_id
    print(f"[merge] stamped schema cols: framework={framework!r}, "
          f"session_type={session_type!r}, perturbation={perturbation!r}, "
          f"policy_id={policy_id!r}")

    print(f"[merge] rows in output: {len(merged)}")
    print(f"[merge] model_key distribution:\n"
          f"{merged['model_key'].value_counts().sort_index().to_string()}")

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"[ok] written: {args.out_csv}")

    if subscale_cols and syc_outcome_cols:
        print("\n[summary] Mean trait scores by model:")
        print(merged.groupby("model_key")[subscale_cols].mean().round(3).to_string())
        print("\n[summary] Mean sycophancy outcomes by model:")
        print(merged.groupby("model_key")[syc_outcome_cols].mean().round(4).to_string())


if __name__ == "__main__":
    main()