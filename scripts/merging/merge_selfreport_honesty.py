#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_selfreport_honesty.py — Instrument-agnostic self-report × Honesty merger (v1)
====================================================================================
Merges a self-report likert_runs CSV (Big5, TPB, …) with a honesty_runs.csv on
the grid match key:
    (model_key, seed, temperature, top_p, persona_label)

honesty_runs.csv has one row per (condition × question). This script aggregates
per-question rows to condition level before joining, mirroring the aggregation
logic in merge_selfreport_iat.py.

Output columns (condition level):
  n_questions              — number of scored questions
  accuracy                 — proportion correct (is_correct_em)
  mean_brier_c1            — mean Brier score at first confidence rating
  mean_brier_c2            — mean Brier score at second confidence rating
  mean_brier_improvement   — mean calibration improvement c1 → c2
  mean_confidence_delta    — mean signed confidence change (C2 − C1)
  mean_abs_confidence_delta — mean |C2 − C1|
  mean_inconsistency_abs   — mean absolute inconsistency between C1 and C2

Usage
-----
# Big5 × Honesty (grid)
python scripts/merging_scripts/merge_selfreport_honesty.py \\
    --selfreport_csv results/psycohere_v1/between/grid/session_sr/big5_psycohere_grid/big5/tpb_likert_runs.csv \\
    --instrument big5 \\
    --honesty_runs_csv results/psycohere_v1/between/grid/session_beh/honesty_psycohere_grid/neutral-honesty/honesty_runs.csv \\
    --out_csv results/psycohere_v1/merged/between/grid/big5_x_honesty.csv

# Big5 × Honesty (personas)
python scripts/merging_scripts/merge_selfreport_honesty.py \\
    --selfreport_csv results/psycohere_v1/between/personas/session_sr/big5_psycohere_personas/big5/tpb_likert_runs.csv \\
    --instrument big5 \\
    --honesty_runs_csv results/psycohere_v1/between/personas/session_beh/honesty_psycohere_personas/neutral-honesty/honesty_runs.csv \\
    --out_csv results/psycohere_v1/merged/between/personas/big5_x_honesty.csv
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

# Column in self-report CSV that identifies the instrument variant
INSTRUMENT_COL = "gen_extra__prompt_variant"
STATUS_COL = "status"

# Subscale auto-detection exclusions (same as other merge scripts)
_SR_EXCLUDE = {
    "mean_k", "loss_rate", "total_payoff", "total_expected_payoff",
    "prop_max_flips", "n_rounds", "max_flips",
}

# Honesty per-question columns → condition-level aggregations
# (output_name, source_col, agg_fn)
HONESTY_AGG_SPEC = [
    ("n_questions",               "brier_c1",          "count"),
    ("accuracy",                  "is_correct_em",     "mean"),
    ("mean_brier_c1",             "brier_c1",          "mean"),
    ("mean_brier_c2",             "brier_c2",          "mean"),
    ("mean_brier_improvement",    "brier_improvement", "mean"),
    ("mean_confidence_delta",     "confidence_delta",  "mean"),
    ("mean_inconsistency_abs",    "inconsistency_abs", "mean"),
]

# Corrupt model label written by runner bug — always excluded
_CORRUPT_MODEL_LABEL = "neutral_honesty"


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


def _load_and_aggregate_honesty(
    path: str,
    exclude_models: List[str],
) -> pd.DataFrame:
    """Load honesty_runs.csv (per-question rows) and aggregate to condition level."""
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    n_raw = len(df)
    print(f"[honesty] loaded {n_raw} per-question rows from {path}")

    # Drop corrupt runner-label rows
    if "model_key" in df.columns and _CORRUPT_MODEL_LABEL in df["model_key"].values:
        n_before = len(df)
        df = df[df["model_key"] != _CORRUPT_MODEL_LABEL].copy()
        print(f"[honesty] dropped {n_before - len(df)} corrupt rows "
              f"(model_key='{_CORRUPT_MODEL_LABEL}')")

    if exclude_models and "model_key" in df.columns:
        before = len(df)
        df = df[~df["model_key"].isin(exclude_models)].copy()
        print(f"[honesty] excluded {exclude_models}: {before - len(df)} rows dropped")

    df = _coerce_key_types(df)

    # Coerce numeric outcome cols
    for _, src_col, _ in HONESTY_AGG_SPEC:
        if src_col in df.columns:
            df[src_col] = pd.to_numeric(df[src_col], errors="coerce")

    # Compute abs_confidence_delta if not present
    if "confidence_delta" in df.columns and "abs_confidence_delta" not in df.columns:
        df["abs_confidence_delta"] = df["confidence_delta"].abs()

    # Aggregate per-question → per-condition
    grp_key = [k for k in MATCH_KEY if k in df.columns]
    agg_dict = {}
    rename_map = {}
    for out_name, src_col, agg_fn in HONESTY_AGG_SPEC:
        if src_col in df.columns:
            agg_dict[src_col] = agg_fn
            rename_map[src_col] = out_name

    if "abs_confidence_delta" in df.columns:
        agg_dict["abs_confidence_delta"] = "mean"
        rename_map["abs_confidence_delta"] = "mean_abs_confidence_delta"

    agg = df.groupby(grp_key, dropna=False).agg(agg_dict).reset_index()
    agg = agg.rename(columns=rename_map)

    print(f"[honesty] aggregated to {len(agg)} condition rows")
    print(f"[honesty] models: {sorted(agg.model_key.unique())}")
    if "mean_brier_c1" in agg.columns:
        print("[honesty] mean_brier_c1 by model:")
        print(agg.groupby("model_key")["mean_brier_c1"].mean().round(4)
              .sort_values().to_string())

    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge self-report subscale means with Honesty behavioral runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--selfreport_csv", required=True,
                    help="Path to tpb_likert_runs.csv (Big5 or other instrument).")
    ap.add_argument("--instrument", default=None,
                    help="gen_extra__prompt_variant value to filter (e.g. 'big5').")
    ap.add_argument("--honesty_runs_csv", required=True,
                    help="Path to honesty_runs.csv (per-question rows).")
    ap.add_argument("--out_csv", required=True,
                    help="Output path for merged condition-level CSV.")
    ap.add_argument("--exclude_models", default="llama31_8b",
                    help="Comma-separated model_key values to exclude (default: llama31_8b).")
    ap.add_argument("--drop_allsame", action="store_true", default=True)
    ap.add_argument("--no_drop_allsame", dest="drop_allsame", action="store_false")
    ap.add_argument("--status_ok_only", action="store_true", default=True)
    ap.add_argument("--how", default="inner", choices=["inner", "left", "outer"])
    args = ap.parse_args()

    exclude_models = [m.strip() for m in (args.exclude_models or "").split(",") if m.strip()]

    sr = _load_selfreport(
        path=args.selfreport_csv,
        instrument=args.instrument,
        exclude_models=exclude_models,
        drop_allsame=args.drop_allsame,
        status_ok_only=args.status_ok_only,
    )

    honesty = _load_and_aggregate_honesty(
        path=args.honesty_runs_csv,
        exclude_models=exclude_models,
    )

    merge_keys = [k for k in MATCH_KEY if k in sr.columns and k in honesty.columns]
    missing = [k for k in MATCH_KEY if k not in merge_keys]
    if missing:
        print(f"[merge] WARNING: match keys not in both files: {missing}")
    print(f"[merge] merging on: {merge_keys}")

    subscale_cols = _detect_subscale_cols(sr)
    sr_carry = list(dict.fromkeys(
        [c for c in merge_keys + subscale_cols + [INSTRUMENT_COL, "model_id", "run_id"]
         if c in sr.columns]
    ))

    honesty_outcome_cols = [out for out, _, _ in HONESTY_AGG_SPEC
                            if out in honesty.columns] + ["mean_abs_confidence_delta"]
    honesty_outcome_cols = [c for c in honesty_outcome_cols if c in honesty.columns]
    honesty_carry = list(dict.fromkeys(
        [c for c in merge_keys + honesty_outcome_cols if c in honesty.columns]
    ))

    merged = sr[sr_carry].merge(
        honesty[honesty_carry],
        on=merge_keys,
        how=args.how,
        suffixes=("__sr", "__honesty"),
    )

    print(f"[merge] rows in output: {len(merged)}")
    print(f"[merge] model_key distribution:\n"
          f"{merged['model_key'].value_counts().sort_index().to_string()}")

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"[ok] written: {args.out_csv}")

    if subscale_cols and honesty_outcome_cols:
        print("\n[summary] Mean trait scores by model:")
        print(merged.groupby("model_key")[subscale_cols].mean().round(3).to_string())
        print("\n[summary] Mean honesty outcomes by model:")
        print(merged.groupby("model_key")[honesty_outcome_cols].mean().round(4).to_string())


if __name__ == "__main__":
    main()
