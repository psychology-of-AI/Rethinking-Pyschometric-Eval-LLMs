#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_selfreport_iat.py — Instrument-agnostic self-report × IAT merger (v1)
============================================================================
Merges a self-report likert_runs CSV (BFI, HEXACO, MFQ, PVQ, TPB, …) with an
IAT behavioral runs CSV on the grid match key:
    (model_key, seed, temperature, top_p, persona_label)

Unlike the CCT merger (1 row per condition), IAT has 21 rows per condition
(7 tests × 3 orders). The output keeps this long structure by default, with
optional aggregation over orders and/or tests.

Output row structure
--------------------
Default (--agg none):
    One row per (condition × test × order) — 21 rows per grid cell.
    Use for order-level or test-level analysis.

--agg orders:
    Average bias over 3 orders per test → 7 rows per condition.
    Standard for IAT analysis (order effects averaged out).

--agg tests:
    Average bias over all tests per condition → 1 row per condition.
    Matches CCT output structure; loses test-specific information.

--agg both:
    Average over orders first, then over tests → 1 row per condition.

Usage
-----
# BIG-5 × IAT (default: keep all orders and tests)
python merge_selfreport_iat.py \\
    --selfreport_csv results/tpb_likert_runs.csv \\
    --instrument big5 \\
    --iat_runs_csv results/iat_runs.csv \\
    --out_csv results/merged/big5_x_iat.csv

# Aggregate over orders only (recommended for most analyses)
python merge_selfreport_iat.py \\
    --selfreport_csv results/tpb_likert_runs.csv \\
    --instrument big5 \\
    --iat_runs_csv results/iat_runs.csv \\
    --out_csv results/merged/big5_x_iat_by_test.csv \\
    --agg orders

# Fully aggregated (1 row per condition, matches CCT structure)
python merge_selfreport_iat.py \\
    --selfreport_csv results/tpb_likert_runs.csv \\
    --instrument big5 \\
    --iat_runs_csv results/iat_runs.csv \\
    --out_csv results/merged/big5_x_iat_flat.csv \\
    --agg both

# Filter to specific tests
python merge_selfreport_iat.py \\
    --selfreport_csv results/tpb_likert_runs.csv \\
    --instrument big5 \\
    --iat_runs_csv results/iat_runs.csv \\
    --out_csv results/merged/big5_x_iat_gender.csv \\
    --tests gender_science gender_career \\
    --agg orders
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

# IAT outcome columns to carry from iat_runs.csv
IAT_OUTCOME_COLS = [
    "bias", "coverage", "label_rate",
    "n_recognized_attr", "n_recognized_label", "n_pairs_parsed",
    "n_sa_xa", "n_sa_xb", "n_sb_xa", "n_sb_xb",
]

# IAT dimension columns (not outcomes but needed for grouping/filtering)
IAT_DIM_COLS = ["test_id", "order_id", "category", "dataset", "sa_label", "sb_label"]

# Self-report subscale auto-detection exclusions
_SR_EXCLUDE = {
    "mean_k", "loss_rate", "total_payoff", "total_expected_payoff",
    "prop_max_flips", "n_rounds", "max_flips",
}

INSTRUMENT_COL = "gen_extra__prompt_variant"
STATUS_COL = "status"


# ---------------------------------------------------------------------------
# Helpers (shared with CCT merger)
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
            print("[selfreport] ERROR: no ok rows remain"); sys.exit(1)

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


def _load_iat(
    path: str,
    exclude_models: List[str],
    tests: Optional[List[str]],
    min_coverage: float,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    n_raw = len(df)

    if exclude_models and "model_key" in df.columns:
        df = df[~df["model_key"].isin(exclude_models)].copy()
        print(f"[iat] excluded {exclude_models}: {len(df)}/{n_raw} rows kept")

    if tests:
        before = len(df)
        df = df[df["test_id"].isin(tests)].copy()
        print(f"[iat] filtered to tests={tests}: {len(df)}/{before} rows kept")

    if "coverage" in df.columns and min_coverage > 0:
        df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce")
        before = len(df)
        df = df[df["coverage"] >= min_coverage].copy()
        n_drop = before - len(df)
        if n_drop:
            print(f"[iat] dropped {n_drop} low-coverage runs (< {min_coverage})")

    for c in IAT_OUTCOME_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return _coerce_key_types(df)


def _aggregate_iat(df: pd.DataFrame, agg: str, subscale_cols: List[str]) -> pd.DataFrame:
    """Aggregate bias over orders and/or tests."""
    if agg == "none":
        return df

    # Numeric cols to aggregate
    agg_cols = [c for c in IAT_OUTCOME_COLS if c in df.columns]

    if agg in ("orders", "both"):
        # Average over order_id within each (condition × test)
        group_keys = MATCH_KEY + subscale_cols + ["test_id", "category", "dataset"]
        group_keys = [k for k in group_keys if k in df.columns]
        df = df.groupby(group_keys, dropna=False)[agg_cols].mean().reset_index()
        print(f"[agg] averaged over orders → {len(df)} rows")

    if agg in ("tests", "both"):
        # Average over test_id within each condition
        group_keys = MATCH_KEY + subscale_cols
        group_keys = [k for k in group_keys if k in df.columns]
        df = df.groupby(group_keys, dropna=False)[agg_cols].mean().reset_index()
        print(f"[agg] averaged over tests → {len(df)} rows")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge self-report subscale means with IAT behavioral runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--selfreport_csv", required=True)
    ap.add_argument("--instrument", default=None,
                    help="gen_extra__prompt_variant value to filter (e.g. 'big5', 'hexaco24').")
    ap.add_argument("--iat_runs_csv", required=True,
                    help="Path to iat_runs.csv.")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--exclude_models", default="llama31_8b",
                    help="Comma-separated model_key values to exclude (default: llama31_8b).")
    ap.add_argument("--drop_allsame", action="store_true", default=True)
    ap.add_argument("--no_drop_allsame", dest="drop_allsame", action="store_false")
    ap.add_argument("--status_ok_only", action="store_true", default=True)
    ap.add_argument("--min_coverage", type=float, default=0.5,
                    help="Drop IAT runs with coverage below this threshold (default: 0.5).")
    ap.add_argument("--tests", nargs="+", default=None,
                    help="Optional: restrict to specific test_id values, e.g. --tests gender_science race_racism")
    ap.add_argument("--agg", default="orders",
                    choices=["none", "orders", "tests", "both"],
                    help="Aggregation level: none=keep all rows, orders=avg over 3 orders per test, "
                         "tests=avg over tests, both=avg orders then tests (default: orders).")
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

    iat = _load_iat(
        path=args.iat_runs_csv,
        exclude_models=exclude_models,
        tests=args.tests,
        min_coverage=args.min_coverage,
    )

    # Resolve merge keys
    merge_keys = [k for k in MATCH_KEY if k in sr.columns and k in iat.columns]
    print(f"[merge] merging on: {merge_keys}")
    print(f"[merge] selfreport rows: {len(sr)} | iat rows: {len(iat)}")
    print(f"[merge] expected output rows (before agg): {len(sr)} conditions × "
          f"{iat.groupby(merge_keys).ngroups // max(len(sr), 1) + 1} iat rows per condition")

    # Columns to carry
    subscale_cols = _detect_subscale_cols(sr)
    sr_carry = list(dict.fromkeys(
        [c for c in merge_keys + subscale_cols + [INSTRUMENT_COL, "model_id", "run_id"]
         if c in sr.columns]
    ))
    iat_carry = list(dict.fromkeys(
        [c for c in merge_keys + IAT_DIM_COLS + IAT_OUTCOME_COLS + ["run_id", "task_seed"]
         if c in iat.columns]
    ))

    merged = sr[sr_carry].merge(
        iat[iat_carry],
        on=merge_keys,
        how=args.how,
        suffixes=("__sr", "__iat"),
    )
    print(f"[merge] rows after merge (before agg): {len(merged)}")
    print(f"[merge] model_key distribution: {merged['model_key'].value_counts().to_dict()}")

    # Aggregate
    merged = _aggregate_iat(merged, args.agg, subscale_cols)

    # Write
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"[ok] written: {args.out_csv} ({len(merged)} rows)")

    # Summary
    if subscale_cols and "bias" in merged.columns:
        print("\n[summary] Mean bias by model:")
        by_model = merged.groupby("model_key")["bias"].agg(["mean", "std", "count"]).round(3)
        print(by_model.to_string())
        if "test_id" in merged.columns:
            print("\n[summary] Mean bias by model × test:")
            print(merged.groupby(["model_key", "test_id"])["bias"].mean().round(3).unstack().to_string())


if __name__ == "__main__":
    main()