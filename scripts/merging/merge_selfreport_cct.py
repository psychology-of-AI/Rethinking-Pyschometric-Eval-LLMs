#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_selfreport_cct.py — Instrument-agnostic self-report × CCT merger (v1)
============================================================================
Merges a self-report likert_runs CSV (BFI, HEXACO, MFQ, PVQ, TPB, …) with a
CCT behavioral runs CSV on the grid match key:
    (model_key, seed, temperature, top_p, persona_label)

Subscale columns are auto-detected as any column ending in ``_mean`` present
in the self-report file.  Swap instruments by changing --instrument and
--selfreport_csv only.

Usage
-----
# BIG-5
python merge_selfreport_cct.py \
    --selfreport_csv results/tpb_likert_runs.csv \
    --instrument big5 \
    --cct_runs_csv results/cct_runs.csv \
    --out_csv results/merged/big5_x_cct.csv

# HEXACO-24
python merge_selfreport_cct.py \
    --selfreport_csv results/tpb_likert_runs.csv \
    --instrument hexaco24 \
    --cct_runs_csv results/cct_runs.csv \
    --out_csv results/merged/hexaco_x_cct.csv

# MFQ
python merge_selfreport_cct.py \
    --selfreport_csv results/tpb_likert_runs.csv \
    --instrument mfq \
    --cct_runs_csv results/cct_runs.csv \
    --out_csv results/merged/mfq_x_cct.csv

# PVQ
python merge_selfreport_cct.py \
    --selfreport_csv results/tpb_likert_runs.csv \
    --instrument pvq \
    --cct_runs_csv results/cct_runs.csv \
    --out_csv results/merged/pvq_x_cct.csv

# TPB self-report
python merge_selfreport_cct.py \
    --selfreport_csv results/tpb_likert_runs.csv \
    --instrument gain_seeking \
    --cct_runs_csv results/cct_runs.csv \
    --out_csv results/merged/tpb_x_cct.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MATCH_KEY = ["model_key", "seed", "temperature", "top_p", "persona_label"]

# CCT outcome columns — excluded from subscale auto-detection
CCT_OUTCOME_COLS = {
    "mean_k", "loss_rate", "total_payoff", "total_expected_payoff",
    "prop_max_flips", "n_rounds", "max_flips",
}

# Column in self-report CSV that identifies the instrument variant
INSTRUMENT_COL = "gen_extra__prompt_variant"

# Status column — keep only ok rows if present
STATUS_COL = "status"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_subscale_cols(df: pd.DataFrame) -> List[str]:
    """Return all *_mean columns that are not CCT outcome cols."""
    return [
        c for c in df.columns
        if c.endswith("_mean") and c not in CCT_OUTCOME_COLS
    ]


def _is_allones(row: pd.Series) -> bool:
    return bool((row == 1.0).all())


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
    drop_allones: bool,
    drop_allsame: bool,
    status_ok_only: bool,
) -> pd.DataFrame:
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    n_raw = len(df)

    # Filter instrument variant
    if instrument and INSTRUMENT_COL in df.columns:
        available = sorted(df[INSTRUMENT_COL].dropna().astype(str).unique().tolist())
        df = df[df[INSTRUMENT_COL].astype(str).str.strip() == instrument].copy()
        print(f"[selfreport] instrument={instrument!r}: {len(df)}/{n_raw} rows")
        if len(df) == 0:
            print(f"[selfreport] ERROR: no rows matched instrument={instrument!r}")
            print(f"[selfreport] Available variants in this file: {available}")
            print(f"[selfreport] Re-run with one of those as --instrument")
            sys.exit(1)
    elif instrument:
        print(f"[selfreport] WARNING: column '{INSTRUMENT_COL}' not found — cannot filter by instrument")

    # Filter status
    if status_ok_only and STATUS_COL in df.columns:
        before = len(df)
        df = df[df[STATUS_COL].astype(str).str.lower() == "ok"].copy()
        print(f"[selfreport] status=ok filter: {len(df)}/{before} rows kept")
        if len(df) == 0:
            print(f"[selfreport] ERROR: no rows with status=ok remain after filtering")
            sys.exit(1)

    # Detect subscale columns
    subscale_cols = _detect_subscale_cols(df)
    if not subscale_cols:
        print("[selfreport] WARNING: no *_mean subscale columns detected")
    else:
        print(f"[selfreport] subscale cols detected: {subscale_cols}")
        for c in subscale_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop collapsed runs (skip if df is empty or no subscale cols)
    if subscale_cols and len(df) > 0:
        if drop_allones:
            mask = df[subscale_cols].apply(_is_allones, axis=1)
            n_drop = mask.sum()
            df = df[~mask].copy()
            if n_drop:
                print(f"[selfreport] dropped {n_drop} all-1 collapsed runs")
        if drop_allsame:
            mask = df[subscale_cols].apply(_is_allsame, axis=1)
            n_drop = mask.sum()
            df = df[~mask].copy()
            if n_drop:
                print(f"[selfreport] dropped {n_drop} all-same collapsed runs")

    # Exclude models
    if exclude_models and "model_key" in df.columns:
        before = len(df)
        df = df[~df["model_key"].isin(exclude_models)].copy()
        print(f"[selfreport] excluded models {exclude_models}: {len(df)}/{before} rows kept")

    df = _coerce_key_types(df)
    return df


def _load_cct(
    path: str,
    exclude_models: List[str],
) -> pd.DataFrame:
    df = pd.read_csv(path)
    n_raw = len(df)

    if exclude_models and "model_key" in df.columns:
        df = df[~df["model_key"].isin(exclude_models)].copy()
        print(f"[cct] excluded models {exclude_models}: {len(df)}/{n_raw} rows kept")

    df = _coerce_key_types(df)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge self-report subscale means with CCT behavioral runs on the PI grid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--selfreport_csv", required=True,
                    help="Path to tpb_likert_runs.csv (or equivalent instrument runs file).")
    ap.add_argument("--instrument", default=None,
                    help="Value of gen_extra__prompt_variant to keep (e.g. 'big5', 'hexaco24', "
                         "'mfq', 'pvq', 'gain_seeking'). Required if file contains multiple instruments.")
    ap.add_argument("--cct_runs_csv", required=True,
                    help="Path to CCT behavioral runs CSV (cct_runs.csv).")
    ap.add_argument("--out_csv", required=True,
                    help="Output path for the merged CSV.")
    ap.add_argument("--exclude_models", default="llama31_8b",
                    help="Comma-separated model_key values to exclude (default: llama31_8b).")
    ap.add_argument("--drop_allones", action="store_true", default=True,
                    help="Drop self-report runs where all subscale means == 1.0 (default: on).")
    ap.add_argument("--no_drop_allones", dest="drop_allones", action="store_false")
    ap.add_argument("--drop_allsame", action="store_true", default=True,
                    help="Drop self-report runs where all subscale means are identical (default: on).")
    ap.add_argument("--no_drop_allsame", dest="drop_allsame", action="store_false")
    ap.add_argument("--status_ok_only", action="store_true", default=True,
                    help="Keep only status==ok rows in self-report file (default: on).")
    ap.add_argument("--how", default="inner", choices=["inner", "left", "outer"],
                    help="Merge type (default: inner — only matched conditions).")
    args = ap.parse_args()

    exclude_models = [m.strip() for m in (args.exclude_models or "").split(",") if m.strip()]

    # Load
    sr = _load_selfreport(
        path=args.selfreport_csv,
        instrument=args.instrument,
        exclude_models=exclude_models,
        drop_allones=args.drop_allones,
        drop_allsame=args.drop_allsame,
        status_ok_only=args.status_ok_only,
    )

    cct = _load_cct(path=args.cct_runs_csv, exclude_models=exclude_models)

    # Resolve merge keys — only use keys present in both
    merge_keys = [k for k in MATCH_KEY if k in sr.columns and k in cct.columns]
    missing = [k for k in MATCH_KEY if k not in merge_keys]
    if missing:
        print(f"[merge] WARNING: match keys not in both files: {missing}")
    print(f"[merge] merging on: {merge_keys}")

    # Identify subscale cols to carry from self-report
    subscale_cols = _detect_subscale_cols(sr)
    sr_carry = [c for c in merge_keys + subscale_cols + [INSTRUMENT_COL, "model_id", "run_id"]
                if c in sr.columns]
    sr_carry = list(dict.fromkeys(sr_carry))  # dedup, preserve order

    # CCT outcome cols to carry
    cct_outcomes = [c for c in CCT_OUTCOME_COLS if c in cct.columns]
    cct_carry = [c for c in merge_keys + cct_outcomes + ["run_id", "prompt_variant",
                 "n_rounds", "max_flips", "task_seed"]
                 if c in cct.columns]
    cct_carry = list(dict.fromkeys(cct_carry))

    merged = sr[sr_carry].merge(
        cct[cct_carry],
        on=merge_keys,
        how=args.how,
        suffixes=("__sr", "__cct"),
    )

    # Diagnostics
    n_matched = merged[subscale_cols[0] if subscale_cols else merge_keys[0]].notna().sum() if subscale_cols else len(merged)
    n_cct_matched = merged[cct_outcomes[0]].notna().sum() if cct_outcomes else len(merged)
    print(f"[merge] rows in output: {len(merged)}")
    print(f"[merge] rows with self-report data: {n_matched}")
    print(f"[merge] rows with CCT data: {n_cct_matched}")
    print(f"[merge] model_key distribution:\n{merged['model_key'].value_counts().to_string()}")

    # Write
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"[ok] written: {args.out_csv}")

    # Quick summary
    if subscale_cols and cct_outcomes:
        print("\n[summary] Mean subscale scores by model:")
        print(merged.groupby("model_key")[subscale_cols].mean().round(3).to_string())
        print("\n[summary] Mean CCT outcomes by model:")
        print(merged.groupby("model_key")[cct_outcomes].mean().round(3).to_string())


if __name__ == "__main__":
    main()