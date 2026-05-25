#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic merger (v5): TPB self-reports (per policy) + behavior runs (per episode/question),
using Option A: derive policy-specific alignment outcomes from the SAME behavior trace.

Fix vs v4:
- Add --exclude_policies: comma-separated policy ids to drop before merging.
  Useful for removing mirror-image policy pairs (e.g. gain_seeking when loss_averse
  is already present) that are perfectly anticorrelated and add no regression information.

Fix vs v3:
- Allow a small set of Python builtins in config expressions (e.g., str/int/float/bool),
  which are needed for expressions like `answer.astype(str)`.

CLI paths only; config contains schema + mapping only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _find_policy_subdirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _eval_expr(expr: str, df: pd.DataFrame) -> Any:
    """
    Evaluate a python expression with:
      - np, pd available
      - a small set of builtins: str, int, float, bool, len, min, max
      - each df column exposed as a variable (Series)
      - df itself exposed as variable 'df'

    Treat configs as trusted code.
    """
    expr = (expr or "").strip()
    if not expr:
        return None

    allowed_builtins = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "len": len,
        "min": min,
        "max": max,
    }
    safe_globals = {"__builtins__": allowed_builtins, "np": np, "pd": pd}
    safe_locals = {"df": df, **{c: df[c] for c in df.columns}}
    return eval(expr, safe_globals, safe_locals)


def _apply_where(df: pd.DataFrame, where_expr: str) -> pd.DataFrame:
    where_expr = (where_expr or "").strip()
    if not where_expr:
        return df
    mask = _eval_expr(where_expr, df)
    if mask is None:
        return df
    mask = pd.Series(mask)
    mask = mask.fillna(False).astype(bool).values
    return df.loc[mask].copy()


def _load_tpb_runs(
    tpb_root: str,
    runs_filename: str = "tpb_likert_runs.csv",
    policy_col: str = "gen_extra__prompt_variant",
    status_ok_only: bool = True,
) -> pd.DataFrame:
    root = Path(tpb_root)

    if root.is_file():
        paths = [root]
    else:
        if not root.exists():
            raise FileNotFoundError(f"TPB root not found: {root}")
        paths = [p / runs_filename for p in _find_policy_subdirs(root) if (p / runs_filename).exists()]
        if not paths:
            raise FileNotFoundError(f"No TPB runs found under {root} (expected */{runs_filename}).")

    dfs: List[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p, on_bad_lines="skip", engine="python")
        df["tpb_source_path"] = str(p)
        folder_policy = p.parent.name

        if policy_col in df.columns:
            df = df.rename(columns={policy_col: "policy_id"})
        elif "prompt_variant" in df.columns:
            df = df.rename(columns={"prompt_variant": "policy_id"})
        else:
            df["policy_id"] = folder_policy

        df["policy_id"] = df["policy_id"].astype(str).replace({"nan": ""}).str.strip()
        df.loc[df["policy_id"].eq(""), "policy_id"] = folder_policy

        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    if status_ok_only and "status" in out.columns:
        out = out[out["status"].astype(str).str.lower().eq("ok")].copy()

    for c in ["temperature", "top_p"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "seed" in out.columns:
        out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype("Int64")

    return out


def _load_behavior_runs(runs_csv: str, where_expr: str = "") -> pd.DataFrame:
    df = pd.read_csv(runs_csv)
    df = _apply_where(df, where_expr)

    for c in ["temperature", "top_p"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")

    return df


def _compute_precols(df: pd.DataFrame, precols: List[Dict[str, Any]]) -> pd.DataFrame:
    if not precols:
        return df
    out = df.copy()
    for item in precols:
        name = item["name"]
        expr = item["expr"]
        out[name] = _eval_expr(expr, out)
    return out


def _compute_alignment_long(cfg: Dict[str, Any], behavior: pd.DataFrame, policy_ids: List[str]) -> pd.DataFrame:
    beh_cfg = cfg.get("behavior", {})
    align_cfg = cfg["alignment"]

    join_keys = _as_list(cfg.get("join_keys"))
    episode_cols = _as_list(beh_cfg.get("episode_cols"))
    carry_cols = _as_list(beh_cfg.get("carry_cols"))

    behavior = _compute_precols(behavior, _as_list(beh_cfg.get("precompute_cols")))

    id_vars: List[str] = []
    for c in join_keys + episode_cols + carry_cols:
        if c and c in behavior.columns and c not in id_vars:
            id_vars.append(c)

    frames: List[pd.DataFrame] = []

    for policy_id, spec in align_cfg.items():
        if policy_ids and policy_id not in set(policy_ids):
            continue

        expr = spec["expr"]
        where = str(spec.get("where", "")).strip()
        clip_min = spec.get("clip_min", None)
        clip_max = spec.get("clip_max", None)

        tmp = behavior.copy()
        tmp = _apply_where(tmp, where)

        tmp["align_score"] = _eval_expr(expr, tmp)

        if clip_min is not None or clip_max is not None:
            tmp["align_score"] = pd.to_numeric(tmp["align_score"], errors="coerce").clip(lower=clip_min, upper=clip_max)

        tmp["policy_id"] = policy_id
        keep = id_vars + ["policy_id", "align_score"]
        keep = [c for c in keep if c in tmp.columns]
        frames.append(tmp[keep])

    if not frames:
        raise ValueError("No alignment rows produced. Check policy_ids and alignment config.")

    return pd.concat(frames, ignore_index=True)


def _merge(cfg: Dict[str, Any], behavior_long: pd.DataFrame, tpb_runs: pd.DataFrame) -> pd.DataFrame:
    join_keys = _as_list(cfg.get("join_keys"))
    keys = [c for c in join_keys if c in behavior_long.columns and c in tpb_runs.columns] + ["policy_id"]
    missing = [c for c in join_keys + ["policy_id"] if c not in keys]
    if missing:
        raise ValueError(f"Missing merge keys in one of the tables: {missing}")
    return behavior_long.merge(tpb_runs, on=keys, how="left", suffixes=("", "__tpb"))


def _aggregate_session(cfg: Dict[str, Any], merged_long: pd.DataFrame) -> pd.DataFrame:
    join_keys = _as_list(cfg.get("join_keys"))
    group_keys = [c for c in join_keys if c in merged_long.columns] + ["policy_id"]

    construct_cols = _as_list(cfg.get("tpb", {}).get(
        "construct_cols",
        ["attitude_mean", "subjective_norm_mean", "pbc_mean", "intention_mean"],
    ))

    agg_map: Dict[str, Any] = {
        "n_episodes": ("align_score", "size"),
        "n_align_nonnull": ("align_score", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        "align_mean": ("align_score", "mean"),
        "align_std": ("align_score", "std"),
    }
    for c in construct_cols:
        if c in merged_long.columns:
            agg_map[c] = (c, "mean")

    return merged_long.groupby(group_keys, dropna=False).agg(**agg_map).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to merge config JSON (no paths inside).")
    ap.add_argument("--tpb_root", required=True, help="TPB sweep root folder (contains per-policy subfolders) OR a single runs CSV.")
    ap.add_argument("--behavior_runs_csv", required=True, help="Behavior runs CSV path (e.g., honesty_runs.csv).")
    ap.add_argument("--out_prefix", required=True, help="Output prefix, e.g., results/.../honesty_x_tpb")
    ap.add_argument("--tpb_runs_filename", default="tpb_likert_runs.csv", help="Filename inside each TPB policy folder.")
    ap.add_argument("--tpb_policy_col", default="gen_extra__prompt_variant", help="Column holding policy variant id in TPB runs.")
    ap.add_argument("--tpb_status_ok_only", action="store_true", help="If set, keep only rows with status == 'ok' when present.")
    ap.add_argument("--behavior_where", default="", help="Optional python boolean expression filter applied to behavior runs.")
    ap.add_argument("--policy_ids", default="", help="Optional comma-separated policy ids to include (default: infer from TPB runs).")
    ap.add_argument("--exclude_policies", default="", help="Optional comma-separated policy ids to exclude (applied after --policy_ids). "
                    "Use to drop mirror-image pairs, e.g. --exclude_policies gain_seeking,intuitive_gut.")
    args = ap.parse_args()

    cfg = _read_json(args.config)

    out_prefix = args.out_prefix
    _ensure_dir(str(Path(out_prefix).parent))

    tpb_runs = _load_tpb_runs(
        tpb_root=args.tpb_root,
        runs_filename=args.tpb_runs_filename,
        policy_col=args.tpb_policy_col,
        status_ok_only=bool(args.tpb_status_ok_only),
    )
    behavior = _load_behavior_runs(args.behavior_runs_csv, where_expr=args.behavior_where)

    for df in [tpb_runs, behavior]:
        if "system_prompt" in df.columns:
            df["system_prompt"] = df["system_prompt"].str.replace("\n", "\\n", regex=False)

    policy_ids = [p.strip() for p in (args.policy_ids or "").split(",") if p.strip()]
    if not policy_ids:
        policy_ids = sorted(tpb_runs["policy_id"].dropna().astype(str).unique().tolist())

    exclude = {p.strip() for p in (args.exclude_policies or "").split(",") if p.strip()}
    if exclude:
        before = policy_ids[:]
        policy_ids = [p for p in policy_ids if p not in exclude]
        print(f"[info] --exclude_policies removed {sorted(exclude)} → keeping {policy_ids} (was {before})")

    behavior_long = _compute_alignment_long(cfg, behavior, policy_ids=policy_ids)
    merged_long = _merge(cfg, behavior_long, tpb_runs)

    long_path = f"{out_prefix}_long.csv"
    merged_long.to_csv(long_path, index=False)

    session = _aggregate_session(cfg, merged_long)
    session_path = f"{out_prefix}_session.csv"
    session.to_csv(session_path, index=False)

    matched = 0
    if "attitude_mean" in merged_long.columns:
        matched = int(pd.to_numeric(merged_long["attitude_mean"], errors="coerce").notna().sum())

    print(f"[ok] wrote: {long_path}")
    print(f"[ok] wrote: {session_path}")
    print(f"[diag] long rows: {len(merged_long)} | rows with TPB match (attitude_mean non-null): {matched}")
    print(f"[diag] policy_ids: {policy_ids}")


if __name__ == "__main__":
    main()