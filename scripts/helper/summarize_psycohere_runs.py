#!/usr/bin/env python3
"""
summarize_psycohere_runs.py — config-driven run status monitor.

Discovers tasks, variants, models, and expected conditions by reading the
config directory. Adding new tasks, policies, or models to configs requires
no changes to this script.

Usage
-----
  python scripts/summarize_psycohere_runs.py \
      --results_root results \
      --config_root  configs

  # Full missing-model list
  ... --verbose

  # Save status CSV
  ... --out_csv results/status_summary.csv

  # TPB SR subscale quality check (floor/ceiling per variant)
  ... --sr_quality

  # Pooled SR means + per-model policy contrasts
  ... --sr_descriptives
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ── Safe-slug helper (mirrors sweep scripts) ───────────────────────────────────

def _safe_slug(s: str) -> str:
    """Convert a string to its slug form, matching the sweep scripts."""
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


# ── Config loading ─────────────────────────────────────────────────────────────

def _count_expected_per_model(grid: dict) -> int:
    """Compute expected condition count per model from a grid block."""
    seeds  = len(grid.get("seeds") or [42])
    temps  = len(grid.get("temperatures") or [0.2])
    top_ps = len(grid.get("top_ps") or [1.0])
    mode   = str(grid.get("persona_prompts_mode", "direct")).strip().lower()
    if mode == "selected_personas":
        sel = grid.get("persona_prompts_selected") or {}
        n_p = int(sel.get("limit") or 30)
    else:
        n_p = len(grid.get("persona_prompts") or [""])
    return seeds * temps * top_ps * n_p


def _parse_contrastive_pairs(cfg: dict, variant_ids: list) -> List[Tuple[str, str]]:
    """
    Return contrastive (var_a, var_b) pairs for --sr_descriptives.

    Priority:
      1. _doc.contrastive_pairs  —  explicit {axis_name: {key: variant_id}} dict
      2. Consecutive pairing of variants list: (v[0],v[1]), (v[2],v[3]), ...
    """
    doc = cfg.get("_doc") or {}
    cp  = doc.get("contrastive_pairs")
    if cp and isinstance(cp, dict):
        pairs = []
        for axis_vals in cp.values():
            ids = [v for v in axis_vals.values() if isinstance(v, str)]
            if len(ids) >= 2:
                pairs.append((ids[0], ids[1]))
        if pairs:
            return pairs
    return [(variant_ids[i], variant_ids[i + 1])
            for i in range(0, len(variant_ids) - 1, 2)]


def _detect_perturbation(filename: str) -> Optional[str]:
    if "_grid" in filename:
        return "grid"
    if "_personas" in filename:
        return "personas"
    return None


def _load_json_safe(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_dilemma_count(cfg: dict) -> Optional[int]:
    """
    For sycophancy-family configs, return the number of dilemmas a condition
    must complete to be 'done'. None if not applicable / unknown.
    """
    syc = cfg.get("sycophancy") or {}
    dids = syc.get("dilemma_ids", None)
    if isinstance(dids, list):
        return len(dids)
    # 'ALL' or absent: try to load the dilemmas file to count
    path = syc.get("dilemmas_json")
    if isinstance(path, str):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(data, list):
                return len(data)
        except Exception:
            return None
    return None


def _runs_csv_candidates(framework: str, task: Optional[str]) -> List[str]:
    """
    Return candidate filenames for the per-variant runs CSV, in priority order.

    The sycophancy sweep script (sweep_sycophancy_variants.py) hardcodes the
    output filename as 'sycophancy_runs.csv' regardless of task name — including
    when the task is 'full_sycophancy'. So for any sycophancy-family task we
    try the task-derived name first, then fall back to 'sycophancy_runs.csv'.
    """
    if framework != "behavior":
        return ["tpb_likert_runs.csv"]

    if not task:
        return ["runs.csv"]

    cands = [f"{task}_runs.csv"]
    if "sycophancy" in task.lower() and "sycophancy_runs.csv" not in cands:
        cands.append("sycophancy_runs.csv")
    return cands


def load_configs(config_root: Path) -> List[dict]:
    """
    Scan config_root/{behavior,tpb,big5}/ and return one entry per JSON file.

    Each entry dict:
        framework         "behavior" | "tpb" | "big5"
        task              e.g. "cct", "honesty"; None for big5 (task-agnostic)
        perturbation      "grid" | "personas"
        experiment_name   from config (used to locate result dirs)
        model_keys        list of model key strings
        n_models          int
        n_per_model       int (conditions per model)
        n_expected        int (n_models x n_per_model)
        variants          list of variant_id strings
        contrastive_pairs list of (var_a, var_b) tuples
        runs_csv          primary filename to look for (back-compat)
        runs_csv_candidates list of filenames to try, in priority order
        n_dilemmas        int | None (sycophancy: rows-per-complete-condition)
    """
    entries = []

    for framework, subdir in [("behavior", "behavior"), ("tpb", "tpb"), ("big5", "big5")]:
        d = config_root / subdir
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            cfg = _load_json_safe(p)
            if cfg is None:
                continue
            pert = _detect_perturbation(p.name)
            if not pert:
                continue

            doc  = cfg.get("_doc") or {}
            task = doc.get("task") or None
            exp  = cfg.get("experiment_name", p.stem)
            keys = list(cfg.get("model_keys") or [])
            grid = cfg.get("grid") or {}

            raw_variants = cfg.get("variants") or []
            variant_ids  = [
                v["variant_id"] for v in raw_variants
                if isinstance(v, dict) and "variant_id" in v
            ]

            n_pm  = _count_expected_per_model(grid)
            pairs = _parse_contrastive_pairs(cfg, variant_ids)

            cands     = _runs_csv_candidates(framework, task)
            runs_csv  = cands[0]
            n_dilem   = _resolve_dilemma_count(cfg) if (framework == "behavior" and task and "sycophancy" in task.lower()) else None

            entries.append({
                "framework":            framework,
                "task":                 task,
                "perturbation":         pert,
                "experiment_name":      exp,
                "model_keys":           keys,
                "n_models":             len(keys),
                "n_per_model":          n_pm,
                "n_expected":           len(keys) * n_pm,
                "variants":             variant_ids,
                "contrastive_pairs":    pairs,
                "runs_csv":             runs_csv,
                "runs_csv_candidates":  cands,
                "n_dilemmas":           n_dilem,
            })

    return entries


# ── CSV helpers ────────────────────────────────────────────────────────────────

def _read_csv(path: Optional[Path]) -> pd.DataFrame:
    """Read a CSV robustly, padding short rows (multi-task combined files)."""
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            return pd.DataFrame()
        header   = rows[0]
        expected = len(header)
        good = []
        for row in rows[1:]:
            diff = expected - len(row)
            if diff == 0:
                good.append(row)
            elif 0 < diff <= 3:
                # row is shorter than header (error rows in combined CSV) — pad
                good.append(row + [""] * diff)
            elif diff < 0:
                # row is longer than header (ok rows with extra outcome cols when
                # header was written from an earlier error row) — truncate
                good.append(row[:expected])
        df = pd.DataFrame(good, columns=header)
        for col in df.columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().any():
                df[col] = converted
        return df
    except Exception as e:
        print(f"  [warn] Could not read {path}: {e}", file=sys.stderr)
        return pd.DataFrame()


# ── Path resolution ────────────────────────────────────────────────────────────
# Different sweep scripts use different conventions: some preserve underscores
# in experiment/variant dir names, others slugify them (underscores -> dashes).
# _find_dir tries both forms so this script works regardless.

def _find_dir(parent: Path, name: str) -> Optional[Path]:
    """Return first existing subdirectory matching raw name or its slug."""
    for candidate in [parent / name, parent / _safe_slug(name)]:
        if candidate.is_dir():
            return candidate
    return None


def _find_csv(parent: Path, filename: str) -> Optional[Path]:
    """Return first existing file matching raw filename or its slug."""
    for candidate in [parent / filename, parent / _safe_slug(filename)]:
        if candidate.exists():
            return candidate
    return None


def _find_csv_any(parent: Path, filenames: List[str]) -> Optional[Path]:
    """Try each candidate filename (and its slug) in order; return the first hit."""
    for fn in filenames:
        hit = _find_csv(parent, fn)
        if hit is not None:
            return hit
    return None


# ── Status helpers ─────────────────────────────────────────────────────────────

def _cond_count(df: pd.DataFrame, *, n_dilemmas: Optional[int] = None) -> int:
    """
    Count unique (model x condition) cells.

    If n_dilemmas is set (sycophancy-family), a condition is only counted
    when it has all n_dilemmas dilemma_id rows present. Otherwise any
    nonempty group counts as 1 (matches legacy behavior for other tasks).
    """
    key_cols = [c for c in ["model_key", "seed", "temperature", "top_p"]
                if c in df.columns]
    if "persona_label" in df.columns:
        non_empty = df["persona_label"].astype(str).str.strip().replace("", pd.NA).notna()
        if non_empty.any():
            key_cols.append("persona_label")
    if not key_cols:
        return len(df)

    if n_dilemmas and n_dilemmas > 0 and "dilemma_id" in df.columns:
        per_cond = df.groupby(key_cols)["dilemma_id"].nunique()
        return int((per_cond >= n_dilemmas).sum())

    return df.groupby(key_cols).ngroups


def _missing_from(df: pd.DataFrame, model_keys: list) -> list:
    if df.empty or "model_key" not in df.columns:
        return sorted(model_keys)
    present = set(df["model_key"].dropna().unique())
    return sorted(set(model_keys) - present)


def _flag(n: int, exp: int) -> str:
    if n >= exp:            return "✅"
    if n >= int(0.7 * exp): return "🟡"
    if n > 0:               return "🔴"
    return "❌"


def _status_line(label: str, n: int, exp: int, missing: list, verbose: bool) -> str:
    flag = _flag(n, exp)
    pct  = int(100 * n / exp) if exp else 0
    line = f"  {flag}  {label:<58}  {n:>4}/{exp}  ({pct:>3}%)"
    if missing and verbose:
        line += f"\n       missing: {missing}"
    elif missing and n < exp:
        line += f"  | missing: {missing[:3]}{'...' if len(missing) > 3 else ''}"
    return line


# ── Section checks ─────────────────────────────────────────────────────────────

def check_behavioral(entries: List[dict], results_root: Path, verbose: bool) -> list:
    """Between-session behavioral runs, one row per task x perturbation x variant."""
    rows = []
    beh  = [e for e in entries if e["framework"] == "behavior"]

    for e in sorted(beh, key=lambda x: (x["task"] or "", x["perturbation"])):
        task, pert = e["task"], e["perturbation"]
        base    = results_root / "between" / pert / "session_beh"
        exp_dir = _find_dir(base, e["experiment_name"])

        for vid in e["variants"]:
            label = f"Beh / {task:<12} / {pert:<8} / {vid}"
            csv_p = None
            if exp_dir:
                v_dir = _find_dir(exp_dir, vid)
                if v_dir:
                    csv_p = _find_csv_any(v_dir, e["runs_csv_candidates"])

            df   = _read_csv(csv_p)
            n    = _cond_count(df, n_dilemmas=e.get("n_dilemmas"))
            miss = _missing_from(df, e["model_keys"])
            extra = ""
            if e.get("n_dilemmas") and not df.empty and "dilemma_id" in df.columns:
                # Show raw dilemma-row count alongside conditions for sycophancy
                extra = f"  [rows={len(df)}, dil/cond={e['n_dilemmas']}]"
            line = _status_line(label, n, e["n_expected"], miss, verbose) + extra
            rows.append({"section": "behavioral", "label": label,
                         "n": n, "exp": e["n_expected"], "missing": miss,
                         "line": line})
    return rows


def check_sr(entries: List[dict], results_root: Path, verbose: bool) -> list:
    """Between-session self-report, one row per framework x task x perturbation x variant."""
    rows = []
    sr   = [e for e in entries if e["framework"] in ("tpb", "big5")]

    for e in sorted(sr, key=lambda x: (x["framework"], x["task"] or "", x["perturbation"])):
        fw, task, pert = e["framework"], e["task"], e["perturbation"]
        base    = results_root / "between" / pert / "session_sr"
        exp_dir = _find_dir(base, e["experiment_name"])
        tag     = f"{fw.upper()}-{task or 'all'}"

        for vid in e["variants"]:
            label = f"SR  / {tag:<18} / {pert:<8} / {vid}"
            csv_p = None
            if exp_dir:
                v_dir = _find_dir(exp_dir, vid)
                if v_dir:
                    csv_p = _find_csv(v_dir, e["runs_csv"])

            df   = _read_csv(csv_p)
            # TPB/Big5: one row per condition
            n    = len(df) if not df.empty else 0
            miss = _missing_from(df, e["model_keys"])
            rows.append({"section": "sr", "label": label,
                         "n": n, "exp": e["n_expected"], "missing": miss,
                         "line": _status_line(label, n, e["n_expected"], miss, verbose)})
    return rows


def check_within(entries: List[dict], results_root: Path, verbose: bool) -> list:
    """
    Within-session combined runs.

    Pairing logic (mirrors sweep_combined_variants.py):
      - TPB (task=X, pert=P) x behavioral (task=X, pert=P)
          -> within/{pert}/{tpb_exp}/{sr_variant}/combined_runs.csv
      - Big5 (pert=P) x each behavioral (task=X, pert=P)
          -> within/{pert}/{big5_exp}/{beh_task}/{big5_variant}/combined_runs.csv
    """
    rows = []
    tpb_entries  = [e for e in entries if e["framework"] == "tpb"]
    big5_entries = [e for e in entries if e["framework"] == "big5"]
    beh_entries  = [e for e in entries if e["framework"] == "behavior"]
    beh_by       = {(e["task"], e["perturbation"]): e for e in beh_entries}

    # TPB within-session
    for sr_e in sorted(tpb_entries, key=lambda x: (x["task"] or "", x["perturbation"])):
        task, pert = sr_e["task"], sr_e["perturbation"]
        if (task, pert) not in beh_by:
            continue
        base    = results_root / "within" / pert
        exp_dir = _find_dir(base, sr_e["experiment_name"])

        for vid in sr_e["variants"]:
            label = f"Within / TPB-{task:<12} / {pert:<8} / {vid}"
            csv_p = None
            if exp_dir:
                # Try with task subdir first (newer sweep_combined_variants behavior),
                # then without (older behavior) — handles both layouts transparently
                if task:
                    task_dir = _find_dir(exp_dir, task)
                    if task_dir:
                        v_dir = _find_dir(task_dir, vid)
                        if v_dir:
                            csv_p = _find_csv(v_dir, "combined_runs.csv")
                if csv_p is None:
                    v_dir = _find_dir(exp_dir, vid)
                    if v_dir:
                        csv_p = _find_csv(v_dir, "combined_runs.csv")

            df = _read_csv(csv_p)
            n_errors = 0
            if not df.empty and "sr_status" in df.columns:
                n_errors = int((df["sr_status"] == "error").sum())
                df = df[df["sr_status"] == "ok"]
            n    = _cond_count(df)
            miss = _missing_from(df, sr_e["model_keys"])
            line = _status_line(label, n, sr_e["n_expected"], miss, verbose)
            if n_errors > 0 and n < sr_e["n_expected"]:
                line += f"  [{n_errors} errors — will retry on resume]"
            rows.append({"section": "within", "label": label, "n_errors": n_errors,
                         "n": n, "exp": sr_e["n_expected"], "missing": miss,
                         "line": line})

    # Big5 within-session
    for sr_e in sorted(big5_entries, key=lambda x: x["perturbation"]):
        pert    = sr_e["perturbation"]
        base    = results_root / "within" / pert
        exp_dir = _find_dir(base, sr_e["experiment_name"])
        beh_for = sorted([e for e in beh_entries if e["perturbation"] == pert],
                         key=lambda x: x["task"] or "")

        for beh_e in beh_for:
            task = beh_e["task"]
            for vid in sr_e["variants"]:
                label = f"Within / Big5-{task:<12} / {pert:<8} / {vid}"
                csv_p = None
                if exp_dir:
                    task_dir = _find_dir(exp_dir, task) if task else exp_dir
                    if task_dir:
                        v_dir = _find_dir(task_dir, vid)
                        if v_dir:
                            csv_p = _find_csv(v_dir, "combined_runs.csv")
                    # Fallback: old layout without task subdir
                    if csv_p is None:
                        v_dir2 = _find_dir(exp_dir, vid)
                        if v_dir2:
                            csv_p = _find_csv(v_dir2, "combined_runs.csv")

                df = _read_csv(csv_p)
                if not df.empty and "sr_status" in df.columns:
                    df = df[df["sr_status"] == "ok"]
                if not df.empty and task and "task" in df.columns:
                    df = df[df["task"] == task]
                n    = _cond_count(df)
                miss = _missing_from(df, sr_e["model_keys"])
                rows.append({"section": "within", "label": label, "n_errors": 0,
                             "n": n, "exp": sr_e["n_expected"], "missing": miss,
                             "line": _status_line(label, n, sr_e["n_expected"], miss, verbose)})
    return rows


# ── SR quality check ───────────────────────────────────────────────────────────

def check_sr_quality(results_root: Path) -> None:
    """Print TPB SR subscale stats — uses glob, auto-discovers all variants."""
    constructs = ["attitude_mean", "subjective_norm_mean", "pbc_mean", "intention_mean"]
    print("\n── TPB SR quality (subscale means by variant, pooled across conditions) ──")
    seen_any = False
    for pert in ["grid", "personas"]:
        sr_root = results_root / "between" / pert / "session_sr"
        if not sr_root.exists():
            continue
        for csv_path in sorted(sr_root.glob("**/tpb_likert_runs.csv")):
            df = _read_csv(csv_path)
            if df.empty:
                continue
            avail = [c for c in constructs if c in df.columns]
            if not avail:
                continue
            variant = csv_path.parent.name
            task    = csv_path.parent.parent.name
            print(f"\n  {task} / {variant} / {pert}")
            for c in avail:
                s = pd.to_numeric(df[c], errors="coerce").dropna()
                if len(s):
                    print(f"    {c:<25}  mean={s.mean():.3f}  std={s.std():.3f}  "
                          f"n={len(s)}  floor={(s<=1.01).mean():.2f}  ceil={(s>=6.99).mean():.2f}")
            seen_any = True
    if not seen_any:
        print("  No TPB SR files found.")


# ── SR descriptives ────────────────────────────────────────────────────────────

def _load_sr_variant(results_root: Path, exp_name: str,
                     variant_id: str, perturbation: str) -> pd.DataFrame:
    base    = results_root / "between" / perturbation / "session_sr"
    exp_dir = _find_dir(base, exp_name)
    if exp_dir is None:
        return pd.DataFrame()
    v_dir = _find_dir(exp_dir, variant_id)
    if v_dir is None:
        return pd.DataFrame()
    csv_p = _find_csv(v_dir, "tpb_likert_runs.csv")
    return _read_csv(csv_p)


def compute_sr_descriptives(entries: List[dict], results_root: Path,
                             out_csv: Optional[str] = None) -> None:
    """
    For each TPB config: pooled means per variant and per-model intention
    contrasts for each contrastive pair (from _doc.contrastive_pairs if
    present, else consecutive pairs).
    """
    constructs  = ["attitude_mean", "subjective_norm_mean", "pbc_mean", "intention_mean"]
    tpb_entries = [e for e in entries if e["framework"] == "tpb"]

    pooled_rows   = []
    contrast_rows = []

    for e in sorted(tpb_entries, key=lambda x: (x["task"] or "", x["perturbation"])):
        task, pert, exp = e["task"], e["perturbation"], e["experiment_name"]

        for vid in e["variants"]:
            df = _load_sr_variant(results_root, exp, vid, pert)
            if df.empty:
                continue
            for c in constructs:
                if c not in df.columns:
                    continue
                s = pd.to_numeric(df[c], errors="coerce").dropna()
                if not len(s):
                    continue
                pooled_rows.append({
                    "task": task, "perturbation": pert, "variant": vid, "construct": c,
                    "mean":  round(float(s.mean()), 4),
                    "std":   round(float(s.std(ddof=1)), 4) if len(s) > 1 else float("nan"),
                    "n":     int(len(s)),
                    "floor": round(float((s <= 1.01).mean()), 4),
                    "ceil":  round(float((s >= 6.99).mean()), 4),
                })

        for var_a, var_b in e["contrastive_pairs"]:
            df_a = _load_sr_variant(results_root, exp, var_a, pert)
            df_b = _load_sr_variant(results_root, exp, var_b, pert)
            if df_a.empty or df_b.empty:
                continue
            if "intention_mean" not in df_a.columns or "intention_mean" not in df_b.columns:
                continue
            m_a = df_a.groupby("model_key")["intention_mean"].apply(
                lambda x: pd.to_numeric(x, errors="coerce").mean())
            m_b = df_b.groupby("model_key")["intention_mean"].apply(
                lambda x: pd.to_numeric(x, errors="coerce").mean())
            contrast = (m_b - m_a).dropna()
            for model, val in contrast.items():
                contrast_rows.append({"task": task, "perturbation": pert,
                                       "variant_a": var_a, "variant_b": var_b,
                                       "model_key": model,
                                       "contrast_intention": round(float(val), 4)})
            contrast_rows.append({"task": task, "perturbation": pert,
                                   "variant_a": var_a, "variant_b": var_b,
                                   "model_key": "__pooled__",
                                   "contrast_intention": round(float(contrast.mean()), 4),
                                   "contrast_sd": round(float(contrast.std(ddof=1)), 4)
                                   if len(contrast) > 1 else float("nan")})

    df_pooled = pd.DataFrame(pooled_rows, columns=[
        "task", "perturbation", "variant", "construct",
        "mean", "std", "n", "floor", "ceil"])
    df_contrast = pd.DataFrame(contrast_rows, columns=[
        "task", "perturbation", "variant_a", "variant_b",
        "model_key", "contrast_intention", "contrast_sd"])

    print("\n" + "=" * 72)
    print("SR DESCRIPTIVES — pooled means by task x perturbation x variant")
    print("=" * 72)
    for e in sorted(tpb_entries, key=lambda x: (x["task"] or "", x["perturbation"])):
        task, pert = e["task"], e["perturbation"]
        for vid in e["variants"]:
            sub = df_pooled[
                (df_pooled.task == task) & (df_pooled.perturbation == pert) &
                (df_pooled.variant == vid) & (df_pooled.construct.isin(constructs))
            ]
            if sub.empty:
                continue
            print(f"\n  {task} / {vid} / {pert}")
            for _, r in sub.iterrows():
                print(f"    {r.construct:<25}  mean={r['mean']:.3f}  std={r['std']:.3f}"
                      f"  n={r.n}  floor={r.floor:.2f}  ceil={r.ceil:.2f}")

    print("\n" + "=" * 72)
    print("SR POLICY CONTRASTS — per-model intention: var_b - var_a")
    print("=" * 72)
    for e in sorted(tpb_entries, key=lambda x: (x["task"] or "", x["perturbation"])):
        task, pert = e["task"], e["perturbation"]
        for var_a, var_b in e["contrastive_pairs"]:
            sub = df_contrast[
                (df_contrast.task == task) & (df_contrast.perturbation == pert) &
                (df_contrast.variant_a == var_a) & (df_contrast.variant_b == var_b)
            ]
            if sub.empty:
                continue
            per_model = sub[sub.model_key != "__pooled__"].sort_values("contrast_intention")
            pooled    = sub[sub.model_key == "__pooled__"]
            print(f"\n  {task} / {pert}  ({var_b} - {var_a})")
            for _, r in per_model.iterrows():
                bar  = "█" * max(0, int(abs(r.contrast_intention) * 4))
                sign = "+" if r.contrast_intention >= 0 else "-"
                print(f"    {r.model_key:<22}  {sign}{abs(r.contrast_intention):.3f}  {bar}")
            if not pooled.empty:
                p      = pooled.iloc[0]
                sd_str = (f"  sd={p.contrast_sd:.3f}"
                          if "contrast_sd" in p and pd.notna(p.get("contrast_sd")) else "")
                print(f"    {'-- pooled mean':<22}  {p.contrast_intention:+.3f}{sd_str}")

    if out_csv:
        stem = out_csv.replace(".csv", "")
        Path(stem + "_sr_descriptives.csv").parent.mkdir(parents=True, exist_ok=True)
        df_pooled.to_csv(stem + "_sr_descriptives.csv", index=False)
        df_contrast.to_csv(stem + "_sr_contrasts.csv", index=False)
        print(f"\nSR descriptives -> {stem}_sr_descriptives.csv")
        print(f"SR contrasts    -> {stem}_sr_contrasts.csv")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Config-driven psycohere run status monitor.")
    ap.add_argument("--results_root", required=True,
                    help="Path to results directory (e.g. results).")
    ap.add_argument("--config_root",  required=True,
                    help="Path to configs directory (e.g. configs).")
    ap.add_argument("--out_csv",      default=None,
                    help="Optional path to save status summary CSV.")
    ap.add_argument("--verbose",      action="store_true",
                    help="Print full list of missing models per run.")
    ap.add_argument("--sr_quality",   action="store_true",
                    help="Print TPB SR subscale statistics per variant.")
    ap.add_argument("--sr_descriptives", action="store_true",
                    help="Print pooled SR means + per-model policy contrasts.")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    config_root  = Path(args.config_root)

    if not results_root.exists():
        print(f"Results root not found: {results_root}", file=sys.stderr); sys.exit(1)
    if not config_root.exists():
        print(f"Config root not found: {config_root}", file=sys.stderr); sys.exit(1)

    entries = load_configs(config_root)
    if not entries:
        print(f"No config files found under {config_root}", file=sys.stderr); sys.exit(1)

    tasks  = sorted({e["task"] for e in entries if e["task"]})
    models = sorted({m for e in entries for m in e["model_keys"]})
    n_beh  = sum(1 for e in entries if e["framework"] == "behavior")
    n_tpb  = sum(1 for e in entries if e["framework"] == "tpb")
    n_big5 = sum(1 for e in entries if e["framework"] == "big5")

    print(f"\npsycohere run status  |  results: {results_root}")
    print(f"Config root: {config_root}")
    print(f"Discovered: {len(tasks)} tasks ({', '.join(tasks)})  |  "
          f"{len(models)} models  |  "
          f"{n_beh} behavior / {n_tpb} TPB / {n_big5} Big5 configs\n")

    all_rows = []

    print("=" * 72)
    print("BEHAVIORAL (between-session)")
    print("=" * 72)
    rows = check_behavioral(entries, results_root, args.verbose)
    for r in rows: print(r["line"])
    all_rows.extend(rows)

    print()
    print("=" * 72)
    print("SELF-REPORT (between-session)")
    print("=" * 72)
    rows = check_sr(entries, results_root, args.verbose)
    for r in rows: print(r["line"])
    all_rows.extend(rows)

    print()
    print("=" * 72)
    print("WITHIN-SESSION (combined SR + behavior)")
    print("=" * 72)
    rows = check_within(entries, results_root, args.verbose)
    for r in rows: print(r["line"])
    all_rows.extend(rows)

    done  = sum(1 for r in all_rows if r["n"] >= r["exp"])
    part  = sum(1 for r in all_rows if 0 < r["n"] < r["exp"])
    empty = sum(1 for r in all_rows if r["n"] == 0)
    total = len(all_rows)
    print(f"\n-- Summary: {done}/{total} complete  {part} partial  {empty} not started --\n")

    if args.sr_quality:
        check_sr_quality(results_root)

    if args.sr_descriptives:
        compute_sr_descriptives(entries, results_root, out_csv=args.out_csv)

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            "section":        r["section"],
            "label":          r["label"],
            "n":              r["n"],
            "expected":       r["exp"],
            "pct":            int(100 * r["n"] / r["exp"]) if r["exp"] else 0,
            "status":         "done" if r["n"] >= r["exp"] else
                              ("partial" if r["n"] > 0 else "empty"),
            "missing_models": ";".join(r["missing"]),
            "n_errors":       r.get("n_errors", 0),
        } for r in all_rows]).to_csv(out, index=False)
        print(f"Status summary saved to: {out}")


if __name__ == "__main__":
    main()