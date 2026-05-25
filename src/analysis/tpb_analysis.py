from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable, Independence


_NEUTRAL_PERSONA_MAP = {
    "": "empty",
    "You are a helpful assistant.": "helpful",
    "Respond to instructions.": "instructions",
}


@dataclass(frozen=True)
class TPBColumns:
    attitude: str = "attitude_mean"
    sn: str = "subjective_norm_mean"
    pbc: str = "pbc_mean"
    intention: str = "intention_mean"
    seed: str = "seed"
    temperature: str = "temperature"
    system_prompt: str = "system_prompt"
    persona_label: str = "persona_label"


def load_tpb_runs(runs_csv: str | Path, *, cols: TPBColumns = TPBColumns()) -> pd.DataFrame:
    """
    Load *_runs.csv robustly:
      - preserve empty strings (don't auto-convert to NaN)
      - add persona_label (empty/helpful/instructions/custom)
      - add categorical versions useful for modeling
      - define condition_id EXCLUDING seed (seed is replicate index)
    """
    runs_csv = Path(runs_csv)
    df = pd.read_csv(runs_csv, keep_default_na=False)

    # Normalize prompt column
    if cols.system_prompt in df.columns:
        df[cols.system_prompt] = df[cols.system_prompt].astype(str)
    else:
        df[cols.system_prompt] = ""

    def _persona_label(p: str) -> str:
        p = (p or "").strip()
        if p in _NEUTRAL_PERSONA_MAP:
            return _NEUTRAL_PERSONA_MAP[p]
        return "custom" if p else "empty"

    df[cols.persona_label] = df[cols.system_prompt].apply(_persona_label)


    # ---- Model label (robust across providers) ----
    model_col = None
    for c in ("model", "model_id", "model_name", "model_key"):
        if c in df.columns:
            model_col = c
            break

    if model_col is None:
        df["model_label"] = "unknown"
    else:
        df["model_label"] = df[model_col].astype(str).replace("", "unknown")

    df["model_cat"] = df["model_label"].astype(str)

    # Ensure top_p exists
    if "top_p" not in df.columns:
        df["top_p"] = 1.0

    # Categorical strings (stable formatting)
    df["seed_cat"] = df[cols.seed].astype(str)
    df["temp_cat"] = df[cols.temperature].astype(str)
    df["persona_cat"] = df[cols.persona_label].astype(str)
    df["top_p_cat"] = df["top_p"].astype(str)

    # ---- Condition id (EXCLUDES seed; seed is replicate) ----
    # Include model + behavior/TACT if present, so conditions don't collide across different runs/tasks/models.
    parts = [df["model_label"].astype(str)]

    for c in ["mode", "behavior", "target", "action", "context", "time", "policy_label"]:
        if c in df.columns:
            parts.append(df[c].astype(str))
        else:
            parts.append(pd.Series([""] * len(df), index=df.index))

    parts.extend([
        df["temp_cat"].astype(str),
        df["top_p_cat"].astype(str),
        df["persona_cat"].astype(str),
    ])

    df["condition_id"] = parts[0]
    for s in parts[1:]:
        df["condition_id"] = df["condition_id"] + "|" + s

    # Replicate identifiers within condition
    df["replicate_id"] = df[cols.seed]  # seed is replicate label
    df["replicate_index"] = df.groupby("condition_id").cumcount()
    df["condition_n"] = df.groupby("condition_id")["condition_id"].transform("size")

    return df


def _ols_fit_with_ci(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit OLS y ~ 1 + x and return mean prediction + 95% CI on a grid."""
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    Xg = sm.add_constant(x_grid)
    pred = model.get_prediction(Xg).summary_frame(alpha=0.05)
    yhat = pred["mean"].to_numpy()
    lo = pred["mean_ci_lower"].to_numpy()
    hi = pred["mean_ci_upper"].to_numpy()
    return yhat, lo, hi


def _fixed_effect_terms_if_vary(df: pd.DataFrame, cols: Sequence[str]) -> str:
    """Return ' + C(col1) + C(col2) ...' for categorical cols that vary; empty string if none."""
    terms = []
    for c in cols:
        if c in df.columns and df[c].nunique(dropna=True) > 1:
            terms.append(f"C({c})")
    return " + ".join(terms)


def fit_intention_ols(
    df: pd.DataFrame,
    *,
    cols: TPBColumns = TPBColumns(),
    cluster_se: bool = True,
    cluster_col: str = "condition_id",
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    OLS baseline with nuisance controls (as fixed effects) + (optionally) cluster-robust SE by condition_id.

    Mean model includes:
      Intention ~ Attitude + SN + PBC + model/temp/persona/top_p (as nuisance fixed effects)
    """
    d = df.copy()

    nuisance = _fixed_effect_terms_if_vary(d, ["model_cat", "temp_cat", "persona_cat", "top_p_cat"])
    formula = f"{cols.intention} ~ {cols.attitude} + {cols.sn} + {cols.pbc}"
    if nuisance:
        formula += " + " + nuisance

    model = smf.ols(formula, data=d)

    if cluster_se and (cluster_col in d.columns) and (d[cluster_col].nunique() > 1):
        return model.fit(cov_type="cluster", cov_kwds={"groups": d[cluster_col]})

    return model.fit(cov_type="HC3")


def fit_intention_gee(
    df: pd.DataFrame,
    *,
    cols: TPBColumns = TPBColumns(),
    cluster: str = "condition_id",
    corr: str = "exchangeable",
):
    """
    GEE:
      - treats rows sharing the same condition_id as correlated (this is your within-condition replicate unit)
      - includes nuisance controls in the mean model, but you don't interpret them

    NOTE: Requires at least 2 rows per cluster.
    """
    d = df.copy()
    if cluster not in d.columns:
        raise ValueError(f"cluster={cluster} not found. Available: {list(d.columns)}")

    if d[cluster].value_counts().max() < 2:
        raise ValueError(f"No repeated rows per {cluster}; cannot model within-condition correlation.")

    nuisance = _fixed_effect_terms_if_vary(d, ["model_cat", "temp_cat", "persona_cat", "top_p_cat"])
    formula = f"{cols.intention} ~ {cols.attitude} + {cols.sn} + {cols.pbc}"
    if nuisance:
        formula += " + " + nuisance

    cov_struct = Exchangeable() if corr == "exchangeable" else Independence()
    model = smf.gee(formula, groups=d[cluster], data=d, cov_struct=cov_struct)
    return model.fit()


def fit_intention_mixedlm(
    df: pd.DataFrame,
    *,
    cols: TPBColumns = TPBColumns(),
    group_col: str = "condition_id",
):
    """
    MixedLM where the within-condition repeats (seed replicates) are correlated.

    Recommended for your design:
      - random intercept for condition_id (seed is replicate, not a condition)
      - no extra variance components (persona/temp) because they’re nested in condition_id
        and will often cause singular covariance in small samples.
    """
    d = df.copy()

    if group_col not in d.columns:
        raise ValueError(f"group_col={group_col} not found. Available: {list(d.columns)}")

    # must have repeated rows per group to learn within-group correlation
    if d[group_col].value_counts().max() < 2:
        raise ValueError(f"No repeated rows per {group_col}; MixedLM random intercept not identifiable.")

    formula = f"{cols.intention} ~ {cols.attitude} + {cols.sn} + {cols.pbc}"

    nuisance = _fixed_effect_terms_if_vary(d, ["model_cat", "temp_cat", "persona_cat", "top_p_cat"])
    if nuisance:
        formula += " + " + nuisance

    model = smf.mixedlm(
        formula,
        data=d,
        groups=d[group_col],
        re_formula="1",
    )

    # try a couple optimizers to reduce failure rate in small samples
    try:
        return model.fit(reml=False, method="lbfgs")
    except Exception:
        return model.fit(reml=False, method="powell", maxiter=2000)


def save_model_summary(res, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(str(res.summary()), encoding="utf-8")


# -------------------------
# Plotting (raw association)
# -------------------------
def plot_predictors_vs_intention(
    df: pd.DataFrame,
    *,
    cols: TPBColumns = TPBColumns(),
    predictors: Sequence[str] = ("attitude_mean", "subjective_norm_mean", "pbc_mean"),
    hue: Optional[str] = "model_label",
    out_path: Optional[str | Path] = None,
    show: bool = True,
    jitter_scale: float = 0.0,
) -> None:
    """Multi-panel scatter plots for predictors vs intention with OLS fit + 95% CI."""
    y_col = cols.intention
    pred_cols = [p for p in predictors if p in df.columns]
    if not pred_cols:
        raise ValueError(f"No predictor columns found from: {list(predictors)}")

    extra_cols: list[str] = []
    if "replicate_index" in df.columns:
        extra_cols.append("replicate_index")
    if "condition_id" in df.columns:
        extra_cols.append("condition_id")

    base_cols = pred_cols + [y_col] + ([hue] if (hue and hue in df.columns) else []) + extra_cols
    plot_df = df[base_cols].copy()
    plot_df = plot_df.replace("", np.nan)

    n = len(pred_cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, x_col in zip(axes, pred_cols):
        tmp = plot_df[[x_col, y_col] + ([hue] if (hue and hue in plot_df.columns) else [])].dropna()
        if tmp.empty:
            ax.set_title(f"{x_col} (no data)")
            continue

        # Optional deterministic x-jitter by replicate_index within condition_id (useful to reveal replicates)
        tmp2 = tmp.copy()
        x_plot = tmp2[x_col].to_numpy(dtype=float)

        if jitter_scale and jitter_scale > 0 and "replicate_index" in tmp2.columns:
            if "condition_id" in tmp2.columns and tmp2["condition_id"].nunique() > 1:
                centered = tmp2["replicate_index"] - tmp2.groupby("condition_id")["replicate_index"].transform("mean")
                x_plot = x_plot + jitter_scale * centered.to_numpy(dtype=float)
            else:
                centered = tmp2["replicate_index"] - tmp2["replicate_index"].mean()
                x_plot = x_plot + jitter_scale * centered.to_numpy(dtype=float)

        tmp2["_x_plot"] = x_plot

        if hue and hue in tmp2.columns:
            for label, g in tmp2.groupby(hue):
                ax.scatter(g["_x_plot"].to_numpy(), g[y_col].to_numpy(), alpha=0.7, label=str(label))
        else:
            ax.scatter(tmp2["_x_plot"].to_numpy(), tmp2[y_col].to_numpy(), alpha=0.7)

        x = tmp[x_col].to_numpy(dtype=float)
        y = tmp[y_col].to_numpy(dtype=float)
        x_grid = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        if len(x) >= 3 and np.nanstd(x) > 1e-12:
            yhat, lo, hi = _ols_fit_with_ci(x, y, x_grid)
            ax.plot(x_grid, yhat, linewidth=2)
            ax.fill_between(x_grid, lo, hi, alpha=0.2)

        pretty = x_col.replace("subjective_norm_mean", "Subjective Norm").replace("_mean", "").replace("_", " ").title()
        ax.set_xlabel(pretty)
        ax.set_ylabel("Intention (1–7)")
        ax.set_xlim(1, 7)
        ax.set_ylim(1, 7)
        ax.set_aspect("equal", adjustable="box")

    if hue and hue in plot_df.columns:
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=min(len(labels), 4),
                frameon=True,
            )
            for ax in axes:
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()

    fig.suptitle("TPB constructs vs Intention (raw; OLS fit ± 95% CI)", y=1.03, fontsize=14)
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# --------------------------------------
# Plotting (partial / controlled association)
# --------------------------------------
def _build_nuisance_terms(df: pd.DataFrame, nuisance_cols: Sequence[str]) -> str:
    """
    Build RHS terms for nuisance controls:
      - categorical controls get wrapped as C(col)
      - numeric controls used as-is
      - drops controls that don't vary (nunique <= 1)
    """
    terms = []
    for col in nuisance_cols:
        if col not in df.columns:
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue

        if pd.api.types.is_numeric_dtype(df[col]) and not col.endswith("_cat"):
            terms.append(col)
        else:
            terms.append(f"C({col})")

    return " + ".join(terms)


def _residualize_on_terms(df: pd.DataFrame, y_col: str, rhs_terms: str) -> pd.Series:
    """Return residuals of y_col after regressing on rhs_terms (or mean-center if rhs_terms empty)."""
    y = pd.to_numeric(df[y_col], errors="coerce")
    out = pd.Series(np.nan, index=df.index, name=f"{y_col}_resid")

    sub = df.copy()
    sub[y_col] = y
    sub = sub.dropna(subset=[y_col])
    if sub.empty:
        return out

    if not rhs_terms.strip():
        out.loc[sub.index] = sub[y_col] - sub[y_col].mean()
        return out

    model = smf.ols(f"{y_col} ~ {rhs_terms}", data=sub).fit()
    out.loc[sub.index] = model.resid
    return out


def plot_partial_predictors_vs_intention(
    df: pd.DataFrame,
    *,
    cols: TPBColumns = TPBColumns(),
    predictors: Sequence[str] = ("attitude_mean", "subjective_norm_mean", "pbc_mean"),
    nuisance_cols: Optional[Sequence[str]] = None,
    hue: Optional[str] = "model_label",
    out_path: Optional[str | Path] = None,
    show: bool = True,
    annotate_seed: bool = False,
    jitter_scale: float = 0.0,
    max_labels: int = 200,
) -> None:
    """
    Partial regression plots:
      - Residualize Intention and each predictor on nuisance controls
      - Plot residual predictor vs residual intention
      - Add OLS fit + 95% CI on residuals

    Default control strategy:
      - If condition_id exists: residualize on C(condition_id) (full condition control, incl. interactions)
      - Else: residualize on temp/persona/top_p/seed cats
    """
    y_col = cols.intention
    pred_cols = [p for p in predictors if p in df.columns]
    if not pred_cols:
        raise ValueError(f"No predictor columns found from: {list(predictors)}")

    if nuisance_cols is None:
        nuisance_cols = ("condition_id",) if "condition_id" in df.columns else ("model_cat", "temp_cat", "top_p_cat", "persona_cat")

    rhs_terms = _build_nuisance_terms(df, nuisance_cols)

    y_resid = _residualize_on_terms(df, y_col, rhs_terms)

    n = len(pred_cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, x_col in zip(axes, pred_cols):
        x_resid = _residualize_on_terms(df, x_col, rhs_terms)

        tmp = pd.DataFrame(
            {
                "x": x_resid,
                "y": y_resid,
                "h": df[hue] if (hue and hue in df.columns) else None,
                "seed": df["seed"] if "seed" in df.columns else None,
                "replicate_index": df["replicate_index"] if "replicate_index" in df.columns else 0,
                "condition_id": df["condition_id"] if "condition_id" in df.columns else "",
            },
            index=df.index,
        ).dropna(subset=["x", "y"])

        if tmp.empty:
            ax.set_title(f"{x_col} (no data)")
            continue

        # Deterministic jitter based on replicate_index within a condition
        x_plot = tmp["x"].to_numpy(dtype=float)
        y_plot = tmp["y"].to_numpy(dtype=float)

        if jitter_scale and jitter_scale > 0 and "replicate_index" in tmp.columns:
            # center replicate_index within each condition (e.g., for 3 reps: -1,0,+1)
            if "condition_id" in tmp.columns and tmp["condition_id"].nunique() > 1:
                centered = tmp["replicate_index"] - tmp.groupby("condition_id")["replicate_index"].transform("mean")
                x_plot = x_plot + jitter_scale * centered.to_numpy(dtype=float)
            else:
                centered = tmp["replicate_index"] - tmp["replicate_index"].mean()
                x_plot = x_plot + jitter_scale * centered.to_numpy(dtype=float)



        tmp = tmp.copy()
        tmp["x_plot"] = x_plot
        tmp["y_plot"] = y_plot

        if hue and hue in df.columns:
            for label, g in tmp.groupby("h"):
                ax.scatter(g["x_plot"].to_numpy(), g["y_plot"].to_numpy(), alpha=0.7, label=str(label))
        else:
            ax.scatter(tmp["x_plot"].to_numpy(), tmp["y_plot"].to_numpy(), alpha=0.7)


        x = tmp["x"].to_numpy(dtype=float)
        y = tmp["y"].to_numpy(dtype=float)
        x_grid = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        if len(x) >= 3 and np.nanstd(x) > 1e-12:
            yhat, lo, hi = _ols_fit_with_ci(x, y, x_grid)
            ax.plot(x_grid, yhat, linewidth=2)
            ax.fill_between(x_grid, lo, hi, alpha=0.2)

        if annotate_seed and ("seed" in tmp.columns) and len(tmp) <= max_labels:
            for _, r in tmp.iterrows():
                if pd.isna(r.get("seed")):
                    continue
                ax.annotate(
                    str(int(r["seed"])) if str(r["seed"]).isdigit() else str(r["seed"]),
                    (r["x_plot"], r["y_plot"]),
                    fontsize=8,
                    alpha=0.8,
                    xytext=(2, 2),
                    textcoords="offset points",
                )


        pretty = x_col.replace("subjective_norm_mean", "Subjective Norm").replace("_mean", "").replace("_", " ").title()
        ax.set_xlabel(f"{pretty} (residualized)")
        ax.set_ylabel("Intention (residualized)")

    if hue and hue in df.columns:
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=min(len(labels), 4),
                frameon=True,
            )
            for ax in axes:
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()

    fig.suptitle(
        "Partial TPB constructs vs Intention (controls via nuisance; OLS fit ± 95% CI)",
        y=1.03,
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def load_tpb_items(items_csv: str | Path) -> pd.DataFrame:
    """
    Load item-level TPB CSV (e.g., *_items.csv). Normalizes:
      - model label columns (model_key/model_label/model/model_id -> model_label)
      - keeps system_prompt as string (empty if missing)
      - response numeric in [1, 7]
    """
    items_csv = Path(items_csv)
    df = pd.read_csv(items_csv)

    # Preserve empty strings / NaNs in prompts
    if "system_prompt" in df.columns:
        df["system_prompt"] = df["system_prompt"].fillna("").astype(str)
    else:
        df["system_prompt"] = ""

    # Model label normalization
    if "model_label" in df.columns:
        df["model_label"] = df["model_label"].fillna("").astype(str)
    elif "model_key" in df.columns:
        df["model_label"] = df["model_key"].fillna("").astype(str)
    elif "model" in df.columns:
        df["model_label"] = df["model"].fillna("").astype(str)
    elif "model_id" in df.columns:
        df["model_label"] = df["model_id"].fillna("").astype(str)
    else:
        df["model_label"] = "unknown"

    df["model_cat"] = df["model_label"].astype("category")

    # Response numeric
    if "response" not in df.columns:
        raise ValueError("items CSV must have a 'response' column.")
    df["response"] = pd.to_numeric(df["response"], errors="coerce")

    # Item id / label
    if "item_code" in df.columns:
        df["item_id"] = df["item_code"].astype(str)
    elif "item_text" in df.columns:
        df["item_id"] = df["item_text"].astype(str)
    else:
        df["item_id"] = df.index.astype(str)

    # Optional subscale for ordering
    if "subscale" in df.columns:
        df["subscale"] = df["subscale"].fillna("").astype(str)
    else:
        df["subscale"] = ""

    df = df.dropna(subset=["response"]).copy()
    return df


def plot_item_response_boxplots_by_model(
    items_df: pd.DataFrame,
    *,
    hue: str = "model_label",
    out_path: Optional[str | Path] = None,
    show: bool = True,
    max_items: int = 24,
    ncols: int = 3,
) -> None:
    """
    For each TPB item/question, draw a small boxplot of response distributions (1-7),
    grouped by model. Useful for spotting degenerate response ranges (always 1/7, no variance).
    """
    if items_df.empty:
        raise ValueError("items_df is empty")

    if hue not in items_df.columns:
        raise ValueError(f"hue '{hue}' not found in items_df columns")

    # Choose item ordering: (subscale, item_id)
    items = (
        items_df[["subscale", "item_id"]]
        .drop_duplicates()
        .sort_values(["subscale", "item_id"])
    )
    item_list = items["item_id"].tolist()[:max_items]

    # Models ordering
    models = sorted(items_df[hue].dropna().astype(str).unique().tolist())

    n_items = len(item_list)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n_items / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.8 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, item_id in zip(axes, item_list):
        sub = items_df[items_df["item_id"] == item_id]
        data = []
        labels = []
        for m in models:
            vals = sub.loc[sub[hue].astype(str) == str(m), "response"].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            data.append(vals)
            labels.append(str(m))

        if not data:
            ax.set_title(f"{item_id} (no data)")
            ax.set_ylim(1, 7)
            continue

        ax.boxplot(
            data,
            labels=labels,
            showfliers=True,
            widths=0.6,
        )
        ax.set_title(str(item_id), fontsize=10)
        ax.set_ylim(1, 7)
        ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
        for tick in ax.get_xticklabels():
            tick.set_rotation(35)
            tick.set_ha("right")

    # Hide unused axes
    for ax in axes[n_items:]:
        ax.axis("off")

    fig.suptitle("TPB item response distributions by model (boxplots)", y=1.02, fontsize=14)
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "na"


def plot_item_response_boxplots_by_model_faceted_subscale(
    items_df: pd.DataFrame,
    *,
    hue: str = "model_label",
    out_dir: str | Path = ".",
    show: bool = False,
    max_items_per_subscale: int = 24,
    ncols: int = 3,
) -> None:
    """
    Subscale-faceted item boxplots: writes one figure per subscale into out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "subscale" not in items_df.columns:
        items_df = items_df.copy()
        items_df["subscale"] = ""

    subscales = items_df["subscale"].fillna("").astype(str).unique().tolist()
    # Ensure deterministic order, keep empty last
    subscales = sorted([s for s in subscales if s.strip() != ""]) + ([""] if any(s.strip()=="" for s in subscales) else [])

    for subscale in subscales:
        sub = items_df[items_df["subscale"].fillna("").astype(str) == str(subscale)].copy()
        if sub.empty:
            continue
        label = subscale.strip() if subscale.strip() else "unspecified"
        out_path = out_dir / f"tpb_item_boxplots_by_model__{_slug(label)}.png"
        plot_item_response_boxplots_by_model(
            sub,
            hue=hue,
            out_path=out_path,
            show=show,
            max_items=max_items_per_subscale,
            ncols=ncols,
        )


def plot_item_response_violinstrip_by_model(
    items_df: pd.DataFrame,
    *,
    hue: str = "model_label",
    out_path: Optional[str | Path] = None,
    show: bool = True,
    max_items: int = 24,
    ncols: int = 3,
    strip_jitter: float = 0.08,
) -> None:
    """
    For each TPB item/question, draw a violin plot (per model) and overlay strip points.
    Helps detect degenerate responses (no variance, always extreme).
    """
    if items_df.empty:
        raise ValueError("items_df is empty")
    if hue not in items_df.columns:
        raise ValueError(f"hue '{hue}' not found in items_df columns")

    # Item ordering: (subscale, item_id)
    items = (
        items_df[["subscale", "item_id"]]
        .drop_duplicates()
        .sort_values(["subscale", "item_id"])
    )
    item_list = items["item_id"].tolist()[:max_items]
    models = sorted(items_df[hue].dropna().astype(str).unique().tolist())

    n_items = len(item_list)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n_items / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.8 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, item_id in zip(axes, item_list):
        sub = items_df[items_df["item_id"] == item_id]
        data = []
        labels = []
        for m in models:
            vals = sub.loc[sub[hue].astype(str) == str(m), "response"].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            data.append(vals)
            labels.append(str(m))

        if not data:
            ax.set_title(f"{item_id} (no data)")
            ax.set_ylim(1, 7)
            continue

        positions = np.arange(1, len(labels) + 1, dtype=float)
        vp = ax.violinplot(data, positions=positions, showmeans=False, showmedians=True, showextrema=False)

        # overlay strip points (deterministic per item)
        rs = np.random.RandomState(abs(hash(str(item_id))) % (2**32))
        for pos, vals in zip(positions, data):
            xs = rs.normal(loc=pos, scale=strip_jitter, size=len(vals))
            ax.scatter(xs, vals, alpha=0.7, s=18)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(str(item_id), fontsize=10)
        ax.set_ylim(1, 7)
        ax.set_yticks([1, 2, 3, 4, 5, 6, 7])

    for ax in axes[n_items:]:
        ax.axis("off")

    fig.suptitle("TPB item response distributions by model (violin + strip)", y=1.02, fontsize=14)
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_item_response_violinstrip_by_model_faceted_subscale(
    items_df: pd.DataFrame,
    *,
    hue: str = "model_label",
    out_dir: str | Path = ".",
    show: bool = False,
    max_items_per_subscale: int = 24,
    ncols: int = 3,
    strip_jitter: float = 0.08,
) -> None:
    """
    Subscale-faceted violin+strip plots: writes one figure per subscale into out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "subscale" not in items_df.columns:
        items_df = items_df.copy()
        items_df["subscale"] = ""

    subscales = items_df["subscale"].fillna("").astype(str).unique().tolist()
    subscales = sorted([s for s in subscales if s.strip() != ""]) + ([""] if any(s.strip()=="" for s in subscales) else [])

    for subscale in subscales:
        sub = items_df[items_df["subscale"].fillna("").astype(str) == str(subscale)].copy()
        if sub.empty:
            continue
        label = subscale.strip() if subscale.strip() else "unspecified"
        out_path = out_dir / f"tpb_item_violinstrip_by_model__{_slug(label)}.png"
        plot_item_response_violinstrip_by_model(
            sub,
            hue=hue,
            out_path=out_path,
            show=show,
            max_items=max_items_per_subscale,
            ncols=ncols,
            strip_jitter=strip_jitter,
        )

