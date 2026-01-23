from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# =========================
# Small utilities
# =========================
def _to_numpy_1d(x) -> np.ndarray:
    if isinstance(x, (pd.Series, pd.Index)):
        return x.to_numpy()
    x = np.asarray(x)
    return x.reshape(-1)


def _ensure_binary_01(y: np.ndarray, name: str = "y") -> None:
    uniq = set(np.unique(y[~pd.isna(y)]))
    extra = uniq - {0, 1}
    if extra:
        raise ValueError(f"{name} must be binary 0/1. Found extra values: {sorted(extra)}")


def _safe_copy_df(df: pd.DataFrame) -> pd.DataFrame:
    # ensures we never mutate callers by surprise
    return df.copy(deep=False)


# ================================================
#  Clean and validate a binary target vector
# ================================================
def clean_binary_target(
    y,
    *,
    name: str = "target",
    allow_nan: bool = False,
    nan_value: int | None = None,
) -> np.ndarray:
    """
    Clean and validate a binary target vector.


    Parameters
    ----------
    y : array-like or Series
    name : str
        Name used in error messages.
    allow_nan : bool
        If False (default), raises error if NaN present.
        If True, replaces NaN with `nan_value`.
    nan_value : int or None
        Value to use for NaN replacement when allow_nan=True.
        Must be 0 or 1.

    Returns
    -------
    y_clean : np.ndarray (int, shape (n,))
    """

    # Convert to pandas Series first 
    if isinstance(y, (pd.Series, pd.Index)):
        s = y.copy()
    else:
        s = pd.Series(y)

    # Force numeric
    s = pd.to_numeric(s, errors="coerce")

    # Handle NaN
    if s.isna().any():
        if not allow_nan:
            raise ValueError(f"{name} contains NaN after numeric conversion.")
        if nan_value not in (0, 1):
            raise ValueError("nan_value must be 0 or 1 when allow_nan=True.")
        s = s.fillna(nan_value)

    # Convert to int numpy array (1-D guaranteed)
    y_arr = s.to_numpy().reshape(-1)

    #  Validate binary
    uniq = set(np.unique(y_arr))
    extra = uniq - {0, 1}
    if extra:
        raise ValueError(f"{name} must be binary 0/1. Found extra values: {sorted(extra)}")

    return y_arr.astype(int)

# example usage:
# y = clean_binary_target(train_df["bad_flag"], name="bad_flag")




# ===========================================
#    Convert predicted proba to binary flag
# ===========================================
def predicted_bad_flag_creation(
    dataset: pd.DataFrame,
    prob_col: str,
    threshold: float,
    pred_flag_col: str = "pred_flag",
) -> pd.DataFrame:
    """
    Convert predicted probability into a binary flag using a threshold.

    Returns a COPY of dataset with `pred_flag_col` added.
    """
    out = _safe_copy_df(dataset)
    if prob_col not in out.columns:
        raise KeyError(f"prob_col '{prob_col}' not found in dataset columns.")
    out[pred_flag_col] = (pd.to_numeric(out[prob_col], errors="coerce") >= threshold).astype("Int64")
    return out

# ===========================================
# Model evaluation with predicted probabilities
# ===========================================
def evaluate_model_with_proba(
    model,
    data_input: pd.DataFrame,
    feature_list: Sequence[str],
    target_flag: str,
    predicted_column_name: str = "pred_proba",
) -> Tuple[pd.DataFrame, Dict, float]:
    """
    Generic evaluator for classifiers exposing:
      - predict_proba(X) -> [:, 1]
      - predict(X)

    Works for sklearn estimators and XGBClassifier (sklearn API).

    Returns:
      (output_df_with_pred_proba, classification_report_dict, auc)
    """
    missing = [c for c in feature_list if c not in data_input.columns]
    if missing:
        raise KeyError(f"Missing features in data_input: {missing}")
    if target_flag not in data_input.columns:
        raise KeyError(f"target_flag '{target_flag}' not found in data_input columns.")

    X = data_input.loc[:, list(feature_list)]
    y_true = pd.to_numeric(data_input[target_flag], errors="coerce").astype("Int64")

    y_true_np = _to_numpy_1d(y_true)
    _ensure_binary_01(y_true_np, name=target_flag)

    # predict
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    y_pred = _to_numpy_1d(y_pred).astype(int)
    _ensure_binary_01(y_pred, name="y_pred")

    out = _safe_copy_df(data_input)
    out[predicted_column_name] = y_proba

    report = classification_report(y_true_np, y_pred, output_dict=True, zero_division=0)
    auc = float(roc_auc_score(y_true_np, y_proba))

    return out, report, auc


# ========================================
# ROC / AUC / Gini (plotting)
# ========================================
def auc_roc_gini_plot(
    data: pd.DataFrame,
    true_flag: str,
    predicted_prob_col: str,
    plot_auc: bool = True,
    plot_gini: bool = True,
    ax=None,
):
    """
    Plot ROC curve and (optionally) Lorenz curve for Gini.

    Notes:
      - Gini = 2*AUC - 1
      - Returns (auc, gini, ax)

    """
    import matplotlib.pyplot as plt  # lazy import

    if true_flag not in data.columns:
        raise KeyError(f"true_flag '{true_flag}' not in data.")
    if predicted_prob_col not in data.columns:
        raise KeyError(f"predicted_prob_col '{predicted_prob_col}' not in data.")

    y_true = pd.to_numeric(data[true_flag], errors="coerce").to_numpy()
    y_proba = pd.to_numeric(data[predicted_prob_col], errors="coerce").to_numpy()

    _ensure_binary_01(y_true, name=true_flag)

    auc = float(roc_auc_score(y_true, y_proba))
    gini = float(2 * auc - 1)

    if ax is None:
        _, ax = plt.subplots()

    if plot_auc:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        label = f"ROC (AUC={auc:.3f}"
        if plot_gini:
            label += f", Gini={gini:.3f}"
        label += ")"
        ax.plot(fpr, tpr, label=label)
        ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(True)
        return auc, gini, ax

    # If plot_auc=False but plot_gini=True, we still compute and return numbers
    return auc, gini, ax


def plot_auc_roc_from_two_datasets(
    real_flag: str,
    df1: pd.DataFrame,
    proba_df1: str,
    model1_name: str = "Model 1",
    df2: Optional[pd.DataFrame] = None,
    proba_df2: Optional[str] = None,
    model2_name: str = "Model 2",
    ax=None,
):
    """
    Compare ROC curves for up to two datasets / score columns.

    """
    import matplotlib.pyplot as plt  # lazy import

    if ax is None:
        _, ax = plt.subplots()

    def _plot_one(df: pd.DataFrame, proba_col: str, name: str):
        if real_flag not in df.columns:
            raise KeyError(f"real_flag '{real_flag}' not in df for {name}")
        if proba_col not in df.columns:
            raise KeyError(f"proba_col '{proba_col}' not in df for {name}")

        y = pd.to_numeric(df[real_flag], errors="coerce").to_numpy()
        s = pd.to_numeric(df[proba_col], errors="coerce").to_numpy()
        _ensure_binary_01(y, name=f"{name}:{real_flag}")

        fpr, tpr, _ = roc_curve(y, s)
        auc = float(roc_auc_score(y, s))
        gini = float(2 * auc - 1)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f}, Gini={gini:.3f})")
        return {"auc": auc, "gini": gini}

    summary = {}
    summary[model1_name] = _plot_one(df1, proba_df1, model1_name)

    if df2 is not None and proba_df2 is not None:
        summary[model2_name] = _plot_one(df2, proba_df2, model2_name)

    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Comparison")
    ax.legend(loc="lower right")
    ax.grid(True)

    return ax, summary


# ===================================================
#   Classification report + PR curve + TopX
# ===================================================
def classification_report_overview(
    data: pd.DataFrame,
    true_flag: str,
    predicted_prob_col: str,
    threshold: float = 0.5,
    plot_PR: bool = True,
    ax=None,
) -> Dict[str, float]:
    """
    Build a small metric dict and optionally plot PR curve.

    """
    import matplotlib.pyplot as plt  

    if true_flag not in data.columns:
        raise KeyError(f"true_flag '{true_flag}' not in data.")
    if predicted_prob_col not in data.columns:
        raise KeyError(f"predicted_prob_col '{predicted_prob_col}' not in data.")

    y_true = pd.to_numeric(data[true_flag], errors="coerce").to_numpy()
    y_proba = pd.to_numeric(data[predicted_prob_col], errors="coerce").to_numpy()
    _ensure_binary_01(y_true, name=true_flag)

    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }

    if plot_PR:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(recall, precision, label=f"PR (AP={metrics['pr_auc']:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(loc="best")

    return metrics


def plot_pr_curve_and_topx(
    df: pd.DataFrame,
    y_col: str,
    score_col: str,
    top_fracs: Sequence[float] = (0.01, 0.05, 0.10, 0.20),
    dropna: bool = True,
    title: Optional[str] = None,
    plot: bool = True,
    ax=None,
) -> pd.DataFrame:
    """
    Plot PR curve (optional) and return a Top-X summary table.

    Top-X logic:
      - rank by score_col descending
      - for each fraction f: take top k = ceil(f*n)
      - compute:
          precision_at_top (bad rate in top)
          recall_at_top
          threshold (minimum score inside top-k)
          base_rate
          lift_vs_base_rate
          pr_auc

    Returns: topx_df (sorted by top_frac)
    """
    import matplotlib.pyplot as plt  # lazy import

    if y_col not in df.columns:
        raise KeyError(f"y_col '{y_col}' not in df.")
    if score_col not in df.columns:
        raise KeyError(f"score_col '{score_col}' not in df.")

    use = df[[y_col, score_col]].copy()
    use[y_col] = pd.to_numeric(use[y_col], errors="coerce")
    use[score_col] = pd.to_numeric(use[score_col], errors="coerce")

    if dropna:
        use = use.dropna(subset=[y_col, score_col])

    if use.empty:
        raise ValueError("No rows to evaluate after filtering/dropna.")

    y = use[y_col].astype(int).to_numpy()
    s = use[score_col].astype(float).to_numpy()

    _ensure_binary_01(y, name=y_col)

    n = len(y)
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("No positive (bad) cases in y; PR curve undefined.")

    base_rate = float(y.mean())

    # PR curve
    precision_arr, recall_arr, _ = precision_recall_curve(y, s)
    pr_auc = float(average_precision_score(y, s))

    if plot:
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(recall_arr, precision_arr, label=f"PR (AP={pr_auc:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title or "Precision-Recall Curve")
        ax.grid(True)

    # TopX table
    order = np.argsort(s)[::-1]  
    rows = []

    for frac in top_fracs:
        frac = float(frac)
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"Each top_fracs value must be in (0,1]. Got {frac}")

        k = int(np.ceil(frac * n))
        k = max(k, 1)

        top_idx = order[:k]
        y_top = y[top_idx]
        s_top = s[top_idx]

        precision_at_top = float(y_top.mean())
        recall_at_top = float(y_top.sum() / n_pos)
        threshold_top = float(np.min(s_top))  # score cutoff at boundary
        lift = float(precision_at_top / base_rate) if base_rate > 0 else np.nan

        rows.append(
            {
                "top_frac": frac,
                "top_pct": frac * 100.0,
                "k": k,
                "n": n,
                "n_pos": n_pos,
                "base_rate": base_rate * 100.0,
                "precision_at_top": precision_at_top * 100.0,
                "recall_at_top": recall_at_top * 100.0,
                "threshold": threshold_top,
                "lift_vs_base_rate": lift,
                "pr_auc": pr_auc,
                "y_col": y_col,
                "score_col": score_col,
            }
        )

        # optional: annotate point on PR curve (precision@top, recall@top)
        if plot and ax is not None:
            ax.scatter([recall_at_top], [precision_at_top], label=f"Top {int(round(frac*100))}%")

    if plot and ax is not None:
        ax.legend(loc="best")

    topx_df = pd.DataFrame(rows).sort_values("top_frac").reset_index(drop=True)

    # light rounding for readability (keep threshold as-is)
    for c in ["base_rate", "precision_at_top", "recall_at_top", "lift_vs_base_rate", "pr_auc"]:
        if c in topx_df.columns:
            topx_df[c] = topx_df[c].astype(float)

    return topx_df


def explain_first_topx_row(topx_df: pd.DataFrame) -> str:
    """
    Return a human-readable explanation for the first row of a Top-X table.
    Assumes table sorted by top_frac (smallest first).
    """
    if topx_df is None or topx_df.empty:
        raise ValueError("topx_df is empty.")

    row = topx_df.iloc[0]

    top_pct = float(row["top_pct"])
    precision_at_top = float(row["precision_at_top"])
    base_rate = float(row["base_rate"])
    lift = float(row["lift_vs_base_rate"])
    recall_at_top = float(row["recall_at_top"])

    msg = (
        f"For the top {top_pct:.0f}% highest-risk accounts, the observed bad rate "
        f"(precision) is {precision_at_top:.2f}%. Compared with an overall portfolio "
        f"bad rate of {base_rate:.2f}%, this represents a lift of {lift:.2f}x over "
        f"random selection. These accounts capture {recall_at_top:.2f}% (recall) of "
        f"all bad outcomes."
    )
    return msg
