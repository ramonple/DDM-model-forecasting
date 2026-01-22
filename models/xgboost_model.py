from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import GridSearchCV

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore


@dataclass
class XGBArtifact:
    model: Any
    feature_cols: List[str]
    target_col: Optional[str] = None
    best_params: Optional[Dict[str, Any]] = None

    # Monotone constraints (optional)
    monotone_map_used: Optional[Dict[str, int]] = None
    monotone_constraints: Optional[List[int]] = None  # aligned to feature_cols

    extra: Optional[Dict[str, Any]] = None


# -------------------------
# Helpers
# -------------------------
def _ensure_xgb_available() -> None:
    if XGBClassifier is None:
        raise ImportError(
            "xgboost is not installed (or failed to import). "
            "Install with: pip install xgboost"
        )


def _validate_columns(df: pd.DataFrame, cols: Sequence[str], name: str = "df") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing columns: {missing}")


def _prep_xy(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    dropna: bool = True,
):
    _validate_columns(df, list(feature_cols) + [target_col], name="training df")
    data = df[list(feature_cols) + [target_col]].copy()
    if dropna:
        data = data.dropna(subset=list(feature_cols) + [target_col])

    X = data[list(feature_cols)]
    y = data[target_col].astype(int).to_numpy()

    extra_vals = set(np.unique(y)) - {0, 1}
    if extra_vals:
        raise ValueError(f"target_col must be binary 0/1. Found extra values: {sorted(extra_vals)}")

    return X, y



# =============================
#  Build monotone constraints
# =============================

def build_monotone_constraints_from_dict(
    feature_cols: Sequence[str],
    monotone_map: Dict[str, int],
    default: int = 0,
    strict: bool = False,
) -> List[int]:
    """
    Build monotone_constraints vector aligned with feature_cols from a dict.

    monotone_map: {feature_name: -1/0/1}
    
    MONOTONE_MAP = {
    "variable 1": 1,
    "variable 2": 1,
    "variable 3": -1,
    "variable 4": -1,
    }

    """
    allowed = {-1, 0, 1}
    bad = {v for v in monotone_map.values() if int(v) not in allowed}
    if bad:
        raise ValueError(f"Invalid constraint values {bad}. Allowed: -1, 0, 1")

    if strict:
        extras = [f for f in monotone_map.keys() if f not in feature_cols]
        if extras:
            raise KeyError(f"Constraints provided for unknown features: {extras}")

    return [int(monotone_map.get(f, default)) for f in feature_cols]


# -------------------------
def _default_xgb_params():

    return dict(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=1.0,
        gamma=0.0,
        random_state=42,
        n_jobs=-1,
    )


# -------------------------
# Train (Fixed Params)
# -------------------------
def train_xgboost(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    *,
    monotone_map: Optional[Dict[str, int]] = None,
    xgb_params: Optional[Dict[str, Any]] = None,
    dropna: bool = True,
) -> XGBArtifact:
    """
    Train an XGBClassifier using fixed parameters (no grid search).
    """
    _ensure_xgb_available()

    X_train, y_train = _prep_xy(train_df, feature_cols, target_col, dropna=dropna)

    params = _default_xgb_params()
    if xgb_params:
        params.update(xgb_params)
        
    if monotone_map is not None:
        monotone_constraints = build_monotone_constraints_from_dict(
            feature_cols,monotone_map=monotone_map,default=0,strict=False,)

            if len(monotone_constraints) != len(feature_cols):
                raise ValueError("monotone_constraints length must match feature_cols.")

        params["monotone_constraints"] = list(monotone_constraints)


    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    return XGBArtifact(
        model=model,
        feature_cols=list(feature_cols),
        target_col=target_col,
        best_params=None,
        extra={
            "n_train": int(len(X_train)),
            "pos_rate_train": float(np.mean(y_train)),
            "params_used": params,
        },
    )


# -------------------------
# Train (GridSearchCV)
# -------------------------
def train_xgboost_with_tuning(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    *,
    base_params: Optional[Dict[str, Any]] = None,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    scoring: str = "roc_auc",
    cv: int = 3,
    n_jobs: int = -1,
    verbose: int = 1,
    monotone_map: Optional[Dict[str, int]] = None,
    dropna: bool = True,
) :
    """
    Train XGBClassifier with GridSearchCV.
    """
    _ensure_xgb_available()

    X_train, y_train = _prep_xy(train_df, feature_cols, target_col, dropna=dropna)

    params = _default_xgb_params()
    if base_params:
        params.update(base_params)
    params["n_jobs"] = n_jobs

    if monotone_map is not None:
        monotone_constraints = build_monotone_constraints_from_dict(
            feature_cols,monotone_map=monotone_map,default=0,strict=False,)

            if len(monotone_constraints) != len(feature_cols):
                raise ValueError("monotone_constraints length must match feature_cols.")
                
        params["monotone_constraints"] = list(monotone_constraints)

    base_model = XGBClassifier(**params)

    if param_grid is None:
        param_grid = {
            "max_depth": [3, 4],
            "learning_rate": [0.03, 0.05],
            "n_estimators": [300, 500],
            "subsample": [0.8, 0.9],
            "colsample_bytree": [0.8, 0.9],
            "reg_lambda": [1.0, 2.0],
        }

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    return XGBArtifact(
        model=best_model,
        feature_cols=list(feature_cols),
        target_col=target_col,
        best_params=dict(grid.best_params_),
        extra={
            "best_score": float(grid.best_score_),
            "cv": int(cv),
            "scoring": scoring,
            "n_train": int(len(X_train)),
            "pos_rate_train": float(np.mean(y_train)),
            "base_params_used": params,
        },
    )


# -------------------------
# Score
# -------------------------
def predict_proba(
    artifact: Union[XGBArtifact, Any],
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    *,
    allow_missing_features: bool = False,
) -> np.ndarray:
    """
    Return positive-class probabilities for df.

    If passing XGBArtifact, feature_cols defaults to artifact.feature_cols.
    If passing a raw model, feature_cols is required.
    """
    if isinstance(artifact, XGBArtifact):
        model = artifact.model
        cols = list(feature_cols) if feature_cols is not None else artifact.feature_cols
    else:
        model = artifact
        if feature_cols is None:
            raise ValueError("feature_cols is required when passing a bare model.")
        cols = list(feature_cols)

    missing = [c for c in cols if c not in df.columns]
    if missing and not allow_missing_features:
        raise KeyError(f"Scoring df is missing features: {missing}")
    if missing and allow_missing_features:
        df = df.copy()
        for c in missing:
            df[c] = np.nan

    X = df[cols]

    if not hasattr(model, "predict_proba"):
        raise TypeError("Model does not support predict_proba (expected XGBClassifier-like model).")

    return model.predict_proba(X)[:, 1].astype(float)


# -------------------------
# Save / Load
# -------------------------
def save_artifact(artifact: XGBArtifact, path: Union[str, Path]) -> None:
    """
    Save XGBArtifact using joblib (single bundle: model + features + metadata).
    Ensures parent folder exists.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(artifact, path)


def load_artifact(path: Union[str, Path]) -> XGBArtifact:
    """Load XGBArtifact saved by save_artifact."""
    return load(Path(path))
