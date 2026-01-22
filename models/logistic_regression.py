from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LogisticRegression


@dataclass
class ModelArtifact:
    """Lightweight wrapper to carry a trained model + config."""
    model: LogisticRegression
    feature_cols: List[str]
    target_col: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["model"] = None
        return d


def _validate_columns(df: pd.DataFrame, cols: Sequence[str], name: str = "df") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing columns: {missing}")


def _prep_xy(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    dropna: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    _validate_columns(df, list(feature_cols) + [target_col], name="training df")

    data = df[list(feature_cols) + [target_col]].copy()
    if dropna:
        data = data.dropna(subset=list(feature_cols) + [target_col])

    X = data[list(feature_cols)]
    y = data[target_col].astype(int).to_numpy()

    extra = set(np.unique(y)) - {0, 1}
    if extra:
        raise ValueError(f"target_col must be binary 0/1. Found extra values: {sorted(extra)}")

    return X, y


def train_logistic_regression(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    *,
    lr_kwargs: Optional[Dict[str, Any]] = None,
    dropna: bool = True,
) -> ModelArtifact:
    """
    Train Logistic Regression on train_df.
    """
    lr_kwargs = lr_kwargs or {}

    X_train, y_train = _prep_xy(train_df, feature_cols, target_col, dropna=dropna)

    # Sensible defaults for many credit-risk settings; override via lr_kwargs.
    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        **lr_kwargs,
    )
    model.fit(X_train, y_train)

    return ModelArtifact(
        model=model,
        feature_cols=list(feature_cols),
        target_col=target_col,
        extra={"n_train": int(len(X_train)), "pos_rate_train": float(np.mean(y_train))},
    )


def predict_proba(
    artifact: Union[ModelArtifact, LogisticRegression],
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    *,
    proba_positive_class: bool = True,
    allow_missing_features: bool = False,
) -> np.ndarray:
    """
    Score probabilities for df.

    If artifact is ModelArtifact and feature_cols is None, uses artifact.feature_cols.
    
    """
    
    if isinstance(artifact, ModelArtifact):
        model = artifact.model
        cols = list(feature_cols) if feature_cols is not None else artifact.feature_cols
    else:
        model = artifact
        if feature_cols is None:
            raise ValueError("feature_cols is required when passing a bare sklearn model.")
        cols = list(feature_cols)

    missing = [c for c in cols if c not in df.columns]
    if missing and not allow_missing_features:
        raise KeyError(f"Scoring df is missing features: {missing}")
    if missing and allow_missing_features:
        df = df.copy()
        for c in missing:
            df[c] = np.nan

    X = df[cols]

    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2 and proba_positive_class:
            return p[:, 1].astype(float)
        return np.asarray(p, dtype=float).ravel()

    # Fallback for rare cases
    if hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(X), dtype=float).ravel()
        # convert to (0,1) best-effort
        s = np.clip(s, -50, 50)
        return 1.0 / (1.0 + np.exp(-s))

    raise TypeError("Model does not support predict_proba or decision_function.")


def coefficients_table(artifact: Union[ModelArtifact, LogisticRegression], feature_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Return a dataframe of coefficients + intercept (for interpretability).
    """
    if isinstance(artifact, ModelArtifact):
        model = artifact.model
        cols = list(feature_cols) if feature_cols is not None else artifact.feature_cols
    else:
        model = artifact
        if feature_cols is None:
            raise ValueError("feature_cols is required when passing a bare sklearn model.")
        cols = list(feature_cols)

    if not hasattr(model, "coef_"):
        raise ValueError("Model has no coef_ attribute (not fitted?).")

    coefs = np.asarray(model.coef_).ravel()
    out = pd.DataFrame({"feature": cols, "coefficient": coefs})
    intercept = float(np.asarray(model.intercept_).ravel()[0])
    out = pd.concat([out, pd.DataFrame([{"feature": "Intercept", "coefficient": intercept}])], ignore_index=True)
    return out


def save_artifact(artifact: ModelArtifact, path: Union[str, Path]) -> None:
    """
    Save ModelArtifact (including sklearn model) using joblib.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True) # mkdir: make directories
    dump(artifact, path)


def load_artifact(path: Union[str, Path]) -> ModelArtifact:
    """
    Load ModelArtifact saved by save_artifact.
    """
    return load(Path(path))
