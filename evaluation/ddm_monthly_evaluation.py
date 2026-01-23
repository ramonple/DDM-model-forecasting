# evaluation/ddm_monthly_evaluation.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# =========================
# Target / score cleaning
# =========================
def clean_binary_target(y, *, name: str = "target") -> np.ndarray:
    """Return int numpy array of shape (n,) with values in {0,1}."""
    s = pd.Series(y).copy()
    s = pd.to_numeric(s, errors="coerce")
    if s.isna().any():
        raise ValueError(f"{name} contains NaN after numeric conversion.")
    arr = s.to_numpy().reshape(-1).astype(int)
    extra = set(np.unique(arr)) - {0, 1}
    if extra:
        raise ValueError(f"{name} must be binary 0/1. Found extra values: {sorted(extra)}")
    return arr


def clean_score(s, *, name: str = "score", clip: bool = True) -> np.ndarray:
    """Return float numpy array of shape (n,) (optionally clipped to [0,1])."""
    x = pd.Series(s).copy()
    x = pd.to_numeric(x, errors="coerce")
    if x.isna().any():
        raise ValueError(f"{name} contains NaN after numeric conversion.")
    arr = x.to_numpy().reshape(-1).astype(float)
    if clip:
        arr = np.clip(arr, 0.0, 1.0)
    return arr


# =========================
# Month parsing
# =========================
def _to_month_period(x: pd.Series) -> pd.PeriodIndex:
    """
    Convert month column to pandas PeriodIndex ('M') robustly.
    Accepts:
      - datetime
      - strings like '2023-07' or '2023-07-31'
      - pandas Period
    """
    if pd.api.types.is_period_dtype(x):
        return x.astype("period[M]")
    if pd.api.types.is_datetime64_any_dtype(x):
        return x.dt.to_period("M")
    # otherwise parse
    dt = pd.to_datetime(x, errors="coerce")
    if dt.isna().any():
        bad_n = int(dt.isna().sum())
        raise ValueError(f"month_col could not be parsed to datetime for {bad_n} rows.")
    return dt.dt.to_period("M")


# =========================
# Binomial CI helpers
# =========================
def wilson_ci(k: Union[int, float], n: Union[int, float], z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.
    Returns (lower, upper). If n==0 returns (nan,nan).
    """
    n = float(n)
    if n <= 0:
        return (np.nan, np.nan)

    p = float(k) / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) + z**2 / (4.0 * n)) / n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)


def normal_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Normal approx CI (for predicted mean proportion).
    Returns (lower, upper). If n==0 returns (nan,nan).
    """
    if n <= 0:
        return (np.nan, np.nan)
    p = float(p)
    se = np.sqrt(max(p * (1 - p), 0.0) / n)
    return (max(0.0, p - z * se), min(1.0, p + z * se))


# =========================
# Monthly aggregation core
# =========================
def month_performance_overview(
    df: pd.DataFrame,
    *,
    month_col: str,
    target_col: str,
    score_col: Optional[str] = None,
    min_n: int = 1,
) -> pd.DataFrame:
    """
    Monthly overview table:
      - n
      - bad_rate (actual)
      - pred_mean (if score_col provided)
      - error = pred_mean - bad_rate (if score_col provided)
    """
    if month_col not in df.columns:
        raise KeyError(f"month_col '{month_col}' not in df.")
    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' not in df.")
    if score_col is not None and score_col not in df.columns:
        raise KeyError(f"score_col '{score_col}' not in df.")

    tmp = df[[month_col, target_col] + ([score_col] if score_col else [])].copy()
    tmp["_month"] = _to_month_period(tmp[month_col])

    # clean target
    tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce")
    tmp = tmp.dropna(subset=[target_col])
    y = tmp[target_col].astype(int)
    extra = set(y.unique()) - {0, 1}
    if extra:
        raise ValueError(f"{target_col} must be binary 0/1. Found extra values: {sorted(extra)}")

    g = tmp.groupby("_month", observed=True)

    out = pd.DataFrame(
        {
            "n": g.size().astype(int),
            "bad_rate": g[target_col].mean().astype(float),
            "bad_count": g[target_col].sum().astype(int),
        }
    )

    if score_col:
        out["pred_mean"] = g[score_col].mean().astype(float)
        out["error"] = (out["pred_mean"] - out["bad_rate"]).astype(float)

    out = out.sort_index()
    out = out[out["n"] >= int(min_n)]
    out.index = out.index.astype(str)  # readable month like '2025-07'
    return out.reset_index(names="month")





def plot_with_confidence_band(
    monthly_df: pd.DataFrame,
    *,
    month_col: str = "month",
    actual_col: str = "bad_rate",
    pred_col: str = "pred_mean",
    n_col: str = "n",
    ci: str = "normal",
    z: float = 1.96,
    title: str = "Actual vs Predicted (Monthly)",
):
    """
    Plot monthly actual vs predicted with a confidence band around predicted mean.

    """
    import plotly.graph_objects as go

    if month_col not in monthly_df.columns:
        raise KeyError(f"{month_col} not in monthly_df.")
    for c in [actual_col, pred_col, n_col]:
        if c not in monthly_df.columns:
            raise KeyError(f"{c} not in monthly_df.")

    x = monthly_df[month_col].astype(str)
    actual = monthly_df[actual_col].astype(float)
    pred = monthly_df[pred_col].astype(float)
    n = monthly_df[n_col].astype(int)

    # compute CI band for predicted
    lo = []
    hi = []
    if ci == "normal":
        for p, nn in zip(pred, n):
            l, h = normal_ci(float(p), int(nn), z=z)
            lo.append(l)
            hi.append(h)
    else:
        lo = [None] * len(pred)
        hi = [None] * len(pred)

    fig = go.Figure()

    # Confidence band (draw upper first, then fill to lower)
    if ci == "normal":
        fig.add_trace(
            go.Scatter(
                x=x,
                y=hi,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=lo,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(0, 100, 200, 0.15)",
                line=dict(width=0),
                name=f"Pred CI (z={z})",
                hoverinfo="skip",
            )
        )

    # Predicted line
    fig.add_trace(
        go.Scatter(
            x=x,
            y=pred,
            mode="lines+markers",
            name="Predicted",
            line=dict(color="blue"),
            hovertemplate="Month=%{x}<br>Pred=%{y:.3%}<extra></extra>",
        )
    )

    # Actual line
    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual,
            mode="lines+markers",
            name="Actual",
            line=dict(color="red"),
            hovertemplate="Month=%{x}<br>Actual=%{y:.3%}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Rate",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        hovermode="x unified",
        template="plotly_white",
    )

    return fig



# ===========================================
# Monthly evaluation views (DDM-style)
# ===========================================
def evaluation_monthly_view_probability(
    df: pd.DataFrame,
    *,
    month_col: str,
    target_col: str,
    score_col: str,
    min_n: int = 1,
) -> pd.DataFrame:
    """
    Wrapper to produce the monthly table used in DDM reporting.
    """
    return month_performance_overview(
        df,
        month_col=month_col,
        target_col=target_col,
        score_col=score_col,
        min_n=min_n,
    )


def evaluation_monthly_view_convert_flag(
    df: pd.DataFrame,
    *,
    month_col: str,
    target_col: str,
    pred_flag_col: str,
    min_n: int = 1,
) -> pd.DataFrame:
    """
    If you have a predicted flag (0/1) instead of proba, evaluate monthly:
      pred_rate = mean(pred_flag)
      error = pred_rate - bad_rate
    """
    if pred_flag_col not in df.columns:
        raise KeyError(f"pred_flag_col '{pred_flag_col}' not in df.")

    tmp = df[[month_col, target_col, pred_flag_col]].copy()
    tmp["_month"] = _to_month_period(tmp[month_col])

    tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce")
    tmp[pred_flag_col] = pd.to_numeric(tmp[pred_flag_col], errors="coerce")
    tmp = tmp.dropna(subset=[target_col, pred_flag_col])

    y = tmp[target_col].astype(int)
    p = tmp[pred_flag_col].astype(int)
    if (set(y.unique()) - {0, 1}) or (set(p.unique()) - {0, 1}):
        raise ValueError("target_col and pred_flag_col must be binary 0/1.")

    g = tmp.groupby("_month", observed=True)
    out = pd.DataFrame(
        {
            "n": g.size().astype(int),
            "bad_rate": g[target_col].mean().astype(float),
            "bad_count": g[target_col].sum().astype(int),
            "pred_rate": g[pred_flag_col].mean().astype(float),
        }
    )
    out["error"] = out["pred_rate"] - out["bad_rate"]
    out = out.sort_index()
    out = out[out["n"] >= int(min_n)]
    out.index = out.index.astype(str)
    return out.reset_index(names="month")


def prediction_table_view(
    df: pd.DataFrame,
    *,
    month_col: str,
    target_col: str,
    score_col: str,
    min_n: int = 1,
) -> pd.DataFrame:
    """
    A slightly richer monthly table used in reports:
      - n, bad_count, bad_rate
      - pred_mean
      - abs_error, squared_error
    """
    m = month_performance_overview(df, month_col=month_col, target_col=target_col, score_col=score_col, min_n=min_n)
    m["abs_error"] = (m["error"]).abs()
    m["squared_error"] = (m["error"]) ** 2
    return m


# =========================
# Forecast-style KPI report
# =========================
@dataclass
class MonthlyForecastSummary:
    overall: Dict[str, float]
    monthly_table: pd.DataFrame
    monthly_table_formatted: pd.DataFrame


def _fmt_pct(x: float, decimals: int = 2) -> str:
    if pd.isna(x):
        return ""
    return f"{100.0 * float(x):.{decimals}f}%"


def _fmt_num(x: float, decimals: int = 4) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.{decimals}f}"


def monthly_forecast_report(
    df: pd.DataFrame,
    *,
    month_col: str,
    target_col: str,
    score_col: str,
    min_n: int = 1,
    z: float = 1.96,
    include_plot: bool = True,
    plot_title: str = "Actual vs Predicted (Monthly)",
) -> MonthlyForecastSummary:
    """
    Build the main DDM-style monthly forecast report.

    Outputs:
      - monthly_table: numeric values
      - monthly_table_formatted: human-friendly % strings
      - overall KPIs: MAE, RMSE, Bias, CI coverage (pred CI vs actual), etc.
      - optionally draws matplotlib plot

    CI coverage:
      We compute a normal approx band for predicted mean each month,
      and count how often actual bad_rate falls within that band.
    """
    mt = month_performance_overview(
        df,
        month_col=month_col,
        target_col=target_col,
        score_col=score_col,
        min_n=min_n,
    )

    # numeric KPIs
    errors = mt["error"].astype(float).to_numpy()
    abs_err = np.abs(errors)
    mse = np.mean(errors**2) if len(errors) else np.nan

    overall = {
        "n_months": float(len(mt)),
        "avg_month_n": float(mt["n"].mean()) if len(mt) else np.nan,
        "bias_mean_error": float(np.mean(errors)) if len(errors) else np.nan,
        "mae": float(np.mean(abs_err)) if len(abs_err) else np.nan,
        "rmse": float(np.sqrt(mse)) if not pd.isna(mse) else np.nan,
        "avg_actual_bad_rate": float(mt["bad_rate"].mean()) if len(mt) else np.nan,
        "avg_pred_bad_rate": float(mt["pred_mean"].mean()) if len(mt) else np.nan,
    }

    # CI coverage: actual inside predicted CI (normal approx)
    cover = []
    for p, a, n in zip(mt["pred_mean"].to_numpy(), mt["bad_rate"].to_numpy(), mt["n"].to_numpy()):
        lo, hi = normal_ci(float(p), int(n), z=z)
        cover.append(1.0 if (a >= lo and a <= hi) else 0.0)
    overall["pred_ci_coverage_rate"] = float(np.mean(cover)) if cover else np.nan

    # Add CI columns (pred CI + actual Wilson CI) for table
    pred_lo = []
    pred_hi = []
    act_lo = []
    act_hi = []
    for bc, nn, pm in zip(mt["bad_count"].to_numpy(), mt["n"].to_numpy(), mt["pred_mean"].to_numpy()):
        plo, phi = normal_ci(float(pm), int(nn), z=z)
        alo, ahi = wilson_ci(float(bc), float(nn), z=z)
        pred_lo.append(plo); pred_hi.append(phi)
        act_lo.append(alo); act_hi.append(ahi)

    mt["pred_ci_lo"] = pred_lo
    mt["pred_ci_hi"] = pred_hi
    mt["actual_ci_lo"] = act_lo
    mt["actual_ci_hi"] = act_hi

    # formatted table
    fmt = mt.copy()
    for c in ["bad_rate", "pred_mean", "error", "abs_error", "squared_error", "pred_ci_lo", "pred_ci_hi", "actual_ci_lo", "actual_ci_hi"]:
        if c in fmt.columns:
            if c in ["error"]:
                fmt[c] = fmt[c].map(lambda v: f"{100.0*float(v):.2f}%" if pd.notna(v) else "")
            elif c in ["squared_error"]:
                fmt[c] = fmt[c].map(lambda v: _fmt_num(v, 6))
            elif c in ["abs_error"]:
                fmt[c] = fmt[c].map(lambda v: f"{100.0*float(v):.2f}%" if pd.notna(v) else "")
            else:
                fmt[c] = fmt[c].map(lambda v: _fmt_pct(v, 2))

    # optional plot
    if include_plot and len(mt):
        _ = plot_with_confidence_band(
            mt,
            month_col="month",
            actual_col="bad_rate",
            pred_col="pred_mean",
            n_col="n",
            ci="normal",
            z=z,
            title=plot_title,
        )

    return MonthlyForecastSummary(
        overall=overall,
        monthly_table=mt,
        monthly_table_formatted=fmt,
    )


# =========================
# Segment monthly drill-down
# =========================
def segment_monthly_report_all(
    df: pd.DataFrame,
    *,
    segment_cols: Sequence[str],
    month_col: str,
    target_col: str,
    score_col: str,
    min_n: int = 1,
    z: float = 1.96,
    use_plotly: bool = True,
) -> Dict[str, Dict[str, object]]:
    """
    Build monthly tables + plots per segment.

    Returns dict:
      results[segment_label] = {
        "monthly_table": numeric_df,
        "monthly_table_formatted": formatted_df,
        "figure": fig_or_ax,
      }

    Plot:
      - if use_plotly=True and plotly is installed, returns plotly fig
      - else returns matplotlib ax
    """
    if any(c not in df.columns for c in segment_cols):
        missing = [c for c in segment_cols if c not in df.columns]
        raise KeyError(f"segment_cols missing in df: {missing}")

    results: Dict[str, Dict[str, object]] = {}

    def _seg_label(vals: Tuple) -> str:
        parts = [f"{c}={v}" for c, v in zip(segment_cols, vals)]
        return " | ".join(parts)

    for seg_vals, g in df.groupby(list(segment_cols), observed=True):
        seg_vals = (seg_vals,) if not isinstance(seg_vals, tuple) else seg_vals
        label = _seg_label(seg_vals)

        mt = month_performance_overview(
            g,
            month_col=month_col,
            target_col=target_col,
            score_col=score_col,
            min_n=min_n,
        )

        # add CI columns
        pred_lo = []
        pred_hi = []
        act_lo = []
        act_hi = []
        for bc, nn, pm in zip(mt["bad_count"].to_numpy(), mt["n"].to_numpy(), mt["pred_mean"].to_numpy()):
            plo, phi = normal_ci(float(pm), int(nn), z=z)
            alo, ahi = wilson_ci(float(bc), float(nn), z=z)
            pred_lo.append(plo); pred_hi.append(phi)
            act_lo.append(alo); act_hi.append(ahi)
        mt["pred_ci_lo"] = pred_lo
        mt["pred_ci_hi"] = pred_hi
        mt["actual_ci_lo"] = act_lo
        mt["actual_ci_hi"] = act_hi

        fmt = mt.copy()
        for c in ["bad_rate", "pred_mean", "error", "pred_ci_lo", "pred_ci_hi"]:
            if c in fmt.columns:
                if c == "error":
                    fmt[c] = fmt[c].map(lambda v: f"{100.0*float(v):.2f}%" if pd.notna(v) else "")
                else:
                    fmt[c] = fmt[c].map(lambda v: _fmt_pct(v, 2))

        fig_or_ax = None

        if use_plotly:
            try:
                import plotly.graph_objects as go

                x = mt["month"].astype(str)
                actual = mt["bad_rate"].astype(float)
                pred = mt["pred_mean"].astype(float)
                lo = mt["pred_ci_lo"].astype(float)
                hi = mt["pred_ci_hi"].astype(float)

                fig = go.Figure()

                # CI band
                fig.add_trace(go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0), showlegend=False))
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=lo,
                        mode="lines",
                        fill="tonexty",
                        line=dict(width=0),
                        name="Pred CI",
                    )
                )
                fig.add_trace(go.Scatter(x=x, y=pred, mode="lines+markers", name="Predicted"))
                fig.add_trace(go.Scatter(x=x, y=actual, mode="lines+markers", name="Actual"))

                fig.update_layout(
                    title=f"Segment Monthly: {label}",
                    xaxis_title="Month",
                    yaxis_title="Rate",
                    yaxis=dict(range=[0, 1]),
                )
                fig_or_ax = fig

            except Exception:
                # fallback to matplotlib
                fig_or_ax = plot_with_confidence_band(
                    mt,
                    month_col="month",
                    actual_col="bad_rate",
                    pred_col="pred_mean",
                    n_col="n",
                    ci="normal",
                    z=z,
                    title=f"Segment Monthly: {label}",
                )
        else:
            fig_or_ax = plot_with_confidence_band(
                mt,
                month_col="month",
                actual_col="bad_rate",
                pred_col="pred_mean",
                n_col="n",
                ci="normal",
                z=z,
                title=f"Segment Monthly: {label}",
            )

        results[label] = {
            "monthly_table": mt,
            "monthly_table_formatted": fmt,
            "figure": fig_or_ax,
        }

    return results
