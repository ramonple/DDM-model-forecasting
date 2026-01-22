# DDM Model Evaluation & Forecasting Toolkit

This repository provides a modular framework for training, evaluating, and monitoring
credit risk / delinquency (DDM) models.

It supports:
- Standard binary classification evaluation (ROC, PR, Top-X, metrics)
- Monthly forecasting-style evaluation (batch bad rate, confidence bands, stability diagnostics)
- Multiple model types (Logistic Regression, XGBoost, rules-based scores)

The design separates **model training**, **generic evaluation**, and **DDM-specific monthly reporting**
so components can be reused across projects.

<img src ="https://github.com/ramonple/DDM-model-forecasting/blob/729acd903766bc3a6813c57a35d8e077602a32d6/DDM%20Forecasting%20icon.png">


---


## Key Concepts

### 1. Model-Agnostic Evaluation
Functions in `evaluation/classification_evaluation.py` operate on:
- true labels (`y_true`)
- predicted probabilities (`pred_proba`)

They can be used with:
- Logistic Regression  
- XGBoost  
- Any model that outputs a score  

Includes:
- ROC / AUC / Gini
- Precision-Recall curve
- Top-X capture analysis
- Classification reports

---

### 2. DDM Monthly Forecasting Evaluation
Functions in `evaluation/ddm_monthly_evaluation.py` evaluate models at the **monthly batch level**:

- Actual vs Predicted monthly bad rate
- Confidence intervals and coverage diagnostics
- Bias / MAE / RMSE over time
- Segment-level monthly drill-down

This layer reflects how DDM models are consumed in production
(decisions based on monthly average risk rather than individual accounts).

---

## Typical Workflow

```python
# 1. Train model
from models.xgboost_model import train_xgboost_with_tuning
model = train_xgboost_with_tuning(...)

# 2. Score datasets
df["pred_proba"] = model.predict_proba(df[feature_cols])[:, 1]

# 3. Standard evaluation
from evaluation.classification_evaluation import plot_pr_curve_and_topx
plot_pr_curve_and_topx(y_true, df["pred_proba"])

# 4. Monthly DDM forecasting evaluation
from evaluation.ddm_monthly_evaluation import monthly_forecast_report
monthly_forecast_report(df, month_col="written_month",
                        target_col="bad_flag",
                        score_col="pred_proba")


