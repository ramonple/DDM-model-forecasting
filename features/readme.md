
---

## README

```markdown
# Features

This project reuses generic feature engineering and feature selection utilities from
the RuleEngine repository (`feature_selection/` module).

Only DDM-specific feature logic should be implemented in this folder:

---
### Planned folder layout:

features/
  README.md
  ddm_feature_engineering.py
  monthly_aggregation.py
  time_index.py
  ddm_schema.py
  config/
    feature_sets.yaml
    monotone_map.yaml

1) DDM specific feature engineering: ddm_feature_engineering.py:
- vintange analysis / tenure features
- lifecycle/ maturity flags: immature month handling rules
- Lag features: lagged behaviour features (t-1, t-2) if avaiable
- Rolling windows: rolling average/sum on account time series (3M/6M)
- policy / rule timing
- score alignment


2) Aggregation layer: monthly_aggregation.py
- Aggregate predicted proba to monthly portfolio PD / bad rate
- Support weights (exposure, balance, account weight, application volume)
- Support segment aggregation (channel, product, risk tier)
- Output standard monthly tables: n_accounts, n_labeled, actual_rate, pred_rate, error
- Optional: binomial CI/Wilson CI helpers can live here or evaluation.

3) Calendar/time utilities tied to the pipeline:  time_index.py
- Parse month column to Period('M') consistently
- Create month_id, sorting, filtering ranges
- Create month_id, sorting, filtering ranges

4) DDM data contract and column management: ddm_schema.py
Keeps notebooks faster/cleaner:
- centralise column names
- validation helpers

5) optional: feature config: /config/
