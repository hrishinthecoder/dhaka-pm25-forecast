# Phase 5 — Gradient Boosting (headline models)

_Generated 2026-06-10._

**Validation:** scikit-learn `TimeSeriesSplit` = expanding-window walk-forward (5 folds, gap 1). No KFold, no shuffling — test fold always strictly after train. Optuna optimised the **walk-forward CV RMSE only**; the 12-month hold-out (n=236, 2024-03-24..2025-03-23) was never seen during tuning. Dev n=1509.


Baseline bar — CV: persistence RMSE **33.12**, climatology **44.45**; hold-out persistence RMSE **35.58**.

## Walk-forward CV (mean ± std across folds) + skill vs persistence

| Model | RMSE | MAE | sMAPE % | R² | skill vs persist | skill vs clim |
|---|---|---|---|---|---|---|
| lightgbm | 30.26 ± 7.52 | 20.33 ± 4.98 | 25.2 ± 4.1 | 0.792 ± 0.102 | +0.086 | +0.319 |
| xgboost | 30.83 ± 7.14 | 21.00 ± 4.61 | 26.0 ± 4.1 | 0.785 ± 0.094 | +0.069 | +0.306 |
| random_forest | 30.77 ± 7.07 | 20.60 ± 4.86 | 25.0 ± 3.6 | 0.785 ± 0.098 | +0.071 | +0.308 |

## Untouched hold-out (scored once) + skill vs persistence

| Model | RMSE | MAE | sMAPE % | R² | skill vs persist |
|---|---|---|---|---|---|
| lightgbm | 30.25 | 20.35 | 22.6 | 0.809 | +0.150 |
| xgboost | 31.17 | 20.66 | 22.4 | 0.798 | +0.124 |
| random_forest | 30.97 | 20.75 | 22.6 | 0.800 | +0.130 |

## Per-fold CV RMSE

| Fold | lightgbm | xgboost | random_forest |
|---|---|---|---|
| 1 | 41.87 | 41.09 | 41.86 |
| 2 | 23.74 | 24.60 | 25.41 |
| 3 | 30.18 | 31.33 | 30.22 |
| 4 | 31.99 | 33.54 | 32.32 |
| 5 | 23.49 | 23.61 | 24.03 |

## Best Optuna params

- **lightgbm**: `{'n_estimators': 819, 'learning_rate': 0.013766698561403674, 'num_leaves': 188, 'max_depth': 3, 'min_child_samples': 76, 'subsample': 0.7448249795473237, 'colsample_bytree': 0.7661811832029592, 'reg_alpha': 4.822627560026533, 'reg_lambda': 0.0011200977594160535}`
- **xgboost**: `{'n_estimators': 1003, 'learning_rate': 0.05174430534450829, 'max_depth': 3, 'min_child_weight': 6, 'subsample': 0.9940501930315925, 'colsample_bytree': 0.9890084994489883, 'gamma': 3.2964825941621356, 'reg_alpha': 0.7092653086457666, 'reg_lambda': 0.0032348517533916554}`

## Verdict

Models beating persistence on CV RMSE: **lightgbm, xgboost, random_forest**.