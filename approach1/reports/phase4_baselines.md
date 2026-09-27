# Phase 4 — Baselines (the bar to beat)

_Generated 2026-06-10._ Splits: expanding-window walk-forward (5 folds, gap 1) on dev (n=1509); untouched 12-month hold-out (n=236, 2024-03-24..2025-03-23).

## Walk-forward CV (mean ± std across folds)

| Baseline | RMSE | MAE | sMAPE % | R² |
|---|---|---|---|---|
| persistence | 33.12 ± 8.87 | 22.00 ± 5.97 | 26.2 ± 4.6 | 0.747 ± 0.140 |
| climatology | 44.45 ± 7.25 | 32.07 ± 5.88 | 40.3 ± 6.9 | 0.553 ± 0.154 |
| seasonal_naive_lag7 | 50.04 ± 10.44 | 34.48 ± 8.74 | 39.5 ± 5.7 | 0.439 ± 0.215 |

## Untouched hold-out

| Baseline | RMSE | MAE | sMAPE % | R² | n |
|---|---|---|---|---|---|
| persistence | 35.58 | 23.00 | 24.0 | 0.736 | 236 |
| climatology | 36.78 | 23.77 | 26.3 | 0.718 | 236 |
| seasonal_naive_lag7 | 44.13 | 31.41 | 34.2 | 0.578 | 221 |

## Per-fold RMSE

| Fold | persistence | climatology | seasonal_naive_lag7 |
|---|---|---|---|
| 1 | 47.88 | 53.74 | 67.11 |
| 2 | 25.04 | 47.96 | 39.89 |
| 3 | 31.21 | 42.92 | 51.19 |
| 4 | 33.61 | 43.61 | 47.88 |
| 5 | 27.85 | 34.03 | 44.10 |