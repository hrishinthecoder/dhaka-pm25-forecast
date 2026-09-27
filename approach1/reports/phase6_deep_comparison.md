# Phase 6 — Deep Comparison (Bi-LSTM + attention): falsification test

_Generated 2026-06-10._

> **Caveat (write this into the manuscript):** N≈1,746 supervised daily samples is tiny for a sequence model. This Bi-LSTM+attention is an honest comparison arm, not the headline. Same target, same expanding-window walk-forward folds, same untouched hold-out, fixed config budget (3 seeds, no Optuna over-search). It is expected to be outperformed by LightGBM.


Look-back 30 days; hidden 64; heads 4; dropout 0.3; early stopping patience 20.

## Walk-forward CV (mean ± std over folds; seed-averaged predictions)

| Metric | Bi-LSTM+attn |
|---|---|
| RMSE | 36.97 ± 6.50 |
| MAE | 26.08 ± 5.10 |
| sMAPE % | 33.9 ± 10.1 |
| R² | 0.691 ± 0.114 |
| skill vs persistence | -0.116 |

## Untouched hold-out (scored once)

- RMSE **30.73**, MAE 21.19, sMAPE 26.6%, R² 0.803, skill vs persistence +0.136.

## Per-fold CV RMSE

| Fold | RMSE |
|---|---|
| 1 | 46.39 |
| 2 | 39.44 |
| 3 | 37.23 |
| 4 | 30.63 |
| 5 | 31.14 |

## Head-to-head (CV RMSE, lower = better)

| Model | CV RMSE | Holdout RMSE |
|---|---|---|
| **LightGBM (primary)** | 30.26 | 30.25 |
| Bi-LSTM+attention | 36.97 | 30.73 |
| persistence baseline | 33.12 | 35.58 |

**Interpretation:** see the verdict line vs LightGBM (30.26 CV / 30.25 holdout). If the deep model is higher (worse), that is the expected null result at this N and is reported as a finding, not hidden.