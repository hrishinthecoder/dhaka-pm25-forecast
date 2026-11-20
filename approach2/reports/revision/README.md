# Revision analyses (manuscript AIRQ-D-26-01116)

New analyses for the major revision. Nothing here changes the frozen v2.0 artifacts in
`models/final/` or their SHA-256 checksums. Run every script from `approach2/` with
`PYTHONPATH=reports/revision` and **xgboost==3.2.0** (other versions do not reproduce the
frozen refit).

| Script | Output | Content |
|---|---|---|
| `r00_reproduce_frozen.py` | `r00_reproduce_frozen.json` | Frozen model and a fresh refit both give RMSE 30.80, R2 0.795 (n = 219) |
| `r01_dataset.py` | `r01_dataset.json`, `r01_coverage_by_*.csv`, `r01_fault985_days.csv` | Record span, missing hours, valid days, pairs, seasonal coverage, 985 fault-code audit |
| `r02_qc985_sensitivity.py` | `r02_qc985_sensitivity.json`, `r02_hours_ge900.csv` | Mask hourly values >= 900, rebuild, retrain |
| `r03_benchmarks.py` | `r03_benchmarks.csv`, `r03_holdout_predictions.csv`, `r03_benchmark_info.json` | Linear, autoregressive and naive benchmarks on the same holdout |
| `r04_ablation.py` | `r04_ablation.csv`, `r04_high_corr_pairs.csv`, `r04_groups_corr.json` | Feature-group ablation, feature correlation |
| `r05_significance.py` | `r05_significance.json`, `r05_bootstrap_intervals.json` | DM specification and variants, Holm, moving-block bootstrap, R2 definition |
| `r06_rolling_folds.py`, `r06b_rolling_pooled.py` | `r06_fold_dates.csv`, `r06_per_fold.csv`, `r06_pooled.csv`, `r06_rolling_predictions.csv` | Fold dates, per-fold and pooled tests |
| `r07_extremes.py` | `r07_error_by_stratum.csv`, `r07_aqi_per_class.csv`, `r07_confusion_*.csv`, `r07_extremes.json` | Error by concentration range and season, AQI recall |
| `r08_group_shap.py` | `r08_group_shap.json`, `r08_shap_per_feature.csv` | Group-level SHAP for the frozen XGBoost model |
| `r09_common_sample_replication.py` | `r09_common_sample_replication.json` | Approach 1 LightGBM and Approach 2 XGBoost on the same holdout days |
| `r10_feature_table.py` | `r10_feature_table.csv`, `r10_feature_table_notes.json` | All 45 features with definition, timing, latency and near-real-time substitute |
| `r11_pooling_evidence.py` | `r11_pooling_evidence.json`, `r11_pooled_share_by_year.csv`, `r11_stations_resolved.csv` | Station composition of the pooled target, from the pooled master in this repository (first pass; superseded by r16) |
| `r12_firms_timing.py` | `r12_firms_timing.json` | UTC/local-day alignment of the FIRMS features |
| `r13_figures.py` | `figures/fig1`-`fig5` (1200 dpi) | Revised manuscript figures |
| `r14_extra_checks.py` | `r14_extra_checks.json`, `r14_longest_gap_by_year.csv` | Yearly data gap, macro-F1 over classes present, PM2.5-ERA5 correlations |
| `r15_number_trace.py` | `r15_number_trace_output.txt` | Screens every number in a .docx against the output files |
| `r16_pooling_from_raw.py` | `r16_pooling_from_raw.json`, `r16_sensor_inventory.csv`, `r16_pooled_share_by_year.csv`, `r16_overlap_with_24434.csv` | Station composition from the raw per-sensor OpenAQ files (the numbers used in the paper). Needs the raw download, which is not redistributed |

Random forest and LightGBM numbers re-run here differ slightly from `reports/metrics.csv`
(30.66 vs 30.64 and 31.69 vs 31.52) because `requirements.txt` does not pin scikit-learn or
lightgbm. XGBoost reproduces exactly.
