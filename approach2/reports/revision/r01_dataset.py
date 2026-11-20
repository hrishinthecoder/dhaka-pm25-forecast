"""R01 - dataset description, coverage, missingness, and the 985 fault-code audit."""
import numpy as np, pandas as pd
from _common import *
from src.evaluate.metrics import season_of

D = load(); df, feat, features = D["df"], D["feat"], D["features"]
t = pd.to_datetime(df["datetime_utc"], utc=True); pm = pd.Series(df["openaq_pm25"].values, index=t)
obs = pm.dropna(); first, last = obs.index.min(), obs.index.max()
span = pm[first:last]
local = obs.index.tz_convert("Asia/Dhaka")
o = {"first_obs_utc": first, "last_obs_utc": last,
     "first_obs_local": local.min(), "last_obs_local": local.max(),
     "hours_in_span": int(len(span)), "hours_observed": int(span.notna().sum()),
     "pct_missing_hours": round(100 * span.isna().mean(), 2)}
d = feat.loc[local.min().normalize().tz_localize(None): local.max().normalize().tz_localize(None)]
o.update({"calendar_days": int(len(d)), "days_with_any_obs": int((d.n_hours.fillna(0) > 0).sum()),
          "valid_days_ge18h": int(d.valid.sum()), "pairs_both_valid": int(d.pair_valid.sum()),
          "pairs_complete_45": int(len(D["X"])), "n_train": int(D["tr"].sum()), "n_holdout": int(D["te"].sum()),
          "train_first": D["X_tr"].index.min().date(), "train_last": D["X_tr"].index.max().date(),
          "holdout_first": D["X_te"].index.min().date(), "holdout_last": D["X_te"].index.max().date(),
          "min": float(obs.min()), "max": float(obs.max()), "n_negative_hours": int((obs < 0).sum()),
          "n_zero_hours": int((obs == 0).sum())})
# why 1,746 -> 1,560: which feature is missing on dropped pair-valid rows
pv = feat[feat.pair_valid]
miss = pv[features].isna()
o["dropped_pairs"] = int(miss.any(axis=1).sum())
o["dropped_by_feature_top"] = miss.sum().sort_values(ascending=False).head(8).astype(int).to_dict()
o["dropped_only_pm_history"] = int((miss[[c for c in features if c.startswith('pm25_')]].any(axis=1) &
                                    ~miss[[c for c in features if not c.startswith('pm25_')]].any(axis=1)).sum())
# season / year coverage
d2 = d.copy(); d2["season"] = [season_of(m) for m in d2.index.month]; d2["year"] = d2.index.year
d2["model_row"] = d2.index.isin(D["X"].index)
by_season = d2.groupby("season").agg(days=("valid", "size"), valid_days=("valid", "sum"),
                                     pair_valid=("pair_valid", "sum"), model_rows=("model_row", "sum"))
by_season["pct_days_not_valid"] = (100 * (1 - by_season.valid_days / by_season.days)).round(1)
by_year = d2.groupby("year").agg(days=("valid", "size"), valid_days=("valid", "sum"),
                                 pair_valid=("pair_valid", "sum"), model_rows=("model_row", "sum"))
by_year["pct_days_not_valid"] = (100 * (1 - by_year.valid_days / by_year.days)).round(1)
by_season.to_csv(OUT / "r01_coverage_by_season.csv"); by_year.to_csv(OUT / "r01_coverage_by_year.csv")
ho = d2[d2.index.isin(D["X_te"].index)].groupby("season").size().to_dict()
trn = d2[d2.index.isin(D["X_tr"].index)].groupby("season").size().to_dict()
o["holdout_rows_by_season_day_t"] = ho; o["train_rows_by_season_day_t"] = trn
# longest gaps
vd = d.index[d.valid]; gaps = pd.Series(vd[1:] - vd[:-1], index=vd[:-1]).dt.days - 1
o["longest_gaps_days"] = {str(k.date()): int(v) for k, v in gaps.sort_values(ascending=False).head(6).items()}
# 985 audit
x = pd.DataFrame({"pm": obs.values}, index=local.tz_localize(None))
x["date"] = x.index.normalize()
o["value_counts_top"] = {float(k): int(v) for k, v in obs.round(1).value_counts().head(5).items()}
hi = x[x.pm >= 900]
o["hours_ge_900"] = int(len(hi)); o["hours_eq_985"] = int((x.pm == 985).sum())
o["distinct_values_ge_900"] = sorted(set(hi.pm.round(1)))
o["next_highest_below_900"] = float(x.pm[x.pm < 900].max())
per_day = hi.groupby("date").size()
info = pd.DataFrame({"n_985_hours": per_day}).join(feat[["n_hours", "valid", "pm25_mean"]])
info["in_model_rows_as_day_t"] = info.index.isin(D["X"].index)
info["in_model_rows_as_target"] = (info.index - pd.Timedelta(days=1)).isin(D["X"].index)
info["in_train_period"] = info.index < D["hs"]
info.to_csv(OUT / "r01_fault985_days.csv")
o["days_with_985"] = int(len(info)); o["valid_days_with_985"] = int(info.valid.sum())
o["invalid_days_with_985"] = int((~info.valid).sum())
o["valid985_hours_per_day_minmax"] = [int(info[info.valid].n_985_hours.min()), int(info[info.valid].n_985_hours.max())]
o["valid985_days_as_day_t_rows"] = int(info[info.valid].in_model_rows_as_day_t.sum())
o["valid985_days_as_target_rows"] = int(info[info.valid].in_model_rows_as_target.sum())
o["valid985_days_all_in_train"] = bool(info[info.valid].in_train_period.all())
o["valid985_months"] = sorted(set(info[info.valid].index.strftime("%Y-%m")))
bad = set(info[info.valid].index)
Xi = D["X"].index
win = [any((dd - pd.Timedelta(days=k)) in bad for k in range(0, 8)) for dd in Xi]
tgt = [(dd + pd.Timedelta(days=1)) in bad for dd in Xi]
o["rows_with_985day_in_history_window_t_to_t-7"] = int(np.sum(win))
o["rows_with_985day_as_target"] = int(np.sum(tgt))
o["rows_touched_any"] = int(np.sum(np.array(win) | np.array(tgt)))
o["rows_touched_in_holdout"] = int(np.sum((np.array(win) | np.array(tgt)) & D["te"]))
dump("r01_dataset.json", o); print(json.dumps(o, indent=1, default=str)); print(by_season); print(by_year); print(info.to_string())
