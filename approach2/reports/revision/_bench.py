"""Benchmark forecasters used by r03 (holdout) and r06 (rolling folds).

Every benchmark sees only rows with day-t before the test window. Linear models
standardise inside the CV pipeline (scaler refit in every tuning fold).
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet

PM_LAGS = ["pm25_t", "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7"]
ALPHAS = np.logspace(-3, 3, 25)

def _gs(est, grid, X, y):
    gs = GridSearchCV(make_pipeline(StandardScaler(), est), grid, cv=TimeSeriesSplit(5),
                      scoring="neg_root_mean_squared_error")
    return gs.fit(X, y)

def linear_benchmarks(X_tr, y_tr, X_te):
    """ridge / lasso / elastic net on all 45 features; OLS on PM lags; AR(1)."""
    out, info = {}, {}
    Xa, ya, Xb = X_tr.values, np.asarray(y_tr), X_te.values
    g = _gs(Ridge(), {"ridge__alpha": ALPHAS}, Xa, ya)
    out["ridge"] = g.predict(Xb); info["ridge"] = {"alpha": float(g.best_params_["ridge__alpha"]), "cv_rmse": float(-g.best_score_)}
    g = _gs(Lasso(max_iter=50000), {"lasso__alpha": np.logspace(-3, 1.5, 25)}, Xa, ya)
    out["lasso"] = g.predict(Xb)
    info["lasso"] = {"alpha": float(g.best_params_["lasso__alpha"]), "cv_rmse": float(-g.best_score_),
                     "n_nonzero": int(np.sum(g.best_estimator_[-1].coef_ != 0)),
                     "nonzero": [c for c, w in zip(X_tr.columns, g.best_estimator_[-1].coef_) if w != 0]}
    g = _gs(ElasticNet(max_iter=50000), {"elasticnet__alpha": np.logspace(-3, 1.5, 25),
                                         "elasticnet__l1_ratio": [0.2, 0.5, 0.8]}, Xa, ya)
    out["elastic_net"] = g.predict(Xb)
    info["elastic_net"] = {k.split("__")[1]: float(v) for k, v in g.best_params_.items()} | {"cv_rmse": float(-g.best_score_)}
    m = LinearRegression().fit(X_tr[PM_LAGS].values, ya)
    out["ols_pm_lags"] = m.predict(X_te[PM_LAGS].values)
    info["ols_pm_lags"] = {"coef": dict(zip(PM_LAGS, map(float, m.coef_))), "intercept": float(m.intercept_)}
    m = LinearRegression().fit(X_tr[["pm25_t"]].values, ya)
    out["ar1"] = m.predict(X_te[["pm25_t"]].values)
    info["ar1"] = {"phi": float(m.coef_[0]), "intercept": float(m.intercept_)}
    return out, info

def smoothed_doy_climatology(daily_train: pd.Series, window=31):
    """Day-of-year mean of valid training daily means, circular centred moving average."""
    s = daily_train.dropna()
    doy = s.groupby(s.index.dayofyear).mean().reindex(range(1, 367))
    doy = doy.interpolate(limit_direction="both")
    k = window // 2
    ext = pd.concat([doy.iloc[-k:], doy, doy.iloc[:k]])
    sm = ext.rolling(window, center=True).mean().iloc[k:-k]
    sm.index = range(1, 367)
    return sm

def naive_benchmarks(feat, train_end_excl, test_index):
    """Smoothed climatology, anomaly persistence, weekly seasonal persistence.
    `feat` = daily frame; training statistics use valid days strictly before train_end_excl."""
    tr = feat[(feat.index < train_end_excl) & feat["valid"]]["pm25_mean"]
    clim = smoothed_doy_climatology(tr)
    t = pd.DatetimeIndex(test_index); t1 = t + pd.Timedelta(days=1)
    c_t = clim.reindex(t.dayofyear).values; c_t1 = clim.reindex(t1.dayofyear).values
    pm_t = feat["pm25_mean"].reindex(t).values
    wk = feat["pm25_mean"].where(feat["valid"]).reindex(t1 - pd.Timedelta(days=7)).values  # same weekday, valid day
    return {"climatology_smoothed": c_t1, "anomaly_persistence": c_t1 + (pm_t - c_t),
            "weekly_seasonal_persistence": wk}

def arima_log(feat, train_end_excl, test_index, order=(2, 0, 1)):
    """ARIMA(order) with constant on log daily mean (valid days; gaps = missing, Kalman filter).
    Parameters estimated on days before train_end_excl, then held fixed; one-step-ahead
    forecasts for each test target day use observations up to day t only.
    Back-transform: lognormal mean, exp(mu + 0.5*var)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    s = np.log(feat["pm25_mean"].where(feat["valid"]).clip(lower=0.5)).asfreq("D")
    tr = s[s.index < train_end_excl]
    res = SARIMAX(tr, order=order, trend="c", enforce_stationarity=True).fit(disp=False, maxiter=200)
    full = res.apply(s)
    pr = full.get_prediction()
    mu, var = pr.predicted_mean, pr.var_pred_mean
    t1 = pd.DatetimeIndex(test_index) + pd.Timedelta(days=1)
    return np.exp(mu.reindex(t1).values + 0.5 * var.reindex(t1).values), {"params": res.params.to_dict(), "aic": float(res.aic)}
