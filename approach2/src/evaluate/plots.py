"""Diagnostic plots: predicted-vs-actual, residual ACF, residual-vs-feature,
confusion matrix. All headless (Agg)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import residual_acf


def pred_vs_actual(y, yhat, fig_dir: str, tag: str) -> None:
    fig_dir = Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    lim = [0, np.nanmax([y.max(), yhat.max()]) * 1.05]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, yhat, s=14, alpha=0.5, edgecolor="none")
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlim=lim, ylim=lim, xlabel="Actual next-day PM2.5 (µg/m³)",
           ylabel="Predicted (µg/m³)", title=f"Predicted vs actual — {tag}")
    fig.tight_layout(); fig.savefig(fig_dir / f"pred_vs_actual_{tag}.png", dpi=130)
    plt.close(fig)


def residual_acf_plot(resid, fig_dir: str, tag: str, nlags: int = 30) -> None:
    fig_dir = Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    ac = residual_acf(resid, nlags=nlags)
    n = np.sum(~np.isnan(np.asarray(resid, float)))
    ci = 1.96 / np.sqrt(max(n, 1))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.stem(range(len(ac)), ac)
    ax.axhline(ci, color="r", ls="--", lw=1); ax.axhline(-ci, color="r", ls="--", lw=1)
    ax.set(xlabel="Lag (days)", ylabel="ACF", title=f"Residual ACF — {tag}")
    fig.tight_layout(); fig.savefig(fig_dir / f"residual_acf_{tag}.png", dpi=130)
    plt.close(fig)


def residual_vs_features(resid, X: pd.DataFrame, feats: list[str],
                         fig_dir: str, tag: str) -> None:
    fig_dir = Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    feats = [f for f in feats if f in X.columns][:6]
    if not feats:
        return
    ncol = 3; nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    r = np.asarray(resid, float)
    for i, f in enumerate(feats):
        ax = axes[i // ncol][i % ncol]
        ax.scatter(X[f].values, r, s=10, alpha=0.4, edgecolor="none")
        ax.axhline(0, color="k", lw=0.8)
        ax.set(xlabel=f, ylabel="residual")
    for j in range(len(feats), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"Residual vs feature — {tag}")
    fig.tight_layout(); fig.savefig(fig_dir / f"residual_vs_features_{tag}.png", dpi=130)
    plt.close(fig)


def confusion_matrix_plot(cm: np.ndarray, labels: list[str],
                          fig_dir: str, tag: str) -> None:
    fig_dir = Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
           xticklabels=labels, yticklabels=labels,
           xlabel="Predicted AQI", ylabel="Actual AQI",
           title=f"AQI confusion matrix — {tag}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(fig_dir / f"confusion_matrix_{tag}.png", dpi=130)
    plt.close(fig)
