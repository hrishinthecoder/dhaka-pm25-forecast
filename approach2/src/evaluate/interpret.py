"""Model interpretation: SHAP (summary + dependence) and permutation importance."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance


def shap_analysis(model, X: pd.DataFrame, feat_names: list[str],
                  fig_dir: str, tag: str, top_k: int = 5) -> pd.DataFrame:
    """TreeSHAP summary plot + top-k dependence plots. Returns mean|SHAP| table."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    Xdf = pd.DataFrame(X, columns=feat_names)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xdf)

    shap.summary_plot(sv, Xdf, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(fig_dir / f"shap_summary_{tag}.png", dpi=130, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(sv).mean(axis=0)
    rank = pd.Series(mean_abs, index=feat_names).sort_values(ascending=False)
    for feat in rank.head(top_k).index:
        shap.dependence_plot(feat, sv, Xdf, show=False)
        plt.tight_layout()
        plt.savefig(fig_dir / f"shap_dep_{tag}_{feat}.png", dpi=130, bbox_inches="tight")
        plt.close()
    return rank.rename("mean_abs_shap").to_frame()


def permutation_importances(model, X_tr, y_tr, X_te, y_te,
                            feat_names: list[str], seed: int) -> pd.DataFrame:
    """Permutation importance: TRAIN estimate + TEST validation, side by side."""
    tr = permutation_importance(model, X_tr, y_tr, n_repeats=10,
                                random_state=seed, n_jobs=-1,
                                scoring="neg_root_mean_squared_error")
    te = permutation_importance(model, X_te, y_te, n_repeats=10,
                                random_state=seed, n_jobs=-1,
                                scoring="neg_root_mean_squared_error")
    df = pd.DataFrame({
        "feature": feat_names,
        "train_importance": tr.importances_mean,
        "train_std": tr.importances_std,
        "test_importance": te.importances_mean,
        "test_std": te.importances_std,
    }).sort_values("test_importance", ascending=False)
    return df.reset_index(drop=True)
