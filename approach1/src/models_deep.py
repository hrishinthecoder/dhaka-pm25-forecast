"""
models_deep.py — compact Bi-LSTM + attention, the Phase-6 FALSIFICATION test.

ROLE (read this before reading the numbers)
    With only ~1,746 supervised daily samples, a sequence model is EXPECTED to overfit and
    is unlikely to beat gradient boosting. This model is included as an honest comparison —
    a falsification test — NOT a contender. It uses the same target, the same walk-forward
    folds, the same hold-out, and a comparable (fixed, not over-searched) budget. If it
    underperforms LightGBM, that null result is itself a reportable finding; we do not tune
    it harder to manufacture a win.

ARCHITECTURE (deliberately small + heavily regularised)
    per-day feature vectors over a 30-day look-back
      -> Bi-LSTM (1 layer, hidden 64)
      -> single multi-head self-attention layer (4 heads), mean-pooled over time
      -> dropout 0.3 -> linear -> next-day daily-mean PM2.5.
    Adam (lr 1e-3, weight_decay 1e-4), early stopping on a time-ordered inner validation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class BiLSTMAttention(nn.Module):
    """Bi-LSTM encoder + one multi-head self-attention layer + mean-pool + linear head.

    Args:
        n_features: number of per-day input features.
        hidden: LSTM hidden size (per direction).
        heads: number of attention heads.
        dropout: dropout probability before the output head.
    """

    def __init__(self, n_features, hidden=64, heads=4, dropout=0.3, num_layers=1):
        """Build the Bi-LSTM encoder, the multi-head self-attention layer, dropout, and the
        linear output head from the given feature/hidden/head sizes."""
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attn = nn.MultiheadAttention(embed_dim=2 * hidden, num_heads=heads,
                                          batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x):
        """Encode the (B, T, F) sequence with the Bi-LSTM, self-attend over time, mean-pool
        across days, and map to a single next-day PM2.5 prediction per sample."""
        h, _ = self.lstm(x)                 # (B, T, 2*hidden)
        a, _ = self.attn(h, h, h)           # self-attention over the time axis
        pooled = a.mean(dim=1)              # mean-pool across the 30 days
        return self.head(self.drop(pooled)).squeeze(-1)


def _inner_split(n, val_frac=0.15, min_val=30):
    """Time-ordered (inner_train, inner_val) split for early stopping (no shuffling)."""
    n_val = max(min_val, int(round(n * val_frac)))
    cut = max(1, n - n_val)
    return np.arange(cut), np.arange(cut, n)


def train_one(Xtr, ytr, Xte, cfg, seed):
    """Train the model on one fold's train rows and predict its test rows.

    Standardisation is fit on TRAIN ONLY (features and target), so the test rows never
    inform scaling — the same leakage discipline as the tree pipeline. Early stopping is
    judged on an inner time-ordered validation slice of the train rows.

    Args:
        Xtr: (n_train, T, F) float array; ytr: (n_train,) targets in µg/m³.
        Xte: (n_test, T, F) float array.
        cfg: config (deep_model.* hyperparameters).
        seed: random seed for this run (we average over several seeds).

    Returns:
        np.ndarray of test predictions in µg/m³.
    """
    dc = cfg["deep_model"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cpu"

    # --- standardise features and target on TRAIN ROWS ONLY ---
    F = Xtr.shape[2]
    flat = Xtr.reshape(-1, F)
    mu, sd = np.nanmean(flat, 0), np.nanstd(flat, 0) + 1e-8
    norm = lambda A: np.nan_to_num((A - mu) / sd, nan=0.0).astype("float32")
    Xtr_n, Xte_n = norm(Xtr), norm(Xte)
    y_mu, y_sd = float(np.mean(ytr)), float(np.std(ytr) + 1e-8)
    ytr_n = ((ytr - y_mu) / y_sd).astype("float32")

    itr, ival = _inner_split(len(Xtr_n))
    to_t = lambda a: torch.tensor(a, device=device)
    Xi, yi = to_t(Xtr_n[itr]), to_t(ytr_n[itr])
    Xv, yv = to_t(Xtr_n[ival]), to_t(ytr_n[ival])

    model = BiLSTMAttention(F, dc["hidden_size"], dc["attention_heads"],
                            dc["dropout"], dc["num_layers"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=dc["learning_rate"],
                           weight_decay=dc["weight_decay"])
    loss_fn = nn.MSELoss()
    bs, best, best_state, wait = dc["batch_size"], np.inf, None, 0

    for epoch in range(dc["max_epochs"]):
        model.train()
        perm = torch.randperm(len(Xi))  # shuffle WITHIN train only (rows are independent samples)
        for s in range(0, len(Xi), bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xi[idx]), yi[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(model(Xv), yv))
        if vloss < best - 1e-4:
            best, best_state, wait = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= dc["early_stopping_patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_n = model(to_t(Xte_n)).cpu().numpy()
    return pred_n * y_sd + y_mu  # de-standardise back to µg/m³
