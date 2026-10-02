"""Local reproduction of the competition portfolio construction for research."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def score_predictions(frame: pd.DataFrame, signal_col: str, return_col: str):
    """Backtest continuous signals with next-period return and 10 bps turnover cost."""
    required = {"timestamp", "asset_id", "is_active", signal_col, return_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    prev = {}
    nav = 1.0
    returns = []
    turns = []
    for _, g in frame.sort_values(["timestamp", "asset_id"]).groupby("timestamp", sort=True):
        g = g[g["is_active"].astype(bool)].copy()
        s = np.clip(g[signal_col].to_numpy(float), -1.0, 1.0)
        den = np.abs(s).sum()
        w = s / den if den > EPS else np.zeros(len(s))
        current = dict(zip(g["asset_id"].astype(int), w))
        turn = sum(abs(current.get(a, 0.0) - prev.get(a, 0.0)) for a in set(current) | set(prev))
        gross = float(np.nan_to_num(g[return_col].to_numpy(float)) @ w)
        net = gross - 0.001 * turn
        nav *= max(1e-8, 1.0 + net)
        returns.append(net)
        turns.append(turn)
        prev = current

    r = np.asarray(returns, float)
    curve = np.cumprod(1.0 + r)
    sharpe = np.sqrt(252.0) * r.mean() / (r.std(ddof=1) + EPS)
    drawdown = np.min(curve / np.maximum.accumulate(curve) - 1.0)
    return {
        "net_cumulative_return": float(nav - 1.0),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown),
        "mean_turnover": float(np.mean(turns)) if turns else 0.0,
    }
