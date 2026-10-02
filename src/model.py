"""Competition model: leakage-safe medium-horizon CatBoost signal model."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from .features import build_features, choose_feature_columns

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except Exception:  # pragma: no cover
    CatBoostRegressor = None
    HAS_CATBOOST = False

from sklearn.ensemble import HistGradientBoostingRegressor

EPS = 1e-10
PRED_LOOKBACK = 128  # > 90-timestamp maximum causal lookback


class Model:
    """Shared nonlinear model following the official train/predict API.

    Final research choice: a single CatBoost model trained on a medium-horizon
    execution-aligned target constructed only from the historical training X.

    For each training row at time t, the target is the return from the next
    execution open (t+1) to the open five trading steps later (t+6):

        y5(t) = Open(t+6) / Open(t+1) - 1

    This future information is used only to construct historical training
    labels. It is never accessed at prediction time and never appears as a
    feature.
    """

    def __init__(self):
        self.model = None
        self.feature_cols: list[str] = []
        self.medians: pd.Series | None = None
        self.signal_scale = 1.0
        self.fitted = False
        self.uses_catboost = False
        self.lookback = PRED_LOOKBACK

    def _feature_matrix(self, X: pd.DataFrame, fit: bool = False):
        F = build_features(X)
        if fit:
            self.feature_cols = choose_feature_columns(F)
        if not self.feature_cols:
            raise ValueError("No usable causal features were constructed")
        A = F[self.feature_cols].replace([np.inf, -np.inf], np.nan)
        if fit:
            self.medians = A.median().fillna(0.0)
        if self.medians is None:
            raise RuntimeError("Feature imputation statistics are unavailable")
        return F, A.fillna(self.medians)

    @staticmethod
    def _auxiliary_target(F: pd.DataFrame, horizon: int = 5) -> pd.Series:
        """Build an execution-aligned future label using training X only."""
        s = F.sort_values(["asset_id", "timestamp"]).copy()
        g = s.groupby("asset_id", sort=False)
        future_open = g["open"].shift(-(horizon + 1))
        exec_open = g["open"].shift(-1)
        y = future_open / exec_open - 1.0
        return y.reindex(F.index)

    @staticmethod
    def _make_catboost(seed: int = 17) -> CatBoostRegressor:
        threads = max(1, min(4, os.cpu_count() or 1))
        return CatBoostRegressor(
            iterations=120,
            depth=5,
            learning_rate=0.03,
            l2_leaf_reg=8,
            loss_function="RMSE",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=threads,
        )

    @staticmethod
    def _fit_hgb(X: pd.DataFrame, y: np.ndarray) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            max_iter=160,
            max_leaf_nodes=15,
            learning_rate=0.03,
            l2_regularization=6.0,
            min_samples_leaf=30,
            random_state=7,
        ).fit(X, y)

    def train(self, X, y=None):
        # The official y is accepted for API compatibility. The final model
        # deliberately uses its own execution-aligned historical label derived
        # from X, so y may be None or unused without changing model behavior.
        X = pd.DataFrame(X).copy()
        X["timestamp"] = pd.to_datetime(X["timestamp"])

        order = X.sort_values(["asset_id", "timestamp"]).index.to_numpy()
        X = X.loc[order].reset_index(drop=True)

        F, A = self._feature_matrix(X, fit=True)
        active = F["is_active"].astype(bool).to_numpy()
        y5 = self._auxiliary_target(F, horizon=5)
        finite_y5 = np.isfinite(y5.to_numpy())
        finite_features = np.isfinite(A.to_numpy()).all(axis=1)
        mask = active & finite_y5 & finite_features
        if mask.sum() < 200:
            raise ValueError("Insufficient finite active rows for auxiliary target")

        train_X = A.loc[mask]
        train_y = y5.loc[mask].to_numpy()

        if HAS_CATBOOST:
            self.model = self._make_catboost(seed=17)
            self.model.fit(train_X, train_y)
            self.uses_catboost = True
        else:
            self.model = self._fit_hgb(train_X, train_y)
            self.uses_catboost = False

        train_pred = np.asarray(self.model.predict(train_X), dtype=float)
        q = float(np.quantile(np.abs(train_pred), 0.995)) if train_pred.size else 1.0
        self.signal_scale = max(q, 1e-8)
        self.fitted = True

    def predict(self, X):
        if not self.fitted or self.model is None:
            raise RuntimeError("Model must be trained before predict()")

        X = pd.DataFrame(X).copy()
        X["timestamp"] = pd.to_datetime(X["timestamp"])
        if X.empty:
            return np.empty(0, dtype=float)

        current_time = X["timestamp"].max()
        current = X[X["timestamp"].eq(current_time)].copy()
        if current.empty:
            return np.empty(0, dtype=float)

        # Only the recent causal tail is needed for all 90-period features.
        cutoff_times = np.sort(X["timestamp"].unique())[-self.lookback:]
        X_tail = X[X["timestamp"].isin(cutoff_times)].copy()
        F, A = self._feature_matrix(X_tail, fit=False)
        cur = F["timestamp"].eq(current_time)
        Ac = A.loc[cur]
        Fc = F.loc[cur]

        raw = np.asarray(self.model.predict(Ac), dtype=float)
        signal = np.clip(raw / self.signal_scale, -1.0, 1.0)

        inactive = ~Fc["is_active"].astype(bool).to_numpy()
        signal[inactive] = 0.0

        order = np.argsort(Fc["asset_id"].to_numpy())
        signal = signal[order]
        if not np.isfinite(signal).all():
            raise FloatingPointError("Non-finite prediction generated")
        return signal.astype(float)
