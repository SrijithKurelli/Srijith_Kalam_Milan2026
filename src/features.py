"""Leakage-safe causal feature engineering for the Milan 2026 challenge."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12
REQUIRED = {"timestamp", "asset_id", "open", "high", "low", "close", "volume", "is_active"}


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build features available at each row using only observations <= that row."""
    missing = REQUIRED.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    d = data.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    g = d.groupby("asset_id", sort=False)

    close = d["close"].astype(float)
    op = d["open"].astype(float)
    high = d["high"].astype(float)
    low = d["low"].astype(float)
    volume = d["volume"].astype(float).clip(lower=0)
    active = d["is_active"].astype(bool)
    ts = d["timestamp"]

    log_close = np.log(close.clip(lower=EPS))
    ret1 = g["close"].pct_change()
    prev_close = g["close"].shift(1)
    log_volume = np.log1p(volume)

    f = pd.DataFrame(index=d.index)
    f["active"] = active.astype(float)

    # Candle / price / volume geometry.
    f["ret1"] = ret1
    f["logret1"] = log_close.groupby(d["asset_id"]).diff()
    f["oc"] = close / op.replace(0, np.nan) - 1.0
    f["gap"] = op / prev_close.replace(0, np.nan) - 1.0
    f["hl"] = high / low.replace(0, np.nan) - 1.0
    f["range_atr"] = (high - low) / close.replace(0, np.nan)
    f["close_loc"] = (close - low) / (high - low + EPS)
    f["upper_wick"] = (high - np.maximum(op, close)) / (high - low + EPS)
    f["lower_wick"] = (np.minimum(op, close) - low) / (high - low + EPS)
    f["body"] = (close - op) / (high - low + EPS)
    f["logvol"] = log_volume
    f["logvol_diff"] = g["volume"].transform(lambda s: np.log1p(s.clip(lower=0)).diff())
    f["dollar_proxy"] = np.log1p((close * volume).clip(lower=0))

    # Multi-horizon return / momentum descriptors.
    for h in (2, 3, 5, 10, 20, 30, 40, 60, 90):
        f[f"ret{h}"] = g["close"].pct_change(h)
        f[f"logret{h}"] = log_close.groupby(d["asset_id"]).diff(h)
        f[f"gap{h}"] = op / g["close"].shift(h).replace(0, np.nan) - 1.0

    for lag in range(1, 11):
        f[f"ret_lag{lag}"] = ret1.groupby(d["asset_id"]).shift(lag - 1)
        f[f"volchg_lag{lag}"] = f["logvol_diff"].groupby(d["asset_id"]).shift(lag - 1)

    # Rolling statistics. Windows are right-aligned and therefore causal.
    for w in (3, 5, 10, 20, 40, 60, 90):
        f[f"vol_{w}"] = ret1.groupby(d["asset_id"]).transform(
            lambda s: s.rolling(w, min_periods=w).std()
        )
        f[f"meanret_{w}"] = ret1.groupby(d["asset_id"]).transform(
            lambda s: s.rolling(w, min_periods=w).mean()
        )
        f[f"absret_{w}"] = ret1.groupby(d["asset_id"]).transform(
            lambda s: s.abs().rolling(w, min_periods=w).mean()
        )
        f[f"medret_{w}"] = ret1.groupby(d["asset_id"]).transform(
            lambda s: s.rolling(w, min_periods=w).median()
        )
        f[f"skew_{w}"] = ret1.groupby(d["asset_id"]).transform(
            lambda s: s.rolling(w, min_periods=w).skew()
        )
        f[f"kurt_{w}"] = ret1.groupby(d["asset_id"]).transform(
            lambda s: s.rolling(w, min_periods=w).kurt()
        )
        f[f"range_mean_{w}"] = f["hl"].groupby(d["asset_id"]).transform(
            lambda s: s.rolling(w, min_periods=w).mean()
        )
        f[f"volmean_{w}"] = f["logvol"].groupby(d["asset_id"]).transform(
            lambda s: s.rolling(w, min_periods=w).mean()
        )
        f[f"volstd_{w}"] = f["logvol"].groupby(d["asset_id"]).transform(
            lambda s: s.rolling(w, min_periods=w).std()
        )
        f[f"price_ma_dist_{w}"] = close / close.groupby(d["asset_id"]).transform(
            lambda s: s.rolling(w, min_periods=w).mean()
        ) - 1.0
        f[f"ema_dist_{w}"] = close / close.groupby(d["asset_id"]).transform(
            lambda s: s.ewm(span=w, adjust=False, min_periods=w).mean()
        ) - 1.0
        f[f"zret_{w}"] = f[f"ret{min(w, 60)}"] / (f[f"vol_{w}"] * np.sqrt(w) + EPS)
        half = max(3, w // 2)
        if f"vol_{half}" in f.columns:
            f[f"vol_ratio_{w}"] = f[f"vol_{half}"] / (f[f"vol_{w}"] + EPS)
        else:
            f[f"vol_ratio_{w}"] = np.nan
        f[f"volz_{w}"] = (f["logvol"] - f[f"volmean_{w}"]) / (f[f"volstd_{w}"] + EPS)

    for span in (5, 10, 20, 40, 60):
        f[f"ewm_vol_{span}"] = ret1.groupby(d["asset_id"]).transform(
            lambda s: s.ewm(span=span, adjust=False, min_periods=span).std()
        )
        f[f"ewm_ret_{span}"] = ret1.groupby(d["asset_id"]).transform(
            lambda s: s.ewm(span=span, adjust=False, min_periods=span).mean()
        )

    # Cross-sectional features at the same timestamp, restricted to active assets.
    for col in (
        "ret1", "ret5", "ret10", "ret20", "ret60", "oc", "gap", "hl", "logvol",
        "logvol_diff", "vol_10", "vol_20", "vol_60", "price_ma_dist_20", "price_ma_dist_60",
    ):
        s = f[col].where(active)
        mean = s.groupby(ts).transform("mean")
        std = s.groupby(ts).transform("std")
        f[f"csmean_{col}"] = mean
        f[f"csstd_{col}"] = std
        f[f"csz_{col}"] = (s - mean) / (std + EPS)
        # Data-driven within-timestamp relative position; not a hard-coded asset ranking.
        f[f"csrnk_{col}"] = s.groupby(ts).rank(pct=True)
        f[f"csdev_{col}"] = s - mean

    market_ret1 = f["ret1"].where(active).groupby(ts).transform("mean")
    f["mkt_ret1"] = market_ret1
    f["mkt_ret5"] = f["ret5"].where(active).groupby(ts).transform("mean")
    f["mkt_ret20"] = f["ret20"].where(active).groupby(ts).transform("mean")
    f["mkt_disp"] = f["ret1"].where(active).groupby(ts).transform("std")
    f["idiosyn_ret1"] = f["ret1"] - market_ret1

    # Rolling beta to the contemporaneous common factor.
    for w in (20, 40, 60):
        rm = f["ret1"].groupby(d["asset_id"]).transform(lambda s: s.rolling(w, min_periods=w).mean())
        mm = market_ret1.groupby(d["asset_id"]).transform(lambda s: s.rolling(w, min_periods=w).mean())
        cov = (
            f["ret1"] * market_ret1
        ).groupby(d["asset_id"]).transform(lambda s: s.rolling(w, min_periods=w).mean()) - rm * mm
        var = (
            market_ret1 * market_ret1
        ).groupby(d["asset_id"]).transform(lambda s: s.rolling(w, min_periods=w).mean()) - mm * mm
        f[f"beta_{w}"] = cov / (var + EPS)

    # Calendar variables are known at prediction time.
    dow = d["timestamp"].dt.dayofweek.astype(float)
    month = d["timestamp"].dt.month.astype(float)
    f["dow_sin"] = np.sin(2 * np.pi * dow / 5.0)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 5.0)
    f["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    f["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    f = f.replace([np.inf, -np.inf], np.nan).copy()
    base = d[["timestamp", "asset_id", "is_active", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    return pd.concat([base, f.reset_index(drop=True)], axis=1)


def choose_feature_columns(features: pd.DataFrame) -> list[str]:
    """Return numeric model inputs; target columns are explicitly excluded."""
    excluded = {
        "timestamp", "asset_id", "is_active", "open", "high", "low", "close", "volume",
        "dow", "month", "y", "y1", "y2", "y3", "y5", "y10", "yvol",
    }
    cols: list[str] = []
    for c in features.columns:
        if c in excluded or c.startswith("y_"):
            continue
        if not pd.api.types.is_numeric_dtype(features[c]):
            continue
        if features[c].notna().mean() < 0.85:
            continue
        if features[c].nunique(dropna=True) <= 1:
            continue
        cols.append(c)
    return cols
