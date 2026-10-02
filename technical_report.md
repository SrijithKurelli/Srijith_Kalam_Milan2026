---
title: "Technical Report - Milan 2026 ML for Algorithmic Trading"
author: ""
date: ""
geometry: margin=0.72in
fontsize: 10pt
---

## 1. Objective and evaluator alignment

The submission follows the supplied `Model` API: the evaluator calls `train(X, y)` once and repeatedly calls `predict(X)` with the history available at the current prediction timestamp. The model returns one continuous signal per current asset in sorted `asset_id` order, bounded to `[-1,1]`. The competition converts signals to portfolio weights as \(w_i = s_i / sum_j abs(s_j)\), executes them at the next open, charges 10 bps on turnover, and ranks primarily by compounded net portfolio return; Sharpe ratio and maximum drawdown are secondary metrics.

The released model is a single CatBoost regressor. It is deliberately not a hand-coded trading strategy: all directional and magnitude decisions come from the trained model.

## 2. Data diagnosis

The supplied data contains 16,800 rows, 14 assets and 1,200 timestamps from approximately 2016-01-04 to 2020-08-07. The supplied file has no missing rows and no duplicate timestamp/asset pairs. Asset 3 has a long inactive period before becoming active, so inactive observations are excluded from model training and receive zero signal when present at prediction time.

The assets have heterogeneous volatility and substantial cross-sectional correlation. Volatility persistence is much stronger than raw-return persistence, and the sample has clear regime variation, especially in 2020. These observations motivate a shared nonlinear model, volatility descriptors, cross-sectional normalization, and robustness checks over chronological folds.

## 3. Causal feature engineering

All submitted features are computable from observations at or before the prediction timestamp. The maximum lookback is 90 timestamps. Feature groups are:

- returns/log returns over 1, 2, 3, 5, 10, 20, 30, 40, 60 and 90 periods;
- OHLC geometry: gap, open-close return, high-low range, body, wick and close-location features;
- lagged return/volume-change features;
- rolling volatility, mean/median return, absolute-return mean, skewness, kurtosis, range and volume statistics;
- EMA distance, price-to-rolling-mean distance, volatility ratios and volume z-scores;
- EWMA return/volatility features;
- cross-sectional mean, standard deviation, z-score, percentile rank and deviation for contemporaneous active assets;
- common-factor, dispersion and idiosyncratic return features plus rolling beta to the common factor;
- calendar sine/cosine descriptors.

Cross-sectional ranks are data-derived features, not hard-coded asset rankings or allocations. Rolling windows are right-aligned; no centered windows or future rows are used. Feature columns explicitly exclude target-like names. Imputation statistics are fitted only from training history.

## 4. Research methodology and model families

The research compared regularized linear models (Ridge, ElasticNet), tree ensembles (ExtraTrees, HistGradientBoosting), boosted models (XGBoost, LightGBM, CatBoost), rankers, sequence models (GRU, MLP, TCN and graph-TCN), and compact Transformer-style families inspired by TFT, PatchTST, iTransformer, TSMixer, TimeMixer and Stockformer. Lightweight RL/policy experiments were also tested. N-BEATS was checked on an independent financial series.

The conclusion was not that nonlinear models are universally best. On generic public time series, regularized linear models often won. The competition data is instead a small multivariate financial problem where CatBoost was the strongest repeatable nonlinear candidate without requiring a large sequence model, GPU, pretrained checkpoint or external data.

Foundation-style models such as TimesFM/Chronos and newer multivariate forecasting models were reviewed but not placed in the submission because the competition requires the final solution to run without internet/external data dependencies. A self-contained CatBoost pipeline is easier to audit for provenance and leakage.

## 5. Target research and temporal validation

Because the public competition CSV contains OHLCV rather than a visible target column, the final model constructs a training-only auxiliary label from historical training observations. For row \(t\), the label is

```text
Open(t+6) / Open(t+1) - 1
```

which represents a five-step holding return beginning at the next execution open. Future values are used only to construct historical labels inside `train()`; they are never features and are never accessed by `predict()`.

Multiple horizons (1, 2, 3, 5 and 10 steps), volatility-normalized targets and target/model blends were tested. The medium five-step target was the most robust target family on the competition data and was also competitive on an independent public MSFT check.

Validation is chronological only. Research used later 120-timestamp blocks as validation periods with no random train/test split. Candidate selection examined net cumulative return together with Sharpe ratio, maximum drawdown, turnover and fold stability. The local backtester implements the next-open execution convention and the official 0.001 turnover cost.

## 6. Competition-data results

Representative honest research results are shown below. Percentages are net portfolio returns after transaction costs. The strongest direct one-step CatBoost run had high regime variance, while the medium-horizon target reduced that instability.

| Model | Mean net | Worst fold | Mean Sharpe | Mean max drawdown |
|---|---:|---:|---:|---:|
| Ridge | -1.72% | -14.11% | -0.86 | -13.50% |
| ElasticNet | -2.14% | -14.08% | -0.88 | -13.93% |
| HistGradientBoosting | -9.76% | -23.25% | -0.56 | -26.03% |
| XGBoost | -1.61% | -18.79% | -0.30 | -23.25% |
| LightGBM | -0.30% | -21.18% | -0.24 | -24.66% |
| CatBoost, one-step | +14.19% | -28.67% | +0.41 | -22.04% |
| MLP-window | +0.75% | -2.39% | +0.45 | -3.16% |
| **CatBoost, five-step** | **+26.68%** | **+1.33%** | **+1.65** | **-20.05%** |

The five-step result above used a 160-tree research configuration. An iteration sweep found the 120-tree setting competitive while reducing capacity and runtime; the released model therefore uses 120 trees. The result table is used as model-family/target evidence, not as a claim about the hidden test period.

## 7. External robustness checks

A separate public MSFT OHLCV check used the same broad execution convention. The 5-step CatBoost variant achieved +2.29% net return with Sharpe 0.47, compared with +0.98% for one-step and +1.28% for two-step. Generic pmdarima series (AirPassengers, Austres, Wineind, Lynx) were also used as sanity checks; no single architecture won every series, supporting the decision to select from financial-domain robustness rather than universal forecasting rankings.

These external checks are research-only. The submission does not download or use external data during evaluation.

## 8. Final architecture and computational requirements

The released model is:

```text
253 causal features
        -> CatBoostRegressor
           iterations=120
           depth=5
           learning_rate=0.03
           l2_leaf_reg=8
        -> training-derived 99.5th-percentile output scale
        -> clip to [-1, 1]
```

The model uses CPU execution and the supplied `requirements.txt` contains only NumPy, pandas, scikit-learn and CatBoost. Training is lightweight on the supplied dataset. At prediction time only the latest 128 timestamps are reconstructed because all submitted features need at most a 90-timestamp lookback.

## 9. Rule-compliance and leakage audit

- Random train/test splitting: **PASS - not used**.
- Future/test observations as input features: **PASS - not used**.
- Future observations for training labels: **PASS - labels only, inside training history**.
- Target leakage into features: **PASS - target-like columns excluded**.
- Test-specific logic or evaluator exploitation: **PASS - none**.
- Hand-coded buy/sell or technical-indicator strategy: **PASS - none; indicators are causal ML features**.
- Hard-coded asset rankings/allocations: **PASS - none**.
- Hard-coded trading thresholds: **PASS - none; output scale is fitted from training predictions**.
- Continuous signals in `[-1,1]`: **PASS**.
- Correct asset ordering: **PASS**.
- Inactive assets masked: **PASS**.
- Next-open objective and 10 bps transaction cost represented in research backtesting: **PASS**.
- No external download/API dependency: **PASS**.
- Official `Model.train(X,y)` / `Model.predict(X)` interface: **PASS**.

## 10. Limitations

The organizer's hidden evaluator, hidden test period and exact construction of the supplied `y` are not public. Therefore offline experiments cannot certify a hidden-test winner. The final choice is the strongest self-contained, leakage-audited candidate supported by chronological financial experiments and independent robustness checks. No result in this report is a hidden-test guarantee.
