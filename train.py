"""Local smoke test. The competition evaluator calls Model.train(X, y)."""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from model import Model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", default=None, help="Optional local target column; the competition API supplies y separately.")
    args = ap.parse_args()

    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    if args.target is not None and args.target in df.columns:
        y = df.pop(args.target).to_numpy(float)
    else:
        y = np.zeros(len(df), dtype=float)
    m = Model()
    m.train(df, y)
    pred = m.predict(df)
    print("catboost:", m.uses_catboost)
    print("n_features:", len(m.feature_cols))
    print("n_predictions:", len(pred))
    print("min/max:", float(pred.min()), float(pred.max()))
    print("finite:", bool(np.isfinite(pred).all()))


if __name__ == "__main__":
    main()
