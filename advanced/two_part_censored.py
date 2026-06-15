"""Advanced technique 1 — Two-part (hurdle) model for the censored target.

The outcome is bounded [0, 100] and ~7.7% of training rows sit at *exactly*
100 (an upper-censoring / ceiling effect, like an assay saturating at its
limit of detection). A single squared-loss regressor is pulled toward that
spike. We model the two parts separately:

    prediction = (1 - p) * E[y | y < 100]  +  p * 100

where
    p              = P(y == 100), an isotonically-calibrated LightGBM classifier
    E[y | y < 100] = a regressor trained ONLY on the uncensored rows

Both parts are produced out-of-fold so the combination is leakage-free, and
scored with the year-weighted MSE (see utils.weighted_mse) that matches the
competition's shifted test distribution.

Run:  python advanced/two_part_censored.py
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

import config
from utils import load_data, make_folds, weighted_mse, year_weights, seed_everything, log

GBDT = dict(n_estimators=1500, learning_rate=0.04, num_leaves=64, subsample=0.85,
            subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, verbosity=-1)


def design_matrix(df):
    X = df.select_dtypes(include=[np.number]).drop(columns=[config.TARGET], errors="ignore").copy()
    for c in config.CAT_COLS:
        if c in df.columns:
            X[c] = df[c].astype("category")
    return X


def main():
    seed_everything()
    tr, te = load_data()
    y = tr[config.TARGET].to_numpy()
    w = year_weights(tr[config.YEAR_COL].to_numpy(), te[config.YEAR_COL].to_numpy())
    X = design_matrix(tr)
    folds = make_folds(y, years=tr[config.YEAR_COL].to_numpy())
    un = y < 100.0

    p_oof = np.zeros(len(y))           # P(y == 100)
    reg_oof = np.zeros(len(y))         # E[y | y < 100]
    plain_oof = np.zeros(len(y))       # single-model baseline (for comparison)
    for tr_idx, va_idx in folds:
        # part 1: censoring classifier
        clf = lgb.LGBMClassifier(**GBDT)
        clf.fit(X.iloc[tr_idx], (y[tr_idx] >= 100).astype(int))
        p_oof[va_idx] = clf.predict_proba(X.iloc[va_idx])[:, 1]
        # part 2: regressor on uncensored rows only
        un_tr = tr_idx[un[tr_idx]]
        reg = lgb.LGBMRegressor(**GBDT)
        reg.fit(X.iloc[un_tr], y[un_tr], sample_weight=w[un_tr])
        reg_oof[va_idx] = reg.predict(X.iloc[va_idx])
        # baseline: a single regressor on all rows
        base = lgb.LGBMRegressor(**GBDT)
        base.fit(X.iloc[tr_idx], y[tr_idx], sample_weight=w[tr_idx])
        plain_oof[va_idx] = base.predict(X.iloc[va_idx])

    # isotonic-calibrate the censoring probability, then combine
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(p_oof, (y >= 100).astype(float), sample_weight=w)
    p_cal = iso.predict(p_oof)
    hurdle = np.clip((1 - p_cal) * np.clip(reg_oof, 0, 100) + p_cal * 100, 0, 100)

    log.info("single-model      weighted MSE: %.3f", weighted_mse(y, np.clip(plain_oof, 0, 100), w))
    log.info("two-part (hurdle) weighted MSE: %.3f", weighted_mse(y, hurdle, w))
    log.info("censoring classifier OOF AUC: %.3f",
             __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score((y >= 100), p_oof))


if __name__ == "__main__":
    main()
