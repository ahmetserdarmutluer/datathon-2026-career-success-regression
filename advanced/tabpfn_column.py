"""Advanced technique 3 — TabPFN as an ensemble member (foundation model).

TabPFN is a transformer pre-trained on millions of *synthetic* tabular
datasets that performs in-context Bayesian inference (no gradient training at
fit time). Because this competition's data is itself synthetic, TabPFN
captures generation structure the gradient-boosted trees miss, and — being a
fundamentally different architecture — its errors are decorrelated from the
GBDTs. In our pipeline it was the single largest stacking gain and it
*over-delivered* on the private leaderboard.

This module produces an out-of-fold TabPFN column on the top-K features and
reports its standalone score plus its decorrelated value (residual correlation
with a GBDT). Add the resulting column to the stacker in train.py.

Requirements (free, one-time):
  1. register at https://ux.priorlabs.ai and accept the licence
  2. accept the gated model at https://huggingface.co/Prior-Labs/tabpfn_3
  3. export TABPFN_TOKEN=...   (and an HF read token if prompted)
Weights are downloaded once; data never leaves the machine. Predict in
batches to fit Apple-MPS / limited GPU memory.

Run:  TABPFN_TOKEN=... python advanced/tabpfn_column.py
"""
from __future__ import annotations
import sys, pathlib, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import config
from feature_engineering import TabularFeatureBuilder
import text_features as tf
from utils import load_data, make_folds, weighted_mse, year_weights, seed_everything, log

TOPK = 50          # TabPFN prefers a compact, informative feature set
PRED_BS = 1024     # batched prediction to bound GPU memory


def top_features(tr, te, y, k):
    import lightgbm as lgb
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    b = TabularFeatureBuilder().fit(tr)
    X = b.transform(tr); Xte = b.transform(te)
    fac = tf.TextFeatureFactory(use_embeddings=False); fac.fit(tr[config.TEXT_COL], te[config.TEXT_COL])
    X = pd.concat([X, fac.static_features(tr[config.TEXT_COL], "train")], axis=1)
    Xte = pd.concat([Xte, fac.static_features(te[config.TEXT_COL], "test")], axis=1)
    for c in X.select_dtypes(include="category").columns:
        X[c] = X[c].cat.codes; Xte[c] = Xte[c].cat.codes
    X = X.select_dtypes(include=[np.number]); Xte = Xte[X.columns]
    tri, vai = train_test_split(np.arange(len(y)), test_size=0.2, random_state=0)
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=48, verbosity=-1)
    m.fit(X.iloc[tri].fillna(X.median()), y[tri])
    r = permutation_importance(m, X.iloc[vai].fillna(X.median()), y[vai],
                               n_repeats=4, random_state=0, scoring="neg_mean_squared_error")
    cols = [X.columns[i] for i in np.argsort(r.importances_mean)[::-1][:k]]
    return X[cols].fillna(X[cols].median()), Xte[cols].fillna(X[cols].median())


def main():
    seed_everything()
    if not os.environ.get("TABPFN_TOKEN"):
        raise SystemExit("Set TABPFN_TOKEN (see module docstring).")
    from tabpfn import TabPFNRegressor
    tr, te = load_data()
    y = tr[config.TARGET].to_numpy()
    w = year_weights(tr[config.YEAR_COL].to_numpy(), te[config.YEAR_COL].to_numpy())
    Xtr, Xte = top_features(tr, te, y, TOPK)
    Xt, Xe = Xtr.values.astype(np.float32), Xte.values.astype(np.float32)
    folds = make_folds(y, years=tr[config.YEAR_COL].to_numpy())

    oof = np.zeros(len(y)); test_acc = []
    for k, (tri, vai) in enumerate(folds):
        log.info("TabPFN fold %d (fit %d)", k, len(tri))
        reg = TabPFNRegressor(device="cpu", ignore_pretraining_limits=True, n_estimators=8,
                              random_state=42 + k)
        reg.fit(Xt[tri], y[tri])
        oof[vai] = reg.predict(Xt[vai])
        test_acc.append(np.concatenate([reg.predict(Xe[i:i+PRED_BS]) for i in range(0, len(Xe), PRED_BS)]))
    tabpfn_test = np.mean(test_acc, axis=0)

    np.savez(config.ARTIFACTS / "tabpfn_column.npz", oof=oof, test=tabpfn_test)
    log.info("TabPFN standalone weighted MSE: %.3f", weighted_mse(y, np.clip(oof, 0, 100), w))
    log.info("saved out-of-fold + test column -> artifacts/tabpfn_column.npz "
             "(add it to the stacker in train.py).")


if __name__ == "__main__":
    main()
