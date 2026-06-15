"""Standalone inference: artifacts + a test CSV -> submission.csv.

Rebuilds the exact training-time feature matrix from saved transformers
(static builder, text factory with full-train meta models, target-encoding
maps, auto-interaction pairs), averages every persisted fold model per base
learner, applies the selected ensemble, clips to the [0, 100] target range.

    python inference.py [--test test_x.csv] [--out submission.csv]
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

import config
from model_training import ADAPTERS  # noqa: F401 (registers adapters & classes for unpickling)
from utils import clip_preds, log, seed_everything, timer


def build_test_matrix(test_df: pd.DataFrame) -> pd.DataFrame:
    meta = json.loads((config.ARTIFACTS / "pipeline_meta.json").read_text())
    builder = joblib.load(config.ARTIFACTS / "feature_builder.joblib")
    factory = joblib.load(config.ARTIFACTS / "text_factory.joblib")
    te_maps = joblib.load(config.ARTIFACTS / "te_maps.joblib")

    with timer("static tabular features"):
        X = builder.transform(test_df).reset_index(drop=True)
    with timer("text features"):
        static_txt, dyn_txt = factory.featurize_new(test_df[config.TEXT_COL])
    X = pd.concat([X, static_txt], axis=1)

    for a, b in meta["auto_pairs"]:
        X[f"ax_{a}__x__{b}"] = X[a] * X[b]
    for c, mp in te_maps.items():
        X[f"te_{c}"] = (test_df[c].astype(str).map(mp["map"])
                        .fillna(mp["prior"]).to_numpy())
    for name, vals in dyn_txt.items():
        X[name] = vals

    missing = set(meta["feature_names"]) - set(X.columns)
    if missing:
        raise RuntimeError(f"feature mismatch, missing: {sorted(missing)[:5]}")
    return X[meta["feature_names"]], meta


def predict(X: pd.DataFrame, meta: dict) -> np.ndarray:
    preds = []
    for m in meta["models"]:
        adapter = ADAPTERS[m]
        per_model = []
        for r in range(meta["repeats"]):
            for k in range(meta["n_folds"]):
                path = config.MODELS_DIR / f"{m}_r{r}_f{k}.joblib"
                model = joblib.load(path)
                per_model.append(adapter.predict(model, X))
        preds.append(np.mean(per_model, axis=0))
        log.info("base model %-10s mean %.3f", m, preds[-1].mean())
    P = np.column_stack(preds)
    ens = joblib.load(config.ARTIFACTS / "ensemble.joblib")
    log.info("applying ensemble: %s", meta["ensemble_method"])
    if meta.get("ensemble_uses_year_extra"):
        years = X[config.YEAR_COL].to_numpy().astype(float)
        p100 = []
        for r in range(meta["repeats"]):
            for k in range(meta["n_folds"]):
                clf = joblib.load(config.MODELS_DIR / f"p100_r{r}_f{k}.joblib")
                p100.append(clf.predict_proba(X)[:, 1])
        extra = np.column_stack([years, np.mean(p100, axis=0)])
        return clip_preds(ens.predict(P, extra=extra))
    return clip_preds(ens.predict(P))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=str(config.DATA_TEST))
    ap.add_argument("--out", default=str(config.SUBMISSION_PATH))
    args = ap.parse_args()
    seed_everything()

    test_df = pd.read_csv(args.test)
    X, meta = build_test_matrix(test_df)
    yhat = predict(X, meta)
    sub = pd.DataFrame({config.ID_COL: test_df[config.ID_COL],
                        config.TARGET: yhat})
    sub.to_csv(args.out, index=False)
    log.info("wrote %s  (n=%d, mean %.3f, std %.3f)",
             args.out, len(sub), yhat.mean(), yhat.std())


if __name__ == "__main__":
    main()
