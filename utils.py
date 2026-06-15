"""Utilities: determinism, folds, metrics, and experiment tracking.

Tracking writes to two sinks: MLflow (rich UI, `mlflow ui`) and a plain
experiments.json (greppable, survives without the mlflow dependency). All
MLflow calls are wrapped so tracking failures can never kill a training run.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold

import config

log = logging.getLogger("pipeline")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                      "%H:%M:%S"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)


# ---------------------------------------------------------------- determinism
def seed_everything(seed: int = config.SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------- folds
def make_folds(y: np.ndarray, n_splits: int = config.N_FOLDS,
               seed: int = config.SEED,
               n_bins: int = config.N_TARGET_BINS,
               years: np.ndarray | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified KFold for regression via quantile-binned target.

    Plain KFold leaves fold target means ~0.5 apart on this data; stratifying
    on 20 quantile bins makes every fold see the full score distribution,
    which stabilises both early stopping and the OOF estimates the ensemble
    is fit on. When `years` is given, strata are target-bin x year so each
    fold mirrors the year mix (the target distribution drifts by year and
    the test set over-samples recent years).
    """
    if years is not None:
        bins = pd.qcut(y, q=config.N_TARGET_BINS_YEAR, labels=False,
                       duplicates="drop")
        strat = bins.astype(str) + "_" + pd.Series(years).astype(str)
    else:
        strat = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr.astype(np.int32), va.astype(np.int32))
            for tr, va in skf.split(np.zeros(len(y)), strat)]


def year_weights(train_years: np.ndarray, test_years: np.ndarray) -> np.ndarray:
    """Density-ratio weights P_test(year)/P_train(year), mean-normalised.

    Uses predictor columns only (no targets), so it is leakage-free. Aligns
    every weighted-MSE estimate with what the leaderboard actually measures.
    """
    ratio = (pd.Series(test_years).value_counts(normalize=True)
             / pd.Series(train_years).value_counts(normalize=True)).to_dict()
    w = np.array([ratio.get(a, 1.0) for a in train_years], dtype=np.float64)
    w = np.clip(w, *config.WEIGHT_CLIP)
    return w / w.mean()


# ---------------------------------------------------------------- metrics
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       w: np.ndarray | None = None) -> dict[str, float]:
    """Plain metrics, plus weighted MSE when w is given (the LB-aligned one)."""
    mse = mean_squared_error(y_true, y_pred)
    out = {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(1.0 - mse / np.var(y_true)),
    }
    if w is not None:
        out["wmse"] = float(np.average((y_true - y_pred) ** 2, weights=w))
        out["wrmse"] = float(np.sqrt(out["wmse"]))
    return out


def weighted_mse(y_true: np.ndarray, y_pred: np.ndarray,
                 w: np.ndarray | None) -> float:
    if w is None:
        return float(np.mean((y_true - y_pred) ** 2))
    return float(np.average((y_true - y_pred) ** 2, weights=w))


def clip_preds(p: np.ndarray) -> np.ndarray:
    return np.clip(p, config.CLIP_MIN, config.CLIP_MAX)


# ---------------------------------------------------------------- timing
@contextmanager
def timer(name: str):
    t0 = time.time()
    log.info("[start] %s", name)
    yield
    log.info("[done ] %s (%.1fs)", name, time.time() - t0)


# ---------------------------------------------------------------- tracking
class Tracker:
    """experiments.json + best-effort MLflow."""

    def __init__(self, run_name: str):
        self.run_name = run_name
        self.record: dict = {"run_name": run_name,
                             "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "params": {}, "metrics": {}, "tables": {}}
        self._mlflow = None
        if getattr(config, "USE_MLFLOW", False):               # opt-in; JSON is the default sink
            try:
                import mlflow
                mlflow.set_tracking_uri(config.MLFLOW_URI)
                mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
                self._mlflow = mlflow
                self._run = mlflow.start_run(run_name=run_name)
            except Exception as e:                             # noqa: BLE001
                log.warning("MLflow unavailable (%s); JSON tracking only", e)

    def _ml(self, fn: Callable, *a, **k):
        if self._mlflow is None:
            return
        try:
            fn(*a, **k)
        except Exception as e:                                 # noqa: BLE001
            log.warning("MLflow call failed: %s", e)

    def log_params(self, params: dict, prefix: str = ""):
        flat = {f"{prefix}{k}": v for k, v in params.items()}
        self.record["params"].update({k: repr(v) for k, v in flat.items()})
        if self._mlflow:
            safe = {k[:250]: str(v)[:500] for k, v in flat.items()}
            self._ml(self._mlflow.log_params, safe)

    def log_metrics(self, metrics: dict, prefix: str = ""):
        flat = {f"{prefix}{k}": float(v) for k, v in metrics.items()}
        self.record["metrics"].update(flat)
        if self._mlflow:
            self._ml(self._mlflow.log_metrics, flat)

    def log_table(self, name: str, rows):
        self.record["tables"][name] = rows

    def log_artifact(self, path):
        if self._mlflow:
            self._ml(self._mlflow.log_artifact, str(path))

    def finish(self):
        self.record["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        existing = []
        if config.EXPERIMENTS_JSON.exists():
            existing = json.loads(config.EXPERIMENTS_JSON.read_text())
        existing.append(self.record)
        config.EXPERIMENTS_JSON.write_text(json.dumps(existing, indent=2))
        if self._mlflow:
            self._ml(self._mlflow.end_run)
        log.info("tracking saved -> %s", config.EXPERIMENTS_JSON)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(config.DATA_TRAIN)
    test = pd.read_csv(config.DATA_TEST)
    return train, test
