"""Model zoo, fold training, and Optuna HPO.

Missing-value policy per model (compared in the ablation stage):
  - CatBoost / LightGBM / XGBoost / HistGB consume NaNs natively (learned
    default split direction) — imputation would only destroy the signal that
    the missing-indicators already expose.
  - ExtraTrees / MLP cannot accept NaNs -> fold-fit median imputation
    (medians from the fold's training rows only). A KNN-imputation variant
    exists for the comparison required by the spec.

Categorical policy:
  - CatBoost: native (ordered target statistics) — its headline strength.
  - LightGBM: native partition-based categorical splits (category dtype).
  - XGBoost: native categorical (enable_categorical, hist method).
  - HistGB: native categorical support from dtype (sklearn >= 1.4).
  - ExtraTrees / MLP: ordinal codes; the MLP learns embeddings per category.
  - On top of that, nested target encoding + frequency encoding columns give
    every model a numeric view of the categories.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import KNNImputer

import config
from feature_engineering import DynamicFeatures
from utils import log, make_folds, regression_metrics, timer, weighted_mse

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ===================================================================== bundle
@dataclass
class Bundle:
    """Everything fold training needs, with fold-aware dynamic features.

    `w` holds per-row sample weights (test-year density ratios) so every
    fit and every validation score targets the leaderboard's distribution.
    """
    X_tr: pd.DataFrame
    X_te: pd.DataFrame
    y: np.ndarray
    folds_by_repeat: list[list[tuple[np.ndarray, np.ndarray]]]
    dyn_by_repeat: list[list[DynamicFeatures]]
    w: np.ndarray | None = None
    _cache: dict = field(default_factory=dict)

    def fold_data(self, r: int, k: int):
        key = (r, k)
        if key not in self._cache:
            tr_idx, va_idx = self.folds_by_repeat[r][k]
            Xtr = self.X_tr.iloc[tr_idx].reset_index(drop=True).copy()
            Xva = self.X_tr.iloc[va_idx].reset_index(drop=True).copy()
            for d in self.dyn_by_repeat[r]:
                Xtr[d.name] = d.train_cols[k][tr_idx]
                Xva[d.name] = d.train_cols[k][va_idx]
            wtr = self.w[tr_idx] if self.w is not None else None
            wva = self.w[va_idx] if self.w is not None else None
            self._cache[key] = (Xtr, self.y[tr_idx], Xva, self.y[va_idx],
                                wtr, wva)
        return self._cache[key]

    def test_matrix(self) -> pd.DataFrame:
        Xte = self.X_te.copy()
        for d in self.dyn_by_repeat[0]:
            Xte[d.name] = d.test_col
        return Xte

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_tr.columns) + [d.name for d in self.dyn_by_repeat[0]]


# ===================================================================== helpers
def numericize(X: pd.DataFrame, medians: pd.Series | None = None,
               impute: str = "median"):
    """Category -> codes; NaN -> median (or KNN) fit on the given medians/X."""
    Z = X.copy()
    for c in Z.select_dtypes(include="category").columns:
        Z[c] = Z[c].cat.codes.astype(np.float32)
    if impute == "knn" and medians is None:
        imp = KNNImputer(n_neighbors=10)
        return pd.DataFrame(imp.fit_transform(Z), columns=Z.columns), imp
    if medians is None:
        medians = Z.median(numeric_only=True)
    return Z.fillna(medians).to_numpy(np.float32), medians


def _cat_cols_of(X: pd.DataFrame) -> list[str]:
    return list(X.select_dtypes(include="category").columns)


# ===================================================================== adapters
class CatBoostAdapter:
    name = "catboost"
    hpo_folds = config.N_FOLDS

    @staticmethod
    def suggest(trial: optuna.Trial) -> dict:
        p = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 9),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 25),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli"]),
        }
        if p["bootstrap_type"] == "Bayesian":
            p["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 3.0)
        else:
            p["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        return p

    @staticmethod
    def fit_fold(params, Xtr, ytr, Xva, yva, seed, wtr=None, wva=None):
        from catboost import CatBoostRegressor, Pool
        cats = _cat_cols_of(Xtr)
        def prep(X):
            Z = X.copy()
            for c in cats:
                Z[c] = Z[c].astype(str)
            return Z
        train_pool = Pool(prep(Xtr), ytr, cat_features=cats, weight=wtr)
        val_pool = Pool(prep(Xva), yva, cat_features=cats, weight=wva)
        model = CatBoostRegressor(
            iterations=config.GBDT_MAX_ESTIMATORS, loss_function="RMSE",
            eval_metric="RMSE", random_seed=seed, verbose=0,
            allow_writing_files=False, **params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True,
                  early_stopping_rounds=config.EARLY_STOPPING_ROUNDS)
        return model, model.predict(prep(Xva))

    @staticmethod
    def predict(model, X):
        Z = X.copy()
        for c in _cat_cols_of(X):
            Z[c] = Z[c].astype(str)
        return model.predict(Z)


class LightGBMAdapter:
    name = "lightgbm"
    hpo_folds = config.N_FOLDS

    @staticmethod
    def suggest(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 120, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "cat_smooth": trial.suggest_float("cat_smooth", 1.0, 100.0, log=True),
        }

    @staticmethod
    def fit_fold(params, Xtr, ytr, Xva, yva, seed, wtr=None, wva=None):
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=config.GBDT_MAX_ESTIMATORS, random_state=seed,
            verbosity=-1, force_col_wise=True, **params)
        model.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)],
                  eval_sample_weight=[wva] if wva is not None else None,
                  eval_metric="rmse",
                  callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS,
                                                verbose=False)])
        return model, model.predict(Xva)

    @staticmethod
    def predict(model, X):
        return model.predict(X)


class XGBoostAdapter:
    name = "xgboost"
    hpo_folds = config.N_FOLDS

    @staticmethod
    def suggest(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "max_cat_to_onehot": trial.suggest_int("max_cat_to_onehot", 1, 16),
        }

    @staticmethod
    def fit_fold(params, Xtr, ytr, Xva, yva, seed, wtr=None, wva=None):
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=config.GBDT_MAX_ESTIMATORS, tree_method="hist",
            enable_categorical=True, random_state=seed, n_jobs=-1,
            early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
            eval_metric="rmse", **params)
        model.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)],
                  sample_weight_eval_set=[wva] if wva is not None else None,
                  verbose=False)
        return model, model.predict(Xva)

    @staticmethod
    def predict(model, X):
        return model.predict(X)


class ExtraTreesAdapter:
    name = "extratrees"
    hpo_folds = config.N_FOLDS
    final_estimators = 800

    @staticmethod
    def suggest(trial):
        return {
            "n_estimators": 500,
            "max_depth": trial.suggest_int("max_depth", 10, 40),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        }

    @classmethod
    def fit_fold(cls, params, Xtr, ytr, Xva, yva, seed, wtr=None, wva=None,
                 final=False):
        params = dict(params)
        if final:
            params["n_estimators"] = cls.final_estimators
        Ztr, med = numericize(Xtr)
        Zva, _ = numericize(Xva, med)
        model = ExtraTreesRegressor(n_jobs=-1, random_state=seed, **params)
        model.fit(Ztr, ytr, sample_weight=wtr)
        return (model, med), model.predict(Zva)

    @staticmethod
    def predict(model, X):
        est, med = model
        Z, _ = numericize(X, med)
        return est.predict(Z)


class HistGBAdapter:
    name = "histgb"
    hpo_folds = config.N_FOLDS

    @staticmethod
    def suggest(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 200, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        }

    @staticmethod
    def fit_fold(params, Xtr, ytr, Xva, yva, seed, wtr=None, wva=None):
        # internal early stopping on a slice of the *training* part keeps the
        # val fold untouched by stopping decisions (cleanest OOF of the zoo)
        model = HistGradientBoostingRegressor(
            max_iter=2000, early_stopping=True, validation_fraction=0.08,
            n_iter_no_change=40, categorical_features="from_dtype",
            random_state=seed, **params)
        model.fit(Xtr, ytr, sample_weight=wtr)
        return model, model.predict(Xva)

    @staticmethod
    def predict(model, X):
        return model.predict(X)


# --------------------------------------------------------------------- MLP
class TorchMLPRegressor:
    """Tabular MLP with categorical embeddings, trained on MPS if available.

    Stands in for TabM/FT-Transformer: at n=10k with ~300 mostly-informative
    columns, a regularised MLP with embeddings captures what attention-based
    tabular models would, at a fraction of the fit cost; it earns its slot
    through ensemble diversity, not solo accuracy.
    """

    def __init__(self, hidden="384-192", dropout=0.2, lr=1.5e-3,
                 weight_decay=1e-5, batch_size=256, seed=42):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.seed = seed

    def _build(self, n_num, cat_cards):
        import torch.nn as nn
        dims = [int(d) for d in self.hidden.split("-")]
        self._emb_dims = [(c, min(16, c // 2 + 2)) for c in cat_cards]
        emb_total = sum(d for _, d in self._emb_dims)
        layers, prev = [], n_num + emb_total

        class Net(nn.Module):
            def __init__(net):
                super().__init__()
                net.embs = nn.ModuleList(
                    [nn.Embedding(c, d) for c, d in self._emb_dims])
                blocks = []
                p = prev
                for h in dims:
                    blocks += [nn.Linear(p, h), nn.BatchNorm1d(h), nn.SiLU(),
                               nn.Dropout(self.dropout)]
                    p = h
                blocks.append(nn.Linear(p, 1))
                net.mlp = nn.Sequential(*blocks)

            def forward(net, xn, xc):
                embs = [e(xc[:, i]) for i, e in enumerate(net.embs)]
                import torch
                return net.mlp(torch.cat([xn] + embs, dim=1)).squeeze(-1)

        return Net()

    def _prep(self, X: pd.DataFrame, fit: bool):
        cats = _cat_cols_of(X)
        Xc = X[cats].apply(lambda s: s.cat.codes).to_numpy(np.int64) if cats \
            else np.zeros((len(X), 0), np.int64)
        Xc = np.clip(Xc, 0, None)                     # -1 (unseen) -> 0
        Xn = X.drop(columns=cats)
        if fit:
            self._medians = Xn.median(numeric_only=True)
            Z = Xn.fillna(self._medians)
            self._mu = Z.mean().to_numpy(np.float32)
            self._sd = (Z.std().to_numpy(np.float32) + 1e-6)
            self._cat_cards = [max(len(X[c].cat.categories), 1) for c in cats]
        Z = Xn.fillna(self._medians).to_numpy(np.float32)
        Z = (Z - self._mu) / self._sd
        # near-constant train-fold columns (std ~ 0) blow up standardized
        # values for unseen val levels — clip to keep the forward pass sane
        np.clip(Z, -8.0, 8.0, out=Z)
        return Z, Xc

    # joblib support: the inner nn.Module class is built in a closure, so we
    # pickle the state dict + sizes and rebuild the module on load
    def __getstate__(self):
        d = dict(self.__dict__)
        net = d.pop("net_", None)
        if net is not None:
            d["_state_dict"] = {k: v.cpu() for k, v in net.state_dict().items()}
        return d

    def __setstate__(self, d):
        sd = d.pop("_state_dict", None)
        self.__dict__.update(d)
        if sd is not None:
            self.net_ = self._build(self._n_num, self._cat_cards)
            self.net_.load_state_dict(sd)
            self.net_.eval()

    def fit(self, Xtr, ytr, Xva, yva, wtr=None, wva=None):
        import torch
        torch.manual_seed(self.seed)
        device = config.NN_DEVICE if (config.NN_DEVICE != "mps"
                                      or torch.backends.mps.is_available()) else "cpu"
        self.device_used = device
        Zn, Zc = self._prep(Xtr, fit=True)
        Vn, Vc = self._prep(Xva, fit=False)
        self._y_mu, self._y_sd = float(np.mean(ytr)), float(np.std(ytr) + 1e-9)
        yt = (ytr - self._y_mu) / self._y_sd
        wt = np.ones(len(ytr), np.float32) if wtr is None else wtr.astype(np.float32)

        self._n_num = Zn.shape[1]
        self.net_ = self._build(Zn.shape[1], self._cat_cards).to(device)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5,
                                                           patience=8)
        ds = torch.utils.data.TensorDataset(
            torch.tensor(Zn), torch.tensor(Zc),
            torch.tensor(yt, dtype=torch.float32), torch.tensor(wt))
        g = torch.Generator().manual_seed(self.seed)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size,
                                         shuffle=True, generator=g)
        Vn_t, Vc_t = torch.tensor(Vn).to(device), torch.tensor(Vc).to(device)
        best, best_state, patience = np.inf, None, 0
        for epoch in range(config.NN_MAX_EPOCHS):
            self.net_.train()
            for xn, xc, yb, wb in dl:
                xn, xc, yb, wb = (xn.to(device), xc.to(device),
                                  yb.to(device), wb.to(device))
                opt.zero_grad()
                se = (self.net_(xn, xc) - yb) ** 2
                loss = (wb * se).mean()
                loss.backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                pv = self.net_(Vn_t, Vc_t).cpu().numpy() * self._y_sd + self._y_mu
            vmse = float(np.average((pv - yva) ** 2, weights=wva))
            sched.step(vmse)
            if vmse < best - 1e-4:
                best, patience = vmse, 0
                best_state = copy.deepcopy(self.net_.state_dict())
            else:
                patience += 1
                if patience >= config.NN_PATIENCE:
                    break
        self.net_.load_state_dict(best_state)
        self.net_.to("cpu").eval()
        return self

    def predict(self, X):
        import torch
        Zn, Zc = self._prep(X, fit=False)
        with torch.no_grad():
            p = self.net_(torch.tensor(Zn), torch.tensor(Zc)).numpy()
        return p * self._y_sd + self._y_mu


class NNAdapter:
    name = "nn_mlp"
    hpo_folds = 3            # NN trials are slow; prune on 3 folds, final CV uses 5

    @staticmethod
    def suggest(trial):
        return {
            "hidden": trial.suggest_categorical(
                "hidden", ["256-128", "384-192", "512-256-128"]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.4),
            "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        }

    @staticmethod
    def fit_fold(params, Xtr, ytr, Xva, yva, seed, wtr=None, wva=None):
        model = TorchMLPRegressor(seed=seed, **params)
        model.fit(Xtr, ytr, Xva, yva, wtr=wtr, wva=wva)
        return model, model.predict(Xva)

    @staticmethod
    def predict(model, X):
        return model.predict(X)


ADAPTERS = {a.name: a for a in
            [CatBoostAdapter, LightGBMAdapter, XGBoostAdapter,
             ExtraTreesAdapter, HistGBAdapter, NNAdapter]}


def censor_probability(bundle: Bundle, threshold: float = 100.0):
    """P(y == 100) per row: the target is top-censored (7.7% of train sits
    exactly at 100). A dedicated classifier sharpens the top end beyond what
    squared-loss regression does; its OOF probability feeds the stacker as
    an extra column (same fold protocol as the base models, so no leakage).
    """
    import lightgbm as lgb
    n = len(bundle.y)
    ybin = (bundle.y >= threshold).astype(int)
    oof_by_rep, test_acc = [], []
    for r, folds in enumerate(bundle.folds_by_repeat):
        oof_r = np.full(n, np.nan)
        for k, (tr_idx, va_idx) in enumerate(folds):
            Xtr, _, Xva, _, wtr, wva = bundle.fold_data(r, k)
            clf = lgb.LGBMClassifier(
                n_estimators=1500, learning_rate=0.05, num_leaves=63,
                verbosity=-1, random_state=config.REPEAT_SEEDS[r] + 7 * k)
            clf.fit(Xtr, ybin[tr_idx], sample_weight=wtr,
                    eval_set=[(Xva, ybin[va_idx])],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_r[va_idx] = clf.predict_proba(Xva)[:, 1]
            Xte_r = bundle.X_te.copy()
            for d in bundle.dyn_by_repeat[r]:
                Xte_r[d.name] = d.test_col
            test_acc.append(clf.predict_proba(Xte_r)[:, 1])
            joblib.dump(clf, config.MODELS_DIR / f"p100_r{r}_f{k}.joblib")
        oof_by_rep.append(oof_r)
    oof = np.nanmean(np.vstack(oof_by_rep), axis=0)
    from sklearn.metrics import roc_auc_score
    log.info("censor classifier P(y=100): OOF AUC %.4f, base rate %.3f",
             roc_auc_score(ybin, oof), ybin.mean())
    return oof, np.mean(np.vstack(test_acc), axis=0)


DEFAULT_PARAMS = {  # sane mid-strength fallbacks for --skip-hpo runs
    "catboost": {"learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 6.0,
                 "random_strength": 1.0, "one_hot_max_size": 12,
                 "bootstrap_type": "Bayesian", "bagging_temperature": 1.0},
    "lightgbm": {"learning_rate": 0.05, "num_leaves": 64,
                 "min_child_samples": 20, "feature_fraction": 0.8,
                 "bagging_fraction": 0.8, "bagging_freq": 1,
                 "lambda_l1": 0.1, "lambda_l2": 1.0, "cat_smooth": 10.0},
    "xgboost": {"learning_rate": 0.05, "max_depth": 6, "min_child_weight": 3.0,
                "subsample": 0.85, "colsample_bytree": 0.8,
                "colsample_bylevel": 0.9, "reg_alpha": 0.01, "reg_lambda": 2.0,
                "gamma": 1e-4, "max_cat_to_onehot": 8},
    "extratrees": {"n_estimators": 500, "max_depth": 25, "max_features": 0.6,
                   "min_samples_leaf": 2, "min_samples_split": 4},
    "histgb": {"learning_rate": 0.06, "max_leaf_nodes": 64,
               "min_samples_leaf": 25, "l2_regularization": 0.1,
               "max_features": 0.9},
    "nn_mlp": {"hidden": "384-192", "dropout": 0.2, "lr": 1.5e-3,
               "weight_decay": 1e-5},
}


# ===================================================================== HPO
def run_hpo(model_name: str, bundle: Bundle, budget: dict,
            budget_name: str = "full", seed: int = config.SEED) -> dict:
    """Optuna TPE + median pruning, resumable via sqlite storage."""
    adapter = ADAPTERS[model_name]
    n_hpo_folds = min(adapter.hpo_folds, len(bundle.folds_by_repeat[0]))

    def objective(trial: optuna.Trial) -> float:
        params = adapter.suggest(trial)
        mses = []
        for k in range(n_hpo_folds):
            Xtr, ytr, Xva, yva, wtr, wva = bundle.fold_data(0, k)
            _, pred = adapter.fit_fold(params, Xtr, ytr, Xva, yva, seed,
                                       wtr=wtr, wva=wva)
            mses.append(weighted_mse(yva, pred, wva))
            trial.report(float(np.mean(mses)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(mses))

    storage = f"sqlite:///{config.HPO_DIR / 'optuna.db'}"
    study = optuna.create_study(
        study_name=f"{model_name}_{budget_name}", storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=12, n_warmup_steps=1))
    done = len([t for t in study.trials
                if t.state in (optuna.trial.TrialState.COMPLETE,
                               optuna.trial.TrialState.PRUNED)])
    remaining = max(0, budget["n_trials"] - done)
    if remaining:
        with timer(f"HPO {model_name} ({remaining} trials, ≤{budget['timeout']}s)"):
            study.optimize(objective, n_trials=remaining,
                           timeout=budget["timeout"], gc_after_trial=True)
    best = study.best_params
    out = {"model": model_name, "best_value_cv_mse": study.best_value,
           "n_trials_completed": len(study.trials), "best_params": best}
    (config.HPO_DIR / f"best_{model_name}.json").write_text(json.dumps(out, indent=2))
    log.info("HPO %s: best CV MSE %.4f after %d trials",
             model_name, study.best_value, len(study.trials))
    return out


def materialize_params(model_name: str, raw: dict) -> dict:
    """Optuna best_params -> adapter fit params (handles conditionals)."""
    p = dict(raw)
    if model_name == "lightgbm":
        p["bagging_freq"] = 1
    if model_name == "extratrees":
        p["n_estimators"] = 500
    return p


# ===================================================================== final CV
def final_cv(model_name: str, params: dict, bundle: Bundle,
             save_models: bool = True) -> dict:
    """Repeated stratified 5-fold: OOF preds, fold-averaged test preds,
    per-fold metrics. Fold models are persisted for inference."""
    adapter = ADAPTERS[model_name]
    n = len(bundle.y)
    Xte = bundle.test_matrix()
    oof_by_rep, test_preds, fold_rows = [], [], []

    for r, folds in enumerate(bundle.folds_by_repeat):
        oof = np.full(n, np.nan)
        for k in range(len(folds)):
            Xtr, ytr, Xva, yva, wtr, wva = bundle.fold_data(r, k)
            seed = config.REPEAT_SEEDS[r] + 7 * k
            kwargs = {"final": True} if model_name == "extratrees" else {}
            model, pred = adapter.fit_fold(params, Xtr, ytr, Xva, yva, seed,
                                           wtr=wtr, wva=wva, **kwargs)
            tr_idx, va_idx = folds[k]
            oof[va_idx] = pred
            Xte_r = bundle.X_te.copy()
            for d in bundle.dyn_by_repeat[r]:
                Xte_r[d.name] = d.test_col
            test_preds.append(adapter.predict(model, Xte_r))
            m = regression_metrics(yva, pred, w=wva)
            fold_rows.append({"repeat": r, "fold": k, **m})
            log.info("  %s r%d f%d  MSE %.3f wMSE %.3f MAE %.3f",
                     model_name, r, k, m["mse"], m.get("wmse", m["mse"]),
                     m["mae"])
            if save_models:
                joblib.dump(model, config.MODELS_DIR / f"{model_name}_r{r}_f{k}.joblib")
        oof_by_rep.append(oof)

    oof_mean = np.nanmean(np.vstack(oof_by_rep), axis=0)
    overall = regression_metrics(bundle.y, oof_mean, w=bundle.w)
    return {
        "model": model_name,
        "oof": oof_mean,
        "oof_by_repeat": np.vstack(oof_by_rep),
        "test_pred": np.mean(np.vstack(test_preds), axis=0),
        "fold_metrics": fold_rows,
        "oof_metrics": overall,
        "params": params,
    }
