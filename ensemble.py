"""Ensembling: optimized blending and stacking, compared honestly.

The trap with ensembling on OOF predictions is evaluating weights on the
same OOF rows they were fit on — that always says "the most flexible
stacker wins". Here every candidate combiner (single models, equal blend,
SLSQP-optimized blend, Ridge stack, CatBoost stack) is scored with repeated
K-fold *over the OOF rows*: fit the combiner on 4/5 of OOF, score on the
held-out 1/5. The winner of that honest comparison is refit on the full OOF
matrix and applied to the test predictions.

Distribution-shift support: all fitting and all scoring accept per-row
weights (test-year density ratios), so the referee ranks combiners by the
leaderboard's metric, not uniform-year MSE. Stackers additionally receive
`extra` columns (application_year) and can learn year-conditional
corrections — e.g. shrink towards the year mean in the noisy recent years.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge, RidgeCV

import config
from utils import log, make_folds, weighted_mse


def _stack_input(P, extra):
    return P if extra is None else np.column_stack([P, extra])


# --------------------------------------------------------------- combiners
class BlendEnsemble:
    """Convex combination, weights >= 0, sum to 1 (SLSQP, weighted MSE)."""

    uses_extra = False

    def __init__(self, model_names):
        self.model_names = list(model_names)

    def fit(self, P: np.ndarray, y: np.ndarray, w=None, extra=None):
        m = P.shape[1]
        x0 = np.full(m, 1.0 / m)

        def mse(wt):
            return weighted_mse(y, P @ wt, w)

        res = minimize(mse, x0, method="SLSQP", bounds=[(0.0, 1.0)] * m,
                       constraints=[{"type": "eq",
                                     "fun": lambda v: np.sum(v) - 1.0}],
                       options={"maxiter": 500, "ftol": 1e-10})
        self.weights_ = res.x / res.x.sum()
        return self

    def predict(self, P: np.ndarray, extra=None) -> np.ndarray:
        return P @ self.weights_


class EqualBlend:
    uses_extra = False

    def __init__(self, model_names):
        self.model_names = list(model_names)

    def fit(self, P, y, w=None, extra=None):
        self.weights_ = np.full(P.shape[1], 1.0 / P.shape[1])
        return self

    def predict(self, P, extra=None):
        return P @ self.weights_


class RidgeStack:
    """Linear meta-model with L2; alpha chosen by internal CV (weighted).
    Allows negative weights (useful when base errors are correlated) and
    sees the year column to absorb year-conditional bias."""

    uses_extra = True

    def __init__(self, model_names):
        self.model_names = list(model_names)

    def fit(self, P, y, w=None, extra=None):
        Z = _stack_input(P, extra)
        self._mu = Z.mean(axis=0)
        self._sd = Z.std(axis=0) + 1e-9
        Zs = (Z - self._mu) / self._sd
        self.model_ = RidgeCV(alphas=np.logspace(-3, 4, 29))
        self.model_.fit(Zs, y, sample_weight=w)
        return self

    def predict(self, P, extra=None):
        Zs = (_stack_input(P, extra) - self._mu) / self._sd
        return self.model_.predict(Zs)


class CatBoostStack:
    """Shallow non-linear meta-model: can learn 'trust model A in this
    prediction regime / this year'. Depth 2 + strong L2 to resist OOF
    overfitting."""

    uses_extra = True

    def __init__(self, model_names):
        self.model_names = list(model_names)

    def fit(self, P, y, w=None, extra=None):
        from catboost import CatBoostRegressor
        self.model_ = CatBoostRegressor(
            iterations=500, learning_rate=0.04, depth=2, l2_leaf_reg=10.0,
            loss_function="RMSE", random_seed=config.SEED, verbose=0,
            allow_writing_files=False)
        self.model_.fit(_stack_input(P, extra), y, sample_weight=w)
        return self

    def predict(self, P, extra=None):
        return self.model_.predict(_stack_input(P, extra))


class SingleModel:
    uses_extra = False

    def __init__(self, idx, name):
        self.idx = idx
        self.model_names = [name]

    def fit(self, P, y, w=None, extra=None):
        return self

    def predict(self, P, extra=None):
        return P[:, self.idx]


# --------------------------------------------------------------- comparison
def honest_comparison(P_oof: np.ndarray, y: np.ndarray, model_names: list[str],
                      w: np.ndarray | None = None,
                      extra: np.ndarray | None = None,
                      n_repeats: int = 3) -> dict[str, float]:
    """Repeated 5-fold CV over OOF rows; mean held-out *weighted* MSE."""
    candidates = {f"single_{n}": (lambda i=i, n=n: SingleModel(i, n))
                  for i, n in enumerate(model_names)}
    candidates["blend_equal"] = lambda: EqualBlend(model_names)
    candidates["blend_optimized"] = lambda: BlendEnsemble(model_names)
    candidates["stack_ridge"] = lambda: RidgeStack(model_names)
    candidates["stack_catboost"] = lambda: CatBoostStack(model_names)

    scores = {k: [] for k in candidates}
    for rep in range(n_repeats):
        folds = make_folds(y, seed=config.SEED + 101 * rep)
        for tr, va in folds:
            wtr = w[tr] if w is not None else None
            wva = w[va] if w is not None else None
            xtr = extra[tr] if extra is not None else None
            xva = extra[va] if extra is not None else None
            for cname, factory in candidates.items():
                mdl = factory().fit(P_oof[tr], y[tr], w=wtr, extra=xtr)
                pred = mdl.predict(P_oof[va], extra=xva)
                scores[cname].append(weighted_mse(y[va], pred, wva))
    means = {k: float(np.mean(v)) for k, v in scores.items()}
    for k, v in sorted(means.items(), key=lambda kv: kv[1]):
        log.info("ensemble %-22s honest wMSE %.4f", k, v)
    return means


def fit_final(method: str, P_oof: np.ndarray, y: np.ndarray,
              model_names: list[str], w: np.ndarray | None = None,
              extra: np.ndarray | None = None):
    if method.startswith("single_"):
        return SingleModel(model_names.index(method[7:]), method[7:]).fit(
            P_oof, y)
    cls = {"blend_equal": EqualBlend, "blend_optimized": BlendEnsemble,
           "stack_ridge": RidgeStack, "stack_catboost": CatBoostStack}[method]
    ens = cls(model_names).fit(P_oof, y, w=w, extra=extra)
    if hasattr(ens, "weights_"):
        log.info("final weights: %s",
                 dict(zip(model_names, np.round(ens.weights_, 4))))
    return ens
