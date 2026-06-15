"""Cross-validation metrics, ablations, and model interpretation (SHAP)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from model_training import ADAPTERS, Bundle
from utils import log, make_folds, regression_metrics, timer, weighted_mse

FIXED_LGBM = {  # mid-of-road params for fair feature-set comparisons
    "learning_rate": 0.05, "num_leaves": 64, "min_child_samples": 20,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l1": 0.1, "lambda_l2": 1.0, "cat_smooth": 10.0,
}


def summarize_folds(fold_rows: list[dict]) -> dict:
    df = pd.DataFrame(fold_rows)
    return {f"{m}_{s}": float(getattr(df[m], s)())
            for m in ("mse", "rmse", "mae") for s in ("mean", "std")}


def quick_cv(bundle: Bundle, drop_cols: list[str] | None = None,
             drop_dyn: list[str] | None = None) -> float:
    """5-fold OOF MSE of a fixed LightGBM on a feature-set variant.

    One fixed, mid-strength model keeps the comparison about the *features*;
    re-tuning per variant would confound feature value with HPO luck.
    """
    adapter = ADAPTERS["lightgbm"]
    y = bundle.y
    oof = np.full(len(y), np.nan)
    drop_dyn = drop_dyn or []
    for k in range(len(bundle.folds_by_repeat[0])):
        Xtr, ytr, Xva, yva, wtr, wva = bundle.fold_data(0, k)
        keep = [c for c in Xtr.columns
                if c not in (drop_cols or []) and c not in drop_dyn]
        _, pred = adapter.fit_fold(FIXED_LGBM, Xtr[keep], ytr, Xva[keep], yva,
                                   config.SEED, wtr=wtr, wva=wva)
        _, va_idx = bundle.folds_by_repeat[0][k]
        oof[va_idx] = pred
    return weighted_mse(y, oof, bundle.w)


def run_ablations(bundle: Bundle) -> dict[str, float]:
    """Measure what each engineered block buys (lower MSE = block helps)."""
    text_static = [c for c in bundle.X_tr.columns
                   if c.startswith(("tfidf_svd_", "emb_pca_", "text_"))]
    auto_inter = [c for c in bundle.X_tr.columns if c.startswith("ax_")]
    te_dyn = [d.name for d in bundle.dyn_by_repeat[0] if d.name.startswith("te_")]
    txt_dyn = [d.name for d in bundle.dyn_by_repeat[0] if d.name.startswith("txt_")]
    freq_cols = [c for c in bundle.X_tr.columns if c.endswith("_freq")]
    miss_ind = [c for c in bundle.X_tr.columns
                if c.endswith("_missing") or c == "n_missing"
                or c == "intern_dur_missing_but_has_intern"]

    variants = {
        "full": ([], []),
        "no_text_at_all": (text_static, txt_dyn),
        "no_text_meta_oof": ([], txt_dyn),
        "no_text_static": (text_static, []),
        "no_target_encoding": ([], te_dyn),
        "no_frequency_encoding": (freq_cols, []),
        "no_auto_interactions": (auto_inter, []),
        "no_missing_indicators": (miss_ind, []),
    }
    results = {}
    for name, (cols, dyn) in variants.items():
        with timer(f"ablation {name}"):
            results[name] = quick_cv(bundle, drop_cols=cols, drop_dyn=dyn)
        log.info("ablation %-24s MSE %.4f", name, results[name])
    return results


def compare_imputation(bundle: Bundle) -> dict[str, float]:
    """Median vs KNN imputation for the only NaN-intolerant tree model
    (ExtraTrees), on fold 0 (KNN imputation is O(n^2); one fold suffices to
    rank the strategies)."""
    from sklearn.ensemble import ExtraTreesRegressor
    from model_training import numericize
    Xtr, ytr, Xva, yva, wtr, wva = bundle.fold_data(0, 0)
    out = {}
    for strat in ("median", "knn"):
        with timer(f"imputation {strat}"):
            if strat == "median":
                Ztr, med = numericize(Xtr)
                Zva, _ = numericize(Xva, med)
            else:
                Ztr_df, imp = numericize(Xtr, impute="knn")
                Ztr = Ztr_df.to_numpy(np.float32)
                Zva_df = Xva.copy()
                for c in Zva_df.select_dtypes(include="category").columns:
                    Zva_df[c] = Zva_df[c].cat.codes.astype(np.float32)
                Zva = imp.transform(Zva_df).astype(np.float32)
            et = ExtraTreesRegressor(n_estimators=400, max_features=0.6,
                                     min_samples_leaf=3, n_jobs=-1,
                                     random_state=config.SEED)
            et.fit(Ztr, ytr, sample_weight=wtr)
            out[strat] = weighted_mse(yva, et.predict(Zva), wva)
        log.info("imputation %-6s fold0 MSE %.4f", strat, out[strat])
    return out


# ===================================================================== interpretation
def shap_analysis(bundle: Bundle, lgbm_params: dict, max_display: int = 25):
    """Global SHAP summary + bar + 3 local waterfalls, from a LightGBM fit
    on fold 0 (TreeExplainer is exact and fast for LightGBM; conclusions
    transfer across the GBDT family which shares the feature matrix)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    adapter = ADAPTERS["lightgbm"]
    Xtr, ytr, Xva, yva, wtr, wva = bundle.fold_data(0, 0)
    model, _ = adapter.fit_fold(lgbm_params, Xtr, ytr, Xva, yva, config.SEED,
                                wtr=wtr, wva=wva)

    explainer = shap.TreeExplainer(model)
    sv = explainer(Xva)

    plt.figure()
    shap.summary_plot(sv, Xva, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "shap_summary.png", dpi=150)
    plt.close("all")

    plt.figure()
    shap.summary_plot(sv, Xva, plot_type="bar", max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "shap_importance_bar.png", dpi=150)
    plt.close("all")

    # local explanations: low / median / high predicted students
    pred = model.predict(Xva)
    for tag, idx in {"low": int(np.argmin(pred)),
                     "median": int(np.argsort(pred)[len(pred) // 2]),
                     "high": int(np.argmax(pred))}.items():
        plt.figure()
        shap.plots.waterfall(sv[idx], max_display=18, show=False)
        plt.tight_layout()
        plt.savefig(config.PLOTS_DIR / f"shap_waterfall_{tag}.png", dpi=150)
        plt.close("all")

    imp = pd.DataFrame({"feature": Xva.columns,
                        "mean_abs_shap": np.abs(sv.values).mean(0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(config.ARTIFACTS / "shap_importance.csv", index=False)
    log.info("SHAP artifacts -> %s", config.PLOTS_DIR)
    return imp


def native_importances(results: dict[str, dict], bundle: Bundle):
    """Per-model native feature importances averaged over saved fold models."""
    import joblib
    rows = []
    feature_names = bundle.feature_names
    for name in results:
        try:
            model = joblib.load(config.MODELS_DIR / f"{name}_r0_f0.joblib")
        except FileNotFoundError:
            continue
        est = model[0] if isinstance(model, tuple) else model
        imp = None
        if hasattr(est, "feature_importances_"):
            imp = np.asarray(est.feature_importances_, dtype=float)
        elif hasattr(est, "get_feature_importance"):
            imp = np.asarray(est.get_feature_importance(), dtype=float)
        if imp is not None and len(imp) == len(feature_names):
            imp = imp / (imp.sum() + 1e-12)
            for f, v in zip(feature_names, imp):
                rows.append({"model": name, "feature": f, "importance": float(v)})
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(config.ARTIFACTS / "native_importances.csv", index=False)
    return df
