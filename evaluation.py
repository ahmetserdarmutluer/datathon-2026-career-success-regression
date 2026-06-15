"""Cross-validation metrics, ablations, and model interpretation (SHAP)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from model_training import ADAPTERS, Bundle
from utils import log, make_folds, regression_metrics, timer, weighted_mse

_FIG = config.ROOT / "figures"


def _savefig(fig, name: str) -> None:
    path = _FIG / name
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("figure -> figures/%s", name)

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


def plot_model_comparison(results: dict, model_names: list[str]) -> None:
    """Bar chart of OOF RMSE per model → figures/07_model_comparison.png."""
    rows = []
    for m in model_names:
        fm = pd.DataFrame(results[m]["fold_metrics"])
        rows.append((m, results[m]["oof_metrics"]["rmse"], fm["rmse"].std()))
    rows.sort(key=lambda x: x[1])
    models, means, stds = zip(*rows)

    fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.6)))
    colors = ["#4CAF50" if i == 0 else "#2196F3" for i in range(len(models))]
    ax.barh(models, means, xerr=stds, capsize=3,
            color=colors, edgecolor="white", alpha=0.88)
    for i, v in enumerate(means):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("OOF RMSE (year-weighted, 5-fold CV)")
    ax.set_title("Model comparison — OOF RMSE  (green = best)", fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    _savefig(fig, "07_model_comparison.png")


def plot_cv_heatmap(results: dict, model_names: list[str]) -> None:
    """Fold-level RMSE heatmap → figures/08_cv_heatmap.png."""
    all_folds = sorted({(r["repeat"], r["fold"])
                        for m in model_names
                        for r in results[m]["fold_metrics"]})
    col_labels = [f"r{r}f{f}" for r, f in all_folds]
    matrix = np.zeros((len(model_names), len(all_folds)))
    for i, m in enumerate(model_names):
        fm = {(r["repeat"], r["fold"]): r["rmse"]
              for r in results[m]["fold_metrics"]}
        for j, key in enumerate(all_folds):
            matrix[i, j] = fm.get(key, np.nan)

    fig, ax = plt.subplots(figsize=(max(6, len(all_folds) * 0.9),
                                    max(3, len(model_names) * 0.55)))
    vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(model_names))); ax.set_yticklabels(model_names, fontsize=9)
    for i in range(len(model_names)):
        for j in range(len(all_folds)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    fig.colorbar(im, ax=ax, shrink=0.6, label="RMSE")
    ax.set_title("CV fold-level RMSE heatmap", fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "08_cv_heatmap.png")


def plot_oof(y: np.ndarray, results: dict, model_names: list[str]) -> None:
    """Actual vs predicted scatter + residuals for the best model → figures/09_best_model_oof.png."""
    best = min(model_names, key=lambda m: results[m]["oof_metrics"]["rmse"])
    oof = results[best]["oof"]
    residuals = y - oof
    rmse = results[best]["oof_metrics"]["rmse"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y, oof, alpha=0.15, s=5, color="#2196F3")
    axes[0].plot([0, 100], [0, 100], "r--", lw=1.5)
    axes[0].set(xlabel="Actual", ylabel="Predicted",
                xlim=(0, 100), ylim=(0, 100),
                title=f"Actual vs Predicted OOF — {best}  (RMSE {rmse:.3f})")
    axes[1].scatter(oof, residuals, alpha=0.15, s=5, color="#FF9800")
    axes[1].axhline(0, color="red", lw=1.5)
    axes[1].set(xlabel="Predicted", ylabel="Residual",
                title="OOF Residuals")
    fig.suptitle(f"Best single model: {best}", fontweight="bold")
    fig.tight_layout()
    _savefig(fig, "09_best_model_oof.png")
