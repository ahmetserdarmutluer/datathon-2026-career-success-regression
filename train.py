"""End-to-end training pipeline.

    python train.py --budget full            # the real run
    python train.py --budget smoke           # 5-minute plumbing check
    python train.py --skip-hpo               # reuse saved best params

Stages: data -> folds -> static features -> text features -> nested dynamic
features -> ablations -> HPO -> repeated-CV training -> honest ensemble
selection -> SHAP -> artifacts + submission.csv. Every stage logs to MLflow
and artifacts/experiments.json.
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

import config
import ensemble as ens
import evaluation
import model_training as mt
import text_features as tf
from feature_engineering import (TabularFeatureBuilder, add_auto_interactions,
                                 build_target_encodings)
from utils import (Tracker, clip_preds, load_data, log, make_folds,
                   regression_metrics, seed_everything, timer, year_weights)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", choices=list(config.BUDGETS), default="full")
    p.add_argument("--models", default=",".join(config.MODEL_NAMES),
                   help="comma-separated subset of models")
    p.add_argument("--emb", choices=["auto", "e5large", "mpnet", "none"],
                   default="auto")
    p.add_argument("--skip-hpo", action="store_true",
                   help="use saved best params (or defaults) instead of Optuna")
    p.add_argument("--no-ablation", action="store_true")
    return p.parse_args()


def build_bundle(train_df, test_df, y, folds_by_repeat, emb_key, tracker,
                 w=None):
    texts_tr = train_df[config.TEXT_COL]
    texts_te = test_df[config.TEXT_COL]

    with timer("static tabular features"):
        builder = TabularFeatureBuilder().fit(train_df)
        X_tr = builder.transform(train_df).reset_index(drop=True)
        X_te = builder.transform(test_df).reset_index(drop=True)

    factory = tf.TextFeatureFactory(emb_key=emb_key or config.PRIMARY_EMBEDDING,
                                    use_embeddings=emb_key is not None)
    factory.fit(texts_tr, texts_te)
    X_tr = pd.concat([X_tr, factory.static_features(texts_tr, "train")], axis=1)
    X_te = pd.concat([X_te, factory.static_features(texts_te, "test")], axis=1)

    auto_pairs = []
    if config.USE_AUTO_INTERACTIONS:
        X_tr, X_te, top = add_auto_interactions(X_tr, X_te, y)
        auto_pairs = [(a, b) for i, a in enumerate(top) for b in top[i + 1:]]
        log.info("auto interactions over top features: %s", top)

    dyn_by_repeat = []
    for r, folds in enumerate(folds_by_repeat):
        with timer(f"nested dynamic features (repeat {r})"):
            dyn = build_target_encodings(train_df, test_df, y, folds)
            dyn += factory.meta_features(y, folds)
            dyn_by_repeat.append(dyn)

    bundle = mt.Bundle(X_tr=X_tr, X_te=X_te, y=y,
                       folds_by_repeat=folds_by_repeat,
                       dyn_by_repeat=dyn_by_repeat, w=w)
    tracker.log_params({"n_features": len(bundle.feature_names),
                        "emb_key": emb_key, "auto_pairs": len(auto_pairs)})
    log.info("feature matrix: %d columns", len(bundle.feature_names))
    return bundle, builder, factory, auto_pairs


def main():
    args = parse_args()
    seed_everything()
    if args.budget == "smoke":           # keep the plumbing check quick
        config.FINAL_REPEATS = 1
        config.NN_MAX_EPOCHS = 25
        config.GBDT_MAX_ESTIMATORS = 600
    model_names = [m for m in config.MODEL_NAMES if m in args.models.split(",")]
    tracker = Tracker(f"train_{args.budget}")
    tracker.log_params({"budget": args.budget, "models": model_names,
                        "n_folds": config.N_FOLDS,
                        "repeats": config.FINAL_REPEATS, "seed": config.SEED})

    train_df, test_df = load_data()
    y = train_df[config.TARGET].to_numpy()
    years_tr = train_df[config.YEAR_COL].to_numpy()
    years_te = test_df[config.YEAR_COL].to_numpy()
    w = year_weights(years_tr, years_te) if config.USE_YEAR_WEIGHTS else None
    if w is not None:
        log.info("year weights: min %.2f max %.2f (test-year density ratio)",
                 w.min(), w.max())
    strat_years = years_tr if config.STRATIFY_BY_YEAR else None
    folds_by_repeat = [make_folds(y, seed=s, years=strat_years)
                       for s in config.REPEAT_SEEDS[:config.FINAL_REPEATS]]

    # ---------------- text representation bake-off ----------------
    emb_key = None if args.emb == "none" else (
        args.emb if args.emb != "auto" else None)
    if args.emb == "auto":
        if args.budget == "smoke":
            emb_key = None
        else:
            cmp = tf.compare_embedding_models(train_df[config.TEXT_COL], y,
                                              folds_by_repeat[0])
            tracker.log_metrics(cmp, prefix="textcmp_")
            log.info("text representation bake-off (Ridge OOF MSE): %s",
                     {k: round(v, 3) for k, v in cmp.items()})
            emb_only = {k: v for k, v in cmp.items()
                        if k in config.EMBEDDING_MODELS}
            emb_key = min(emb_only, key=emb_only.get) if emb_only else None

    w_train = w if (w is not None and config.USE_YEAR_WEIGHTS_TRAIN) else None
    bundle, builder, factory, auto_pairs = build_bundle(
        train_df, test_df, y, folds_by_repeat, emb_key, tracker, w=w_train)

    # ---------------- ablations ----------------
    if not args.no_ablation and args.budget != "smoke":
        with timer("feature ablations"):
            tracker.log_metrics(evaluation.run_ablations(bundle), prefix="ablation_")
        with timer("imputation comparison"):
            tracker.log_metrics(evaluation.compare_imputation(bundle),
                                prefix="impute_")

    # ---------------- HPO ----------------
    best_params = {}
    for m in model_names:
        saved = config.HPO_DIR / f"best_{m}.json"
        if args.skip_hpo:
            best_params[m] = (json.loads(saved.read_text())["best_params"]
                              if saved.exists() else mt.DEFAULT_PARAMS[m])
            log.info("skip-hpo: %s params %s", m, best_params[m])
        else:
            res = mt.run_hpo(m, bundle, config.BUDGETS[args.budget][m],
                             budget_name=args.budget)
            best_params[m] = res["best_params"]
            tracker.log_metrics({f"hpo_{m}_best_mse": res["best_value_cv_mse"],
                                 f"hpo_{m}_trials": res["n_trials_completed"]})
        best_params[m] = mt.materialize_params(m, best_params[m])

    # ---------------- final repeated-CV training ----------------
    results = {}
    for m in model_names:
        with timer(f"final CV {m}"):
            results[m] = mt.final_cv(m, best_params[m], bundle)
        om = results[m]["oof_metrics"]
        tracker.log_metrics({f"{m}_oof_{k}": v for k, v in om.items()})
        tracker.log_table(f"fold_metrics_{m}", results[m]["fold_metrics"])
        tracker.log_params({f"params_{m}": best_params[m]})
        log.info("%s OOF  MSE %.4f  RMSE %.4f  MAE %.4f",
                 m, om["mse"], om["rmse"], om["mae"])

    P_oof = np.column_stack([results[m]["oof"] for m in model_names])
    P_test = np.column_stack([results[m]["test_pred"] for m in model_names])
    np.savez(config.ARTIFACTS / "predictions.npz",
             y=y, models=model_names, P_oof=P_oof, P_test=P_test,
             fold_assign_r0=np.concatenate(
                 [np.full(len(va), k) for k, (_, va) in
                  enumerate(folds_by_repeat[0])])[
                 np.argsort(np.concatenate(
                     [va for _, va in folds_by_repeat[0]]))])

    # ---------------- ensemble selection ----------------
    with timer("censoring classifier P(y=100)"):
        p100_oof, p100_test = mt.censor_probability(bundle)
    extra_tr = np.column_stack([years_tr.astype(float), p100_oof])
    extra_te = np.column_stack([years_te.astype(float), p100_test])
    with timer("honest ensemble comparison"):
        comparison = ens.honest_comparison(P_oof, y, model_names, w=w,
                                           extra=extra_tr)
    tracker.log_metrics(comparison, prefix="ens_")
    best_method = min(comparison, key=comparison.get)
    log.info("selected ensemble: %s (honest wMSE %.4f)",
             best_method, comparison[best_method])
    final_ens = ens.fit_final(best_method, P_oof, y, model_names, w=w,
                              extra=extra_tr)
    final_oof = clip_preds(final_ens.predict(P_oof, extra=extra_tr))
    tracker.log_metrics(regression_metrics(y, final_oof, w=w),
                        prefix="final_oof_")
    test_pred = clip_preds(final_ens.predict(P_test, extra=extra_te))

    # ---------------- interpretation ----------------
    if args.budget != "smoke":
        with timer("SHAP analysis"):
            evaluation.shap_analysis(bundle, best_params.get(
                "lightgbm", evaluation.FIXED_LGBM))
        evaluation.native_importances(results, bundle)
        for png in config.PLOTS_DIR.glob("shap_*.png"):
            tracker.log_artifact(png)

    # ---------------- persist inference artifacts ----------------
    joblib.dump(builder, config.ARTIFACTS / "feature_builder.joblib")
    factory.fit_full_train_meta_models(y)
    factory.save(config.ARTIFACTS / "text_factory.joblib")
    te_maps = {}
    for c in config.TE_COLS:
        prior = float(y.mean())
        s = pd.Series(y).groupby(train_df[c].astype(str)).agg(["sum", "count"])
        te_maps[c] = {"prior": prior,
                      "map": ((s["sum"] + prior * config.TE_SMOOTHING)
                              / (s["count"] + config.TE_SMOOTHING)).to_dict()}
    joblib.dump(te_maps, config.ARTIFACTS / "te_maps.joblib")
    joblib.dump(final_ens, config.ARTIFACTS / "ensemble.joblib")
    meta = {"models": model_names, "repeats": config.FINAL_REPEATS,
            "n_folds": config.N_FOLDS, "feature_names": bundle.feature_names,
            "auto_pairs": auto_pairs, "ensemble_method": best_method,
            "emb_key": emb_key, "budget": args.budget,
            "ensemble_honest_mse": comparison[best_method],
            "ensemble_uses_year_extra": bool(getattr(final_ens, "uses_extra",
                                                     False)),
            "extra_cols": ["application_year", "p100"],
            "year_weighted": w is not None}
    (config.ARTIFACTS / "pipeline_meta.json").write_text(json.dumps(meta, indent=2))

    # ---------------- submission ----------------
    sub = pd.DataFrame({config.ID_COL: test_df[config.ID_COL],
                        config.TARGET: test_pred})
    sub.to_csv(config.SUBMISSION_PATH, index=False)
    tracker.log_artifact(config.SUBMISSION_PATH)
    log.info("submission written -> %s  (mean %.3f, std %.3f)",
             config.SUBMISSION_PATH, test_pred.mean(), test_pred.std())
    tracker.finish()


if __name__ == "__main__":
    main()
