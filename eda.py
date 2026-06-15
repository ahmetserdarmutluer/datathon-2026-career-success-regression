"""Exploratory Data Analysis for the career-success-score dataset.

A condensed, reproducible EDA focused on the findings that actually shaped the
modelling strategy (rather than an exhaustive variable-by-variable report):

  1. Target distribution + upper censoring at 100  -> motivates a two-part model
  2. Missingness map + informative missingness       -> motivates missing-indicators
  3. Correlation of predictors with the target       -> signal ranking
  4. Permutation feature importance                  -> non-linear signal ranking
  5. Mentor-feedback text signal                     -> is the free text useful?
  6. Distribution shift (train vs test, by year)     -> the key generalisation issue

Run:  python eda.py        # writes PNGs to figures/ and prints a report
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

FIG = config.ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10})


def _save(fig, name):
    fig.savefig(FIG / name, facecolor="white")
    plt.close(fig)
    print(f"  figure -> figures/{name}")


def target_analysis(tr):
    """Distribution of the outcome and its upper-censoring at 100."""
    y = tr[config.TARGET]
    print("\n[1] TARGET (career_success_score)")
    print(f"    mean={y.mean():.2f}  sd={y.std():.2f}  min={y.min():.2f}  max={y.max():.2f}")
    at_max = (y >= 100).mean()
    print(f"    fraction at exactly 100 (ceiling/censoring): {at_max:.3f}")
    print(f"    baseline MSE (predict the mean): {y.var():.1f}")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(y, bins=60, color="#3b7dd8")
    ax[0].set(title="Target distribution", xlabel="career_success_score", ylabel="count")
    ax[0].axvline(100, color="crimson", ls="--", lw=1, label="censoring at 100")
    ax[0].legend()
    by_year = tr.groupby("application_year")[config.TARGET].agg(["mean", "std"])
    ax[1].errorbar(by_year.index, by_year["mean"], yerr=by_year["std"], marker="o", capsize=3)
    ax[1].set(title="Target mean ± sd by application year",
              xlabel="application_year", ylabel="career_success_score")
    _save(fig, "01_target.png")
    print("    -> recent years have a lower, NOISIER target (drift); 7.7% sit at 100.")


def missingness(tr):
    print("\n[2] MISSINGNESS")
    miss = tr.isna().mean()
    miss = miss[miss > 0].sort_values(ascending=False)
    print(miss.round(3).to_string())
    y = tr[config.TARGET].values
    print("    informative-missingness check (target mean when missing vs present):")
    for c in miss.index:
        m = tr[c].isna().values
        print(f"      {c:32s} missing={y[m].mean():.2f}  present={y[~m].mean():.2f}")
    fig, ax = plt.subplots(figsize=(7, 4))
    miss.plot.barh(ax=ax, color="#d8893b")
    ax.set(title="Missing-value rate by column", xlabel="fraction missing")
    _save(fig, "02_missingness.png")
    print("    -> missingness correlates with a LOWER target => add missing indicators.")


def correlations(tr):
    print("\n[3] CORRELATION WITH TARGET (numeric)")
    num = tr.select_dtypes(include=[np.number]).drop(columns=[config.TARGET])
    corr = num.corrwith(tr[config.TARGET]).sort_values(key=np.abs, ascending=False)
    print(corr.head(15).round(3).to_string())
    fig, ax = plt.subplots(figsize=(7, 6))
    corr.head(20).iloc[::-1].plot.barh(ax=ax, color="#3bd87d")
    ax.set(title="Top-20 |Pearson r| with target")
    _save(fig, "03_correlation.png")
    print(f"    -> strongest: {corr.index[0]} (r={corr.iloc[0]:.2f}); cgpa/attendance ~0.")


def importance(tr):
    print("\n[4] PERMUTATION FEATURE IMPORTANCE (LightGBM)")
    import lightgbm as lgb
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    cats = [c for c in config.CAT_COLS if c in tr.columns]
    X = tr.select_dtypes(include=[np.number]).drop(columns=[config.TARGET]).copy()
    for c in cats:
        X[c] = tr[c].astype("category").cat.codes
    y = tr[config.TARGET].values
    tri, vai = train_test_split(np.arange(len(y)), test_size=0.2, random_state=0)
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=48, verbosity=-1)
    m.fit(X.iloc[tri], y[tri])
    r = permutation_importance(m, X.iloc[vai], y[vai], n_repeats=5, random_state=0,
                               scoring="neg_mean_squared_error")
    imp = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
    print(imp.head(15).round(3).to_string())
    fig, ax = plt.subplots(figsize=(7, 6))
    imp.head(20).iloc[::-1].plot.barh(ax=ax, color="#9b59b6")
    ax.set(title="Top-20 permutation importance")
    _save(fig, "04_importance.png")


def text_signal(tr):
    print("\n[5] MENTOR-FEEDBACK TEXT SIGNAL")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    txt = tr[config.TEXT_COL].fillna("")
    y = tr[config.TARGET].values
    lens = txt.str.len()
    print(f"    char length: mean={lens.mean():.0f} p95={np.percentile(lens,95):.0f}")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2, sublinear_tf=True)
    M = tfidf.fit_transform(txt)
    oof = np.zeros(len(y))
    for trn, va in KFold(5, shuffle=True, random_state=0).split(M):
        oof[va] = Ridge(alpha=2.0).fit(M[trn], y[trn]).predict(M[va])
    r2 = 1 - mean_squared_error(y, oof) / y.var()
    print(f"    text-only Ridge(TF-IDF) OOF R^2 = {r2:.3f}  (real but mostly redundant with tabular)")


def distribution_shift(tr, te):
    print("\n[6] DISTRIBUTION SHIFT (train vs test)")
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    yr_tr = tr["application_year"].value_counts(normalize=True).sort_index()
    yr_te = te["application_year"].value_counts(normalize=True).sort_index()
    comp = pd.DataFrame({"train": yr_tr, "test": yr_te})
    print("    application_year distribution:")
    print(comp.round(3).to_string())
    X = pd.concat([tr.drop(columns=[config.TARGET]), te], ignore_index=True)
    X = X.drop(columns=[config.ID_COL, config.TEXT_COL])
    for c in X.select_dtypes(include="object").columns:
        X[c] = X[c].astype("category")
    lab = np.r_[np.zeros(len(tr)), np.ones(len(te))]
    p = cross_val_predict(lgb.LGBMClassifier(n_estimators=300, verbosity=-1),
                          X, lab, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          method="predict_proba")[:, 1]
    print(f"    adversarial AUC (train vs test) = {roc_auc_score(lab, p):.3f}  (0.5 = no shift)")
    fig, ax = plt.subplots(figsize=(7, 4))
    comp.plot.bar(ax=ax)
    ax.set(title="application_year: train vs test (test over-samples recent years)",
           ylabel="proportion")
    _save(fig, "06_year_shift.png")
    print("    -> test over-samples 2024-2026 => optimise a YEAR-WEIGHTED objective.")


def main():
    if not config.DATA_TRAIN.exists():
        raise SystemExit(f"Place train.csv/test_x.csv in {config.DATA_DIR}/ first.")
    tr = pd.read_csv(config.DATA_TRAIN)
    te = pd.read_csv(config.DATA_TEST)
    print(f"train: {tr.shape}   test: {te.shape}")
    target_analysis(tr)
    missingness(tr)
    correlations(tr)
    importance(tr)
    text_signal(tr)
    distribution_shift(tr, te)
    print("\nEDA complete. Figures written to figures/.")


if __name__ == "__main__":
    main()
