"""Tabular feature engineering.

Two kinds of features:

1. Static features (`TabularFeatureBuilder`) — functions of the predictor
   columns only. Statistics (z-score params, role group means, frequency
   tables) are fit on train and applied identically to test.

2. Fold-aware dynamic features (`nested_feature`) — anything that uses the
   target (target encoding here; text meta-models in text_features.py).
   These are recomputed per outer CV fold with an *inner* CV inside the
   fold's training part, so a model validated on fold k never sees a feature
   derived from fold k's targets. This is the leakage-safe "nested CV"
   construction; the alternative (one global OOF column) leaks val-fold
   targets into train rows and inflates CV scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from utils import make_folds

# ---------------------------------------------------------------- groups
TECH_CORE = ["coding_score", "problem_solving_score", "data_structures_score",
             "sql_score"]
ENGINEERING = ["backend_score", "frontend_score", "cloud_score", "devops_score"]
INTERVIEW = ["technical_interview_score", "hr_interview_score",
             "communication_score", "presentation_score"]
SOFT = ["communication_score", "teamwork_score", "leadership_score",
        "presentation_score"]
PORTFOLIO_RAW = ["portfolio_score", "linkedin_profile_score", "cv_quality_score"]
EXPERIENCE_RAW = ["project_quality_score", "internship_duration_months"]

# Which skill columns matter for each target role (domain-driven
# Categorical x Technical interaction). Trees can only find this with many
# splits; giving it directly is cheap and interpretable.
ROLE_SKILLS = {
    "Backend Developer": ["backend_score", "sql_score", "data_structures_score"],
    "Frontend Developer": ["frontend_score", "coding_score"],
    "Software Developer": ["coding_score", "data_structures_score",
                           "problem_solving_score"],
    "Data Scientist": ["machine_learning_score", "sql_score",
                       "problem_solving_score"],
    "Data Analyst": ["sql_score", "problem_solving_score"],
    "Cloud Engineer": ["cloud_score", "devops_score"],
    "AI Engineer": ["machine_learning_score", "coding_score"],
    "DevOps Engineer": ["devops_score", "cloud_score"],
    "Cybersecurity Analyst": ["problem_solving_score", "sql_score"],
    "MLOps Engineer": ["machine_learning_score", "devops_score", "cloud_score"],
    "Product Analyst": ["sql_score", "communication_score",
                        "problem_solving_score"],
}


class TabularFeatureBuilder:
    """Fit on train, transform train/test identically. joblib-picklable."""

    def fit(self, df: pd.DataFrame) -> "TabularFeatureBuilder":
        self.z_stats_ = {}
        for c in (PORTFOLIO_RAW + EXPERIENCE_RAW
                  + ["github_avg_stars"]):
            self.z_stats_[c] = (float(df[c].mean()), float(df[c].std() + 1e-9))
        for c in ["github_repo_count", "open_source_contribution_count",
                  "freelance_project_count", "internship_count",
                  "real_client_project_count"]:
            v = np.log1p(df[c])
            self.z_stats_["log1p_" + c] = (float(v.mean()), float(v.std() + 1e-9))

        # role-conditional means of key scores (unsupervised group stats)
        self.role_means_ = {}
        tmp = self._base(df)
        for c in ["technical_strength", "engineering_strength",
                  "project_quality_score", "technical_interview_score",
                  "role_skill_match"]:
            src = tmp[c] if c in tmp.columns else df[c]
            self.role_means_[c] = src.groupby(df["target_role"]).mean().to_dict()

        self.freq_maps_ = {c: (df[c].value_counts(normalize=True)).to_dict()
                           for c in config.CAT_COLS}
        self.cat_categories_ = {c: sorted(df[c].astype(str).unique())
                                for c in config.CAT_COLS}
        return self

    # -------------------------------------------------- internal helpers
    def _z(self, s: pd.Series, key: str) -> pd.Series:
        m, sd = self.z_stats_[key]
        return (s - m) / sd

    def _base(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features that need no fitted stats (safe to call inside fit)."""
        X = pd.DataFrame(index=df.index)

        # ---- aggregates
        X["academic_strength"] = df["cgpa"] * df["attendance_rate"]
        X["technical_strength"] = df[TECH_CORE].mean(axis=1)
        X["engineering_strength"] = df[ENGINEERING].mean(axis=1)
        X["interview_strength"] = df[INTERVIEW].mean(axis=1)
        X["soft_strength"] = df[SOFT].mean(axis=1)
        X["data_ml_strength"] = df[["sql_score", "machine_learning_score"]].mean(axis=1)
        X["all_tech_mean"] = df[TECH_CORE + ENGINEERING
                                + ["machine_learning_score"]].mean(axis=1)
        X["tech_score_std"] = df[TECH_CORE + ENGINEERING
                                 + ["machine_learning_score"]].std(axis=1)
        X["tech_score_max"] = df[TECH_CORE + ENGINEERING
                                 + ["machine_learning_score"]].max(axis=1)

        # ---- role alignment (Categorical x Technical)
        match = np.full(len(df), np.nan)
        for role, cols in ROLE_SKILLS.items():
            mask = (df["target_role"] == role).to_numpy()
            if mask.any():
                match[mask] = df.loc[mask, cols].mean(axis=1).to_numpy()
        X["role_skill_match"] = match
        X["role_skill_gap"] = X["role_skill_match"] - X["all_tech_mean"]

        # ---- ratios / rates
        X["interview_conversion"] = df["interviews_attended"] / (df["applications_sent"] + 1)
        X["hackathon_win_rate"] = df["hackathon_awards"] / (df["hackathon_count"] + 1)
        X["osc_per_repo"] = df["open_source_contribution_count"] / (df["github_repo_count"] + 1)
        X["stars_total"] = df["github_repo_count"] * df["github_avg_stars"]
        X["months_per_internship"] = df["internship_duration_months"] / (df["internship_count"] + 1)
        X["failed_ratio"] = df["failed_courses_count"] / (df["cgpa"] + 0.1)

        # ---- timeline
        X["years_since_graduation"] = df["application_year"] - df["graduation_year"]
        X["age_at_graduation"] = df["age"] - X["years_since_graduation"]

        # ---- curated interactions (Academic/Tech/Portfolio/Experience/Soft)
        X["pq_x_tech_interview"] = df["project_quality_score"] * df["technical_interview_score"]
        X["pq_x_portfolio"] = df["project_quality_score"] * df["portfolio_score"]
        X["pq_x_real_clients"] = df["project_quality_score"] * np.log1p(df["real_client_project_count"])
        X["tech_x_interview"] = X["technical_strength"] * X["interview_strength"]
        X["tech_x_portfolio"] = X["technical_strength"] * df["portfolio_score"]
        X["soft_x_interview"] = X["soft_strength"] * df["hr_interview_score"]
        X["acad_x_tech"] = X["academic_strength"] * X["technical_strength"]
        X["tech_interview_minus_tech"] = df["technical_interview_score"] - X["all_tech_mean"]
        return X

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._base(df)

        # ---- missing indicators (EDA: missingness is informative, e.g.
        # missing LinkedIn score -> mean target 73.9 vs 77.2)
        for c in config.MISSING_COLS:
            X[c + "_missing"] = df[c].isna().astype(np.int8)
        X["github_profile_missing"] = df["github_avg_stars"].isna().astype(np.int8)
        X["intern_dur_missing_but_has_intern"] = (
            df["internship_duration_months"].isna() & (df["internship_count"] > 0)
        ).astype(np.int8)
        X["n_missing"] = df[config.MISSING_COLS].isna().sum(axis=1).astype(np.int8)

        # ---- z-score composites (NaN-aware means over standardized parts)
        zp = [self._z(df["portfolio_score"], "portfolio_score"),
              self._z(df["linkedin_profile_score"], "linkedin_profile_score"),
              self._z(df["cv_quality_score"], "cv_quality_score"),
              self._z(np.log1p(df["github_repo_count"]), "log1p_github_repo_count"),
              self._z(df["github_avg_stars"], "github_avg_stars"),
              self._z(np.log1p(df["open_source_contribution_count"]),
                      "log1p_open_source_contribution_count")]
        X["portfolio_strength"] = pd.concat(zp, axis=1).mean(axis=1)

        ze = [self._z(np.log1p(df["internship_count"]), "log1p_internship_count"),
              self._z(df["internship_duration_months"], "internship_duration_months"),
              self._z(np.log1p(df["freelance_project_count"]), "log1p_freelance_project_count"),
              self._z(df["project_quality_score"], "project_quality_score"),
              self._z(np.log1p(df["real_client_project_count"]), "log1p_real_client_project_count")]
        X["experience_strength"] = pd.concat(ze, axis=1).mean(axis=1)

        X["exp_x_tech"] = X["experience_strength"] * X["technical_strength"]
        X["portfolio_x_interview"] = X["portfolio_strength"] * X["interview_strength"]

        # ---- role-relative scores (is this student above peers aiming at
        # the same role?)
        for c, means in self.role_means_.items():
            base = X[c] if c in X.columns else df[c]
            X[c + "_vs_role"] = base - df["target_role"].map(means).astype(float)

        # ---- ordinal + frequency encodings
        X["university_tier_ord"] = df["university_tier"].map(config.TIER_ORDER).astype(float)
        if config.USE_FREQUENCY_ENCODING:
            for c in config.CAT_COLS:
                X[c + "_freq"] = df[c].map(self.freq_maps_[c]).astype(float)

        # ---- passthrough raw columns
        raw_num = df.select_dtypes(include=[np.number]).drop(
            columns=[config.TARGET], errors="ignore")
        X = pd.concat([raw_num, X], axis=1)

        # ---- categoricals as pandas category dtype (shared category sets so
        # train/test codes align for LightGBM/XGBoost/HistGB native handling)
        for c in config.CAT_COLS:
            X[c] = pd.Categorical(df[c].astype(str),
                                  categories=self.cat_categories_[c])
        return X


# ------------------------------------------------------------------ auto interactions
def add_auto_interactions(X_tr: pd.DataFrame, X_te: pd.DataFrame, y: np.ndarray,
                          top_n: int = config.AUTO_INTERACTIONS_TOP_N):
    """Pairwise products of the top-|corr| numeric features.

    GBDTs approximate interactions through sequential splits; explicit
    products of the few globally strongest features give them (and the MLP)
    smooth access to the same structure. Selecting pairs via train
    correlations touches y only through a global ranking — negligible
    selection leakage at n=10k, and the ablation validates the net effect.
    """
    num = X_tr.select_dtypes(include=[np.number])
    corr = num.apply(lambda s: np.abs(np.corrcoef(s.fillna(s.median()), y)[0, 1]))
    top = corr.sort_values(ascending=False).head(top_n).index.tolist()
    for i, a in enumerate(top):
        for b in top[i + 1:]:
            name = f"ax_{a}__x__{b}"
            X_tr[name] = X_tr[a] * X_tr[b]
            X_te[name] = X_te[a] * X_te[b]
    return X_tr, X_te, top


# ------------------------------------------------------------------ nested dynamic features
class DynamicFeatures:
    """Fold-aware feature columns.

    train_cols[k] : full-length array valid when validating outer fold k
                    (val rows: fit on fold-k train; train rows: inner OOF)
    test_col      : fit on all training data, applied to test
    """

    def __init__(self, name: str, train_cols: list[np.ndarray], test_col: np.ndarray):
        self.name = name
        self.train_cols = train_cols
        self.test_col = test_col


def nested_feature(name: str, build, y: np.ndarray,
                   outer_folds: list, n_test: int,
                   inner_splits: int = config.INNER_SPLITS,
                   seed: int = config.SEED) -> DynamicFeatures:
    """Generic nested-CV feature constructor.

    `build(fit_idx)` must return a callable: predict(row_idx | "test") -> array.
    """
    n = len(y)
    train_cols = []
    for k, (tr_idx, va_idx) in enumerate(outer_folds):
        col = np.full(n, np.nan)
        col[va_idx] = build(tr_idx)(va_idx)
        inner = make_folds(y[tr_idx], n_splits=inner_splits, seed=seed + 31 * k)
        for itr, iva in inner:
            col[tr_idx[iva]] = build(tr_idx[itr])(tr_idx[iva])
        train_cols.append(col)
    test_col = build(np.arange(n))("test")
    return DynamicFeatures(name, train_cols, test_col)


def target_encoding_feature(col: str, cat_values_train: pd.Series,
                            cat_values_test: pd.Series, y: np.ndarray,
                            outer_folds: list,
                            smoothing: float = config.TE_SMOOTHING) -> DynamicFeatures:
    """Leakage-safe smoothed target encoding via nested CV."""
    tr_vals = cat_values_train.astype(str).to_numpy()
    te_vals = cat_values_test.astype(str).to_numpy()

    def build(fit_idx):
        prior = y[fit_idx].mean()
        s = pd.Series(y[fit_idx]).groupby(tr_vals[fit_idx]).agg(["sum", "count"])
        enc = ((s["sum"] + prior * smoothing) / (s["count"] + smoothing)).to_dict()

        def predict(idx):
            vals = te_vals if isinstance(idx, str) else tr_vals[idx]
            return pd.Series(vals).map(enc).fillna(prior).to_numpy()
        return predict

    return nested_feature(f"te_{col}", build, y, outer_folds,
                          n_test=len(te_vals))


def build_target_encodings(train_df, test_df, y, outer_folds) -> list[DynamicFeatures]:
    if not config.USE_TARGET_ENCODING:
        return []
    return [target_encoding_feature(c, train_df[c], test_df[c], y, outer_folds)
            for c in config.TE_COLS]
