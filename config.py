"""Central configuration for the career-success regression pipeline.

Every knob that affects an experiment lives here so a run is fully described
by (git state of this file, RANDOM_SEED). train.py --budget {smoke,fast,full}
overlays the BUDGETS table onto HPO so the same code path is used end-to-end.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"               # place train.csv / test_x.csv here
DATA_TRAIN = DATA_DIR / "train.csv"
DATA_TEST = DATA_DIR / "test_x.csv"
ARTIFACTS = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
TEXT_CACHE = ARTIFACTS / "text_cache"
PLOTS_DIR = ARTIFACTS / "plots"
HPO_DIR = ARTIFACTS / "hpo"
EXPERIMENTS_JSON = ARTIFACTS / "experiments.json"
SUBMISSION_PATH = ROOT / "submission.csv"
USE_MLFLOW = False                               # opt-in; experiments.json is the default tracker
MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT = "career_success_score"

for _p in (ARTIFACTS, MODELS_DIR, TEXT_CACHE, PLOTS_DIR, HPO_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- columns
ID_COL = "student_id"
TARGET = "career_success_score"
TEXT_COL = "mentor_feedback_text"

# hobby and preferred_social_media_platform were dropped: rigorous multi-seed
# CV showed they are a-priori lifestyle noise (dropping them was the best CV
# variant, 96.80 vs 96.99 baseline) and the GBDTs gain nothing from them.
# The other low-importance columns are KEPT — dropping more than these two
# made CV worse (drop-all-4: 97.11; broad set: 97.88).
DROPPED_NOISE_COLS = ["hobby", "preferred_social_media_platform"]
CAT_COLS = [
    "department",
    "university_tier",
    "target_role",
]

# Tier is ordinal — keep a numeric version too so split-based models can
# exploit the order directly instead of learning it from 4 unordered levels.
TIER_ORDER = {"Tier 1": 1, "Tier 2": 2, "Tier 3": 3, "Tier 4": 4}

# Columns with missing values (train and test show the same pattern).
# EDA: github_avg_stars and open_source_contribution_count are missing on the
# same rows, and missingness correlates with a LOWER target -> indicators are
# genuine features, not noise.
MISSING_COLS = [
    "english_exam_score",
    "internship_duration_months",
    "portfolio_score",
    "github_avg_stars",
    "open_source_contribution_count",
    "linkedin_profile_score",
    "hr_interview_score",
]

# ---------------------------------------------------------------- CV
SEED = 42
N_FOLDS = 5
N_TARGET_BINS = 20          # quantile bins for stratified regression CV
FINAL_REPEATS = 3           # repeated 5-fold for final models (seeds below)
REPEAT_SEEDS = [42, 2025, 7]
INNER_SPLITS = 4            # inner folds for nested (leakage-safe) features

# -------- distribution shift correction (found via adversarial validation:
# AUC 0.65; test over-samples 2024-2026 where the target is noisier/lower;
# year-reweighted OOF reproduced the leaderboard score to within 0.21).
YEAR_COL = "application_year"
USE_YEAR_WEIGHTS = True       # weighted metrics/referee/stack (LB-aligned)
USE_YEAR_WEIGHTS_TRAIN = False  # A/B verdict: weighted base fits lose
                                # (85.40 vs 84.82 honest wMSE) — recent years
                                # are noisier, not differently structured, so
                                # reweighting just burns effective sample size
WEIGHT_CLIP = (0.25, 3.0)
STRATIFY_BY_YEAR = True     # fold strata = target-bin x year
N_TARGET_BINS_YEAR = 10     # coarser bins when crossing with 8 years

# ---------------------------------------------------------------- text
# Mentor feedback is Turkish -> multilingual encoders only. instructor-xl is
# English-instruction-tuned and was ruled out for this corpus.
EMBEDDING_MODELS = {
    "e5large": "intfloat/multilingual-e5-large",       # primary (asked for)
    "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
}
PRIMARY_EMBEDDING = "e5large"
E5_PREFIX = "query: "        # e5 family expects a prefix at encode time
EMB_BATCH_SIZE = 64

TFIDF_WORD = dict(ngram_range=(1, 3), max_features=5000, min_df=2,
                  sublinear_tf=True)
TFIDF_CHAR = dict(analyzer="char_wb", ngram_range=(3, 5), max_features=30000,
                  min_df=3, sublinear_tf=True)   # char n-grams suit Turkish morphology
SVD_DIMS_TFIDF = 24
SVD_DIMS_EMB = 32
KNN_TEXT_NEIGHBORS = 32

# ---------------------------------------------------------------- features
USE_TARGET_ENCODING = True      # nested-CV target encoding (leakage-safe)
TE_SMOOTHING = 20.0
TE_COLS = ["target_role", "department", "university_tier", "application_year"]
USE_FREQUENCY_ENCODING = True
USE_AUTO_INTERACTIONS = True    # pairwise products of top-correlated features
AUTO_INTERACTIONS_TOP_N = 8

# ---------------------------------------------------------------- models / HPO
MODEL_NAMES = ["catboost", "lightgbm", "xgboost", "extratrees", "histgb", "nn_mlp"]

# n_trials=200 is the request; wall-clock timeouts are safety rails so one
# slow model cannot starve the rest of the pipeline. Reached-timeout runs
# report the number of completed trials honestly.
BUDGETS = {
    "full": {
        # catboost/lightgbm consumed their wall budgets in the first session
        # (51 and 34 completed trials); n_trials matches so a resumed run
        # skips straight to the unfinished models.
        "catboost":  dict(n_trials=51, timeout=4200),
        "lightgbm":  dict(n_trials=34, timeout=2400),
        "xgboost":   dict(n_trials=200, timeout=2700),
        "extratrees": dict(n_trials=200, timeout=2100),
        "histgb":    dict(n_trials=200, timeout=1800),
        "nn_mlp":    dict(n_trials=30,  timeout=2400),
    },
    "fast": {
        "catboost":  dict(n_trials=40, timeout=1200),
        "lightgbm":  dict(n_trials=60, timeout=700),
        "xgboost":   dict(n_trials=60, timeout=800),
        "extratrees": dict(n_trials=30, timeout=600),
        "histgb":    dict(n_trials=40, timeout=500),
        "nn_mlp":    dict(n_trials=10, timeout=700),
    },
    "smoke": {m: dict(n_trials=2, timeout=240) for m in MODEL_NAMES},
    # fresh Optuna studies under the year-weighted objective
    "fullw": {
        "catboost":  dict(n_trials=150, timeout=5400),
        "lightgbm":  dict(n_trials=150, timeout=3600),
        "xgboost":   dict(n_trials=150, timeout=3600),
        "extratrees": dict(n_trials=60, timeout=1500),
        "histgb":    dict(n_trials=120, timeout=1500),
        "nn_mlp":    dict(n_trials=25,  timeout=1500),
    },
}

GBDT_MAX_ESTIMATORS = 4000
EARLY_STOPPING_ROUNDS = 200

NN_DEVICE = "mps" if os.environ.get("NN_DEVICE", "auto") == "auto" else os.environ["NN_DEVICE"]
NN_MAX_EPOCHS = 200
NN_PATIENCE = 20

# Target is bounded in [0, 100]; clipping predictions can only reduce MSE.
CLIP_MIN, CLIP_MAX = 0.0, 100.0
