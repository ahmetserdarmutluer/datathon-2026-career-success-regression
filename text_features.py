"""Mentor-feedback text features (the corpus is Turkish).

Three complementary views, all of which feed the tabular models:

1. Static compressed representations — TruncatedSVD of TF-IDF (word 1-3
   grams per the spec, plus char_wb 3-5 grams because Turkish is
   agglutinative and char n-grams normalise suffixes), and PCA of sentence
   embeddings. Small dim counts: trees dilute over wide blocks.

2. Lexicon/shape statistics — counts of positive/negative mentor phrases,
   contrast markers ("ancak/fakat/ama"), lengths. Interpretable and robust.

3. Nested meta-model features — out-of-fold predictions of Ridge-on-TFIDF,
   Ridge-on-embeddings and KNN-on-embeddings. These compress the whole text
   block into 3 strong columns; built with the same nested-CV machinery as
   target encoding so downstream CV stays honest.

Embeddings: multilingual-e5-large (primary, asked for) vs
paraphrase-multilingual-mpnet-base-v2 (comparison) — both multilingual, so
they handle Turkish. instructor-xl/BGE-en are English-tuned: ruled out.
Embeddings are cached to artifacts/text_cache keyed by (model, corpus hash).
"""
from __future__ import annotations

import hashlib

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

import config
from feature_engineering import DynamicFeatures, nested_feature
from utils import log, timer

POSITIVE_LEX = ["güçlü", "dikkat çekici", "etkileyici", "başarılı", "umut verici",
                "yüksek", "mükemmel", "üstün", "yetkin", "uzman", "olumlu",
                "parlak", "sağlam", "potansiyel", "ileri düzey", "takdir"]
NEGATIVE_LEX = ["geliştirmesi", "gelişim göster", "eksik", "zayıf", "yetersiz",
                "düşük", "ihtiyaç", "daha fazla", "çalışması gerek", "odaklanmalı",
                "geride", "sınırlı"]
CONTRAST_LEX = ["ancak", "fakat", "ama ", "buna rağmen", "öte yandan"]


def _corpus_hash(texts: list[str]) -> str:
    h = hashlib.md5()
    for t in texts:
        h.update(t.encode("utf-8"))
    return h.hexdigest()[:12]


def encode_texts(texts: list[str], model_key: str) -> np.ndarray:
    """Sentence embeddings with on-disk cache. Uses MPS (Apple GPU) if present."""
    model_name = config.EMBEDDING_MODELS[model_key]
    cache = config.TEXT_CACHE / f"emb_{model_key}_{_corpus_hash(texts)}.npy"
    if cache.exists():
        return np.load(cache)
    import torch
    from sentence_transformers import SentenceTransformer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    with timer(f"embedding {len(texts)} texts with {model_name} on {device}"):
        model = SentenceTransformer(model_name, device=device)
        prefixed = ([config.E5_PREFIX + t for t in texts]
                    if model_key.startswith("e5") else list(texts))
        emb = model.encode(prefixed, batch_size=config.EMB_BATCH_SIZE,
                           normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)
    np.save(cache, emb)
    return emb


import re

# Skill keyword -> canonical name. The mentor feedback is rigidly templated:
# praised skills come before a contrast marker ("Ancak", "Bununla birlikte"),
# weaknesses after it. If the generating template chose skills from latent
# (noise-free) attribute values, these mentions carry *denoised* signal the
# tabular columns cannot — probe: +2.1 MSE over raw tabular from counts alone.
MENTION_SKILLS = {
    "kodlama": "coding", "yazılım": "coding", "problem çözme": "probsolve",
    "analitik": "probsolve", "veri yapıları": "datastruct", "sql": "sql",
    "veritabanı": "sql", "makine öğren": "ml", "yapay zeka": "ml",
    "veri bilimi": "ml", "backend": "backend", "frontend": "frontend",
    "bulut": "cloud", "cloud": "cloud", "devops": "devops",
    "iletişim": "comm", "takım": "team", "ekip": "team", "işbirliği": "team",
    "liderlik": "lead", "sunum": "present", "proje kalite": "projq",
    "proje": "proj", "portföy": "portfolio", "github": "portfolio",
    "staj": "intern", "müşteri": "client", "teknik görüşme": "techint",
    "görüşme": "interview", "ingilizce": "english", "sertifika": "cert",
    "hackathon": "hack",
}
CONTRAST_SPLIT = r"(?:ancak|fakat|bununla birlikte|buna rağmen|öte yandan|ama )"


def mention_features(texts: pd.Series) -> pd.DataFrame:
    """Praise/criticism skill mentions split on the contrast marker."""
    rows = []
    for t in texts:
        tl = t.lower()
        parts = re.split(CONTRAST_SPLIT, tl, maxsplit=1)
        praise, crit = parts[0], (parts[1] if len(parts) > 1 else "")
        r = {}
        for kw, name in MENTION_SKILLS.items():
            r[f"m_praise_{name}"] = r.get(f"m_praise_{name}", 0) + praise.count(kw)
            r[f"m_crit_{name}"] = r.get(f"m_crit_{name}", 0) + crit.count(kw)
        for name in set(MENTION_SKILLS.values()):
            r[f"m_net_{name}"] = r[f"m_praise_{name}"] - r[f"m_crit_{name}"]
        r["m_has_contrast"] = int(len(parts) > 1)
        r["m_praise_total"] = sum(v for k, v in r.items()
                                  if k.startswith("m_praise_"))
        r["m_crit_total"] = sum(v for k, v in r.items()
                                if k.startswith("m_crit_"))
        rows.append(r)
    return pd.DataFrame(rows).fillna(0).astype(np.float32)


def lexicon_features(texts: pd.Series) -> pd.DataFrame:
    low = texts.str.lower()
    X = pd.DataFrame(index=texts.index)
    X["text_char_len"] = texts.str.len()
    X["text_word_count"] = texts.str.split().str.len()
    X["text_sentence_count"] = texts.str.count(r"\.") + 1
    X["text_pos_hits"] = sum(low.str.count(w) for w in POSITIVE_LEX)
    X["text_neg_hits"] = sum(low.str.count(w) for w in NEGATIVE_LEX)
    X["text_contrast_hits"] = sum(low.str.count(w) for w in CONTRAST_LEX)
    X["text_sentiment_balance"] = X["text_pos_hits"] - X["text_neg_hits"]
    return X.astype(np.float32)


class TextFeatureFactory:
    """Fits all text transforms on train; persists for inference."""

    def __init__(self, emb_key: str = config.PRIMARY_EMBEDDING,
                 use_embeddings: bool = True):
        self.emb_key = emb_key
        self.use_embeddings = use_embeddings

    # ------------------------------------------------------------ fitting
    def fit(self, texts_train: pd.Series, texts_test: pd.Series):
        with timer("TF-IDF (word 1-3 + char_wb 3-5)"):
            self.tfidf_word_ = TfidfVectorizer(**config.TFIDF_WORD)
            self.tfidf_char_ = TfidfVectorizer(**config.TFIDF_CHAR)
            Tw = self.tfidf_word_.fit_transform(texts_train)
            Tc = self.tfidf_char_.fit_transform(texts_train)
            self.M_tfidf_train_ = sparse.hstack([Tw, Tc]).tocsr()
            self.M_tfidf_test_ = sparse.hstack(
                [self.tfidf_word_.transform(texts_test),
                 self.tfidf_char_.transform(texts_test)]).tocsr()
            self.svd_tfidf_ = TruncatedSVD(config.SVD_DIMS_TFIDF,
                                           random_state=config.SEED)
            self.svd_tfidf_.fit(self.M_tfidf_train_)

        if self.use_embeddings:
            self.E_train_ = encode_texts(texts_train.tolist(), self.emb_key)
            self.E_test_ = encode_texts(texts_test.tolist(), self.emb_key)
            self.pca_emb_ = PCA(config.SVD_DIMS_EMB, random_state=config.SEED)
            self.pca_emb_.fit(self.E_train_)
        return self

    # ------------------------------------------------------------ static
    def static_features(self, texts: pd.Series, split: str) -> pd.DataFrame:
        """split in {'train','test'} — picks the right cached matrices."""
        M = self.M_tfidf_train_ if split == "train" else self.M_tfidf_test_
        out = [lexicon_features(texts).reset_index(drop=True),
               mention_features(texts).reset_index(drop=True),
               pd.DataFrame(self.svd_tfidf_.transform(M),
                            columns=[f"tfidf_svd_{i}" for i in range(config.SVD_DIMS_TFIDF)])]
        if self.use_embeddings:
            E = self.E_train_ if split == "train" else self.E_test_
            out.append(pd.DataFrame(self.pca_emb_.transform(E),
                                    columns=[f"emb_pca_{i}" for i in range(config.SVD_DIMS_EMB)]))
        return pd.concat(out, axis=1)

    # ------------------------------------------------------------ nested meta features
    def _matrix_feature(self, name, M_train, M_test, model_factory,
                        y, folds) -> DynamicFeatures:
        def build(fit_idx):
            mdl = model_factory()
            mdl.fit(M_train[fit_idx], y[fit_idx])

            def predict(idx):
                X = M_test if isinstance(idx, str) else M_train[idx]
                return mdl.predict(X)
            return predict
        return nested_feature(name, build, y, folds, n_test=M_test.shape[0])

    def meta_features(self, y: np.ndarray, folds: list) -> list[DynamicFeatures]:
        feats = [self._matrix_feature(
            "txt_ridge_tfidf", self.M_tfidf_train_, self.M_tfidf_test_,
            lambda: Ridge(alpha=2.0), y, folds)]
        if self.use_embeddings:
            feats.append(self._matrix_feature(
                "txt_ridge_emb", self.E_train_, self.E_test_,
                lambda: Ridge(alpha=1.0), y, folds))
            feats.append(self._matrix_feature(
                "txt_knn_emb", self.E_train_, self.E_test_,
                lambda: KNeighborsRegressor(
                    n_neighbors=config.KNN_TEXT_NEIGHBORS, weights="distance"),
                y, folds))
        return feats

    # ------------------------------------------------------------ inference support
    def fit_full_train_meta_models(self, y: np.ndarray):
        """Full-train text meta models, used to featurize unseen test data."""
        self.full_ridge_tfidf_ = Ridge(alpha=2.0).fit(self.M_tfidf_train_, y)
        if self.use_embeddings:
            self.full_ridge_emb_ = Ridge(alpha=1.0).fit(self.E_train_, y)
            self.full_knn_emb_ = KNeighborsRegressor(
                n_neighbors=config.KNN_TEXT_NEIGHBORS,
                weights="distance").fit(self.E_train_, y)

    def featurize_new(self, texts: pd.Series):
        """Featurize unseen texts at inference time.

        Returns (static_df, dyn_dict) matching the training-time column
        names; dyn values come from the full-train meta models.
        """
        M = sparse.hstack([self.tfidf_word_.transform(texts),
                           self.tfidf_char_.transform(texts)]).tocsr()
        out = [lexicon_features(texts).reset_index(drop=True),
               mention_features(texts).reset_index(drop=True),
               pd.DataFrame(self.svd_tfidf_.transform(M),
                            columns=[f"tfidf_svd_{i}" for i in range(config.SVD_DIMS_TFIDF)])]
        dyn = {"txt_ridge_tfidf": self.full_ridge_tfidf_.predict(M)}
        if self.use_embeddings:
            E = encode_texts(texts.tolist(), self.emb_key)
            out.append(pd.DataFrame(self.pca_emb_.transform(E),
                                    columns=[f"emb_pca_{i}" for i in range(config.SVD_DIMS_EMB)]))
            dyn["txt_ridge_emb"] = self.full_ridge_emb_.predict(E)
            dyn["txt_knn_emb"] = self.full_knn_emb_.predict(E)
        return pd.concat(out, axis=1), dyn

    def save(self, path):
        # embeddings live in the npy cache; don't duplicate them in the pickle
        E_tr, E_te = getattr(self, "E_train_", None), getattr(self, "E_test_", None)
        M_tr, M_te = self.M_tfidf_train_, self.M_tfidf_test_
        self.E_train_ = self.E_test_ = None
        self.M_tfidf_train_ = self.M_tfidf_test_ = None
        joblib.dump(self, path)
        self.E_train_, self.E_test_ = E_tr, E_te
        self.M_tfidf_train_, self.M_tfidf_test_ = M_tr, M_te
        log.info("text artifacts saved -> %s", path)


def compare_embedding_models(texts_train, y, folds) -> dict[str, float]:
    """Quick bake-off: 5-fold Ridge OOF MSE per representation."""
    from sklearn.metrics import mean_squared_error
    results = {}
    tf = TfidfVectorizer(**config.TFIDF_WORD)
    M = tf.fit_transform(texts_train)
    oof = np.zeros(len(y))
    for tr, va in folds:
        oof[va] = Ridge(alpha=2.0).fit(M[tr], y[tr]).predict(M[va])
    results["tfidf_word"] = mean_squared_error(y, oof)
    for key in config.EMBEDDING_MODELS:
        try:
            E = encode_texts(texts_train.tolist(), key)
        except Exception as e:                                  # noqa: BLE001
            log.warning("embedding %s failed (%s); skipping", key, e)
            continue
        oof = np.zeros(len(y))
        for tr, va in folds:
            oof[va] = Ridge(alpha=1.0).fit(E[tr], y[tr]).predict(E[va])
        results[key] = mean_squared_error(y, oof)
    return results
