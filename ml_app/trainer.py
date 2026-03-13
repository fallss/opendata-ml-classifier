"""
trainer.py
==========
Training 5 model ML dengan GridSearchCV untuk klasifikasi
label 'group' dan 'organization'.

Strategi:
- Pipeline: FeatureUnion(word TF-IDF + char TF-IDF) → Model
- Scoring: f1_weighted (realistis untuk data tidak seimbang)
- 3-fold CV: kelas kecil butuh lebih banyak data training per fold
- class_weight='balanced': kelas minoritas diberi bobot lebih
- Grid lebar: cari kombinasi parameter optimal
"""

import os
import json
import shutil
import logging
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import joblib

from feature_extractor import (
    load_training_data, get_training_samples,
    fit_vectorizer, MODELS_DIR, CSV_PATH,
    DomainKeywordTransformer,
)

logger = logging.getLogger(__name__)
COMPARISON_PATH = os.path.join(MODELS_DIR, "model_comparison.json")


# ================================================================
# Pipeline
# ================================================================

def _make_pipeline(model, label_col: str = "organization") -> Pipeline:
    """
    FeatureUnion: word TF-IDF + char TF-IDF + DomainKeywordTransformer
    DomainKeywordTransformer memberi sinyal eksplisit domain per org/group
    sehingga classifier tidak sepenuhnya bergantung pada TF-IDF yang
    terganggu oleh teks lorem ipsum dalam data training.
    """
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=6000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=1,
        max_features=3000,
        sublinear_tf=True,
    )
    keyword_feats = DomainKeywordTransformer(label_col=label_col)

    union = FeatureUnion([
        ("word",    word_tfidf),
        ("char",    char_tfidf),
        ("keyword", keyword_feats),
    ])
    return Pipeline([("tfidf", union), ("clf", model)])


def _make_nb_pipeline(model, label_col: str = "organization") -> Pipeline:
    """ComplementNB tidak bisa pakai char TF-IDF + keyword (float), gunakan word only + keyword."""
    from scipy.sparse import hstack as sp_hstack
    from sklearn.pipeline import Pipeline
    # ComplementNB needs non-negative features — keyword counts are non-negative ✓
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=6000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    keyword_feats = DomainKeywordTransformer(label_col=label_col)
    union = FeatureUnion([("word", word_tfidf), ("keyword", keyword_feats)])
    return Pipeline([("tfidf", union), ("clf", model)])


# ================================================================
# 5 Model + Grid Parameter
# ================================================================
MODEL_CONFIGS = {
    "Logistic Regression": {
        "pipeline": "default",
        "model": LogisticRegression(
            random_state=42, max_iter=2000, class_weight="balanced"
        ),
        "params": {
            "clf__C":       [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
            "clf__solver":  ["lbfgs", "liblinear"],
            "clf__penalty": ["l2"],
        },
    },
    "Linear SVC": {
        "pipeline": "default",
        "model": LinearSVC(
            random_state=42, max_iter=5000, class_weight="balanced"
        ),
        "params": {
            "clf__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        },
    },
    "Random Forest": {
        "pipeline": "default",
        "model": RandomForestClassifier(
            random_state=42, class_weight="balanced"
        ),
        "params": {
            "clf__n_estimators":      [100, 200, 300],
            "clf__max_depth":         [None, 10, 20, 30],
            "clf__min_samples_split": [2, 3, 5],
        },
    },
    "Extra Trees": {
        "pipeline": "default",
        "model": ExtraTreesClassifier(
            random_state=42, class_weight="balanced"
        ),
        "params": {
            "clf__n_estimators":     [100, 200, 300],
            "clf__max_depth":        [None, 10, 20],
            "clf__min_samples_leaf": [1, 2, 3],
        },
    },
    "Complement NB": {
        "pipeline": "nb",
        "model": ComplementNB(),
        "params": {
            "clf__alpha": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            "clf__norm":  [True, False],
        },
    },
}


def train_all_models(
    label_col: str,
    csv_path: str = CSV_PATH,
    progress_callback=None,
) -> dict:
    """Train semua 5 model dengan GridSearchCV (f1_weighted, 3-fold)."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_training_data(csv_path)
    texts, labels = get_training_samples(df, label_col)

    if len(texts) < 5:
        raise ValueError(
            f"Data terlalu sedikit ({len(texts)} baris) untuk '{label_col}'"
        )

    n_classes = len(set(labels))
    logger.info(f"Training '{label_col}': {len(texts)} sampel, {n_classes} kelas")

    # Hapus kelas dengan hanya 1 sampel (tidak bisa di-CV)
    from collections import Counter
    count = Counter(labels)
    valid_mask = [count[l] >= 2 for l in labels]
    texts_f  = [t for t, m in zip(texts, valid_mask) if m]
    labels_f = [l for l, m in zip(labels, valid_mask) if m]
    if len(texts_f) < len(texts):
        dropped = len(texts) - len(texts_f)
        logger.warning(f"Drop {dropped} sampel dari kelas singleton")
        texts, labels = texts_f, labels_f

    fit_vectorizer(texts, save=True)

    # 3-fold lebih cocok untuk dataset kecil dengan banyak kelas
    n_splits = min(3, min(Counter(labels).values()))
    n_splits = max(n_splits, 2)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    logger.info(f"Menggunakan {n_splits}-fold CV")

    results = {}
    total_models = len(MODEL_CONFIGS)

    for idx, (model_name, cfg) in enumerate(MODEL_CONFIGS.items(), start=1):
        logger.info(f"[{idx}/{total_models}] {model_name}")
        if progress_callback:
            progress_callback(model_name, idx, total_models)

        # Pilih pipeline sesuai model
        if cfg["pipeline"] == "nb":
            pipeline = _make_nb_pipeline(cfg["model"], label_col=label_col)
        else:
            pipeline = _make_pipeline(cfg["model"], label_col=label_col)
        
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=cfg["params"],
            cv=cv,
            scoring="f1_weighted",   # realistis untuk data tidak seimbang
            n_jobs=-1,
            refit=True,
            error_score=0.0,
        )
        grid.fit(texts, labels)

        best = grid.best_estimator_
        best_params = {
            k.replace("clf__", ""): v
            for k, v in grid.best_params_.items()
        }

        preds = best.predict(texts)
        acc   = round(accuracy_score(labels, preds), 4)
        f1_tr = round(f1_score(labels, preds, average="weighted", zero_division=0), 4)
        cv_f1 = round(grid.best_score_, 4)

        results[model_name] = {
            "accuracy":    acc,
            "f1_weighted": f1_tr,
            "cv_f1":       cv_f1,
            "best_params": best_params,
        }

        model_path = os.path.join(
            MODELS_DIR,
            f"{label_col}_{model_name.lower().replace(' ', '_')}.joblib"
        )
        joblib.dump(best, model_path)
        logger.info(f"  ✓ CV F1_weighted={cv_f1:.4f} | Params={best_params}")

    best_model_name = max(results, key=lambda k: results[k]["cv_f1"])
    results.update({
        "_best_model": best_model_name,
        "_label_col":  label_col,
        "_trained_at": datetime.now().isoformat(),
        "_n_samples":  len(texts),
        "_n_classes":  len(set(labels)),
        "_scoring":    f"f1_weighted ({n_splits}-fold CV)",
    })

    best_src  = os.path.join(
        MODELS_DIR,
        f"{label_col}_{best_model_name.lower().replace(' ', '_')}.joblib"
    )
    shutil.copy2(best_src, os.path.join(MODELS_DIR, f"best_{label_col}.joblib"))
    logger.info(
        f"Best '{label_col}': {best_model_name}"
        f" (CV F1={results[best_model_name]['cv_f1']})"
    )
    return results


def save_comparison(group_results: dict, org_results: dict):
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(COMPARISON_PATH, "w") as f:
        json.dump({"group": group_results, "organization": org_results}, f, indent=2)
    logger.info(f"Disimpan: {COMPARISON_PATH}")


def load_comparison() -> dict | None:
    if os.path.exists(COMPARISON_PATH):
        with open(COMPARISON_PATH) as f:
            return json.load(f)
    return None


def load_best_model(label_col: str):
    path = os.path.join(MODELS_DIR, f"best_{label_col}.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None
