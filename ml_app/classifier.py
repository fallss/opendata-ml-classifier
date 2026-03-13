"""
classifier.py
=============
Modul untuk melakukan prediksi label (group, organization) pada file-file
menggunakan model terbaik yang sudah ditraining.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime

import joblib

from feature_extractor import (
    tokenize_filename, load_vectorizer,
    get_all_dataset_files, BASE_DIR
)
from trainer import load_best_model

logger = logging.getLogger(__name__)

DATASET_FOLDER   = os.path.join(BASE_DIR, "dataset")
CSV_PATH         = os.path.join(BASE_DIR, "training_data_enriched.csv")
RESULTS_CSV      = os.path.join(BASE_DIR, "classification_results.csv")
RESULTS_COLS     = ["filename", "pred_group", "pred_organization", "classified_at", "source"]


# ================================================================
# Helper: top-K prediksi dengan confidence score
# ================================================================

def _get_topk_predictions(model, text: str, top_k: int = 3) -> list:
    """
    Ambil top-K prediksi dari model beserta confidence score (0–100%).

    Mendukung:
    - predict_proba  → LR, Random Forest, Extra Trees, Complement NB
    - decision_function → LinearSVC (dinormalisasi via softmax)

    Returns list of dict: [{"rank": int, "label": str, "confidence": float}, ...]
    """
    if model is None:
        return []

    # Ambil daftar kelas
    try:
        classes = model.classes_
    except AttributeError:
        try:
            classes = model.named_steps["clf"].classes_
        except Exception:
            return []

    # Coba predict_proba terlebih dahulu
    try:
        proba  = model.predict_proba([text])[0]
        scores = proba  # sudah [0, 1]
    except AttributeError:
        # Fallback: decision_function (LinearSVC) → softmax
        try:
            df_raw = model.decision_function([text])[0]
            exp_s  = np.exp(df_raw - np.max(df_raw))
            scores = exp_s / exp_s.sum()
        except Exception as e:
            logger.warning(f"Tidak bisa hitung confidence: {e}")
            return []

    sorted_idx = np.argsort(scores)[::-1]
    k = min(top_k, len(classes))

    results = []
    for rank, idx in enumerate(sorted_idx[:k], start=1):
        results.append({
            "rank":       rank,
            "label":      str(classes[idx]),
            "confidence": round(float(scores[idx]) * 100, 1),   # persen
        })
    return results


# ================================================================
# predict_file — prediksi tunggal (kompatibel dengan kode lama)
# ================================================================

def predict_file(filename: str) -> dict:
    """
    Prediksi group & organization untuk satu nama file.
    Kembalikan dict kompatibel dengan kode lama PLUS top-3 candidates.
    """
    model_group = load_best_model("group")
    model_org   = load_best_model("organization")
    text        = tokenize_filename(filename)

    group_topk = _get_topk_predictions(model_group, text, top_k=3)
    org_topk   = _get_topk_predictions(model_org,   text, top_k=3)

    result = {
        "filename":          filename,
        "pred_group":        group_topk[0]["label"] if group_topk else "N/A",
        "pred_organization": org_topk[0]["label"]   if org_topk   else "N/A",
        "classified_at":     datetime.now().isoformat(),
        "group_topk":        group_topk,
        "org_topk":          org_topk,
    }

    # Fallback: jika topk kosong, coba predict biasa
    if not group_topk and model_group:
        try:
            result["pred_group"] = model_group.predict([text])[0]
        except Exception as e:
            logger.warning(f"Gagal prediksi group untuk '{filename}': {e}")

    if not org_topk and model_org:
        try:
            result["pred_organization"] = model_org.predict([text])[0]
        except Exception as e:
            logger.warning(f"Gagal prediksi organization untuk '{filename}': {e}")

    return result


# ================================================================
# predict_file_topk — eksplisit top-K (bisa atur jumlah kandidat)
# ================================================================

def predict_file_topk(filename: str, top_k: int = 3) -> dict:
    """
    Prediksi top-K group & organization untuk satu file beserta confidence score.

    Returns:
    {
        "filename":          str,
        "classified_at":     str,
        "pred_group":        str  (label #1),
        "pred_organization": str  (label #1),
        "group_topk":  [{"rank":1,"label":"...","confidence":85.2}, ...],
        "org_topk":    [{"rank":1,"label":"...","confidence":72.1}, ...],
    }
    """
    model_group = load_best_model("group")
    model_org   = load_best_model("organization")
    text        = tokenize_filename(filename)

    group_topk = _get_topk_predictions(model_group, text, top_k)
    org_topk   = _get_topk_predictions(model_org,   text, top_k)

    return {
        "filename":          filename,
        "classified_at":     datetime.now().isoformat(),
        "group_topk":        group_topk,
        "org_topk":          org_topk,
        "pred_group":        group_topk[0]["label"] if group_topk else "N/A",
        "pred_organization": org_topk[0]["label"]   if org_topk   else "N/A",
    }


# ================================================================
# classify_all_files
# ================================================================

def classify_all_files(dataset_folder: str = DATASET_FOLDER) -> pd.DataFrame:
    """
    Klasifikasi semua file di folder dataset.
    Kembalikan DataFrame hasil prediksi.
    """
    files = get_all_dataset_files(dataset_folder)
    if not files:
        logger.warning("Folder dataset kosong atau tidak ditemukan.")
        return pd.DataFrame(columns=RESULTS_COLS)

    rows = []
    for fname in files:
        pred = predict_file(fname)
        pred["source"] = "auto"
        rows.append(pred)

    return pd.DataFrame(rows)


# ================================================================
# save_results
# ================================================================

def save_results(df: pd.DataFrame, results_csv: str = RESULTS_CSV):
    """Simpan / append hasil klasifikasi ke CSV."""
    # Hanya simpan kolom yang ada di RESULTS_COLS
    save_cols = [c for c in RESULTS_COLS if c in df.columns]
    df_save   = df[save_cols]

    if os.path.exists(results_csv):
        existing = pd.read_csv(results_csv)
        df_save  = df_save[~df_save["filename"].isin(existing["filename"])]
        combined = pd.concat([existing, df_save], ignore_index=True)
    else:
        combined = df_save

    combined.to_csv(results_csv, index=False)
    logger.info(f"Hasil disimpan: {results_csv} ({len(combined)} baris)")
    return combined


# ================================================================
# save_uploaded_to_training
# ================================================================

def save_uploaded_to_training(
    filename: str,
    pred_group: str,
    pred_organization: str,
    csv_path: str = CSV_PATH,
):
    """
    Tambahkan satu baris baru ke training_data_enriched.csv
    untuk file yang baru diupload + sudah diklasifikasi.
    """
    df = pd.read_csv(csv_path)
    name_slug     = os.path.splitext(filename)[0].lower()
    relative_path = os.path.join("dataset", filename)

    new_row = {
        "title":        filename,
        "name":         name_slug,
        "group":        pred_group        if pred_group        != "N/A" else "",
        "organization": pred_organization if pred_organization != "N/A" else "",
        "notes":        f"[Auto-tagged {datetime.now().strftime('%Y-%m-%d')}]",
        "link_file":    relative_path,
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"Baris baru ditambahkan ke training data: {filename}")
    return df
