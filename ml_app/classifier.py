"""
classifier.py
=============
Modul untuk melakukan prediksi label (group, organization) pada file-file
menggunakan model terbaik yang sudah ditraining.
"""

import os
import logging
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


def predict_file(filename: str) -> dict:
    """
    Prediksi group & organization untuk satu nama file.
    Kembalikan dict {"pred_group": ..., "pred_organization": ...}.
    """
    model_group = load_best_model("group")
    model_org   = load_best_model("organization")

    text = tokenize_filename(filename)

    result = {
        "filename":          filename,
        "pred_group":        "N/A",
        "pred_organization": "N/A",
        "classified_at":     datetime.now().isoformat(),
    }

    if model_group:
        try:
            result["pred_group"] = model_group.predict([text])[0]
        except Exception as e:
            logger.warning(f"Gagal prediksi group untuk '{filename}': {e}")

    if model_org:
        try:
            result["pred_organization"] = model_org.predict([text])[0]
        except Exception as e:
            logger.warning(f"Gagal prediksi organization untuk '{filename}': {e}")

    return result


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


def save_results(df: pd.DataFrame, results_csv: str = RESULTS_CSV):
    """Simpan / append hasil klasifikasi ke CSV."""
    if os.path.exists(results_csv):
        existing = pd.read_csv(results_csv)
        # Hindari duplikasi berdasarkan filename
        df = df[~df["filename"].isin(existing["filename"])]
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df
    combined.to_csv(results_csv, index=False)
    logger.info(f"Hasil disimpan: {results_csv} ({len(combined)} baris)")
    return combined


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
    name_slug = os.path.splitext(filename)[0].lower()
    relative_path = os.path.join("dataset", filename)

    new_row = {
        "title":        filename,
        "name":         name_slug,
        "group":        pred_group if pred_group != "N/A" else "",
        "organization": pred_organization if pred_organization != "N/A" else "",
        "notes":        f"[Auto-tagged {datetime.now().strftime('%Y-%m-%d')}]",
        "link_file":    relative_path,
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"Baris baru ditambahkan ke training data: {filename}")
    return df
