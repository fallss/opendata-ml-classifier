"""
enrich_csv.py
=============
Modul untuk mencocokkan file-file yang sudah didownload
dengan baris-baris di training_data.csv berdasarkan title/name,
lalu menambahkan kolom 'link_file' dan menyimpan CSV baru.
"""

import os
import re
import logging
from difflib import SequenceMatcher, get_close_matches

import pandas as pd

from config import TRAINING_CSV, OUTPUT_CSV, DATASET_FOLDER, FUZZY_MATCH_THRESHOLD

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """
    Normalisasi teks untuk perbandingan:
    - Lowercase
    - Ganti karakter non-alfanumerik dengan spasi
    - Hilangkan spasi berlebih
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _slug_to_words(slug: str) -> str:
    """Ubah slug (dash-separated) menjadi kata-kata biasa."""
    return slug.replace("-", " ").replace("_", " ")


def _similarity(a: str, b: str) -> float:
    """Hitung rasio kemiripan antara dua string."""
    return SequenceMatcher(None, a, b).ratio()


def build_file_index(dataset_folder: str) -> dict[str, str]:
    """
    Buat indeks dari nama file (tanpa ekstensi, dinormalisasi) → path absolut.
    """
    index = {}
    if not os.path.isdir(dataset_folder):
        logger.warning(f"Folder dataset tidak ditemukan: {dataset_folder}")
        return index

    for fname in os.listdir(dataset_folder):
        fpath = os.path.join(dataset_folder, fname)
        if os.path.isfile(fpath):
            stem = os.path.splitext(fname)[0]
            key = _normalize(_slug_to_words(stem))
            index[key] = fpath

    logger.info(f"Total file di folder dataset: {len(index)}")
    return index


def match_row_to_file(title: str, name: str, file_index: dict[str, str]) -> str:
    """
    Cocokkan satu baris CSV ke file lokal.
    Prioritas:
      1. Exact match pada 'name' (slug)
      2. Exact match pada normalized 'title'
      3. Fuzzy match terbaik di antara semua key file
    Kembalikan path relatif atau "" jika tidak cocok.
    """
    norm_name = _normalize(_slug_to_words(name))
    norm_title = _normalize(title)

    # 1. Exact match slug
    if norm_name in file_index:
        return os.path.relpath(file_index[norm_name], start=os.path.dirname(TRAINING_CSV))

    # 2. Exact match title
    if norm_title in file_index:
        return os.path.relpath(file_index[norm_title], start=os.path.dirname(TRAINING_CSV))

    # 3. Fuzzy match — cari kandidat terbaik
    all_keys = list(file_index.keys())
    if not all_keys:
        return ""

    # Coba cocokkan nama (slug) dulu
    best_key = None
    best_score = 0.0

    for key in all_keys:
        score_name = _similarity(norm_name, key)
        score_title = _similarity(norm_title, key)
        score = max(score_name, score_title)
        if score > best_score:
            best_score = score
            best_key = key

    if best_score >= FUZZY_MATCH_THRESHOLD and best_key:
        rel_path = os.path.relpath(file_index[best_key], start=os.path.dirname(TRAINING_CSV))
        logger.debug(f"  Fuzzy match [{best_score:.2f}]: '{name}' → '{best_key}'")
        return rel_path

    return ""


def enrich_csv() -> pd.DataFrame:
    """
    Baca training_data.csv, tambahkan kolom 'link_file',
    simpan ke training_data_enriched.csv, dan kembalikan DataFrame hasilnya.
    """
    logger.info(f"Membaca: {TRAINING_CSV}")
    df = pd.read_csv(TRAINING_CSV)

    # Pastikan kolom yang dibutuhkan ada
    for col in ["title", "name"]:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di training_data.csv!")

    # Bangun indeks file dari folder dataset
    file_index = build_file_index(DATASET_FOLDER)

    # Cocokkan setiap baris
    link_files = []
    matched = 0
    for _, row in df.iterrows():
        link = match_row_to_file(
            title=str(row.get("title", "")),
            name=str(row.get("name", "")),
            file_index=file_index,
        )
        link_files.append(link)
        if link:
            matched += 1

    df["link_file"] = link_files

    logger.info(f"Total baris: {len(df)} | Berhasil dicocokkan: {matched} | Tidak cocok: {len(df) - matched}")

    # Simpan CSV baru
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"CSV baru disimpan di: {OUTPUT_CSV}")

    return df
