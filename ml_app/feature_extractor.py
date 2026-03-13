"""
feature_extractor.py
====================
Ekstraksi fitur dari nama file dan metadata CSV.

Dua pendekatan fitur:
1. TF-IDF (word + char n-gram) atas teks gabungan
2. DomainKeywordTransformer: pencocokan kata kunci domain → sinyal langsung
   per organisasi. Ini critical untuk menembus ceiling TF-IDF yang terhalang
   oleh data lorem ipsum / dummy.
"""

import os
import re
import logging
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

logger = logging.getLogger(__name__)

# Path default
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH   = os.path.join(BASE_DIR, "training_data_enriched.csv")
MODELS_DIR = os.path.join(BASE_DIR, "ml_app", "models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")

# ================================================================
# Domain keyword map — derived from actual CSV data analysis
# Each list = terms strongly associated with that org's datasets
# ================================================================
ORG_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "sdppi": [
        "frekuensi", "spektrum", "radio", "alat", "perangkat",
        "sertifikat", "sertifikasi", "kalibrasi", "monitoring", "smfr",
        "satelit", "filing", "stasiun", "ikrap", "skor", "reor",
        "imei", "uji", "pengujian", "gangguan", "penertiban",
        "notifikasi", "itu", "hak labuh", "pengguna spektrum",
        "band frekuensi", "lhu", "sp2",
    ],
    "djppi": [
        "penyiaran", "pos", "seluler", "bandwidth", "pelanggan",
        "teledensitas", "enodeb", "isp", "sinyal", "kapasitas",
        "lembaga penyiaran", "lpb", "lps", "lpk", "tv", "radio",
        "pertumbuhan izin", "penyelenggaraan", "kiriman", "produksi",
        "jasa penyiaran", "jaringan bergerak", "jaringan tetap",
        "persebaran", "cakupan sinyal",
    ],
    "djaplikasi": [
        "aplikasi", "e-commerce", "domain", "sistem elektronik",
        "pse", "keamanan informasi", "insiden", "startup",
        "umkm", "pegi", "e-government", "go online",
        "digital", "informatika", "kominfo",
    ],
    "sekjen": [
        "pedoman", "keputusan", "anggaran", "kepuasan",
        "cio", "walidata", "nelayan", "administrasi", "reviu",
        "kebijakan", "informasi wajib",
    ],
    "balitbang": [
        "penggunaan internet", "internet", "sebaran pengguna",
        "pengalaman insiden", "paket langganan",
    ],
    "bati": [
        "aksesibilitas", "akses broadband", "desa",
    ],
    "djikp": [
        "komunikasi publik", "rekapitulasi penggunaan frekuensi",
        "daftar filing satelit baru",
    ],
    "inspektorat": [
        "inspektorat", "sensus", "audit",
    ],
    "menteri": [
        "keputusan menteri", "peraturan menteri", "kebijakan menteri",
    ],
}

GROUP_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "frekuensi-radio": [
        "frekuensi", "spektrum", "radio", "satelit", "filing",
        "stasiun", "monitoring", "gangguan", "penertiban", "skor", "reor",
        "ikrap", "band", "hak labuh", "itu",
    ],
    "alat-dan-perangkat": [
        "alat", "perangkat", "sertifikat", "sertifikasi",
        "kalibrasi", "uji", "pengujian", "imei", "lhu", "sp2",
    ],
    "telekomunikasi": [
        "telekomunikasi", "seluler", "pelanggan", "bandwidth", "isp",
        "enodeb", "jaringan", "teledensitas", "kapasitas", "pertumbuhan izin",
    ],
    "penyiaran": [
        "penyiaran", "tv", "lembaga penyiaran", "lpb", "lps", "lpk",
        "jasa penyiaran",
    ],
    "pos": ["pos", "kiriman", "produksi", "surat pos", "paket pos"],
    "internet": ["internet", "penggunaan internet", "sebaran pengguna", "paket langganan"],
    "e-commerce": ["e-commerce", "toko online", "umkm", "startup", "capaian"],
    "e-government": ["e-government", "pegi"],
    "keamanan-informasi": ["keamanan", "insiden", "serangan", "siber"],
    "sistem-elektronik": ["sistem elektronik", "pse"],
    "domain": ["domain", "nama domain"],
}


class DomainKeywordTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer: input list of text strings →
    output dense matrix of keyword-match counts per domain category.

    Ini menghasilkan 'sinyal keras' yang langsung menunjukkan
    katogori organisasi/group, berbeda dari TF-IDF yang statistical.
    """

    def __init__(self, keyword_map: dict | None = None, label_col: str = "organization"):
        self.keyword_map = keyword_map
        self.label_col = label_col

    def fit(self, X, y=None):
        if self.keyword_map is None:
            self.keyword_map_ = (
                ORG_DOMAIN_KEYWORDS
                if self.label_col == "organization"
                else GROUP_DOMAIN_KEYWORDS
            )
        else:
            self.keyword_map_ = self.keyword_map
        self.feature_names_out_ = list(self.keyword_map_.keys())
        return self

    def transform(self, X):
        results = np.zeros((len(X), len(self.feature_names_out_)))
        kmap = self.keyword_map_
        keys = self.feature_names_out_
        for i, text in enumerate(X):
            text_lower = text.lower()
            for j, key in enumerate(keys):
                score = sum(1 for kw in kmap[key] if kw in text_lower)
                results[i, j] = score
        return results

    def get_feature_names_out(self):
        return np.array(self.feature_names_out_)


logger = logging.getLogger(__name__)

# Path default
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH   = os.path.join(BASE_DIR, "training_data_enriched.csv")
MODELS_DIR = os.path.join(BASE_DIR, "ml_app", "models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")


def tokenize_filename(filename: str) -> str:
    """
    Ubah nama file menjadi string kata-kata yang dapat di-TF-IDF-kan.
    Contoh: 'data-stasiun-radio-2019.xlsx' → 'data stasiun radio 2019 xlsx'
    """
    stem = os.path.splitext(filename)[0]  # hapus ekstensi
    # Pisah berdasarkan -, _, titik, atau transisi huruf-angka
    tokens = re.split(r"[-_\s\.]+", stem.lower())
    # Hapus token kosong
    tokens = [t for t in tokens if t]
    return " ".join(tokens)


def build_text_feature(row: pd.Series) -> str:
    """
    Gabungkan beberapa kolom teks dari satu baris CSV menjadi satu string fitur.
    """
    parts = []
    # Nama/slug dataset
    if pd.notna(row.get("name", "")) and str(row.get("name", "")):
        parts.append(tokenize_filename(str(row["name"])))
    # Judul (bobot tinggi — ulangi 2x)
    if pd.notna(row.get("title", "")) and str(row.get("title", "")):
        title_text = str(row["title"]).lower()
        parts.append(title_text)
        parts.append(title_text)  # duplikasi untuk bobot lebih tinggi
    # Notes (deskripsi paling kaya informasi)
    if pd.notna(row.get("notes", "")) and str(row.get("notes", "")).strip():
        notes_clean = str(row["notes"]).lower()
        # Bersihkan markdown/simbol
        notes_clean = re.sub(r"\*+", " ", notes_clean)
        notes_clean = re.sub(r"\s+", " ", notes_clean).strip()
        parts.append(notes_clean)
    # Nama file lokal (dari link_file)
    if pd.notna(row.get("link_file", "")) and str(row.get("link_file", "")):
        fname = os.path.basename(str(row["link_file"]))
        parts.append(tokenize_filename(fname))
    return " ".join(parts)


def load_training_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """
    Baca CSV, bersihkan dan kembalikan DataFrame siap pakai.
    Baris tanpa label group DAN organization di-drop.
    """
    df = pd.read_csv(csv_path)

    # Normalisasi kolom label
    for col in ["group", "organization"]:
        if col in df.columns:
            df[col] = df[col].fillna("").str.strip()

    # Buat kolom fitur teks gabungan
    df["_text_feature"] = df.apply(build_text_feature, axis=1)

    return df


def get_training_samples(df: pd.DataFrame, label_col: str):
    """
    Ambil baris yang memiliki label (label_col tidak kosong).
    Kembalikan (texts, labels).
    """
    mask = df[label_col].str.len() > 0
    sub = df[mask].copy()
    return sub["_text_feature"].tolist(), sub[label_col].tolist()


def fit_vectorizer(texts: list[str], save: bool = True) -> TfidfVectorizer:
    """
    Fit TF-IDF vectorizer terhadap semua teks training.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000,
        sublinear_tf=True,
    )
    vec.fit(texts)
    if save:
        joblib.dump(vec, VECTORIZER_PATH)
        logger.info(f"Vectorizer disimpan: {VECTORIZER_PATH}")
    return vec


def load_vectorizer() -> TfidfVectorizer | None:
    """Load vectorizer yang sudah di-fit."""
    if os.path.exists(VECTORIZER_PATH):
        return joblib.load(VECTORIZER_PATH)
    return None


def feature_from_filename(filename: str, vectorizer: TfidfVectorizer):
    """Transform satu nama file menjadi feature matrix."""
    text = tokenize_filename(filename)
    return vectorizer.transform([text])


def get_all_dataset_files(dataset_folder: str) -> list[str]:
    """Kembalikan list nama file dari folder dataset."""
    if not os.path.isdir(dataset_folder):
        return []
    return [
        f for f in os.listdir(dataset_folder)
        if os.path.isfile(os.path.join(dataset_folder, f))
        and not f.startswith(".")
    ]
