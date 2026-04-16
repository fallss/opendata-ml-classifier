"""
model_loader.py
===============
Singleton loader untuk model joblib ML Auto-Tagging.

Model dimuat satu kali saat modul pertama kali di-import, kemudian
di-cache ke variabel modul-level. Pendekatan ini lebih ringan daripada
threading.Lock dan aman untuk lingkungan single-process (Uvicorn default).

Logika tokenisasi & prediksi direproduksi di sini secara mandiri
(tidak import langsung dari ml_app) agar FastAPI service dapat
berjalan sebagai layanan terpisah tanpa ketergantungan path.
"""

import logging
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Path resolving
# ──────────────────────────────────────────────────────────────

# ml_service/ berada di ML Auto-Tagging Dashboard/ml_service/
# Model berada di    ML Auto-Tagging Dashboard/ml_app/models/
_SERVICE_DIR = Path(__file__).resolve().parent          # .../ml_service/
_DASHBOARD_DIR = _SERVICE_DIR.parent                    # .../ML Auto-Tagging Dashboard/
_ML_APP_DIR = _DASHBOARD_DIR / "ml_app"
_MODELS_DIR = _ML_APP_DIR / "models"

# Tambahkan ml_app ke sys.path agar dapat import DomainKeywordTransformer
# (class kustom yang dibutuhkan joblib untuk de-serialisasi pipeline)
if str(_ML_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_APP_DIR))


# ──────────────────────────────────────────────────────────────
# Tokenisasi nama file (mirror dari feature_extractor.py)
# ──────────────────────────────────────────────────────────────

def tokenize_filename(filename: str) -> str:
    """
    Ubah nama file menjadi string token yang bisa di-vectorize.

    Contoh:
        'data-frekuensi-radio-2019.xlsx' → 'data frekuensi radio 2019 xlsx'
    """
    stem = os.path.splitext(filename)[0]
    tokens = re.split(r"[-_\s\.]+", stem.lower())
    return " ".join(t for t in tokens if t)


def build_text_feature(
    filename: str,
    title: Optional[str] = None,
    name: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """
    Bangun string fitur teks gabungan dari metadata dataset.

    Mengikuti logika `build_text_feature` di feature_extractor.py:
    - name: tokenized
    - title: raw lowercase (duplikat 2× untuk bobot)
    - notes: bersihkan markdown, lowercase
    - filename: tokenized
    """
    parts = []

    if name:
        parts.append(tokenize_filename(str(name)))

    if title:
        title_text = str(title).lower()
        parts.append(title_text)
        parts.append(title_text)  # duplikasi untuk bobot lebih tinggi

    if notes and str(notes).strip():
        notes_clean = str(notes).lower()
        notes_clean = re.sub(r"\*+", " ", notes_clean)
        notes_clean = re.sub(r"\s+", " ", notes_clean).strip()
        parts.append(notes_clean)

    # Nama file selalu disertakan
    parts.append(tokenize_filename(filename))

    return " ".join(parts)


# ──────────────────────────────────────────────────────────────
# Singleton model holder
# ──────────────────────────────────────────────────────────────

class _ModelHolder:
    """Menyimpan model yang sudah dimuat di memori."""

    def __init__(self) -> None:
        self.group: Optional[object] = None
        self.organization: Optional[object] = None
        self._loaded: bool = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load kedua model dari disk (dipanggil sekali saat startup)."""
        group_path = _MODELS_DIR / "best_group.joblib"
        org_path = _MODELS_DIR / "best_organization.joblib"

        if not group_path.exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan: {group_path}\n"
                "Pastikan training sudah dilakukan terlebih dahulu."
            )
        if not org_path.exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan: {org_path}\n"
                "Pastikan training sudah dilakukan terlebih dahulu."
            )

        logger.info("Memuat model best_group.joblib …")
        self.group = joblib.load(group_path)
        logger.info("Memuat model best_organization.joblib …")
        self.organization = joblib.load(org_path)
        self._loaded = True
        logger.info("✓ Kedua model berhasil dimuat.")


# Instance tunggal — inilah singleton-nya
_holder = _ModelHolder()


def load_models() -> None:
    """
    Panggil saat startup FastAPI (lifespan event).
    Aman dipanggil berkali-kali; model hanya dimuat sekali.
    """
    if not _holder.loaded:
        _holder.load()


def get_models() -> _ModelHolder:
    """Kembalikan holder yang berisi model siap pakai."""
    if not _holder.loaded:
        raise RuntimeError(
            "Model belum dimuat. Pastikan load_models() dipanggil saat startup."
        )
    return _holder


# ──────────────────────────────────────────────────────────────
# Top-K prediksi
# ──────────────────────────────────────────────────────────────

def _get_topk_predictions(model: object, text: str, top_k: int = 3) -> list[dict]:
    """
    Ambil top-K prediksi dari pipeline model beserta confidence score.

    Mendukung:
    - predict_proba  → LR, Random Forest, Extra Trees, Complement NB, SGD(log)
    - decision_function → LinearSVC, Ridge (dinormalisasi via softmax)

    Returns:
        list of dict: [{"rank": int, "label": str, "confidence": float}, ...]
    """
    if model is None:
        return []

    # Ambil daftar kelas dari pipeline
    try:
        classes = model.classes_  # type: ignore[union-attr]
    except AttributeError:
        try:
            classes = model.named_steps["clf"].classes_  # type: ignore[union-attr]
        except Exception:
            logger.warning("Tidak bisa mengambil classes dari model.")
            return []

    # Coba predict_proba terlebih dahulu
    try:
        proba = model.predict_proba([text])[0]  # type: ignore[union-attr]
        scores = proba
    except AttributeError:
        # Fallback: decision_function (LinearSVC, Ridge) → softmax normalisasi
        try:
            df_raw = model.decision_function([text])[0]  # type: ignore[union-attr]
            exp_s = np.exp(df_raw - np.max(df_raw))
            scores = exp_s / exp_s.sum()
        except Exception as exc:
            logger.warning("Tidak bisa menghitung confidence score: %s", exc)
            return []

    sorted_idx = np.argsort(scores)[::-1]
    k = min(top_k, len(classes))

    return [
        {
            "rank": rank,
            "label": str(classes[idx]),
            "confidence": round(float(scores[idx]) * 100, 1),
        }
        for rank, idx in enumerate(sorted_idx[:k], start=1)
    ]


# ──────────────────────────────────────────────────────────────
# Fungsi prediksi utama
# ──────────────────────────────────────────────────────────────

def predict(
    filename: str,
    title: Optional[str] = None,
    name: Optional[str] = None,
    notes: Optional[str] = None,
    top_k: int = 3,
) -> dict:
    """
    Lakukan prediksi group & organization untuk satu file/dataset.

    Args:
        filename: Nama file (wajib).
        title:    Judul dataset (opsional).
        name:     Slug dataset (opsional).
        notes:    Deskripsi dataset (opsional).
        top_k:    Jumlah kandidat teratas.

    Returns:
        dict dengan pred_group, pred_organization, group_topk, org_topk.
    """
    holder = get_models()
    text = build_text_feature(filename, title=title, name=name, notes=notes)

    group_topk = _get_topk_predictions(holder.group, text, top_k)
    org_topk = _get_topk_predictions(holder.organization, text, top_k)

    result: dict = {
        "pred_group": group_topk[0]["label"] if group_topk else "N/A",
        "pred_organization": org_topk[0]["label"] if org_topk else "N/A",
        "group_topk": group_topk,
        "org_topk": org_topk,
    }

    # Fallback predict biasa jika topk gagal
    if not group_topk and holder.group:
        try:
            result["pred_group"] = holder.group.predict([text])[0]  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("Gagal prediksi group (fallback): %s", exc)

    if not org_topk and holder.organization:
        try:
            result["pred_organization"] = holder.organization.predict([text])[0]  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("Gagal prediksi organization (fallback): %s", exc)

    return result
