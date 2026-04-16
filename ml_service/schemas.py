"""
schemas.py
==========
Pydantic schema untuk request dan response endpoint FastAPI ML Service.

Mendukung dua mode input:
1. Hanya `filename` — cocok untuk prediksi cepat dari nama file saja.
2. Multi-field (`filename`, `title`, `name`, `notes`) — menghasilkan fitur
   teks gabungan yang lebih kaya, sama seperti pipeline training.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# Request
# ──────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Payload untuk endpoint POST /predict.

    Minimal harus ada `filename`. Field lainnya bersifat opsional
    dan akan memperkaya konteks fitur teks sebelum prediksi.
    """

    filename: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Nama file yang akan diklasifikasikan (termasuk ekstensi).",
        examples=["data-frekuensi-radio-2023.xlsx"],
    )
    title: Optional[str] = Field(
        default=None,
        max_length=1024,
        description="Judul dataset (opsional, memperkaya konteks fitur).",
        examples=["Data Frekuensi Radio Tahun 2023"],
    )
    name: Optional[str] = Field(
        default=None,
        max_length=512,
        description="Slug/nama dataset (opsional).",
        examples=["data-frekuensi-radio-2023"],
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=4096,
        description="Deskripsi / catatan dataset (opsional, memperkaya konteks fitur).",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Jumlah kandidat prediksi teratas yang dikembalikan.",
    )


# ──────────────────────────────────────────────────────────────
# Response: item top-K
# ──────────────────────────────────────────────────────────────

class TopKItem(BaseModel):
    """Satu kandidat prediksi beserta confidence score."""

    rank: int = Field(..., description="Peringkat prediksi (1 = terbaik).")
    label: str = Field(..., description="Label kelas yang diprediksi.")
    confidence: float = Field(
        ...,
        description="Confidence score dalam persen (0–100).",
    )


# ──────────────────────────────────────────────────────────────
# Response: prediksi lengkap
# ──────────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    """
    Hasil prediksi dari endpoint POST /predict.
    """

    filename: str = Field(..., description="Nama file yang diprediksi.")
    pred_group: str = Field(
        ..., description="Label group terbaik (rank #1)."
    )
    pred_organization: str = Field(
        ..., description="Label organization terbaik (rank #1)."
    )
    group_topk: list[TopKItem] = Field(
        default_factory=list,
        description="Top-K kandidat prediksi group.",
    )
    org_topk: list[TopKItem] = Field(
        default_factory=list,
        description="Top-K kandidat prediksi organization.",
    )
    classified_at: datetime = Field(
        ..., description="Waktu prediksi dilakukan (ISO 8601)."
    )


# ──────────────────────────────────────────────────────────────
# Response: health check
# ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response untuk endpoint GET /health."""

    status: str = Field(default="ok")
    models_loaded: bool = Field(
        ..., description="True jika kedua model berhasil dimuat."
    )
    message: str = Field(default="ML Service berjalan normal.")
