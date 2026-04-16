"""
main.py
=======
FastAPI ML Service — endpoint prediksi auto-tagging dataset.

Menjalankan:
    uvicorn main:app --reload --host 0.0.0.0 --port 8001

Atau menggunakan environment variable:
    ML_SERVICE_HOST=0.0.0.0 ML_SERVICE_PORT=8001 python main.py
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from model_loader import load_models, predict as ml_predict
from schemas import HealthResponse, PredictRequest, PredictResponse, TopKItem

# ──────────────────────────────────────────────────────────────
# Konfigurasi logging
# ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("ml_service")


# ──────────────────────────────────────────────────────────────
# Lifespan: load model sekali saat startup
# ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Startup: muat model joblib ke memori.
    Shutdown: tidak ada resource yang perlu dibebaskan secara eksplisit.
    """
    logger.info("🚀 ML Service sedang diinisialisasi …")
    try:
        load_models()
        logger.info("✅ Model berhasil dimuat. Service siap menerima request.")
    except FileNotFoundError as exc:
        logger.critical("❌ Gagal memuat model: %s", exc)
        raise RuntimeError(str(exc)) from exc
    yield
    logger.info("🛑 ML Service dihentikan.")


# ──────────────────────────────────────────────────────────────
# Inisialisasi FastAPI
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="ML Auto-Tagging Service",
    description=(
        "Microservice prediksi label **group** dan **organization** "
        "untuk dataset Portal Open Data Komdigi menggunakan model ML terlatih."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — izinkan semua origin di development; batasi di production
_ALLOWED_ORIGINS = os.getenv("ML_CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Monitoring"],
)
async def health_check() -> HealthResponse:
    """
    Periksa status service dan apakah model sudah dimuat.
    Digunakan oleh Laravel atau orchestrator untuk memverifikasi
    ketersediaan service sebelum mengirim request prediksi.
    """
    try:
        from model_loader import get_models
        holder = get_models()
        models_loaded = holder.loaded
    except RuntimeError:
        models_loaded = False

    return HealthResponse(
        status="ok" if models_loaded else "degraded",
        models_loaded=models_loaded,
        message="ML Service berjalan normal." if models_loaded
                else "Model belum dimuat — service mungkin baru saja restart.",
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Prediksi group & organization dataset",
    tags=["Prediksi"],
)
async def predict_endpoint(payload: PredictRequest) -> PredictResponse:
    """
    Prediksi label **group** dan **organization** untuk satu file/dataset.

    ### Minimal Request
    ```json
    {"filename": "data-frekuensi-radio-2023.xlsx"}
    ```

    ### Request Lengkap (akurasi lebih tinggi)
    ```json
    {
        "filename": "data-frekuensi-radio-2023.xlsx",
        "title": "Data Frekuensi Radio Tahun 2023",
        "name": "data-frekuensi-radio-2023",
        "notes": "Dataset berisi data alokasi frekuensi radio ...",
        "top_k": 3
    }
    ```

    ### Response
    Mengembalikan prediksi terbaik (`pred_group`, `pred_organization`)
    beserta daftar top-K kandidat dan confidence score masing-masing.
    """
    logger.info(
        "Prediksi diminta | filename='%s' | top_k=%d",
        payload.filename,
        payload.top_k,
    )

    try:
        result = ml_predict(
            filename=payload.filename,
            title=payload.title,
            name=payload.name,
            notes=payload.notes,
            top_k=payload.top_k,
        )
    except RuntimeError as exc:
        logger.error("Model belum dimuat: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model belum dimuat. Coba beberapa saat lagi.",
        ) from exc
    except Exception as exc:
        logger.exception("Kesalahan tidak terduga saat prediksi: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Kesalahan internal server: {exc}",
        ) from exc

    # Konversi list[dict] → list[TopKItem]
    group_topk = [TopKItem(**item) for item in result.get("group_topk", [])]
    org_topk = [TopKItem(**item) for item in result.get("org_topk", [])]

    response = PredictResponse(
        filename=payload.filename,
        pred_group=result.get("pred_group", "N/A"),
        pred_organization=result.get("pred_organization", "N/A"),
        group_topk=group_topk,
        org_topk=org_topk,
        classified_at=datetime.now(tz=timezone.utc),
    )

    logger.info(
        "Prediksi selesai | group='%s' | org='%s'",
        response.pred_group,
        response.pred_organization,
    )
    return response


# ──────────────────────────────────────────────────────────────
# Entry point (opsional — untuk `python main.py`)
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("ML_SERVICE_HOST", "127.0.0.1")
    port = int(os.getenv("ML_SERVICE_PORT", "8001"))

    logger.info("Menjalankan Uvicorn pada %s:%d", host, port)
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
