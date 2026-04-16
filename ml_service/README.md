# ML Auto-Tagging FastAPI Service

Microservice Python/FastAPI yang mengekspos model ML Auto-Tagging melalui REST API. Portal Open Data (Laravel) menggunakan service ini untuk memprediksi label `group` dan `organization` secara otomatis berdasarkan nama file atau metadata dataset.

---

## Prasyarat

- Python ≥ 3.11
- Model sudah ditraining (file `best_group.joblib` & `best_organization.joblib` ada di `ml_app/models/`)
- (Opsional) virtualenv atau conda

---

## Instalasi

```bash
# Dari folder root ML Auto-Tagging Dashboard/
cd ml_service

# Buat virtual environment (opsional tapi direkomendasikan)
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .\.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Menjalankan Service

### Development (auto-reload)

```bash
# Dari dalam folder ml_service/
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

### Production

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --workers 2
```

### Menggunakan environment variable

```bash
ML_SERVICE_HOST=0.0.0.0 ML_SERVICE_PORT=8001 python main.py
```

---

## Environment Variables

| Variable            | Default       | Keterangan                                    |
|---------------------|---------------|-----------------------------------------------|
| `ML_SERVICE_HOST`   | `127.0.0.1`   | Host yang didengarkan Uvicorn                 |
| `ML_SERVICE_PORT`   | `8001`        | Port yang digunakan                           |
| `ML_CORS_ORIGINS`   | `*`           | Allowed CORS origins (pisahkan dengan koma)   |

---

## Endpoints

### `GET /health`
Health check — cek apakah service dan model berjalan normal.

```bash
curl http://localhost:8001/health
```

Response:
```json
{
  "status": "ok",
  "models_loaded": true,
  "message": "ML Service berjalan normal."
}
```

---

### `POST /predict`
Prediksi `group` dan `organization` untuk satu file/dataset.

**Minimal request:**
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"filename": "data-frekuensi-radio-2023.xlsx"}'
```

**Request lengkap (akurasi lebih tinggi):**
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "data-pelanggan-seluler-q3-2023.csv",
    "title": "Data Pelanggan Seluler Triwulan III 2023",
    "name": "data-pelanggan-seluler-q3-2023",
    "notes": "Dataset jumlah pelanggan seluler per operator ...",
    "top_k": 3
  }'
```

**Response:**
```json
{
  "filename": "data-frekuensi-radio-2023.xlsx",
  "pred_group": "frekuensi-radio",
  "pred_organization": "sdppi",
  "group_topk": [
    {"rank": 1, "label": "frekuensi-radio", "confidence": 87.3},
    {"rank": 2, "label": "alat-dan-perangkat", "confidence": 8.1},
    {"rank": 3, "label": "telekomunikasi", "confidence": 4.6}
  ],
  "org_topk": [
    {"rank": 1, "label": "sdppi", "confidence": 92.0},
    {"rank": 2, "label": "djppi", "confidence": 5.4},
    {"rank": 3, "label": "djaplikasi", "confidence": 2.6}
  ],
  "classified_at": "2026-04-13T08:30:00.000Z"
}
```

---

## Dokumentasi Interaktif

Setelah service berjalan, buka di browser:
- Swagger UI: http://localhost:8001/docs
- ReDoc:       http://localhost:8001/redoc

---

## Integrasi dengan Laravel

Tambahkan ke `.env` portal Open Data:
```env
ML_SERVICE_URL=http://localhost:8001
```

Lalu gunakan `App\Services\MLService` untuk memanggil endpoint dari Laravel.
