# 🏷️ OpenData ML Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://opendata-ml-classifier-8vryjxsngcksc7udfnnliw.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?logo=streamlit)

Sistem klasifikasi otomatis dan microservice berbasis Machine Learning untuk mengkategorikan dataset publik (Portal Open Data Komdigi) ke dalam kelompok tematik (*group*) dan unit kerja (*organization*) menggunakan pipeline hybrid NLP (TF-IDF & Domain Keyword Transformer).

🌐 **Live Demo:** [opendata-ml-classifier.streamlit.app](https://opendata-ml-classifier-8vryjxsngcksc7udfnnliw.streamlit.app/)

---

## 🚀 Fitur Utama

- **🏠 Overview** — Statistik dataset, distribusi label (grup & organisasi), dan visualisasi interaktif via Plotly.
- **⚙️ Training & Model Comparison** — Train & benchmarking model ML dengan GridSearchCV (3-fold Stratified CV, metric F1-weighted).
- **🏷️ Auto-Tag Dataset** — Prediksi otomatis multi-output (*group* & *organization*) untuk file dataset dengan skor keyakinan (*confidence score*).
- **📤 Upload & Classify** — Antarmuka upload file baru, inferensi instan top-K, dan integrasi penambahan ke basis data training.
- **⚡ FastAPI Microservice** — REST API berkinerja tinggi (`/predict` & `/health`) yang siap dikonsumsi langsung oleh backend portal.

---

## 🧠 Model ML & Feature Engineering

Model dilatih menggunakan GridSearchCV untuk mengoptimalkan skor `f1_weighted`:
1. **Logistic Regression**
2. **Linear SVC**
3. **Random Forest**
4. **Extra Trees**
5. **Complement Naive Bayes**
6. **Ridge & SGD Classifiers**

**Pipeline Fitur Hybrid:**
- Word-level TF-IDF (1-2 gram)
- Character-level TF-IDF (3-5 gram)
- Domain Keyword Transformer (pencocokan istilah khusus domain instansi)

---

## 📁 Struktur Project

```text
opendata-ml-classifier/
├── ml_app/                     # Dashboard Streamlit & Engine Klasifikasi
│   ├── app.py                  # Entrypoint Streamlit UI
│   ├── feature_extractor.py    # Ekstraksi fitur teks & transformer domain
│   ├── trainer.py              # Pipeline training & GridSearchCV
│   ├── classifier.py           # Engine prediksi & inferensi top-K
│   ├── models/                 # Model terlatih (*.joblib)
│   └── requirements.txt        # Spesifikasi dependensi Streamlit
├── ml_service/                 # FastAPI REST Microservice
│   ├── main.py                 # Endpoint API FastAPI (/health, /predict)
│   ├── model_loader.py         # Singleton model loader
│   └── schemas.py              # Skema validasi Pydantic
├── dataset/                    # Direktori sampel dataset
├── training_data_enriched.csv  # Dataset berlabel untuk pelatihan
├── requirements.txt            # Dependensi root aplikasi
├── Dockerfile                  # Containerization image
└── docker-compose.yml          # Orkestrasi Docker multi-service
```

---

## ⚙️ Instalasi & Menjalankan Lokal

```bash
# 1. Clone repository
git clone https://github.com/fallss/opendata-ml-classifier.git
cd opendata-ml-classifier

# 2. Buat & aktifkan virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Jalankan Dashboard Streamlit
streamlit run ml_app/app.py
```

### Menjalankan FastAPI Microservice:
```bash
cd ml_service
uvicorn main:app --reload --port 8001
```
Buka dokumentasi interaktif API di `http://localhost:8001/docs`.

---

## 🛠️ Tech Stack

- **Machine Learning:** `scikit-learn`, `joblib`, `numpy`, `pandas`
- **Dashboard UI:** `Streamlit`, `Plotly`, Custom Glassmorphism CSS
- **API Microservice:** `FastAPI`, `Uvicorn`, `Pydantic`
- **Deployment:** Streamlit Community Cloud, Docker, Docker Compose
