# 🏷️ MachineLearning Auto-Tagging Dashboard

Streamlit dashboard untuk mengklasifikasikan file dataset Komdigi secara otomatis menggunakan Machine Learning.

## 🚀 Fitur Utama

- **🏠 Overview** — Statistik dataset & distribusi label (group & organization)
- **⚙️ Training & Model Comparison** — Train & bandingkan 5 model ML dengan GridSearchCV
- **🏷️ Auto-Tag Dataset** — Klasifikasi file satu per satu dengan koreksi manual
- **📤 Upload & Classify** — Upload file baru, prediksi otomatis, simpan ke training data

## 🧠 Model ML

5 model ditraining dengan GridSearchCV (f1_weighted, 3-fold Stratified CV):
1. Logistic Regression
2. Linear SVC
3. Random Forest
4. Extra Trees
5. Complement Naive Bayes

**Fitur:**  Word TF-IDF + Char TF-IDF + Domain Keyword Transformer

## 📁 Struktur Project

```
dataAnalyst/
├── ml_app/
│   ├── app.py                  # Dashboard Streamlit
│   ├── feature_extractor.py    # Ekstraksi fitur dari nama file & CSV
│   ├── trainer.py              # Training & simpan model ML
│   └── classifier.py           # Prediksi & simpan hasil
├── dataset/                    # Folder file yang akan di-tag
├── training_data_enriched.csv  # Data training
├── classification_results.csv  # Hasil klasifikasi
└── prompt.md                   # Prompt pembuatan project
```

## ⚙️ Instalasi

```bash
# Clone repo
git clone https://github.com/fallss/MachineLearning-Auto-Tagging-Dashboard.git
cd MachineLearning-Auto-Tagging-Dashboard

# Buat virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install streamlit plotly pandas scikit-learn joblib

# Jalankan dashboard
streamlit run ml_app/app.py
```

## 🛠️ Tech Stack

- **Python** + **Streamlit**
- **scikit-learn** (GridSearchCV, TF-IDF, 5 classifier)
- **Plotly** (visualisasi interaktif)
- **pandas** / **numpy**
- UI: Dark mode glassmorphism, Google Fonts Inter
