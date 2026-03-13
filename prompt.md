Buatkan Streamlit dashboard bernama "ML Auto-Tagging Dashboard" untuk mengklasifikasikan file dataset Komdigi secara otomatis.

## Stack
- Python, Streamlit, Plotly, scikit-learn, pandas
- UI: dark mode glassmorphism, font Inter dari Google Fonts

## Struktur File
- app.py          → dashboard utama Streamlit
- feature_extractor.py → ekstraksi fitur dari nama file
- trainer.py      → training & simpan model ML
- classifier.py   → prediksi & simpan hasil klasifikasi
- dataset/        → folder file yang akan di-tag
- training_data_enriched.csv → data training (kolom: title, name, group, organization, link_file)
- classification_results.csv → hasil tag

## 4 Halaman Dashboard

### 🏠 Overview
- Metric cards: total entri CSV, jumlah file di dataset/, unique group, unique org
- Bar chart distribusi Group (horizontal)
- Pie chart distribusi Organization
- Preview tabel data training (30 baris)

### ⚙️ Training & Model Comparison
- Tombol "Mulai Training" dengan progress bar real-time
- Train 5 model (Logistic Regression, Random Forest, Linear SVC, Gradient Boosting, Multinomial NB) menggunakan GridSearchCV 5-fold stratified CV
- Target label: [group](cci:1://file:///home/ifhalfaizi/dataAnalyst/ml_app/app.py:257:12-259:79) dan `organization`
- Tampilkan hasil: tabel + grouped bar chart (CV F1 vs Train Accuracy per model)
- Tandai model terbaik dengan badge ⭐
- Simpan model terbaik sebagai file .joblib

### 🏷️ Auto-Tag Dataset
- Filter file: Belum di-tag / Sudah di-tag / Semua
- Search nama file
- Pilih file → tampilkan prediksi group & organization otomatis
- Override prediksi via selectbox (opsional)
- Simpan tag ke classification_results.csv
- Download tag sebagai CSV
- Tab riwayat semua file yang sudah di-tag

### 📤 Upload & Classify
- Upload file (CSV, XLSX, PDF, JSON, ZIP, DOCX, TXT)
- File disimpan ke dataset/, diprediksi otomatis
- Tampilkan hasil prediksi semua file upload
- Tombol konfirmasi → tambah ke training data
- Download hasil klasifikasi sebagai CSV
- Tampilkan riwayat klasifikasi sebelumnya

## Fitur Tambahan
- Sidebar: logo, navigasi radio, info jumlah file & baris training, badge "Model Terlatih"
- st.cache_data untuk load CSV
- Semua operasi file bersifat idempoten (append jika file sudah ada)
