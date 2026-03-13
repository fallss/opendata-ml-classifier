import os

# ============================================================
# Konfigurasi Scraper
# ============================================================

# URL halaman listing dataset Komdigi
BASE_URL = "https://dev-opendata.komdigi.go.id"
DATASET_LIST_URL = f"{BASE_URL}/opendata/dataset"

# Header HTTP agar tidak terdeteksi sebagai bot
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
}

# Timeout per request (detik)
REQUEST_TIMEOUT = 20

# Delay antar request (detik) agar tidak membebani server
REQUEST_DELAY = 1.0

# ============================================================
# Konfigurasi File & Folder
# ============================================================

# Root project — satu level di atas folder crawler/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Folder penyimpanan dataset yang di-download
DATASET_FOLDER = os.path.join(BASE_DIR, "dataset")

# File training data input
TRAINING_CSV = os.path.join(BASE_DIR, "training_data.csv")

# File output CSV yang sudah diperkaya
OUTPUT_CSV = os.path.join(BASE_DIR, "training_data_enriched.csv")

# ============================================================
# Konfigurasi Matching
# ============================================================

# Ambang batas kemiripan nama untuk fuzzy matching (0.0 - 1.0)
FUZZY_MATCH_THRESHOLD = 0.6
