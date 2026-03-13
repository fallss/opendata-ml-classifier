"""
main.py
=======
Entry point untuk menjalankan seluruh proses:
1. Scrape daftar dataset dari portal Komdigi (sudah termasuk download URLs)
2. Download semua file dataset ke folder 'dataset/'
3. Cocokkan file lokal dengan training_data.csv dan tambah kolom link_file
4. Simpan training_data_enriched.csv
"""

import os
import logging
import requests

from config import DATASET_FOLDER, OUTPUT_CSV
from scraper import get_dataset_list
from downloader import download_all
from enrich_csv import enrich_csv

# Setup logging global
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run():
    logger.info("=" * 60)
    logger.info(" KOMDIGI DATASET CRAWLER - MULAI")
    logger.info("=" * 60)

    # Buat folder dataset jika belum ada
    os.makedirs(DATASET_FOLDER, exist_ok=True)

    # Gunakan satu Session untuk semua request
    session = requests.Session()

    # ----------------------------------------------------------
    # STEP 1: Scrape daftar dataset + URL download sekaligus
    # (download_urls sudah tersedia dari JSON halaman listing)
    # ----------------------------------------------------------
    logger.info("\n[STEP 1] Scraping daftar dataset dari portal...")
    datasets = get_dataset_list(session)

    if not datasets:
        logger.error("Tidak ada dataset yang ditemukan! Periksa koneksi atau struktur halaman.")
        return

    total_with_files = sum(1 for d in datasets if d.get("download_urls"))
    logger.info(f"Total dataset: {len(datasets)} | Dataset dengan file: {total_with_files}")

    # ----------------------------------------------------------
    # STEP 2: Download semua file ke folder dataset/
    # ----------------------------------------------------------
    logger.info(f"\n[STEP 2] Mendownload file ke folder: {DATASET_FOLDER}")
    results = download_all(datasets, session)

    total_downloaded = sum(len(r["local_files"]) for r in results)
    logger.info(f"Total file berhasil didownload: {total_downloaded}")

    # ----------------------------------------------------------
    # STEP 3: Enrich training_data.csv dengan kolom link_file
    # ----------------------------------------------------------
    logger.info("\n[STEP 3] Memperkaya training_data.csv dengan kolom link_file...")
    df = enrich_csv()

    # ----------------------------------------------------------
    # SELESAI
    # ----------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info(" SELESAI!")
    logger.info(f" File output   : {OUTPUT_CSV}")
    logger.info(f" Folder dataset: {DATASET_FOLDER}")
    logger.info(f" Total file    : {total_downloaded}")
    logger.info(f" Total baris CSV: {len(df)}")
    filled = (df["link_file"] != "").sum()
    logger.info(f" Baris dengan link_file: {filled}/{len(df)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
