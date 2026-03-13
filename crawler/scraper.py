"""
scraper.py
==========
Modul untuk melakukan scraping daftar dataset dan link download
dari portal Open Data Komdigi.

CATATAN TEKNIS:
  Website ini menggunakan Inertia.js (Laravel + Vue). Semua data dataset
  TIDAK dirender sebagai HTML biasa, melainkan di-embed sebagai JSON
  di dalam atribut `data-page` pada elemen <div id="app">.

  Oleh karena itu pendekatan yang digunakan adalah:
  1. GET halaman listing → parse HTML → ambil atribut data-page
  2. Parse JSON dari data-page → ekstrak datasetsArray
  3. Setiap dataset sudah mengandung list resources (termasuk URL download)
     sehingga TIDAK perlu mengunjungi halaman detail satu per satu.
"""

import html
import json
import time
import logging
import requests
from bs4 import BeautifulSoup

from config import BASE_URL, DATASET_LIST_URL, HEADERS, REQUEST_TIMEOUT, REQUEST_DELAY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get_page_json(url: str, session: requests.Session) -> dict | None:
    """
    GET halaman URL, parse HTML, ambil atribut data-page dari div#app,
    decode HTML entities, kembalikan sebagai dict Python.
    """
    try:
        resp = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Gagal mengakses {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    app_div = soup.find("div", {"id": "app"})
    if not app_div:
        logger.error("Elemen <div id='app'> tidak ditemukan. Struktur halaman mungkin berubah.")
        return None

    raw = app_div.get("data-page", "")
    if not raw:
        logger.error("Atribut data-page kosong atau tidak ditemukan.")
        return None

    # HTML entities di-encode (&quot; → " dst.), perlu di-unescape dulu
    decoded = html.unescape(raw)
    try:
        return json.loads(decoded)
    except json.JSONDecodeError as e:
        logger.error(f"Gagal parse JSON dari data-page: {e}")
        return None


def get_dataset_list(session: requests.Session) -> list[dict]:
    """
    Crawl semua halaman listing dataset Komdigi.
    Kembalikan list dict dengan format:
    [
        {
            "title": "...",
            "slug": "...",
            "detail_url": "...",
            "download_urls": ["url1", "url2", ...]
        }
    ]

    Data resource/download URL sudah tersedia di JSON halaman listing,
    tidak perlu visit halaman detail.
    """
    # Ambil halaman pertama untuk deteksi total halaman
    logger.info(f"Mengambil halaman 1: {DATASET_LIST_URL}")
    page_data = _get_page_json(DATASET_LIST_URL, session)
    if not page_data:
        return []

    props = page_data.get("props", {})
    datasets_array = props.get("datasetsArray", {}).get("results", {})
    last_page = datasets_array.get("last_page", 1)
    logger.info(f"Total halaman terdeteksi: {last_page}")

    all_datasets = []

    def _extract_from_page(data: dict):
        """Ekstrak list dataset dari objek datasetsArray.results."""
        results = data.get("props", {}).get("datasetsArray", {}).get("results", {})
        items = results.get("data", [])
        for item in items:
            slug = item.get("name", "")
            title = item.get("title", slug)
            detail_url = f"{BASE_URL}/opendata/dataset/{slug}"

            # Ekstrak URL download dari resources yang sudah ada di JSON
            resources = item.get("resources", [])
            download_urls = [r["url"] for r in resources if r.get("url")]

            all_datasets.append({
                "title": title,
                "slug": slug,
                "detail_url": detail_url,
                "download_urls": download_urls,
            })

    # Proses halaman pertama
    _extract_from_page(page_data)
    time.sleep(REQUEST_DELAY)

    # Proses halaman 2..N
    for page in range(2, last_page + 1):
        url = f"{DATASET_LIST_URL}?page={page}"
        logger.info(f"Mengambil halaman {page}/{last_page}: {url}")
        page_data = _get_page_json(url, session)
        if page_data:
            _extract_from_page(page_data)
        time.sleep(REQUEST_DELAY)

    logger.info(f"Total dataset ditemukan: {len(all_datasets)}")
    return all_datasets


def get_download_links(detail_url: str, session: requests.Session) -> list[str]:
    """
    [OPSIONAL] Kunjungi halaman detail dataset untuk mengambil download URL.
    Fungsi ini hanya digunakan sebagai fallback jika dataset tidak punya
    resources di halaman listing. Dalam kondisi normal, download URL
    sudah tersedia dari get_dataset_list().
    """
    page_data = _get_page_json(detail_url, session)
    if not page_data:
        return []

    props = page_data.get("props", {})
    dataset = props.get("dataset", {})
    resources = dataset.get("resources", [])
    return [r["url"] for r in resources if r.get("url")]
