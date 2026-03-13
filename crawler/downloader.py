"""
downloader.py
=============
Modul untuk mendownload file dataset dari URL ke folder lokal.
"""

import os
import time
import logging
import requests
from urllib.parse import urlparse, unquote
from tqdm import tqdm
import urllib3

# SSL cert pada dev-ckan.kominfo.go.id sudah expired.
# Nonaktifkan verifikasi SSL dan suppress peringatannya.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from config import HEADERS, REQUEST_TIMEOUT, REQUEST_DELAY, DATASET_FOLDER

logger = logging.getLogger(__name__)


def _get_filename_from_url(url: str, response: requests.Response) -> str:
    """
    Tentukan nama file dari:
    1. Header Content-Disposition
    2. URL path terakhir
    """
    # Cek Content-Disposition header
    cd = response.headers.get("Content-Disposition", "")
    if "filename=" in cd:
        filename = cd.split("filename=")[-1].strip().strip('"').strip("'")
        if filename:
            return filename

    # Ambil dari URL path
    parsed = urlparse(url)
    path = unquote(parsed.path)
    filename = path.split("/")[-1]

    # Jika tidak ada ekstensi, tambahkan .csv sebagai default
    if "." not in filename:
        filename = filename + ".csv"

    return filename


def _sanitize_filename(filename: str) -> str:
    """Bersihkan karakter tidak valid dari nama file."""
    invalid_chars = r'\/:*?"<>|'
    for ch in invalid_chars:
        filename = filename.replace(ch, "_")
    return filename.strip()


def download_file(url: str, session: requests.Session, dest_folder: str = DATASET_FOLDER) -> str | None:
    """
    Download file dari URL ke dest_folder.
    - Jika file sudah ada, skip (tidak download ulang).
    - Return path lokal file jika berhasil, atau None jika gagal.
    """
    os.makedirs(dest_folder, exist_ok=True)

    try:
        # Lakukan streaming request agar file besar tidak habiskan RAM
        # verify=False: bypass SSL cert expired pada dev-ckan.kominfo.go.id
        with session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, stream=True, verify=False) as resp:
            resp.raise_for_status()

            filename = _sanitize_filename(_get_filename_from_url(url, resp))
            if not filename:
                logger.warning(f"Nama file tidak dapat ditentukan dari: {url}")
                return None

            dest_path = os.path.join(dest_folder, filename)

            # Skip jika sudah ada
            if os.path.exists(dest_path):
                logger.info(f"  [SKIP] File sudah ada: {filename}")
                return dest_path

            # Ambil ukuran file untuk progress bar
            total_size = int(resp.headers.get("Content-Length", 0))

            logger.info(f"  [DOWNLOAD] {filename} ({total_size / 1024:.1f} KB)")

            with open(dest_path, "wb") as f, tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=filename[:40],
                leave=False,
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        time.sleep(REQUEST_DELAY)
        return dest_path

    except requests.exceptions.RequestException as e:
        logger.error(f"  [GAGAL] Download dari {url}: {e}")
        return None
    except OSError as e:
        logger.error(f"  [ERROR] Gagal menyimpan file: {e}")
        return None


def download_all(dataset_with_links: list[dict], session: requests.Session) -> list[dict]:
    """
    Download semua file dari list dataset.
    Input: [{"title": "...", "slug": "...", "download_urls": [...]}]
    Output: list yang sama, ditambah key "local_files": [path, ...]
    """
    results = []
    for item in dataset_with_links:
        local_files = []
        for url in item.get("download_urls", []):
            local_path = download_file(url, session)
            if local_path:
                local_files.append(local_path)
        results.append({**item, "local_files": local_files})
    return results
