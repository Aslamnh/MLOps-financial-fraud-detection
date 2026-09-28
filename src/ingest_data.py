"""
Module: ingest_data.py
Description: Script untuk melakukan data ingestion secara otomatis dari CoinGecko API.
Supports non-destructive file storage based on timestamping for Continual Learning.
"""

import os
import sys
import time
import pandas as pd
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv


def fetch_market_data(api_key: str, max_retries: int = 3, delay: int = 5) -> dict:
    """
    Mengambil data pasar dari CoinGecko API dengan penanganan error koneksi.
    
    Args:
        api_key (str): Key otentikasi CoinGecko.
        max_retries (int): Batas maksimal percobaan ulang jika gagal.
        delay (int): Waktu tunggu (detik) antar percobaan.
        
    Returns:
        dict: Respons JSON dari API.
    """
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 100,
        "page": 1,
        "sparkline": False,
    }
    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": api_key,
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"[ERROR] HTTP error terjadi (Percobaan {attempt}/{max_retries}): {http_err}")
        except requests.exceptions.ConnectionError as conn_err:
            print(f"[ERROR] Koneksi gagal (Percobaan {attempt}/{max_retries}): {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            print(f"[ERROR] Request timeout (Percobaan {attempt}/{max_retries}): {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            print(f"[ERROR] Fatal error pada request (Percobaan {attempt}/{max_retries}): {req_err}")

        if attempt < max_retries:
            print(f"Menunggu {delay} detik sebelum mencoba ulang...")
            time.sleep(delay)

    raise RuntimeError("Gagal mengambil data dari API setelah beberapa kali percobaan.")


def run_ingestion() -> None:
    """
    Fungsi utama untuk menjalankan pipeline ingesti data dan menyimpan ke data/raw/.
    """
    load_dotenv()
    api_key = os.getenv("COINGECKO_API_KEY")

    if not api_key:
        raise ValueError("COINGECKO_API_KEY tidak ditemukan di environment variables / file .env!")

    print("--- MEMULAI DATA INGESTION ---")
    data = fetch_market_data(api_key)
    df = pd.DataFrame(data)

    # Menambahkan metadata timestamp Waktu UTC
    ingest_datetime = datetime.now(timezone.utc)
    current_time_str = ingest_datetime.strftime("%Y-%m-%d %H:%M:%S")
    df["ingested_at"] = current_time_str

    if "last_updated" in df.columns:
        df["market_last_updated"] = pd.to_datetime(df["last_updated"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    selected_columns = [
        "ingested_at",
        "id",
        "current_price",
        "total_volume",
        "high_24h",
        "low_24h",
        "price_change_percentage_24h",
    ]
    df_selected = df[selected_columns].copy()

    # Penyimpanan Non-Destruktif di folder data/raw/
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    filename_timestamp = ingest_datetime.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(raw_dir, f"raw_market_data_{filename_timestamp}.csv")

    df_selected.to_csv(output_path, index=False)

    print(f"[SUCCESS] Ingesti data berhasil!")
    print(f"Timestamp    : {current_time_str} UTC")
    print(f"Saved File   : {output_path}")
    print(f"Total Rows   : {len(df_selected)}\n")


if __name__ == "__main__":
    try:
        run_ingestion()
    except Exception as e:
        print(f"[FATAL] Program terhenti: {e}")
        sys.exit(1)