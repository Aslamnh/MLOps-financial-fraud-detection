import os
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

def run_ingestion():
    # 1. Load API Key dari .env
    load_dotenv()
    API_KEY = os.getenv("COINGECKO_API_KEY")

    if not API_KEY:
        raise ValueError("API Key tidak ditemukan di file .env!")

    # 2. Setup Request ke API CoinGecko
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 100,
        "page": 1,
        "sparkline": False
    }
    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": API_KEY
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        
        # Metadata timestamp sesuai perancangan LK-03
        ingest_datetime = datetime.now(timezone.utc)
        current_time_str = ingest_datetime.strftime("%Y-%m-%d %H:%M:%S")
        df["ingested_at"] = current_time_str
        
        if "last_updated" in df.columns:
            df["market_last_updated"] = pd.to_datetime(df["last_updated"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Ekstraksi atribut mentah yang dibutuhkan
        selected_columns = [
            "ingested_at", 
            "id", 
            "current_price", 
            "total_volume", 
            "high_24h", 
            "low_24h", 
            "price_change_percentage_24h"
        ]
        
        df_selected = df[selected_columns].copy()
        
        # 3. Simpan ke folder data/raw/ dengan nama file timestamped (non-destruktif)
        raw_dir = os.path.join("data", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        
        filename_timestamp = ingest_datetime.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(raw_dir, f"raw_market_data_{filename_timestamp}.csv")
        
        df_selected.to_csv(output_file, index=False)
        
        print(f"=== INGESTION BERHASIL (HTTP {response.status_code}) ===")
        print(f"Waktu Pengambilan (UTC): {current_time_str}")
        print(f"Data mentah disimpan di: {output_file}")
        print(f"Jumlah entri: {len(df_selected)} baris\n")
    else:
        print(f"Ingestion Gagal! Status Code: {response.status_code}")
        print(f"Pesan Error: {response.text}")

if __name__ == "__main__":
    run_ingestion()