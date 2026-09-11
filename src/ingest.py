import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# 1. Load API Key dari .env
load_dotenv()
API_KEY = os.getenv("COINGECKO_API_KEY")

if not API_KEY:
    raise ValueError("API Key tidak ditemukan di file .env!")

# 2. Setup Request
url = "https://api.coingecko.com/api/v3/coins/markets"
params = {
    "vs_currency": "usd",
    "order": "market_cap_desc",
    "per_page": 5,
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
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df["ingested_at"] = current_time
    
    if "last_updated" in df.columns:
        df["market_last_updated"] = pd.to_datetime(df["last_updated"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    
    selected_columns = [
        "ingested_at", 
        "id", 
        "current_price", 
        "total_volume", 
        "high_24h", 
        "low_24h", 
        "price_change_percentage_24h"
    ]
    
    print(f"=== FETCH BERHASIL (HTTP {response.status_code}) ===")
    print(f"Waktu Pengambilan: {current_time}\n")
    print(df[selected_columns].to_string(index=False))
else:
    print(f"Fetch Gagal! Status Code: {response.status_code}")
    print(f"Pesan Error: {response.text}")