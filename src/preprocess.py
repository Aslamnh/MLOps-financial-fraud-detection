"""
Module: preprocess.py
Description: Script prapemrosesan data mentah, mencakup Data Cleaning, 
Schema Validation, dan Feature Engineering sesuai spesifikasi dokumen LK-03.
"""

import glob
import os
import sys
import pandas as pd


def get_latest_raw_file(raw_dir: str) -> str:
    """
    Mencari file data mentah terbaru di direktori raw.
    """
    pattern = os.path.join(raw_dir, "raw_market_data_*.csv")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"Tidak ada file data mentah yang cocok dengan pola {pattern}")

    latest_file = max(files, key=os.path.getctime)
    return latest_file


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan pembersihan data, eliminasi duplikat, serta validasi skema.
    """
    initial_count = len(df)

    # 1. Hapus Duplikasi
    df = df.drop_duplicates(subset=["id", "ingested_at"]).copy()

    # 2. Penanganan Missing Values pada Kolom Kritis
    critical_cols = ["id", "current_price", "total_volume", "high_24h", "low_24h"]
    df = df.dropna(subset=critical_cols).copy()

    # 3. Schema Validation sesuai Spesifikasi LK-03
    # current_price > 0, total_volume >= 0, high_24h >= low_24h
    valid_schema_mask = (
        (df["current_price"] > 0)
        & (df["total_volume"] >= 0)
        & (df["high_24h"] >= df["low_24h"])
    )
    df_clean = df[valid_schema_mask].copy()

    # Formating tipe data
    df_clean["ingested_at"] = pd.to_datetime(df_clean["ingested_at"])

    cleaned_count = len(df_clean)
    print(f"[INFO] Cleaning Selesai: {initial_count - cleaned_count} baris tidak valid dibuang.")
    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membentuk fitur finansial baru (Spread dan Z-Score Volume).
    """
    df_feat = df.copy()

    # High-Low Spread
    df_feat["high_low_spread"] = (df_feat["high_24h"] - df_feat["low_24h"]) / df_feat["current_price"]

    # Z-Score Volume Transaksi
    vol_mean = df_feat["total_volume"].mean()
    vol_std = df_feat["total_volume"].std()

    if vol_std > 0:
        df_feat["z_score_volume"] = (df_feat["total_volume"] - vol_mean) / vol_std
    else:
        df_feat["z_score_volume"] = 0.0

    return df_feat


def run_preprocessing() -> None:
    """
    Fungsi utama untuk eksekusi prapemrosesan data.
    """
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    print("--- MEMULAI PRAPEMROSESAN DATA ---")
    latest_file = get_latest_raw_file(raw_dir)
    print(f"Reading file : {latest_file}")

    df_raw = pd.read_csv(latest_file)

    # Eksekusi pipeline pembersihan dan rekayasa fitur
    df_cleaned = clean_data(df_raw)
    df_processed = engineer_features(df_cleaned)

    # Simpan hasil pemrosesan
    output_path = os.path.join(processed_dir, "market_data_processed.csv")
    df_processed.to_csv(output_path, index=False)

    print(f"[SUCCESS] Prapemrosesan berhasil!")
    print(f"Saved File   : {output_path}")
    print(f"New Features : ['high_low_spread', 'z_score_volume']")
    print(f"Clean Rows   : {len(df_processed)}\n")


if __name__ == "__main__":
    try:
        run_preprocessing()
    except Exception as e:
        print(f"[FATAL] Prapemrosesan terhenti: {e}")
        sys.exit(1)