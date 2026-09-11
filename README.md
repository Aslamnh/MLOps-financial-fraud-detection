# MLOps-financial-fraud-detection
Project MLOps yang akan menganalisis dan mendeteksi transaksi keuangan yang mencurigakan secara otomatis dengan proses data ingestion (CoinGecko API), data processing, training model and automated retraining, evaluation.

## Struktur Direktori

```text
.
├── config/                  # Berkas konfigurasi proyek
├── data/                    # Penyimpanan dataset
│   ├── raw/                 # Data mentah hasil ekstraksi API ( CoinGecko )
│   ├── processed/           # Data yang sudah dibersihkan dan siap diproses model
│   └── .gitkeep             # Menjaga struktur folder tetap terdeteksi Git saat kosong
├── docs/                    # Dokumentasi proyek, diagram arsitektur, dan aset visual
├── models/                  # Tempat menyimpan berkas model terlatih
│   └── .gitkeep             # Menjaga struktur folder tetap terdeteksi Git saat kosong
├── notebooks/               # Jupyter Notebooks untuk analisis eksploratif & eksperimen
├── src/                     # Source code utama proyek
│   ├── hello.py             # Skrip testing / contoh skrip Python
│   └── ingest.py            # Skrip ekstraksi & penarikan data dari CoinGecko API
├── tests/                   # Folder untuk skrip pengujian otomatis (unit testing)
├── .env                     # Variabel lingkungan lokal (berisi API Key asli, di-ignore Git)
├── .gitignore               # Menentukan berkas/folder yang tidak di-push ke GitHub
├── LICENSE                  # Lisensi hak cipta penggunaan proyek
├── README.md                # Dokumentasi utama proyek
└── requirements.txt         # Daftar pustaka (dependencies) Python yang dibutuhkan

```

## Cara Menjalankan di GitHub Codespaces
1. Buka halaman utama repo ini di GitHub
2. Klik tombol hijau "<> Code"
3. Pilih tab "Codespaces" lalu klik "Create codespace on main"
4. Melakukan konfigurasi Environment Variables dengan menambahkan file .env ke projek
5. Buka file .env dan masukkan CoinGecko API Key (COINGECKO_API_KEY=kunci_api)
6. Install dependencies dengan menjalankan "pip install -r requirements.txt"
7. Setup selesai, jalankan "python src/ingest.py" untuk mencoba proses penarikan data