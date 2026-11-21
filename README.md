# MLOps-financial-fraud-detection
Project MLOps yang akan menganalisis dan mendeteksi transaksi keuangan yang mencurigakan secara otomatis dengan proses data ingestion (CoinGecko API), data processing, training model and automated retraining, evaluation.

Link GitHub: https://github.com/Aslamnh/MLOps-financial-fraud-detection/tree/main

## Struktur Direktori

```
.
├── config/                  # Berkas konfigurasi proyek
├── data/                    # Penyimpanan dataset (dilacak DVC, bukan Git)
│   ├── raw/                 # Data mentah hasil ekstraksi API ( CoinGecko )
│   ├── processed/           # Data yang sudah dibersihkan dan siap diproses model
│   └── .gitkeep             # Menjaga struktur folder tetap terdeteksi Git saat kosong
├── docs/                    # Dokumentasi proyek, diagram arsitektur, dan aset visual
├── models/                  # Tempat menyimpan berkas model terlatih
│   └── .gitkeep             # Menjaga struktur folder tetap terdeteksi Git saat kosong
├── notebooks/               # Jupyter Notebooks untuk analisis eksploratif & eksperimen
├── src/                     # Source code utama proyek
│   ├── hello.py             # Skrip testing / contoh skrip Python
│   ├── ingest_data.py       # Skrip ekstraksi & penarikan data dari CoinGecko API
│   ├── preprocess.py        # Skrip pembersihan dan pelabelan data
│   ├── train.py             # Pelatihan model + pelacakan eksperimen MLflow (LK-06)
│   └── model_registry.py    # Pengelolaan Model Registry dan inferensi (LK-07)
├── tests/                   # Folder untuk skrip pengujian otomatis (unit testing)
├── .dvc/                    # Konfigurasi DVC (remote Backblaze B2, tanpa kredensial)
├── .env                     # (Opsional, hanya lokal) variabel lingkungan, di-ignore Git
├── .gitignore               # Menentukan berkas/folder yang tidak di-push ke GitHub
├── LICENSE                  # Lisensi hak cipta penggunaan proyek
├── README.md                # Dokumentasi utama proyek
└── requirements.txt         # Daftar pustaka (dependencies) Python yang dibutuhkan

```

## Cara Menjalankan di GitHub Codespaces
1. Buka halaman utama repo ini di GitHub.
2. Klik tombol hijau "<> Code".
3. Pilih tab "Codespaces" lalu klik "Create codespace on main".
4. Melakukan konfigurasi Environment Variables dengan menambahkan file .env ke projek.
5. Buka file .env dan masukkan CoinGecko API Key (COINGECKO_API_KEY=kunci_api).
6. Install dependencies dengan menjalankan `pip install -r requirements.txt`.
7. Setup selesai, jalankan "python src/ingest.py" untuk mencoba proses penarikan data.

## Cara Melakukan Data Ingestion dan Data Preprocessing
1. Pastikan .env sudah dimasukkan CoinGecko API Key (COINGECKO_API_KEY=kunci_api).
2. Pastikan dependencies sudah terinstal dengan `pip install -r requirements.txt`.
3. Jalankan data ingestion: `python src/ingest_data.py`.
4. Jalankan data preprocessing: `python src/preprocess.py`.
5. Folder data/raw menyimpan data murni sedangkan data/processed menyimpan data yang sudah dilakukan preprocessing.

## Alur Versioning Data (DVC)
Data versioning dengan DVC penting untuk reproduksibilitas model. Data transaksi bersifat dinamis setiap dilakukan data ingestion, pelacakan hash data fisik dengan .dvc memungkinkan untuk kembali ke versi dataset spesifik yang ingin digunakan.

Langkah-Langkah Data Versioning dengan DVC:
1. Inisialisasi DVC
    ```
    dvc init
    ```
2. Mendaftarkan direktori data ke DVC agar menghasilkan file penunjuk (.dvc)
    ```
    dvc add data/raw/
    dvc add data/processed/
    ```
3. Saat data baru ditarik melalui skrip ingestion dan preprocess
    ```
    python src/ingest_data.py
    python src/preprocess.py
    dvc add data/raw/
    dvc add data/processed/
    ```
4. Untuk melakukan audit transisi versi data
    ```
    dvc diff
    ```

## Otomatisasi Pipeline (CI/CD dengan GitHub Actions)

Setiap `push` ke `main` dan setiap `pull request` ke `main` memicu alur kerja
`.github/workflows/mlops-automation.yaml` (prinsip *Code as a Trigger*):

| Tahap | Job / langkah | Isi |
|---|---|---|
| 1 | `test` | `pytest tests` (logika pelabelan, pemuatan data, metrik, pengecekan threshold) |
| 2 | Automated Training | `dvc pull data/processed.dvc` dari Backblaze B2, lalu `python src/train.py --no-register` |
| 3 | Evaluasi dan validasi | `python src/evaluate_model.py` membandingkan metrik dengan `config/thresholds.json`; gagal bila ada metrik di bawah ambang |
| 4 | Auto-registry | `python src/register_staging.py` mendaftarkan model ke MLflow Model Registry dengan alias `staging` |

Tahap 2 sampai 4 hanya berjalan bila tahap sebelumnya lulus. Model yang tidak memenuhi threshold tidak pernah masuk registry.

### Secret yang dibutuhkan
Tambahkan di **Settings > Secrets and variables > Actions** (terpisah dari Codespaces secret):
`AWS_ACCESS_KEY_ID` dan `AWS_SECRET_ACCESS_KEY` (kunci Backblaze B2 untuk `dvc pull`).

### Mengubah threshold
Ubah nilai pada `config/thresholds.json`. Perubahan ini juga memicu pipeline.

### Menjalankan secara lokal
```bash
pip install pytest
python -m pytest tests -v
python src/train.py --no-register
python src/evaluate_model.py
python src/register_staging.py
```

Catatan: registry pada CI memakai SQLite sementara di runner, sehingga setiap eksekusi membuat registry baru
(`mlflow.db` dan laporan evaluasi disimpan sebagai artifact eksekusi). Untuk registry yang berkelanjutan
diperlukan MLflow tracking server terpusat.

## Menjalankan Seluruh Sistem dengan Docker Compose

Dua layanan berjalan bersama dalam satu jaringan Docker (`fraud-detection-net`):

| Layanan | Peran | Port host |
|---|---|---|
| `mlflow-server` | Tracking server dan Model Registry MLflow (backend SQLite di volume persisten) | 5000 |
| `api-service` | API inferensi FastAPI; menarik model `models:/fraud-detection-rf@production` dari `mlflow-server` | 8000 |

Satu layanan tambahan, `mlflow-init`, hanya berjalan sekali untuk menyalin registry hasil `dvc pull` ke volume
`mlflow-data`. `api-service` baru dimulai setelah `mlflow-server` sehat (`depends_on` dengan `service_healthy`).

### Prasyarat
- Docker Desktop (atau Docker Engine) dengan Docker Compose v2.
- Registry dan model sudah diambil dari remote DVC. Ini membutuhkan kredensial Backblaze B2
  (`AWS_ACCESS_KEY_ID` dan `AWS_SECRET_ACCESS_KEY`):
  ```bash
  pip install -r requirements.txt
  dvc pull
  ```
  Setelah itu `mlflow.db`, `mlruns/`, dan `models/production/` ada di folder proyek.

### Satu perintah untuk menjalankan semuanya
```bash
docker compose up -d --build
```

### Memeriksa status
```bash
docker compose ps
```
Kedua layanan harus berstatus `running (healthy)`.

- MLflow UI: http://localhost:5000
- Dokumentasi API: http://localhost:8000/docs
- Pemeriksaan kesehatan: http://localhost:8000/health
- Informasi model: http://localhost:8000/model-info (`model_source` bernilai `tracking-server`)

### Mencoba prediksi
PowerShell:
```powershell
$body = @{ transactions = @(@{ current_price = 95000; total_volume = 3.2e10; high_24h = 97000; low_24h = 93500 }) } | ConvertTo-Json -Depth 4
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5
```
Bash:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"transactions":[{"current_price":95000,"total_volume":3.2e10,"high_24h":97000,"low_24h":93500}]}'
```

### Menghentikan dan mengatur ulang
```bash
docker compose down        # hentikan kontainer; registry di volume tetap ada
docker compose up -d       # jalankan lagi; data masih sama
docker compose down -v     # hentikan dan HAPUS volume (registry disalin ulang dari folder proyek)
```

### Pemecahan masalah
- **`mlflow-init` gagal dengan pesan `dvc pull`:** `mlflow.db` atau `mlruns/` belum ada. Jalankan `dvc pull`.
- **`api-service` memakai `model_source: local-fallback`:** tracking server tidak dapat mengakses artefak model.
  MLflow menyimpan path artefak secara absolut di `mlflow.db`; periksa path yang tersimpan:
  ```bash
  python -c "import sqlite3; c=sqlite3.connect('mlflow.db'); print(c.execute('select name, artifact_location from experiments').fetchall())"
  ```
  Jika path-nya bukan `/workspaces/MLOps-financial-fraud-detection/...`, jalankan dengan
  `REGISTRY_ROOT=<path-yang-tersimpan> docker compose up -d --build`, lalu `docker compose down -v` bila
  volume sudah terisi dengan path yang berbeda.
- **Mengganti model aktif:** pindahkan alias `production` di MLflow UI, lalu `curl -X POST http://localhost:8000/reload`.

