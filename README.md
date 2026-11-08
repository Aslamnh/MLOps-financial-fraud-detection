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

## Model Registry dan Inferensi

Model fraud-detection-rf dikelola di MLflow Model Registry dengan src/model_registry.py.

Model aktif untuk inferensi
Versi aktif: v1 (alias production), berasal dari run rf-balanced-deep (200 pohon, max_depth 8, min_samples_leaf 1, class_weight balanced)
URI inferensi: models:/fraud-detection-rf@production
Pembanding: v2 (rf-tuned-v2: 300 pohon, max_depth 10) di Staging
| Versi | cv_f1_mean | cv_f1_std | f1 uji | precision | recall | roc_auc |
|---|---|---|---|---|---|---|
| v1 | (Production) | 0,3667 | 0,3055 | 0,4000 | 0,3333 | 0,5000 | 0,9643 |
| v2 | (Staging) | 0,3333 | 0,2789 | 0,5000 | 0,5000 | 0,5000 |	0,9643 |

Alasan memilih v1: cv_f1_mean lebih tinggi dan menjadi metrik utama, perbedaan performa metrik lain sangat kecil sehingga model yang lebih sederhana dipilih karena lebih cepat dan efisien. Namun, v2 memiliki precision dan recall yang lebih tinggi.

Berkas metadata
* models/registry_metadata.yaml: referensi model, metrik, hash data, hash model, dan git commit
* models/production.dvc: pointer DVC artefak model Production
* docs/model_version_comparison.md: perbandingan versi
* docs/stage_transitions.md: riwayat transisi stage
* docs/mlflow_runs.csv: ekspor seluruh run
* mlflow.db.dvc, mlruns.dvc: cadangan MLflow tracking (DVC)

Cara menjalankan
```
bash
dvc pull                                         # ambil data dan cadangan MLflow dari B2
python src/model_registry.py describe            # deskripsi dan tag metadata
python src/model_registry.py new-version --run-name rf-tuned-v2 --n-estimators 300 --max-depth 10 --min-samples-leaf 1
python src/model_registry.py compare             # bandingkan versi
python src/model_registry.py promote --version 1 --stage Staging
python src/model_registry.py promote --version 2 --stage Staging
python src/model_registry.py promote --version 1 --stage Production --note "cv_f1_mean tertinggi"
python src/model_registry.py verify              # uji load_model dan prediksi
python src/model_registry.py publish             # ekspor model, dvc add, tulis metadata YAML
dvc push
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Catatan: mlflow.db dan mlruns/ dicadangkan lewat DVC; pemulihan di Codespace baru mengandalkan path repositori yang sama (/workspaces/MLOps-financial-fraud-detection). Jalankan train.py hanya sekali pada registry kosong, karena setiap eksekusi mendaftarkan versi baru.