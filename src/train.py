"""
Module: train.py
Description: Pelatihan model deteksi transaksi/pasar mencurigakan dengan
manajemen eksperimen MLflow (LK-06).

Alur:
    1. Muat data/processed/market_data_processed.csv (hasil preprocess.py + DVC LK-05).
    2. Bentuk label `is_suspicious` (aturan anomali) dan fitur model.
    3. Jalankan beberapa konfigurasi hyperparameter (minimal 3 run), tiap run
       dicatat ke MLflow: parameter, metrik, artefak, dan model.
    4. Pilih run terbaik berdasarkan metrik cross-validation, daftarkan ke
       MLflow Model Registry, dan tulis metadata ke models/best_model_metadata.json.

Cara pakai:
    python src/train.py
    python src/train.py --data-path data/processed/market_data_processed.csv --seed 42
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

DEFAULT_DATA_PATH = os.path.join("data", "processed", "market_data_processed.csv")
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "financial-fraud-detection"
REGISTERED_MODEL_NAME = "fraud-detection-rf"
METADATA_PATH = os.path.join("models", "best_model_metadata.json")
SELECTION_METRIC = "cv_f1_mean"

FEATURE_COLUMNS = ["log_price", "log_volume", "high_low_spread"]

# Ambang aturan pelabelan anomali (dicatat sebagai parameter di setiap run).
LABEL_RULES = {"volume_z_threshold": 2.0, "price_change_z_threshold": 2.0}

# Konfigurasi eksperimen (hyperparameter tuning). Minimal 3 run.
RUN_CONFIGS = [
    {"run_name": "rf-baseline", "n_estimators": 50, "max_depth": 3, "min_samples_leaf": 5, "class_weight": None},
    {"run_name": "rf-balanced-medium", "n_estimators": 100, "max_depth": 5, "min_samples_leaf": 2, "class_weight": "balanced"},
    {"run_name": "rf-balanced-deep", "n_estimators": 200, "max_depth": 8, "min_samples_leaf": 1, "class_weight": "balanced"},
    {"run_name": "rf-balanced-subsample", "n_estimators": 300, "max_depth": None, "min_samples_leaf": 2, "class_weight": "balanced_subsample"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training + MLflow tracking (LK-06)")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--no-register", action="store_true", help="Lewati registrasi Model Registry")
    return parser.parse_args()


def file_sha256(path: str) -> str:
    """Hash isi dataset agar versi data tiap run dapat ditelusuri."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}. Jalankan ingest_data.py & preprocess.py "
            "(atau `dvc pull`) terlebih dahulu."
        )
    df = pd.read_csv(path)
    required = {
        "current_price", "total_volume", "high_low_spread",
        "z_score_volume", "price_change_percentage_24h",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ada pada dataset: {sorted(missing)}")
    return df.dropna(subset=list(required)).reset_index(drop=True)


def build_features_and_labels(df: pd.DataFrame):
    """
    Dataset CoinGecko tidak memiliki label fraud asli, sehingga label `is_suspicious`
    dibentuk dari aturan anomali: lonjakan volume ekstrem ATAU perubahan harga 24 jam
    ekstrem. Fitur model sengaja TIDAK memuat z_score_volume dan
    price_change_percentage_24h agar model tidak sekadar menghafal aturan label.
    """
    price_change = df["price_change_percentage_24h"]
    price_change_std = price_change.std()
    price_change_z = (
        (price_change - price_change.mean()) / price_change_std if price_change_std > 0 else price_change * 0.0
    )

    label = (
        (df["z_score_volume"] > LABEL_RULES["volume_z_threshold"])
        | (price_change_z.abs() > LABEL_RULES["price_change_z_threshold"])
    ).astype(int)

    features = pd.DataFrame(
        {
            "log_price": np.log1p(df["current_price"]),
            "log_volume": np.log1p(df["total_volume"]),
            "high_low_spread": df["high_low_spread"],
        }
    )[FEATURE_COLUMNS]
    return features, label


def evaluate(model, x_test, y_test) -> dict:
    y_pred = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }
    if y_test.nunique() > 1:
        metrics["roc_auc"] = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    return metrics


def train_one_run(config, x_train, x_test, y_train, y_test, args, dataset_info) -> dict:
    hyperparams = {k: v for k, v in config.items() if k != "run_name"}
    n_splits = int(min(5, y_train.value_counts().min()))
    cv = StratifiedKFold(n_splits=max(n_splits, 2), shuffle=True, random_state=args.seed)

    with mlflow.start_run(run_name=config["run_name"]) as run:
        # --- Parameter ---
        mlflow.log_params({**hyperparams, "random_state": args.seed, "test_size": args.test_size})
        mlflow.log_params({f"label_{k}": v for k, v in LABEL_RULES.items()})
        mlflow.log_param("features", ",".join(FEATURE_COLUMNS))
        mlflow.log_param("cv_folds", cv.get_n_splits())

        # --- Tag keterlacakan / reproduksibilitas ---
        mlflow.set_tags(
            {
                "git_commit": dataset_info["git_commit"],
                "data_path": args.data_path,
                "data_sha256": dataset_info["sha256"],
                "task_type": "classification",
            }
        )

        model = RandomForestClassifier(**hyperparams, random_state=args.seed, n_jobs=-1)
        cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="f1")
        model.fit(x_train, y_train)

        # --- Metrik ---
        metrics = evaluate(model, x_test, y_test)
        metrics["cv_f1_mean"] = float(cv_scores.mean())
        metrics["cv_f1_std"] = float(cv_scores.std())
        mlflow.log_metrics(metrics)
        mlflow.log_metrics(
            {"n_train": len(x_train), "n_test": len(x_test), "positive_rate": float(y_train.mean())}
        )

        # --- Artefak tambahan + model ---
        importances = dict(zip(FEATURE_COLUMNS, map(float, model.feature_importances_)))
        mlflow.log_dict(importances, "feature_importances.json")

        signature = infer_signature(x_train, model.predict(x_train))
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=x_train.head(3),
            serialization_format="cloudpickle",
        )

        print(
            f"[RUN] {config['run_name']:<24} cv_f1={metrics['cv_f1_mean']:.4f} "
            f"test_f1={metrics['f1_score']:.4f} run_id={run.info.run_id}"
        )
        return {"run_id": run.info.run_id, "config": config, "metrics": metrics}


def select_best_run(experiment_name: str) -> dict:
    """Pilih run terbaik langsung dari data MLflow (bukan dari memori skrip)."""
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{SELECTION_METRIC} DESC", "metrics.f1_score DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError("Tidak ada run selesai yang dapat dipilih.")
    return runs.iloc[0].to_dict()


def write_metadata(best: dict, dataset_info: dict, version) -> None:
    params = {k.replace("params.", ""): v for k, v in best.items() if k.startswith("params.")}
    metrics = {k.replace("metrics.", ""): v for k, v in best.items() if k.startswith("metrics.")}
    metadata = {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "model_version": version,
        "run_id": best["run_id"],
        "run_name": best.get("tags.mlflow.runName"),
        "selection_metric": SELECTION_METRIC,
        "best_parameters": params,
        "metrics": metrics,
        "dataset": {"path": dataset_info["path"], "sha256": dataset_info["sha256"], "rows": dataset_info["rows"]},
        "git_commit": dataset_info["git_commit"],
        "model_uri": f"runs:/{best['run_id']}/model",
    }
    Path(METADATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"[SUCCESS] Metadata model disimpan: {METADATA_PATH}")


def run_training() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("--- MEMULAI TRAINING & EXPERIMENT TRACKING ---")
    df = load_dataset(args.data_path)
    x, y = build_features_and_labels(df)
    print(f"Dataset      : {args.data_path} ({len(df)} baris, {int(y.sum())} berlabel suspicious)")

    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise ValueError(
            "Kelas positif terlalu sedikit untuk stratified split. Tambah data (jalankan "
            "ingestion beberapa kali) atau longgarkan LABEL_RULES."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    dataset_info = {
        "path": args.data_path,
        "sha256": file_sha256(args.data_path),
        "rows": len(df),
        "git_commit": get_git_commit(),
    }

    for config in RUN_CONFIGS:
        train_one_run(config, x_train, x_test, y_train, y_test, args, dataset_info)

    best = select_best_run(EXPERIMENT_NAME)
    print(f"[BEST] run_id={best['run_id']} {SELECTION_METRIC}={best[f'metrics.{SELECTION_METRIC}']:.4f}")

    version = None
    if not args.no_register:
        registered = mlflow.register_model(f"runs:/{best['run_id']}/model", REGISTERED_MODEL_NAME)
        version = registered.version
        print(f"[SUCCESS] Model terdaftar: {REGISTERED_MODEL_NAME} v{version}")

    write_metadata(best, dataset_info, version)
    print("Buka dashboard: mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    try:
        run_training()
    except Exception as e:
        print(f"[FATAL] Training terhenti: {e}")
        sys.exit(1)