"""
Mendaftarkan model hasil pelatihan ke MLflow Model Registry dengan status Staging.

Hanya dijalankan setelah evaluate_model.py lulus. Model diambil dari run terbaik
yang dicatat di models/best_model_metadata.json.

Status Staging diberikan lewat alias `staging` (cara yang dianjurkan MLflow 3),
tag `stage`, dan API stage klasik bila masih tersedia.
"""
import json
import os
import sys
import warnings
from pathlib import Path

import mlflow
from mlflow import MlflowClient

METADATA_PATH = os.path.join("models", "best_model_metadata.json")
REPORT_PATH = os.path.join("models", "evaluation_report.json")
OUTPUT_PATH = os.path.join("models", "staging_registration.json")
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def main() -> int:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))

    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, encoding="utf-8") as f:
            if not json.load(f).get("passed", False):
                print("[GAGAL] Laporan evaluasi menyatakan model tidak lulus. Registrasi dibatalkan.")
                return 1
    else:
        print("[GAGAL] Laporan evaluasi tidak ditemukan. Jalankan evaluate_model.py terlebih dahulu.")
        return 1

    name = metadata["registered_model_name"]
    run_id = metadata["run_id"]
    metrics = metadata.get("metrics", {})
    git_commit = os.getenv("GIT_COMMIT") or metadata.get("git_commit", "unknown")

    client = MlflowClient()
    registered = mlflow.register_model(f"runs:/{run_id}/model", name)
    version = str(registered.version)

    client.update_model_version(
        name,
        version,
        description=(
            f"Dilatih otomatis oleh pipeline CI pada commit {git_commit}. "
            f"Run {metadata.get('run_name')}; "
            f"cv_f1_mean={metrics.get('cv_f1_mean')}, f1={metrics.get('f1_score')}, "
            f"recall={metrics.get('recall')}, roc_auc={metrics.get('roc_auc')}. "
            "Lulus validasi threshold (config/thresholds.json)."
        ),
    )
    for key, value in {
        "stage": "Staging",
        "git_commit": git_commit,
        "run_id": run_id,
        "data_sha256": metadata.get("dataset", {}).get("sha256", ""),
        "validated": "true",
        "registered_by": "ci-pipeline",
    }.items():
        client.set_model_version_tag(name, version, key, value)

    client.set_registered_model_alias(name, "staging", version)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            client.transition_model_version_stage(name, version, stage="Staging")
    except Exception as exc:  # API stage klasik sudah deprecated; alias sudah cukup.
        print(f"[INFO] API stage klasik dilewati ({type(exc).__name__}); alias 'staging' sudah diatur.")

    output = {"model": name, "version": version, "alias": "staging", "run_id": run_id, "git_commit": git_commit}
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"[SUCCESS] {name} v{version} terdaftar dengan alias 'staging' (commit {git_commit[:7]}).")

    summary_file = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(f"### Registry: `{name}` v{version} masuk Staging (commit `{git_commit[:7]}`)\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[FATAL] Registrasi gagal: {exc}")
        sys.exit(1)
