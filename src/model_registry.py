"""
Module: model_registry.py
Description: LK-07 - Model Registry, Versioning, dan Kesiapan Inferensi.

Melanjutkan LK-06 (src/train.py). Skrip ini mengelola siklus hidup model
`fraud-detection-rf` di MLflow Model Registry.

Subcommand (dijalankan berurutan):
    python src/model_registry.py describe      # metadata lengkap pada model & tiap versi
    python src/model_registry.py new-version   # latih varian baru -> versi baru (v2)
    python src/model_registry.py compare       # bandingkan versi, tulis docs/model_version_comparison.md
    python src/model_registry.py promote --version 1 --stage Staging
    python src/model_registry.py promote --version 2 --stage Production
    python src/model_registry.py verify        # uji load_model + prediksi data baru
    python src/model_registry.py publish       # ekspor model + catat metadata (DVC + YAML)

Prasyarat: `python src/train.py` sudah dijalankan (membuat versi 1) pada
backend MLflow yang sama (sqlite:///mlflow.db, atau MLFLOW_TRACKING_URI).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.artifacts
import mlflow.pyfunc
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split

import train  # modul LK-06: src/train.py (aman diimpor, ada guard __main__)

COMPARISON_PATH = os.path.join("docs", "model_version_comparison.md")
TRANSITIONS_PATH = os.path.join("docs", "stage_transitions.md")
EXPORT_DIR = os.path.join("models", "production")
REGISTRY_METADATA_PATH = os.path.join("models", "registry_metadata.yaml")
DATA_DVC_POINTER = os.path.join("data", "processed.dvc")
VALID_STAGES = ["None", "Staging", "Production", "Archived"]

MODEL_DESCRIPTION = (
    "Random Forest classifier untuk mendeteksi data pasar kripto yang mencurigakan "
    "(anomali volume atau perubahan harga 24 jam) pada proyek MLOps financial fraud detection. "
    "Sumber data: CoinGecko API (ingest_data.py), diproses preprocess.py, diversioning dengan DVC. "
    "Fitur: log_price, log_volume, high_low_spread. "
    "Label `is_suspicious` dibentuk dari aturan anomali (z-score volume > 2 atau |z-score perubahan "
    "harga 24 jam| > 2), bukan label fraud asli. Metrik seleksi: cv_f1_mean."
)


# --------------------------------------------------------------------------- #
# Utilitas
# --------------------------------------------------------------------------- #
def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def read_dvc_md5(pointer_path: str) -> str:
    """Ambil hash md5 dari file .dvc (silsilah data/model)."""
    if not os.path.exists(pointer_path):
        return "tidak-ditemukan"
    with open(pointer_path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f) or {}
    outs = content.get("outs") or [{}]
    return str(outs[0].get("md5", "tidak-ditemukan"))


def setup(args):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", train.DEFAULT_TRACKING_URI))
    mlflow.set_experiment(train.EXPERIMENT_NAME)
    return MlflowClient(), args.model_name


def get_versions(client: MlflowClient, name: str) -> list:
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(
            f"Model '{name}' belum terdaftar. Jalankan `python src/train.py` terlebih dahulu "
            "(pada backend MLflow yang sama)."
        )
    return sorted(versions, key=lambda v: int(v.version))


def run_details(client: MlflowClient, run_id: str) -> dict:
    run = client.get_run(run_id)
    return {
        "name": run.data.tags.get("mlflow.runName", run_id[:8]),
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "tags": dict(run.data.tags),
    }


def fmt(value, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def prepare_data(args):
    """Muat data dan buat split identik dengan train.py (seed & stratifikasi sama)."""
    df = train.load_dataset(args.data_path)
    x, y = train.build_features_and_labels(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    dataset_info = {
        "path": args.data_path,
        "sha256": train.file_sha256(args.data_path),
        "rows": len(df),
        "git_commit": train.get_git_commit(),
    }
    return df, (x_train, x_test, y_train, y_test), dataset_info


def features_from_market(price: float, volume: float, high: float, low: float) -> pd.DataFrame:
    """Bentuk fitur model dari satu data pasar baru (simulasi input layanan inferensi)."""
    import numpy as np

    return pd.DataFrame(
        {
            "log_price": [float(np.log1p(price))],
            "log_volume": [float(np.log1p(volume))],
            "high_low_spread": [float((high - low) / price)],
        }
    )[train.FEATURE_COLUMNS]


# --------------------------------------------------------------------------- #
# describe: metadata lengkap pada model terdaftar dan setiap versi
# --------------------------------------------------------------------------- #
def describe_version(client: MlflowClient, name: str, version) -> None:
    details = run_details(client, version.run_id)
    m, p, t = details["metrics"], details["params"], details["tags"]
    param_text = ", ".join(
        f"{k}={p[k]}" for k in ["n_estimators", "max_depth", "min_samples_leaf", "class_weight"] if k in p
    )
    description = (
        f"Versi {version.version} dari run '{details['name']}' (run_id {version.run_id[:8]}). "
        f"Hyperparameter: {param_text}. "
        f"Metrik: cv_f1_mean={fmt(m.get('cv_f1_mean'))}, f1_score={fmt(m.get('f1_score'))}, "
        f"precision={fmt(m.get('precision'))}, recall={fmt(m.get('recall'))}, "
        f"roc_auc={fmt(m.get('roc_auc'))}. "
        f"Data: {t.get('data_path', '-')} (sha256 {t.get('data_sha256', '-')[:12]}...), "
        f"git commit {t.get('git_commit', '-')}."
    )
    client.update_model_version(name, version.version, description=description)

    tags = {
        "run_id": version.run_id,
        "run_name": details["name"],
        "cv_f1_mean": fmt(m.get("cv_f1_mean")),
        "f1_score": fmt(m.get("f1_score")),
        "recall": fmt(m.get("recall")),
        "data_sha256": t.get("data_sha256", "-"),
        "git_commit": t.get("git_commit", "-"),
        "dvc_data_md5": read_dvc_md5(DATA_DVC_POINTER),
        "task_type": "classification",
    }
    for key, value in tags.items():
        client.set_model_version_tag(name, version.version, key, str(value))


def cmd_describe(args) -> None:
    client, name = setup(args)
    client.update_registered_model(name, description=MODEL_DESCRIPTION)
    client.set_registered_model_tag(name, "project", "mlops-financial-fraud-detection")
    client.set_registered_model_tag(name, "owner", "aslamnh")
    for version in get_versions(client, name):
        describe_version(client, name, version)
        print(f"[SUCCESS] Metadata versi {version.version} dilengkapi (deskripsi + tag).")


# --------------------------------------------------------------------------- #
# new-version: latih varian baru lalu daftarkan sebagai versi berikutnya
# --------------------------------------------------------------------------- #
def cmd_new_version(args) -> None:
    client, name = setup(args)
    existing = get_versions(client, name)
    _, splits, dataset_info = prepare_data(args)
    x_train, x_test, y_train, y_test = splits

    config = {
        "run_name": args.run_name,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "class_weight": None if args.class_weight == "none" else args.class_weight,
    }
    result = train.train_one_run(config, x_train, x_test, y_train, y_test, args, dataset_info)

    registered = mlflow.register_model(f"runs:/{result['run_id']}/model", name)
    version = client.get_model_version(name, registered.version)
    describe_version(client, name, version)
    client.set_model_version_tag(name, version.version, "change_note", args.note)
    print(
        f"[SUCCESS] Versi baru terdaftar: {name} v{version.version} "
        f"(sebelumnya {len(existing)} versi). Catatan: {args.note}"
    )


# --------------------------------------------------------------------------- #
# compare: perbandingan antar versi + dasar pemilihan
# --------------------------------------------------------------------------- #
def cmd_compare(args) -> None:
    client, name = setup(args)
    rows = []
    for version in get_versions(client, name):
        details = run_details(client, version.run_id)
        stage = getattr(version, "current_stage", "None") or "None"
        rows.append(
            {
                "version": int(version.version),
                "run_name": details["name"],
                "stage": stage,
                "aliases": ",".join(getattr(version, "aliases", []) or []) or "-",
                "n_estimators": details["params"].get("n_estimators", "-"),
                "max_depth": details["params"].get("max_depth", "-"),
                "min_samples_leaf": details["params"].get("min_samples_leaf", "-"),
                "class_weight": details["params"].get("class_weight", "-"),
                "cv_f1_mean": details["metrics"].get("cv_f1_mean"),
                "cv_f1_std": details["metrics"].get("cv_f1_std"),
                "f1_score": details["metrics"].get("f1_score"),
                "precision": details["metrics"].get("precision"),
                "recall": details["metrics"].get("recall"),
                "roc_auc": details["metrics"].get("roc_auc"),
            }
        )

    best = max(rows, key=lambda r: (r["cv_f1_mean"] or 0.0, r["f1_score"] or 0.0))
    lines = [
        f"# Perbandingan Versi Model `{name}`",
        "",
        f"Dibuat: {now_iso()}",
        "",
        "| Versi | Run | Stage | Alias | n_est | max_depth | leaf | class_weight | cv_f1_mean | cv_f1_std | f1 | precision | recall | roc_auc |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| v{r['version']} | {r['run_name']} | {r['stage']} | {r['aliases']} | {r['n_estimators']} | "
            f"{r['max_depth']} | {r['min_samples_leaf']} | {r['class_weight']} | {fmt(r['cv_f1_mean'])} | "
            f"{fmt(r['cv_f1_std'])} | {fmt(r['f1_score'])} | {fmt(r['precision'])} | {fmt(r['recall'])} | "
            f"{fmt(r['roc_auc'])} |"
        )
    others = [r for r in rows if r["version"] != best["version"]]
    deltas = ", ".join(
        f"v{r['version']}: {fmt((best['cv_f1_mean'] or 0) - (r['cv_f1_mean'] or 0))}" for r in others
    ) or "-"
    lines += [
        "",
        f"**Versi terbaik berdasarkan `{train.SELECTION_METRIC}`: v{best['version']}** "
        f"({best['run_name']}, {train.SELECTION_METRIC}={fmt(best['cv_f1_mean'])}, "
        f"f1_score={fmt(best['f1_score'])}).",
        f"Selisih {train.SELECTION_METRIC} terhadap versi lain: {deltas}.",
        "",
        "Catatan analisis (lengkapi di laporan): jelaskan trade-off cv_f1_mean vs f1_score uji, "
        "kestabilan (cv_f1_std), dan keterbatasan jumlah data/kelas positif.",
    ]
    Path(COMPARISON_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(COMPARISON_PATH).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\n[SUCCESS] Perbandingan disimpan: {COMPARISON_PATH}")


# --------------------------------------------------------------------------- #
# promote: None -> Staging -> Production (stage API + alias + tag + log)
# --------------------------------------------------------------------------- #
def append_transition_log(name, version, old_stage, new_stage, methods, note) -> None:
    path = Path(TRANSITIONS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# Riwayat Transisi Stage Model\n\n"
            "| Waktu | Model | Versi | Dari | Ke | Mekanisme | Catatan |\n"
            "|---|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    with path.open("a", encoding="utf-8") as f:
        f.write(f"| {now_iso()} | {name} | v{version} | {old_stage} | {new_stage} | {methods} | {note} |\n")


def cmd_promote(args) -> None:
    client, name = setup(args)
    stage = args.stage
    mv = client.get_model_version(name, str(args.version))
    old_stage = getattr(mv, "current_stage", "None") or "None"
    methods = []

    # 1) API stage klasik (deprecated sejak MLflow 2.9; dipertahankan karena diminta di LK-07)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            client.transition_model_version_stage(
                name, str(args.version), stage=stage, archive_existing_versions=(stage == "Production")
            )
        methods.append("stage-api")
    except Exception as exc:  # API dapat dihapus pada rilis MLflow berikutnya
        print(f"[WARN] API stage tidak tersedia/ditolak ({type(exc).__name__}): {exc}")

    # 2) Alias (cara yang dianjurkan MLflow 3): staging / production
    if stage in ("Staging", "Production"):
        client.set_registered_model_alias(name, stage.lower(), str(args.version))
        methods.append(f"alias:{stage.lower()}")
        if stage == "Production":
            # Versi yang naik ke Production tidak lagi berstatus Staging
            try:
                holder = client.get_model_version_by_alias(name, "staging")
                if str(holder.version) == str(args.version):
                    client.delete_registered_model_alias(name, "staging")
            except Exception:
                pass  # alias staging memang tidak ada / sudah dihapus

    # 3) Tag agar status selalu terbaca di UI dan riwayat
    client.set_model_version_tag(name, str(args.version), "stage", stage)
    client.set_model_version_tag(name, str(args.version), "stage_changed_at", now_iso())
    methods.append("tag")

    append_transition_log(name, args.version, old_stage, stage, "+".join(methods), args.note)
    print(f"[SUCCESS] {name} v{args.version}: {old_stage} -> {stage} via {'+'.join(methods)}")
    print(f"Riwayat transisi: {TRANSITIONS_PATH}")


# --------------------------------------------------------------------------- #
# verify: simulasi kesiapan inferensi
# --------------------------------------------------------------------------- #
def cmd_verify(args) -> None:
    client, name = setup(args)
    uri = args.uri or f"models:/{name}@production"

    start = time.perf_counter()
    try:
        model = mlflow.pyfunc.load_model(uri)
    except Exception as exc:
        legacy_uri = f"models:/{name}/Production"
        print(f"[WARN] Gagal memuat {uri} ({type(exc).__name__}); mencoba {legacy_uri}")
        uri = legacy_uri
        model = mlflow.pyfunc.load_model(uri)
    load_seconds = time.perf_counter() - start
    print(f"[OK] Model dimuat dari registry: {uri} ({load_seconds:.2f} detik)")

    df = train.load_dataset(args.data_path)
    x, _ = train.build_features_and_labels(df)
    sample = x.head(args.n)

    start = time.perf_counter()
    predictions = model.predict(sample)
    batch_ms = (time.perf_counter() - start) * 1000

    result = pd.DataFrame(
        {"coin_id": df.loc[sample.index, "id"].values, "prediction": list(predictions)}
    )
    print(result.to_string(index=False))

    new_row = features_from_market(args.price, args.volume, args.high, args.low)
    start = time.perf_counter()
    single = model.predict(new_row)
    single_ms = (time.perf_counter() - start) * 1000
    print(
        f"\nData baru (price={args.price}, volume={args.volume}, high={args.high}, low={args.low}) "
        f"-> prediksi: {list(single)[0]} (is_suspicious)"
    )

    assert len(predictions) == len(sample), "Jumlah prediksi tidak sama dengan jumlah input"
    assert set(int(p) for p in predictions) <= {0, 1}, "Prediksi di luar {0,1}"
    print(
        f"\n[SUCCESS] Inferensi siap: load={load_seconds:.2f}s, "
        f"batch {len(sample)} baris={batch_ms:.1f}ms, 1 baris={single_ms:.1f}ms"
    )


# --------------------------------------------------------------------------- #
# publish: ekspor model, lacak dengan DVC, tulis metadata YAML
# --------------------------------------------------------------------------- #
def cmd_publish(args) -> None:
    client, name = setup(args)
    if args.version:
        mv = client.get_model_version(name, str(args.version))
    else:
        mv = client.get_model_version_by_alias(name, "production")
    version = str(mv.version)
    details = run_details(client, mv.run_id)

    export_path = Path(EXPORT_DIR)
    if export_path.exists():
        shutil.rmtree(export_path)
    export_path.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}/{version}", dst_path=str(export_path))
    print(f"[OK] Model v{version} diekspor ke {EXPORT_DIR}")

    model_pointer = f"{EXPORT_DIR}.dvc"
    try:
        subprocess.run([sys.executable, "-m", "dvc", "add", EXPORT_DIR], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"`dvc add {EXPORT_DIR}` gagal ({exc}). Pasang DVC dan jalankan manual: dvc add {EXPORT_DIR}"
        ) from exc

    aliases = list(getattr(mv, "aliases", []) or [])
    if "production" in aliases:
        stage = "Production"
    elif "staging" in aliases:
        stage = "Staging"
    else:
        stage = getattr(mv, "current_stage", "None") or "None"
    metadata = {
        "registered_model": name,
        "version": int(version),
        "stage": stage,
        "aliases": aliases,
        "inference_uri": f"models:/{name}@production",
        "run_id": mv.run_id,
        "run_name": details["name"],
        "selection_metric": train.SELECTION_METRIC,
        "metrics": {k: round(float(v), 6) for k, v in details["metrics"].items()},
        "parameters": details["params"],
        "data_lineage": {
            "dataset_path": details["tags"].get("data_path", args.data_path),
            "dataset_sha256": details["tags"].get("data_sha256", "-"),
            "dvc_pointer": DATA_DVC_POINTER,
            "dvc_md5": read_dvc_md5(DATA_DVC_POINTER),
        },
        "model_artifact": {
            "path": EXPORT_DIR,
            "dvc_pointer": model_pointer,
            "dvc_md5": read_dvc_md5(model_pointer),
        },
        "git_commit": train.get_git_commit(),
        "published_at": now_iso(),
    }
    Path(REGISTRY_METADATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_METADATA_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False, allow_unicode=True)

    print(f"[SUCCESS] Metadata model disimpan: {REGISTRY_METADATA_PATH}")
    print("Langkah berikut:")
    print("  dvc push")
    print(f"  git add {model_pointer} models/.gitignore {REGISTRY_METADATA_PATH} {DATA_DVC_POINTER} docs/")
    print('  git commit -m "feat: publish model registry metadata (LK-07)"')


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LK-07 Model Registry & Inferensi")
    parser.add_argument("--model-name", default=train.REGISTERED_MODEL_NAME)
    parser.add_argument("--data-path", default=train.DEFAULT_DATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.3)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("describe", help="Lengkapi deskripsi & tag model dan semua versi").set_defaults(func=cmd_describe)

    nv = sub.add_parser("new-version", help="Latih varian baru dan daftarkan sebagai versi baru")
    nv.add_argument("--run-name", default="rf-tuned-v2")
    nv.add_argument("--n-estimators", type=int, default=150)
    nv.add_argument("--max-depth", type=int, default=6)
    nv.add_argument("--min-samples-leaf", type=int, default=2)
    nv.add_argument("--class-weight", choices=["none", "balanced", "balanced_subsample"], default="balanced")
    nv.add_argument("--note", default="Penyetelan hyperparameter lanjutan (LK-07)")
    nv.set_defaults(func=cmd_new_version)

    sub.add_parser("compare", help="Bandingkan semua versi").set_defaults(func=cmd_compare)

    pr = sub.add_parser("promote", help="Ubah stage model (None -> Staging -> Production)")
    pr.add_argument("--version", type=int, required=True)
    pr.add_argument("--stage", choices=VALID_STAGES, required=True)
    pr.add_argument("--note", default="-")
    pr.set_defaults(func=cmd_promote)

    vf = sub.add_parser("verify", help="Uji muat model dari registry dan inferensi")
    vf.add_argument("--uri", default=None, help="Default: models:/<nama>@production")
    vf.add_argument("--n", type=int, default=5, help="Jumlah baris sampel dari dataset")
    vf.add_argument("--price", type=float, default=95000.0)
    vf.add_argument("--volume", type=float, default=3.2e10)
    vf.add_argument("--high", type=float, default=97000.0)
    vf.add_argument("--low", type=float, default=93500.0)
    vf.set_defaults(func=cmd_verify)

    pb = sub.add_parser("publish", help="Ekspor model, lacak dengan DVC, tulis metadata YAML")
    pb.add_argument("--version", type=int, default=None, help="Default: versi beralias production")
    pb.add_argument("--stage", default=None)
    pb.set_defaults(func=cmd_publish)
    return parser


if __name__ == "__main__":
    try:
        cli_args = build_parser().parse_args()
        cli_args.func(cli_args)
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}")
        sys.exit(1)