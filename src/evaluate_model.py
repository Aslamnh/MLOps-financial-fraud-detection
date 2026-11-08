"""
Validasi otomatis model hasil pelatihan terhadap ambang batas (threshold).

Membaca metrik dari models/best_model_metadata.json (ditulis oleh train.py) dan
membandingkannya dengan config/thresholds.json. Keluar dengan kode 1 jika ada
metrik yang tidak memenuhi syarat, sehingga pipeline berhenti sebelum registry.

Pemakaian:
    python src/evaluate_model.py
    python src/evaluate_model.py --metadata models/best_model_metadata.json --thresholds config/thresholds.json
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

DEFAULT_METADATA = os.path.join("models", "best_model_metadata.json")
DEFAULT_THRESHOLDS = os.path.join("config", "thresholds.json")
REPORT_PATH = os.path.join("models", "evaluation_report.json")


def check_thresholds(metrics: dict, minimum: dict) -> list:
    """Kembalikan daftar hasil per metrik: nama, nilai, minimum, lulus/tidak."""
    results = []
    for name, threshold in minimum.items():
        value = metrics.get(name)
        valid = isinstance(value, (int, float)) and not math.isnan(float(value))
        passed = bool(valid and float(value) >= float(threshold))
        results.append(
            {
                "metric": name,
                "value": float(value) if valid else None,
                "minimum": float(threshold),
                "passed": passed,
            }
        )
    return results


def format_table(results: list) -> str:
    lines = ["| Metrik | Nilai | Minimum | Status |", "|---|---|---|---|"]
    for r in results:
        value = "tidak ada" if r["value"] is None else f"{r['value']:.4f}"
        status = "LULUS" if r["passed"] else "GAGAL"
        lines.append(f"| {r['metric']} | {value} | {r['minimum']:.4f} | {status} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validasi model terhadap threshold")
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    args = parser.parse_args()

    with open(args.metadata, encoding="utf-8") as f:
        metadata = json.load(f)
    with open(args.thresholds, encoding="utf-8") as f:
        minimum = json.load(f)["minimum"]

    results = check_thresholds(metadata.get("metrics", {}), minimum)
    all_passed = all(r["passed"] for r in results)

    table = format_table(results)
    print(f"Model   : {metadata.get('registered_model_name')} (run {metadata.get('run_name')}, id {metadata.get('run_id')})")
    print(table)

    report = {"run_id": metadata.get("run_id"), "passed": all_passed, "results": results}
    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    summary_file = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_file:
        verdict = "LULUS validasi" if all_passed else "GAGAL validasi"
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(f"### Evaluasi model: {verdict}\n\n{table}\n")

    if not all_passed:
        failed = ", ".join(r["metric"] for r in results if not r["passed"])
        print(f"[GAGAL] Model tidak memenuhi threshold: {failed}. Model tidak didaftarkan ke registry.")
        return 1
    print("[LULUS] Seluruh metrik memenuhi threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
