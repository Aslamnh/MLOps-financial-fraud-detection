"""Uji unit untuk logika pelatihan dan validasi. Tidak memerlukan data DVC maupun jaringan."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

import evaluate_model
import train


def make_dataframe(n: int = 40, with_outlier: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "current_price": rng.uniform(1, 100000, n),
            "total_volume": rng.uniform(1e6, 1e9, n),
            "high_low_spread": rng.uniform(0.0, 0.2, n),
            "z_score_volume": rng.normal(0, 0.5, n),
            "price_change_percentage_24h": rng.normal(0, 1, n),
        }
    )
    if with_outlier:
        df.loc[0, "z_score_volume"] = 5.0
    return df


def test_labels_flag_volume_outlier():
    features, label = train.build_features_and_labels(make_dataframe())
    assert label.iloc[0] == 1
    assert set(label.unique()) <= {0, 1}


def test_features_have_expected_columns_and_no_nan():
    features, _ = train.build_features_and_labels(make_dataframe())
    assert list(features.columns) == train.FEATURE_COLUMNS
    assert not features.isna().any().any()


def test_no_outlier_gives_no_label_from_volume():
    df = make_dataframe(with_outlier=False)
    df["z_score_volume"] = 0.0
    _, label = train.build_features_and_labels(df)
    assert label.sum() <= 3  # hanya sisa dari aturan perubahan harga


def test_load_dataset_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        train.load_dataset(str(tmp_path / "tidak_ada.csv"))


def test_load_dataset_missing_columns(tmp_path):
    path = tmp_path / "rusak.csv"
    pd.DataFrame({"current_price": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        train.load_dataset(str(path))


def test_load_dataset_drops_missing_rows(tmp_path):
    df = make_dataframe(10)
    df.loc[2, "total_volume"] = np.nan
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    assert len(train.load_dataset(str(path))) == 9


def test_evaluate_returns_expected_metrics():
    features, label = train.build_features_and_labels(make_dataframe(80))
    label.iloc[:10] = 1  # pastikan kedua kelas ada
    label.iloc[10:20] = 0
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(features, label)
    metrics = train.evaluate(model, features, label)
    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_run_configs_are_valid():
    assert len(train.RUN_CONFIGS) >= 3
    names = [c["run_name"] for c in train.RUN_CONFIGS]
    assert len(names) == len(set(names))


def test_threshold_check_passes_and_fails():
    minimum = {"recall": 0.4, "roc_auc": 0.8}
    ok = evaluate_model.check_thresholds({"recall": 0.5, "roc_auc": 0.9}, minimum)
    assert all(r["passed"] for r in ok)
    bad = evaluate_model.check_thresholds({"recall": 0.2, "roc_auc": 0.9}, minimum)
    assert [r["metric"] for r in bad if not r["passed"]] == ["recall"]


def test_threshold_check_missing_or_nan_metric_fails():
    results = evaluate_model.check_thresholds({"recall": float("nan")}, {"recall": 0.4, "roc_auc": 0.8})
    assert not any(r["passed"] for r in results)


def test_threshold_config_file_is_valid():
    path = Path(__file__).resolve().parent.parent / "config" / "thresholds.json"
    minimum = json.loads(path.read_text(encoding="utf-8"))["minimum"]
    assert minimum
    assert all(0.0 <= v <= 1.0 for v in minimum.values())
