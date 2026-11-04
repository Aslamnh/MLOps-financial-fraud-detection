# Perbandingan Versi Model `fraud-detection-rf`

Dibuat: 2026-11-04 02:24:28 UTC

| Versi | Run | Stage | Alias | n_est | max_depth | leaf | class_weight | cv_f1_mean | cv_f1_std | f1 | precision | recall | roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 | rf-balanced-deep | None | - | 200 | 8 | 1 | balanced | 0.3667 | 0.3055 | 0.4000 | 0.3333 | 0.5000 | 0.9643 |
| v2 | rf-tuned-v2 | None | - | 300 | 10 | 1 | balanced | 0.3333 | 0.2789 | 0.5000 | 0.5000 | 0.5000 | 0.9643 |

**Versi terbaik berdasarkan `cv_f1_mean`: v1** (rf-balanced-deep, cv_f1_mean=0.3667, f1_score=0.4000).
Selisih cv_f1_mean terhadap versi lain: v2: 0.0333.

Catatan analisis (lengkapi di laporan): jelaskan trade-off cv_f1_mean vs f1_score uji, kestabilan (cv_f1_std), dan keterbatasan jumlah data/kelas positif.
