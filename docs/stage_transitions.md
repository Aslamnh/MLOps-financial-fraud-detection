# Riwayat Transisi Stage Model

| Waktu | Model | Versi | Dari | Ke | Mekanisme | Catatan |
|---|---|---|---|---|---|---|
| 2026-11-04 02:26:48 UTC | fraud-detection-rf | v1 | None | Staging | stage-api+alias:staging+tag | - |
| 2026-11-04 02:27:01 UTC | fraud-detection-rf | v2 | None | Staging | stage-api+alias:staging+tag | - |
| 2026-11-04 02:30:17 UTC | fraud-detection-rf | v1 | Staging | Production | stage-api+alias:production+tag | cv_f1_mean lebih tinggi dan menjadi metrik utama, perbedaan performa metrik lain sangat kecil sehingga model yang lebih sederhana lebih cepat dan efisien. Namun, cv_f1_mean lebih tinggi dan menjadi metrik utama, perbedaan performa metrik lain sangat kecil sehingga model yang lebih sederhana lebih cepat dan efisien. Namun, v2 hanya unggul tipis pada f1 dan precision uji; recall dan ROC-AUC identik. |
