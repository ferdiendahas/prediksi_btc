# Daftar Variabel Percobaan

Penelitian: prediksi harga BTC/IDR (Indodax) beberapa menit ke depan dengan empat model boosting, untuk mencari kombinasi model, optimasi, pembagian data, dan time frame yang paling akurat.

Semua pilihan di bawah diatur di `config.py` atau lewat argumen `run.py`.

---

## 1. Data

| Kode | Sumber | Resolusi | Kolom | Jumlah bar seminggu |
|---|---|---|---|---|
| `candle_1m` | Candle OHLCV Indodax | 1 menit | open, high, low, close, volume | ±10.080 |
| `candle_5m` | Agregasi candle 1 menit (open pertama, high maks, low min, close terakhir, volume jumlah) | 5 menit | open, high, low, close, volume | ±2.016 |

`ticker_5m` tidak dipakai lagi sejak revisi dosen 11-11-2026 (Indodax tidak menyediakan candle 5 menit, jadi candle 5 menit dibentuk dari candle 1 menit). Bar 5 menit yang datanya kurang dari 5 menit dicatat sebagai `n_parsial` di `hasil/deskripsi_data.csv`.

**Periode:** 14 Oktober 2026 00:00 – 20 Oktober 2026 23:59 UTC (7 hari penuh). Bisa diganti di `START_UTC` dan `END_UTC`.

**Definisi harga** (`HARGA` di config):

| Data | Pilihan | Keterangan |
|---|---|---|
| candle | `typical` (bawaan) | (high + low + close) / 3 |
| candle | `close` | Harga penutupan |
| candle | `ohlc4` | (open + high + low + close) / 4 |
| ticker | `mid` (bawaan) | (bid + ask) / 2 |
| ticker | `last` | Harga transaksi terakhir |

> Kenapa bawaannya bukan `close`/`last`: harga transaksi terakhir meloncat-loncat antara harga bid dan ask (*bid-ask bounce*). Di data 1 menit, autokorelasi return `close` = −0,30, sedangkan `typical` = −0,13. Pola loncatan ini bisa "ditebak" model, tetapi bukan pergerakan harga yang sebenarnya dan tidak bisa dimanfaatkan trader karena terpotong spread. `close` dan `last` tetap bisa dipakai sebagai uji sensitivitas.

---

## 2. Time frame prediksi (horizon)

Horizon = berapa menit ke depan harga diprediksi.

| Horizon | `candle_1m` | `candle_5m` | Contoh pengguna |
|---|---|---|---|
| 1 menit | ✅ | ✗ (bukan kelipatan 5) | Scalper sangat cepat |
| 5 menit | ✅ | ✅ | Scalper |
| 15 menit | ✅ | ✅ | Scalper / intraday |
| 30 menit | ✅ | ✅ | Intraday |
| 60 menit | ✅ | ✅ | Intraday / swing pendek |

Horizon lain (misal 10, 20, 45, 120) bisa ditambahkan di `HORIZONS`, asal kelipatan resolusi data.

**Cara kerja:** data tidak di-*resample*. Setiap baris tetap satu bar (1 atau 5 menit), dan targetnya adalah return dari sekarang sampai h menit ke depan. Dengan begitu, jumlah baris tetap banyak untuk semua horizon. Contoh: horizon 60 menit di data 1 menit tetap punya ±10 ribu baris, bukan 168.

---

## 3. Pembagian data (berbasis waktu)

| Bagian | Isi |
|---|---|
| Blok uji final | 20% bar terakhir per (dataset, horizon), mulai waktu `t_u`. Tetap untuk semua run |
| Data pengembangan | Baris dengan waktu target (t + horizon) < `t_u` |
| Tahap 3 (`frac_latih`) | 1.0 / 0.75 / 0.5 = porsi **akhir** data pengembangan yang dipakai melatih. Blok uji tidak berubah |

- Pembagian berurutan waktu: data pengembangan selalu lebih awal dari blok uji.
- *Purging* berbasis waktu, juga di tiap fold walk-forward CV.

---

## 4. Model

| Model | Library | Jenis |
|---|---|---|
| XGBoost | `xgboost` | Gradient boosting, pertumbuhan pohon per level |
| LightGBM | `lightgbm` | Gradient boosting, pertumbuhan pohon per daun (*leaf-wise*) |
| CatBoost | `catboost` | Gradient boosting, *ordered boosting*, pohon simetris |
| AdaBoost | `scikit-learn` (AdaBoost.R2) | Boosting klasik dengan pembobotan ulang sampel |
| **Naive** (pembanding) | — | Harga h menit lagi = harga sekarang |

---

## 5. Optimasi

### 5a. Metode pencarian hyperparameter (`OPTIMIZERS`)

| Kode | Metode | Kelompok | Cara kerja singkat | Jumlah percobaan |
|---|---|---|---|---|
| `default` | Tanpa tuning | Pembanding | Setelan bawaan library | 1 |
| `grid` | Grid Search | Pencarian menyeluruh | Coba semua kombinasi dari daftar nilai tetap | 27 (3×3×3) |
| `random` | Random Search | Pencarian acak | Ambil kombinasi secara acak dari rentang | `N_TRIALS` |
| `tpe` | Bayesian Optimization (TPE) | Berbasis model probabilistik | Belajar dari trial sebelumnya untuk memilih kombinasi berikutnya | `N_TRIALS` |
| `cmaes` | CMA-ES | Evolusioner (*evolution strategy*) | Menggeser dan menyesuaikan sebaran pencarian ke arah nilai yang lebih baik | `N_TRIALS` |
| `ga` | Genetic Algorithm (NSGA-II) | Evolusioner (genetika) | Seleksi, *crossover*, dan mutasi antar generasi | `N_TRIALS` |
| `qmc` | Quasi-Monte Carlo (Sobol) | Pencarian acak terstruktur | Titik pencarian disebar merata di seluruh rentang | `N_TRIALS` |

Semua metode (kecuali `default`) memakai fungsi tujuan yang sama: **rata-rata MAE return pada walk-forward CV di data latih**. Data uji tidak pernah dipakai saat tuning.

### 5b. Teknik optimasi tambahan (bisa dikombinasikan dengan metode di atas)

| Pengaturan | Pilihan | Fungsi |
|---|---|---|
| `N_TRIALS` | angka (bawaan 30) | Jatah percobaan tuning, sama untuk semua metode |
| `N_FOLDS` | angka (bawaan 5) | Jumlah fold walk-forward CV |
| `PRUNER` | `None`, `median`, `hyperband` | Menghentikan trial yang jelek lebih awal sehingga tuning lebih cepat |
| `EARLY_STOP` | `True`/`False` | Jumlah pohon ditentukan otomatis: berhenti saat error validasi tidak membaik 50 pohon berturut-turut (XGBoost, LightGBM, CatBoost) |
| `TOP_K` | `None` atau angka | Seleksi fitur: hanya pakai K fitur dengan *importance* tertinggi (dihitung dari data latih) |
| `LOSS` | `l2`, `l1` | Fungsi loss saat training: squared error atau absolute error |

### 5c. Ruang pencarian hyperparameter

| Model | Parameter dan rentang (random/tpe/cmaes/ga/qmc) | Grid Search |
|---|---|---|
| XGBoost | n_estimators 10–1000, max_depth 1–8, learning_rate 0,001–0,3, min_child_weight 1–100, subsample 0,5–1, colsample_bytree 0,3–1, reg_lambda 0,001–100, reg_alpha 0,001–10 | n_estimators {50, 200, 500}, max_depth {2, 4, 6}, learning_rate {0,001; 0,01; 0,1} |
| LightGBM | n_estimators 10–1000, num_leaves 2–63, max_depth 1–8, learning_rate 0,001–0,3, min_child_samples 5–300, subsample 0,5–1, colsample_bytree 0,3–1, reg_lambda 0,001–100, reg_alpha 0,001–10 | n_estimators {50, 200, 500}, num_leaves {4, 15, 31}, learning_rate {0,001; 0,01; 0,1} |
| CatBoost | iterations 10–1000, depth 1–8, learning_rate 0,001–0,3, l2_leaf_reg 0,1–100, random_strength 0,001–10, bagging_temperature 0–1 | iterations {50, 200, 500}, depth {2, 4, 6}, learning_rate {0,001; 0,01; 0,1} |
| AdaBoost | n_estimators 10–500, learning_rate 0,001–1, loss {linear, square, exponential}, max_depth pohon dasar 1–6 | n_estimators {50, 150, 300}, max_depth {2, 3, 5}, learning_rate {0,01; 0,1; 1} |

---

## 6. Fitur (42 fitur untuk candle 1 menit dan candle 5 menit)

| Kelompok | Fitur | Data |
|---|---|---|
| Return masa lalu | Return 12 bar terakhir | Keduanya |
| Statistik bergulir (5, 15, 30, 60, 240 menit) | Jumlah return, standar deviasi return, porsi bar tanpa perubahan harga | Keduanya |
| Candle | Range (high−low)/close, posisi close dalam bar, log volume, rata-rata range bergulir, rasio volume terhadap rata-ratanya | candle |
| Ticker | Spread relatif, posisi harga dalam range 24 jam, perubahan volume 24 jam | ticker |
| Waktu | Jam dalam sehari (sin dan cos) | Keduanya |

---

## 7. Evaluasi

| Metrik | Arti |
|---|---|
| MAE (Rp) | Rata-rata selisih absolut prediksi dan harga aktual |
| RMSE (Rp) | Seperti MAE, tetapi kesalahan besar dihukum lebih berat |
| MAPE (%) | Rata-rata kesalahan dalam persen harga |
| Akurasi arah (%) | Seberapa sering arah naik/turun tepat. Bar dengan selisih harga < Rp1 tidak dihitung. Pembanding: `arah_baseline_pct` = tebak arah dengan sign(ret_lag1) |
| Uji Holm | Koreksi uji berganda per tahap pada p_MAE (`ringkas.py --uji-final`) |
| MAE/RMSE vs Naive (%) | Selisih terhadap Naive. **Negatif = lebih akurat dari Naive** |
| Uji Diebold-Mariano | Apakah selisih dengan Naive signifikan (loss absolut dan kuadrat, koreksi HLN, autokovarians sampai lag h−1) |

**Cara menentukan time frame optimal:** jangan pakai MAE rupiah mentah, karena MAE pasti naik untuk horizon yang lebih jauh (harga sempat bergerak lebih jauh). Bandingkan **MAE vs Naive (%)** dan signifikansinya. Time frame optimal adalah horizon dengan keunggulan terbesar atas Naive yang signifikan.

---

## 8. Jumlah kombinasi

| Variabel | Jumlah pilihan |
|---|---|
| Dataset | 2 |
| Horizon | 5 (candle_1m) + 4 (candle_5m) = 9 pasangan dataset-horizon |
| Panjang data latih (Tahap 3) | 3 |
| Model | 4 |
| Metode optimasi | 6 (+1 opsional `qmc`) |
| **Total run (full factorial)** | 9 × 3 × 4 × 6 = **648 run** |

Ditambah variasi teknik tambahan (early stopping, pruning, seleksi fitur, loss L1), jumlahnya bisa berlipat. Karena itu percobaan disarankan bertahap (lihat README bagian "Urutan percobaan").
