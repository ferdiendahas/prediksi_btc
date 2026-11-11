# Prediksi Harga BTC/IDR dengan Model Boosting

Kode percobaan skripsi: membandingkan **XGBoost, LightGBM, CatBoost, dan AdaBoost** untuk memprediksi harga BTC/IDR (Indodax) beberapa menit ke depan, dengan berbagai metode optimasi, rasio pembagian data, dan time frame.

Daftar lengkap semua variabel percobaan ada di **`DAFTAR_PERCOBAAN.md`**.

---

## 1. Struktur folder

```
prediksi_btc/
├── config.py              <- SEMUA pengaturan percobaan diubah di sini
├── run.py                 <- menjalankan percobaan
├── ringkas.py             <- membuat tabel perbandingan dari hasil
├── requirements.txt
├── DAFTAR_PERCOBAAN.md    <- daftar model, optimasi, split, time frame
├── btcpred/
│   ├── data.py            <- baca CSV, definisi harga, fitur, target
│   ├── models.py          <- 4 model, ruang hyperparameter, grid
│   ├── optim.py           <- metode optimasi (grid, random, TPE, CMA-ES, GA, ...)
│   └── evaluate.py        <- metrik, uji Diebold-Mariano, seleksi fitur
├── data/
│   ├── btcidr_candle_1menit.csv
│   └── btcidr_ticker_5menit.csv
└── hasil/                 <- dibuat otomatis
    ├── hasil_semua.csv    <- satu baris per run (semua metrik)
    ├── params/            <- hyperparameter terbaik + daftar fitur per run
    ├── prediksi/          <- prediksi data uji per run
    └── ringkasan/         <- tabel dari ringkas.py
```

---

## 2. Setup

### Di Mac (pakai virtual environment)

```bash
cd prediksi_btc
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Kalau LightGBM error soal `libomp` di Mac: `brew install libomp`.

### Di Google Colab

1. Upload folder `prediksi_btc` (zip) ke Colab, lalu:
   ```python
   !unzip -q prediksi_btc.zip
   %cd prediksi_btc
   !pip install -q -r requirements.txt
   ```
2. Jalankan percobaan dengan `!python run.py ...` (contoh di bagian 3).
3. Colab bisa terputus kalau terlalu lama. Simpan folder `hasil/` ke Google Drive, atau jalankan per tahap. Run yang sudah selesai tidak diulang.

### Data

Taruh CSV di folder `data/` dengan nama:
- `btcidr_candle_1menit.csv` (kolom: waktu_utc, waktu_wib, open, high, low, close, volume)
- `btcidr_ticker_5menit.csv` (kolom: waktu_utc, waktu_wib, last_price, buy, sell, high_24h, low_24h, vol_btc_24h, vol_idr_24h)

Ini format bawaan `indodax ekspor` dari collector. Kalau nama file berbeda, ubah `DATA` di `config.py`.

---

## 3. Cara menjalankan

### Semua kombinasi di `config.py`

```bash
python run.py
```

### Mengganti variasi tanpa mengedit config (argumen menimpa config)

```bash
# hanya data 1 menit, horizon 5 dan 15 menit, metode TPE dan Random Search
python run.py --datasets candle_1m --horizons 5 15 --optimizers tpe random

# satu model, semua metode optimasi, 50 trial
python run.py --models CatBoost --optimizers default grid random tpe cmaes ga qmc --trials 50

# teknik tambahan: early stopping, seleksi 15 fitur, loss L1, pruning
python run.py --early-stop --top-k 15 --loss l1 --pruner median

# periode lain dan definisi harga lain
python run.py --start "2026-10-10 00:00" --end "2026-10-16 23:59" --harga-candle close

# beri label supaya mudah difilter saat meringkas
python run.py --optimizers tpe --splits 0.8 --tag tahap1
```

Semua argumen:

| Argumen | Isi | Contoh |
|---|---|---|
| `--datasets` | `candle_1m`, `ticker_5m` | `--datasets candle_1m` |
| `--horizons` | horizon dalam menit | `--horizons 5 15 30 60` |
| `--splits` | porsi data latih | `--splits 0.8 0.7 0.6` |
| `--models` | `XGBoost LightGBM CatBoost AdaBoost` | `--models XGBoost CatBoost` |
| `--optimizers` | `default grid random tpe cmaes ga qmc` | `--optimizers tpe ga` |
| `--trials` | jumlah trial tuning | `--trials 50` |
| `--folds` | jumlah fold CV | `--folds 5` |
| `--pruner` | `median`, `hyperband`, `none` | `--pruner median` |
| `--early-stop` | aktifkan early stopping | `--early-stop` |
| `--top-k` | seleksi K fitur teratas | `--top-k 15` |
| `--loss` | `l1` atau `l2` | `--loss l1` |
| `--harga-candle` | `typical`, `close`, `ohlc4` | `--harga-candle close` |
| `--harga-ticker` | `mid`, `last` | `--harga-ticker last` |
| `--start`, `--end` | periode data (UTC) | `--start "2026-10-14 00:00"` |
| `--tag` | label bebas | `--tag tahap1` |

### Meringkas hasil

```bash
python ringkas.py                                   # semua run
python ringkas.py --tag tahap1                      # hanya run berlabel tahap1
python ringkas.py --dataset candle_1m --split 80:20 --optimasi tpe
python ringkas.py --metrik RMSE_vs_naive_pct        # bandingkan dengan RMSE
```

### Mengulang dari awal

Hapus atau ganti nama folder `hasil/`. Selama file `hasil/hasil_semua.csv` masih ada, run dengan konfigurasi yang sama persis **dan `--tag` yang sama** akan dilewati. Konfigurasi sama tapi tag berbeda akan dijalankan lagi, supaya setiap tahap punya baris hasilnya sendiri.

---

## 4. Urutan percobaan yang disarankan

Full factorial ada 648 run (lihat `DAFTAR_PERCOBAAN.md`). Supaya hemat waktu dan alurnya jelas, jalankan bertahap. **Tetapkan urutan dan kriteria ini sebelum melihat hasil**, supaya pemilihan tidak terkesan mengikuti hasil yang paling enak dilihat.

| Tahap | Tujuan | Perintah |
|---|---|---|
| 1 | Cari **time frame** terbaik | `python run.py --optimizers tpe --splits 0.8 --tag tahap1` |
| 2 | Bandingkan **metode optimasi** di time frame terbaik | `python run.py --horizons <H> --optimizers default grid random tpe cmaes ga qmc --splits 0.8 --tag tahap2` |
| 3 | Uji **rasio split** | `python run.py --horizons <H> --optimizers <metode terbaik> --splits 0.8 0.7 0.6 --tag tahap3` |
| 4 | Uji **teknik tambahan** | `python run.py --horizons <H> --optimizers <metode terbaik> --splits 0.8 --early-stop --tag tahap4a`, lalu `--top-k 15`, `--loss l1`, `--pruner median` |
| 5 (opsional) | Full factorial | `python run.py --tag lengkap` |

**Kriteria "terbaik" untuk setiap tahap:** MAE vs Naive (%) paling negatif dengan uji DM signifikan (p < 0,05). Kalau tidak ada yang signifikan, ambil yang paling dekat ke Naive dan laporkan apa adanya.

---

## 5. Perkiraan waktu

Diukur untuk data seminggu, 30 trial, 5 fold. Angka kasar, tergantung jumlah core CPU.

| Data | Per run XGBoost/LightGBM | Per run CatBoost | Per run AdaBoost | Grid (27 kombinasi) |
|---|---|---|---|---|
| `ticker_5m` (±2.000 baris) | < 1 menit | 1–2 menit | 1–3 menit | 1–5 menit |
| `candle_1m` (±10.000 baris) | 1–3 menit | 2–5 menit | 5–15 menit | 5–20 menit |

Tahap 1 (9 dataset-horizon × 4 model × TPE) sekitar 1–3 jam. Full factorial bisa 1–2 hari, jadi jalankan di Mac yang dibiarkan menyala atau dicicil per tahap.

---

## 6. Catatan metodologi (penting untuk Bab 3)

1. **Target berupa return, bukan harga mentah.** Model pohon tidak bisa memprediksi di luar rentang data latih. Prediksi return lalu dikonversi ke rupiah: `harga_prediksi = harga_sekarang × exp(return_prediksi)`.
2. **Tidak ada kebocoran data.**
   - Fitur hanya memakai data sampai waktu t.
   - Tuning hanya di data latih dengan walk-forward CV (`TimeSeriesSplit`) dan jeda sebesar horizon.
   - h bar terakhir data latih dibuang (*purging*) karena targetnya melewati batas uji.
   - Data uji hanya dipakai sekali untuk laporan akhir.
3. **Naive wajib jadi pembanding.** Untuk horizon pendek, Naive (harga tetap) sangat sulit dikalahkan.
4. **Time frame optimal dinilai dari keunggulan atas Naive (%),** bukan MAE rupiah, karena MAE pasti membesar untuk horizon lebih jauh.
5. **Metode optimasi berbasis acak memakai seed yang sama.** Random, TPE, CMA-ES, dan GA mengawali dengan beberapa trial acak yang sama. Dengan trial sedikit (< 10), hasilnya bisa identik. Pakai minimal 30 trial supaya perbedaan metode terlihat.
6. **Grid Search punya jumlah trial sendiri (27)**, karena ditentukan oleh jumlah kombinasi grid.
7. **AdaBoost tidak memakai pengaturan `LOSS` dan `EARLY_STOP`.** AdaBoost.R2 punya parameter `loss` sendiri (linear/square/exponential) yang sudah ikut dituning.
8. **Hasil tergantung periode.** Satu minggu hanya mewakili kondisi pasar minggu itu. Untuk uji ketahanan, ganti `--start`/`--end` ke minggu lain.
