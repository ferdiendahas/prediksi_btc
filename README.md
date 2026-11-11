# Prediksi Harga BTC/IDR dengan Model Boosting

Kode percobaan skripsi: membandingkan **XGBoost, LightGBM, CatBoost, dan AdaBoost** untuk memprediksi harga BTC/IDR (Indodax) beberapa menit ke depan, dengan berbagai metode optimasi, rasio pembagian data, dan time frame.

Daftar lengkap semua variabel percobaan ada di **`DAFTAR_PERCOBAAN.md`**. Perubahan versi v4 (revisi dosen 11-11-2026) ada di **`CHANGELOG.md`**.

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
│   └── btcidr_candle_1menit_<stamp>.csv   (candle 5 menit dibentuk dari file ini)
└── hasil/                 <- dibuat otomatis
    ├── hasil_validasi.csv     <- skor validasi per run (dipakai memilih Tahap 1-2)
    ├── hasil_uji_final.csv    <- metrik blok uji per run (dibuka setelah dikunci)
    ├── deskripsi_data.csv     <- deskripsi data per (dataset, horizon)
    ├── keputusan_terkunci.json<- pilihan Tahap 1-2 + hash validasi
    ├── lingkungan.txt         <- Python, pip freeze, CPU, RAM, OS
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
- (ticker tidak dipakai lagi di v4)

Ini format bawaan `indodax ekspor` dari collector. Kalau nama file berbeda, ubah `DATA` di `config.py`.

---

## 3. Cara menjalankan

### Semua kombinasi di `config.py`

```bash
python run.py
```

### Argumen

| Argumen | Isi | Contoh |
|---|---|---|
| `--datasets` | `candle_1m`, `candle_5m` | `--datasets candle_1m` |
| `--horizons` | horizon dalam menit | `--horizons 5 15 30 60` |
| `--train-fracs` | porsi akhir data pengembangan untuk melatih (bawaan 1.0) | `--train-fracs 1.0 0.75 0.5` |
| `--models` | `XGBoost LightGBM CatBoost AdaBoost` | `--models XGBoost CatBoost` |
| `--optimizers` | `default grid random tpe cmaes ga qmc` | `--optimizers tpe ga` |
| `--trials` | jumlah trial tuning | `--trials 50` |
| `--folds` | jumlah fold CV | `--folds 5` |
| `--pruner` | `median`, `hyperband`, `none` | `--pruner median` |
| `--early-stop` | aktifkan early stopping | `--early-stop` |
| `--top-k` | seleksi K fitur teratas (di dalam tiap fold) | `--top-k 15` |
| `--loss` | `l1` atau `l2` | `--loss l1` |
| `--harga-candle` | `typical`, `close`, `ohlc4` | `--harga-candle close` |
| `--start`, `--end` | periode data (UTC) | `--start "2026-10-14 00:00"` |
| `--tag` | label tahap (`tahap1`, `tahap2`, `tahap3`, `tahap4_*`) | `--tag tahap1` |

### Pembagian data

- **Blok uji final** = 20% bar terakhir (`FINAL_TEST_FRAC`), mulai waktu `t_u`. Sama untuk semua run dengan dataset dan horizon yang sama.
- **Data pengembangan** = baris yang waktu targetnya (t + horizon) sebelum `t_u` (*purging* berbasis waktu).
- Tuning memakai walk-forward CV 5 fold di data pengembangan. Di tiap fold, baris latih yang waktu targetnya tidak sebelum awal validasi dibuang.
- Objektif tuning = **MAE harga (rupiah)** rata-rata fold. Skor pembanding Naive di fold yang sama ikut dicatat (`cv_vs_naive_pct`).

### Mengulang dari awal

Hapus atau ganti nama folder `hasil/`. Selama `hasil/hasil_uji_final.csv` masih ada, run dengan konfigurasi sama persis **dan `--tag` sama** akan dilewati.

---

## 4. Urutan percobaan (aturan v4)

```bash
python run.py --optimizers tpe --tag tahap1                       # 9 pasangan x 4 model
python ringkas.py --tahap 1                                       # -> DS, H (dari validasi)
python run.py --datasets <DS> --horizons <H> \
  --optimizers default grid random tpe cmaes ga qmc --tag tahap2
python ringkas.py --tahap 2                                       # -> OPT, tulis keputusan_terkunci.json
python run.py --datasets <DS> --horizons <H> --optimizers <OPT> --train-fracs 1.0 0.75 0.5 --tag tahap3
for t in "--early-stop" "--top-k 15" "--loss l1" "--pruner median"; do
  python run.py --datasets <DS> --horizons <H> --optimizers <OPT> $t --tag "tahap4_$(echo $t | tr -d ' -')"
done
python ringkas.py --uji-final                                     # baru membuka hasil_uji_final.csv
```

- **Tahap 1:** pasangan (dataset, horizon) dengan `cv_vs_naive_pct` terkecil di antara 4 model. Seri → horizon lebih pendek → `candle_1m`.
- **Tahap 2:** strategi dengan rata-rata `cv_vs_naive_pct` 4 model terkecil. `default` boleh terpilih. Seri → waktu tuning terpendek.
- **Uji final:** koreksi Holm per tahap pada `p_MAE`. Status `lebih_baik` / `lebih_buruk` / `tidak_berbeda`.
- `--early-stop` dan `--loss l1` tidak berlaku untuk AdaBoost (dicatat `status=tidak_berlaku`).
- Log `run.py` hanya menampilkan skor validasi; metrik blok uji dibaca lewat `ringkas.py --uji-final`.

---

## 5. Perkiraan waktu

Diukur untuk data seminggu, 30 trial, 5 fold. Angka kasar, tergantung jumlah core CPU.

| Data | Per run XGBoost/LightGBM | Per run CatBoost | Per run AdaBoost | Grid (27 kombinasi) |
|---|---|---|---|---|
| `candle_5m` (±2.000 baris) | < 1 menit | 1–2 menit | 1–3 menit | 1–5 menit |
| `candle_1m` (±10.000 baris) | 1–3 menit | 2–5 menit | 5–15 menit | 5–20 menit |

Tahap 1 (9 dataset-horizon × 4 model × TPE) sekitar 1–3 jam. Full factorial bisa 1–2 hari, jadi jalankan di Mac yang dibiarkan menyala atau dicicil per tahap.

---

## 6. Catatan metodologi (penting untuk Bab 3)

1. **Target berupa return, bukan harga mentah.** Model pohon tidak bisa memprediksi di luar rentang data latih. Prediksi return lalu dikonversi ke rupiah: `harga_prediksi = harga_sekarang × exp(return_prediksi)`.
2. **Tidak ada kebocoran data.**
   - Fitur hanya memakai data sampai waktu t.
   - Tuning hanya di data pengembangan dengan walk-forward CV (`TimeSeriesSplit`) dan *purging* berbasis waktu di tiap fold.
   - Baris yang waktu targetnya (t + horizon) masuk blok uji dibuang dari data pengembangan.
   - Seleksi fitur top-k dihitung di dalam tiap fold, bukan di seluruh data.
   - Pemilihan Tahap 1-2 hanya memakai skor validasi; blok uji dibuka setelah pilihan dikunci.
3. **Naive wajib jadi pembanding.** Untuk horizon pendek, Naive (harga tetap) sangat sulit dikalahkan.
4. **Time frame optimal dinilai dari keunggulan atas Naive (%),** bukan MAE rupiah, karena MAE pasti membesar untuk horizon lebih jauh.
5. **Metode optimasi berbasis acak memakai seed yang sama.** Random, TPE, CMA-ES, dan GA mengawali dengan beberapa trial acak yang sama. Dengan trial sedikit (< 10), hasilnya bisa identik. Pakai minimal 30 trial supaya perbedaan metode terlihat.
6. **Grid Search punya jumlah trial sendiri (27)**, karena ditentukan oleh jumlah kombinasi grid.
7. **AdaBoost tidak memakai pengaturan `LOSS` dan `EARLY_STOP`.** AdaBoost.R2 punya parameter `loss` sendiri (linear/square/exponential) yang sudah ikut dituning.
8. **Hasil tergantung periode.** Satu minggu hanya mewakili kondisi pasar minggu itu. Untuk uji ketahanan, ganti `--start`/`--end` ke minggu lain.
