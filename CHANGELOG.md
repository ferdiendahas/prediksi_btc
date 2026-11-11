# CHANGELOG — `prediksi_btc` v4 (revisi dari review dosen 11-11-2026)

Branch `revisi-dosen-v4`, dibuat dari commit `fc06845` (kode P2). Hasil lama (`hasil/`, P1 dan P2) tidak disentuh.
Ruang pencarian, *seed* (42), jumlah *trial* (30; Grid 27), jumlah fold (5), model, dan fitur **tidak diubah**.

| Tanggal | Bagian | Berkas | Perubahan | Alasan (review dosen) |
|---|---|---|---|---|
| 2026-11-12 | K1 | `btcpred/data.py`, `config.py` | Fungsi `agregasi_5m()` (open pertama, high maks, low min, close terakhir, volume jumlah, `n_menit`; `origin="epoch"`, waktu = awal bar). Dataset `candle_5m` dibentuk dari candle 1 menit (`sumber_tf=1`). `ticker_5m` dikeluarkan dari daftar default (definisinya tetap ada). `load()` sekarang mengembalikan `(df, info)` | Indodax tidak menyediakan candle 5 menit; pembanding 1 menit vs 5 menit harus dari jenis data yang sama |
| 2026-11-12 | K1, K2 | `btcpred/data.py`, `run.py` | `info`: `tanggal_awal/akhir`, `n_baris_mentah`, `n_bar`, `n_bar_grid`, `n_hilang`, `n_celah`, `celah_terpanjang_mnt`, `n_parsial`. Ditulis di log dan `hasil/deskripsi_data.csv` (satu baris per dataset × horizon × periode, ditambah `n_setelah_fitur`, `n_pengembangan`, `n_uji_final`, `t_u`) | Deskripsi data wajib dilaporkan |
| 2026-11-12 | K3 | `run.py`, `config.py` | `SPLITS` diganti `FINAL_TEST_FRAC = 0.2` dan `TRAIN_FRACS = [1.0, 0.75, 0.5]`. `t_u = index[floor(0,8 n)]`; data pengembangan = baris dengan `t + horizon < t_u`; blok uji = `index >= t_u`; `assert` purging. `--splits` dihapus, `--train-fracs` ditambah (bawaan 1.0). Kolom `split` diganti `frac_latih`; `t_u` disimpan | Blok uji harus tetap antar skenario; Tahap 3 menguji panjang data latih, bukan memindah blok uji; purging berbasis waktu |
| 2026-11-12 | K4 | `btcpred/optim.py` | `TimeSeriesSplit(n_splits=5)` tanpa `gap`; di tiap fold baris latih dengan `t + horizon >= awal validasi` dibuang (`_folds()`). Argumen `gap=hb` diganti `horizon_menit=hz` | Jeda berbasis waktu, aman bila ada celah data |
| 2026-11-12 | K5 | `btcpred/optim.py` | Objektif Optuna = rata-rata **MAE harga (Rp)** 5 fold (`harga_now × exp(prediksi)`). `trial.report` memakai MAE harga. `tune()` mengembalikan `cv_mae_rp`, `cv_naive_rp`, `cv_vs_naive_pct`. Early stopping tetap memakai loss model | Objektif tuning disamakan dengan metrik utama dan bisa dibandingkan dengan Naive di validasi |
| 2026-11-12 | K6 | `btcpred/optim.py`, `run.py`, `btcpred/evaluate.py` | `select_top_k` dipindah ke dalam fold (importance model bawaan pada data latih fold; di-*cache* per fold karena tidak bergantung pada trial). Model akhir: `select_top_k` sekali pada seluruh data pengembangan. `fitur_final` dicatat di `hasil_validasi.csv`; `fitur_fold` dan `fitur_final` di `params/*.json` | Seleksi fitur di luar fold membocorkan informasi validasi |
| 2026-11-12 | K7 | `run.py`, `ringkas.py` | Keluaran dipisah: `hasil_validasi.csv` (tanpa metrik blok uji) dan `hasil_uji_final.csv`. Log `run.py` hanya menampilkan skor validasi. `ringkas.py --tahap 1/2` hanya membaca validasi; Tahap 1 tanpa syarat signifikansi; Tahap 2 `default` boleh terpilih dan selisih tiap strategi terhadap `default` dilaporkan. `--tahap 2` menulis `hasil/keputusan_terkunci.json` (tidak ditimpa tanpa `--paksa`). `--uji-final` menolak jalan bila file kunci belum ada | Pemilihan tidak boleh melihat blok uji |
| 2026-11-12 | K7 | `ringkas.py` | `keputusan_terkunci.json` menyimpan `sha256_hasil_validasi` (hash file saat dikunci, sesuai brief) **dan** `sha256_baris_tahap1_2` (hash kolom pemilihan baris Tahap 1-2 dengan format angka tetap). `--uji-final` mengecek hash baris | Hash file pasti berubah setelah Tahap 3-4 menambah baris, jadi pengecekan memakai hash baris; hash file tetap dicatat |
| 2026-11-12 | K8 | `btcpred/evaluate.py`, `run.py` | `dm_test(d, h, freq_menit)`: `d` = Series selisih loss ber-index waktu, di-reindex ke grid rapat; autokovarians lag 1..h−1 hanya dari pasangan yang keduanya tidak NaN, dibagi `n` = jumlah `d` tidak NaN. HLN dan df `n−1` tetap | Uji DM benar walaupun blok uji punya celah |
| 2026-11-12 | K8 | `ringkas.py`, `requirements.txt` | Holm (`statsmodels`) per keluarga tahap (`tahap4_*` = satu keluarga). Konfigurasi duplikat (run_id tanpa tag) dihitung sekali. Kolom `p_MAE_holm`, `status_uji` (`lebih_baik`/`lebih_buruk`/`tidak_berbeda`), `beda_arah_rmse`. Tabel sebaran (n_unik, rasio_sebaran, korelasi, datar) dipertahankan. Keluaran `hasil/ringkasan/uji_final_holm.csv` | Koreksi uji berganda |
| 2026-11-12 | K9 | `run.py` | `hasil/lingkungan.txt` (tanggal, Python, OS, CPU, RAM, `pip freeze`) ditulis sekali di awal bila belum ada | Reproduksibilitas |
| 2026-11-12 | K10 | `btcpred/evaluate.py` | `metrics()`: harga tetap bila \|harga_next − harga_now\| < Rp1; `Akurasi_arah_pct` dihitung pada bar yang berubah; kolom `n_arah`, `pct_tetap`, `pct_naik`, `pct_turun`, `arah_baseline_pct` (= sign(ret_lag1)) | Akurasi arah perlu pembanding |
| 2026-11-12 | K10 | `run.py` | AdaBoost dengan `--early-stop` atau `--loss l1` tidak dijalankan; ditulis `status="tidak_berlaku"` tanpa metrik di kedua file | Teknik itu tidak berlaku untuk AdaBoost.R2 |
| 2026-11-12 | K10 | `ringkas.py` | Rekap jumlah skenario dan konfigurasi unik per tahap (dicetak di setiap mode) | Laporan jumlah percobaan |
| 2026-11-12 | dok | `README.md`, `DAFTAR_PERCOBAAN.md` | Disesuaikan dengan alur v4 | — |

## Yang sengaja belum dikerjakan (menunggu keputusan)
- QMC 30 atau 32 trial.
- `has_time=True` untuk CatBoost.
- Beberapa seed untuk Tahap 2.

## Catatan
- `START_UTC`/`END_UTC` di `config.py` masih 14–20 Okt 2026. Untuk eksperimen Februari, ubah di config atau beri `--start`/`--end`.
- Fitur untuk `candle_5m` memakai jendela bergulir minimal 2 bar, sehingga jendela 5 menit menjadi 10 menit (nama kolom `*_10m`), sama seperti perilaku lama untuk data 5 menit.
- Baris yang jendela fiturnya melewati celah data tetap dibuang (perilaku lama).
