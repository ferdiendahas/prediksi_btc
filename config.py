"""
KONFIGURASI PERCOBAAN
=====================
Ubah daftar di bawah untuk menentukan variasi percobaan. run.py akan menjalankan
SEMUA kombinasi (dataset x horizon x frac_latih x model x optimasi).
Semua daftar juga bisa ditimpa lewat argumen command line, contoh:
    python run.py --datasets candle_1m --horizons 5 15 --optimizers tpe random
Lihat DAFTAR_PERCOBAAN.md untuk penjelasan tiap pilihan.
"""

# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------
CANDLE_1M = "data/btcidr_candle_1menit_20261110_1642.csv"
DATA = {
    # nama_dataset: path, jenis (candle/ticker), resolusi dalam menit,
    # sumber_tf = resolusi file sumber bila bar dibentuk lewat agregasi
    "candle_1m": {"path": CANDLE_1M, "jenis": "candle", "tf_menit": 1},
    # candle 5 menit = agregasi candle 1 menit (Indodax tidak menyediakan tf=5).
    # Waktu bar = AWAL bar (UTC). Bar dengan < 5 menit data dicatat sebagai n_parsial.
    "candle_5m": {"path": CANDLE_1M, "jenis": "candle", "tf_menit": 5, "sumber_tf": 1},
    # ticker_5m tidak lagi dipakai (review dosen 11-11-2026); definisi disimpan
    # supaya hasil lama tetap bisa direproduksi bila file CSV-nya tersedia.
    "ticker_5m": {"path": "data/btcidr_ticker_5menit_20261110_1642.csv", "jenis": "ticker", "tf_menit": 5},
}

# Rentang data yang dipakai (UTC). Seminggu penuh = 7 x 24 jam.
START_UTC = "2026-10-14 00:00"
END_UTC   = "2026-10-20 23:59"

# Definisi "harga" per jenis data
#   candle: "typical" = (high+low+close)/3  | "close" | "ohlc4" = (open+high+low+close)/4
#   ticker: "mid" = (bid+ask)/2            | "last" = last_price
HARGA = {"candle": "typical", "ticker": "mid"}

# ---------------------------------------------------------------------------
# 2. VARIASI PERCOBAAN
# ---------------------------------------------------------------------------
DATASETS   = ["candle_1m", "candle_5m"]

# Horizon prediksi dalam MENIT (berapa menit ke depan harga diprediksi).
# Horizon yang bukan kelipatan resolusi data otomatis dilewati
# (mis. horizon 1 menit tidak bisa untuk candle_5m).
HORIZONS   = [1, 5, 15, 30, 60]

# Pembagian data berbasis waktu (review dosen 11-11-2026):
#   blok uji final = FINAL_TEST_FRAC bar terakhir (sama untuk semua run per dataset x horizon).
#   Tahap 3 memvariasikan PANJANG data latih: porsi akhir data pengembangan yang dipakai.
#   run.py memakai --train-fracs (default 1.0); daftar di bawah dipakai untuk Tahap 3.
FINAL_TEST_FRAC = 0.2
TRAIN_FRACS     = [1.0, 0.75, 0.5]

MODELS     = ["XGBoost", "LightGBM", "CatBoost", "AdaBoost"]

# Metode optimasi hyperparameter:
#   default : tanpa tuning (setelan bawaan library)
#   grid    : Grid Search (kombinasi tetap, lihat GRID di btcpred/models.py)
#   random  : Random Search
#   tpe     : Bayesian Optimization (Tree-structured Parzen Estimator)
#   cmaes   : CMA-ES (Evolution Strategy)
#   ga      : Genetic Algorithm (NSGA-II)
#   qmc     : Quasi-Monte Carlo (Sobol)
OPTIMIZERS = ["default", "grid", "random", "tpe", "cmaes", "ga"]

# ---------------------------------------------------------------------------
# 3. PENGATURAN OPTIMASI TAMBAHAN (berlaku untuk semua run)
# ---------------------------------------------------------------------------
N_TRIALS   = 30        # jatah percobaan tuning per model (sama untuk semua metode, kecuali grid)
N_FOLDS    = 5         # fold walk-forward (TimeSeriesSplit) saat tuning
PRUNER     = None      # None | "median" | "hyperband"  -> hentikan trial jelek lebih awal
EARLY_STOP = False     # True = early stopping (XGBoost, LightGBM, CatBoost)
TOP_K      = None      # None = semua fitur | angka = seleksi K fitur terpenting
LOSS       = "l2"      # "l2" = squared error | "l1" = absolute error (MAE)

# ---------------------------------------------------------------------------
# 4. FITUR
# ---------------------------------------------------------------------------
N_LAG       = 12                      # jumlah lag return (dalam bar)
WINDOWS_MNT = [5, 15, 30, 60, 240]    # jendela statistik bergulir (menit)

# ---------------------------------------------------------------------------
# 5. LAIN-LAIN
# ---------------------------------------------------------------------------
SEED           = 42
OUT_DIR        = "hasil"
SIMPAN_PREDIKSI = True                # simpan prediksi data uji per run
