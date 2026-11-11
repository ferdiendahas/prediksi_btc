"""
KONFIGURASI PERCOBAAN
=====================
Ubah daftar di bawah untuk menentukan variasi percobaan. run.py akan menjalankan
SEMUA kombinasi (dataset x horizon x split x model x optimasi).
Semua daftar juga bisa ditimpa lewat argumen command line, contoh:
    python run.py --datasets candle_1m --horizons 5 15 --optimizers tpe random
Lihat DAFTAR_PERCOBAAN.md untuk penjelasan tiap pilihan.
"""

# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------
DATA = {
    # nama_dataset: path, jenis (candle/ticker), resolusi asli dalam menit
    "candle_1m": {"path": "data/btcidr_candle_1menit_20261110_1642.csv", "jenis": "candle", "tf_menit": 1},
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
DATASETS   = ["candle_1m", "ticker_5m"]

# Horizon prediksi dalam MENIT (berapa menit ke depan harga diprediksi).
# Horizon yang bukan kelipatan resolusi data otomatis dilewati
# (mis. horizon 1 menit tidak bisa untuk ticker_5m).
HORIZONS   = [1, 5, 15, 30, 60]

# Porsi data latih (sisanya data uji): 0.8 = 80:20, 0.7 = 70:30, 0.6 = 60:40
SPLITS     = [0.8, 0.7, 0.6]

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
