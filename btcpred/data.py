"""Memuat data Indodax dan membangun fitur + target untuk horizon tertentu."""
import numpy as np
import pandas as pd


def load(path, jenis, tf_menit, harga, start=None, end=None):
    """Baca CSV, rapikan timestamp ke grid TF, isi kolom 'harga'.
    Bar yang hilang dibiarkan NaN (tidak diisi), supaya fitur/target yang
    melewati gap otomatis terbuang."""
    df = pd.read_csv(path)
    df["waktu"] = pd.to_datetime(df["waktu_utc"]).dt.floor(f"{tf_menit}min")
    df = df.drop_duplicates("waktu").sort_values("waktu").set_index("waktu")

    if jenis == "candle":
        if harga == "typical":
            df["harga"] = (df["high"] + df["low"] + df["close"]) / 3
        elif harga == "ohlc4":
            df["harga"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        elif harga == "close":
            df["harga"] = df["close"]
        else:
            raise ValueError(f"harga candle tidak dikenal: {harga}")
    elif jenis == "ticker":
        df["mid"] = (df["buy"] + df["sell"]) / 2          # buy = bid, sell = ask
        if harga == "mid":
            df["harga"] = df["mid"]
        elif harga == "last":
            df["harga"] = df["last_price"]
        else:
            raise ValueError(f"harga ticker tidak dikenal: {harga}")
    else:
        raise ValueError(jenis)

    full = pd.date_range(df.index.min(), df.index.max(), freq=f"{tf_menit}min")
    n_gap = len(full) - len(df)
    df = df.reindex(full)
    if start or end:
        # ambil sedikit data sebelum START untuk pemanasan fitur (hanya data masa lalu)
        buf = pd.Timedelta(minutes=300)
        s = pd.Timestamp(start) - buf if start else None
        df = df.loc[s:end]
    return df, n_gap


def build(df, jenis, tf_menit, horizon_menit, n_lag, windows_mnt, start=None):
    """Kembalikan DataFrame berisi fitur, 'target' (log return h ke depan),
    'harga_now', 'harga_next'. Semua fitur hanya memakai data sampai waktu t."""
    h = horizon_menit // tf_menit                     # horizon dalam bar
    p = df["harga"]
    r = np.log(p).diff()                              # return bar t-1 -> t
    X = pd.DataFrame(index=df.index)

    for k in range(1, n_lag + 1):
        X[f"ret_lag{k}"] = r.shift(k - 1)

    wins = sorted({max(2, w // tf_menit) for w in windows_mnt})
    for w in wins:
        m = w * tf_menit
        X[f"ret_sum_{m}m"] = r.rolling(w).sum()
        X[f"ret_std_{m}m"] = r.rolling(w).std()
        X[f"nol_{m}m"] = (r == 0).astype(float).where(r.notna()).rolling(w).mean()

    if jenis == "candle":
        rng = (df["high"] - df["low"]) / df["close"]
        X["range_rel"] = rng
        hl = (df["high"] - df["low"]).replace(0, np.nan)
        X["posisi_bar"] = ((df["close"] - df["low"]) / hl).fillna(0.5).where(df["close"].notna())
        lv = np.log1p(df["volume"])
        X["log_vol"] = lv
        for w in wins:
            m = w * tf_menit
            X[f"range_mean_{m}m"] = rng.rolling(w).mean()
            X[f"vol_rasio_{m}m"] = lv - lv.rolling(w).mean()
    else:
        X["spread_rel"] = (df["sell"] - df["buy"]) / df["mid"]
        r24 = (df["high_24h"] - df["low_24h"]).replace(0, np.nan)
        X["posisi_24h"] = (p - df["low_24h"]) / r24
        X["dvol_24h"] = np.log(df["vol_btc_24h"]).diff()

    jam = df.index.hour + df.index.minute / 60
    X["jam_sin"] = np.sin(2 * np.pi * jam / 24)
    X["jam_cos"] = np.cos(2 * np.pi * jam / 24)

    out = X.copy()
    out["target"] = np.log(p.shift(-h) / p)
    out["harga_now"] = p
    out["harga_next"] = p.shift(-h)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    if start:
        out = out.loc[start:]
    return out, list(X.columns), h
