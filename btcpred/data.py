"""Memuat data Indodax dan membangun fitur + target untuk horizon tertentu."""
import numpy as np
import pandas as pd


def agregasi_5m(df1):
    """df1: index waktu awal candle 1 menit (UTC), kolom open, high, low, close, volume.
    Hasil: candle 5 menit, waktu = AWAL bar (UTC). n_menit < 5 = bar parsial."""
    g = df1.resample("5min", label="left", closed="left", origin="epoch")
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(),
        "volume": g["volume"].sum(min_count=1),
        "n_menit": g["close"].count(),
    })
    out = out[out["n_menit"] > 0]
    return out


def _celah(idx, tf_menit, start, end):
    """Statistik bar hilang pada grid TF di dalam [start, end]."""
    s = pd.Timestamp(start).floor(f"{tf_menit}min") if start else idx.min()
    e = pd.Timestamp(end).floor(f"{tf_menit}min") if end else idx.max()
    grid = pd.date_range(s, e, freq=f"{tf_menit}min")
    ada = grid.isin(idx)
    hilang = ~ada
    n_hilang = int(hilang.sum())
    if n_hilang == 0:
        return dict(n_bar_grid=len(grid), n_hilang=0, n_celah=0, celah_terpanjang_mnt=0)
    blok = pd.Series(hilang).ne(pd.Series(hilang).shift()).cumsum()
    panjang = pd.Series(hilang)[hilang].groupby(blok[hilang]).size()
    return dict(n_bar_grid=len(grid), n_hilang=n_hilang, n_celah=int(len(panjang)),
                celah_terpanjang_mnt=int(panjang.max() * tf_menit))


def load(path, jenis, tf_menit, harga, start=None, end=None, sumber_tf=None):
    """Baca CSV, rapikan timestamp ke grid TF, isi kolom 'harga'.
    Bar yang hilang dibiarkan NaN (tidak diisi), supaya fitur/target yang
    melewati gap otomatis terbuang.

    sumber_tf=1 dengan tf_menit=5 (jenis candle): candle 5 menit dibentuk dari
    candle 1 menit lewat agregasi_5m().

    Mengembalikan (df, info). info memuat deskripsi data di dalam [start, end]:
    tanggal awal/akhir, n_baris_mentah, n_bar, n_parsial, n_hilang, n_celah,
    celah_terpanjang_mnt."""
    df = pd.read_csv(path)
    agregasi = sumber_tf is not None and sumber_tf != tf_menit
    floor_tf = sumber_tf if agregasi else tf_menit
    df["waktu"] = pd.to_datetime(df["waktu_utc"]).dt.floor(f"{floor_tf}min")
    df = df.drop_duplicates("waktu").sort_values("waktu").set_index("waktu")

    dalam = slice(start, end)
    n_mentah = int(len(df.loc[dalam]))
    if agregasi:
        if jenis != "candle" or sumber_tf != 1 or tf_menit != 5:
            raise ValueError("agregasi hanya didukung untuk candle 1 menit -> 5 menit")
        df = agregasi_5m(df[["open", "high", "low", "close", "volume"]])
    else:
        df["n_menit"] = tf_menit if jenis == "candle" else np.nan

    info = dict(tanggal_awal=str(df.loc[dalam].index.min()),
                tanggal_akhir=str(df.loc[dalam].index.max()),
                n_baris_mentah=n_mentah, n_bar=int(len(df.loc[dalam])),
                n_parsial=int((df.loc[dalam, "n_menit"] < tf_menit).sum()) if jenis == "candle" else 0)
    info.update(_celah(df.index, tf_menit, start, end))

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
    df = df.reindex(full)
    if start or end:
        # ambil sedikit data sebelum START untuk pemanasan fitur (hanya data masa lalu)
        buf = pd.Timedelta(minutes=300)
        s = pd.Timestamp(start) - buf if start else None
        df = df.loc[s:end]
    return df, info


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
