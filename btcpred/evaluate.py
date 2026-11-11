"""Metrik evaluasi, uji Diebold-Mariano, dan seleksi fitur."""
import numpy as np
import pandas as pd
from scipy import stats

from . import models as M

AMBANG_TETAP_RP = 1.0     # harga dianggap tetap bila |harga_next - harga_now| < 1 rupiah


def metrics(h_true, h_pred, r_pred, h_now, ret_lag1=None):
    """Metrik harga + akurasi arah.

    Akurasi arah hanya dihitung pada bar yang harganya berubah
    (|harga_next - harga_now| >= 1 rupiah). arah_baseline_pct = akurasi arah
    bila arah ditebak dengan sign(ret_lag1) (arah return bar terakhir)."""
    h_true, h_pred, h_now = map(np.asarray, (h_true, h_pred, h_now))
    e = h_true - h_pred
    dh = h_true - h_now
    gerak = np.abs(dh) >= AMBANG_TETAP_RP
    arah_aktual = np.sign(dh[gerak])
    arah = (np.sign(np.asarray(r_pred)[gerak]) == arah_aktual).mean() * 100 if gerak.any() else np.nan
    out = dict(MAE_Rp=float(np.abs(e).mean()),
               RMSE_Rp=float(np.sqrt((e ** 2).mean())),
               MAPE_pct=float((np.abs(e) / h_true).mean() * 100),
               Akurasi_arah_pct=float(arah),
               n_arah=int(gerak.sum()),
               pct_tetap=float((~gerak).mean() * 100),
               pct_naik=float((dh >= AMBANG_TETAP_RP).mean() * 100),
               pct_turun=float((dh <= -AMBANG_TETAP_RP).mean() * 100))
    if ret_lag1 is not None and gerak.any():
        b = np.sign(np.asarray(ret_lag1)[gerak])
        out["arah_baseline_pct"] = float((b == arah_aktual).mean() * 100)
    else:
        out["arah_baseline_pct"] = np.nan
    return out


def dm_test(d, h, freq_menit):
    """Diebold-Mariano dengan koreksi Harvey-Leybourne-Newbold.

    d          : pd.Series selisih loss (loss_model - loss_naive), index = waktu.
    h          : horizon dalam bar (autokovarians lag 1 .. h-1).
    freq_menit : resolusi grid (1 atau 5 menit).

    Seri di-reindex ke grid waktu rapat, sehingga bar yang hilang menjadi NaN dan
    jarak lag tetap benar walaupun ada celah. Autokovarians lag k memakai hanya
    pasangan yang keduanya tidak NaN, lalu dibagi n = jumlah d yang tidak NaN.
    DM < 0 berarti model lebih akurat daripada Naive."""
    d = d.sort_index()
    grid = pd.date_range(d.index.min(), d.index.max(), freq=f"{freq_menit}min")
    x = d.reindex(grid).to_numpy(dtype=float)
    ok = ~np.isnan(x)
    n = int(ok.sum())
    if n < 2:
        return np.nan, np.nan
    mean = np.nanmean(x)
    dc = np.where(ok, x - mean, 0.0)          # NaN -> 0 supaya pasangan berisi NaN tidak ikut
    gamma0 = np.dot(dc, dc) / n
    var = gamma0
    for k in range(1, h):
        if k >= len(dc):
            break
        var += 2 * np.dot(dc[k:], dc[:-k]) / n
    if var <= 0:
        var = gamma0
    if var == 0:
        return np.nan, np.nan
    dm = mean / np.sqrt(var / n)
    koreksi = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    dm *= koreksi
    p = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)


def select_top_k(name, X, y, k, loss, seed):
    """Seleksi fitur: latih model bawaan pada (X, y), ambil K fitur dengan
    importance tertinggi. Dipanggil di dalam tiap fold CV (data latih fold saja)
    dan sekali pada seluruh data pengembangan untuk model akhir."""
    m = M.make(name, M.DEFAULT[name], loss, seed)
    m.fit(X, y)
    imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
    return list(imp.index[:k])
