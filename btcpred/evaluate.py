"""Metrik evaluasi, uji Diebold-Mariano, dan seleksi fitur."""
import numpy as np
import pandas as pd
from scipy import stats

from . import models as M


def metrics(h_true, h_pred, r_true, r_pred):
    e = h_true - h_pred
    gerak = r_true != 0          # bar tanpa perubahan harga tidak dihitung arahnya
    arah = (np.sign(r_pred[gerak]) == np.sign(r_true[gerak])).mean() * 100 if gerak.any() else np.nan
    return dict(MAE_Rp=float(np.abs(e).mean()),
                RMSE_Rp=float(np.sqrt((e ** 2).mean())),
                MAPE_pct=float((np.abs(e) / h_true).mean() * 100),
                Akurasi_arah_pct=float(arah))


def dm_test(e_model, e_naive, h, power=1):
    """Diebold-Mariano dengan koreksi Harvey-Leybourne-Newbold.
    power=1: loss absolut (MAE), power=2: loss kuadrat (RMSE).
    Autokovarians sampai lag h-1 karena target horizon h saling tumpang tindih.
    DM < 0 berarti model lebih akurat daripada Naive."""
    d = np.abs(e_model) ** power - np.abs(e_naive) ** power
    n = len(d)
    dc = d - d.mean()
    gamma0 = np.dot(dc, dc) / n
    var = gamma0
    for k in range(1, h):
        var += 2 * np.dot(dc[k:], dc[:-k]) / n
    if var <= 0:
        var = gamma0
    if var == 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(var / n)
    koreksi = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    dm *= koreksi
    p = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)


def select_top_k(name, X, y, k, loss, seed):
    """Seleksi fitur: latih model bawaan di data latih, ambil K fitur dengan
    importance tertinggi. Hanya memakai data latih."""
    m = M.make(name, M.DEFAULT[name], loss, seed)
    m.fit(X, y)
    imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
    return list(imp.index[:k])
