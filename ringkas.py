"""
Meringkas hasil percobaan dan menerapkan aturan pemilihan (review dosen 11-11-2026).

Urutan pemakaian:
    python ringkas.py                      # rekap jumlah skenario (hanya hasil_validasi.csv)
    python ringkas.py --tahap 1            # pilih (dataset, horizon) dari VALIDASI
    python ringkas.py --tahap 2            # pilih strategi optimasi dari VALIDASI,
                                           #   lalu tulis hasil/keputusan_terkunci.json
    python ringkas.py --uji-final          # baru membuka hasil_uji_final.csv (+ Holm)

Aturan:
  Tahap 1 : pasangan (dataset, horizon) dengan rata-rata cv_vs_naive_pct 4 model
            terkecil, sama seperti aturan Tahap 2. Pasangan yang belum lengkap 4 model
            dibuang. Seri -> horizon lebih pendek -> candle_1m. Tanpa syarat signifikansi.
  Tahap 2 : strategi dengan rata-rata cv_vs_naive_pct 4 model terkecil. 'default'
            boleh terpilih. Seri -> total waktu tuning terpendek.
  Uji final: RM3 -> satu keluarga Holm = semua baris model-vs-Naive dalam satu tahap
            (tag tahap4_* digabung jadi satu keluarga 'tahap4'). Konfigurasi duplikat
            (run_id tanpa tag) dihitung sekali.
            RM2 -> uji berpasangan antarkonfigurasi pada blok uji final, tiga
            keluarga Holm terpisah (lihat uji_berpasangan): strategi terpilih vs
            default, data latih 75%/50% vs 100%, dan tiap teknik Tahap 4 vs tanpa
            teknik. Pembandingnya baris Tahap 2 dengan strategi terpilih, jadi
            tidak perlu run tambahan.
  Uji berpasangan: tiga keluarga Holm tersendiri, membandingkan dua ramalan dari
            model yang sama pada blok uji final (DM loss absolut + HLN):
            (1) strategi terpilih Tahap 2 vs default,
            (2) data latih 75% dan 50% vs 100% (Tahap 3),
            (3) setiap teknik Tahap 4 vs konfigurasi dasar tanpa teknik.
"""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

OUT = Path(C.OUT_DIR)
F_VAL, F_UJI = OUT / "hasil_validasi.csv", OUT / "hasil_uji_final.csv"
F_KUNCI = OUT / "keputusan_terkunci.json"
URUT_DATASET = {"candle_1m": 0, "candle_5m": 1}
pd.set_option("display.width", 220)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_baris(df):
    """Hash isi baris tertentu (urut run_id), supaya bisa dicek ulang walaupun
    hasil_validasi.csv bertambah baris Tahap 3-4 setelah dikunci."""
    kol = ["run_id", "dataset", "horizon_menit", "model", "optimasi",
           "cv_mae_rp", "cv_naive_rp", "cv_vs_naive_pct", "detik_tuning"]
    d = df[kol].sort_values("run_id").copy()
    d["horizon_menit"] = d["horizon_menit"].astype(int)
    for c in kol[5:]:
        d[c] = d[c].map(lambda x: f"{float(x):.10e}")   # format tetap, tidak bergantung dtype kolom
    return hashlib.sha256(d.to_csv(index=False).encode()).hexdigest()


def baca_validasi():
    v = pd.read_csv(F_VAL)
    return v.drop_duplicates("run_id", keep="last")


def keluarga(tag):
    m = re.match(r"^(tahap\d)", str(tag))
    return m.group(1) if m else str(tag)


def kunci_tanpa_tag(run_id):
    return str(run_id).rsplit("|", 1)[0]


# ---------------------------------------------------------------------------
# rekap (K10)
# ---------------------------------------------------------------------------
def rekap(v):
    v = v[v.model != "Naive"].copy()
    v["tahap"] = v.tag.map(keluarga)
    v["kunci"] = v.run_id.map(kunci_tanpa_tag)
    r = v.groupby("tahap").agg(skenario=("run_id", "size"),
                               konfigurasi_unik=("kunci", "nunique"),
                               ok=("status", lambda s: (s == "ok").sum()),
                               tidak_berlaku=("status", lambda s: (s == "tidak_berlaku").sum()))
    print("\n=== Rekap skenario (dari hasil_validasi.csv) ===")
    print(r.to_string())
    print(f"  total skenario: {len(v)} | konfigurasi unik lintas tahap: {v.kunci.nunique()}")


# ---------------------------------------------------------------------------
# Tahap 1 dan 2 (hanya validasi)
# ---------------------------------------------------------------------------
def pilih_tahap1(v, tag):
    t = v[(v.tag == tag) & (v.status == "ok") & (v.model != "Naive")]
    if t.empty:
        raise SystemExit(f"Tidak ada baris validasi dengan tag '{tag}'.")
    n_model = t.groupby(["dataset", "horizon_menit"]).model.nunique()
    kurang = n_model[n_model < 4]
    if len(kurang):
        print("PERINGATAN: pasangan berikut belum lengkap 4 model dan tidak ikut dipilih:\n",
              kurang.to_string())
    pv = t.pivot_table(index=["dataset", "horizon_menit"], columns="model",
                       values="cv_vs_naive_pct", aggfunc="mean")
    s = t.groupby(["dataset", "horizon_menit"]).cv_vs_naive_pct.mean().rename("skor").reset_index()
    s["n_model"] = s.set_index(["dataset", "horizon_menit"]).index.map(n_model)
    s = s[s.n_model == 4].drop(columns="n_model")
    if s.empty:
        raise SystemExit("Tidak ada pasangan (dataset, horizon) yang lengkap 4 model.")
    s["urut_ds"] = s.dataset.map(URUT_DATASET).fillna(9)
    s = s.sort_values(["skor", "horizon_menit", "urut_ds"]).drop(columns="urut_ds")
    print(f"\n=== Tahap 1 (validasi, tag={tag}): cv_vs_naive_pct per model ===")
    print(pv.round(3).to_string())
    print("\nSkor pasangan = rata-rata cv_vs_naive_pct 4 model:")
    print(s.round(3).to_string(index=False))
    best = s.iloc[0]
    print(f"\n-> Terpilih: dataset={best.dataset}, horizon={int(best.horizon_menit)} menit "
          f"(skor {best.skor:+.3f}%)")
    (OUT / "ringkasan").mkdir(exist_ok=True)
    s.to_csv(OUT / "ringkasan" / "tahap1_validasi.csv", index=False)
    return dict(dataset=best.dataset, horizon_menit=int(best.horizon_menit),
                skor_cv_vs_naive_pct=float(best.skor)), t


def pilih_tahap2(v, tag):
    t = v[(v.tag == tag) & (v.status == "ok") & (v.model != "Naive")]
    if t.empty:
        raise SystemExit(f"Tidak ada baris validasi dengan tag '{tag}'.")
    pv = t.pivot_table(index="optimasi", columns="model", values="cv_vs_naive_pct", aggfunc="mean")
    n_model = t.groupby("optimasi").model.nunique()
    r = t.groupby("optimasi").agg(rata2=("cv_vs_naive_pct", "mean"),
                                  detik_tuning=("detik_tuning", "sum")).reset_index()
    r["n_model"] = r.optimasi.map(n_model)
    r = r.sort_values(["rata2", "detik_tuning"])
    print(f"\n=== Tahap 2 (validasi, tag={tag}): cv_vs_naive_pct per model ===")
    print(pv.round(3).to_string())
    print("\nRata-rata 4 model (seri -> waktu tuning terpendek):")
    print(r.round(3).to_string(index=False))
    if "default" in pv.index:
        print("\nSelisih tiap strategi terhadap 'default' per model (poin persen; negatif = lebih baik):")
        print((pv - pv.loc["default"]).round(3).to_string())
    if (r.n_model < 4).any():
        print("PERINGATAN: ada strategi yang belum lengkap 4 model.")
    best = r.iloc[0]
    print(f"\n-> Terpilih: optimasi={best.optimasi} (rata-rata {best.rata2:+.3f}%)")
    (OUT / "ringkasan").mkdir(exist_ok=True)
    r.to_csv(OUT / "ringkasan" / "tahap2_validasi.csv", index=False)
    return dict(optimasi=best.optimasi, rata2_cv_vs_naive_pct=float(best.rata2)), t


def kunci(v, tag1, tag2, paksa):
    if F_KUNCI.exists() and not paksa:
        print(f"\n{F_KUNCI} sudah ada; tidak ditimpa (pakai --paksa bila memang perlu).")
        print(F_KUNCI.read_text())
        return
    p1, t1 = pilih_tahap1(v, tag1)
    p2, t2 = pilih_tahap2(v, tag2)
    if not ((t2.dataset == p1["dataset"]) & (t2.horizon_menit == p1["horizon_menit"])).all():
        print("PERINGATAN: baris Tahap 2 tidak semuanya memakai pasangan terpilih Tahap 1.")
    isi = dict(dibuat_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               tag_tahap1=tag1, tag_tahap2=tag2, tahap1=p1, tahap2=p2,
               sha256_hasil_validasi=sha256(F_VAL),
               sha256_baris_tahap1_2=sha256_baris(pd.concat([t1, t2])),
               n_baris_tahap1=len(t1), n_baris_tahap2=len(t2))
    F_KUNCI.write_text(json.dumps(isi, indent=2))
    print(f"\nKeputusan dikunci -> {F_KUNCI}")
    print(json.dumps(isi, indent=2))


# ---------------------------------------------------------------------------
# Uji final (K8)
# ---------------------------------------------------------------------------
def cek_kunci(v):
    if not F_KUNCI.exists():
        raise SystemExit("Belum ada hasil/keputusan_terkunci.json. Jalankan "
                         "'python ringkas.py --tahap 2' dulu.")
    k = json.loads(F_KUNCI.read_text())
    t = v[v.tag.isin([k["tag_tahap1"], k["tag_tahap2"]]) & (v.status == "ok") & (v.model != "Naive")]
    if sha256_baris(t) != k["sha256_baris_tahap1_2"]:
        raise SystemExit("Baris validasi Tahap 1/2 berubah sejak dikunci. Uji final dibatalkan.")
    status_file = "sama" if sha256(F_VAL) == k["sha256_hasil_validasi"] else \
        "berbeda (wajar bila Tahap 3-4 sudah menambah baris)"
    print(f"Keputusan terkunci {k['dibuat_utc']}: {k['tahap1']} | {k['tahap2']}")
    print(f"  baris Tahap 1-2 tidak berubah (hash cocok); hash file validasi: {status_file}")
    return k


def holm(u):
    from statsmodels.stats.multitest import multipletests
    u = u[(u.model != "Naive") & (u.status == "ok")].copy()
    u["tahap"] = u.tag.map(keluarga)
    u["kunci"] = u.run_id.map(kunci_tanpa_tag)
    u["duplikat"] = u.duplicated(["tahap", "kunci"])
    u["p_MAE_holm"] = np.nan
    for _, g in u[~u.duplikat & u.p_MAE.notna()].groupby("tahap"):
        u.loc[g.index, "p_MAE_holm"] = multipletests(g.p_MAE.values, method="holm")[1]
    # duplikat mengikuti nilai baris pertamanya
    first = u[~u.duplikat].set_index(["tahap", "kunci"])["p_MAE_holm"]
    dup = u.duplikat
    u.loc[dup, "p_MAE_holm"] = [first.get((a, b), np.nan) for a, b in zip(u.loc[dup, "tahap"], u.loc[dup, "kunci"])]
    sig = u.p_MAE_holm < 0.05
    u["status_uji"] = np.select(
        [(u.MAE_vs_naive_pct < 0) & (u.DM_MAE < 0) & sig, (u.DM_MAE > 0) & sig],
        ["lebih_baik", "lebih_buruk"], "tidak_berbeda")
    u["beda_arah_rmse"] = np.sign(u.DM_RMSE) != np.sign(u.DM_MAE)
    return u


def sebaran(u):
    """Uji model datar dari file prediksi (lihat LOG_PERUBAHAN P2)."""
    dirp = OUT / "prediksi"
    baris = []
    for _, r in u.iterrows():
        safe = str(r["run_id"]).replace("|", "__").replace(":", "-").replace(" ", "_")
        f = dirp / f"{safe}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        pr = np.log(d["prediksi"] / d["harga_now"])
        ak = np.log(d["harga_aktual"] / d["harga_now"])
        sd = ak.std()
        baris.append(dict(run_id=r["run_id"], n_unik=pr.round(12).nunique(),
                          rasio_sebaran=pr.std() / sd if sd > 0 else np.nan,
                          korelasi=np.corrcoef(pr, ak)[0, 1] if pr.std() > 0 else np.nan,
                          geser_rata2=pr.mean()))
    if not baris:
        return u.assign(n_unik=np.nan, rasio_sebaran=np.nan, korelasi=np.nan, geser_rata2=np.nan,
                        datar=np.nan)
    s = pd.DataFrame(baris)
    s["datar"] = (s.n_unik <= 3) | (s.rasio_sebaran < 0.01)
    return u.merge(s, on="run_id", how="left")


# ---------------------------------------------------------------------------
# RM2: uji berpasangan perlakuan vs konfigurasi dasarnya (revisi 15-11-2026)
#   keluarga 1 : strategi terpilih Tahap 2 vs 'default'
#   keluarga 2 : data latih 75% dan 50% (Tahap 3) vs 100%
#   keluarga 3 : tiap teknik tambahan (Tahap 4) vs tanpa teknik tersebut
# Konfigurasi dasar ketiganya = baris Tahap 2 dengan strategi terpilih
# (frac_latih 1.0, tanpa teknik tambahan), jadi tidak perlu run tambahan.
# ---------------------------------------------------------------------------
KOL_PERLAKUAN = ["frac_latih", "loss", "early_stop", "top_k", "pruner"]


def baca_prediksi(run_id):
    safe = str(run_id).replace("|", "__").replace(":", "-").replace(" ", "_")
    d = pd.read_csv(OUT / "prediksi" / f"{safe}.csv", parse_dates=["waktu_target"])
    return d.set_index("waktu_target")


def dm_pasangan(a, b):
    """DM loss absolut dan kuadrat antara dua ramalan pada blok uji final.
    a = perlakuan, b = pembanding. DM < 0 berarti perlakuan lebih akurat."""
    from btcpred import evaluate as E
    pa, pb = baca_prediksi(a.run_id), baca_prediksi(b.run_id)
    pa, pb = pa.align(pb, join="inner", axis=0)
    ea, eb = pa.harga_aktual - pa.prediksi, pb.harga_aktual - pb.prediksi
    hb, tf = int(a.horizon_bar), int(a.tf_data_menit)
    dm1, p1 = E.dm_test(ea.abs() - eb.abs(), hb, tf)
    dm2, p2 = E.dm_test(ea ** 2 - eb ** 2, hb, tf)
    return dict(n=int(len(ea)), DM=dm1, p=p1, DM_L2=dm2, p_L2=p2)


def label_perlakuan(a, b):
    """Apa yang berbeda antara perlakuan dan pembandingnya."""
    beda = [f"optimasi={a.optimasi}"] if str(a.optimasi) != str(b.optimasi) else []
    beda += [f"{c}={a[c]}" for c in KOL_PERLAKUAN if str(a[c]) != str(b[c])]
    return ";".join(beda) if beda else "(sama)"


def pasangkan(a, b, nama):
    """Satu baris uji berpasangan; None bila berkas prediksinya tidak ada."""
    lab = label_perlakuan(a, b)
    if lab == "(sama)":
        return None
    if (a.dataset, a.horizon_menit) != (b.dataset, b.horizon_menit):
        print(f"  PERINGATAN: {a.run_id} dan pembandingnya beda dataset/horizon; dilewati.")
        return None
    try:
        d = dm_pasangan(a, b)
    except FileNotFoundError as e:
        print(f"  lewati {a.model} {lab}: berkas prediksi tidak ada ({e.filename}); "
              "pastikan C.SIMPAN_PREDIKSI = True saat run dijalankan.")
        return None
    return dict(keluarga=nama, model=a.model, perlakuan=lab,
                MAE_perlakuan=a.MAE_Rp, MAE_dasar=b.MAE_Rp,
                selisih_pct=(a.MAE_Rp / b.MAE_Rp - 1) * 100, **d,
                run_perlakuan=a.run_id, run_dasar=b.run_id)


def uji_keluarga(nama, baris, judul, n_harap=None):
    """Holm dalam satu keluarga; DM < 0 dan p_holm < 0,05 -> perlakuan lebih baik."""
    from statsmodels.stats.multitest import multipletests
    r = pd.DataFrame([b for b in baris if b])
    print(f"\n=== RM2: {judul} (keluarga Holm '{nama}') ===")
    if r.empty:
        print("  tidak ada pasangan yang bisa diuji.")
        return r
    # pasangan dengan ramalan identik (selisih loss selalu nol) tidak bisa diuji
    # dan tidak ikut menghitung Holm.
    bisa = r.p.notna()
    r["p_holm"] = np.nan
    r.loc[bisa, "p_holm"] = multipletests(r.loc[bisa, "p"].values, method="holm")[1]
    sig = r.p_holm < 0.05
    r["status"] = np.select([~bisa, (r.DM < 0) & sig, (r.DM > 0) & sig],
                            ["ramalan_identik", "lebih_baik", "lebih_buruk"], "tidak_berbeda")
    r["beda_arah_l2"] = bisa & (np.sign(r.DM_L2) != np.sign(r.DM))
    print(f"  {int(bisa.sum())} uji dalam keluarga Holm ini"
          + (f" ({int((~bisa).sum())} pasangan ramalannya identik, tidak diuji)"
             if (~bisa).any() else ""))
    print(r.drop(columns=["keluarga", "run_perlakuan", "run_dasar"]).round(4).to_string(index=False))
    print("  status:", r.status.value_counts().to_dict())
    if n_harap is not None and len(r) != n_harap:
        print(f"  PERINGATAN: jumlah pasangan {len(r)}, seharusnya {n_harap} bila semua "
              "run tersedia.")
    if r.beda_arah_l2.any():
        print("  catatan: ada uji yang arah loss kuadratnya berbeda; keputusan tetap "
              "mengikuti loss absolut.")
    return r


def baris_dasar(u, k, opt):
    """Baris konfigurasi dasar per model = Tahap 2 dengan strategi terpilih."""
    t2 = u[(u.tag == k["tag_tahap2"]) & (u.optimasi == opt)]
    d = {}
    for m, g in t2.groupby("model"):
        if len(g) > 1:
            print(f"PERINGATAN: ada {len(g)} baris dasar untuk {m}; dipakai yang pertama.")
        d[m] = g.iloc[0]
    return d


def uji_berpasangan(u_semua, k):
    opt = k["tahap2"]["optimasi"]
    u = u_semua[(u_semua.status == "ok") & (u_semua.model != "Naive")].copy()
    dasar = baris_dasar(u, k, opt)
    if not dasar:
        print(f"\nRM2: tidak ada baris Tahap 2 dengan optimasi '{opt}'; uji berpasangan dilewati.")
        return None
    n_model = len(dasar)
    print(f"\nKonfigurasi dasar RM2: tag={k['tag_tahap2']}, optimasi={opt}, "
          f"{n_model} model, frac_latih 1.0, tanpa teknik tambahan")
    out = []

    # keluarga 1: strategi terpilih vs default
    if opt == "default":
        print("\nRM2 keluarga 1: strategi terpilih = default, uji vs default tidak dijalankan.")
    else:
        t2d = u[(u.tag == k["tag_tahap2"]) & (u.optimasi == "default")]
        baris = [pasangkan(dasar[m], g.iloc[0], "tahap2_vs_default")
                 for m, g in t2d.groupby("model") if m in dasar]
        out.append(uji_keluarga("tahap2_vs_default", baris,
                                f"strategi terpilih '{opt}' vs default", n_model))

    # keluarga 2: panjang data latih
    t3 = u[(u.tag.map(keluarga) == "tahap3") & (u.frac_latih < 1.0)]
    baris = [pasangkan(r, dasar[r.model], "tahap3_vs_latih100")
             for _, r in t3.iterrows() if r.model in dasar]
    n3 = n_model * t3.frac_latih.nunique() if len(t3) else None
    out.append(uji_keluarga("tahap3_vs_latih100", baris,
                            "data latih sebagian vs 100%", n3))

    # keluarga 3: teknik tambahan
    t4 = u[u.tag.map(keluarga) == "tahap4"]
    tb4 = u_semua[(u_semua.tag.map(keluarga) == "tahap4")
                  & (u_semua.status == "tidak_berlaku")]
    if len(tb4):
        print(f"\nTahap 4: {len(tb4)} run tidak_berlaku (mis. AdaBoost dengan "
              "--early-stop/--loss l1) tidak ikut diuji.")
    baris = [pasangkan(r, dasar[r.model], "tahap4_vs_tanpa_teknik")
             for _, r in t4.iterrows() if r.model in dasar]
    out.append(uji_keluarga("tahap4_vs_tanpa_teknik", baris,
                            "tiap teknik tambahan vs tanpa teknik", len(t4)))

    r = [x for x in out if x is not None and not x.empty]
    if not r:
        return None
    r = pd.concat(r, ignore_index=True)
    (OUT / "ringkasan").mkdir(exist_ok=True)
    f = OUT / "ringkasan" / "uji_berpasangan.csv"
    r.to_csv(f, index=False)
    print(f"\nTersimpan: {f}")
    return r


def uji_final(v):
    k = cek_kunci(v)
    u = pd.read_csv(F_UJI).drop_duplicates("run_id", keep="last")
    h = sebaran(holm(u))
    kol = ["tahap", "tag", "dataset", "horizon_menit", "frac_latih", "model", "optimasi", "loss",
           "early_stop", "top_k", "pruner", "MAE_vs_naive_pct", "RMSE_vs_naive_pct", "DM_MAE",
           "p_MAE", "p_MAE_holm", "status_uji", "beda_arah_rmse", "Akurasi_arah_pct",
           "arah_baseline_pct", "pct_tetap", "korelasi", "rasio_sebaran", "datar", "duplikat"]
    print("\n=== Uji final per tahap (Holm per tahap pada p_MAE) ===")
    for tahap, g in h.groupby("tahap"):
        print(f"\n--- {tahap}: {len(g)} baris ({(~g.duplikat).sum()} unik dalam keluarga Holm) ---")
        print(g[kol[1:]].drop(columns=["duplikat"]).round(4).to_string(index=False))
        print("  status:", g[~g.duplikat].status_uji.value_counts().to_dict())
    tb = pd.read_csv(F_UJI)
    tb = tb[tb.status == "tidak_berlaku"]
    if len(tb):
        print(f"\nTidak berlaku (tidak dijalankan): {len(tb)} baris, mis. AdaBoost dengan --early-stop/--loss l1")
    (OUT / "ringkasan").mkdir(exist_ok=True)
    h[kol + ["run_id"]].to_csv(OUT / "ringkasan" / "uji_final_holm.csv", index=False)
    print(f"\nTersimpan: {OUT / 'ringkasan' / 'uji_final_holm.csv'}")
    uji_berpasangan(u, k)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--tahap", type=int, choices=[1, 2])
    a.add_argument("--uji-final", action="store_true")
    a.add_argument("--tag1", default="tahap1")
    a.add_argument("--tag2", default="tahap2")
    a.add_argument("--paksa", action="store_true", help="timpa keputusan_terkunci.json")
    a = a.parse_args()

    v = baca_validasi()
    rekap(v)
    if a.tahap == 1:
        pilih_tahap1(v, a.tag1)
    elif a.tahap == 2:
        kunci(v, a.tag1, a.tag2, a.paksa)
    if a.uji_final:
        uji_final(v)


if __name__ == "__main__":
    main()
