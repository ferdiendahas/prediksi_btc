"""
Meringkas hasil/hasil_semua.csv menjadi tabel perbandingan.

Contoh:
    python ringkas.py                                  # semua tabel, semua run
    python ringkas.py --dataset candle_1m --split 80:20 --optimasi tpe
    python ringkas.py --tag tahap1

Tabel yang dibuat (juga disimpan ke hasil/ringkasan/):
  1. horizon_x_model   : MAE vs Naive (%) per horizon dan model  -> cari TF optimal
  2. optimasi_x_model  : MAE vs Naive (%) per metode optimasi dan model
  3. split_x_model     : MAE vs Naive (%) per rasio split dan model
  4. sebaran_prediksi  : uji model datar -- apakah model benar-benar memprediksi
  5. peringkat         : 15 konfigurasi terbaik
Angka negatif = lebih akurat dari Naive. Bintang (*) = DM signifikan p < 0,05.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import config as C


def sebaran(t, out):
    """Uji model datar.

    Model pohon bisa dituning sampai praktis berhenti memprediksi: ia mengeluarkan
    satu konstanta untuk semua baris uji. Hasilnya tampak "menang tipis dari Naive"
    di tabel MAE, padahal kandungan informasinya nol -- yang terjadi cuma Naive
    digeser sebesar konstanta itu. Tabel ini membongkarnya dengan membaca file
    prediksi tiap run.

      n_unik        : banyaknya nilai prediksi berbeda di seluruh data uji (1 = datar)
      rasio_sebaran : std(return prediksi) / std(return aktual).
                      ~0 = prediksi datar; makin besar = model makin berani bergerak.
      geser_rata2   : rata-rata return prediksi. Kalau ini jauh dari nol dan searah
                      tren periode uji, keunggulan atas Naive bisa berasal dari
                      pergeseran konstan itu saja, bukan dari prediksi per waktu.
      status        : DATAR bila n_unik <= 3 atau rasio_sebaran < 0,01
    """
    dirp = out.parent / "prediksi"
    baris = []
    for _, r in t.iterrows():
        safe = str(r["run_id"]).replace("|", "__").replace(":", "-").replace(" ", "_")
        f = dirp / f"{safe}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        pr = np.log(d["prediksi"] / d["harga_now"])
        ak = np.log(d["harga_aktual"] / d["harga_now"])
        sd_ak = ak.std()
        baris.append({
            "dataset": r["dataset"], "horizon": r["horizon_menit"], "split": r["split"],
            "model": r["model"], "optimasi": r["optimasi"],
            "MAE_vs_naive_pct": r["MAE_vs_naive_pct"],
            "n_unik": pr.nunique(),
            "rasio_sebaran": pr.std() / sd_ak if sd_ak > 0 else np.nan,
            "geser_rata2": pr.mean(),
        })
    if not baris:
        print("\n=== sebaran_prediksi ===\n  (file prediksi tidak ditemukan; "
              "SIMPAN_PREDIKSI mungkin False saat run)")
        return
    s = pd.DataFrame(baris)
    s["status"] = np.where((s.n_unik <= 3) | (s.rasio_sebaran < 0.01), "DATAR", "ok")
    s = s.sort_values("rasio_sebaran")
    print("\n=== sebaran_prediksi (uji model datar) ===")
    print("  n_unik=1 berarti model mengeluarkan satu konstanta untuk SEMUA baris uji:")
    print("  keunggulan atas Naive dari run seperti itu bukan kemampuan prediksi.")
    tampil = s.copy()
    for k in ["rasio_sebaran", "geser_rata2"]:
        tampil[k] = tampil[k].map(lambda v: f"{v:.2e}" if pd.notna(v) else "")
    print(tampil.round(3).to_string(index=False))
    n = (s.status == "DATAR").sum()
    print(f"  -> {n} dari {len(s)} run berstatus DATAR.")
    s.to_csv(out / "sebaran_prediksi.csv", index=False)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--dataset"); a.add_argument("--split"); a.add_argument("--optimasi")
    a.add_argument("--horizon", type=int); a.add_argument("--tag")
    a.add_argument("--metrik", default="MAE_vs_naive_pct",
                   choices=["MAE_vs_naive_pct", "RMSE_vs_naive_pct", "Akurasi_arah_pct"])
    a = a.parse_args()

    t = pd.read_csv(Path(C.OUT_DIR) / "hasil_semua.csv")
    t = t[t["model"] != "Naive"]
    for col, val in [("dataset", a.dataset), ("split", a.split), ("optimasi", a.optimasi),
                     ("horizon_menit", a.horizon), ("tag", a.tag)]:
        if val is not None:
            t = t[t[col] == val]
    if t.empty:
        print("Tidak ada run yang cocok dengan filter."); return

    out = Path(C.OUT_DIR) / "ringkasan"; out.mkdir(exist_ok=True)
    m = a.metrik
    pcol = "p_RMSE" if m == "RMSE_vs_naive_pct" else "p_MAE"
    pd.set_option("display.width", 200)

    def pivot(idx, nama):
        pv = t.pivot_table(index=idx, columns="model", values=m, aggfunc="mean")
        sig = t.pivot_table(index=idx, columns="model", values=pcol, aggfunc="max")
        print(f"\n=== {nama} ({m}, rata-rata bila ada beberapa run) ===")
        print(pv.round(3).to_string())
        if m != "Akurasi_arah_pct":
            print("  p-value DM terbesar per sel (signifikan bila < 0,05):")
            print(sig.round(3).to_string())
        pv.to_csv(out / f"{nama}.csv")

    pivot(["dataset", "horizon_menit"], "horizon_x_model")
    pivot("optimasi", "optimasi_x_model")
    pivot("split", "split_x_model")
    sebaran(t, out)

    asc = m != "Akurasi_arah_pct"
    top = t.sort_values(m, ascending=asc).head(15)
    kol = ["dataset", "horizon_menit", "split", "model", "optimasi", "MAE_Rp",
           "MAE_vs_naive_pct", "RMSE_vs_naive_pct", "Akurasi_arah_pct", "p_MAE", "p_RMSE",
           "std_pred_return"]
    print("\n=== 15 konfigurasi terbaik ===")
    print(top[kol].round(3).to_string(index=False))
    top[kol].to_csv(out / "peringkat.csv", index=False)
    print(f"\nTabel disimpan di {out}/")


if __name__ == "__main__":
    main()
