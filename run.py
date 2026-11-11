"""
Menjalankan semua kombinasi percobaan dari config.py (bisa ditimpa lewat argumen).

Contoh:
    python run.py --optimizers tpe --tag tahap1
    python run.py --datasets candle_1m --horizons 5 --optimizers default grid random tpe cmaes ga qmc --tag tahap2
    python run.py --datasets candle_1m --horizons 5 --optimizers tpe --train-fracs 1.0 0.75 0.5 --tag tahap3
    python run.py --datasets candle_1m --horizons 5 --optimizers tpe --top-k 15 --tag tahap4_topk15

Pembagian data (berbasis waktu, revisi dosen 11-11-2026):
  - Blok uji final = 20% bar terakhir, mulai waktu t_u. Sama untuk semua run
    dengan (dataset, horizon) yang sama.
  - Data pengembangan = semua baris yang waktu targetnya (t + horizon) < t_u.
  - --train-fracs 0.75 / 0.5 memakai hanya bagian akhir data pengembangan
    (Tahap 3: panjang data latih). Blok uji tidak berubah.

Keluaran:
  hasil/hasil_validasi.csv   : identitas run + skor validasi (dipakai untuk memilih)
  hasil/hasil_uji_final.csv  : identitas run + metrik blok uji (dibuka setelah dikunci)
  hasil/deskripsi_data.csv   : satu baris per (dataset, horizon, periode)
  hasil/lingkungan.txt       : versi Python, paket, CPU, RAM, OS
  hasil/params/, hasil/prediksi/

Run yang sudah selesai (run_id sama, termasuk --tag) otomatis dilewati, jadi
script aman dihentikan lalu dijalankan lagi.
"""
import argparse
import itertools
import json
import os
import platform
import subprocess
import sys
import time
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from btcpred import data as D, models as M, optim as O, evaluate as E


def parse():
    a = argparse.ArgumentParser()
    a.add_argument("--datasets", nargs="+")
    a.add_argument("--horizons", nargs="+", type=int)
    a.add_argument("--train-fracs", nargs="+", type=float, default=[1.0],
                   help="porsi akhir data pengembangan yang dipakai melatih (Tahap 3)")
    a.add_argument("--models", nargs="+")
    a.add_argument("--optimizers", nargs="+")
    a.add_argument("--trials", type=int)
    a.add_argument("--folds", type=int)
    a.add_argument("--pruner", choices=["median", "hyperband", "none"])
    a.add_argument("--early-stop", action="store_true")
    a.add_argument("--top-k", type=int)
    a.add_argument("--loss", choices=["l1", "l2"])
    a.add_argument("--harga-candle", choices=["typical", "close", "ohlc4"])
    a.add_argument("--harga-ticker", choices=["mid", "last"])
    a.add_argument("--start")
    a.add_argument("--end")
    a.add_argument("--tag", default="", help="label bebas untuk menandai run")
    return a.parse_args()


# ---------------------------------------------------------------------------
# kolom keluaran
# ---------------------------------------------------------------------------
ID = ["run_id", "status", "dataset", "harga", "tf_data_menit", "horizon_menit", "horizon_bar",
      "frac_latih", "model", "optimasi", "loss", "early_stop", "top_k", "pruner",
      "n_trials_cfg", "n_folds", "periode", "t_u", "n_latih", "latih_mulai", "latih_akhir", "tag"]
KOLOM_VALIDASI = ID + ["cv_mae_rp", "cv_naive_rp", "cv_vs_naive_pct", "n_trials", "n_pruned",
                       "n_fitur", "fitur_final", "detik_tuning", "detik_total"]
KOLOM_UJI = ID + ["n_uji", "uji_mulai", "uji_akhir", "MAE_Rp", "RMSE_Rp", "MAPE_pct",
                  "Akurasi_arah_pct", "n_arah", "pct_tetap", "pct_naik", "pct_turun",
                  "arah_baseline_pct", "MAE_vs_naive_pct", "RMSE_vs_naive_pct",
                  "DM_MAE", "p_MAE", "DM_RMSE", "p_RMSE", "std_pred_return"]
KOLOM_DESKRIPSI = ["dataset", "horizon_menit", "periode", "tanggal_awal", "tanggal_akhir",
                   "n_baris_mentah", "n_bar", "n_bar_grid", "n_hilang", "n_celah",
                   "celah_terpanjang_mnt", "n_parsial", "n_setelah_fitur", "n_pengembangan",
                   "n_uji_final", "t_u"]


def _append(path, row, kolom):
    df = pd.DataFrame([{k: row.get(k, np.nan) for k in kolom}])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _upsert_deskripsi(path, row):
    new = pd.DataFrame([{k: row.get(k, np.nan) for k in KOLOM_DESKRIPSI}])
    if path.exists():
        old = pd.read_csv(path)
        key = (old.dataset == row["dataset"]) & (old.horizon_menit == row["horizon_menit"]) \
            & (old.periode == row["periode"])
        new = pd.concat([old[~key], new], ignore_index=True)
    new.to_csv(path, index=False)


def tulis_lingkungan(path):
    """K9: catat lingkungan sekali (kalau file belum ada)."""
    if path.exists():
        return

    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as e:
            return f"(gagal: {e})"

    cpu, ram = platform.processor(), ""
    if sys.platform == "darwin":
        cpu = sh(["sysctl", "-n", "machdep.cpu.brand_string"]) or cpu
        try:
            ram = f"{int(sh(['sysctl', '-n', 'hw.memsize'])) / 2**30:.1f} GB"
        except ValueError:
            ram = ""
    elif Path("/proc/cpuinfo").exists():
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip(); break
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                ram = f"{int(line.split()[1]) / 2**20:.1f} GB"; break
    teks = [f"tanggal : {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            f"python  : {sys.version.split()[0]} ({sys.executable})",
            f"OS      : {platform.platform()}",
            f"CPU     : {cpu} | {os.cpu_count()} core logis",
            f"RAM     : {ram}", "", "== pip freeze ==",
            sh([sys.executable, "-m", "pip", "freeze"])]
    path.write_text("\n".join(teks) + "\n")


def main():
    a = parse()
    cfg = dict(
        DATASETS=a.datasets or C.DATASETS, HORIZONS=a.horizons or C.HORIZONS,
        TRAIN_FRACS=a.train_fracs, MODELS=a.models or C.MODELS,
        OPTIMIZERS=a.optimizers or C.OPTIMIZERS, N_TRIALS=a.trials or C.N_TRIALS,
        N_FOLDS=a.folds or C.N_FOLDS,
        PRUNER=(None if a.pruner == "none" else a.pruner) if a.pruner else C.PRUNER,
        EARLY_STOP=a.early_stop or C.EARLY_STOP, TOP_K=a.top_k or C.TOP_K,
        LOSS=a.loss or C.LOSS, SEED=C.SEED,
        HARGA={"candle": a.harga_candle or C.HARGA["candle"],
               "ticker": a.harga_ticker or C.HARGA["ticker"]},
        START=a.start or C.START_UTC, END=a.end or C.END_UTC,
    )
    out = Path(C.OUT_DIR)
    (out / "params").mkdir(parents=True, exist_ok=True)
    (out / "prediksi").mkdir(parents=True, exist_ok=True)
    tulis_lingkungan(out / "lingkungan.txt")
    fval, fuji, fdes = out / "hasil_validasi.csv", out / "hasil_uji_final.csv", out / "deskripsi_data.csv"
    done = set(pd.read_csv(fuji)["run_id"]) if fuji.exists() else set()
    periode = f"{cfg['START']}..{cfg['END']}"

    total = (len(cfg["DATASETS"]) * len(cfg["HORIZONS"]) * len(cfg["TRAIN_FRACS"])
             * len(cfg["MODELS"]) * len(cfg["OPTIMIZERS"]))
    print(f"Maksimum {total} run (kombinasi tidak valid dilewati). Hasil: {fval}, {fuji}")
    t_all, ke = time.time(), 0

    for ds in cfg["DATASETS"]:
        info = C.DATA[ds]
        jenis, tf = info["jenis"], info["tf_menit"]
        harga = cfg["HARGA"][jenis]
        raw, dinfo = D.load(info["path"], jenis, tf, harga, cfg["START"], cfg["END"],
                            sumber_tf=info.get("sumber_tf"))
        print(f"\n=== {ds} | harga={harga} | n_bar={dinfo['n_bar']} | n_parsial={dinfo['n_parsial']}"
              f" | n_hilang={dinfo['n_hilang']} (celah {dinfo['n_celah']}, terpanjang "
              f"{dinfo['celah_terpanjang_mnt']} menit)")

        for hz in cfg["HORIZONS"]:
            if hz % tf != 0:
                print(f"  lewati horizon {hz} menit (bukan kelipatan TF {tf} menit)")
                continue
            data, feats, hb = D.build(raw, jenis, tf, hz, C.N_LAG, C.WINDOWS_MNT, cfg["START"])
            n = len(data)

            # ---- K3: blok uji final + purging berbasis waktu ----
            t_u = data.index[int(np.floor((1 - C.FINAL_TEST_FRAC) * n))]
            t_target = data.index + pd.Timedelta(minutes=hz)
            dev_full = data[t_target < t_u]
            test = data[data.index >= t_u]
            assert (dev_full.index + pd.Timedelta(minutes=hz) < t_u).all(), "purging gagal"

            _upsert_deskripsi(fdes, dict(dataset=ds, horizon_menit=hz, periode=periode,
                                         n_setelah_fitur=n, n_pengembangan=len(dev_full),
                                         n_uji_final=len(test), t_u=str(t_u), **dinfo))

            yte = test["target"]
            h_now, h_true = test["harga_now"].values, test["harga_next"].values
            naive = h_now.copy()
            m_naive = E.metrics(h_true, naive, np.zeros(len(yte)), h_now, test["ret_lag1"].values)
            e_naive = pd.Series(h_true - naive, index=test.index)

            for frac in cfg["TRAIN_FRACS"]:
                dev = dev_full
                if frac < 1.0:
                    t0f = dev.index.max() - (dev.index.max() - dev.index.min()) * frac
                    dev = dev[dev.index >= t0f]
                ydev = dev["target"]

                base = dict(dataset=ds, harga=harga, tf_data_menit=tf, horizon_menit=hz,
                            horizon_bar=hb, frac_latih=frac, loss=cfg["LOSS"],
                            early_stop=cfg["EARLY_STOP"], top_k=cfg["TOP_K"] or 0,
                            pruner=cfg["PRUNER"] or "none", n_trials_cfg=cfg["N_TRIALS"],
                            n_folds=cfg["N_FOLDS"], periode=periode, t_u=str(t_u),
                            n_latih=len(dev), latih_mulai=str(dev.index.min()),
                            latih_akhir=str(dev.index.max()), tag=a.tag)
                uji_id = dict(n_uji=len(test), uji_mulai=str(test.index.min()),
                              uji_akhir=str(test.index.max()))

                # baris Naive (sekali per dataset x horizon x frac x tag), hanya di uji final
                nid = f"{ds}|{harga}|h{hz}|fl{frac}|Naive|{cfg['START']}|{cfg['END']}|{a.tag}"
                if nid not in done:
                    row = dict(run_id=nid, status="ok", model="Naive", optimasi="-", **base,
                               **uji_id, **m_naive, MAE_vs_naive_pct=0.0, RMSE_vs_naive_pct=0.0)
                    row["Akurasi_arah_pct"] = np.nan
                    _append(fuji, row, KOLOM_UJI)
                    done.add(nid)

                for model, opt in itertools.product(cfg["MODELS"], cfg["OPTIMIZERS"]):
                    ke += 1
                    rid = (f"{ds}|{harga}|h{hz}|fl{frac}|{model}|{opt}|{cfg['LOSS']}|"
                           f"es{int(cfg['EARLY_STOP'])}|k{cfg['TOP_K'] or 0}|"
                           f"pr{cfg['PRUNER'] or 'none'}|t{cfg['N_TRIALS']}|f{cfg['N_FOLDS']}|"
                           f"{cfg['START']}|{cfg['END']}|{a.tag}")
                    if rid in done:
                        continue

                    # ---- K10: teknik yang tidak berlaku untuk AdaBoost ----
                    if model == "AdaBoost" and (cfg["EARLY_STOP"] or cfg["LOSS"] == "l1"):
                        row = dict(run_id=rid, status="tidak_berlaku", model=model,
                                   optimasi=opt, **base)
                        _append(fval, row, KOLOM_VALIDASI)
                        _append(fuji, {**row, **uji_id}, KOLOM_UJI)
                        done.add(rid)
                        print(f"  [{ke}] {ds} h{hz} fl{frac} {model:8s} {opt:7s} | tidak_berlaku")
                        continue

                    t0 = time.time()
                    Xdev = dev[feats]
                    tuned = O.tune(model, Xdev, ydev, opt, cfg, horizon_menit=hz,
                                   harga_now=dev["harga_now"], harga_next=dev["harga_next"])

                    # ---- K6: seleksi top-k final pada seluruh data pengembangan ----
                    cols = feats
                    if cfg["TOP_K"]:
                        cols = E.select_top_k(model, Xdev, ydev, cfg["TOP_K"], cfg["LOSS"], cfg["SEED"])
                    mdl = M.make(model, tuned["params"], cfg["LOSS"], cfg["SEED"])
                    mdl.fit(dev[cols], ydev)
                    t_total = time.time() - t0

                    vrow = dict(run_id=rid, status="ok", model=model, optimasi=opt, **base,
                                cv_mae_rp=tuned["cv_mae_rp"], cv_naive_rp=tuned["cv_naive_rp"],
                                cv_vs_naive_pct=tuned["cv_vs_naive_pct"],
                                n_trials=tuned["n_trials"], n_pruned=tuned["n_pruned"],
                                n_fitur=len(cols),
                                fitur_final=";".join(cols) if cfg["TOP_K"] else "",
                                detik_tuning=tuned["detik_tuning"], detik_total=t_total)
                    _append(fval, vrow, KOLOM_VALIDASI)

                    # ---- blok uji final ----
                    r_pred = mdl.predict(test[cols])
                    h_pred = h_now * np.exp(r_pred)
                    met = E.metrics(h_true, h_pred, r_pred, h_now, test["ret_lag1"].values)
                    e_m = pd.Series(h_true - h_pred, index=test.index)
                    dm1, p1 = E.dm_test(e_m.abs() - e_naive.abs(), hb, tf)
                    dm2, p2 = E.dm_test(e_m ** 2 - e_naive ** 2, hb, tf)
                    urow = dict(run_id=rid, status="ok", model=model, optimasi=opt, **base,
                                **uji_id, **met,
                                MAE_vs_naive_pct=(met["MAE_Rp"] / m_naive["MAE_Rp"] - 1) * 100,
                                RMSE_vs_naive_pct=(met["RMSE_Rp"] / m_naive["RMSE_Rp"] - 1) * 100,
                                DM_MAE=dm1, p_MAE=p1, DM_RMSE=dm2, p_RMSE=p2,
                                std_pred_return=float(np.std(r_pred)))
                    _append(fuji, urow, KOLOM_UJI)
                    done.add(rid)

                    safe = rid.replace("|", "__").replace(":", "-").replace(" ", "_")
                    json.dump(dict(params=tuned["params"], fitur_final=cols,
                                   fitur_fold=tuned["fitur_fold"]),
                              open(out / "params" / f"{safe}.json", "w"), indent=2, default=str)
                    if C.SIMPAN_PREDIKSI:
                        pd.DataFrame({"waktu_target": test.index + pd.Timedelta(minutes=hz),
                                      "harga_now": h_now, "harga_aktual": h_true,
                                      "prediksi": h_pred, "naive": naive}).to_csv(
                            out / "prediksi" / f"{safe}.csv", index=False)

                    # log hanya menampilkan skor validasi (blok uji dibuka lewat ringkas.py)
                    print(f"  [{ke}] {ds} h{hz} fl{frac} {model:8s} {opt:7s} | "
                          f"CV MAE Rp{tuned['cv_mae_rp']:,.0f} ({tuned['cv_vs_naive_pct']:+.3f}% vs Naive"
                          f" validasi) | {t_total:.0f}s", flush=True)

    print(f"\nSelesai dalam {(time.time()-t_all)/60:.1f} menit.")


if __name__ == "__main__":
    main()
