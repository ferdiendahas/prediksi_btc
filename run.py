"""
Menjalankan semua kombinasi percobaan dari config.py (bisa ditimpa lewat argumen).

Contoh:
    python run.py                                   # semua kombinasi di config.py
    python run.py --datasets candle_1m --horizons 5 15 30 60 --optimizers tpe --splits 0.8
    python run.py --models CatBoost --optimizers default grid random tpe cmaes ga --trials 50
    python run.py --early-stop --top-k 15 --loss l1 --pruner median

Hasil setiap run langsung ditambahkan ke hasil/hasil_semua.csv.
Run yang sudah pernah selesai (konfigurasi sama persis, termasuk --tag) otomatis
dilewati, jadi script aman dihentikan lalu dijalankan lagi. Karena --tag ikut masuk
run_id, konfigurasi yang sama dengan tag berbeda tetap dicatat ulang (perlu supaya
tahap 2/3/4 tidak kehilangan baris yang kebetulan sudah dijalankan di tahap 1).
"""
import argparse
import itertools
import json
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from btcpred import data as D, models as M, optim as O, evaluate as E


def parse():
    a = argparse.ArgumentParser()
    a.add_argument("--datasets", nargs="+")
    a.add_argument("--horizons", nargs="+", type=int)
    a.add_argument("--splits", nargs="+", type=float)
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


def main():
    a = parse()
    cfg = dict(
        DATASETS=a.datasets or C.DATASETS, HORIZONS=a.horizons or C.HORIZONS,
        SPLITS=a.splits or C.SPLITS, MODELS=a.models or C.MODELS,
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
    fres = out / "hasil_semua.csv"
    done = set(pd.read_csv(fres)["run_id"]) if fres.exists() else set()

    total = (len(cfg["DATASETS"]) * len(cfg["HORIZONS"]) * len(cfg["SPLITS"])
             * len(cfg["MODELS"]) * len(cfg["OPTIMIZERS"]))
    print(f"Maksimum {total} run (kombinasi tidak valid dilewati). Hasil: {fres}")
    t_all, ke = time.time(), 0

    for ds in cfg["DATASETS"]:
        info = C.DATA[ds]
        jenis, tf = info["jenis"], info["tf_menit"]
        harga = cfg["HARGA"][jenis]
        raw, n_gap = D.load(info["path"], jenis, tf, harga, cfg["START"], cfg["END"])
        print(f"\n=== {ds} | harga={harga} | bar hilang di file: {n_gap}")

        for hz in cfg["HORIZONS"]:
            if hz % tf != 0:
                print(f"  lewati horizon {hz} menit (bukan kelipatan TF {tf} menit)")
                continue
            data, feats, hb = D.build(raw, jenis, tf, hz, C.N_LAG, C.WINDOWS_MNT, cfg["START"])
            n = len(data)

            for sp in cfg["SPLITS"]:
                cut = int(n * sp)
                train = data.iloc[: cut - hb]          # purge: buang hb baris terakhir
                test = data.iloc[cut:]                 # agar target latih tidak masuk periode uji
                ytr, yte = train["target"], test["target"]
                h_now, h_true = test["harga_now"].values, test["harga_next"].values
                naive = h_now.copy()
                e_naive = h_true - naive
                m_naive = E.metrics(h_true, naive, yte.values, np.zeros(len(yte)))
                split_lbl = f"{int(round(sp*100))}:{int(round((1-sp)*100))}"

                base = dict(dataset=ds, harga=harga, tf_data_menit=tf, horizon_menit=hz,
                            horizon_bar=hb, split=split_lbl, n_latih=len(train),
                            n_uji=len(test), latih_mulai=str(train.index.min()),
                            uji_mulai=str(test.index.min()), uji_akhir=str(test.index.max()),
                            loss=cfg["LOSS"], early_stop=cfg["EARLY_STOP"],
                            top_k=cfg["TOP_K"] or 0, pruner=cfg["PRUNER"] or "none",
                            n_trials_cfg=cfg["N_TRIALS"], n_folds=cfg["N_FOLDS"],
                            periode=f"{cfg['START']}..{cfg['END']}", tag=a.tag)

                # baris Naive (sekali per dataset x horizon x split)
                nid = f"{ds}|{harga}|h{hz}|{split_lbl}|Naive|{cfg['START']}|{cfg['END']}|{a.tag}"
                if nid not in done:
                    row = dict(run_id=nid, model="Naive", optimasi="-", **base, **m_naive,
                               MAE_vs_naive_pct=0.0, RMSE_vs_naive_pct=0.0)
                    row["Akurasi_arah_pct"] = np.nan
                    _append(fres, row)
                    done.add(nid)

                for model, opt in itertools.product(cfg["MODELS"], cfg["OPTIMIZERS"]):
                    ke += 1
                    rid = (f"{ds}|{harga}|h{hz}|{split_lbl}|{model}|{opt}|{cfg['LOSS']}|"
                           f"es{int(cfg['EARLY_STOP'])}|k{cfg['TOP_K'] or 0}|"
                           f"pr{cfg['PRUNER'] or 'none'}|t{cfg['N_TRIALS']}|f{cfg['N_FOLDS']}|"
                           f"{cfg['START']}|{cfg['END']}|{a.tag}")
                    if rid in done:
                        continue
                    t0 = time.time()
                    cols = feats
                    if cfg["TOP_K"]:
                        cols = E.select_top_k(model, train[feats], ytr, cfg["TOP_K"],
                                              cfg["LOSS"], cfg["SEED"])
                    Xtr, Xte = train[cols], test[cols]

                    tuned = O.tune(model, Xtr, ytr, opt, cfg, gap=hb)
                    mdl = M.make(model, tuned["params"], cfg["LOSS"], cfg["SEED"])
                    mdl.fit(Xtr, ytr)
                    r_pred = mdl.predict(Xte)
                    h_pred = h_now * np.exp(r_pred)
                    met = E.metrics(h_true, h_pred, yte.values, r_pred)
                    e_m = h_true - h_pred
                    dm1, p1 = E.dm_test(e_m, e_naive, hb, power=1)
                    dm2, p2 = E.dm_test(e_m, e_naive, hb, power=2)

                    row = dict(run_id=rid, model=model, optimasi=opt, **base, **met,
                               MAE_vs_naive_pct=(met["MAE_Rp"] / m_naive["MAE_Rp"] - 1) * 100,
                               RMSE_vs_naive_pct=(met["RMSE_Rp"] / m_naive["RMSE_Rp"] - 1) * 100,
                               DM_MAE=dm1, p_MAE=p1, DM_RMSE=dm2, p_RMSE=p2,
                               cv_mae_return=tuned["cv_mae"], n_trials=tuned["n_trials"],
                               n_pruned=tuned["n_pruned"], n_fitur=len(cols),
                               std_pred_return=float(np.std(r_pred)),
                               detik_tuning=tuned["detik_tuning"],
                               detik_total=time.time() - t0)
                    _append(fres, row)
                    done.add(rid)

                    safe = rid.replace("|", "__").replace(":", "-").replace(" ", "_")
                    json.dump(dict(params=tuned["params"], fitur=cols), open(
                        out / "params" / f"{safe}.json", "w"), indent=2, default=str)
                    if C.SIMPAN_PREDIKSI:
                        pd.DataFrame({"waktu_target": test.index + pd.Timedelta(minutes=hz),
                                      "harga_now": h_now, "harga_aktual": h_true,
                                      "prediksi": h_pred, "naive": naive}).to_csv(
                            out / "prediksi" / f"{safe}.csv", index=False)

                    print(f"  [{ke}] {ds} h{hz} {split_lbl} {model:8s} {opt:7s} | "
                          f"MAE Rp{met['MAE_Rp']:,.0f} ({row['MAE_vs_naive_pct']:+.2f}% vs Naive) "
                          f"| RMSE {row['RMSE_vs_naive_pct']:+.2f}% | arah {met['Akurasi_arah_pct']:.1f}% "
                          f"| {row['detik_total']:.0f}s", flush=True)

    print(f"\nSelesai dalam {(time.time()-t_all)/60:.1f} menit.")


KOLOM = ["run_id", "dataset", "harga", "tf_data_menit", "horizon_menit", "horizon_bar",
         "split", "model", "optimasi", "loss", "early_stop", "top_k", "pruner",
         "n_trials_cfg", "n_folds", "MAE_Rp", "RMSE_Rp", "MAPE_pct", "Akurasi_arah_pct",
         "MAE_vs_naive_pct", "RMSE_vs_naive_pct", "DM_MAE", "p_MAE", "DM_RMSE", "p_RMSE",
         "cv_mae_return", "n_trials", "n_pruned", "n_fitur", "std_pred_return",
         "detik_tuning", "detik_total", "n_latih", "n_uji", "latih_mulai", "uji_mulai",
         "uji_akhir", "periode", "tag"]


def _append(path, row):
    df = pd.DataFrame([{k: row.get(k, np.nan) for k in KOLOM}])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


if __name__ == "__main__":
    main()
