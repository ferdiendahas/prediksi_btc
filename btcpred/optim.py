"""Optimasi hyperparameter: default, grid, random, tpe, cmaes, ga, qmc.
Semua metode memakai fungsi objektif yang sama: rata-rata MAE harga (rupiah)
pada walk-forward CV (TimeSeriesSplit + purging berbasis waktu) di data
pengembangan saja."""
import time
import warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit

from . import models as M
from .evaluate import select_top_k

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def make_sampler(method, seed):
    s = optuna.samplers
    if method == "random":
        return s.RandomSampler(seed=seed)
    if method == "tpe":
        return s.TPESampler(seed=seed, n_startup_trials=10)
    if method == "cmaes":
        return s.CmaEsSampler(seed=seed, n_startup_trials=5, warn_independent_sampling=False)
    if method == "ga":
        return s.NSGAIISampler(seed=seed, population_size=10)
    if method == "qmc":
        return s.QMCSampler(qmc_type="sobol", scramble=True, seed=seed,
                            warn_independent_sampling=False)
    raise ValueError(f"metode optimasi tidak dikenal: {method}")


def make_pruner(pruner):
    if pruner is None:
        return optuna.pruners.NopPruner()
    if pruner == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    if pruner == "hyperband":
        return optuna.pruners.HyperbandPruner()
    raise ValueError(pruner)


def _folds(X, n_folds, horizon_menit):
    """TimeSeriesSplit tanpa gap, lalu purging berbasis waktu: baris latih fold
    dibuang bila waktu target (t + horizon) tidak lebih awal dari awal validasi."""
    hz = pd.Timedelta(minutes=horizon_menit)
    out = []
    for tr, va in TimeSeriesSplit(n_splits=n_folds).split(X):
        t_val0 = X.index[va[0]]
        tr = tr[(X.index[tr] + hz) < t_val0]
        if len(tr) == 0:
            raise ValueError("fold tanpa data latih setelah purging; kurangi n_folds")
        out.append((tr, va))
    return out


def cv_score(name, params, X, y, harga_now, harga_next, folds, loss, seed, early_stop,
             fitur_fold=None, trial=None):
    """Walk-forward CV. Objektif = MAE harga (rupiah) rata-rata fold.

    folds      : daftar (idx_latih, idx_validasi) dari _folds().
    fitur_fold : None (semua kolom) atau daftar kolom per fold (seleksi top-k di
                 dalam fold).
    Return dict: mae_rp, naive_rp, best_n."""
    hn, hx = harga_now.to_numpy(), harga_next.to_numpy()

    def one(i, tr, va):
        cols = fitur_fold[i] if fitur_fold is not None else list(X.columns)
        Xtr, Xva = X.iloc[tr][cols], X.iloc[va][cols]
        m = M.make(name, params, loss, seed, early_stop and name != "AdaBoost")
        bn = M.fit(name, m, Xtr, y.iloc[tr], Xva, y.iloc[va], early_stop)
        p_hat = hn[va] * np.exp(m.predict(Xva))
        mae_m = float(np.mean(np.abs(hx[va] - p_hat)))
        mae_n = float(np.mean(np.abs(hx[va] - hn[va])))
        return mae_m, mae_n, bn

    if name == "AdaBoost":
        # AdaBoost (scikit-learn) hanya memakai 1 core, jadi fold dijalankan paralel.
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=-1)(delayed(one)(i, tr, va) for i, (tr, va) in enumerate(folds))
        if trial is not None:
            for i in range(len(res)):
                trial.report(float(np.mean([r[0] for r in res[: i + 1]])), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
    else:
        res = []
        for i, (tr, va) in enumerate(folds):
            res.append(one(i, tr, va))
            if trial is not None:
                trial.report(float(np.mean([r[0] for r in res])), i)
                if trial.should_prune():
                    raise optuna.TrialPruned()
    best_n = [r[2] for r in res if r[2] is not None]
    return dict(mae_rp=float(np.mean([r[0] for r in res])),
                naive_rp=float(np.mean([r[1] for r in res])),
                best_n=int(np.mean(best_n)) if best_n else None)


def tune(name, X, y, method, cfg, horizon_menit, harga_now, harga_next):
    """Tuning pada data pengembangan saja.

    Return dict: params, cv_mae_rp, cv_naive_rp, cv_vs_naive_pct, n_trials,
    n_pruned, detik_tuning, fitur_fold (daftar kolom per fold bila top-k)."""
    t0 = time.time()
    es = cfg["EARLY_STOP"]
    folds = _folds(X, cfg["N_FOLDS"], horizon_menit)
    fitur_fold = None
    if cfg["TOP_K"]:
        # importance dihitung dengan model bawaan pada data latih tiap fold.
        # Tidak bergantung pada hyperparameter trial, jadi cukup sekali per fold.
        fitur_fold = [select_top_k(name, X.iloc[tr], y.iloc[tr], cfg["TOP_K"],
                                   cfg["LOSS"], cfg["SEED"]) for tr, _ in folds]
    kw = dict(harga_now=harga_now, harga_next=harga_next, folds=folds, loss=cfg["LOSS"],
              seed=cfg["SEED"], early_stop=es, fitur_fold=fitur_fold)

    def default_params():
        p = dict(M.DEFAULT[name])
        if es and name != "AdaBoost":
            p[M.N_TREES_KEY[name]] = M.ES_MAX_TREES
        return p

    if method == "default":
        params = default_params()
        sc = cv_score(name, params, X, y, **kw)
        n_tr, n_pr = 1, 0
    else:
        hasil = {}

        def objective(t):
            p = M.grid_params(name, t, es) if method == "grid" else M.space(name, t, es)
            sc_t = cv_score(name, p, X, y, trial=t, **kw)
            hasil[t.number] = sc_t
            return sc_t["mae_rp"]

        if method == "grid":
            g = M.grid_space(name, es)
            sampler = optuna.samplers.GridSampler(g, seed=cfg["SEED"])
            n_trials = int(np.prod([len(v) for v in g.values()]))
        else:
            sampler = make_sampler(method, cfg["SEED"])
            n_trials = cfg["N_TRIALS"]
        study = optuna.create_study(direction="minimize", sampler=sampler,
                                    pruner=make_pruner(cfg["PRUNER"]))
        study.optimize(objective, n_trials=n_trials)
        n_tr = len(study.trials)
        n_pr = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
        if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            # semua trial gagal/dipangkas -> pakai setelan bawaan supaya run tidak batal
            params = default_params()
            sc = cv_score(name, params, X, y, **kw)
        else:
            bt = study.best_trial
            params = dict(bt.params)
            if name == "LightGBM":
                params["subsample_freq"] = 1
            if es and name != "AdaBoost":
                params[M.N_TREES_KEY[name]] = M.ES_MAX_TREES
            sc = hasil[bt.number]

    # Early stopping: model final memakai rata-rata jumlah pohon terbaik dari CV
    if es and name != "AdaBoost" and sc["best_n"] is not None:
        params[M.N_TREES_KEY[name]] = max(1, sc["best_n"])
    return dict(params=params, cv_mae_rp=sc["mae_rp"], cv_naive_rp=sc["naive_rp"],
                cv_vs_naive_pct=(sc["mae_rp"] / sc["naive_rp"] - 1) * 100,
                n_trials=n_tr, n_pruned=n_pr, detik_tuning=time.time() - t0,
                fitur_fold=fitur_fold)
