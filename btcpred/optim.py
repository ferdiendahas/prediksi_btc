"""Optimasi hyperparameter: default, grid, random, tpe, cmaes, ga, qmc.
Semua metode memakai fungsi objektif yang sama: rata-rata MAE return pada
walk-forward CV (TimeSeriesSplit) di data latih saja."""
import time
import warnings
import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

from . import models as M

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


def cv_score(name, params, X, y, n_folds, gap, loss, seed, early_stop, trial=None):
    """Walk-forward CV. Return (MAE rata-rata, jumlah pohon terbaik rata-rata/None)."""
    tscv = TimeSeriesSplit(n_splits=n_folds, gap=gap)
    if name == "AdaBoost":
        # AdaBoost (scikit-learn) hanya memakai 1 core, jadi fold dijalankan paralel.
        from joblib import Parallel, delayed

        def one(tr, va):
            m = M.make(name, params, loss, seed)
            m.fit(X.iloc[tr], y.iloc[tr])
            return mean_absolute_error(y.iloc[va], m.predict(X.iloc[va]))
        errs = Parallel(n_jobs=-1)(delayed(one)(tr, va) for tr, va in tscv.split(X))
        if trial is not None:
            for i in range(len(errs)):
                trial.report(float(np.mean(errs[: i + 1])), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(errs)), None
    errs, best_n = [], []
    for i, (tr, va) in enumerate(tscv.split(X)):
        m = M.make(name, params, loss, seed, early_stop and name != "AdaBoost")
        bn = M.fit(name, m, X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va], early_stop)
        if bn is not None:
            best_n.append(bn)
        errs.append(mean_absolute_error(y.iloc[va], m.predict(X.iloc[va])))
        if trial is not None:
            trial.report(float(np.mean(errs)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(errs)), (int(np.mean(best_n)) if best_n else None)


def tune(name, X, y, method, cfg, gap):
    """Return dict: params final, cv_mae, n_trials, n_pruned, detik."""
    t0 = time.time()
    es = cfg["EARLY_STOP"]
    kw = dict(n_folds=cfg["N_FOLDS"], gap=gap, loss=cfg["LOSS"], seed=cfg["SEED"], early_stop=es)

    if method == "default":
        params = dict(M.DEFAULT[name])
        if es and name != "AdaBoost":
            params[M.N_TREES_KEY[name]] = M.ES_MAX_TREES
        cv, bn = cv_score(name, params, X, y, **kw)
        n_tr, n_pr = 1, 0
    else:
        best_n = {}

        def objective(t):
            p = M.grid_params(name, t, es) if method == "grid" else M.space(name, t, es)
            score, bn = cv_score(name, p, X, y, trial=t, **kw)
            best_n[t.number] = bn
            return score

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
        if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            # semua trial gagal/dipangkas -> pakai setelan bawaan supaya run tidak batal
            params = dict(M.DEFAULT[name])
            if es and name != "AdaBoost":
                params[M.N_TREES_KEY[name]] = M.ES_MAX_TREES
            cv, bn = cv_score(name, params, X, y, **kw)
            return dict(params=params, cv_mae=cv, n_trials=len(study.trials),
                        n_pruned=sum(t.state == optuna.trial.TrialState.PRUNED
                                     for t in study.trials),
                        detik_tuning=time.time() - t0)
        bt = study.best_trial
        params = dict(bt.params)
        if name == "LightGBM":
            params["subsample_freq"] = 1
        if es and name != "AdaBoost":
            params[M.N_TREES_KEY[name]] = M.ES_MAX_TREES
        cv, bn = bt.value, best_n.get(bt.number)
        n_tr = len(study.trials)
        n_pr = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)

    # Early stopping: model final memakai rata-rata jumlah pohon terbaik dari CV
    if es and name != "AdaBoost" and bn is not None:
        params[M.N_TREES_KEY[name]] = max(1, bn)
    return dict(params=params, cv_mae=cv, n_trials=n_tr, n_pruned=n_pr,
                detik_tuning=time.time() - t0)
