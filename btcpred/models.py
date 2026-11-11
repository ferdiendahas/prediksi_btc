"""Definisi model, ruang pencarian hyperparameter, dan grid."""
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

ES_ROUNDS = 50          # kesabaran early stopping
ES_MAX_TREES = 2000     # jumlah pohon maksimum saat early stopping aktif

# Nama parameter "jumlah pohon" per model
N_TREES_KEY = {"XGBoost": "n_estimators", "LightGBM": "n_estimators",
               "CatBoost": "iterations", "AdaBoost": "n_estimators"}

DEFAULT = {"XGBoost": {}, "LightGBM": {}, "CatBoost": {}, "AdaBoost": {"max_depth": 3}}

# Grid Search: 3 x 3 x 3 = 27 kombinasi per model
GRID = {
    "XGBoost":  {"n_estimators": [50, 200, 500], "max_depth": [2, 4, 6],
                 "learning_rate": [0.001, 0.01, 0.1]},
    "LightGBM": {"n_estimators": [50, 200, 500], "num_leaves": [4, 15, 31],
                 "learning_rate": [0.001, 0.01, 0.1]},
    "CatBoost": {"iterations": [50, 200, 500], "depth": [2, 4, 6],
                 "learning_rate": [0.001, 0.01, 0.1]},
    "AdaBoost": {"n_estimators": [50, 150, 300], "max_depth": [2, 3, 5],
                 "learning_rate": [0.01, 0.1, 1.0]},
}


def space(name, t, early_stop=False):
    """Ruang pencarian untuk random/tpe/cmaes/ga/qmc."""
    if name == "XGBoost":
        p = dict(
            n_estimators=t.suggest_int("n_estimators", 10, 1000, log=True),
            max_depth=t.suggest_int("max_depth", 1, 8),
            learning_rate=t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            min_child_weight=t.suggest_int("min_child_weight", 1, 100, log=True),
            subsample=t.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_lambda=t.suggest_float("reg_lambda", 1e-3, 100, log=True),
            reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True),
        )
    elif name == "LightGBM":
        p = dict(
            n_estimators=t.suggest_int("n_estimators", 10, 1000, log=True),
            num_leaves=t.suggest_int("num_leaves", 2, 63, log=True),
            max_depth=t.suggest_int("max_depth", 1, 8),
            learning_rate=t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            min_child_samples=t.suggest_int("min_child_samples", 5, 300, log=True),
            subsample=t.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_lambda=t.suggest_float("reg_lambda", 1e-3, 100, log=True),
            reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True),
        )
        p["subsample_freq"] = 1
    elif name == "CatBoost":
        p = dict(
            iterations=t.suggest_int("iterations", 10, 1000, log=True),
            depth=t.suggest_int("depth", 1, 8),
            learning_rate=t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            l2_leaf_reg=t.suggest_float("l2_leaf_reg", 0.1, 100, log=True),
            random_strength=t.suggest_float("random_strength", 1e-3, 10, log=True),
            bagging_temperature=t.suggest_float("bagging_temperature", 0.0, 1.0),
        )
    elif name == "AdaBoost":
        p = dict(
            n_estimators=t.suggest_int("n_estimators", 10, 500, log=True),
            learning_rate=t.suggest_float("learning_rate", 1e-3, 1.0, log=True),
            loss=t.suggest_categorical("loss", ["linear", "square", "exponential"]),
            max_depth=t.suggest_int("max_depth", 1, 6),
        )
    else:
        raise ValueError(name)
    if early_stop and name != "AdaBoost":
        p[N_TREES_KEY[name]] = ES_MAX_TREES
    return p


def grid_params(name, t, early_stop=False):
    g = GRID[name]
    p = {k: t.suggest_categorical(k, v) for k, v in g.items()
         if not (early_stop and name != "AdaBoost" and k == N_TREES_KEY[name])}
    if name == "LightGBM":
        p["subsample_freq"] = 1
    if early_stop and name != "AdaBoost":
        p[N_TREES_KEY[name]] = ES_MAX_TREES
    return p


def grid_space(name, early_stop=False):
    g = dict(GRID[name])
    if early_stop and name != "AdaBoost":
        g.pop(N_TREES_KEY[name])
    return g


def make(name, params, loss="l2", seed=42, early_stop=False):
    prm = dict(params)
    if name == "XGBoost":
        if loss == "l1":
            prm["objective"] = "reg:absoluteerror"
        if early_stop:
            prm["early_stopping_rounds"] = ES_ROUNDS
        return XGBRegressor(**prm, random_state=seed, n_jobs=-1, verbosity=0)
    if name == "LightGBM":
        if loss == "l1":
            prm["objective"] = "l1"
        return LGBMRegressor(**prm, random_state=seed, n_jobs=-1, verbose=-1)
    if name == "CatBoost":
        if loss == "l1":
            prm["loss_function"] = "MAE"
        return CatBoostRegressor(**prm, random_seed=seed, verbose=0, thread_count=-1,
                                 allow_writing_files=False)
    if name == "AdaBoost":
        # AdaBoost.R2 punya parameter 'loss' sendiri (linear/square/exponential);
        # pengaturan LOSS l1/l2 tidak berlaku untuk AdaBoost.
        depth = prm.pop("max_depth", 3)
        return AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=depth),
                                 **prm, random_state=seed)
    raise ValueError(name)


def fit(name, model, Xtr, ytr, Xva=None, yva=None, early_stop=False):
    """Latih model; bila early stopping aktif, pakai (Xva, yva) sebagai data validasi.
    Mengembalikan jumlah pohon terbaik (atau None)."""
    if not early_stop or name == "AdaBoost" or Xva is None:
        model.fit(Xtr, ytr)
        return None
    if name == "XGBoost":
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return int(model.best_iteration) + 1
    if name == "LightGBM":
        import lightgbm as lgb
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False)])
        return int(model.best_iteration_ or model.n_estimators)
    if name == "CatBoost":
        model.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=ES_ROUNDS)
        return int(model.get_best_iteration()) + 1
