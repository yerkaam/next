"""Модель: подбор гиперпараметров на временных разбиениях, взвешенный ансамбль
регрессий на доходность следующего дня, классификатор направления и квантили."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

QUANTILES = (0.1, 0.9)

# Модель и сетка гиперпараметров, которые перебираются при обучении.
CANDIDATES = {
    "ridge": (
        make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 30))),
        {},
    ),
    "gbm": (
        GradientBoostingRegressor(loss="huber", subsample=0.8, random_state=0),
        {"n_estimators": [150, 400], "max_depth": [2, 3], "learning_rate": [0.01, 0.03]},
    ),
    "hgb": (
        HistGradientBoostingRegressor(loss="absolute_error", random_state=0),
        {"max_depth": [2, 4], "learning_rate": [0.02, 0.06], "max_iter": [150, 400], "l2_regularization": [0.0, 1.0]},
    ),
    "rf": (
        RandomForestRegressor(n_estimators=300, max_features=0.4, n_jobs=-1, random_state=0),
        {"max_depth": [4, 8], "min_samples_leaf": [20, 60]},
    ),
    "et": (
        ExtraTreesRegressor(n_estimators=300, max_features=0.5, n_jobs=-1, random_state=0),
        {"max_depth": [4, 8], "min_samples_leaf": [20, 60]},
    ),
}


def _classifiers():
    return {
        "logit": make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)),
        "hgb_clf": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=200, l2_regularization=1.0, random_state=0),
    }


def _quantile_model(q: float):
    return GradientBoostingRegressor(
        loss="quantile", alpha=q, n_estimators=200, max_depth=2, learning_rate=0.05, random_state=0
    )


@dataclass
class Ensemble:
    """Обученные на одном наборе данных модели с весами."""
    params: dict
    weights: dict
    models: dict = field(default_factory=dict)
    classifiers: dict = field(default_factory=dict)

    def fit(self, X, y):
        for name, p in self.params.items():
            self.models[name] = clone(CANDIDATES[name][0]).set_params(**p).fit(X, y)
        for name, c in _classifiers().items():
            self.classifiers[name] = c.fit(X, (y > 0).astype(int))
        return self

    def predict_all(self, X) -> dict[str, np.ndarray]:
        out = {k: m.predict(X) for k, m in self.models.items()}
        out["ensemble"] = sum(self.weights[k] * out[k] for k in self.models)
        return out

    def prob_up(self, X) -> np.ndarray:
        return np.mean([c.predict_proba(X)[:, 1] for c in self.classifiers.values()], axis=0)


def tune(X: pd.DataFrame, y: pd.Series, n_splits: int = 4, fast: bool = False) -> tuple[dict, dict, dict]:
    """Перебирает гиперпараметры на TimeSeriesSplit (обучение всегда в прошлом,
    проверка — в будущем). Возвращает лучшие параметры, веса ансамбля и MAE моделей."""
    cv = TimeSeriesSplit(n_splits=n_splits)
    best_params, best_mae = {}, {}
    for name, (base, grid) in CANDIDATES.items():
        combos = [{}] if fast or not grid else list(ParameterGrid(grid))
        scores = []
        for p in combos:
            errs = []
            for tr, va in cv.split(X):
                m = clone(base).set_params(**p).fit(X.iloc[tr], y.iloc[tr])
                errs.append(np.mean(np.abs(m.predict(X.iloc[va]) - y.iloc[va])))
            scores.append(np.mean(errs))
        i = int(np.argmin(scores))
        best_params[name], best_mae[name] = combos[i], float(scores[i])
        log.info("%s: MAE %.4f, параметры %s", name, scores[i], combos[i])

    # Веса ∝ 1/MAE² — точные модели весят больше, но слабые не выкидываются полностью.
    inv = {k: 1 / v**2 for k, v in best_mae.items()}
    total = sum(inv.values())
    weights = {k: v / total for k, v in inv.items()}
    return best_params, weights, best_mae


@dataclass
class Prediction:
    last_date: str
    last_close: float
    predicted_return: float
    predicted_price: float
    low_price: float
    high_price: float
    prob_up: float
    per_model: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    cv_mae: dict = field(default_factory=dict)
    backtest: dict = field(default_factory=dict)
    top_features: list = field(default_factory=list)
    n_features: int = 0
    n_train_days: int = 0


def walk_forward(features: pd.DataFrame, params: dict, weights: dict, test_days: int = 250, step: int = 20) -> dict:
    """Честная проверка: модель учится только на прошлом и предсказывает следующие дни."""
    data = features.dropna()
    X, y = data.drop(columns="target"), data["target"]
    test_days = min(test_days, len(data) // 3)
    start = len(data) - test_days
    preds, probs, actual, dates = [], [], [], []
    for i in range(start, len(data), step):
        end = min(i + step, len(data))
        ens = Ensemble(params, weights).fit(X.iloc[:i], y.iloc[:i])
        preds.extend(ens.predict_all(X.iloc[i:end])["ensemble"])
        probs.extend(ens.prob_up(X.iloc[i:end]))
        actual.extend(y.iloc[i:end])
        dates.extend(data.index[i:end])
    preds, probs, actual = np.array(preds), np.array(probs), np.array(actual)
    nonzero = actual != 0
    up = (actual > 0).astype(float)
    return {
        "days": int(len(actual)),
        "mae_model": float(np.mean(np.abs(preds - actual))),
        "mae_naive": float(np.mean(np.abs(actual))),  # «завтра цена = сегодня»
        "direction_accuracy": float(np.mean(np.sign(preds[nonzero]) == np.sign(actual[nonzero]))),
        "classifier_accuracy": float(np.mean((probs > 0.5) == (actual > 0))),
        "brier": float(np.mean((probs - up) ** 2)),
        "brier_naive": float(np.mean((up.mean() - up) ** 2)),
        "share_up_days": float(np.mean(actual > 0)),
        "series": pd.DataFrame({"pred": preds, "prob_up": probs, "actual": actual}, index=pd.DatetimeIndex(dates)),
    }


def predict_next(features: pd.DataFrame, last_close: float, backtest_days: int = 250, fast: bool = False) -> Prediction:
    train = features.dropna()
    X, y = train.drop(columns="target"), train["target"]
    x_last = features.drop(columns="target").iloc[[-1]].ffill().fillna(0)

    # Параметры подбираются только на данных ДО периода проверки, чтобы бэктест был честным.
    bt_days = min(backtest_days, len(train) // 3)
    params, weights, cv_mae = tune(X.iloc[:-bt_days], y.iloc[:-bt_days], fast=fast)

    ens = Ensemble(params, weights).fit(X, y)
    point = ens.predict_all(x_last)
    r = float(point["ensemble"][0])
    q = {qq: float(_quantile_model(qq).fit(X, y).predict(x_last)[0]) for qq in QUANTILES}
    lo, hi = min(q[QUANTILES[0]], r), max(q[QUANTILES[1]], r)
    prob_up = float(ens.prob_up(x_last)[0])

    imp = pd.Series(0.0, index=X.columns)
    for name in ("gbm", "rf", "et"):
        imp += weights[name] * pd.Series(ens.models[name].feature_importances_, index=X.columns)
    imp = (imp / imp.sum()).sort_values(ascending=False).head(10)

    bt = walk_forward(features, params, weights, test_days=backtest_days)
    return Prediction(
        last_date=str(features.index[-1].date()),
        last_close=last_close,
        predicted_return=r,
        predicted_price=last_close * np.exp(r),
        low_price=last_close * np.exp(lo),
        high_price=last_close * np.exp(hi),
        prob_up=prob_up,
        per_model={k: float(v[0]) for k, v in point.items()},
        weights=weights,
        params=params,
        cv_mae=cv_mae,
        backtest=bt,
        top_features=[(k, float(v)) for k, v in imp.items()],
        n_features=X.shape[1],
        n_train_days=len(X),
    )
