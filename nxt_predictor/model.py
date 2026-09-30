"""Модель: ансамбль регрессий на доходность следующего дня + квантили для диапазона."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

QUANTILES = (0.1, 0.9)


def _point_models():
    return {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 20))),
        "gbm": GradientBoostingRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.03, subsample=0.8, random_state=0
        ),
    }


def _quantile_model(q: float):
    return GradientBoostingRegressor(
        loss="quantile", alpha=q, n_estimators=150, max_depth=2, learning_rate=0.05, random_state=0
    )


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
    backtest: dict = field(default_factory=dict)
    top_features: list = field(default_factory=list)


def _fit_predict(X_train, y_train, X_pred) -> dict[str, np.ndarray]:
    out = {}
    for name, m in _point_models().items():
        m.fit(X_train, y_train)
        out[name] = m.predict(X_pred)
    out["ensemble"] = np.mean([out[k] for k in ("ridge", "gbm")], axis=0)
    return out


def walk_forward(features: pd.DataFrame, test_days: int = 120, step: int = 20) -> dict:
    """Честная проверка: модель учится только на прошлом и предсказывает следующие дни."""
    data = features.dropna()
    X, y = data.drop(columns="target"), data["target"]
    test_days = min(test_days, len(data) // 3)
    start = len(data) - test_days
    preds, actual, dates = [], [], []
    for i in range(start, len(data), step):
        end = min(i + step, len(data))
        p = _fit_predict(X.iloc[:i], y.iloc[:i], X.iloc[i:end])["ensemble"]
        preds.extend(p)
        actual.extend(y.iloc[i:end])
        dates.extend(data.index[i:end])
    preds, actual = np.array(preds), np.array(actual)
    nonzero = actual != 0
    return {
        "days": int(len(actual)),
        "mae_model": float(np.mean(np.abs(preds - actual))),
        "mae_naive": float(np.mean(np.abs(actual))),  # «завтра цена = сегодня»
        "direction_accuracy": float(np.mean(np.sign(preds[nonzero]) == np.sign(actual[nonzero]))),
        "share_up_days": float(np.mean(actual > 0)),
        "series": pd.DataFrame({"pred": preds, "actual": actual}, index=pd.DatetimeIndex(dates)),
    }


def predict_next(features: pd.DataFrame, last_close: float, backtest_days: int = 120) -> Prediction:
    train = features.dropna()
    X, y = train.drop(columns="target"), train["target"]
    x_last = features.drop(columns="target").iloc[[-1]].ffill().fillna(0)

    point = _fit_predict(X, y, x_last)
    r = float(point["ensemble"][0])
    q = {}
    for qq in QUANTILES:
        m = _quantile_model(qq).fit(X, y)
        q[qq] = float(m.predict(x_last)[0])
    lo, hi = min(q[QUANTILES[0]], r), max(q[QUANTILES[1]], r)

    # Вероятность роста: сдвигаем эмпирическое распределение ошибок модели на прогноз.
    resid = y.to_numpy() - _fit_predict(X, y, X)["ensemble"]
    prob_up = float(np.mean(resid + r > 0))

    gbm = _point_models()["gbm"].fit(X, y)
    imp = sorted(zip(X.columns, gbm.feature_importances_), key=lambda t: -t[1])[:8]

    bt = walk_forward(features, test_days=backtest_days)
    return Prediction(
        last_date=str(features.index[-1].date()),
        last_close=last_close,
        predicted_return=r,
        predicted_price=last_close * np.exp(r),
        low_price=last_close * np.exp(lo),
        high_price=last_close * np.exp(hi),
        prob_up=prob_up,
        per_model={k: float(v[0]) for k, v in point.items()},
        backtest=bt,
        top_features=[(k, float(v)) for k, v in imp],
    )
