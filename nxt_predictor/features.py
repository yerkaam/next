"""Признаки: технический анализ, поток сделок и тональность новостей."""
from __future__ import annotations

import numpy as np
import pandas as pd

_analyzer = None

# VADER обучен на соцсетях и не знает биржевой лексики — добавляем её.
FINANCE_LEXICON = {
    "beat": 2.0, "beats": 2.0, "raises": 1.5, "raised": 1.5, "upgrade": 2.5, "upgraded": 2.5,
    "outperform": 2.0, "record": 1.5, "surge": 2.5, "surges": 2.5, "soar": 2.5, "soars": 2.5,
    "jump": 2.0, "jumps": 2.0, "rally": 2.0, "rallies": 2.0, "gain": 1.5, "gains": 1.5,
    "rise": 1.5, "rises": 1.5, "climb": 1.5, "climbs": 1.5, "bullish": 2.5, "buyback": 1.5,
    "order": 0.8, "contract": 0.8, "wins": 2.0, "growth": 1.5, "strong": 1.5,
    "miss": -2.0, "misses": -2.0, "missed": -2.0, "downgrade": -2.5, "downgraded": -2.5,
    "cut": -1.5, "cuts": -1.5, "lowers": -1.5, "lowered": -1.5, "fall": -2.0, "falls": -2.0,
    "drop": -2.0, "drops": -2.0, "plunge": -3.0, "plunges": -3.0, "slump": -2.5, "slumps": -2.5,
    "tumble": -2.5, "tumbles": -2.5, "sink": -2.0, "sinks": -2.0, "decline": -1.5, "declines": -1.5,
    "bearish": -2.5, "underperform": -2.0, "headwinds": -1.5, "tariff": -1.0, "tariffs": -1.0,
    "lawsuit": -2.0, "probe": -1.5, "investigation": -1.5, "weak": -1.5, "selloff": -2.5,
}


def sentiment(text: str) -> float:
    """Тональность текста от -1 до 1 (VADER + финансовый словарь)."""
    global _analyzer
    if _analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _analyzer = SentimentIntensityAnalyzer()
        _analyzer.lexicon.update(FINANCE_LEXICON)
    return _analyzer.polarity_scores(text or "")["compound"]


def score_news(news: pd.DataFrame) -> pd.DataFrame:
    news = news.copy()
    news["sentiment"] = [sentiment(f"{t}. {s}") for t, s in zip(news["title"], news["summary"])]
    return news


def daily_news_features(news: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Привязывает каждую новость к последнему торговому дню, на момент закрытия
    (или после) которого она уже известна для прогноза следующего дня."""
    out = pd.DataFrame(0.0, index=trading_days, columns=["news_count", "news_sent", "news_sent_max", "news_sent_min"])
    if news.empty:
        return out
    if "sentiment" not in news:
        news = score_news(news)
    local_dates = news["time"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    pos = trading_days.searchsorted(local_dates, side="right") - 1
    mask = pos >= 0
    tmp = pd.DataFrame({"day": trading_days[pos[mask]], "s": news["sentiment"].to_numpy()[mask]})
    g = tmp.groupby("day")["s"]
    agg = pd.DataFrame({"news_count": g.size(), "news_sent": g.mean(), "news_sent_max": g.max(), "news_sent_min": g.min()})
    out.update(agg)
    return out


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def context_features(daily: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Рынок и конкуренты: их доходности, относительная сила NXT, бета и корреляция."""
    f = pd.DataFrame(index=daily.index)
    ctx = context.reindex(daily.index).ffill(limit=3)
    own = np.log(daily["Close"]).diff()
    for t in ctx.columns:
        r = np.log(ctx[t]).diff()
        name = t.lower()
        f[f"{name}_ret_1"] = r
        f[f"{name}_ret_5"] = np.log(ctx[t] / ctx[t].shift(5))
        f[f"rel_{name}_5"] = np.log(daily["Close"] / daily["Close"].shift(5)) - f[f"{name}_ret_5"]
        if t in ("SPY", "TAN"):
            cov = own.rolling(60).cov(r)
            f[f"beta_{name}_60"] = cov / r.rolling(60).var()
            f[f"corr_{name}_60"] = own.rolling(60).corr(r)
    return f


def build_features(
    daily: pd.DataFrame, news: pd.DataFrame | None = None, context: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Таблица признаков на каждый торговый день + целевая переменная target
    (лог-доходность следующего дня)."""
    c, h, l, o, v = daily["Close"], daily["High"], daily["Low"], daily["Open"], daily["Volume"]
    logret = np.log(c).diff()
    f = pd.DataFrame(index=daily.index)

    for n in (1, 2, 3, 5, 10, 20):
        f[f"ret_{n}"] = np.log(c / c.shift(n))
    for n in (5, 20, 50):
        f[f"dist_sma{n}"] = c / c.rolling(n).mean() - 1
    f["vol_5"] = logret.rolling(5).std()
    f["vol_20"] = logret.rolling(20).std()
    f["vol_ratio"] = f["vol_5"] / f["vol_20"]
    f["rsi_14"] = _rsi(c) / 100
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9).mean()) / c
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_14"] = tr.rolling(14).mean() / c
    f["gap"] = np.log(o / c.shift())
    f["intraday_ret"] = np.log(c / o)
    f["range"] = (h - l) / c

    # Поток сделок (на дневных барах): где закрылись в диапазоне дня, аномальный объём, денежный поток.
    rng = (h - l).replace(0, np.nan)
    clv = ((c - l) - (h - c)) / rng
    f["close_location"] = clv
    logv = np.log(v.replace(0, np.nan))
    f["volume_z"] = (logv - logv.rolling(20).mean()) / logv.rolling(20).std()
    f["money_flow_5"] = (clv * v).rolling(5).sum() / v.rolling(5).sum()
    f["obv_slope"] = (np.sign(logret) * v).rolling(10).sum() / v.rolling(10).sum()

    f["dow"] = daily.index.dayofweek / 4
    for n in (5, 20):
        f[f"up_share_{n}"] = (logret > 0).rolling(n).mean()
    f["skew_20"] = logret.rolling(20).skew()
    f["high_52w"] = c / c.rolling(252, min_periods=60).max() - 1
    f["low_52w"] = c / c.rolling(252, min_periods=60).min() - 1

    if context is not None and not context.empty:
        f = f.join(context_features(daily, context))

    news_f = daily_news_features(news if news is not None else pd.DataFrame(columns=["time", "title", "summary"]), daily.index)
    f = f.join(news_f)
    f["news_sent_3d"] = f["news_sent"].rolling(3, min_periods=1).mean()
    f["news_count_5d"] = f["news_count"].rolling(5, min_periods=1).sum()

    f["target"] = logret.shift(-1)
    return f.replace([np.inf, -np.inf], np.nan)


def intraday_summary(intraday: pd.DataFrame) -> dict:
    """Сводка по сделкам последней сессии: последняя цена, VWAP, давление покупателей."""
    if intraday is None or intraday.empty:
        return {}
    last_day = intraday.index.normalize().max()
    s = intraday[intraday.index.normalize() == last_day]
    typical = (s["High"] + s["Low"] + s["Close"]) / 3
    vwap = float((typical * s["Volume"]).sum() / s["Volume"].sum())
    up = s["Close"].diff() > 0
    buy_vol = float(s.loc[up, "Volume"].sum())
    total = float(s["Volume"].sum())
    return {
        "session": str(last_day.date()),
        "last_trade_time": str(s.index[-1]),
        "last_price": float(s["Close"].iloc[-1]),
        "vwap": vwap,
        "session_volume": total,
        "buy_pressure": buy_vol / total if total else 0.5,
        "minutes": len(s),
    }
