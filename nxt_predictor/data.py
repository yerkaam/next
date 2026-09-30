"""Загрузка данных: дневные цены, внутридневные сделки (минутные бары) и новости."""
from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Рынок, солнечный сектор и конкуренты Nextracker — их движения тоже влияют на NXT.
CONTEXT_TICKERS = ["SPY", "QQQ", "TAN", "FSLR", "ARRY", "SHLS", "ENPH"]


def _normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)[PRICE_COLUMNS].dropna()
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize() if (idx == idx.normalize()).all() else idx
    df.index.name = "Date"
    return df.sort_index()


def fetch_daily_prices(ticker: str, years: int = 10) -> pd.DataFrame:
    """Дневные OHLCV. Сначала Yahoo Finance, при ошибке — Stooq."""
    try:
        import yfinance as yf

        df = yf.Ticker(ticker).history(period=f"{years}y", interval="1d", auto_adjust=True)
        if not df.empty:
            return _normalize_prices(df)
    except Exception as exc:  # noqa: BLE001
        log.warning("Yahoo Finance недоступен: %s", exc)

    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), parse_dates=["Date"], index_col="Date")
    if df.empty:
        raise RuntimeError(f"Не удалось получить цены для {ticker}")
    cutoff = df.index.max() - pd.DateOffset(years=years)
    return _normalize_prices(df[df.index >= cutoff])


def fetch_context(tickers: list[str], years: int = 10) -> pd.DataFrame:
    """Цены закрытия рыночных индексов и конкурентов (колонка = тикер)."""
    closes = {}
    for t in tickers:
        try:
            closes[t] = fetch_daily_prices(t, years)["Close"]
        except Exception as exc:  # noqa: BLE001
            log.warning("Нет данных по %s: %s", t, exc)
    return pd.DataFrame(closes)


def fetch_intraday_trades(ticker: str, days: int = 7) -> pd.DataFrame:
    """Минутные бары (агрегированные сделки) за последние дни.

    Бесплатные источники не отдают поштучную ленту сделок, поэтому используем
    1-минутные бары Yahoo: каждая минута = все сделки за минуту (OHLC + объём).
    """
    import yfinance as yf

    df = yf.Ticker(ticker).history(period=f"{min(days, 7)}d", interval="1m", auto_adjust=False)
    if df.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    return _normalize_prices(df)


def fetch_news(ticker: str, company: str = "Nextracker") -> pd.DataFrame:
    """Новости: Yahoo Finance + Google News RSS. Колонки: time, title, summary, source."""
    rows: list[dict] = []
    try:
        import yfinance as yf

        for item in yf.Ticker(ticker).news or []:
            content = item.get("content", item)
            ts = content.get("pubDate") or content.get("providerPublishTime")
            rows.append(
                {
                    "time": pd.to_datetime(ts, unit="s" if isinstance(ts, (int, float)) else None, utc=True),
                    "title": content.get("title", ""),
                    "summary": content.get("summary", "") or "",
                    "source": "yahoo",
                }
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("Новости Yahoo недоступны: %s", exc)

    try:
        import feedparser

        q = requests.utils.quote(f"{company} OR {ticker} stock")
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en")
        for e in feed.entries:
            rows.append(
                {
                    "time": pd.to_datetime(e.get("published"), utc=True, errors="coerce"),
                    "title": e.get("title", ""),
                    "summary": "",
                    "source": "google",
                }
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("Google News недоступен: %s", exc)

    df = pd.DataFrame(rows, columns=["time", "title", "summary", "source"])
    df = df.dropna(subset=["time"]).drop_duplicates(subset=["title"])
    return df.sort_values("time").reset_index(drop=True)


def load_prices_csv(path: str) -> pd.DataFrame:
    """CSV с ценами: формат Yahoo, Nasdaq.com (Close/Last, $-цены) или свой."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    date_col = next(c for c in df.columns if c.lower() in ("date", "datetime", "time"))
    df = df.set_index(pd.to_datetime(df.pop(date_col)))
    aliases = {"close/last": "Close", "price": "Close", "vol.": "Volume"}
    df = df.rename(columns=lambda c: aliases.get(c.lower(), c))
    if "Close" not in df and "Adj Close" in df:
        df = df.rename(columns={"Adj Close": "Close"})
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"[$,\s]", "", regex=True), errors="coerce")
    return _normalize_prices(df)


def load_news_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for col in ("summary", "source"):
        if col not in df:
            df[col] = ""
    return df[["time", "title", "summary", "source"]].sort_values("time").reset_index(drop=True)


def synthetic_data(days: int = 1000, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Синтетические цены, минутки и новости — для демо и тестов без сети."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=datetime.now().date(), periods=days)
    news_shock = rng.choice([0, 1, -1], size=days, p=[0.85, 0.08, 0.07])
    rets = rng.normal(0.0006, 0.028, days) + 0.02 * np.roll(news_shock, 1)
    close = 40 * np.exp(np.cumsum(rets))
    open_ = close * np.exp(rng.normal(0, 0.008, days))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.012, days)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.012, days)))
    vol = rng.lognormal(15, 0.35, days) * (1 + 0.8 * np.abs(news_shock))
    daily = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=dates)
    daily.index.name = "Date"

    minutes = []
    for d, row in daily.tail(5).iterrows():
        t = pd.date_range(d + timedelta(hours=9, minutes=30), periods=390, freq="1min")
        path = row.Open * np.exp(np.cumsum(rng.normal(0, 0.0012, 390)))
        path = path * (row.Close / path[-1]) ** np.linspace(0, 1, 390)
        minutes.append(
            pd.DataFrame(
                {"Open": path, "High": path * 1.0007, "Low": path * 0.9993, "Close": path,
                 "Volume": rng.lognormal(9, 0.6, 390)},
                index=t,
            )
        )
    intraday = pd.concat(minutes)
    intraday.index.name = "Date"

    pos = ["Nextracker beats earnings estimates, raises guidance", "Nextracker wins record solar tracker order"]
    neg = ["Nextracker shares fall after downgrade", "Nextracker faces tariff headwinds and weak demand"]
    news = []
    for d, s in zip(dates, news_shock):
        if s:
            title = rng.choice(pos if s > 0 else neg)
            news.append({"time": pd.Timestamp(d, tz="UTC") + timedelta(hours=21), "title": title,
                         "summary": "", "source": "synthetic"})
    news_df = pd.DataFrame(news, columns=["time", "title", "summary", "source"])
    return daily, intraday, news_df


def synthetic_context(daily: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """Синтетические рынок и конкуренты, скоррелированные с ценой из synthetic_data."""
    rng = np.random.default_rng(seed)
    r = np.log(daily["Close"]).diff().fillna(0).to_numpy()
    out = {}
    for t, (beta, noise) in {"SPY": (0.25, 0.008), "QQQ": (0.3, 0.011), "TAN": (0.6, 0.015),
                             "FSLR": (0.7, 0.025), "ARRY": (0.9, 0.03)}.items():
        out[t] = 100 * np.exp(np.cumsum(beta * r + rng.normal(0, noise, len(r))))
    return pd.DataFrame(out, index=daily.index)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
