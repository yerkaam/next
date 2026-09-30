"""CLI: python -m nxt_predictor [--ticker NXT] [--demo] [--prices-csv FILE] [--news-csv FILE]"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from . import data
from .features import build_features, intraday_summary, score_news
from .model import predict_next
from .report import html_report, text_report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Прогноз следующей цены акции по ценам, сделкам и новостям")
    ap.add_argument("--ticker", default="NXT")
    ap.add_argument("--company", default="Nextracker", help="название компании для поиска новостей")
    ap.add_argument("--years", type=int, default=10, help="сколько лет истории загрузить")
    ap.add_argument("--context", default=",".join(data.CONTEXT_TICKERS),
                    help="рынок/конкуренты через запятую; 'none' — не использовать")
    ap.add_argument("--backtest-days", type=int, default=250, help="длина честной проверки на истории")
    ap.add_argument("--fast", action="store_true", help="без подбора гиперпараметров (быстрее)")
    ap.add_argument("-v", "--verbose", action="store_true", help="показывать ход обучения")
    ap.add_argument("--demo", action="store_true", help="синтетические данные, без интернета")
    ap.add_argument("--prices-csv", help="свои дневные цены (Date,Open,High,Low,Close,Volume)")
    ap.add_argument("--news-csv", help="свои новости (time,title[,summary])")
    ap.add_argument("--out", default="output", help="папка для отчётов")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")

    context_tickers = [] if args.context.lower() == "none" else [t.strip().upper() for t in args.context.split(",") if t.strip()]
    if args.demo:
        daily, intraday, news = data.synthetic_data()
        context = data.synthetic_context(daily) if context_tickers else pd.DataFrame()
    else:
        daily = data.load_prices_csv(args.prices_csv) if args.prices_csv else data.fetch_daily_prices(args.ticker, args.years)
        news = data.load_news_csv(args.news_csv) if args.news_csv else data.fetch_news(args.ticker, args.company)
        try:
            intraday = data.fetch_intraday_trades(args.ticker)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Минутные сделки недоступны: %s", exc)
            intraday = pd.DataFrame()
        context = data.fetch_context(context_tickers, args.years) if context_tickers else pd.DataFrame()

    news = score_news(news)
    features = build_features(daily, news, context)
    print(f"Обучение на {len(features.dropna())} днях, {features.shape[1] - 1} признаков"
          f"{' (быстрый режим)' if args.fast else ', подбор гиперпараметров'}...", flush=True)
    last_close = float(daily["Close"].iloc[-1])
    pred = predict_next(features, last_close=last_close, backtest_days=args.backtest_days, fast=args.fast)
    print("Прогноз цены открытия...", flush=True)
    pred_open = predict_next(build_features(daily, news, context, target="open"), last_close=last_close,
                             backtest_days=args.backtest_days, fast=args.fast)
    intra = intraday_summary(intraday)

    print(text_report(args.ticker, pred, intra, news, pred_open))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.ticker}_report.html").write_text(html_report(args.ticker, pred, daily, intra, news, pred_open), encoding="utf-8")
    summary = {k: v for k, v in pred.__dict__.items() if k != "backtest"}
    summary["backtest"] = {k: v for k, v in pred.backtest.items() if k != "series"}
    summary["intraday"] = intra
    summary["open"] = {
        "predicted_price": pred_open.predicted_price,
        "low_price": pred_open.low_price,
        "high_price": pred_open.high_price,
        "predicted_return": pred_open.predicted_return,
        "backtest": {k: v for k, v in pred_open.backtest.items() if k != "series"},
    }
    (out / f"{args.ticker}_prediction.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nОтчёт: {out / f'{args.ticker}_report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
