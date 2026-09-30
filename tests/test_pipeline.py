import json

import numpy as np
import pandas as pd

from nxt_predictor import data
from nxt_predictor.__main__ import main
from nxt_predictor.features import build_features, context_features, daily_news_features, intraday_summary, sentiment
from nxt_predictor.model import predict_next, tune


def test_finance_sentiment_direction():
    assert sentiment("Nextracker beats earnings estimates, raises guidance") > 0.3
    assert sentiment("Nextracker shares fall after downgrade") < -0.3


def test_news_after_close_maps_to_same_day_and_weekend_to_friday():
    days = pd.bdate_range("2026-09-21", "2026-09-25")  # пн–пт
    news = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-09-22 21:30", "2026-09-26 15:00"], utc=True),  # вт после закрытия, сб
            "title": ["a", "b"],
            "summary": ["", ""],
            "sentiment": [0.5, -0.5],
        }
    )
    f = daily_news_features(news, days)
    assert f.loc["2026-09-22", "news_sent"] == 0.5
    assert f.loc["2026-09-25", "news_sent"] == -0.5
    assert f["news_count"].sum() == 2


def test_features_have_no_lookahead():
    daily, _, news = data.synthetic_data(days=300)
    f1 = build_features(daily, news)
    cut = daily.iloc[:-10]
    known = news[news["time"] < pd.Timestamp(cut.index[-1] + pd.Timedelta(days=1), tz="UTC")]
    f2 = build_features(cut, known)
    common = f2.drop(columns="target").dropna().index
    pd.testing.assert_frame_equal(
        f1.loc[common].drop(columns="target"), f2.loc[common].drop(columns="target")
    )
    assert np.isnan(f1["target"].iloc[-1])


def test_prediction_is_sane():
    daily, _, news = data.synthetic_data(days=400)
    from nxt_predictor.features import score_news

    feats = build_features(daily, score_news(news), data.synthetic_context(daily))
    p = predict_next(feats, float(daily["Close"].iloc[-1]), backtest_days=60, fast=True)
    assert p.low_price <= p.predicted_price <= p.high_price
    assert 0 <= p.prob_up <= 1
    assert abs(p.predicted_return) < 0.2
    assert abs(sum(p.weights.values()) - 1) < 1e-9
    assert set(p.weights) == {"ridge", "gbm", "hgb", "rf", "et"}
    assert 0 <= p.backtest["classifier_accuracy"] <= 1


def test_tuning_picks_from_grid():
    daily, _, news = data.synthetic_data(days=300)
    f = build_features(daily, news).dropna()
    params, weights, mae = tune(f.drop(columns="target"), f["target"], n_splits=2)
    assert params["gbm"]["max_depth"] in (2, 3)
    assert all(v > 0 for v in mae.values())
    assert abs(sum(weights.values()) - 1) < 1e-9


def test_context_features_no_lookahead():
    daily, _, _ = data.synthetic_data(days=300)
    ctx = data.synthetic_context(daily)
    f1 = context_features(daily, ctx)
    f2 = context_features(daily.iloc[:-5], ctx.iloc[:-5])
    common = f2.dropna().index
    pd.testing.assert_frame_equal(f1.loc[common], f2.loc[common])
    assert "beta_tan_60" in f1 and "rel_spy_5" in f1


def test_intraday_summary():
    _, intraday, _ = data.synthetic_data(days=50)
    s = intraday_summary(intraday)
    assert s["minutes"] == 390
    assert 0 <= s["buy_pressure"] <= 1


def test_cli_demo_writes_reports(tmp_path):
    assert main(["--demo", "--fast", "--backtest-days", "60", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "NXT_report.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    summary = json.loads((tmp_path / "NXT_prediction.json").read_text(encoding="utf-8"))
    assert summary["predicted_price"] > 0


def test_csv_inputs(tmp_path):
    daily, _, news = data.synthetic_data(days=300)
    daily.to_csv(tmp_path / "p.csv")
    news.to_csv(tmp_path / "n.csv", index=False)
    loaded = data.load_prices_csv(str(tmp_path / "p.csv"))
    assert len(loaded) == 300
    assert len(data.load_news_csv(str(tmp_path / "n.csv"))) == len(news)


def test_nasdaq_csv_format(tmp_path):
    p = tmp_path / "nasdaq.csv"
    p.write_text(
        "Date,Close/Last,Volume,Open,High,Low\n"
        "09/29/2026,$77.71,2345678,$76.50,$78.10,$76.20\n"
        "09/26/2026,$76.40,1987654,$75.90,$77.00,$75.10\n"
    )
    df = data.load_prices_csv(str(p))
    assert list(df.columns) == data.PRICE_COLUMNS
    assert df.index[0] == pd.Timestamp("2026-09-26")
    assert df["Close"].iloc[-1] == 77.71


def test_open_target_is_gap_to_next_open():
    daily, _, _ = data.synthetic_data(days=100)
    f = build_features(daily, target="open")
    expected = np.log(daily["Open"].iloc[1] / daily["Close"].iloc[0])
    assert abs(f["target"].iloc[0] - expected) < 1e-12
    assert np.isnan(f["target"].iloc[-1])
