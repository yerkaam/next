"""Отчёт: текст в консоль и самодостаточный HTML с графиком."""
from __future__ import annotations

import html

import pandas as pd

from .model import Prediction


def text_report(ticker: str, p: Prediction, intraday: dict, news: pd.DataFrame) -> str:
    bt = p.backtest
    arrow = "▲" if p.predicted_return > 0 else "▼"
    lines = [
        f"=== {ticker}: прогноз на следующую торговую сессию ===",
        f"Последнее закрытие ({p.last_date}): ${p.last_close:.2f}",
    ]
    if intraday:
        lines.append(
            f"Последняя сделка ({intraday['last_trade_time']}): ${intraday['last_price']:.2f}, "
            f"VWAP ${intraday['vwap']:.2f}, давление покупок {intraday['buy_pressure']:.0%}"
        )
    lines += [
        f"Прогноз закрытия: ${p.predicted_price:.2f}  {arrow} {p.predicted_return * 100:+.2f}%",
        f"Диапазон 80%:     ${p.low_price:.2f} – ${p.high_price:.2f}",
        f"Вероятность роста: {p.prob_up:.0%}",
        "",
        f"Проверка на истории ({bt['days']} дней, walk-forward):",
        f"  средняя ошибка модели {bt['mae_model'] * 100:.2f}% vs «цена не изменится» {bt['mae_naive'] * 100:.2f}%",
        f"  угадано направление: {bt['direction_accuracy']:.0%} (доля дней роста {bt['share_up_days']:.0%})",
        "",
        "Самые важные признаки: " + ", ".join(k for k, _ in p.top_features[:5]),
    ]
    if not news.empty:
        lines += ["", "Последние новости:"]
        for _, r in news.tail(8).iloc[::-1].iterrows():
            lines.append(f"  [{r['sentiment']:+.2f}] {r['time']:%Y-%m-%d} {r['title'][:110]}")
    lines += ["", "Это статистическая оценка, а не инвестиционная рекомендация."]
    return "\n".join(lines)


def _svg_chart(closes: pd.Series, p: Prediction, w: int = 760, h: int = 300) -> str:
    vals = list(closes.values) + [p.low_price, p.high_price]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08 or 1
    lo, hi = lo - pad, hi + pad
    n = len(closes) + 1

    def x(i):
        return 40 + i * (w - 60) / (n - 1)

    def y(v):
        return h - 25 - (v - lo) * (h - 45) / (hi - lo)

    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(closes.values))
    xl, xp = x(n - 2), x(n - 1)
    color = "var(--up)" if p.predicted_return > 0 else "var(--down)"
    ticks = "".join(
        f'<text x="4" y="{y(v) + 4:.1f}" class="t">{v:.0f}</text>'
        f'<line x1="36" x2="{w - 16}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="g"/>'
        for v in [lo + (hi - lo) * k / 4 for k in range(5)]
    )
    return f"""<svg viewBox="0 0 {w} {h}" role="img" aria-label="График цены и прогноз">
{ticks}
<polyline points="{pts}" fill="none" stroke="var(--line)" stroke-width="2"/>
<polygon points="{xl:.1f},{y(p.last_close):.1f} {xp:.1f},{y(p.high_price):.1f} {xp:.1f},{y(p.low_price):.1f}" fill="{color}" opacity=".18"/>
<line x1="{xl:.1f}" y1="{y(p.last_close):.1f}" x2="{xp:.1f}" y2="{y(p.predicted_price):.1f}" stroke="{color}" stroke-width="2" stroke-dasharray="5 4"/>
<circle cx="{xp:.1f}" cy="{y(p.predicted_price):.1f}" r="5" fill="{color}"/>
<text x="{xp - 6:.1f}" y="{y(p.predicted_price) - 10:.1f}" text-anchor="end" class="t b">${p.predicted_price:.2f}</text>
<text x="40" y="{h - 6}" class="t">{closes.index[0]:%Y-%m-%d}</text>
<text x="{xl:.1f}" y="{h - 6}" text-anchor="end" class="t">{closes.index[-1]:%Y-%m-%d}</text>
</svg>"""


def html_report(ticker: str, p: Prediction, daily: pd.DataFrame, intraday: dict, news: pd.DataFrame) -> str:
    bt = p.backtest
    sign = "up" if p.predicted_return > 0 else "down"
    news_rows = "".join(
        f"<tr><td>{r['time']:%Y-%m-%d %H:%M}</td><td class='{'up' if r['sentiment'] > 0.05 else 'down' if r['sentiment'] < -0.05 else ''}'>"
        f"{r['sentiment']:+.2f}</td><td>{html.escape(str(r['title']))}</td></tr>"
        for _, r in news.tail(25).iloc[::-1].iterrows()
    ) or "<tr><td colspan=3>Новостей нет</td></tr>"
    trade = (
        f"<div class='card'><div class='k'>Последняя сделка</div><div class='v'>${intraday['last_price']:.2f}</div>"
        f"<div class='s'>VWAP ${intraday['vwap']:.2f} · покупки {intraday['buy_pressure']:.0%} · {intraday['minutes']} мин</div></div>"
        if intraday else ""
    )
    feats = "".join(f"<li>{html.escape(k)} <span class='s'>{v:.1%}</span></li>" for k, v in p.top_features)
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{ticker} прогноз цены</title>
<style>
:root{{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--border:#e4e3df;--line:#3a5a8c;--up:#1d8a5a;--down:#c2412d}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161616;--fg:#ecebe7;--muted:#9a9993;--card:#1f1f1f;--border:#333;--line:#8fb0e6;--up:#4cc28c;--down:#f07a64}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}}
main{{max-width:820px;margin:auto}} h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 8px}}
.s,.t{{color:var(--muted);font-size:13px}} .t{{fill:var(--muted);font-size:11px}} .b{{font-weight:600;fill:var(--fg)}}
.g{{stroke:var(--border)}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:16px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px 14px}}
.k{{color:var(--muted);font-size:13px}} .v{{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}}
.up{{color:var(--up)}} .down{{color:var(--down)}} svg{{width:100%;height:auto;margin-top:12px}}
table{{width:100%;border-collapse:collapse;font-size:14px}} td{{padding:6px 4px;border-top:1px solid var(--border);vertical-align:top}}
td:first-child{{white-space:nowrap;color:var(--muted)}} ul{{padding-left:18px}}
</style></head><body><main>
<h1>{ticker}: следующая цена</h1>
<div class="s">Данные по {p.last_date}. Статистическая оценка, не инвестиционная рекомендация.</div>
<div class="grid">
<div class="card"><div class="k">Прогноз закрытия</div><div class="v {sign}">${p.predicted_price:.2f}</div><div class="s {sign}">{p.predicted_return * 100:+.2f}% к ${p.last_close:.2f}</div></div>
<div class="card"><div class="k">Диапазон 80%</div><div class="v">${p.low_price:.0f}–{p.high_price:.0f}</div><div class="s">${p.low_price:.2f} – ${p.high_price:.2f}</div></div>
<div class="card"><div class="k">Вероятность роста</div><div class="v">{p.prob_up:.0%}</div><div class="s">по ошибкам модели</div></div>
{trade}
</div>
{_svg_chart(daily["Close"].tail(90), p)}
<h2>Насколько модели можно верить</h2>
<p>Walk-forward проверка на последних {bt['days']} днях: средняя ошибка <b>{bt['mae_model'] * 100:.2f}%</b>
против <b>{bt['mae_naive'] * 100:.2f}%</b> у наивного прогноза «цена не изменится».
Направление угадано в <b>{bt['direction_accuracy']:.0%}</b> случаев (доля дней роста — {bt['share_up_days']:.0%}).</p>
<h2>Что влияет сильнее всего</h2><ul>{feats}</ul>
<h2>Новости и тональность</h2><table>{news_rows}</table>
</main></body></html>"""
