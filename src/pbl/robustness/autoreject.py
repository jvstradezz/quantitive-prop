"""Автоотказ по списку сделок (VALIDATION §6, SPEC §4 B2–B4).

Каждый детектор отвечает `True` (техника найдена → отказ без MC), `False` (чисто) или `None`
(нет данных: например, мартингейл не виден без риска сделки, а движок считает все сделки при 1R).
Пороги — `robustness.thresholds` (P-10, P-36).
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from pbl.robustness.thresholds import (
    AVERAGING_ADVERSE_R,
    GRID_MIN_POSITIONS,
    GRID_STEP_RATIO,
    MARTINGALE_JUMP,
    MARTINGALE_SHARE,
    MARTINGALE_TOL,
    MAX_TRADES_DAY,
    MIN_HOLD_SEC,
    SCALP_MEDIAN_HOLD_SEC,
    SCALP_TAKE_SPREADS,
    SHORT_PROFIT_SHARE,
)
from pbl.robustness.trades import trade_day
from pbl.schemas import ServerTime, Trade

FORBIDDEN_FAMILIES = frozenset(
    {
        "martingale",
        "grid",
        "averaging",
        "hft",
        "tick_scalping",
        "latency_arbitrage",
        "hedge_accounts",
        "copy_trading",
        "challenge_passing_ea",
    }
)
"""Семейства, запрещённые фирмами или §9 сами по себе (SPEC §2, PROP_RULES FN 2.6)."""

MAX_IDS = 5


@dataclass(frozen=True)
class Finding:
    code: str
    name: str
    triggered: bool | None
    detail: str
    trade_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AutoReject:
    findings: tuple[Finding, ...]
    n_trades: int

    @property
    def rejected(self) -> bool:
        return any(f.triggered for f in self.findings)

    @property
    def reasons(self) -> list[str]:
        return [f"{f.name}: {f.detail}" for f in self.findings if f.triggered]

    def get(self, code: str) -> Finding:
        return next(f for f in self.findings if f.code == code)


def _ids(trades: Iterable[Trade]) -> tuple[str, ...]:
    return tuple(t.trade_id for t in trades)[:MAX_IDS]


def no_stop(trades: Sequence[Trade]) -> Finding:
    bad = []
    for t in trades:
        sign = 1.0 if t.side == "long" else -1.0
        ok = math.isfinite(t.sl) and math.isfinite(t.stop_dist) and sign * (t.entry - t.sl) > 0
        if not ok:
            bad.append(t)
    return Finding(
        "no_stop",
        "сделка без стопа",
        bool(bad),
        f"{len(bad)} сделок без стопа или со стопом не с той стороны" if bad else "у всех стоп",
        _ids(bad),
    )


def hft(trades: Sequence[Trade], server: ServerTime) -> Finding:
    """HFT: > 200 сделок в день (B4) или медианное удержание < 30 с."""
    per_day = Counter(trade_day(t.open_utc, server)[0] for t in trades)
    busy = max(per_day.values(), default=0)
    med = float(np.median([t.hold_sec for t in trades])) if trades else math.nan
    hit = busy > MAX_TRADES_DAY or (not math.isnan(med) and med < MIN_HOLD_SEC)
    detail = f"максимум сделок в день: {busy}; медиана удержания {med:.0f} с"
    return Finding("hft", "HFT", hit, detail)


def quick_strike(trades: Sequence[Trade]) -> Finding:
    """Quick Strike FN 2.6: доля прибыли сделок короче 30 с ≥ 20% (порог предупреждения фирмы).

    Убыточная сделка короче 30 с (стоп в первые секунды) правило не нарушает (P-36).
    """
    name = "Quick Strike (B2)"
    profit = sum(t.r_net for t in trades if t.r_net > 0)
    short = [t for t in trades if t.hold_sec < MIN_HOLD_SEC]
    if profit <= 0:
        return Finding("quick_strike", name, None, "нет прибыли", _ids(short))
    share = sum(t.r_net for t in short if t.r_net > 0) / profit
    return Finding(
        "quick_strike",
        name,
        share >= SHORT_PROFIT_SHARE,
        f"{100 * share:.0f}% прибыли — сделки короче {MIN_HOLD_SEC} с ({len(short)} сделок)",
        _ids(t for t in short if t.r_net > 0),
    )


def tick_scalping(trades: Sequence[Trade]) -> Finding:
    """Медианное удержание < 2 мин или медианный тейк < 3 × спреда сделки (P-10)."""
    if not trades:
        return Finding("scalping", "тик-скальпинг", None, "нет сделок")
    med_hold = float(np.median([t.hold_sec for t in trades]))
    ratios = [
        abs(t.tp - t.entry) / (t.cost_spread_r * t.stop_dist)
        for t in trades
        if t.tp is not None and t.cost_spread_r > 0
    ]
    med_take = float(np.median(ratios)) if ratios else math.nan
    hit = med_hold < SCALP_MEDIAN_HOLD_SEC or (
        not math.isnan(med_take) and med_take < SCALP_TAKE_SPREADS
    )
    take = "—" if math.isnan(med_take) else f"{med_take:.1f}"
    detail = f"медиана удержания {med_hold / 60:.1f} мин; медианный тейк {take} спредов"
    return Finding("scalping", "тик-скальпинг", hit, detail)


def martingale(trades: Sequence[Trade], risk: Sequence[float] | None) -> Finding:
    """Риск сделки растёт после убытка: один рост ≥ ×1.5 или рост > 5% после ≥ 50% убытков."""
    name = "мартингейл"
    if risk is None:
        return Finding("martingale", name, None, "нет риска сделок (у движка все сделки по 1R)")
    if len(risk) != len(trades):
        raise ValueError("risk и trades разной длины")
    order = sorted(range(len(trades)), key=lambda i: trades[i].open_utc)
    by_close = sorted(range(len(trades)), key=lambda i: trades[i].close_utc)
    closes = [trades[i].close_utc for i in by_close]
    after_loss = jumps = ups = 0
    hits: list[Trade] = []
    for j in order:
        k = bisect_right(closes, trades[j].open_utc) - 1
        if k < 0:
            continue
        prev = by_close[k]
        if trades[prev].r_net >= 0 or risk[prev] <= 0:
            continue
        after_loss += 1
        ratio = risk[j] / risk[prev]
        if ratio > 1 + MARTINGALE_TOL:
            ups += 1
            hits.append(trades[j])
        if ratio >= MARTINGALE_JUMP:
            jumps += 1
    hit = jumps > 0 or (ups >= 3 and ups >= MARTINGALE_SHARE * after_loss)
    detail = (
        f"после {after_loss} убытков риск рос {ups} раз, из них ≥ ×{MARTINGALE_JUMP:g}: {jumps}"
    )
    return Finding("martingale", name, hit, detail, _ids(hits))


def _concurrent(trades: Sequence[Trade]) -> Iterable[tuple[Trade, list[Trade]]]:
    """Для каждой сделки — открытые в момент её входа позиции того же символа и стороны."""
    groups: dict[tuple[str, str], list[Trade]] = {}
    for t in sorted(trades, key=lambda x: x.open_utc):
        groups.setdefault((t.symbol, t.side), []).append(t)
    for group in groups.values():
        live: list[Trade] = []
        for t in group:
            live = [o for o in live if o.close_utc > t.open_utc]
            yield t, list(live)
            live.append(t)


def grid(trades: Sequence[Trade]) -> Finding:
    """≥ 3 открытых позиции одной стороны по символу с ≈ равным шагом цены входа против позиции.

    Шаг по ходу позиции (каждый вход лучше прошлого для этой стороны) — пирамидинг в прибыль,
    он разрешён (инструкции §1, P-36).
    """
    hits: list[Trade] = []
    for t, live in _concurrent(trades):
        if len(live) + 1 < GRID_MIN_POSITIONS:
            continue
        sign = 1.0 if t.side == "long" else -1.0
        d = sign * np.diff(np.array([o.entry for o in live] + [t.entry]))
        if not np.all(d < 0):
            continue
        a = np.abs(d)
        if a.max() / a.min() <= GRID_STEP_RATIO:
            hits.append(t)
    return Finding(
        "grid",
        "сетка",
        bool(hits),
        f"{len(hits)} входов в сетку из ≥ {GRID_MIN_POSITIONS} позиций" if hits else "нет",
        _ids(hits),
    )


def averaging(trades: Sequence[Trade]) -> Finding:
    """Новый вход в ту же сторону, пока открытая позиция в минусе больше 0.1R."""
    hits: list[Trade] = []
    for t, live in _concurrent(trades):
        sign = 1.0 if t.side == "long" else -1.0
        if any(sign * (t.entry - o.entry) / o.stop_dist < -AVERAGING_ADVERSE_R for o in live):
            hits.append(t)
    return Finding(
        "averaging",
        "усреднение против позиции",
        bool(hits),
        f"{len(hits)} доливок в убыточную позицию" if hits else "нет",
        _ids(hits),
    )


def family(name: str | None) -> Finding:
    if name is None:
        return Finding("family", "запрещённое семейство", None, "семейство не указано")
    hit = name in FORBIDDEN_FAMILIES
    return Finding(
        "family", "запрещённое семейство", hit, f"«{name}»" + (" запрещено" if hit else "")
    )


@dataclass(frozen=True)
class AutoRejectInput:
    trades: Sequence[Trade]
    server: ServerTime
    risk: Sequence[float] | None = None
    family: str | None = None
    extra: tuple[Finding, ...] = field(default_factory=tuple)


def autoreject(inp: AutoRejectInput) -> AutoReject:
    t = inp.trades
    findings = (
        no_stop(t),
        martingale(t, inp.risk),
        grid(t),
        averaging(t),
        hft(t, inp.server),
        quick_strike(t),
        tick_scalping(t),
        family(inp.family),
        *inp.extra,
    )
    return AutoReject(findings=findings, n_trades=len(t))
