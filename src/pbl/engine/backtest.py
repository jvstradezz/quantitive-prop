"""Событийный бэктест и векторный скан (ARCHITECTURE §9, SPEC F-ENG-1…4).

`backtest` — сделки `Trade` (вход проп-MC) с издержками по статьям в R, MAE/MFE и поминутной
эквити; применяет лимит позиций и правила фирмы, меняющие сделки (окно 00:00–02:00 сервера,
лимит сделок в день, закрытие перед выходными). `scan` — то же ядро без лимитов позиций и
сделок в день, результат — массивы R для массового перебора; в решения без `backtest` не идёт.

R: 1R = |цена входа − стоп| с учётом спреда и проскальзывания входа — от этой дистанции бот
считает объём. `r_gross` — по mid-ценам; спред, проскальзывание, комиссия и своп — отдельно.
MAE/MFE — по mid (как `r_gross`), издержки проп-MC вычитает в момент входа (D-020).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import polars as pl

from pbl.engine import kernel as K
from pbl.engine.market import Market
from pbl.instruments.costs import swap_nights
from pbl.schemas import Trade
from pbl.schemas.trade import Sample
from pbl.strategies.base import Signals

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Exits:
    """Выходы позиции; 0 — правило выключено. Уровни в R от фактического риска сделки."""

    be_at_r: float = 0.0  # перенос стопа в безубыток после хода на be_at_r
    be_offset_r: float = 0.0  # стоп безубытка = вход + offset (покрыть издержки)
    trail_start_r: float = 0.0  # трейлинг включается после хода на trail_start_r
    trail_dist_r: float = 0.0  # дистанция трейлинга от лучшей цены
    trail_step_r: float = 0.1  # мин. шаг переноса стопа (троттлинг сообщений, SPEC B5)
    partial_r: float = 0.0  # частичный выход лимитом на partial_r
    partial_frac: float = 0.5
    max_hold_min: float = 0.0  # выход по времени удержания
    close_at: str | None = None  # «HH:MM» сервера: закрыть до этого времени торгового дня

    def __post_init__(self) -> None:
        if not 0.0 < self.partial_frac < 1.0:
            raise ValueError("partial_frac — доля в (0, 1)")
        if min(self.be_at_r, self.trail_start_r, self.trail_dist_r, self.partial_r) < 0:
            raise ValueError("уровни выходов не могут быть отрицательными")
        if self.trail_dist_r > 0 and self.trail_step_r <= 0:
            raise ValueError("trail_step_r > 0 обязателен для трейлинга (B5)")


@dataclass(frozen=True)
class RunMeta:
    strategy_id: str
    variant_id: str = "base"
    sample: Sample = "IS"
    profit_to_usd: float | None = None  # курс валюты прибыли к $ (GER30: EUR, O-20)


@dataclass(frozen=True)
class KernelOut:
    status: npt.NDArray[np.int64]
    ints: I64
    floats: F64


def _params(market: Market, exits: Exits, max_positions: int, max_trades_day: int) -> F64:
    p = np.zeros(K.N_EXIT_PARAMS)
    p[K.E_BE_AT] = exits.be_at_r
    p[K.E_BE_OFF] = exits.be_offset_r
    p[K.E_TR_START] = exits.trail_start_r
    p[K.E_TR_DIST] = exits.trail_dist_r
    p[K.E_TR_STEP] = exits.trail_step_r
    p[K.E_PART_R] = exits.partial_r
    p[K.E_PART_Q] = exits.partial_frac
    p[K.E_MAX_HOLD] = round(exits.max_hold_min * 60)
    if exits.close_at is None:
        p[K.E_CLOSE_MIN] = -1
    else:
        hh, mm = exits.close_at.split(":")
        p[K.E_CLOSE_MIN] = int(hh) * 60 + int(mm)
    p[K.E_MAX_POS] = max_positions
    p[K.E_MAX_DAY] = max_trades_day
    return p


def run_kernel(
    market: Market,
    signals: Signals,
    exits: Exits,
    *,
    max_positions: int,
    max_trades_day: int,
    jit: bool = True,
) -> KernelOut:
    sig = signals.sorted()
    m = len(sig)
    out_i = np.zeros((m, K.N_OUT_I), dtype=np.int64)
    out_f = np.zeros((m, K.N_OUT_F), dtype=np.float64)
    fn = K.kernel if jit else K._run
    fn(
        market.ts,
        market.mid_o,
        market.mid_h,
        market.mid_l,
        market.mid_c,
        market.spread,
        market.slip,
        market.day,
        market.minute,
        market.live,
        market.entry_ok,
        market.force_close,
        sig.bar,
        sig.side,
        sig.order,
        sig.level,
        sig.expire_s,
        sig.sl,
        sig.sl_dist,
        sig.tp,
        sig.tp_r,
        sig.oco,
        _params(market, exits, max_positions, max_trades_day),
        out_i,
        out_f,
    )
    return KernelOut(status=out_i[:, K.I_STATUS], ints=out_i, floats=out_f)


@dataclass(frozen=True)
class CostsR:
    """Комиссия и своп сделок в R и цена пункта в $ (векторно по принятым сделкам)."""

    comm_r: F64
    swap_r: F64
    pv_usd: F64
    stop_points: F64


def _costs_r(
    market: Market,
    out: KernelOut,
    sides: npt.NDArray[np.int8],
    profit_to_usd: float | None,
    swap_mult: float,
    comm_mult: float,
) -> CostsR:
    c = market.costs
    acc = np.flatnonzero(out.status == K.ACCEPTED)
    entry = out.floats[acc, K.F_ENTRY]
    risk = out.floats[acc, K.F_RISK]
    stop_points = risk / c.point
    pv = np.asarray([c.pv_usd(float(p), profit_to_usd) for p in entry])
    comm_usd = np.asarray([c.commission_usd(float(p), profit_to_usd) for p in entry]) * comm_mult
    comm_r = comm_usd / (stop_points * pv)
    e_day = market.day[out.ints[acc, K.I_ENTRY]]
    x_day = market.day[out.ints[acc, K.I_EXIT]]
    nights = np.asarray(
        [swap_nights(int(a), int(b), c.triple_weekday) for a, b in zip(e_day, x_day, strict=True)],
        dtype=np.float64,
    )
    per_night = np.asarray([c.swap_per_night(int(s)) for s in sides[acc]])
    if c.swap_in_points:
        swap_gain_r = per_night * nights / stop_points
    else:
        swap_gain_r = per_night * nights / (stop_points * pv)
    return CostsR(
        comm_r=comm_r, swap_r=-swap_gain_r * swap_mult, pv_usd=pv, stop_points=stop_points
    )


@dataclass(frozen=True)
class ScanResult:
    """Массивы по исполненным сигналам (векторный скан)."""

    r_net: F64
    r_gross: F64
    cost_r: F64
    hold_sec: I64
    n_signals: int
    entry_bar: I64 = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    side: npt.NDArray[np.int8] = field(default_factory=lambda: np.zeros(0, dtype=np.int8))

    @property
    def n_trades(self) -> int:
        return int(self.r_net.shape[0])

    def summary(self) -> dict[str, float]:
        return summarize_r(self.r_net, self.r_gross)


def summarize_r(r_net: F64, r_gross: F64) -> dict[str, float]:
    n = int(r_net.shape[0])
    if n == 0:
        return {"n": 0.0}
    wins, losses = r_net[r_net > 0].sum(), -r_net[r_net < 0].sum()
    gross_pos = float(r_gross.sum())
    return {
        "n": float(n),
        "mean_r": float(r_net.mean()),
        "sum_r": float(r_net.sum()),
        "win_rate": float((r_net > 0).mean()),
        "pf": float(wins / losses) if losses > 0 else float("inf"),
        "cost_share_of_gross": float((r_gross - r_net).sum() / gross_pos)
        if gross_pos > 0
        else float("nan"),
    }


def scan(
    market: Market,
    signals: Signals,
    exits: Exits,
    *,
    profit_to_usd: float | None = None,
    swap_mult: float = 1.0,
    commission_mult: float = 1.0,
) -> ScanResult:
    """Каждый сигнал — отдельная сделка (без лимита позиций и сделок в день; OCO действует)."""
    sig = signals.sorted()
    out = run_kernel(market, sig, exits, max_positions=len(sig) + 1, max_trades_day=len(sig) + 1)
    acc = out.status == K.ACCEPTED
    cr = _costs_r(market, out, sig.side, profit_to_usd, swap_mult, commission_mult)
    f = out.floats[acc]
    risk = f[:, K.F_RISK]
    r_gross = f[:, K.F_GROSS] / risk
    cost = (f[:, K.F_SPREAD] + f[:, K.F_SLIP]) / risk + cr.comm_r + cr.swap_r
    hold = (
        market.ts[out.ints[acc, K.I_EXIT]]
        + 60 * out.ints[acc, K.I_XCLOSE]
        - market.ts[out.ints[acc, K.I_ENTRY]]
    )
    return ScanResult(
        r_net=r_gross - cost,
        r_gross=r_gross,
        cost_r=cost,
        hold_sec=hold.astype(np.int64),
        n_signals=len(sig),
        entry_bar=out.ints[acc, K.I_ENTRY].astype(np.int64),
        side=sig.side[acc],
    )


@dataclass(frozen=True)
class BacktestResult:
    trades: list[Trade]
    equity: pl.DataFrame  # trade_id, ts (UTC), r_worst, r_close — поминутно, чистые R
    signal_status: dict[str, int]
    stats: dict[str, float] = field(default_factory=dict)


def _utc(sec: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(sec, dt.UTC)


def backtest(
    market: Market,
    signals: Signals,
    exits: Exits,
    meta: RunMeta,
    *,
    max_positions: int = 1,
    swap_mult: float = 1.0,
    commission_mult: float = 1.0,
    with_equity: bool = True,
    jit: bool = True,
) -> BacktestResult:
    sig = signals.sorted()
    max_day = market.firm.max_trades_day or len(sig) + 1
    out = run_kernel(
        market, sig, exits, max_positions=max_positions, max_trades_day=max_day, jit=jit
    )
    cr = _costs_r(market, out, sig.side, meta.profit_to_usd, swap_mult, commission_mult)
    acc = np.flatnonzero(out.status == K.ACCEPTED)
    trades: list[Trade] = []
    eq_parts: list[pl.DataFrame] = []
    for n_t, k in enumerate(acc):
        fi, ii = out.floats[k], out.ints[k]
        d = int(sig.side[k])
        risk = fi[K.F_RISK]
        r_gross = fi[K.F_GROSS] / risk
        c_spread = fi[K.F_SPREAD] / risk
        c_slip = fi[K.F_SLIP] / risk
        c_comm = float(cr.comm_r[n_t])
        c_swap = float(cr.swap_r[n_t])
        r_net = r_gross - (c_spread + c_comm + c_swap + c_slip)
        e_bar, x_bar = int(ii[K.I_ENTRY]), int(ii[K.I_EXIT])
        close_s = int(market.ts[x_bar]) + 60 * int(ii[K.I_XCLOSE])
        trade_id = f"{meta.strategy_id}-{meta.variant_id}-{n_t:05d}"
        trades.append(
            Trade(
                trade_id=trade_id,
                strategy_id=meta.strategy_id,
                variant_id=meta.variant_id,
                symbol=market.symbol,
                side="long" if d > 0 else "short",
                open_utc=_utc(int(market.ts[e_bar])),
                close_utc=_utc(close_s),
                entry=float(fi[K.F_ENTRY]),
                exit=float(fi[K.F_EXIT]),
                sl=float(fi[K.F_SL0]),
                tp=None if np.isnan(fi[K.F_TP0]) else float(fi[K.F_TP0]),
                stop_dist=float(cr.stop_points[n_t]),
                point_value=float(cr.pv_usd[n_t]),
                r_gross=float(r_gross),
                cost_spread_r=float(c_spread),
                cost_comm_r=c_comm,
                cost_swap_r=c_swap,
                cost_slip_r=float(c_slip),
                r_net=float(r_net),
                mae_r=float(fi[K.F_MAE]),
                mfe_r=float(fi[K.F_MFE]),
                hold_sec=close_s - int(market.ts[e_bar]),
                exit_reason=K.EXIT_REASONS[int(ii[K.I_REASON])],  # type: ignore[arg-type]
                mods=int(ii[K.I_MODS]),
                sample=meta.sample,
            )
        )
        if with_equity:
            eq_parts.append(
                _trade_equity(market, trades[-1], fi, ii, d, c_comm, c_swap, exits.partial_frac)
            )
    status = {name: int((out.status == code).sum()) for code, name in enumerate(K.STATUSES) if name}
    equity = (
        pl.concat(eq_parts)
        if eq_parts
        else pl.DataFrame(
            schema={
                "trade_id": pl.String,
                "ts": pl.Datetime("ms", "UTC"),
                "r_worst": pl.Float64,
                "r_close": pl.Float64,
            }
        )
    )
    return BacktestResult(
        trades=trades,
        equity=equity,
        signal_status=status,
        stats=trade_stats(trades, market.day[out.ints[acc, K.I_ENTRY]], market.firm.min_hold_sec),
    )


def _trade_equity(
    market: Market,
    t: Trade,
    fi: F64,
    ii: I64,
    d: int,
    comm_r: float,
    swap_r_total: float,
    partial_frac: float,
) -> pl.DataFrame:
    """Поминутная плавающая прибыль сделки в чистых R: худшая цена бара (лонг — low по bid,
    шорт — high по ask) и закрытие бара. Комиссия — в момент входа, своп — по ночам.
    На баре выхода `r_close = r_net`."""
    e, x = int(ii[K.I_ENTRY]), int(ii[K.I_EXIT])
    sl = slice(e, x + 1)
    risk, entry = fi[K.F_RISK], fi[K.F_ENTRY]
    half = market.spread[sl] / 2.0
    adv = (market.mid_l[sl] if d > 0 else market.mid_h[sl]) - d * half
    close = market.mid_c[sl] - d * half
    q = np.ones(x - e + 1)
    realized = np.zeros(x - e + 1)
    pb = int(ii[K.I_PART])
    if pb >= 0:
        after = np.arange(e, x + 1) > pb
        q[after] = 1.0 - partial_frac
        realized[after] = partial_frac * d * (fi[K.F_PART_FILL] - entry) / risk
    days = market.day[sl]
    total_n = swap_nights(int(market.day[e]), int(market.day[x]), market.costs.triple_weekday)
    per_n = swap_r_total / total_n if total_n else 0.0
    swap_acc = (
        np.asarray(
            [swap_nights(int(market.day[e]), int(dd), market.costs.triple_weekday) for dd in days]
        )
        * per_n
    )
    r_worst = realized + q * d * (adv - entry) / risk - comm_r - swap_acc
    r_close = realized + q * d * (close - entry) / risk - comm_r - swap_acc
    # вне сессии у фирмы нет котировок: эквити стоит на последнем значении сессии
    if not market.live[sl].all():
        keep = np.maximum.accumulate(np.where(market.live[sl], np.arange(x - e + 1), 0))
        r_worst, r_close = r_worst[keep], r_close[keep]
    r_close[-1] = t.r_net
    held_through = t.exit_reason in ("tp", "time", "eod", "friday_close")
    r_worst[-1] = min(t.r_net, r_worst[-1]) if held_through else t.r_net
    r_worst = np.minimum(r_worst, r_close)
    return pl.DataFrame(
        {
            "trade_id": [t.trade_id] * (x - e + 1),
            "ts": (market.ts[sl] * 1000).astype("datetime64[ms]"),
            "r_worst": r_worst,
            "r_close": r_close,
        }
    ).with_columns(pl.col("ts").dt.replace_time_zone("UTC"))


def trade_stats(trades: list[Trade], server_day: I64, min_hold_sec: int | None) -> dict[str, float]:
    """Сводка R плюс проверки правил фирмы: Quick Strike и сообщения в день (SPEC B2, B5)."""
    r_net = np.asarray([t.r_net for t in trades])
    r_gross = np.asarray([t.r_gross for t in trades])
    out = summarize_r(r_net, r_gross)
    if trades and min_hold_sec:
        hold = np.asarray([t.hold_sec for t in trades])
        profit = np.clip(r_net, 0, None)
        total = profit.sum()
        # Quick Strike FN: доля прибыли сделок короче порога (на M1 выход в баре входа = 0 с)
        out["quick_strike_profit_share"] = (
            float(profit[hold < min_hold_sec].sum() / total) if total > 0 else 0.0
        )
        out["quick_strike_trades"] = float((hold < min_hold_sec).sum())
    if trades:
        # открытие + закрытие + модификации; по дню открытия на сервере
        msgs = np.asarray([t.mods + 2 for t in trades], dtype=np.int64)
        _, inv = np.unique(server_day, return_inverse=True)
        out["messages_max_day"] = float(np.bincount(inv, weights=msgs).max())
        out["trades_max_day"] = float(np.bincount(inv).max())
    return out
