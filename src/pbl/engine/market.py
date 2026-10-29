"""Рынок для движка: M1-бары bid/ask → массивы numpy с эффективным спредом, временем сервера
и флагами правил фирмы (ARCHITECTURE §6, §9).

Цены движка — mid; bid/ask восстанавливаются как `mid ∓ spread/2`, где `spread` — модель
издержек (`CostModel.spread`). Так спред можно заменить худшим (стресс), не трогая данные.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from pbl.instruments.costs import WEEKDAYS, CostModel, InstrumentCosts, instrument_costs
from pbl.instruments.servertime import server_clock
from pbl.schemas import InstrumentSpec, ServerTime
from pbl.schemas.rules import Commission, TradeRules

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
B8 = npt.NDArray[np.bool_]

BAR_COLS = ("ts", "bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c")
FRIDAY_CLOSE = "23:45"  # закрытие перед выходными, если фирма запрещает их держать (B10)


@dataclass(frozen=True)
class FirmTradeProfile:
    """Правила фирмы, меняющие сделки (D-012): окно без торговли, лимит сделок, выходные."""

    settlement_window: tuple[str, str] | None = None  # по серверу, «HH:MM»
    max_trades_day: int | None = None
    weekend_close: bool = False
    friday_close: str = FRIDAY_CLOSE
    min_hold_sec: int | None = None  # Quick Strike: только статистика, порог доли прибыли

    @classmethod
    def from_rules(cls, rules: TradeRules, *, funded: bool) -> FirmTradeProfile:
        hold = rules.weekend_holding_funded if funded else rules.weekend_holding_challenge
        return cls(
            settlement_window=rules.settlement_window_server,
            max_trades_day=rules.max_trades_day,
            weekend_close=hold is False,
            min_hold_sec=rules.quick_strike.hold_sec if rules.quick_strike else None,
        )


@dataclass(frozen=True)
class Market:
    symbol: str
    ts: I64  # начало бара, секунды UTC
    mid_o: F64
    mid_h: F64
    mid_l: F64
    mid_c: F64
    spread: F64  # эффективный спред бара, единицы цены
    slip: F64  # проскальзывание market/stop-исполнения в баре, единицы цены
    day: I64  # ordinal торгового дня сервера
    minute: I64  # минута по часам сервера (0–1439)
    weekday: I64  # 0 — понедельник
    live: B8  # бар внутри торговой сессии инструмента: вне её у фирмы нет котировок
    entry_ok: B8  # можно открыть позицию: сессия инструмента и правила фирмы
    force_close: B8  # к открытию бара позиция должна быть закрыта (выходные)
    costs: InstrumentCosts
    firm: FirmTradeProfile

    def __len__(self) -> int:
        return int(self.ts.shape[0])


def _hhmm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def session_mask(spec: InstrumentSpec, weekday: I64, minute: I64) -> B8:
    """Бар внутри торговой сессии инструмента (часы сервера; «00:00» в конце — полночь)."""
    out = np.zeros(weekday.shape, dtype=np.bool_)
    for s in spec.sessions:
        d = WEEKDAYS.index(s.day)
        a, b = _hhmm(s.start), _hhmm(s.end) or 1440
        if b > a:
            out |= (weekday == d) & (minute >= a) & (minute < b)
        else:  # через полночь
            out |= ((weekday == d) & (minute >= a)) | ((weekday == (d + 1) % 7) & (minute < b))
    return out


def prepare_market(
    bars: pl.DataFrame,
    spec: InstrumentSpec,
    server: ServerTime,
    cost: CostModel | None = None,
    firm: FirmTradeProfile | None = None,
    program_commission: Commission | None = None,
) -> Market:
    """Бары `pbl.data.bars.build_bars(clock="utc")` → `Market`."""
    missing = [c for c in BAR_COLS if c not in bars.columns]
    if missing:
        raise ValueError(f"в барах нет колонок: {missing}")
    if bars.is_empty():
        raise ValueError("нет баров")
    cost = cost or CostModel()
    firm = firm or FirmTradeProfile()
    b = bars.sort("ts")
    ts = b["ts"].dt.epoch("s").to_numpy().astype(np.int64)
    if np.any(np.diff(ts) <= 0):
        raise ValueError("метки баров должны строго возрастать")

    def col(name: str) -> F64:
        return b[name].to_numpy().astype(np.float64)

    mid = {x: (col(f"bid_{x}") + col(f"ask_{x}")) / 2.0 for x in "ohlc"}
    if "spread_mean" in b.columns:
        data_spread = col("spread_mean")
    else:
        data_spread = ((col("ask_o") - col("bid_o")) + (col("ask_c") - col("bid_c"))) / 2.0

    day, minute_from_reset, weekday = server_clock(ts, server)
    reset = _hhmm(server.day_reset)
    minute = (minute_from_reset + reset) % 1440
    hour = minute // 60
    costs = instrument_costs(spec, program_commission)
    typical, worst = costs.typical[hour], costs.worst[hour]
    has_spec = bool(np.isfinite(costs.typical).all())
    if cost.spread in ("typical", "worst") and not has_spec:
        raise ValueError(f"{spec.symbol}: в спеке нет spread_by_hour, режим {cost.spread!r}")
    if cost.spread == "data":
        spread = data_spread
    elif cost.spread == "typical":
        spread = typical
    elif cost.spread == "worst":
        spread = worst
    else:
        spread = np.fmax(data_spread, typical) if has_spec else data_spread
    if cost.slip_points is not None:
        slip = np.full(ts.shape, cost.slip_points * spec.point)
    else:
        slip = cost.slip_spread_mult * (typical if has_spec else data_spread)

    live = session_mask(spec, weekday, minute)
    entry_ok = live.copy()
    force_close = np.zeros(ts.shape, dtype=np.bool_)
    if firm.settlement_window is not None:
        a, z = (_hhmm(x) for x in firm.settlement_window)
        window = (minute >= a) & (minute < z) if z > a else (minute >= a) | (minute < z)
        entry_ok &= ~window
    if firm.weekend_close:
        force_close = ((weekday == 4) & (minute >= _hhmm(firm.friday_close))) | (weekday >= 5)
        entry_ok &= ~force_close
    return Market(
        symbol=spec.symbol,
        ts=ts,
        mid_o=mid["o"],
        mid_h=np.maximum(mid["h"], np.maximum(mid["o"], mid["c"])),
        mid_l=np.minimum(mid["l"], np.minimum(mid["o"], mid["c"])),
        mid_c=mid["c"],
        spread=np.ascontiguousarray(spread, dtype=np.float64),
        slip=np.ascontiguousarray(slip, dtype=np.float64),
        day=day,
        minute=minute,
        weekday=weekday,
        live=live,
        entry_ok=entry_ok,
        force_close=force_close,
        costs=costs,
        firm=firm,
    )


def strategy_frame(bars: pl.DataFrame, server: ServerTime) -> pl.DataFrame:
    """Кадр баров для стратегии: исходные bid/ask плюс время сервера (`srv_day`, `srv_min`,
    `srv_wd`)."""
    b = bars.sort("ts")
    ts = b["ts"].dt.epoch("s").to_numpy().astype(np.int64)
    day, minute_from_reset, weekday = server_clock(ts, server)
    minute = (minute_from_reset + _hhmm(server.day_reset)) % 1440
    return b.select(BAR_COLS).with_columns(
        pl.Series("srv_day", day), pl.Series("srv_min", minute), pl.Series("srv_wd", weekday)
    )
