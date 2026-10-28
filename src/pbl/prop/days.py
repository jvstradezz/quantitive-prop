"""Сделки → дневные записи под профиль фирмы (MONTE_CARLO §2.1–2.2).

Всё в R при номинальном риске 1R на сделку. Путь дня — точки эквити относительно `E_SOD`.

Пока движок (M2) не отдаёт поминутную эквити, путь строится из MAE/MFE сделки
(DECISIONS P-25): стоп-выход — `0 → MFE → MAE = close` (MFE обязан быть раньше стопа),
выход по лучшей цене — `0 → MAE → close`, прочие — `0 → MFE → MAE → close` (худший порядок).
Издержки сделки списываются в момент входа. Пересекающиеся по времени сделки склеиваются
в группу: низ группы = сумма MAE (все худшие точки одновременно), верх = лучший MFE одной сделки.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

from pbl.prop.servertime import server_date
from pbl.schemas import ServerTime, Trade

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
B8 = npt.NDArray[np.bool_]

_TOL = 1e-9


@dataclass(frozen=True)
class LotSpec:
    """Мин. лот и шаг лота инструмента (для округления объёма, MONTE_CARLO §3.1)."""

    min_lot: float
    lot_step: float


@dataclass(frozen=True)
class TradeArrays:
    """Колонки сделок в numpy: вход стрессов и построителя дней."""

    day: I64  # серверная дата открытия, ordinal
    open_s: I64  # время открытия, секунды UTC
    close_s: I64
    r_net: F64
    r_gross: F64
    cost_spread: F64
    cost_comm: F64
    cost_swap: F64
    cost_slip: F64
    mae: F64
    mfe: F64
    dv: F64  # stop_dist × point_value, $ риска на 1 лот
    min_lot: F64  # nan — спеки нет, κ = 1
    lot_step: F64
    news: B8
    close_day: I64

    def __len__(self) -> int:
        return int(self.r_net.shape[0])

    def take(self, mask_or_idx: npt.NDArray[np.bool_] | I64) -> TradeArrays:
        return TradeArrays(**{k: v[mask_or_idx] for k, v in self.__dict__.items()})

    def with_(self, **cols: npt.NDArray[np.generic]) -> TradeArrays:
        return replace(self, **cols)  # type: ignore[arg-type]


def trades_to_arrays(
    trades: Sequence[Trade],
    server: ServerTime,
    lot_specs: Mapping[str, LotSpec] | None = None,
) -> TradeArrays:
    if not trades:
        raise ValueError("пустой список сделок")
    specs = lot_specs or {}

    def col(values: list[float]) -> F64:
        return np.asarray(values, dtype=np.float64)

    nan = float("nan")
    return TradeArrays(
        day=np.asarray([server_date(t.open_utc, server).toordinal() for t in trades], np.int64),
        close_day=np.asarray(
            [server_date(t.close_utc, server).toordinal() for t in trades], np.int64
        ),
        open_s=np.asarray([int(t.open_utc.timestamp()) for t in trades], np.int64),
        close_s=np.asarray([int(t.close_utc.timestamp()) for t in trades], np.int64),
        r_net=col([t.r_net for t in trades]),
        r_gross=col([t.r_gross for t in trades]),
        cost_spread=col([t.cost_spread_r for t in trades]),
        cost_comm=col([t.cost_comm_r for t in trades]),
        cost_swap=col([t.cost_swap_r for t in trades]),
        cost_slip=col([t.cost_slip_r for t in trades]),
        mae=col([t.mae_r for t in trades]),
        mfe=col([t.mfe_r for t in trades]),
        dv=col([t.stop_dist * t.point_value for t in trades]),
        min_lot=col([specs[t.symbol].min_lot if t.symbol in specs else nan for t in trades]),
        lot_step=col([specs[t.symbol].lot_step if t.symbol in specs else nan for t in trades]),
        news=np.asarray([t.news_flag for t in trades], dtype=np.bool_),
    )


@dataclass(frozen=True)
class DayTable:
    """Дневные записи в формате CSR (MONTE_CARLO §13): вход ядра MC."""

    dates: I64  # ordinal серверной даты
    n_opened: I64
    db: F64  # изменение баланса за день, R
    b_max: F64  # максимум накопленного закрытого результата внутри дня
    c_in: F64
    c_out: F64
    path_off: I64  # точки дня d: path_val[path_off[d]:path_off[d+1]]
    path_val: F64
    risk_open_max: F64  # максимум одновременно открытых сделок (по 1R)
    news_pos: F64  # прибыль сделок с news_flag, R
    single_loss: F64  # худший плавающий убыток одной сделки, R (≥ 0)
    cost_spread: F64
    cost_comm: F64
    cost_swap: F64
    cost_slip: F64
    gross: F64
    # сделки → дни (для κ)
    trade_day: I64
    trade_dv: F64
    trade_min_lot: F64
    trade_lot_step: F64
    eps: float = 0.0
    r_win_mean: float = 1.0
    n_trades: int = 0
    overnight_trades: int = 0
    weekend_folded: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_days(self) -> int:
        return int(self.dates.shape[0])

    @property
    def n_trading_days(self) -> int:
        return int((self.n_opened > 0).sum())

    def years(self) -> I64:
        return np.asarray(
            [dt.date.fromordinal(int(d)).year for d in self.dates], dtype=np.int64
        )

    def take(self, idx: I64) -> DayTable:
        """Подтаблица (или псевдо-история) из дней `idx` в заданном порядке."""
        idx = np.asarray(idx, dtype=np.int64)
        lens = self.path_off[idx + 1] - self.path_off[idx]
        off = np.zeros(idx.shape[0] + 1, dtype=np.int64)
        np.cumsum(lens, out=off[1:])
        pos = np.repeat(self.path_off[idx] - off[:-1], lens) + np.arange(off[-1])
        # сделки: дни могут повторяться — размножаем сделки под каждое вхождение
        order = np.argsort(self.trade_day, kind="stable")
        tday = self.trade_day[order]
        starts = np.searchsorted(tday, idx, side="left")
        ends = np.searchsorted(tday, idx, side="right")
        cnt = ends - starts
        t_idx = (
            np.concatenate([order[s:e] for s, e in zip(starts, ends, strict=True)])
            if cnt.sum()
            else np.zeros(0, dtype=np.int64)
        )
        return replace(
            self,
            dates=self.dates[idx],
            n_opened=self.n_opened[idx],
            db=self.db[idx],
            b_max=self.b_max[idx],
            c_in=self.c_in[idx],
            c_out=self.c_out[idx],
            path_off=off,
            path_val=self.path_val[pos],
            risk_open_max=self.risk_open_max[idx],
            news_pos=self.news_pos[idx],
            single_loss=self.single_loss[idx],
            cost_spread=self.cost_spread[idx],
            cost_comm=self.cost_comm[idx],
            cost_swap=self.cost_swap[idx],
            cost_slip=self.cost_slip[idx],
            gross=self.gross[idx],
            trade_day=np.repeat(np.arange(idx.shape[0], dtype=np.int64), cnt),
            trade_dv=self.trade_dv[t_idx],
            trade_min_lot=self.trade_min_lot[t_idx],
            trade_lot_step=self.trade_lot_step[t_idx],
        )

    def kappa(self, money_risk_usd: float) -> tuple[F64, float]:
        """κ_d = max κ_i по сделкам дня (MONTE_CARLO §3.1) и доля сделок с κ_i > 1.5.

        Без спеки инструмента (min_lot = nan) κ = 1: лот не округляется.
        """
        k_day = np.ones(self.n_days, dtype=np.float64)
        if self.trade_day.shape[0] == 0:
            return k_day, 0.0
        if money_risk_usd <= 0:
            raise ValueError("риск в $ должен быть > 0")
        raw = money_risk_usd / self.trade_dv
        step = self.trade_lot_step
        known = np.isfinite(step) & np.isfinite(self.trade_min_lot)
        safe_step = np.where(known, step, 1.0)
        lots = np.floor(raw / safe_step + 1e-9) * safe_step
        lots = np.maximum(np.where(known, self.trade_min_lot, 0.0), lots)
        k_trade = np.where(known, lots * self.trade_dv / money_risk_usd, 1.0)
        k_day.fill(0.0)
        np.maximum.at(k_day, self.trade_day, k_trade)
        k_day[k_day == 0.0] = 1.0
        return k_day, float((k_trade > 1.5).mean())


def _trade_points(r_net: float, r_gross: float, mae: float, mfe: float) -> tuple[float, float]:
    """Низ и верх плавающего результата сделки в чистых R (издержки — в момент входа)."""
    cost = r_gross - r_net
    low = min(mae - cost, r_net)
    high = max(mfe - cost, r_net)
    return low, high


def _weekday_calendar(first: int, last: int) -> I64:
    days = np.arange(first, last + 1, dtype=np.int64)
    # ordinal 1 = понедельник 0001-01-01 → weekday = (ordinal − 1) % 7
    return days[(days - 1) % 7 < 5]


def build_day_table(ta: TradeArrays, *, eps: float = 0.0) -> DayTable:
    """Дневные записи из сделок: календарь Пн–Пт от первого до последнего дня, пустые дни тоже.

    Сделки выходного дня (по серверу) переносятся на предыдущую пятницу.
    """
    n = len(ta)
    if n == 0:
        raise ValueError("пустой список сделок")
    day = ta.day.copy()
    wd = (day - 1) % 7
    weekend = wd >= 5
    day[weekend] -= wd[weekend] - 4
    cal = _weekday_calendar(int(day.min()), int(day.max()))
    pos_of = {int(d): i for i, d in enumerate(cal)}
    nd = cal.shape[0]

    n_opened = np.zeros(nd, dtype=np.int64)
    db = np.zeros(nd)
    b_max = np.zeros(nd)
    risk_open = np.zeros(nd)
    news_pos = np.zeros(nd)
    single_loss = np.zeros(nd)
    costs = {k: np.zeros(nd) for k in ("spread", "comm", "swap", "slip", "gross")}
    paths: list[list[float]] = [[0.0] for _ in range(nd)]
    trade_day = np.asarray([pos_of[int(d)] for d in day], dtype=np.int64)

    order = np.lexsort((ta.open_s, trade_day))
    i = 0
    while i < n:
        d = int(trade_day[order[i]])
        j = i
        while j < n and int(trade_day[order[j]]) == d:
            j += 1
        idx = order[i:j]
        n_opened[d] = j - i
        costs["spread"][d] = ta.cost_spread[idx].sum()
        costs["comm"][d] = ta.cost_comm[idx].sum()
        costs["swap"][d] = ta.cost_swap[idx].sum()
        costs["slip"][d] = ta.cost_slip[idx].sum()
        costs["gross"][d] = ta.r_gross[idx].sum()
        news_pos[d] = float(np.clip(ta.r_net[idx][ta.news[idx]], 0.0, None).sum())

        # максимум одновременно открытых сделок
        ev = sorted(
            [(int(ta.open_s[k]), 1) for k in idx] + [(int(ta.close_s[k]), -1) for k in idx],
            key=lambda e: (e[0], e[1]),
        )
        cur = peak = 0
        for _, delta in ev:
            cur += delta
            peak = max(peak, cur)
        risk_open[d] = float(max(peak, 1))

        # группы пересекающихся сделок → точки пути
        path = paths[d]
        realized = 0.0
        best_realized = 0.0
        worst_single = 0.0
        g = 0
        m = idx.shape[0]
        while g < m:
            grp = [int(idx[g])]
            g_end = int(ta.close_s[idx[g]])
            h = g + 1
            while h < m and int(ta.open_s[idx[h]]) < g_end:
                grp.append(int(idx[h]))
                g_end = max(g_end, int(ta.close_s[idx[h]]))
                h += 1
            lows: list[float] = []
            highs: list[float] = []
            r_sum = 0.0
            for k in grp:
                lo, hi = _trade_points(
                    float(ta.r_net[k]), float(ta.r_gross[k]), float(ta.mae[k]), float(ta.mfe[k])
                )
                lows.append(lo)
                highs.append(hi)
                r_sum += float(ta.r_net[k])
                worst_single = max(worst_single, -lo)
            low = sum(lows)
            high = max(highs) if len(grp) > 1 else highs[0]
            low = min(low, r_sum)
            high = max(high, r_sum)
            pts = [high, low, r_sum] if high > r_sum + _TOL else [low, r_sum]
            for v in pts:
                val = realized + v
                if abs(val - path[-1]) > _TOL:
                    path.append(val)
            realized += r_sum
            best_realized = max(best_realized, realized)
            g = h
        db[d] = realized
        b_max[d] = best_realized
        single_loss[d] = worst_single
        i = j

    off = np.zeros(nd + 1, dtype=np.int64)
    np.cumsum([len(p) for p in paths], out=off[1:])
    path_val = np.fromiter((v for p in paths for v in p), dtype=np.float64, count=int(off[-1]))

    wins = ta.r_net[ta.r_net > 0]
    warnings: list[str] = []
    n_td = int((n_opened > 0).sum())
    if n_td < 250 or n < 100:
        warnings.append(f"МАЛО ДАННЫХ: {n_td} торговых дней, {n} сделок (нужно ≥ 250 и ≥ 100)")
    overnight = int((ta.close_day != ta.day).sum())
    if overnight:
        warnings.append(
            f"перенос через полночь сервера: {overnight} сделок ({overnight / n:.1%}) — "
            "результат отнесён ко дню открытия"
        )
    if weekend.any():
        warnings.append(f"сделки выходного дня перенесены на пятницу: {int(weekend.sum())}")

    zeros = np.zeros(nd)
    return DayTable(
        dates=cal,
        n_opened=n_opened,
        db=db,
        b_max=b_max,
        c_in=zeros.copy(),
        c_out=zeros.copy(),
        path_off=off,
        path_val=path_val,
        risk_open_max=risk_open,
        news_pos=news_pos,
        single_loss=single_loss,
        cost_spread=costs["spread"],
        cost_comm=costs["comm"],
        cost_swap=costs["swap"],
        cost_slip=costs["slip"],
        gross=costs["gross"],
        trade_day=trade_day,
        trade_dv=ta.dv.copy(),
        trade_min_lot=ta.min_lot.copy(),
        trade_lot_step=ta.lot_step.copy(),
        eps=eps,
        r_win_mean=float(wins.mean()) if wins.size else 1.0,
        n_trades=n,
        overnight_trades=overnight,
        weekend_folded=int(weekend.sum()),
        warnings=tuple(warnings),
    )


def day_table_from_trades(
    trades: Sequence[Trade],
    server: ServerTime,
    lot_specs: Mapping[str, LotSpec] | None = None,
) -> DayTable:
    return build_day_table(trades_to_arrays(trades, server, lot_specs))
