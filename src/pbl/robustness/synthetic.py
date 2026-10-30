"""Синтетические стратегии с известным ответом для проверки гейтов (D-007). Это не рынок.

Одна сделка в день на инструмент: выход по стопу (−1R) или тейку (+rr R), издержки `cost_r`.
Случайность по умолчанию общая для всех вариантов (одни и те же дни, стороны и броски): соседние
параметры дают почти те же сделки, как у настоящей стратегии. `lookback` меняет только долю
взятых дней (вложенные подмножества), фильтр дня — какие корзины «день × сторона» берутся.
Истинная вероятность выигрыша задаётся функцией `edge`, поэтому правильный вердикт известен.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import polars as pl

from pbl.robustness.study import FilterDef, Params, Space, TradesStudy, params_key
from pbl.robustness.trades import TRADE_SCHEMA
from pbl.schemas import ServerTime, Trade

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
I8 = npt.NDArray[np.int8]

# (день недели, сторона), которые разрешает «фильтр автора» (как у идеи №1): пн, ср — лонг,
# чт — шорт, вт и пт — обе стороны
AUTHOR_BUCKETS = frozenset({(0, 1), (1, 1), (1, -1), (2, 1), (3, -1), (4, 1), (4, -1)})

UTC_SERVER = ServerTime(winter_utc_offset=0, summer_utc_offset=0, dst_calendar=None, verified=True)

Edge = Callable[[Params, str, I64, I8, F64], F64]
"""(параметры, символ, день недели, сторона, доля истории 0…1) → P(выигрыша) каждой сделки."""


def author_mask(wd: npt.ArrayLike, side: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    pairs = zip(np.asarray(wd).tolist(), np.asarray(side).tolist(), strict=True)
    return np.array([(int(w), int(s)) in AUTHOR_BUCKETS for w, s in pairs], dtype=bool)


DAY_FILTER = FilterDef(
    name="фильтр дня",
    param="dow",
    off="none",
    allowed=lambda t: pl.Series(author_mask(t["wd"], t["side"])),
)


def flat(p_win: float) -> Edge:
    def f(params: Params, sym: str, wd: I64, side: I8, frac: F64) -> F64:
        return np.full(wd.shape, p_win)

    return f


@dataclass
class SyntheticStudy(TradesStudy):
    edge: Edge
    symbols: tuple[str, ...] = ("NAS100", "US30", "SPX500")
    start: dt.date = dt.date(2019, 1, 1)
    end: dt.date = dt.date(2024, 12, 31)
    trade_prob: float = 0.9
    cost_r: float = 0.05
    seed: int = 11
    independent: bool = False  # свой поток случайности у каждого варианта (чистый шум)
    strategy_id: str = "synthetic"
    family: str | None = "synthetic"
    server: ServerTime = field(default_factory=lambda: UTC_SERVER)

    @property
    def period(self) -> tuple[dt.date, dt.date]:
        return self.start, self.end

    def _draws(self, params: Params, k: int) -> tuple[I64, I8, F64, F64, F64, F64]:
        days = np.arange(self.start.toordinal(), self.end.toordinal() + 1, dtype=np.int64)
        days = days[(days - 1) % 7 < 5]
        seq: list[int] = [self.seed, k]
        if self.independent:
            h = hashlib.sha256(params_key(params).encode()).digest()
            seq.append(int.from_bytes(h[:4], "little"))
        rng = np.random.default_rng(seq)
        u_take, u_win, u_min, u_hold = rng.random((4, days.size))
        side = np.where(rng.random(days.size) < 0.5, 1, -1).astype(np.int8)
        return days, side, u_take, u_win, u_min, u_hold

    def _arrays(self, params: Params) -> dict[str, npt.NDArray[np.generic]]:
        lookback = float(params.get("lookback", 30))
        share = self.trade_prob * min(1.0, 0.7 + 0.3 * lookback / 50)
        rr = float(params.get("rr", 1.0))
        span = self.end.toordinal() - self.start.toordinal()
        cols: dict[str, list[npt.NDArray[np.generic]]] = {}
        for k, sym in enumerate(self.symbols):
            days, side, u_take, u_win, u_min, u_hold = self._draws(params, k)
            take = u_take < share
            wd = ((days - 1) % 7).astype(np.int64)
            if params.get("dow", "none") != "none":
                take &= author_mask(wd, side)
            days, side, wd = days[take], side[take], wd[take]
            u_win, u_min, u_hold = u_win[take], u_min[take], u_hold[take]
            frac = (days - self.start.toordinal()) / span
            win = u_win < self.edge(params, sym, wd, side, frac)
            r_gross = np.where(win, rr, -1.0)
            part = {
                "day": days,
                "wd": wd,
                "minute": (14 * 60 + 30 + u_min * 60).astype(np.int64),
                "symbol": np.full(days.size, sym),
                "side": side,
                "r_net": r_gross - self.cost_r,
                "r_gross": r_gross,
                "cost_r": np.full(days.size, self.cost_r),
                "hold_sec": (600 + u_hold * 10_200).astype(np.int64),
            }
            for c, v in part.items():
                cols.setdefault(c, []).append(np.asarray(v))
        return {c: np.concatenate(v) for c, v in cols.items()}

    def frame(self, params: Params) -> pl.DataFrame:
        a = self._arrays(params)
        return pl.DataFrame({c: a[c] for c in TRADE_SCHEMA}).cast(TRADE_SCHEMA).sort("day")  # type: ignore[arg-type]

    def trades(self, params: Params) -> list[Trade]:
        a = self._arrays(params)
        rr = float(params.get("rr", 1.0))
        stop, spread, slip = 50.0, 0.6 * self.cost_r, 0.4 * self.cost_r
        out: list[Trade] = []
        for i in np.argsort(a["day"], kind="stable"):
            day = dt.date.fromordinal(int(a["day"][i]))
            minute = int(a["minute"][i])
            open_utc = dt.datetime.combine(day, dt.time(minute // 60, minute % 60), dt.UTC)
            hold = int(a["hold_sec"][i])
            sign = 1.0 if int(a["side"][i]) > 0 else -1.0
            entry = 10_000.0
            win = float(a["r_gross"][i]) > 0
            out.append(
                Trade(
                    trade_id=f"{self.strategy_id}-{len(out):06d}",
                    strategy_id=self.strategy_id,
                    variant_id=params_key(params),
                    symbol=str(a["symbol"][i]),
                    side="long" if sign > 0 else "short",
                    open_utc=open_utc,
                    close_utc=open_utc + dt.timedelta(seconds=hold),
                    entry=entry,
                    exit=entry + sign * stop * (rr if win else -1.0),
                    sl=entry - sign * stop,
                    tp=entry + sign * stop * rr,
                    stop_dist=stop,
                    point_value=1.0,
                    r_gross=float(a["r_gross"][i]),
                    cost_spread_r=spread,
                    cost_comm_r=0.0,
                    cost_swap_r=0.0,
                    cost_slip_r=slip,
                    r_net=float(a["r_gross"][i]) - spread - slip,
                    mae_r=-0.5 if win else -1.0,
                    mfe_r=rr if win else 0.5,
                    hold_sec=hold,
                    exit_reason="tp" if win else "sl",
                    sample="IS",
                )
            )
        return out


# --- случаи с известным ответом ------------------------------------------------------------

LOOKBACKS = (20, 30, 40)


def _space(*, dow: bool = False) -> Space:
    filters = ("none", "author") if dow else ("none",)
    variants: list[Params] = [{"lookback": lb, "dow": d} for lb in LOOKBACKS for d in filters]
    return Space(
        variants=variants,
        numeric=("lookback",),
        filters=(DAY_FILTER,),
        integer=frozenset({"lookback"}),
    )


def _by_symbol(table: dict[str, float], default: float) -> Edge:
    def f(params: Params, sym: str, wd: I64, side: I8, frac: F64) -> F64:
        return np.full(wd.shape, table.get(sym, default))

    return f


def _decay(before: float, after: float, at: float = 0.7) -> Edge:
    def f(params: Params, sym: str, wd: I64, side: I8, frac: F64) -> F64:
        return np.where(frac < at, before, after)

    return f


def _buckets(allowed: float, forbidden: float) -> Edge:
    def f(params: Params, sym: str, wd: I64, side: I8, frac: F64) -> F64:
        return np.where(author_mask(wd, side), allowed, forbidden)

    return f


def _spike(at: int, p_win: float, other: float) -> Edge:
    def f(params: Params, sym: str, wd: I64, side: I8, frac: F64) -> F64:
        return np.full(wd.shape, p_win if params.get("lookback") == at else other)

    return f


CASES: dict[str, tuple[str, str]] = {
    "edge": ("перевес на всех инструментах и параметрах", "в демо"),
    "noise": ("нет перевеса, издержки 0.05R", "отклонена на S1"),
    "decay": ("перевес только в первых 70% истории", "отклонена на holdout"),
    "single": ("перевес только на одном инструменте", "отклонена: инструменты"),
    "fake-filter": ("фильтр дня без эффекта", "фильтр не держится, финалист без фильтра"),
    "real-filter": ("перевес только в корзинах фильтра", "финалист с фильтром"),
    "fragile": ("перевес только при lookback = 30", "отклонена: чувствительность"),
}
"""Имя → (что заложено, правильный ответ)."""


def synthetic_case(name: str, *, seed: int = 1) -> tuple[SyntheticStudy, Space]:
    if name == "edge":
        return SyntheticStudy(flat(0.62), seed=seed, strategy_id="synth-edge"), _space()
    if name == "noise":
        return SyntheticStudy(flat(0.5), seed=seed, strategy_id="synth-noise"), _space()
    if name == "decay":
        return SyntheticStudy(_decay(0.62, 0.45), seed=seed, strategy_id="synth-decay"), _space()
    if name == "single":
        study = SyntheticStudy(
            _by_symbol({"NAS100": 0.75}, 0.5),
            symbols=("NAS100", "US30"),
            seed=seed,
            strategy_id="synth-single",
        )
        return study, _space()
    if name == "fake-filter":
        return SyntheticStudy(flat(0.62), seed=seed, strategy_id="synth-fake-filter"), _space(
            dow=True
        )
    if name == "real-filter":
        study = SyntheticStudy(_buckets(0.68, 0.42), seed=seed, strategy_id="synth-real-filter")
        return study, _space(dow=True)
    if name == "fragile":
        study = SyntheticStudy(
            _spike(30, 0.66, 0.5), seed=seed, independent=True, strategy_id="synth-fragile"
        )
        return study, Space(
            variants=[{"lookback": 30, "dow": "none"}],
            numeric=("lookback",),
            integer=frozenset({"lookback"}),
        )
    raise KeyError(f"нет синтетики {name!r}; есть: {', '.join(CASES)}")
