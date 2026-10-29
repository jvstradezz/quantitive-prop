"""Счётчик вариантов для DSR и PBO (VALIDATION §4.3, D-028).

Каждый прогнанный вариант идеи — строка в `research/registry/variants/<IDEA-id>.parquet`, даже если
он провалил S1: N в DSR считается по всем пробам, а не по выжившим. Вариант = стратегия с
параметрами × инструмент × ТФ × сессия × выходы × фильтры × профиль фирмы × версия кода.

В файл попадают только сделки IS/WF (внутри ~70% истории, на которых вариант выбирается).
Holdout и демо сюда не пишутся: так holdout нельзя «подсмотреть» через реестр (VALIDATION §2).

Дневная доходность — сумма `r_net` сделок дня при номинальном риске 1R, день — по серверу FN,
выходные относим к пятнице (D-022). Это вход для DSR при фиксированном риске.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import Field

from pbl.config import REGISTRY
from pbl.prop.servertime import server_date
from pbl.schemas import IdeaCard, Trade
from pbl.schemas.common import StrictModel
from pbl.schemas.firm import ServerTime
from pbl.schemas.strategy import ExitSpec, FilterSpec, ParamValue, Timeframe

F64 = npt.NDArray[np.float64]
_ONE_DAY = np.timedelta64(1, "D")


def _d64(day: dt.date) -> np.datetime64:
    return np.datetime64(day, "D")


ALLOWED_SAMPLES = frozenset({"IS", "WF"})
MAX_SHARD_BYTES = 50 * 2**20
"""Предел размера одного файла: GitHub не принимает > 100 МБ, а diff бинарника не читается."""

_COLUMNS: dict[str, pl.DataType | type[pl.DataType]] = {
    "variant_id": pl.String,
    "idea_id": pl.String,
    "family": pl.String,
    "version": pl.String,
    "instrument": pl.String,
    "timeframe": pl.String,
    "session": pl.String,
    "firm_profile": pl.String,
    "spec_json": pl.String,
    "run_id": pl.String,
    "logged_on": pl.Date,
    "is_start": pl.Date,
    "is_end": pl.Date,
    "n_trades": pl.Int32,
    "n_days": pl.Int32,
    "net_r": pl.Float64,
    "pf": pl.Float64,
    "sharpe_daily": pl.Float64,
    "skew": pl.Float64,
    "kurt": pl.Float64,
    "series_dates": pl.List(pl.Date),
    "series_r": pl.List(pl.Float32),
}
SCHEMA = pl.Schema(_COLUMNS)


class VariantConflict(ValueError):
    """Тот же вариант вернул другой результат: поднимите `version` (новая проба = новая строка)."""


class VariantSpec(StrictModel):
    idea_id: str
    family: str
    version: str = "v1"
    instrument: str
    timeframe: Timeframe
    session: str | None = None
    params: dict[str, ParamValue] = Field(default_factory=dict)
    exits: list[ExitSpec] = Field(min_length=1)
    filters: list[FilterSpec] = Field(default_factory=list)
    firm_profile: str

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )

    @property
    def variant_id(self) -> str:
        """Стабильный id: тот же `Trade.variant_id`, что пишет движок."""
        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        return f"V-{digest[:16]}"


@dataclass(frozen=True, eq=False)
class DailyR:
    """Дни со сделками и сумма `r_net` за день; окно IS задаёт календарь для статистик."""

    start: dt.date
    end: dt.date
    dates: tuple[dt.date, ...]
    r: tuple[float, ...]
    n_trades: int
    win_r: float
    loss_r: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("окно IS: end раньше start")
        if len(self.dates) != len(self.r):
            raise ValueError("dates и r разной длины")
        if list(self.dates) != sorted(set(self.dates)):
            raise ValueError("dates должны быть уникальны и отсортированы")
        if any(d.weekday() >= 5 for d in self.dates):
            raise ValueError("в дневном ряду только дни Пн–Пт")
        if self.dates and not (self.start <= self.dates[0] and self.dates[-1] <= self.end):
            raise ValueError("дни вне окна IS")

    @property
    def n_days(self) -> int:
        return int(np.busday_count(_d64(self.start), _d64(self.end) + _ONE_DAY))


def trading_day(ts_utc: dt.datetime, server: ServerTime) -> dt.date:
    """День по серверу; суббота и воскресенье → пятница (D-022)."""
    day = server_date(ts_utc, server)
    return day - dt.timedelta(days=max(day.weekday() - 4, 0))


def daily_r_from_trades(
    trades: Iterable[Trade], *, start: dt.date, end: dt.date, server: ServerTime
) -> DailyR:
    """Свёртка сделок IS/WF в дневной ряд. HOLDOUT и DEMO отвергаются."""
    by_day: dict[dt.date, float] = {}
    n = 0
    win = loss = 0.0
    for t in trades:
        if t.sample not in ALLOWED_SAMPLES:
            raise ValueError(f"{t.trade_id}: sample={t.sample}; в счётчик идут только IS и WF")
        day = trading_day(t.close_utc, server)
        if not start <= day <= end:
            raise ValueError(f"{t.trade_id}: день {day} вне окна IS {start}…{end}")
        by_day[day] = by_day.get(day, 0.0) + t.r_net
        n += 1
        if t.r_net > 0:
            win += t.r_net
        else:
            loss += t.r_net
    days = sorted(by_day)
    return DailyR(
        start=start,
        end=end,
        dates=tuple(days),
        r=tuple(by_day[d] for d in days),
        n_trades=n,
        win_r=win,
        loss_r=loss,
    )


def _dense(daily: DailyR) -> F64:
    x = np.zeros(daily.n_days)
    if daily.dates:
        idx = np.busday_count(_d64(daily.start), np.array(daily.dates, dtype="datetime64[D]"))
        x[idx] = np.asarray(daily.r, dtype=np.float64)
    return x


def daily_stats(daily: DailyR) -> dict[str, float]:
    """Sharpe дневной (ddof=1), асимметрия и эксцесс (нормальное = 3) по календарю Пн–Пт."""
    x = _dense(daily)
    mean = float(x.mean()) if x.size else 0.0
    std = float(x.std(ddof=1)) if x.size > 1 else 0.0
    if std == 0.0:
        return {"sharpe_daily": 0.0, "skew": 0.0, "kurt": 3.0}
    z = (x - mean) / float(x.std(ddof=0))
    return {
        "sharpe_daily": mean / std,
        "skew": float(np.mean(z**3)),
        "kurt": float(np.mean(z**4)),
    }


@dataclass(frozen=True, eq=False)
class ReturnsMatrix:
    """T × N дневных доходностей вариантов; NaN вне окна IS самого варианта."""

    dates: list[dt.date]
    ids: list[str]
    values: F64


class VariantLog:
    """Лог проб одной идеи. Изменения копятся в памяти до `save()`."""

    def __init__(self, path: Path, idea_id: str, frame: pl.DataFrame | None = None) -> None:
        self.path = path
        self.idea_id = idea_id
        self._frame = frame if frame is not None else pl.DataFrame(schema=SCHEMA)

    @classmethod
    def shard_path(cls, idea_id: str, registry: Path | None = None) -> Path:
        return (registry or REGISTRY) / "variants" / f"{idea_id}.parquet"

    @classmethod
    def for_idea(cls, idea_id: str, registry: Path | None = None) -> VariantLog:
        path = cls.shard_path(idea_id, registry)
        frame = pl.read_parquet(path).select(list(SCHEMA)).cast(SCHEMA) if path.is_file() else None
        return cls(path, idea_id, frame)

    def __len__(self) -> int:
        return self._frame.height

    def frame(self) -> pl.DataFrame:
        return self._frame

    def ids(self) -> list[str]:
        return self._frame["variant_id"].to_list()

    def count(self, *, family: str | None = None, instrument: str | None = None) -> int:
        f = self._frame
        if family is not None:
            f = f.filter(pl.col("family") == family)
        if instrument is not None:
            f = f.filter(pl.col("instrument") == instrument)
        return f.height

    def add(
        self,
        spec: VariantSpec,
        daily: DailyR,
        *,
        run_id: str,
        on: dt.date,
        keep_series: bool = True,
    ) -> bool:
        """Записать пробу. True — новая; False — та же проба уже была с тем же результатом."""
        if spec.idea_id != self.idea_id:
            raise ValueError(f"вариант идеи {spec.idea_id} в логе {self.idea_id}")
        vid = spec.variant_id
        net_r = float(sum(daily.r))
        if vid in set(self.ids()):
            old = self._frame.filter(pl.col("variant_id") == vid).row(0, named=True)
            same = (
                old["n_trades"] == daily.n_trades
                and old["is_start"] == daily.start
                and old["is_end"] == daily.end
                and abs(old["net_r"] - net_r) <= 1e-9 * max(1.0, abs(net_r))
            )
            if not same:
                raise VariantConflict(
                    f"{vid}: результат изменился (было {old['n_trades']} сделок, "
                    f"{old['net_r']:.6f} R; стало {daily.n_trades}, {net_r:.6f} R); "
                    "поднимите version варианта"
                )
            return False
        pf = daily.win_r / abs(daily.loss_r) if daily.loss_r < 0 else 0.0
        row: dict[str, Any] = {
            "variant_id": vid,
            "idea_id": spec.idea_id,
            "family": spec.family,
            "version": spec.version,
            "instrument": spec.instrument,
            "timeframe": spec.timeframe,
            "session": spec.session,
            "firm_profile": spec.firm_profile,
            "spec_json": spec.canonical_json(),
            "run_id": run_id,
            "logged_on": on,
            "is_start": daily.start,
            "is_end": daily.end,
            "n_trades": daily.n_trades,
            "n_days": daily.n_days,
            "net_r": net_r,
            "pf": pf,
            **daily_stats(daily),
            "series_dates": list(daily.dates) if keep_series else None,
            "series_r": list(daily.r) if keep_series else None,
        }
        self._frame = pl.concat(
            [self._frame, pl.DataFrame([row], schema=SCHEMA)], how="vertical", rechunk=True
        )
        return True

    def save(self) -> None:
        """Атомарная запись; строки отсортированы по id — минимум шума в git."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".parquet.tmp")
        self._frame.sort("variant_id").write_parquet(tmp, compression="zstd")
        if tmp.stat().st_size > MAX_SHARD_BYTES:
            tmp.unlink()
            raise ValueError(
                f"{self.path.name} > {MAX_SHARD_BYTES // 2**20} МБ: пишите keep_series=False "
                "для вариантов, которые не прошли S1"
            )
        os.replace(tmp, self.path)

    def returns_matrix(self, ids: Sequence[str] | None = None) -> ReturnsMatrix:
        """Матрица для PBO: дни Пн–Пт, NaN вне окна варианта. Нужны сохранённые ряды."""
        want = list(ids) if ids is not None else self.ids()
        if not want:
            raise ValueError("нет вариантов")
        rows = {
            r["variant_id"]: r
            for r in self._frame.filter(pl.col("variant_id").is_in(want)).iter_rows(named=True)
        }
        missing = [v for v in want if v not in rows]
        if missing:
            raise KeyError(f"нет в логе: {missing[:3]}")
        no_series = [v for v in want if rows[v]["series_dates"] is None]
        if no_series:
            raise ValueError(f"ряды не сохранены (keep_series=False): {no_series[:3]}")
        lo = min(rows[v]["is_start"] for v in want)
        hi = max(rows[v]["is_end"] for v in want)
        days = np.arange(_d64(lo), _d64(hi) + _ONE_DAY, dtype="datetime64[D]")
        days = days[np.is_busday(days)]
        mat = np.full((days.size, len(want)), np.nan)
        for j, vid in enumerate(want):
            r = rows[vid]
            a = int(np.searchsorted(days, _d64(r["is_start"]), side="left"))
            b = int(np.searchsorted(days, _d64(r["is_end"]), side="right"))
            mat[a:b, j] = 0.0
            if r["series_dates"]:
                at = np.searchsorted(days, np.array(r["series_dates"], dtype="datetime64[D]"))
                mat[at, j] = np.asarray(r["series_r"], dtype=np.float64)
        return ReturnsMatrix(dates=[d.astype(dt.date) for d in days], ids=want, values=mat)


def refresh_card_counter(card: IdeaCard, log: VariantLog) -> IdeaCard:
    """Карточка с `variants_tested`, равным числу проб в логе (дубль счётчика, IDEAS §3)."""
    return card.model_copy(update={"variants_tested": len(log)})
