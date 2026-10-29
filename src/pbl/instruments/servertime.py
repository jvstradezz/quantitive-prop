"""Время сервера фирмы: граница торгового дня для проп-правил (D-006, MONTE_CARLO §2.2).

FundedNext: GMT+2 зимой / GMT+3 летом, переход по календарю США (O-10 закрыт 2026-10-28,
D-019). Храним всё в UTC и переводим в серверную дату только здесь.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import numpy.typing as npt

from pbl.schemas import ServerTime

I64 = npt.NDArray[np.int64]


def _nth_sunday(year: int, month: int, n: int) -> dt.date:
    first = dt.date(year, month, 1)
    offset = (6 - first.weekday()) % 7
    return first + dt.timedelta(days=offset + 7 * (n - 1))


def _last_sunday(year: int, month: int) -> dt.date:
    nxt = dt.date(year + month // 12, month % 12 + 1, 1)
    last = nxt - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - 6) % 7)


def dst_window_utc(year: int, calendar: str) -> tuple[dt.datetime, dt.datetime]:
    """Интервал летнего времени [start, end) в UTC для календаря `us` или `eu`."""
    if calendar == "us":
        # 2:00 местного: второе вс марта (EST, UTC−5) → первое вс ноября (EDT, UTC−4)
        start = dt.datetime.combine(_nth_sunday(year, 3, 2), dt.time(7), dt.UTC)
        end = dt.datetime.combine(_nth_sunday(year, 11, 1), dt.time(6), dt.UTC)
    elif calendar == "eu":
        start = dt.datetime.combine(_last_sunday(year, 3), dt.time(1), dt.UTC)
        end = dt.datetime.combine(_last_sunday(year, 10), dt.time(1), dt.UTC)
    else:
        raise ValueError(f"неизвестный календарь перехода: {calendar!r}")
    return start, end


def utc_offset_hours(ts_utc: dt.datetime, server: ServerTime) -> int:
    """Смещение сервера от UTC в момент `ts_utc`."""
    if server.dst_calendar is None:
        return server.winter_utc_offset
    start, end = dst_window_utc(ts_utc.year, server.dst_calendar)
    return server.summer_utc_offset if start <= ts_utc < end else server.winter_utc_offset


def to_server(ts_utc: dt.datetime, server: ServerTime) -> dt.datetime:
    """Наивное серверное время (без tzinfo) для момента `ts_utc`."""
    if ts_utc.tzinfo is None:
        raise ValueError("ожидается tz-aware UTC")
    utc = ts_utc.astimezone(dt.UTC)
    return (utc + dt.timedelta(hours=utc_offset_hours(utc, server))).replace(tzinfo=None)


def server_date(ts_utc: dt.datetime, server: ServerTime) -> dt.date:
    """Торговый день по серверу: сутки [reset, reset + 24 ч) серверного времени."""
    hh, mm = (int(x) for x in server.day_reset.split(":"))
    local = to_server(ts_utc, server) - dt.timedelta(hours=hh, minutes=mm)
    return local.date()


def server_clock(ts_s: I64, server: ServerTime) -> tuple[I64, I64, I64]:
    """Векторно: секунды UTC → (ordinal серверной даты торгового дня, минута дня, день недели).

    День и минута отсчитываются от `day_reset` (у FN 00:00). День недели: 0 — понедельник.
    """
    ts = np.asarray(ts_s, dtype=np.int64)
    offset = np.full(ts.shape, server.winter_utc_offset * 3600, dtype=np.int64)
    if server.dst_calendar is not None and ts.size:
        years = np.arange(
            dt.datetime.fromtimestamp(int(ts.min()), dt.UTC).year,
            dt.datetime.fromtimestamp(int(ts.max()), dt.UTC).year + 1,
        )
        for y in years:
            start, end = dst_window_utc(int(y), server.dst_calendar)
            summer = (ts >= int(start.timestamp())) & (ts < int(end.timestamp()))
            offset[summer] = server.summer_utc_offset * 3600
    hh, mm = (int(x) for x in server.day_reset.split(":"))
    local = ts + offset - (hh * 3600 + mm * 60)
    days = np.floor_divide(local, 86_400)
    # 1970-01-01 — ordinal 719163, четверг
    ordinal = days + 719_163
    minute = np.floor_divide(local - days * 86_400, 60)
    weekday = (days + 3) % 7
    return ordinal.astype(np.int64), minute.astype(np.int64), weekday.astype(np.int64)
