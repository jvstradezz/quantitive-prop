"""Синтетические сделки с известными свойствами (D-007, MONTE_CARLO §12).

Выгрузки сделок нет, поэтому симулятор проверяется на синтетике: монетка ±1R для задачи
о разорении, стратегия с заданным винрейтом и RR — для эталонного отчёта. Это не данные рынка.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from pbl.schemas import Trade


def synthetic_trades(
    *,
    n_days: int = 750,
    start: dt.date = dt.date(2023, 1, 2),
    trade_prob: float = 0.8,
    win_rate: float = 0.45,
    rr: float = 1.5,
    cost_r: float = 0.06,
    loser_mfe_max: float = 0.8,
    winner_mae_max: float = 0.8,
    news_prob: float = 0.0,
    exact_wins: bool = False,
    seed: int = 7,
    symbol: str = "NAS100",
    strategy_id: str = "synthetic",
    point_value: float = 10.0,
    stop_points: float = 50.0,
) -> list[Trade]:
    """Одна сделка в день Пн–Пт с вероятностью `trade_prob`, выход по стопу (−1R) или тейку (+rr R).

    Издержки `cost_r` делятся: 60% спред, 40% проскальзывание. Время входа — 14:30–15:30 UTC
    (открытие NY), удержание 10–180 мин. `exact_wins` — ровно round(win_rate · n) прибыльных.
    """
    rng = np.random.default_rng(seed)
    days = [start + dt.timedelta(days=i) for i in range(n_days * 7 // 5 + 7)]
    days = [d for d in days if d.weekday() < 5][:n_days]
    trade_days = [d for d in days if rng.random() < trade_prob]
    n = len(trade_days)
    if exact_wins:
        wins = np.zeros(n, dtype=np.bool_)
        wins[: round(win_rate * n)] = True
        rng.shuffle(wins)
    else:
        wins = rng.random(n) < win_rate
    out: list[Trade] = []
    entry = 18_000.0
    for i, (day, win) in enumerate(zip(trade_days, wins, strict=True)):
        open_utc = dt.datetime.combine(day, dt.time(14, 30), dt.UTC) + dt.timedelta(
            minutes=float(rng.uniform(0, 60))
        )
        open_utc = open_utc.replace(microsecond=0)
        close_utc = open_utc + dt.timedelta(minutes=int(rng.uniform(10, 180)))
        long = bool(rng.random() < 0.5)
        sign = 1.0 if long else -1.0
        sl = entry - sign * stop_points
        tp = entry + sign * stop_points * rr
        r_gross = rr if win else -1.0
        mae = -float(rng.uniform(0, winner_mae_max)) if win else -1.0
        mfe = rr if win else float(rng.uniform(0, loser_mfe_max))
        spread, slip = 0.6 * cost_r, 0.4 * cost_r
        out.append(
            Trade(
                trade_id=f"{strategy_id}-{i:05d}",
                strategy_id=strategy_id,
                variant_id="base",
                symbol=symbol,
                side="long" if long else "short",
                open_utc=open_utc,
                close_utc=close_utc,
                entry=entry,
                exit=tp if win else sl,
                sl=sl,
                tp=tp,
                stop_dist=stop_points,
                point_value=point_value,
                r_gross=r_gross,
                cost_spread_r=spread,
                cost_comm_r=0.0,
                cost_swap_r=0.0,
                cost_slip_r=slip,
                r_net=r_gross - spread - slip,
                mae_r=mae,
                mfe_r=mfe,
                hold_sec=int((close_utc - open_utc).total_seconds()),
                exit_reason="tp" if win else "sl",
                news_flag=bool(rng.random() < news_prob),
                sample="WF",
            )
        )
    return out


def coin_trades(n_days: int, p_win: float, *, seed: int = 1) -> list[Trade]:
    """Монетка ±1R без издержек, одна сделка каждый день, ровно round(p · n) выигрышей.

    Путь сделки — прямой ход к ±1R (MAE победителя и MFE проигравшего = 0).
    """
    return synthetic_trades(
        n_days=n_days,
        trade_prob=1.0,
        win_rate=p_win,
        rr=1.0,
        cost_r=0.0,
        loser_mfe_max=0.0,
        winner_mae_max=0.0,
        exact_wins=True,
        seed=seed,
        strategy_id="coin",
    )
