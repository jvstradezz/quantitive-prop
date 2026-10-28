"""Ядро MC: проход путей по дням (MONTE_CARLO §3).

Одна функция `_kernel` на чистом Python в стиле numba; `kernel` — её numba-версия.
`simulate(..., jit=False)` гоняет ту же логику интерпретатором (проверка V5).

Единицы: баланс и уровни — доли `B0` (старт = 1.0); путь дня — R; `s = f · κ_d` — доля `B0`
на 1R. Календарь пути: день 0 — понедельник покупки, торгуем Пн–Пт.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numba
import numpy as np
import numpy.typing as npt

from pbl.prop.days import DayTable
from pbl.prop.rules import RuleSet

F64 = npt.NDArray[np.float64]
I32 = npt.NDArray[np.int32]
I8 = npt.NDArray[np.int8]

# причины провала (взаимоисключающие, по первому событию; MONTE_CARLO §3.8)
OK, DAILY, MAX, GUARD, TIMEOUT, INACTIVITY, RULE = 0, 1, 2, 3, 4, 5, 6
REASONS = ("ok", "daily", "max", "guard_total", "timeout", "inactivity", "rule")

# индексы вектора параметров
(
    P_START,
    P_STOP,
    P_FUNDED,
    P_HAS_DAILY,
    P_DL,
    P_DL_REF,
    P_DL_BASIS,
    P_MAX_TYPE,
    P_D,
    P_CAP,
    P_LOCK_TRIG,
    P_FLOOR,
    P_GUARDS,
    P_GAMMA_D,
    P_GAMMA_T,
    P_SLIP,
    P_BUFFER,
    P_F_AFTER,
    P_R1_X,
    P_R1_K,
    P_R2,
    P_R_WIN,
    P_INACT,
    P_PAUSE,
    P_HORIZON,
    P_EV_H,
    P_HAS_PAYOUT,
    P_SPLIT,
    P_FIRST_AFTER,
    P_CYCLE,
    P_CYCLE_B,
    P_MIN_PROFIT,
    P_CONSIST,
    P_PDAYS_N,
    P_PDAYS_MIN,
    P_NEWS,
    P_REFUND_K,
    P_RR,
    P_STRIKE,
    P_STRIKES_N,
    P_EPS,
    N_PARAMS,
) = range(42)

_BIG = 1e300
_TOL = 1e-12


def _kernel(
    idx: I32,
    n_opened: npt.NDArray[np.int64],
    db: F64,
    b_max: F64,
    c_in: F64,
    path_off: npt.NDArray[np.int64],
    path_val: F64,
    risk_open: F64,
    news_pos: F64,
    single_loss: F64,
    kappa: F64,
    slot_target: F64,
    slot_min_days: npt.NDArray[np.int64],
    slot_max_days: npt.NDArray[np.int64],
    slot_f: F64,
    prm: F64,
    out_fail: I8,
    out_fail_slot: I8,
    out_pass_cal: I32,
    out_tdays: I32,
    out_end_cal: I32,
    out_pay1_cal: I32,
    out_pay1_amt: F64,
    out_pay_sum: F64,
    out_npay: I32,
    out_refund: npt.NDArray[np.bool_],
    out_bal: F64,
) -> None:
    n_paths = idx.shape[0]
    width = idx.shape[1]
    start = int(prm[P_START])
    stop = int(prm[P_STOP])
    funded = int(prm[P_FUNDED])
    has_daily = prm[P_HAS_DAILY] > 0
    dl = prm[P_DL]
    dl_ref = int(prm[P_DL_REF])
    dl_basis = int(prm[P_DL_BASIS])
    mt = int(prm[P_MAX_TYPE])
    dist = prm[P_D]
    cap = prm[P_CAP]
    lock_trig = prm[P_LOCK_TRIG]
    floor_resets = int(prm[P_FLOOR]) == 1
    guards = prm[P_GUARDS] > 0
    gamma_d = prm[P_GAMMA_D]
    gamma_t = prm[P_GAMMA_T]
    slip = prm[P_SLIP]
    buffer = prm[P_BUFFER]
    f_after = prm[P_F_AFTER]
    r1_x = prm[P_R1_X]
    r1_k = prm[P_R1_K]
    r2 = prm[P_R2] > 0
    r_win = prm[P_R_WIN]
    inact = int(prm[P_INACT])
    pause = int(prm[P_PAUSE])
    horizon = int(prm[P_HORIZON])
    ev_h = int(prm[P_EV_H])
    has_payout = prm[P_HAS_PAYOUT] > 0
    split = prm[P_SPLIT]
    first_after = int(prm[P_FIRST_AFTER])
    cycle = int(prm[P_CYCLE])
    cycle_b = int(prm[P_CYCLE_B])
    min_profit = prm[P_MIN_PROFIT]
    consist = prm[P_CONSIST]
    pdays_n = int(prm[P_PDAYS_N])
    pdays_min = prm[P_PDAYS_MIN]
    news_counted = prm[P_NEWS]
    refund_k = int(prm[P_REFUND_K])
    rr_max = prm[P_RR]
    strike_thr = prm[P_STRIKE]
    strikes_n = int(prm[P_STRIKES_N])
    eps = prm[P_EPS]

    for p in range(n_paths):
        slot = start
        cal = 0
        h = 0
        fail = OK
        fail_slot = -1
        # состояние фазы
        bal = 1.0
        h_eod = 1.0
        h_bal = 1.0
        m_eq = 1.0
        frozen = False
        l_frozen = 0.0
        tdays = 0
        slot_start = 0
        last_trade = 0
        target_hit = False
        # funded
        f_started = False
        next_check = 0
        bcount = 0
        best_day = 0.0
        pdays = 0
        news_cum = 0.0
        strikes = 0
        npay = 0
        pay1_cal = -1
        pay1_amt = 0.0
        pay_sum = 0.0
        refund = False

        while True:
            wd = cal % 7
            if wd >= 5:
                cal += 7 - wd
            if pay1_cal < 0:
                if cal >= horizon or h >= width:
                    fail = TIMEOUT
                    fail_slot = slot
                    break
                if slot_max_days[slot] > 0 and cal - slot_start >= slot_max_days[slot]:
                    fail = TIMEOUT
                    fail_slot = slot
                    break
            elif cal >= ev_h or h >= width:
                break
            if inact > 0 and cal - last_trade > inact:
                if pay1_cal < 0:
                    fail = INACTIVITY
                    fail_slot = slot
                break

            d = idx[p, h]
            h += 1
            is_funded = slot == funded
            tgt = slot_target[slot]
            traded = n_opened[d] > 0

            # уровень общего лимита на начало дня (для R1)
            if frozen:
                lm0 = l_frozen
            elif mt == 0:
                lm0 = 1.0 - dist
            elif mt == 1:
                lm0 = min(h_eod - dist, cap)
            elif mt == 2:
                lm0 = min(h_bal - dist, cap)
            else:
                lm0 = min(m_eq - dist, cap)

            # риск дня: фаза → после цели → R1 → R2 (MONTE_CARLO §2.4)
            f = slot_f[slot]
            if target_hit:
                f = f_after
            else:
                if r1_x > 0 and (lm0 + dist - bal) / dist >= r1_x - _TOL:
                    f = f * r1_k
                if r2 and tgt > 0:
                    rem = 1.0 + tgt + buffer - bal
                    if rem < f * r_win:
                        f = max(f_after, rem / r_win)
            s = f * kappa[slot, d]

            # правила funded на открытый риск (Risk Rule FN, Striking FP)
            if is_funded and traded:
                if rr_max > 0 and s * risk_open[d] > rr_max + _TOL:
                    if pay1_cal < 0:
                        fail = RULE
                        fail_slot = slot
                    break
                if strike_thr > 0 and s * single_loss[d] > strike_thr + _TOL:
                    strikes += 1
                    if strikes >= strikes_n:
                        if pay1_cal < 0:
                            fail = RULE
                            fail_slot = slot
                        break

            b_sod = bal
            e_sod = bal + s * c_in[d]
            if has_daily:
                ref = b_sod if dl_ref == 0 else max(b_sod, e_sod)
                if dl_basis == 0:
                    amt = dl
                elif dl_basis == 1:
                    amt = dl * ref
                else:
                    amt = dl * min(1.0, ref)
                l_d = ref - amt
                a_d = amt
            else:
                ref = b_sod
                l_d = -_BIG
                a_d = dist
            g_d = ref - gamma_d * a_d
            if mt == 2 and not frozen:
                h_bal = max(h_bal, b_sod + s * b_max[d])
            if frozen:
                l_m = l_frozen
            elif mt == 0:
                l_m = 1.0 - dist
            elif mt == 1:
                l_m = min(h_eod - dist, cap)
            elif mt == 2:
                l_m = min(h_bal - dist, cap)
            else:
                l_m = min(m_eq - dist, cap)
            g_t = 1.0 - gamma_t * dist if mt == 0 else l_m + (1.0 - gamma_t) * dist
            t_lvl = 1.0 + tgt + buffer if (tgt > 0 and not target_hit) else _BIG

            # проход по точкам пути: лимит фирмы → предохранители → цель (§3.6)
            event = 0  # 1 провал, 2 дневной предохранитель, 3 цель
            b_new = 0.0
            ev_fail = OK
            for k in range(path_off[d], path_off[d + 1]):
                e = e_sod + s * path_val[k]
                e_lo = e - s * eps
                if mt == 3 and not frozen:
                    if e > m_eq:
                        m_eq = e
                    l_m = min(m_eq - dist, cap)
                    if lock_trig > 0 and m_eq >= 1.0 + lock_trig - _TOL:
                        frozen = True
                        l_frozen = l_m
                    g_t = l_m + (1.0 - gamma_t) * dist
                if e_lo < l_d:
                    event, ev_fail = 1, DAILY
                    break
                if e_lo < l_m:
                    event, ev_fail = 1, MAX
                    break
                if guards:
                    if e_lo <= g_t:
                        x = g_t - slip * s
                        ev_fail = MAX if x < l_m else (DAILY if x < l_d else GUARD)
                        event = 1
                        break
                    if e_lo <= g_d:
                        x = g_d - slip * s
                        if x < l_m:
                            event, ev_fail = 1, MAX
                        elif x < l_d:
                            event, ev_fail = 1, DAILY
                        elif x <= g_t:
                            event, ev_fail = 1, GUARD
                        else:
                            event, b_new = 2, x
                        break
                if e >= t_lvl:
                    event, b_new = 3, t_lvl
                    break
            if event == 1:
                if pay1_cal < 0:
                    fail = ev_fail
                    fail_slot = slot
                break
            bal = b_sod + s * db[d] if event == 0 else b_new
            if event == 3:
                target_hit = True

            # конец дня
            if traded:
                tdays += 1
                last_trade = cal
            h_eod = max(h_eod, bal)
            h_bal = max(h_bal, bal)
            m_eq = max(m_eq, bal)
            if lock_trig > 0 and not frozen:
                peak = h_eod if mt == 1 else (h_bal if mt == 2 else m_eq)
                if peak >= 1.0 + lock_trig - _TOL:
                    frozen = True
                    l_frozen = min(peak - dist, cap)

            if not is_funded:
                if target_hit:
                    if bal < 1.0 + tgt - _TOL:
                        target_hit = False
                    elif tdays >= slot_min_days[slot]:
                        out_pass_cal[p, slot] = cal
                        out_tdays[p, slot] = tdays
                        slot += 1
                        if slot > stop:
                            break
                        bal = 1.0
                        h_eod = 1.0
                        h_bal = 1.0
                        m_eq = 1.0
                        frozen = False
                        tdays = 0
                        target_hit = False
                        cal += 1 + pause
                        slot_start = cal
                        last_trade = cal
                        continue
            else:
                day_pnl = bal - b_sod
                if not f_started and traded:
                    f_started = True
                    if first_after > 0:
                        next_check = cal + first_after
                    elif cycle > 0:
                        next_check = cal + cycle
                    else:
                        next_check = cal
                    bcount = 0
                if f_started:
                    news_cum += s * news_pos[d]
                    if day_pnl > best_day:
                        best_day = day_pnl
                    if day_pnl > 0 and day_pnl >= pdays_min - _TOL:
                        pdays += 1
                    bcount += 1
                    eligible = False
                    if first_after > 0 or cycle > 0:
                        if cal >= next_check:
                            eligible = True
                            step = cycle if cycle > 0 else 1
                            next_check += step * ((cal - next_check) // step + 1)
                    elif cycle_b > 0:
                        if bcount >= cycle_b:
                            eligible = True
                            bcount = 0
                    else:
                        eligible = True
                    if eligible and has_payout:
                        gross = bal - 1.0
                        profit = gross - (1.0 - news_counted) * news_cum
                        can_pay = profit > 0.0
                        if profit < min_profit - _TOL:
                            can_pay = False
                        if consist > 0 and best_day > consist * gross + _TOL:
                            can_pay = False
                        if pdays_n > 0 and pdays < pdays_n:
                            can_pay = False
                        if can_pay:
                            pay_amt = split * profit
                            npay += 1
                            if pay1_cal < 0:
                                pay1_cal = cal
                                pay1_amt = pay_amt
                            if cal < ev_h:
                                pay_sum += pay_amt
                                if refund_k > 0 and npay == refund_k:
                                    refund = True
                            bal = 1.0
                            best_day = 0.0
                            pdays = 0
                            news_cum = 0.0
                            if floor_resets:
                                h_eod = 1.0
                                h_bal = 1.0
                                m_eq = 1.0
                                frozen = False
                            if ev_h <= 0 or cal >= ev_h:
                                break
            cal += 1

        out_fail[p] = fail
        out_fail_slot[p] = fail_slot
        out_end_cal[p] = cal
        out_pay1_cal[p] = pay1_cal
        out_pay1_amt[p] = pay1_amt
        out_pay_sum[p] = pay_sum
        out_npay[p] = npay
        out_refund[p] = refund
        out_bal[p] = bal


kernel = cast(Callable[..., None], numba.njit(cache=True, nogil=True)(_kernel))


@dataclass(frozen=True)
class RiskScheme:
    """Схема риска (MONTE_CARLO §2.4). `f` — риск на сделку по слотам (фазы…, funded), доли B0."""

    f: tuple[float, ...]
    gamma_d: float = 0.5
    gamma_t: float = 0.6
    guards: bool = True
    guard_slip_r: float = 0.1
    target_buffer: float = 0.001
    f_after_target: float = 0.001
    r1_x: float | None = None
    r1_k: float = 0.5
    r2: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.gamma_d <= 0.7:
            raise ValueError("γ_D должна быть в (0, 0.7] (инструкции §10)")
        if not 0 < self.gamma_t <= 1:
            raise ValueError("γ_T должна быть в (0, 1]")
        if any(x <= 0 for x in self.f):
            raise ValueError("риск на сделку должен быть > 0")

    def label(self) -> str:
        parts = " / ".join(f"{100 * x:g}" for x in self.f)
        extra = ""
        if self.r1_x is not None:
            extra += f", R1 {100 * self.r1_x:g}%"
        if self.r2:
            extra += ", R2"
        return f"{parts}%{extra}"


@dataclass(frozen=True)
class SimSettings:
    horizon_cal: int = 365  # P-13
    ev_horizon_cal: int = 180  # MONTE_CARLO §8
    pause_days: int = 2  # P-17


@dataclass(frozen=True)
class PathResults:
    fail: I8
    fail_slot: I8
    pass_cal: I32  # [путь, слот] — календарный день прохождения фазы, −1
    tdays: I32
    end_cal: I32
    pay1_cal: I32
    pay1_amt: F64  # доля B0, после сплита
    pay_sum: F64  # сумма выплат до горизонта EV, доля B0
    npay: I32
    refund: npt.NDArray[np.bool_]
    bal: F64  # баланс в конце пути (доля B0)
    start_slot: int
    stop_slot: int

    @property
    def n(self) -> int:
        return int(self.fail.shape[0])


def index_width(settings: SimSettings) -> int:
    """Сколько торговых дней нужно на путь: горизонт в днях Пн–Пт плюс запас."""
    return max(settings.horizon_cal, settings.ev_horizon_cal) * 5 // 7 + 8


def build_params(
    rules: RuleSet,
    risk: RiskScheme,
    table: DayTable,
    settings: SimSettings,
    *,
    start_slot: int,
    stop_slot: int,
) -> F64:
    prm = np.zeros(N_PARAMS)
    prm[P_START] = start_slot
    prm[P_STOP] = stop_slot
    prm[P_FUNDED] = rules.n_phases
    prm[P_HAS_DAILY] = 1.0 if rules.daily_pct is not None else 0.0
    prm[P_DL] = rules.daily_pct or 0.0
    prm[P_DL_REF] = rules.daily_ref
    prm[P_DL_BASIS] = rules.daily_basis
    prm[P_MAX_TYPE] = rules.max_type
    prm[P_D] = rules.max_dist
    prm[P_CAP] = min(rules.lock_cap, _BIG)
    prm[P_LOCK_TRIG] = rules.lock_trigger
    prm[P_FLOOR] = rules.floor_after_payout
    prm[P_GUARDS] = 1.0 if risk.guards else 0.0
    prm[P_GAMMA_D] = risk.gamma_d
    prm[P_GAMMA_T] = risk.gamma_t
    prm[P_SLIP] = risk.guard_slip_r
    prm[P_BUFFER] = risk.target_buffer
    prm[P_F_AFTER] = risk.f_after_target
    prm[P_R1_X] = risk.r1_x if risk.r1_x is not None else -1.0
    prm[P_R1_K] = risk.r1_k
    prm[P_R2] = 1.0 if risk.r2 else 0.0
    prm[P_R_WIN] = table.r_win_mean
    prm[P_INACT] = rules.inactivity_days
    prm[P_PAUSE] = settings.pause_days
    prm[P_HORIZON] = settings.horizon_cal
    prm[P_EV_H] = settings.ev_horizon_cal
    po = rules.payout
    prm[P_HAS_PAYOUT] = 1.0 if po is not None else 0.0
    if po is not None:
        prm[P_SPLIT] = po.split
        prm[P_FIRST_AFTER] = po.first_after_days
        prm[P_CYCLE] = po.cycle_days
        prm[P_CYCLE_B] = po.cycle_business_days
        prm[P_MIN_PROFIT] = po.min_profit
        prm[P_CONSIST] = po.consistency
        prm[P_PDAYS_N] = po.profitable_days_n
        prm[P_PDAYS_MIN] = po.profitable_days_min
        prm[P_NEWS] = po.news_counted
    prm[P_REFUND_K] = rules.refund_on_payout
    prm[P_RR] = rules.risk_rule_max
    prm[P_STRIKE] = rules.striking_thr
    prm[P_STRIKES_N] = rules.strikes_to_close
    prm[P_EPS] = table.eps
    return prm


def kappa_matrix(table: DayTable, rules: RuleSet, risk: RiskScheme) -> tuple[F64, float]:
    """κ по слотам и дням для номинального риска слота; вторым — худшая доля κ_i > 1.5."""
    rows = []
    worst = 0.0
    for f in risk.f:
        k, share = table.kappa(f * rules.size_usd)
        rows.append(k)
        worst = max(worst, share)
    return np.vstack(rows), worst


def simulate(
    table: DayTable,
    rules: RuleSet,
    risk: RiskScheme,
    idx: I32,
    settings: SimSettings | None = None,
    *,
    start_slot: int = 0,
    stop_slot: int | None = None,
    jit: bool = True,
) -> PathResults:
    """Прогон путей `idx` (индексы дней таблицы) по правилам `rules` при риске `risk`.

    `start_slot..stop_slot` — какие фазы гонять (для раздельного поиска риска, §10).
    По умолчанию — от первой фазы до первой выплаты (или до funded, если выплат нет).
    """
    settings = settings or SimSettings()
    if len(risk.f) != rules.n_slots:
        raise ValueError(
            f"нужно {rules.n_slots} значений риска (фазы + funded), дано {len(risk.f)}"
        )
    last = rules.n_phases if rules.payout is not None else rules.n_phases - 1
    stop = last if stop_slot is None else stop_slot
    if stop < start_slot:
        raise ValueError("нечего моделировать: нет фаз и нет выплат")
    n = idx.shape[0]
    kap, _ = kappa_matrix(table, rules, risk)
    prm = build_params(rules, risk, table, settings, start_slot=start_slot, stop_slot=stop)
    s = rules.n_slots
    out: dict[str, Any] = {
        "fail": np.zeros(n, np.int8),
        "fail_slot": np.zeros(n, np.int8),
        "pass_cal": np.full((n, s), -1, np.int32),
        "tdays": np.zeros((n, s), np.int32),
        "end_cal": np.zeros(n, np.int32),
        "pay1_cal": np.zeros(n, np.int32),
        "pay1_amt": np.zeros(n),
        "pay_sum": np.zeros(n),
        "npay": np.zeros(n, np.int32),
        "refund": np.zeros(n, np.bool_),
        "bal": np.zeros(n),
    }
    fn = kernel if jit else _kernel
    fn(
        np.ascontiguousarray(idx, dtype=np.int32),
        table.n_opened,
        table.db,
        table.b_max,
        table.c_in,
        table.path_off,
        table.path_val,
        table.risk_open_max,
        table.news_pos,
        table.single_loss,
        kap,
        np.asarray([*rules.targets, 0.0]),
        np.asarray([*rules.min_days, 0], dtype=np.int64),
        np.asarray([*rules.max_days, -1], dtype=np.int64),
        np.asarray(risk.f, dtype=np.float64),
        prm,
        out["fail"],
        out["fail_slot"],
        out["pass_cal"],
        out["tdays"],
        out["end_cal"],
        out["pay1_cal"],
        out["pay1_amt"],
        out["pay_sum"],
        out["npay"],
        out["refund"],
        out["bal"],
    )
    return PathResults(**out, start_slot=start_slot, stop_slot=stop)
