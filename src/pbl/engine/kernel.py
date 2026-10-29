"""Ядро движка на numba: исполнение заявок и сопровождение позиции по M1-барам (ARCHITECTURE §9).

Одна функция `_run` на чистом Python в стиле numba; `kernel` — её numba-версия (как в
`pbl.prop.sim`). Цены — mid; bid/ask бара = `mid ∓ spread/2`.

Правила, которые нельзя восстановить по бару, — в худшую сторону (D-013):
- стоп и тейк (или частичный выход) в одном баре — стоп;
- в баре входа по stop/limit-заявке весь диапазон бара считается пройденным после входа;
- гэп за стоп или тейк — исполнение по цене открытия бара;
- безубыток и трейлинг считаются по закрытому бару и действуют со следующего бара
  (не чаще одного переноса стопа за бар — троттлинг сообщений, SPEC B5).

Шаг 1 — исполнение каждой заявки независимо от позиций. Шаг 2 — по времени исполнения:
OCO, лимит одновременных позиций, лимит сделок в день, сопровождение до выхода.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numba
import numpy as np
import numpy.typing as npt

from pbl.strategies.base import LIMIT, MARKET, STOP

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
I8 = npt.NDArray[np.int8]
B8 = npt.NDArray[np.bool_]

# статус заявки
ACCEPTED, NO_FILL, BUSY, DAY_CAP, OCO_CANCELLED, BAD_STOP = range(6)
STATUSES = ("accepted", "no_fill", "busy", "day_cap", "oco_cancelled", "bad_stop")

# причина выхода — индексы в порядке `pbl.schemas.trade.ExitReason`
X_SL, X_TP, X_TRAIL, X_BE, X_PARTIAL, X_TIME, X_GUARD, X_FRIDAY, X_EOD = range(9)
EXIT_REASONS = ("sl", "tp", "trail", "be", "partial", "time", "guard", "friday_close", "eod")

# параметры выходов
(
    E_BE_AT,
    E_BE_OFF,
    E_TR_START,
    E_TR_DIST,
    E_TR_STEP,
    E_PART_R,
    E_PART_Q,
    E_MAX_HOLD,
    E_CLOSE_MIN,
    E_MAX_POS,
    E_MAX_DAY,
    N_EXIT_PARAMS,
) = range(12)

# колонки выхода
I_STATUS, I_ENTRY, I_EXIT, I_REASON, I_MODS, I_PART, N_OUT_I = range(7)
(
    F_ENTRY_MID,
    F_ENTRY,
    F_EXIT,
    F_SL0,
    F_TP0,
    F_RISK,
    F_GROSS,
    F_SPREAD,
    F_SLIP,
    F_MAE,
    F_MFE,
    F_PART_FILL,
    F_SL_LAST,
    N_OUT_F,
) = range(14)


def _run(
    ts: I64,
    mo: F64,
    mh: F64,
    ml: F64,
    mc: F64,
    spr: F64,
    slp: F64,
    day: I64,
    minute: I64,
    entry_ok: B8,
    force_close: B8,
    s_bar: I64,
    s_side: I8,
    s_order: I8,
    s_level: F64,
    s_expire: I64,
    s_sl: F64,
    s_sldist: F64,
    s_tp: F64,
    s_tpr: F64,
    s_oco: I64,
    prm: F64,
    out_i: I64,
    out_f: F64,
) -> None:
    n = ts.shape[0]
    m = s_bar.shape[0]
    close_min = int(prm[E_CLOSE_MIN])
    max_hold = int(prm[E_MAX_HOLD])
    max_pos = int(prm[E_MAX_POS])
    max_day = int(prm[E_MAX_DAY])
    be_at = prm[E_BE_AT]
    be_off = prm[E_BE_OFF]
    tr_start = prm[E_TR_START]
    tr_dist = prm[E_TR_DIST]
    tr_step = prm[E_TR_STEP]
    part_r = prm[E_PART_R]
    part_q = prm[E_PART_Q] if part_r > 0.0 else 0.0

    # --- шаг 1: исполнение заявок -------------------------------------------------------
    fill = np.full(m, n, dtype=np.int64)  # n — не исполнена (в конце сортировки)
    emid = np.zeros(m)
    eslip = np.zeros(m)
    for k in range(m):
        out_i[k, I_STATUS] = NO_FILL
        out_i[k, I_ENTRY] = -1
        out_i[k, I_EXIT] = -1
        out_i[k, I_REASON] = -1
        out_i[k, I_MODS] = 0
        out_i[k, I_PART] = -1
        for c in range(out_f.shape[1]):
            out_f[k, c] = np.nan
        d = float(s_side[k])
        j0 = s_bar[k] + 1
        if j0 >= n or j0 < 1:
            continue
        if s_order[k] == MARKET:
            if entry_ok[j0] and not (close_min >= 0 and minute[j0] >= close_min):
                fill[k] = j0
                emid[k] = mo[j0]
                eslip[k] = 1.0
            continue
        for j in range(j0, n):
            if ts[j] >= s_expire[k]:
                break
            if not entry_ok[j] or (close_min >= 0 and minute[j] >= close_min):
                continue
            trig = s_level[k] - d * spr[j] / 2.0  # уровень заявки в mid
            if s_order[k] == STOP:
                fav = mh[j] if d > 0 else ml[j]
                if d * (mo[j] - trig) >= 0.0:
                    fill[k], emid[k], eslip[k] = j, mo[j], 1.0
                    break
                if d * (fav - trig) >= 0.0:
                    fill[k], emid[k], eslip[k] = j, trig, 1.0
                    break
            elif s_order[k] == LIMIT:
                adv = ml[j] if d > 0 else mh[j]
                if d * (mo[j] - trig) <= 0.0:
                    fill[k], emid[k], eslip[k] = j, mo[j], 0.0
                    break
                if d * (adv - trig) <= 0.0:
                    fill[k], emid[k], eslip[k] = j, trig, 0.0
                    break

    # --- шаг 2: позиции по времени исполнения ---------------------------------------------
    order = np.argsort(fill, kind="mergesort")
    max_oco = 0
    for k in range(m):
        max_oco = max(max_oco, s_oco[k] + 1)
    oco_used = np.zeros(max_oco + 1, dtype=np.bool_)
    active = np.zeros(m + 1, dtype=np.int64)  # бары выхода открытых позиций
    n_act = 0
    cur_day = -1
    cnt_day = 0
    for idx in range(m):
        k = order[idx]
        j = fill[k]
        if j >= n:
            break
        if s_oco[k] >= 0 and oco_used[s_oco[k]]:
            out_i[k, I_STATUS] = OCO_CANCELLED
            continue
        keep = 0
        for a in range(n_act):
            if active[a] >= j:  # бар выхода = бар входа: порядок неизвестен → занято
                active[keep] = active[a]
                keep += 1
        n_act = keep
        if n_act >= max_pos:
            out_i[k, I_STATUS] = BUSY
            continue
        if day[j] != cur_day:
            cur_day = day[j]
            cnt_day = 0
        if cnt_day >= max_day:
            out_i[k, I_STATUS] = DAY_CAP
            continue
        d = float(s_side[k])
        s0 = spr[j]
        entry = emid[k] + d * (s0 / 2.0 + eslip[k] * slp[j])
        sl = s_sl[k] if not np.isnan(s_sl[k]) else entry - d * s_sldist[k]
        risk = d * (entry - sl)
        tp = s_tp[k]
        if np.isnan(tp) and not np.isnan(s_tpr[k]):
            tp = entry + d * s_tpr[k] * risk
        if not risk > 0.0 or (not np.isnan(tp) and d * (tp - entry) <= 0.0):
            out_i[k, I_STATUS] = BAD_STOP
            continue
        out_i[k, I_STATUS] = ACCEPTED
        cnt_day += 1
        if s_oco[k] >= 0:
            oco_used[s_oco[k]] = True
        em = emid[k]
        p_level = entry + d * part_r * risk if part_q > 0.0 else np.nan

        q_left = 1.0
        gross = 0.0  # реализованный результат в цене (mid), с весом доли
        spread_c = s0 / 2.0
        slip_c = eslip[k] * slp[j]
        sl_cur = sl
        moved = 0  # 1 — безубыток, 2 — трейлинг
        mods = 0
        mae = 0.0
        mfe = 0.0
        best = -np.inf  # лучший bid лонга / ask шорта, со знаком d
        part_bar = -1
        part_fill = np.nan
        x_bar = -1
        reason = -1
        x_fill = np.nan
        b = j
        while b < n:
            sb = spr[b]
            h = d * sb / 2.0
            sl_mid = sl_cur + h
            tp_mid = tp + h
            x_mid = np.nan
            x_slip = 0.0
            if b > j:
                o = mo[b]
                if d * (o - sl_mid) <= 0.0:
                    reason = X_TRAIL if moved == 2 else (X_BE if moved == 1 else X_SL)
                    x_mid, x_slip = o, 1.0
                elif not np.isnan(tp) and d * (o - tp_mid) >= 0.0:
                    reason = X_TP
                    x_mid = o
                elif force_close[b]:
                    reason = X_FRIDAY
                    x_mid, x_slip = o, 1.0
                elif close_min >= 0 and (day[b] != day[j] or minute[b] >= close_min):
                    reason = X_EOD
                    x_mid, x_slip = o, 1.0
                elif max_hold > 0 and ts[b] - ts[j] >= max_hold:
                    reason = X_TIME
                    x_mid, x_slip = o, 1.0
                if reason >= 0:
                    gross += q_left * d * (x_mid - em)
                    spread_c += q_left * sb / 2.0
                    slip_c += q_left * x_slip * slp[b]
                    x_fill = x_mid - d * (sb / 2.0 + x_slip * slp[b])
                    mae = min(mae, gross / risk)
                    mfe = max(mfe, gross / risk)
                    x_bar = b
                    break
                if part_bar < 0 and part_q > 0.0 and d * (o - (p_level + h)) >= 0.0:
                    gross += part_q * d * (o - em)
                    spread_c += part_q * sb / 2.0
                    q_left -= part_q
                    part_bar, part_fill = b, o - h
                    mods += 1
            adv = ml[b] if d > 0 else mh[b]
            fav = mh[b] if d > 0 else ml[b]
            if d * (adv - sl_mid) <= 0.0:
                reason = X_TRAIL if moved == 2 else (X_BE if moved == 1 else X_SL)
                gross += q_left * d * (sl_mid - em)
                spread_c += q_left * sb / 2.0
                slip_c += q_left * slp[b]
                x_fill = sl_cur - d * slp[b]
                mae = min(mae, gross / risk)
                x_bar = b
                break
            mae = min(mae, (gross + q_left * d * (adv - em)) / risk)
            if part_bar < 0 and part_q > 0.0 and d * (fav - (p_level + h)) >= 0.0:
                gross += part_q * d * (p_level + h - em)
                spread_c += part_q * sb / 2.0
                q_left -= part_q
                part_bar, part_fill = b, p_level
                mods += 1
            if not np.isnan(tp) and d * (fav - tp_mid) >= 0.0:
                reason = X_TP
                gross += q_left * d * (tp_mid - em)
                spread_c += q_left * sb / 2.0
                x_fill = tp
                mfe = max(mfe, gross / risk)
                x_bar = b
                break
            mfe = max(mfe, (gross + q_left * d * (fav - em)) / risk)
            # сопровождение по закрытому бару, действует со следующего
            best = max(best, d * fav - sb / 2.0)
            best_r = (best - d * entry) / risk
            if be_at > 0.0 and best_r >= be_at:
                new = entry + d * be_off * risk
                if d * (new - sl_cur) > 1e-12 * risk:
                    sl_cur = new
                    mods += 1
                    moved = max(moved, 1)
            if tr_dist > 0.0 and best_r >= tr_start:
                cand = d * best - d * tr_dist * risk
                if d * (cand - sl_cur) >= max(tr_step * risk, 1e-12 * risk):
                    sl_cur = cand
                    mods += 1
                    moved = 2
            b += 1
        if x_bar < 0:  # данные кончились — закрываем по последней цене
            b = n - 1
            sb = spr[b]
            reason = X_TIME
            gross += q_left * d * (mc[b] - em)
            spread_c += q_left * sb / 2.0
            slip_c += q_left * slp[b]
            x_fill = mc[b] - d * (sb / 2.0 + slp[b])
            mae = min(mae, gross / risk)
            mfe = max(mfe, gross / risk)
            x_bar = b
        active[n_act] = x_bar
        n_act += 1

        out_i[k, I_ENTRY] = j
        out_i[k, I_EXIT] = x_bar
        out_i[k, I_REASON] = reason
        out_i[k, I_MODS] = mods
        out_i[k, I_PART] = part_bar
        out_f[k, F_ENTRY_MID] = em
        out_f[k, F_ENTRY] = entry
        out_f[k, F_EXIT] = x_fill
        out_f[k, F_SL0] = sl
        out_f[k, F_TP0] = tp
        out_f[k, F_RISK] = risk
        out_f[k, F_GROSS] = gross
        out_f[k, F_SPREAD] = spread_c
        out_f[k, F_SLIP] = slip_c
        out_f[k, F_MAE] = min(mae, 0.0, gross / risk)
        out_f[k, F_MFE] = max(mfe, 0.0, gross / risk)
        out_f[k, F_PART_FILL] = part_fill
        out_f[k, F_SL_LAST] = sl_cur


kernel = cast(Callable[..., None], numba.njit(cache=True, nogil=True)(_run))
