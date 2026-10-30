"""Отчёт автопоиска: `reports/search-<campaign>-<дата>/` и сводка до 30 строк для чата.

Файлы: `summary.md` (воронка, топ, оговорки), `leaderboard.csv` (все кандидаты событийного
бэктеста), `manifest.json` (воспроизводимость), `top-N-<ключ>/` — полный отчёт проп-MC для
лучших (таблица программ, стрессы, внешний бутстрэп).
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import polars as pl

from pbl.config import REPO_ROOT, load_firms
from pbl.prop.engine import MCConfig
from pbl.prop.report import run_report, write_report
from pbl.research.search import Candidate, SearchParams, SearchResult, Windows

COSTS_HEAVY = 0.30
"""Издержки > 30% валовой — флаг (инструкции §5, SPEC §6)."""
TOP_ROWS = 10


def _f(x: float, spec: str = ".2f") -> str:
    return "—" if x is None or math.isnan(x) else format(x, spec)


def _p(x: float) -> str:
    return "—" if math.isnan(x) else f"{100 * x:.1f}%"


def mark(v: bool | None) -> str:
    return "—" if v is None else ("✓" if v else "✗")


def gates_text(c: Candidate) -> str:
    return " ".join(f"{k} {mark(v)}" for k, v in c.gates.items())


def leaderboard(res: SearchResult) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for rank, c in enumerate(res.candidates, 1):
        u, mc = c.unit, c.mc
        rows.append(
            {
                "rank": rank,
                "family": u.cfg.family.family,
                "key": u.cfg.key,
                "group": u.group,
                "symbols": "+".join(u.symbols),
                "variant_id": u.variant_id,
                "n": int(u.stats["n"]),
                "pf": u.stats["pf"],
                "mean_r": u.stats["mean_r"],
                "e15": u.stats["e15"],
                "cost_share": u.stats["cost_share"],
                "costs_heavy": bool(u.stats["cost_share"] > COSTS_HEAVY),
                "sharpe_sel": c.sharpe,
                "dsr": c.dsr,
                "eval_sample": c.eval_sample,
                "eval_n": int(c.eval_stats.get("n", 0)),
                "eval_mean_r": c.eval_stats.get("mean_r", math.nan),
                "eval_pf": c.eval_stats.get("pf", math.nan),
                "eval_quick_strike": c.eval_stats.get("qs", math.nan),
                "parity_gap": c.parity_gap,
                "gates": gates_text(c),
                "mc_program": None if mc is None else mc.label,
                "p1": math.nan if mc is None else mc.p1,
                "p1_lo": math.nan if mc is None else mc.p1_ci[0],
                "p1_hi": math.nan if mc is None else mc.p1_ci[1],
                "p_funded": math.nan if mc is None else mc.p_funded,
                "ev1": math.nan if mc is None else mc.ev1,
                "days_funded_median": math.nan if mc is None else mc.days_funded_median,
                "risk": None if mc is None else mc.risk,
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def window_lines(w: Windows) -> list[str]:
    lines = [
        f"- Данные: {w.first} … {w.last}; holdout с {w.cut} поиск **не видел** "
        "(бары после отсечения не загружались в рынок)"
    ]
    if w.forward:
        lines.append(
            f"- Выбор вариантов: {w.first} … {w.sel_end} (S1, DSR/PBO, Sharpe); "
            f"оценка и MC: {w.sel_end} … {w.cut} (тесты WF без переоптимизации)"
        )
    else:
        lines.append(
            f"- Выбор и MC на одном и том же отрезке {w.first} … {w.cut}: "
            "**IN-SAMPLE — справка, не для решений** (VALIDATION §2)"
        )
    return lines


def verdict(res: SearchResult) -> str:
    n_ok = sum(1 for c in res.candidates if c.passed and c.mc is not None and c.mc.ev_ok)
    if res.funnel["прошли S1"] == 0:
        return "Ни один вариант не прошёл S1: edge в перебранном пространстве не найден."
    if not res.candidates:
        return "Выжившие S1 не прошли DSR/PBO: на фоне числа проб отличить их от шума нельзя."
    if n_ok == 0:
        return "Кандидатов с EV₁ > 0 после всех гейтов поиска нет."
    tag = "" if res.windows.forward else " (IN-SAMPLE, справка)"
    return f"{n_ok} кандидат(ов) на S2: гейты поиска пройдены, EV₁ > 0{tag}."


def _eval_cell(c: Candidate) -> str:
    st = c.eval_stats
    mean_r, pf = st.get("mean_r", math.nan), st.get("pf", math.nan)
    return f"{int(st.get('n', 0))} / {_f(mean_r, '+.3f')} / {_f(pf)}"


def top_table(res: SearchResult, limit: int = TOP_ROWS) -> list[str]:
    head = (
        "| # | Конфигурация | Корзина | Сделок | PF | E при ×1.5 | Издержки | Sharpe/д | DSR "
        "| Оценка: n / mean R / PF | P(1-я выплата) [CI] | EV₁ | Дни до funded | Риск | Гейты |"
    )
    out = [head, "|" + "---|" * 15]
    for i, c in enumerate(res.candidates[:limit], 1):
        u, mc = c.unit, c.mc
        flag = " ⚠COSTS_HEAVY" if u.stats["cost_share"] > COSTS_HEAVY else ""
        p1 = "—" if mc is None else f"**{_p(mc.p1)}** [{_p(mc.p1_ci[0])}–{_p(mc.p1_ci[1])}]"
        out.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    f"`{u.cfg.key}`",
                    u.group,
                    str(int(u.stats["n"])),
                    _f(u.stats["pf"]),
                    _f(u.stats["e15"], "+.3f"),
                    _p(u.stats["cost_share"]) + flag,
                    _f(c.sharpe, ".3f"),
                    _f(c.dsr, ".2f"),
                    _eval_cell(c),
                    p1,
                    "—" if mc is None else f"{mc.ev1:+,.0f} $".replace(",", " "),
                    "—" if mc is None else _f(mc.days_funded_median, ".0f"),
                    "—" if mc is None else mc.risk,
                    gates_text(c),
                ]
            )
            + " |"
        )
    return out


def summary_markdown(res: SearchResult) -> str:
    p, w, d = res.params, res.windows, res.deflation
    lines = [
        f"# Автопоиск {p.campaign}: {p.on}",
        "",
        f"Вердикт: {verdict(res)}",
        "",
        "## Данные и режим",
        "",
        f"- Режим `{p.mode}`, генератор {p.version}, seed {p.seed}, путей MC {p.n_paths}",
        *window_lines(w),
        "- Корзины: " + "; ".join(f"{g}: {'+'.join(s)}" for g, s in res.groups.items()),
        "- Программы MC: "
        + ", ".join(p.programs)
        + f" (размеры ${', '.join(str(s) for s in p.sizes_usd)}); сделки оценки помечены "
        f"{res.candidates[0].eval_sample if res.candidates else '—'}",
        "",
        "## Воронка",
        "",
        "| Этап | Число |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in res.funnel.items()],
        "",
        f"Время: {res.elapsed:.0f} с"
        + (" — **прервано по бюджету времени**" if res.truncated else ""),
        f"DSR: N = {d.n_trials} проб, SR₀ = {_f(d.sr0, '.4f')}; "
        + (
            f"PBO среди выживших S1 = {_p(d.pbo.pbo)} ({d.pbo.n_variants} вариантов, "
            f"{d.pbo.n_blocks} блоков)"
            if d.pbo is not None
            else "PBO не посчитан (мало выживших или дней)"
        ),
        "",
        "## Семейства",
        "",
        "| Семейство | Единиц | Прошли S1 |",
        "|---|---|---|",
        *[f"| {k} | {v['units']} | {v['s1']} |" for k, v in sorted(res.by_family.items())],
        "",
        f"## Топ-{TOP_ROWS} по P(1-я выплата)",
        "",
        *(top_table(res) if res.candidates else ["Кандидатов нет."]),
        "",
        "## Оговорки",
        "",
        "- Экран «OOS E>0» в поиске предварительный: он отбирает из сотен вариантов и не "
        "заменяет holdout. Holdout открывают гейты S2 для финалистов, один раз.",
        "- Каждая проба (конфигурация × инструмент и × корзина) записана в счётчик "
        f"`research/registry/variants/{p.campaign}.parquet`, в том числе провалившие S1.",
        "- Один вход в день на инструмент, закрытие к концу сессии; событийный бэктест "
        "сверен со сканом (колонка `parity_gap` в `leaderboard.csv`).",
    ]
    if p.explore:
        lines.append(
            "- **Разведка (`--explore`)**: гейты только помечают кандидатов, не отсеивают."
        )
    return "\n".join(lines) + "\n"


def chat_summary(res: SearchResult, out: Path | None = None) -> str:
    lines = [
        f"Автопоиск {res.params.campaign}: {verdict(res)}",
        f"режим {res.params.mode}, {res.windows.first} … {res.windows.cut} (holdout не открывался)",
        *[f"  {k}: {v}" for k, v in res.funnel.items()],
    ]
    for i, c in enumerate(res.candidates[:5], 1):
        mc = c.mc
        tail = "MC нет" if mc is None else f"P1 {_p(mc.p1)}, EV₁ {mc.ev1:+.0f} $"
        lines.append(f"  #{i} {c.unit.key} — {tail}; {gates_text(c)}")
    if res.truncated:
        lines.append("  ⚠ прервано по бюджету времени")
    if out is not None:
        lines.append(f"отчёт: {out}")
    return "\n".join(lines[:30])


def git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return r.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def data_fingerprint(bars: dict[str, pl.DataFrame]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for sym, b in sorted(bars.items()):
        first, last = str(b["ts"].min()), str(b["ts"].max())
        key = f"{sym}:{b.height}:{first}:{last}"
        out[sym] = {
            "rows": b.height,
            "first": first,
            "last": last,
            "hash": hashlib.sha256(key.encode()).hexdigest()[:16],
        }
    return out


def manifest(res: SearchResult, bars: dict[str, pl.DataFrame]) -> dict[str, Any]:
    p: SearchParams = res.params
    return {
        "campaign": p.campaign,
        "generator": p.version,
        "git_sha": git_sha(),
        "mode": p.mode,
        "seed": p.seed,
        "n_paths": p.n_paths,
        "on": str(p.on),
        "windows": {
            "first": str(res.windows.first),
            "sel_end": str(res.windows.sel_end),
            "cut": str(res.windows.cut),
            "last": str(res.windows.last),
        },
        "budget": {
            "max_scans": p.budget.max_scans,
            "max_event_runs": p.budget.max_event_runs,
            "mc_top": p.budget.mc_top,
            "max_seconds": p.budget.max_seconds,
        },
        "gates": {
            "s1": {
                "min_trades": p.s1.min_trades,
                "min_pf": p.s1.min_pf,
                "cost_mult": p.s1.cost_mult,
            },
            "min_dsr": p.min_dsr,
            "max_pbo": p.max_pbo,
            "min_eval_trades": p.min_eval_trades,
        },
        "explore": p.explore,
        "funnel": res.funnel,
        "truncated": res.truncated,
        "data": data_fingerprint(bars),
    }


def write_search_report(
    res: SearchResult, out_dir: Path, bars: dict[str, pl.DataFrame], *, full_top: int = 3
) -> Path:
    """Пишет отчёт; `full_top` лучших кандидатов с MC получают полный отчёт проп-MC."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(summary_markdown(res), encoding="utf-8")
    board = leaderboard(res)
    if not board.is_empty():
        board.write_csv(out_dir / "leaderboard.csv")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest(res, bars), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    p = res.params
    firm = next(f for f in load_firms() if f.id == p.firm_id)
    done = 0
    for i, c in enumerate(res.candidates, 1):
        if done >= full_top or c.mc is None:
            continue
        cfg = p.mc or MCConfig(n_paths=p.n_paths, seed=p.seed)
        rep = run_report(
            c.eval_trades, [firm], [], cfg, program_ids=list(p.programs), top_k=1, detail_keys=()
        )
        write_report(rep, out_dir / f"top-{i}")
        done += 1
    return out_dir / "summary.md"
