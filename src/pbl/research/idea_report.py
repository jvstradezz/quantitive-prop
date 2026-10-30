"""Отчёт прогона идеи №1: `reports/<run_id>/` и сводка по IDEAS §4 (до 30 строк для чата)."""

from __future__ import annotations

import datetime as dt
import json
import math
import statistics
from pathlib import Path
from typing import Any

import polars as pl

from pbl.prop.io import write_trades_csv
from pbl.prop.report import costs_section, program_table, write_report
from pbl.research.idea_eval import FilterCheck
from pbl.research.idea_run import (
    GATES,
    HOLDOUT_GATES,
    Candidate,
    HoldoutStage,
    ISStage,
    RunParams,
    verdict,
)
from pbl.research.registry import advance
from pbl.schemas import IdeaCard
from pbl.schemas.idea import IdeaStatus

TOP_PROGRAMS = 6


def mark(v: bool | None) -> str:
    return "—" if v is None else ("✓" if v else "✗")


def _f(x: float, spec: str = ".2f") -> str:
    return "—" if math.isnan(x) else format(x, spec)


def _check(fc: FilterCheck | None) -> str:
    if fc is None or fc.diff is None:
        return "нет данных"
    d = fc.diff
    ci = "выше нуля" if d.holds else "включает ноль"
    return (
        f"разность mean R {d.diff:+.3f} (разрешённые {d.mean_a:+.3f}, "
        f"запрещённые {d.mean_b:+.3f}), 90% интервал {ci} [{d.low:+.3f}; {d.high:+.3f}]; "
        f"mean R × √n: с фильтром {fc.score_author:.2f}, без {fc.score_none:.2f} — "
        f"{'держится' if fc.ok else 'не держится'}"
    )


def filter_answer(stage: ISStage, ho: HoldoutStage | None) -> str:
    """Ответ на главный вопрос идеи: вердикт — только по OOS (VALIDATION §4.5)."""
    if ho is not None:
        key = stage.finalist.cfg.key if stage.finalist else "—"
        return f"OOS, тесты WF + holdout, параметры финалиста `{key}`: {_check(ho.filt)}"
    pre = next((c for c in stage.candidates if c.filt is not None), None)
    if pre is None:
        return "на OOS не проверялся: нет тестовых окон WF или ни один вариант не прошёл S1"
    return (
        f"предварительно, только тесты WF (holdout не открыт, вердикта нет), параметры "
        f"`{pre.cfg.key}`: {_check(pre.filt)}"
    )


def candidates_frame(stage: ISStage) -> pl.DataFrame:
    rows = []
    for rank, c in enumerate(stage.candidates, 1):
        rows.append(
            {
                "rank": rank,
                "key": c.cfg.key,
                "n": int(c.row["n"]),
                "pf": c.row["pf"],
                "mean_r": c.row["mean_r"],
                "e15": c.row["e15"],
                "cost_share": c.row["cost_share"],
                "sharpe_daily": c.row["sharpe_daily"],
                "dsr": c.dsr,
                "sr0": c.sr0,
                **{g: c.gates[g] for g in GATES},
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def candidates_table(stage: ISStage) -> list[str]:
    head = "| # | Вариант | Сделок | PF | mean R | E при ×1.5 | Доля изд. | DSR | " + " | ".join(
        GATES
    )
    out = [head + " |", "|" + "---|" * (8 + len(GATES))]
    for rank, c in enumerate(stage.candidates, 1):
        flag = " COSTS_HEAVY" if c.row["cost_share"] > 0.3 else ""
        out.append(
            f"| {rank} | `{c.cfg.key}` | {int(c.row['n'])} | {_f(c.row['pf'])} | "
            f"{c.row['mean_r']:+.3f} | {c.row['e15']:+.3f} | "
            f"{_f(c.row['cost_share'], '.0%')}{flag} | {_f(c.dsr, '.3f')} | "
            + " | ".join(mark(c.gates[g]) for g in GATES)
            + " |"
        )
    return out


def filter_section(stage: ISStage, ho: HoldoutStage | None) -> list[str]:
    fg = stage.filter_grid
    out = [
        "## 2. Фильтр дня недели (главный вопрос идеи)",
        "",
        f"**Ответ:** {filter_answer(stage, ho)}",
        "",
    ]
    out += [
        "Вердикт «держится» выносится только по OOS (тесты WF и holdout), IS служит для отбора "
        "вариантов (D-032, VALIDATION §4.5).",
        "",
    ]
    if fg.height:
        out.append(
            f"Сетка ({fg.height} сканов без фильтра), разрешённые корзины лучше запрещённых:"
        )
        for label, name in (("is", "IS (для отбора)"), ("oos", "OOS, тесты WF")):
            col = f"allowed_better_{label}"
            if col in fg.columns:
                med_a = statistics.median(fg[f"mean_r_allowed_{label}"].drop_nulls().to_list())
                med_f = statistics.median(fg[f"mean_r_forbidden_{label}"].drop_nulls().to_list())
                out.append(
                    f"- {name}: в {int(fg[col].sum())} из {fg.height} сканов; медиана mean R "
                    f"{med_a:+.3f} против {med_f:+.3f}."
                )
        out.append("")
    for c in stage.candidates:
        if c.cfg.filt == "author":
            out.append(f"- `{c.cfg.key}` на OOS (тесты WF): {_check(c.filt)}.")
    out.append("")
    return out


def holdout_section(ho: HoldoutStage) -> list[str]:
    r = ho.result
    out = [
        "## 3. Holdout (открыт один раз)",
        "",
        f"Открыт {r.opened_at}; сделок {r.n_trades}; расхождение быстрого скана и событийного "
        f"бэктеста по сумме R: {ho.scan_vs_backtest:.1%}.",
        "",
        "| Метрика | Holdout | 5% | 95% | В полосе |",
        "|---|---|---|---|---|",
    ]
    for k, b in r.band.bands.items():
        mand = " (обязательная)" if k in ("mean_r", "max_dd") else ""
        out.append(
            f"| {k}{mand} | {_f(b.observed, '.3f')} | {_f(b.low, '.3f')} | "
            f"{_f(b.high, '.3f')} | {mark(b.inside)} |"
        )
    out += ["", "| Инструмент | Сделок | mean R |", "|---|---|---|"]
    out += [f"| {s} | {n} | {m:+.3f} |" for s, (n, m) in sorted(r.per_instrument.items())]
    out += [
        "",
        "Гейты holdout: " + ", ".join(f"{g} {mark(ho.gates[g])}" for g in HOLDOUT_GATES),
        "",
    ]
    return out


def costs_fallback(c: Candidate) -> list[str]:
    s = c.row
    share = s["cost_share"]
    flag = " **COSTS_HEAVY**" if share > 0.3 else ""
    return [
        "## 4. Издержки",
        "",
        f"Проп-MC не считали, поэтому $ и % счёта не определены. Лучший вариант `{c.cfg.key}` "
        f"на IS: {int(s['n'])} сделок, mean R {s['mean_r']:+.3f}, при издержках ×1.5 "
        f"{s['e15']:+.3f}, доля издержек от валовой {_f(share, '.0%')}{flag}.",
        "",
    ]


def summary_markdown(stage: ISStage, ho: HoldoutStage | None, p: RunParams, final: str) -> str:
    sp = stage.splits
    n_wf = len(stage.picks)
    lines = [
        f"# IDEA-0001 · {p.run_id}",
        "",
        f"**Вердикт:** {final}",
        "",
        f"Данные: {sp.first} … {sp.last}; IS до {sp.cut} (не включая), holdout с {sp.cut}. "
        f"Инструменты: {', '.join(sorted({s for t in stage.tables.values() for s in t}))}.",
        f"Проб в счётчике (N для DSR): {stage.n_trials}; новых в этом прогоне: {stage.new_rows}. "
        f"WFE {_f(stage.wfe, '.2f')} по {n_wf} окнам walk-forward"
        + (" (меньше 4, P-01 не выполнено)" if n_wf < 4 else "")
        + f"; PBO {_f(stage.pbo.pbo, '.2f') if stage.pbo else '—'} по корзинам сетки.",
        "",
        "## 1. Лучшие варианты (до 5, отбор по Sharpe на IS среди прошедших S1)",
        "",
    ]
    lines += candidates_table(stage) if stage.candidates else ["Ни один вариант не прошёл S1."]
    lines += [""]
    lines += filter_section(stage, ho)
    if ho is not None:
        lines += holdout_section(ho)
    if ho is not None and ho.prop is not None:
        lines += ["## 4. Издержки стратегии (OOS: WF-тесты и holdout)", *costs_section(ho.prop)[1:]]
        lines += ["", "## 5. Проп-MC (S3)", ""]
        ranked = sorted(ho.prop.rows, key=lambda r: r.stats.objective, reverse=True)
        lines += program_table(ranked[:TOP_PROGRAMS])
        lines += ["", f"Полный отчёт: `prop/summary.md`. Гейт S3: {mark(ho.s3_ok)}.", ""]
    elif stage.candidates:
        lines += costs_fallback(stage.candidates[0])
    lines += [
        "## 6. Что не проверено и допущения",
        "",
        "- Вариант «снять ордера через 2 часа после окна» и таймфрейм M5 не тестировались: "
        "ордера живут до закрытия NY (23:00 сервера), сигналы на M1.",
        "- GER30 торгуется у FN с 10:15 сервера: окна Азии и Лондона для него не торгуемы, "
        "в корзину не входят.",
        "- Допущения D-029…D-032 в `docs/DECISIONS.md` приняты 2026-10-29.",
        "",
    ]
    return "\n".join(lines)


def chat_summary(stage: ISStage, ho: HoldoutStage | None, p: RunParams, final: str) -> str:
    """Сводка до 30 строк для чата (CLAUDE.md)."""
    sp = stage.splits
    lines = [
        f"IDEA-0001 {p.run_id}: {final}",
        f"Данные {sp.first}…{sp.last}, holdout с {sp.cut}; проб {stage.n_trials}, "
        f"WFE {_f(stage.wfe, '.2f')}, PBO {_f(stage.pbo.pbo, '.2f') if stage.pbo else '—'}.",
    ]
    for rank, c in enumerate(stage.candidates, 1):
        lines.append(
            f"{rank}. {c.cfg.key}: n {int(c.row['n'])}, PF {_f(c.row['pf'])}, "
            f"mean R {c.row['mean_r']:+.3f}, DSR {_f(c.dsr, '.2f')}; "
            + " ".join(f"{g} {mark(c.gates[g])}" for g in GATES)
        )
    lines.append(f"Фильтр дня, {filter_answer(stage, ho)}")
    if ho is not None:
        lines.append(
            "Holdout: "
            + ", ".join(f"{g} {mark(ho.gates[g])}" for g in HOLDOUT_GATES)
            + f"; S3 {mark(ho.s3_ok)}."
        )
        if ho.prop is not None and ho.prop.recommended is not None:
            st = ho.prop.recommended.stats
            lines.append(
                f"Лучшая программа: {ho.prop.recommended.rules.label}, "
                f"P(1-я выплата) {100 * st.p1:.1f}%, EV₁ {st.ev1:+.0f} $."
            )
    return "\n".join(lines[:30])


def write_idea_report(stage: ISStage, ho: HoldoutStage | None, p: RunParams, out_dir: Path) -> Path:
    """Файлы прогона в `out_dir`; возвращает путь к summary.md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    final = verdict(stage, ho)
    stage.s1.write_csv(out_dir / "s1_grid.csv")
    cand = candidates_frame(stage)
    if cand.height:
        cand.write_csv(out_dir / "candidates.csv")
    stage.filter_grid.write_csv(out_dir / "filter_grid.csv")
    rows: list[dict[str, Any]] = []
    for c in stage.candidates:
        rows += [{"key": c.cfg.key, "check": s.label, "n": s.n, "pf": s.pf} for s in c.sens]
    if rows:
        pl.DataFrame(rows).write_csv(out_dir / "sensitivity.csv")
    pl.DataFrame(
        [
            {
                "train_start": pk.window.train_start,
                "test_start": pk.window.test_start,
                "test_end": pk.window.test_end,
                "key": pk.key,
                "n_train": pk.n_train,
                "train_r": pk.train_r,
                "test_r": pk.test_r,
                "n_test": pk.n_test,
            }
            for pk in stage.picks
        ]
    ).write_csv(out_dir / "wf_picks.csv")
    if ho is not None:
        write_trades_csv(ho.trades, out_dir / "trades_oos.csv")
        if ho.prop is not None:
            write_report(ho.prop, out_dir / "prop")
    manifest = {
        "run_id": p.run_id,
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "params": {
            "start": str(p.start),
            "end": str(p.end),
            "version": p.version,
            "n_paths": p.n_paths,
            "seed": p.seed,
            "wf": p.wf,
            "is_fraction": p.is_fraction,
            "top": p.top,
        },
        "splits": {k: str(v) for k, v in vars(stage.splits).items()},
        "n_trials": stage.n_trials,
        "wfe": stage.wfe,
        "pbo": stage.pbo.pbo if stage.pbo else None,
        "verdict": final,
        "holdout_opened_at": str(ho.result.opened_at) if ho else None,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "chat.md").write_text(chat_summary(stage, ho, p, final) + "\n", encoding="utf-8")
    path = out_dir / "summary.md"
    path.write_text(summary_markdown(stage, ho, p, final), encoding="utf-8")
    return path


def card_after_run(
    card: IdeaCard, stage: ISStage, ho: HoldoutStage | None, p: RunParams, final: str
) -> IdeaCard:
    """Карточка после прогона: статус по вердикту, счётчик проб, дата открытия holdout."""
    if card.status == "Формализована":
        card = advance(
            card, "Кодится", note="стратегия RangeSession и сетка сканов написаны", on=p.on
        )
    if card.status == "Кодится":
        card = advance(card, "Тест", note=f"запущен прогон S1–S3 {p.run_id}", on=p.on)
    if card.status != "Тест":
        raise ValueError(f"{card.id}: прогон возможен из статуса Тест, а не {card.status}")
    to: IdeaStatus = "Отклонена" if final.startswith("отклонена") else "Результат"
    card = advance(card, to, note=f"{p.run_id}: {final}", on=p.on, verdict=final)
    result = f"{p.run_id}: {final}; отчёт reports/{p.run_id}/summary.md"
    return card.model_copy(
        update={
            "variants_tested": stage.n_trials,
            "holdout_opened_at": ho.result.opened_at if ho else card.holdout_opened_at,
            "results": [*card.results, result],
        }
    )
