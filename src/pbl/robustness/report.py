"""Отчёт конвейера гейтов: `gates.md` (сводка ≤ 60 строк) и `gates.json` в reports/<run>/."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from pbl.prop.report import write_report
from pbl.robustness.autoreject import AutoReject
from pbl.robustness.gates import IS_GATES, Candidate, GateReport


def _mark(v: bool | None) -> str:
    return "—" if v is None else ("✓" if v else "✗")


def _f(x: float, fmt: str = ".2f") -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, fmt)


def _params(c: Candidate) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(c.params.items()))


def reject_lines(rej: AutoReject) -> list[str]:
    out = ["| Детектор | Найдено | Детали |", "|---|---|---|"]
    out += [f"| {f.name} | {_mark(f.triggered)} | {f.detail} |" for f in rej.findings]
    return out


def to_markdown(rep: GateReport) -> list[str]:
    sp = rep.splits
    pbo = "—" if rep.pbo is None else f"{rep.pbo.pbo:.2f}"
    out = [
        f"# Гейты S1–S3: {rep.strategy_id}",
        "",
        f"**Вердикт:** {rep.verdict}",
        "",
        f"IS {sp.first} … {sp.cut} · holdout {sp.cut} … {sp.last} · проб {rep.n_trials} "
        f"(N_eff {rep.clusters.n_eff}) · WF-окон {len(rep.picks)} · WFE {_f(rep.wfe)} · "
        f"PBO {pbo} · прошли S1: {int(rep.s1['s1'].sum())}",
        "",
        "## Кандидаты на IS (топ по Sharpe среди прошедших S1)",
        "",
        "| Вариант | n | PF | E×1.5 | DSR (N_eff / сырое) | " + " | ".join(IS_GATES) + " |",
        "|---" * (5 + len(IS_GATES)) + "|",
    ]
    for c in rep.candidates:
        gates = " | ".join(_mark(c.gates[g]) for g in IS_GATES)
        out.append(
            f"| {_params(c)} | {int(c.s1['n'])} | {_f(c.s1['pf'])} | {_f(c.s1['e15'], '.3f')} | "
            f"{_f(c.dsr.dsr, '.3f')} / {_f(c.dsr.dsr_raw, '.3f')} | {gates} |"
        )
    fin = rep.finalist
    if fin is not None:
        worst = min((s.pf for s in fin.sens), default=math.nan)
        out += ["", f"Финалист: {_params(fin)}; худший сосед ±10/20% — PF {_f(worst)}"]
        for f in fin.filters:
            d = "—"
            if f.diff is not None:
                d = f"{f.diff.diff:+.3f} [{f.diff.low:+.3f}; {f.diff.high:+.3f}]"
            out.append(
                f"{f.name} на IS: разность корзин {d}, "
                f"mean R×√n {f.score:.1f} vs {f.twin_score:.1f} без фильтра"
            )
    if rep.holdout is not None:
        ho = rep.holdout
        out += ["", f"## Holdout (открыт {ho.opened_at})", ""]
        out += ["| Метрика | Holdout | 5% | 95% | Не хуже полосы |", "|---|---|---|---|---|"]
        for k, b in ho.band.bands.items():
            out.append(
                f"| {k} | {b.observed:.3f} | {b.low:.3f} | {b.high:.3f} | {_mark(not b.worse)} |"
            )
        out.append("")
        out.append("Гейты holdout: " + ", ".join(f"{k} {_mark(v)}" for k, v in ho.gates.items()))
    if rep.s3 is not None:
        s3 = rep.s3
        out += ["", "## S3 — проп-MC на OOS (тесты WF + holdout)", ""]
        if s3.best is not None:
            st = s3.best.stats
            out.append(
                f"Лучше всего: {s3.best.rules.label} — P(1-я выплата) {100 * st.p1:.1f}%, "
                f"EV₁ {st.ev1:+.0f} $, риск {s3.best.opt.best.risk.label()}"
            )
        out.append(f"S3: {_mark(s3.passed)} · сделок OOS {len(s3.trades)}")
        out += ["", "Аудит:"]
        out += [f"- {a.name}: {_mark(a.flagged)} {a.detail}" for a in s3.audit]
    best = fin or (rep.candidates[0] if rep.candidates else None)
    if best is not None and best.reject is not None:
        out += ["", "## Автоотказ (сделки IS)", "", *reject_lines(best.reject)]
    return out


def to_json(rep: GateReport) -> dict[str, Any]:
    def cand(c: Candidate) -> dict[str, Any]:
        return {
            "params": c.params,
            "s1": c.s1,
            "dsr": c.dsr.dsr,
            "dsr_raw": c.dsr.dsr_raw,
            "n_eff": c.dsr.n_eff,
            "gates": c.gates,
            "sensitivity": [s.__dict__ for s in c.sens],
            "reject": [f.__dict__ for f in c.reject.findings] if c.reject else [],
        }

    ho = rep.holdout
    return {
        "strategy_id": rep.strategy_id,
        "verdict": rep.verdict,
        "splits": {
            "first": str(rep.splits.first),
            "cut": str(rep.splits.cut),
            "last": str(rep.splits.last),
        },
        "n_trials": rep.n_trials,
        "n_eff": rep.clusters.n_eff,
        "wfe": rep.wfe,
        "pbo": None if rep.pbo is None else rep.pbo.pbo,
        "candidates": [cand(c) for c in rep.candidates],
        "holdout": None
        if ho is None
        else {
            "opened_at": str(ho.opened_at),
            "gates": ho.gates,
            "band": {k: b.__dict__ for k, b in ho.band.bands.items()},
            "stats": ho.stats,
        },
        "s3": None
        if rep.s3 is None
        else {
            "passed": rep.s3.passed,
            "best": None if rep.s3.best is None else rep.s3.best.rules.key,
            "audit": [a.__dict__ for a in rep.s3.audit],
        },
    }


def write_gates_report(rep: GateReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "gates.md"
    path.write_text("\n".join(to_markdown(rep)) + "\n", encoding="utf-8")
    (out_dir / "gates.json").write_text(
        json.dumps(to_json(rep), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    if rep.s3 is not None:
        write_report(rep.s3.prop, out_dir / "prop")
    return path
