"""Таблица программ, стрессы, рекомендация и отчёт в reports/ (MONTE_CARLO §9, SPEC §6)."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from pbl.config import REPO_ROOT
from pbl.prop.days import LotSpec, build_day_table, trades_to_arrays
from pbl.prop.engine import MCConfig, MCEngine
from pbl.prop.io import trades_hash
from pbl.prop.metrics import Stats
from pbl.prop.optimize import OptResult, optimize
from pbl.prop.rules import RuleSet, compile_offer, compile_template
from pbl.prop.stress import STRESSES, worst_period_days
from pbl.schemas import Firm, PropTypeTemplate, Trade


@dataclass(frozen=True)
class Row:
    rules: RuleSet
    opt: OptResult
    no_guards: Stats
    sample_ci: tuple[float, float, float, float] | None = None
    stresses: tuple[tuple[str, Stats], ...] = ()

    @property
    def stats(self) -> Stats:
        return self.opt.best.stats


@dataclass(frozen=True)
class PropReport:
    strategy_id: str
    sample: str
    n_trades: int
    period: tuple[dt.date, dt.date]
    n_days: int
    n_trading_days: int
    block: float
    cfg: MCConfig
    rows: tuple[Row, ...]
    skipped: tuple[str, ...]
    warnings: tuple[str, ...]
    costs: dict[str, float]
    recommended: Row | None
    manifest: dict[str, object] = field(default_factory=dict)


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def recommend(rows: Sequence[Row]) -> Row | None:
    """§9.2: среди программ фирм с EV₁ > 0 — максимум P(1-я выплата); в пределах шума MC —
    выше EV₁; затем меньше медиана дней до funded."""
    pool = [r for r in rows if not r.rules.is_template and r.stats.ev1 > 0]
    if not pool:
        return None
    best = max(pool, key=lambda r: r.stats.p1)
    lo, hi = best.stats.p1_ci
    tol = (hi - lo) / 2
    near = [r for r in pool if r.stats.p1 >= best.stats.p1 - tol]
    return max(near, key=lambda r: (r.stats.ev1, -r.stats.days_funded_median))


def cost_breakdown(trades: Sequence[Trade]) -> dict[str, float]:
    gross = sum(t.r_gross for t in trades)
    out = {
        "gross_r": gross,
        "spread_r": sum(t.cost_spread_r for t in trades),
        "comm_r": sum(t.cost_comm_r for t in trades),
        "swap_r": sum(t.cost_swap_r for t in trades),
        "slip_r": sum(t.cost_slip_r for t in trades),
        "net_r": sum(t.r_net for t in trades),
    }
    total = out["spread_r"] + out["comm_r"] + out["swap_r"] + out["slip_r"]
    out["costs_r"] = total
    out["cost_share_of_gross"] = total / gross if gross > 0 else math.inf
    return out


def run_report(
    trades: Sequence[Trade],
    firms: Sequence[Firm],
    templates: Sequence[PropTypeTemplate] = (),
    cfg: MCConfig | None = None,
    *,
    lot_specs: dict[str, LotSpec] | None = None,
    top_k: int = 3,
    ea_only: bool = True,
    program_ids: Sequence[str] | None = None,
) -> PropReport:
    """Полный прогон: каждая программа фирмы × размер × вариант выплат + шаблоны.

    Дневные записи строятся под сервер каждой фирмы; шаблоны — под сервер первой фирмы.
    Стрессы и внешний бутстрэп — для `top_k` лучших строк и рекомендации.
    """
    cfg = cfg or MCConfig()
    if not firms:
        raise ValueError("нужна хотя бы одна фирма")
    rows: list[Row] = []
    skipped: list[str] = []
    engines: dict[str, MCEngine] = {}
    warnings: list[str] = []

    def engine_for(firm: Firm) -> MCEngine:
        if firm.id not in engines:
            ta = trades_to_arrays(trades, firm.server_time, lot_specs)
            table = build_day_table(ta)
            engines[firm.id] = MCEngine(table, cfg, ta)
            warnings.extend(f"{firm.name}: {w}" for w in table.warnings)
        return engines[firm.id]

    def add(engine: MCEngine, rules: RuleSet) -> None:
        if rules.n_phases == 0 and rules.payout is None:
            skipped.append(f"{rules.label}: нет фаз и нет данных о выплатах")
            return
        if rules.payout is None and not rules.is_template:
            skipped.append(f"{rules.label}: {'; '.join(rules.notes) or 'нет условий выплат'}")
            return
        opt = optimize(engine, rules)
        no_guards = engine.run(rules, replace(opt.best.risk, guards=False))
        rows.append(Row(rules=rules, opt=opt, no_guards=no_guards))

    for firm in firms:
        eng = engine_for(firm)
        for prog in firm.programs:
            if program_ids and prog.id not in program_ids:
                continue
            offers = list(prog.iter_offers(ea_only=ea_only))
            if not offers:
                skipped.append(f"{firm.name} {prog.name}: нет вариантов выплат в источнике")
                continue
            for offer in offers:
                add(eng, compile_offer(firm.id, prog, offer, firm.name))
    base_engine = engine_for(firms[0])
    for tpl in templates:
        add(base_engine, compile_template(tpl))

    rec = recommend(rows)
    # стрессы и неопределённость по выборке — для лучших строк
    ranked = sorted(
        [r for r in rows if not r.rules.is_template],
        key=lambda r: (r.stats.ev1 > 0, r.stats.objective),
        reverse=True,
    )[:top_k]
    if rec is not None and rec.rules.key not in {r.rules.key for r in ranked}:
        ranked.append(rec)
    detailed: dict[str, Row] = {}
    for r in ranked:
        eng = engines[r.rules.firm or firms[0].id]
        risk = r.opt.best.risk
        stresses = tuple((s.label, eng.run_stress(r.rules, risk, s)) for s in STRESSES)
        ci = eng.outer_bootstrap(r.rules, risk) if cfg.outer_boot > 0 else None
        detailed[r.rules.key] = replace(r, stresses=stresses, sample_ci=ci)
    rows = [detailed.get(r.rules.key, r) for r in rows]
    rec = detailed.get(rec.rules.key) if rec is not None else None

    eng0 = engines[firms[0].id]
    t0 = eng0.table
    _, worst_label = worst_period_days(t0)
    manifest: dict[str, object] = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "seed": cfg.seed,
        "n_paths": cfg.n_paths,
        "opt_paths": cfg.opt_paths,
        "outer_boot": cfg.outer_boot,
        "outer_paths": cfg.outer_paths,
        "block_L": {k: e.block for k, e in engines.items()},
        "eps": t0.eps,
        "horizon_cal": cfg.settings.horizon_cal,
        "ev_horizon_cal": cfg.settings.ev_horizon_cal,
        "trades_hash": trades_hash(trades),
        "profiles_hash": hashlib.sha256(
            "".join(f.model_dump_json() for f in firms).encode()
            + "".join(t.model_dump_json() for t in templates).encode()
        ).hexdigest()[:16],
        "worst_period": worst_label,
    }
    samples = sorted({t.sample for t in trades})
    return PropReport(
        strategy_id=trades[0].strategy_id,
        sample="/".join(samples),
        n_trades=len(trades),
        period=(
            min(t.open_utc for t in trades).date(),
            max(t.open_utc for t in trades).date(),
        ),
        n_days=t0.n_days,
        n_trading_days=t0.n_trading_days,
        block=eng0.block,
        cfg=cfg,
        rows=tuple(rows),
        skipped=tuple(skipped),
        warnings=tuple(dict.fromkeys(warnings)),
        costs=cost_breakdown(trades),
        recommended=rec,
        manifest=manifest,
    )


# --- форматирование -----------------------------------------------------------------------


def _p(x: float) -> str:
    return "—" if math.isnan(x) else f"{100 * x:.1f}%"


def _usd(x: float) -> str:
    if math.isnan(x):
        return "—"
    if math.isinf(x):
        return "∞"
    return f"{x:+,.0f} $".replace(",", " ")


def _d(x: float) -> str:
    return "—" if math.isnan(x) else f"{x:.0f}"


def _risk(row: Row) -> str:
    r = row.opt.best.risk
    parts = [f"{100 * f:g}" for f in r.f[: row.rules.n_phases]]
    if row.rules.payout is not None:
        parts.append(f"{100 * r.f[-1]:g}")
    s = " / ".join(parts) + "%"
    if r.r1_x is not None:
        s += f" R1@{100 * r.r1_x:g}%"
    if r.r2:
        s += " R2"
    return s


def _p1_cell(row: Row) -> str:
    st = row.stats
    if math.isnan(st.p1):
        return "—"
    if row.sample_ci is not None:
        return f"**{_p(st.p1)}** [{_p(row.sample_ci[0])}–{_p(row.sample_ci[1])}]"
    lo, hi = st.p1_ci
    return f"{_p(st.p1)} (MC {_p(lo)}–{_p(hi)})"


def _fails(st: Stats) -> str:
    fs = st.fail_share
    return " / ".join(
        _p(fs.get(k, 0.0)) for k in ("daily", "max", "guard_total", "timeout", "rule")
    )


def _phase(st: Stats, k: int) -> str:
    return _p(st.p_phase[k]) if k < len(st.p_phase) else "—"


def program_table(rows: Sequence[Row]) -> list[str]:
    head = (
        "| Программа | Взнос + add-on | Риск Ф1 / Ф2 / funded | P(Ф1) | P(Ф2 \\| Ф1) | P(funded) "
        "| P(выплата \\| funded) | P(1-я выплата) [5–95%] | Дни до funded, мед. / p90 "
        "| EV₁ | EV₁₈₀ | Провалы: daily / max / guard / timeout / rule |"
    )
    out = [head, "|" + "---|" * 12]
    for r in rows:
        st = r.stats
        cost = r.rules.cost_usd
        out.append(
            "| "
            + " | ".join(
                [
                    r.rules.label,
                    "—" if cost is None else f"{cost:.2f} $",
                    _risk(r),
                    _phase(st, 0),
                    _phase(st, 1),
                    _p(st.p_funded),
                    _p(st.p_payout_given_funded),
                    _p1_cell(r),
                    f"{_d(st.days_funded_median)} / {_d(st.days_funded_p90)}",
                    _usd(st.ev1),
                    _usd(st.ev180),
                    _fails(st),
                ]
            )
            + " |"
        )
    return out


def _sentence(rep: PropReport) -> str:
    rec = rep.recommended
    if rec is None:
        return (
            "Нет программы с EV₁ > 0 при найденном риске: для этой стратегии покупка челленджа "
            "в минус. Нет edge — так и пишем."
        )
    st = rec.stats
    ci = rec.sample_ci
    ci_txt = f" ({_p(ci[0])}–{_p(ci[1])} по выборке)" if ci else ""
    fails = {k: v for k, v in st.fail_share.items() if v > 0}
    main = max(fails.items(), key=lambda kv: kv[1]) if fails else ("—", 0.0)
    share = main[1] / max(1e-12, 1 - st.p1)
    others = sorted(
        [r for r in rep.rows if not r.rules.is_template and r is not rec and r.stats.ev1 > 0],
        key=lambda r: r.stats.p1,
        reverse=True,
    )
    second = ""
    if others:
        o = others[0]
        second = f" Вторая — {o.rules.label}: P {_p(o.stats.p1)}, EV₁ {_usd(o.stats.ev1)}."
    unverified = (
        " Правила этой программы не сверены с официальным источником — сверить до покупки."
        if not rec.rules.verified
        else ""
    )
    return (
        f"Лучше всего — {rec.rules.label}: P(1-я выплата) {_p(st.p1)}{ci_txt}, "
        f"EV на взнос {_usd(st.ev1)}, риск {_risk(rec)}. Главная причина провалов — "
        f"{main[0]} ({share:.0%} провалов).{second}{unverified}"
    )


def _costs_section(rep: PropReport) -> list[str]:
    """Издержки по статьям в $, % счёта и R (SPEC §6) при риске Ф1 рекомендованной программы."""
    k = rep.costs
    rec = rep.recommended
    usd_per_r = rec.opt.best.risk.f[0] * rec.rules.size_usd if rec is not None else math.nan
    size = rec.rules.size_usd if rec is not None else math.nan
    basis = (
        f"$ и % счёта — при риске Ф1 {100 * rec.opt.best.risk.f[0]:g}% на {rec.rules.label}"
        if rec is not None
        else "$ и % счёта не считаем: нет рекомендованной программы"
    )
    out = [
        "## Издержки стратегии (весь период)",
        "",
        basis + ".",
        "",
        "| Статья | $ | % счёта | R |",
        "|---|---|---|---|",
    ]

    def line(label: str, r: float) -> str:
        r += 0.0
        return f"| {label} | {_usd(r * usd_per_r)} | {_p(r * usd_per_r / size)} | {r:+.1f} |"

    out.append(line("Валовая прибыль (до издержек)", k["gross_r"]))
    out.append(line("Спред", -k["spread_r"]))
    out.append(line("Комиссия", -k["comm_r"]))
    out.append(line("Своп", -k["swap_r"]))
    out.append(line("Проскальзывание", -k["slip_r"]))
    share = k["cost_share_of_gross"]
    flag = " **COSTS_HEAVY**" if share > 0.3 else ""
    out.append(f"| Доля издержек от валовой | | | {share:.1%}{flag} |")
    out.append(line("**Чистый результат**", k["net_r"]))
    return out


def to_markdown(rep: PropReport) -> str:
    lines: list[str] = []
    c = rep.cfg
    lines += [
        f"# Проп-MC: {rep.strategy_id}",
        "",
        f"Выборка: **{rep.sample}** · сделок {rep.n_trades} · период {rep.period[0]} … "
        f"{rep.period[1]} · дней Пн–Пт {rep.n_days}, торговых {rep.n_trading_days}",
        f"MC: {c.n_paths} путей, seed {c.seed}, блок L = {rep.block:.1f} дн., горизонт "
        f"{c.settings.horizon_cal} дн. до 1-й выплаты; оптимизатор {c.opt_paths} путей; "
        f"внешний бутстрэп {c.outer_boot} × {c.outer_paths}.",
        "",
    ]
    if "IS" in rep.sample:
        lines += ["> **IN-SAMPLE** — справка, не основание для решений (MONTE_CARLO §1).", ""]
    if rep.warnings:
        lines += ["**Предупреждения:**", *[f"- {w}" for w in rep.warnings], ""]

    lines += ["## Рекомендация", "", _sentence(rep), ""]

    lines += [*_costs_section(rep), ""]

    firm_rows = sorted(
        [r for r in rep.rows if not r.rules.is_template],
        key=lambda r: -r.stats.objective,
    )
    lines += [
        "## Программы фирм (EA-доступные размеры)",
        "",
        "P(1-я выплата) жирным — с интервалом по выборке (внешний бутстрэп), иначе — шум MC. "
        "Провалы — доли всех путей. Риск подобран на тех же данных: P слегка оптимистична.",
        "",
        *program_table(firm_rows),
        "",
    ]
    tpl = [r for r in rep.rows if r.rules.is_template]
    if tpl:
        lines += [
            "## Шаблоны типов пропов (для сравнения, без выплат)",
            "",
            *program_table(tpl),
            "",
        ]

    detailed = [r for r in rep.rows if r.stresses]
    if detailed:
        lines += [
            "## Стрессы (риск — найденный на базовом сценарии)",
            "",
            "| Программа | "
            + " | ".join(lbl for lbl, _ in detailed[0].stresses)
            + " | Без предохранителей |",
            "|---|" + "---|" * (len(detailed[0].stresses) + 1),
        ]
        for r in detailed:
            cells = [f"{_p(s.objective)} · {_usd(s.ev1)}" for _, s in r.stresses]
            cells.append(f"{_p(r.no_guards.objective)} · {_usd(r.no_guards.ev1)}")
            lines.append(f"| {r.rules.label} | " + " | ".join(cells) + " |")
        lines += ["", "В ячейке: P(1-я выплата) · EV₁.", ""]

    if rep.recommended is not None and rep.recommended.opt.curves:
        rec = rep.recommended
        names = [f"Ф{k + 1}" for k in range(rec.rules.n_phases)] + ["funded"]
        lines += [
            f"## Кривая P(f) по фазам: {rec.rules.label}",
            "",
            "| Риск на сделку | " + " | ".join(names[s] for s in rec.opt.curves) + " |",
            "|---|" + "---|" * len(rec.opt.curves),
        ]
        grid = sorted({f for cur in rec.opt.curves.values() for f, _ in cur})
        for f in grid:
            cells = []
            for cur in rec.opt.curves.values():
                m = dict(cur)
                cells.append(_p(m[f]) if f in m else "нельзя")
            lines.append(f"| {100 * f:g}% | " + " | ".join(cells) + " |")
        lines += ["", "funded — P(1-я выплата | funded) при старте с funded.", ""]

    notes = sorted(
        {n for r in rep.rows for n in r.rules.notes if "условия в источнике" not in n}
        | {f"{r.rules.label}: {n}" for r in rep.rows for n in r.opt.notes}
    )
    if rep.skipped or notes:
        lines += ["## Что не посчитано и допущения", ""]
        lines += [f"- пропущено: {s}" for s in rep.skipped]
        lines += [f"- {n}" for n in notes]
        lines += [""]
    lines += [
        "Причины провалов: daily — дневной лимит фирмы, max — общий лимит, guard — общий "
        "предохранитель бота (60% общего лимита, бот остановлен), timeout — не успел к горизонту, "
        "rule — Risk Rule FN / Striking FP.",
        "",
    ]
    return "\n".join(lines)


def rows_to_csv(rep: PropReport, path: Path) -> None:
    fields = [
        "key",
        "firm",
        "label",
        "size_usd",
        "cost_usd",
        "risk",
        "p_phase1",
        "p_phase2",
        "p_funded",
        "p_payout_given_funded",
        "p1",
        "p1_lo_mc",
        "p1_hi_mc",
        "p1_lo_sample",
        "p1_hi_sample",
        "days_funded_median",
        "days_funded_p90",
        "days_payout_median",
        "ev1",
        "ev180",
        "breakeven_p",
        *[f"fail_{k}" for k in ("daily", "max", "guard_total", "timeout", "inactivity", "rule")],
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for r in rep.rows:
            st = r.stats
            ci = r.sample_ci or (math.nan, math.nan, math.nan, math.nan)
            w.writerow(
                [
                    r.rules.key,
                    r.rules.firm or "",
                    r.rules.label,
                    r.rules.size_usd,
                    r.rules.cost_usd if r.rules.cost_usd is not None else "",
                    _risk(r),
                    st.p_phase[0] if st.p_phase else "",
                    st.p_phase[1] if len(st.p_phase) > 1 else "",
                    st.p_funded,
                    st.p_payout_given_funded,
                    st.p1,
                    st.p1_ci[0],
                    st.p1_ci[1],
                    ci[0],
                    ci[1],
                    st.days_funded_median,
                    st.days_funded_p90,
                    st.days_payout_median,
                    st.ev1,
                    st.ev180,
                    st.breakeven_p,
                    *[
                        st.fail_share.get(k, 0.0)
                        for k in ("daily", "max", "guard_total", "timeout", "inactivity", "rule")
                    ],
                ]
            )


def write_report(rep: PropReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(to_markdown(rep), encoding="utf-8")
    rows_to_csv(rep, out_dir / "prop_table.csv")
    (out_dir / "manifest.json").write_text(
        json.dumps(rep.manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return out_dir / "summary.md"


def _json_default(o: object) -> object:
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))
