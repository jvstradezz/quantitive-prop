"""Правила программы → плоский набор чисел для ядра MC (MONTE_CARLO §2.3, §3).

Все суммы — доли `B0` (5% → 0.05). Неизвестные условия выплат не выдумываем: вариант
без единого условия (срок, цикл, порог, consistency) не считается, причина — в `notes`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pbl.schemas import Program, PropTypeTemplate
from pbl.schemas.firm import ProgramOffer
from pbl.schemas.rules import AccountRules, PayoutOption

# коды для ядра
MAX_STATIC, MAX_BALANCE_EOD, MAX_BALANCE_RT, MAX_EQUITY_RT = 0, 1, 2, 3
DAILY_REF_SOD_BALANCE, DAILY_REF_MAX_BAL_EQ = 0, 1
DAILY_BASIS_INITIAL, DAILY_BASIS_REFERENCE, DAILY_BASIS_MIN = 0, 1, 2
FLOOR_NEVER_LOWERS, FLOOR_RESETS = 0, 1

DEFAULT_MIN_PROFIT = 0.01  # P-15 / O-11: мин. прибыль для выплаты, если источник молчит
EA_ADDON_ID = "ea"


@dataclass(frozen=True)
class PayoutSpec:
    id: str
    label: str
    split: float
    first_after_days: int = -1
    cycle_days: int = -1
    cycle_business_days: int = -1
    min_profit: float = DEFAULT_MIN_PROFIT
    consistency: float = -1.0  # лучший день ≤ доля прибыли; −1 — нет
    profitable_days_n: int = 0
    profitable_days_min: float = 0.0
    news_counted: float = 1.0  # доля прибыли новостных сделок, которая засчитывается


@dataclass(frozen=True)
class RuleSet:
    """Одна строка таблицы программ: программа × размер × вариант выплат."""

    key: str
    label: str
    size_usd: float
    targets: tuple[float, ...]
    min_days: tuple[int, ...]
    max_days: tuple[int, ...]
    daily_pct: float | None
    daily_ref: int
    daily_basis: int
    max_type: int
    max_dist: float
    lock_cap: float
    lock_trigger: float
    floor_after_payout: int
    inactivity_days: int
    payout: PayoutSpec | None
    risk_rule_max: float = -1.0
    striking_thr: float = -1.0
    strikes_to_close: int = 0
    fee_usd: float | None = None
    addon_usd: float = 0.0
    refund_on_payout: int = -1
    firm: str | None = None
    program_id: str | None = None
    is_template: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_phases(self) -> int:
        return len(self.targets)

    @property
    def n_slots(self) -> int:
        """Фазы челленджа + funded."""
        return self.n_phases + 1

    @property
    def cost_usd(self) -> float | None:
        return None if self.fee_usd is None else self.fee_usd + self.addon_usd

    @property
    def refund_usd(self) -> float:
        return self.fee_usd or 0.0


def _payout_spec(opt: PayoutOption, news_counted: float) -> PayoutSpec | None:
    timing = (
        opt.first_after_days,
        opt.cycle_days,
        opt.cycle_business_days,
        opt.min_profit_pct,
        opt.profitable_days,
        opt.consistency_best_day_max_pct,
    )
    if all(v is None for v in timing):
        return None
    pdays = opt.profitable_days
    return PayoutSpec(
        id=opt.id,
        label=opt.label,
        split=opt.split_pct / 100,
        first_after_days=opt.first_after_days if opt.first_after_days is not None else -1,
        cycle_days=opt.cycle_days if opt.cycle_days is not None else -1,
        cycle_business_days=opt.cycle_business_days if opt.cycle_business_days is not None else -1,
        min_profit=opt.min_profit_pct / 100
        if opt.min_profit_pct is not None
        else DEFAULT_MIN_PROFIT,
        consistency=opt.consistency_best_day_max_pct / 100
        if opt.consistency_best_day_max_pct is not None
        else -1.0,
        profitable_days_n=pdays[0] if pdays else 0,
        profitable_days_min=pdays[1] / 100 if pdays else 0.0,
        news_counted=news_counted,
    )


def _news_counted(rules: AccountRules) -> float:
    news = rules.trade_rules.news
    if news is None or news.funded == "prohibited":
        return 1.0  # окно новостей вырезает движок
    if news.funded == "deduct_profit":
        return 0.0
    assert news.funded_profit_counted_pct is not None
    return news.funded_profit_counted_pct / 100


def compile_rules(
    rules: AccountRules,
    *,
    key: str,
    label: str,
    size_usd: float,
    payout: PayoutOption | None = None,
    fee_usd: float | None = None,
    addon_usd: float = 0.0,
    refund_on_payout: int | None = None,
    firm: str | None = None,
    program_id: str | None = None,
    is_template: bool = False,
) -> RuleSet:
    notes: list[str] = []
    min_days = [p.min_days for p in rules.phases]
    if payout is not None and payout.min_days_override is not None:
        min_days = [payout.min_days_override for _ in min_days]

    dl = rules.daily_loss
    if dl is not None and dl.check != "realtime_equity":
        notes.append("дневной лимит проверяется по эквити в реальном времени (консервативно)")
    ml = rules.max_loss
    if ml.check != "realtime_equity":
        notes.append("общий лимит проверяется по эквити в реальном времени (консервативно)")
    dist = ml.pct / 100 if ml.pct is not None else (ml.abs_usd or 0.0) / size_usd
    if ml.type == "static":
        mtype = MAX_STATIC
    else:
        mtype = {
            "balance_eod": MAX_BALANCE_EOD,
            "balance_realtime": MAX_BALANCE_RT,
            "equity_realtime": MAX_EQUITY_RT,
        }[ml.trail_source or "balance_realtime"]

    payout_spec: PayoutSpec | None = None
    if payout is not None:
        payout_spec = _payout_spec(payout, _news_counted(rules))
        if payout_spec is None:
            notes.append(f"вариант выплат «{payout.label}»: условия в источнике не указаны")

    funded = rules.funded
    rr = funded.risk_rule.max_open_risk_pct / 100 if funded.risk_rule else -1.0
    striking_thr, strikes = -1.0, 0
    st = funded.striking
    if st is not None and (
        st.applies_from_size_exclusive is None or size_usd > st.applies_from_size_exclusive
    ):
        striking_thr, strikes = st.single_idea_floating_loss_pct / 100, st.strikes_to_close
    if funded.profit_concentration is not None:
        notes.append("Profit Concentration FP не моделируется")

    return RuleSet(
        key=key,
        label=label,
        size_usd=size_usd,
        targets=tuple(p.target_pct / 100 for p in rules.phases),
        min_days=tuple(min_days),
        max_days=tuple(p.max_days if p.max_days is not None else -1 for p in rules.phases),
        daily_pct=dl.pct / 100 if dl is not None else None,
        daily_ref=DAILY_REF_SOD_BALANCE
        if dl is None or dl.reference == "sod_balance"
        else DAILY_REF_MAX_BAL_EQ,
        daily_basis={
            "initial": DAILY_BASIS_INITIAL,
            "reference": DAILY_BASIS_REFERENCE,
            "min_initial_reference": DAILY_BASIS_MIN,
        }[dl.amount_basis if dl is not None else "initial"],
        max_type=mtype,
        max_dist=dist,
        lock_cap=1.0 if ml.lock_level == "initial_balance" else math.inf,
        lock_trigger=ml.lock_trigger_profit_pct / 100
        if ml.lock_trigger_profit_pct is not None
        else -1.0,
        floor_after_payout=FLOOR_RESETS
        if ml.payout_floor_behavior == "resets"
        else FLOOR_NEVER_LOWERS,
        inactivity_days=rules.inactivity_days or -1,
        payout=payout_spec,
        risk_rule_max=rr,
        striking_thr=striking_thr,
        strikes_to_close=strikes,
        fee_usd=fee_usd,
        addon_usd=addon_usd,
        refund_on_payout=refund_on_payout if refund_on_payout is not None else -1,
        firm=firm,
        program_id=program_id,
        is_template=is_template,
        notes=tuple(notes),
    )


def compile_offer(firm_id: str, program: Program, offer: ProgramOffer) -> RuleSet:
    """Программа фирмы × размер × вариант выплат. Стоимость = взнос + EA Add-On (D-002)."""
    size = offer.size
    addon = sum(
        a.fee_usd or 0.0
        for a in program.addons
        if a.id == EA_ADDON_ID
        and (a.applies_to_sizes is None or size.size_usd in a.applies_to_sizes)
    )
    return compile_rules(
        program,
        key=f"{program.id}/{size.size_usd // 1000}k/{offer.payout.id}",
        label=f"{program.name} ${size.size_usd // 1000}K · {offer.payout.label}",
        size_usd=float(size.size_usd),
        payout=offer.payout,
        fee_usd=size.fee_usd,
        addon_usd=addon,
        refund_on_payout=program.fee_refund_on_payout,
        firm=firm_id,
        program_id=program.id,
    )


def compile_template(template: PropTypeTemplate, size_usd: float | None = None) -> RuleSet:
    """Шаблон типа пропа: без цены и выплат — считаем только прохождение фаз."""
    size = size_usd or float(template.reference_balance_usd or 25_000)
    return compile_rules(
        template,
        key=f"template/{template.id}",
        label=f"Шаблон: {template.name}",
        size_usd=size,
        is_template=True,
    )
