"""Все YAML в репозитории валидны, а ключевые числа совпадают с PROP_RULES и DECISIONS."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pbl.config import (
    CONFIGS,
    REPO_ROOT,
    load_firms,
    load_prop_types,
    validate_all,
)
from pbl.schemas import Firm, Program

DECISIONS = (REPO_ROOT / "docs" / "DECISIONS.md").read_text(encoding="utf-8")
KNOWN_OPEN_ITEMS = set(re.findall(r"^\| (O-\d{2}) \|", DECISIONS, flags=re.MULTILINE))


@pytest.fixture(scope="module")
def firms() -> dict[str, Firm]:
    return {f.id: f for f in load_firms()}


def programs(firm: Firm) -> dict[str, Program]:
    return {p.id: p for p in firm.programs}


def test_validate_all_clean() -> None:
    assert validate_all() == []


def test_prop_type_templates_complete() -> None:
    ids = {t.id for t in load_prop_types()}
    assert ids == {
        "cfd_1step",
        "cfd_2step",
        "cfd_3step",
        "cfd_instant_balance_trailing",
        "cfd_instant_equity_trailing",
        "futures_eod_trailing",
        "futures_intraday_trailing",
    }
    for path in (CONFIGS / "prop_types").glob("*.yaml"):
        assert path.stem in ids  # имя файла = id


def test_programs_reference_existing_templates(firms: dict[str, Firm]) -> None:
    ids = {t.id for t in load_prop_types()}
    for firm in firms.values():
        for prog in firm.programs:
            assert prog.template in ids, prog.id


def _all_open_items(obj: object) -> set[str]:
    """Собирает все ссылки O-xx из любого pydantic-объекта рекурсивно."""
    found: set[str] = set()
    if hasattr(obj, "model_dump"):
        stack: list[object] = [obj.model_dump()]
    else:
        stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for key, val in cur.items():
                if key in ("open_items", "inactivity_open_items"):
                    found.update(val)
                else:
                    stack.append(val)
        elif isinstance(cur, list):
            stack.extend(cur)
    return found


def test_open_items_exist_in_decisions(firms: dict[str, Firm]) -> None:
    used: set[str] = set()
    for firm in firms.values():
        used |= _all_open_items(firm)
    for tpl in load_prop_types():
        used |= _all_open_items(tpl)
    assert used, "ожидаются ссылки на открытые вопросы"
    assert used <= KNOWN_OPEN_ITEMS, used - KNOWN_OPEN_ITEMS


def test_fundednext_ea_offers_match_d002(firms: dict[str, Firm]) -> None:
    """D-002: EA-программы FN — 2-Step 6/15/25K, 1-Step 6/15/25K, Lite 5/10/25K, Instant 2–20K."""
    fn = programs(firms["fundednext"])
    ea_sizes = {pid: sorted(s.size_usd for s in p.sizes if s.ea_allowed) for pid, p in fn.items()}
    assert ea_sizes == {
        "fn-stellar-2step": [6000, 15000, 25000],
        "fn-stellar-1step": [6000, 15000, 25000],
        "fn-stellar-lite": [5000, 10000, 25000],
        "fn-stellar-instant": [2000, 5000, 10000, 20000],
    }
    # D-003: бот только на счетах < $50K
    for p in fn.values():
        assert p.ea.max_account_size_exclusive == 50000
        assert all(s.size_usd < 50000 for s in p.sizes if s.ea_allowed)


def test_fundednext_stellar_2step_25k(firms: dict[str, Firm]) -> None:
    """План по умолчанию (D-002): Stellar 2-Step $25K + EA Add-On $5."""
    p = programs(firms["fundednext"])["fn-stellar-2step"]
    size = next(s for s in p.sizes if s.size_usd == 25000)
    assert size.fee_usd == 199.99
    ea = next(a for a in p.addons if a.id == "ea")
    assert ea.fee_usd == 5
    assert [ph.target_pct for ph in p.phases] == [8, 5]
    assert [ph.min_days for ph in p.phases] == [5, 5]
    assert p.daily_loss is not None
    assert p.daily_loss.pct == 5
    assert p.daily_loss.reference == "sod_balance"
    assert p.max_loss.type == "static"
    assert p.max_loss.pct == 10
    assert p.fee_refund_on_payout == 1
    standard = next(o for o in p.payout_options if o.id == "standard")
    assert (standard.split_pct, standard.first_after_days, standard.cycle_days) == (80, 21, 14)


def test_fundednext_rules_from_prop_rules(firms: dict[str, Firm]) -> None:
    fn = programs(firms["fundednext"])
    for p in fn.values():
        tr = p.trade_rules
        assert tr.quick_strike is not None
        assert tr.quick_strike.hold_sec == 30
        assert tr.max_trades_day == 200
        assert tr.max_messages_day == 2000
        assert tr.settlement_window_server == ("00:00", "02:00")
        assert tr.news is not None
        assert tr.news.funded_profit_counted_pct == 40
        assert p.funded.risk_rule is not None
        assert p.funded.risk_rule.max_open_risk_pct == 3
    one = fn["fn-stellar-1step"]
    assert one.daily_loss is not None
    assert (one.daily_loss.pct, one.max_loss.pct) == (3, 6)
    lite = fn["fn-stellar-lite"]
    assert lite.daily_loss is not None
    assert (lite.daily_loss.pct, lite.max_loss.pct, lite.max_loss.alt_pct) == (4, 8, 10)
    # O-01: консервативные 5 дней по умолчанию, оптимистичные 2 — в варианте выплат
    assert [ph.min_days for ph in lite.phases] == [5, 5]
    assert {o.id: o.min_days_override for o in lite.payout_options}["cycle_10d"] == 2


def test_instant_has_no_daily_loss_and_trails_balance(firms: dict[str, Firm]) -> None:
    inst = programs(firms["fundednext"])["fn-stellar-instant"]
    assert inst.daily_loss is None
    assert inst.phases == []
    ml = inst.max_loss
    assert (ml.type, ml.pct, ml.trail_source, ml.lock_level) == (
        "trailing",
        6,
        "balance_realtime",
        "initial_balance",
    )
    assert inst.fee_refund_on_payout is None


def test_server_time(firms: dict[str, Firm]) -> None:
    fn = firms["fundednext"].server_time
    assert (fn.winter_utc_offset, fn.summer_utc_offset, fn.dst_calendar) == (2, 3, "us")
    assert fn.verified
    assert fn.open_items == []
    fp = firms["fundingpips"].server_time
    assert (fp.winter_utc_offset, fp.summer_utc_offset, fp.dst_calendar) == (3, 3, None)
    assert not fp.verified


def test_fundingpips_marked_unverified(firms: dict[str, Firm]) -> None:
    fp = firms["fundingpips"]
    assert fp.role == "backup"
    for p in fp.programs:
        assert not p.verified
        assert p.data_source == "third_party_review"
        assert p.ea.vps == "prohibited"
        assert "O-03" in p.open_items


def test_fundingpips_numbers(firms: dict[str, Firm]) -> None:
    fp = programs(firms["fundingpips"])
    std = fp["fp-2step-standard"]
    assert {s.size_usd: s.fee_usd for s in std.sizes} == {
        5000: 34,
        10000: 63,
        25000: 168,
        50000: 285,
        100000: 544,
    }
    assert std.daily_loss is not None
    assert (std.daily_loss.pct, std.daily_loss.alt_pct, std.daily_loss.alt_min_days) == (5, 3, 0)
    assert std.daily_loss.amount_basis == "min_initial_reference"  # O-13: консервативно
    assert std.max_loss.pct == 10
    assert std.fee_refund_on_payout == 4
    assert std.funded.striking is not None
    assert std.funded.striking.single_idea_floating_loss_pct == 1.2
    zero = fp["fp-zero"]
    assert zero.phases == []
    assert zero.max_loss.trail_source == "equity_realtime"
    assert zero.max_loss.lock_trigger_profit_pct == 5
    assert fp["fp-1step-flex"].funded.striking is not None
    assert fp["fp-1step-flex"].funded.striking.single_idea_floating_loss_pct == 1


def test_offers_per_program(firms: dict[str, Firm]) -> None:
    fn = programs(firms["fundednext"])
    # размеры EA × варианты выплат
    assert len(list(fn["fn-stellar-2step"].iter_offers())) == 3 * 3
    assert len(list(fn["fn-stellar-instant"].iter_offers())) == 4 * 2
    assert len(list(fn["fn-stellar-2step"].iter_offers(ea_only=False))) == 6 * 3


def test_unknown_addon_prices_are_null_not_guessed(firms: dict[str, Firm]) -> None:
    fn = programs(firms["fundednext"])["fn-stellar-2step"]
    prices = {a.id: a.fee_usd for a in fn.addons}
    assert prices["no_min_days"] is None
    assert prices["vps_ea"] is None


def test_yaml_files_are_not_empty() -> None:
    for path in Path(CONFIGS).rglob("*.yaml"):
        assert path.stat().st_size > 0
