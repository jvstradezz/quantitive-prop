"""Схемы отклоняют неверные данные и принимают корректные."""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from pbl.config import REPO_ROOT
from pbl.schemas import (
    DayRecord,
    IdeaCard,
    InstrumentSpec,
    RunManifest,
    Strategy,
    Trade,
)
from pbl.schemas.rules import DailyLoss, MaxLoss, NewsRule
from pbl.schemas.strategy import ExitSpec

UTC = dt.UTC
T0 = dt.datetime(2026, 1, 5, 14, 30, tzinfo=UTC)


def trade_kwargs(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "trade_id": "t1",
        "strategy_id": "s1",
        "variant_id": "v1",
        "symbol": "SYNTH",
        "side": "long",
        "open_utc": T0,
        "close_utc": T0 + dt.timedelta(minutes=5),
        "entry": 100.0,
        "exit": 101.0,
        "sl": 99.0,
        "tp": 101.0,
        "stop_dist": 100.0,
        "point_value": 1.0,
        "r_gross": 1.0,
        "cost_spread_r": 0.05,
        "cost_comm_r": 0.02,
        "cost_swap_r": 0.0,
        "cost_slip_r": 0.01,
        "r_net": 0.92,
        "mae_r": -0.3,
        "mfe_r": 1.1,
        "hold_sec": 300,
        "exit_reason": "tp",
        "sample": "IS",
    }
    base.update(over)
    return base


def test_trade_ok() -> None:
    t = Trade(**trade_kwargs())
    assert t.r_net == pytest.approx(0.92)


def test_trade_rejects_inconsistent_net() -> None:
    with pytest.raises(ValidationError, match="r_net"):
        Trade(**trade_kwargs(r_net=1.0))


def test_trade_rejects_naive_time() -> None:
    with pytest.raises(ValidationError, match="UTC"):
        Trade(**trade_kwargs(open_utc=dt.datetime(2026, 1, 5, 14, 30)))


def test_trade_rejects_non_utc_offset() -> None:
    msk = dt.timezone(dt.timedelta(hours=3))
    with pytest.raises(ValidationError, match="UTC"):
        Trade(**trade_kwargs(open_utc=dt.datetime(2026, 1, 5, 17, 30, tzinfo=msk)))


def test_trade_requires_stop_distance() -> None:
    with pytest.raises(ValidationError):
        Trade(**trade_kwargs(stop_dist=0))


def test_trade_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError, match="extra"):
        Trade(**trade_kwargs(foo=1))


@given(
    gross=st.floats(-3, 5),
    spread=st.floats(0, 0.5),
    comm=st.floats(0, 0.5),
    swap=st.floats(-0.2, 0.2),
    slip=st.floats(0, 0.5),
)
def test_trade_net_identity_holds(
    gross: float, spread: float, comm: float, swap: float, slip: float
) -> None:
    net = gross - (spread + comm + swap + slip)
    t = Trade(
        **trade_kwargs(
            r_gross=gross,
            cost_spread_r=spread,
            cost_comm_r=comm,
            cost_swap_r=swap,
            cost_slip_r=slip,
            r_net=net,
        )
    )
    assert t.r_net == net


def test_day_record() -> None:
    d = DayRecord(
        firm="fundednext",
        date=dt.date(2026, 1, 5),
        n_opened=2,
        c_in=0.0,
        c_out=0.0,
        db=0.5,
        b_max=1.2,
        path=[0.0, -0.4, 1.2, 0.5],
        risk_open_max=1.0,
    )
    assert d.weekday == 0
    assert d.year == 2026
    with pytest.raises(ValidationError):
        DayRecord(
            firm="fundednext",
            date=dt.date(2026, 1, 5),
            n_opened=0,
            c_in=0,
            c_out=0,
            db=0,
            b_max=0,
            path=[0.0],
            risk_open_max=0,
        )


def test_max_loss_exactly_one_amount() -> None:
    with pytest.raises(ValidationError, match="pct и abs_usd"):
        MaxLoss(type="static")
    with pytest.raises(ValidationError, match="pct и abs_usd"):
        MaxLoss(type="static", pct=10, abs_usd=1000)


def test_max_loss_trailing_needs_source_and_static_forbids_it() -> None:
    with pytest.raises(ValidationError, match="trail_source"):
        MaxLoss(type="trailing", pct=6)
    with pytest.raises(ValidationError, match="trailing"):
        MaxLoss(type="static", pct=6, trail_source="balance_eod")


def test_news_rule_profit_factor_consistency() -> None:
    with pytest.raises(ValidationError):
        NewsRule(window_min=5, challenge="allowed", funded="profit_factor")
    with pytest.raises(ValidationError):
        NewsRule(
            window_min=5, challenge="allowed", funded="deduct_profit", funded_profit_counted_pct=40
        )


def test_daily_loss_requires_basis() -> None:
    with pytest.raises(ValidationError):
        DailyLoss.model_validate({"pct": 5})


def test_invalid_open_item_id_rejected() -> None:
    with pytest.raises(ValidationError, match="O-01"):
        MaxLoss(type="static", pct=10, open_items=["X-1"])


def test_strategy() -> None:
    s = Strategy(
        id="s1",
        family="range_breakout",
        params={"range_min": 30, "buffer": 0.1},
        instruments=["NAS100"],
        timeframe="M5",
        exits=[ExitSpec(kind="rr", params={"rr": 1.0})],
        firm_profile="fundednext",
    )
    assert s.exits[0].kind == "rr"
    with pytest.raises(ValidationError):
        Strategy(
            id="s1",
            family="x",
            instruments=[],
            timeframe="M5",
            exits=[ExitSpec(kind="rr")],
            firm_profile="fundednext",
        )
    with pytest.raises(ValidationError):
        ExitSpec.model_validate({"kind": "martingale"})


def test_run_manifest() -> None:
    m = RunManifest(
        run_id="r1",
        git_sha="f3cdf1f",
        config_hash="abc",
        data_hash="def",
        seed=42,
        sample="OOS",
        created_at=T0,
    )
    assert m.seed == 42


def _instrument(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "symbol": "SYNTH",
        "firm": "fundednext",
        "asset_class": "indices",
        "contract_size": 1,
        "point": 0.01,
        "point_value_usd": 0.01,
        "min_lot": 0.01,
        "lot_step": 0.01,
        "commission": {"per_lot_usd": 0},
        "spread_by_hour": [{"hour": 10, "typical": 1.0, "worst": 3.0}],
        "swap_long": -1.0,
        "swap_short": 0.5,
        "swap_unit": "points",
        "triple_swap_day": "fri",
        "sessions": [{"day": "mon", "start": "01:00", "end": "23:55"}],
        "server_tz": "fundednext",
        "source": "mt5_specification",
    }
    base.update(over)
    return base


def test_instrument_spec_synthetic() -> None:
    InstrumentSpec.model_validate(_instrument())


def test_instrument_estimate_needs_open_item() -> None:
    with pytest.raises(ValidationError, match="O-04"):
        InstrumentSpec.model_validate(_instrument(source="conservative_estimate"))
    InstrumentSpec.model_validate(_instrument(source="conservative_estimate", open_items=["O-04"]))


def test_instrument_empty_spread_needs_open_item() -> None:
    with pytest.raises(ValidationError, match="O-04"):
        InstrumentSpec.model_validate(_instrument(spread_by_hour=[]))
    InstrumentSpec.model_validate(_instrument(spread_by_hour=[], open_items=["O-04"]))


def test_instrument_point_value_rules() -> None:
    with pytest.raises(ValidationError, match="point_value_usd"):
        InstrumentSpec.model_validate(_instrument(point_value_usd=None))
    with pytest.raises(ValidationError, match="стоимость пункта"):
        InstrumentSpec.model_validate(_instrument(profit_currency="JPY", point_value_usd=None))
    InstrumentSpec.model_validate(
        _instrument(profit_currency="JPY", point_value_usd=None, point_value_profit_ccy=100.0)
    )


def test_instrument_spread_worst_not_below_typical() -> None:
    with pytest.raises(ValidationError, match="worst"):
        InstrumentSpec.model_validate(
            _instrument(spread_by_hour=[{"hour": 1, "typical": 3.0, "worst": 1.0}])
        )


def _idea_blocks() -> list[dict[str, Any]]:
    text = (REPO_ROOT / "docs" / "IDEAS.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```yaml\n(.*?)```", text, flags=re.DOTALL)
    return [yaml.safe_load(b) for b in blocks]


def test_idea_template_from_docs_validates() -> None:
    """Шаблон карточки в docs/IDEAS.md §3 проходит схему IdeaCard."""
    (raw,) = _idea_blocks()
    card = IdeaCard.model_validate(raw)
    assert card.status == "Новая"
    assert card.id == "IDEA-0001"


def test_idea_card_rejects_bad_id_and_many_questions() -> None:
    raw = _idea_blocks()[-1]
    with pytest.raises(ValidationError, match="IDEA-0001"):
        IdeaCard.model_validate({**raw, "id": "idea-1"})
    with pytest.raises(ValidationError):
        IdeaCard.model_validate({**raw, "questions": ["1", "2", "3", "4"]})
    with pytest.raises(ValidationError):
        IdeaCard.model_validate({**raw, "status": "Готово"})
