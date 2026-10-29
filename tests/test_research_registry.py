"""Реестр идей: карточки, переходы статусов, согласованность (docs/IDEAS.md §2–3)."""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from pbl.cli import app
from pbl.config import CONFIGS, REGISTRY, load_ideas
from pbl.research import registry
from pbl.schemas import IdeaCard

D0 = dt.date(2026, 10, 29)
CARD_FILE = REGISTRY / "IDEA-0001-range-bot-sniper-v2.yaml"


def _card() -> IdeaCard:
    return registry.load_card(CARD_FILE)


def _coding() -> IdeaCard:
    return registry.advance(_card(), "Кодится", note="движок смержен", on=D0)


def test_real_registry_is_consistent() -> None:
    assert registry.check_registry() == []
    ids = [c.id for c in load_ideas()]
    assert ids == sorted(set(ids)), "id карточек уникальны"
    assert "IDEA-0001" in ids


def test_idea_one_is_formalized_without_open_questions() -> None:
    card = _card()
    assert card.status == "Формализована"
    assert card.questions == []
    assert [h.status for h in card.history] == ["Новая", "Формализована"]
    assert card.variants_tested == 0
    assert card.holdout_opened_at is None


def test_idea_one_instruments_exist_in_firm_specs() -> None:
    specs = {p.stem for p in (CONFIGS / "instruments" / "fundednext").glob("*.yaml")}
    assert set(_card().markets.instruments) <= specs


def _hhmm(text: str) -> int:
    h, m = text.split(":")
    return int(h) * 60 + int(m)


def _fmt(minutes: int) -> str:
    return f"{minutes // 60 % 24:02d}:{minutes % 60:02d}"


def _shift(window: str, hours: int) -> str:
    start, end = (_hhmm(x) for x in window.split("-"))
    return f"{_fmt(start + hours * 60)}-{_fmt(end + hours * 60)}"


def test_idea_one_session_windows_follow_et_offsets() -> None:
    """Сервер FN = ET + 7 ч, МСК = ET + 7 ч летом США и ET + 8 ч зимой (D-019)."""
    for s in _card().sessions:
        assert s.server_fn == _shift(s.et, 7), s.name
        summer, winter = (x.strip() for x in s.msk.split("/"))
        assert summer == _shift(s.et, 7), s.name
        assert winter == _shift(s.et, 8), s.name


def test_dump_roundtrip_is_lossless() -> None:
    card = _card()
    text = registry.dump_card(card)
    assert text.startswith("id: IDEA-0001\n")
    assert "!!" not in text, "никаких python-тегов в YAML"
    again = IdeaCard.model_validate(yaml.safe_load(text))
    assert again == card
    assert registry.dump_card(again) == text


def test_advance_happy_path_records_history_and_keeps_original() -> None:
    card = _card()
    coding = registry.advance(card, "Кодится", note="движок смержен", on=D0)
    assert card.status == "Формализована"
    assert len(card.history) == 2, "исходная не меняется"
    testing = registry.advance(coding, "Тест", note="S1–S3", on=D0)
    result = registry.advance(
        testing, "Результат", note="отчёт готов", on=D0, verdict="доработать: фильтр дня"
    )
    demo = registry.advance(result, "Демо", note="прошла S3", on=D0, verdict="в демо")
    assert [h.status for h in demo.history][-4:] == ["Кодится", "Тест", "Результат", "Демо"]
    assert registry.check_card(demo) == []


@pytest.mark.parametrize(
    ("start", "to"),
    [("Формализована", "Тест"), ("Формализована", "Результат"), ("Новая", "Кодится")],
)
def test_advance_rejects_skipped_steps(start: str, to: str) -> None:
    card = _card().model_copy(update={"status": start})
    with pytest.raises(ValueError, match="не предусмотрен"):
        registry.advance(card, to, note="x", on=D0)  # type: ignore[arg-type]


def test_advance_requires_note_and_verdict() -> None:
    testing = registry.advance(_coding(), "Тест", note="S1", on=D0)
    with pytest.raises(ValueError, match="note"):
        registry.advance(testing, "Результат", note=" ", on=D0, verdict="в демо")
    with pytest.raises(ValueError, match="вердикт"):
        registry.advance(testing, "Результат", note="готово", on=D0)
    result = registry.advance(testing, "Результат", note="готово", on=D0, verdict="в демо")
    with pytest.raises(ValueError, match="в демо"):
        registry.advance(result, "Демо", note="ок", on=D0, verdict="доработать: x")
    with pytest.raises(ValueError, match="отклонена"):
        registry.advance(result, "Отклонена", note="нет edge", on=D0, verdict="в демо")


def test_reject_and_rework_keep_prior_verdict_in_history() -> None:
    rejected = registry.advance(
        _coding(), "Отклонена", note="S1 провален", on=D0, verdict="отклонена: PF 1.05 < 1.3"
    )
    assert registry.check_card(rejected) == []
    again = registry.advance(rejected, "Кодится", note="другой диапазон", on=D0)
    assert again.verdict is None
    assert "отклонена: PF 1.05 < 1.3" in again.history[-1].note
    assert registry.check_card(again) == []


def test_open_questions_block_progress() -> None:
    card = _card().model_copy(update={"status": "Новая", "questions": ["окно диапазона?"]})
    with pytest.raises(ValueError, match="вопросы"):
        registry.advance(card, "Формализована", note="x", on=D0)


def test_check_card_finds_inconsistencies() -> None:
    card = _card()
    broken = card.model_copy(update={"status": "Тест"})
    assert any("history" in p for p in registry.check_card(broken))
    no_verdict = broken.model_copy(update={"status": "Результат"})
    assert any("вердикта" in p for p in registry.check_card(no_verdict))
    q = card.model_copy(update={"questions": ["?"]})
    assert any("вопросы" in p for p in registry.check_card(q))
    assert any("имя файла" in p for p in registry.check_card(card, Path("IDEA-0002-x.yaml")))
    backwards = card.model_copy(
        update={"history": [card.history[1], card.history[0].model_copy(update={"date": D0})]}
    )
    assert registry.check_card(backwards)


def test_card_filename_validation() -> None:
    assert registry.card_filename("IDEA-0007", "x-y") == "IDEA-0007-x-y.yaml"
    with pytest.raises(ValueError, match="slug"):
        registry.card_filename("IDEA-0007", "Bad Slug")


def test_check_registry_flags_counter_mismatch(tmp_path: Path) -> None:
    shutil.copy(CARD_FILE, tmp_path / CARD_FILE.name)
    assert registry.check_registry(tmp_path) == []
    path, card = registry.find_card("IDEA-0001", tmp_path)
    registry.save_card(card.model_copy(update={"variants_tested": 3}), path)
    (problem,) = registry.check_registry(tmp_path)
    assert "variants_tested=3" in problem


def test_find_card_missing(tmp_path: Path) -> None:
    with pytest.raises(LookupError):
        registry.find_card("IDEA-0009", tmp_path)


def test_cli_list_check_advance(tmp_path: Path) -> None:
    shutil.copy(CARD_FILE, tmp_path / CARD_FILE.name)
    run = CliRunner()
    base = ["idea"]
    out = run.invoke(app, [*base, "list", "--directory", str(tmp_path)])
    assert out.exit_code == 0, out.output
    assert "IDEA-0001" in out.output
    assert "Формализована" in out.output
    assert run.invoke(app, [*base, "check", "--directory", str(tmp_path)]).exit_code == 0

    args = ["advance", "IDEA-0001", "Кодится", "--note", "движок смержен", "--on", "2026-10-29"]
    moved = run.invoke(app, [*base, *args, "--directory", str(tmp_path)])
    assert moved.exit_code == 0, moved.output
    assert registry.find_card("IDEA-0001", tmp_path)[1].status == "Кодится"

    bad = run.invoke(
        app,
        [*base, "advance", "IDEA-0001", "Демо", "--note", "x", "--directory", str(tmp_path)],
    )
    assert bad.exit_code != 0
    unknown = run.invoke(
        app,
        [*base, "advance", "IDEA-0001", "Готово", "--note", "x", "--directory", str(tmp_path)],
    )
    assert unknown.exit_code != 0
