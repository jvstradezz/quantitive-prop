from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pbl.api.site_data import build_site_data, parse_md_tables, write_site_data
from pbl.cli import app
from pbl.config import REPO_ROOT

ROADMAP = """# ROADMAP

## Обзор

| Веха | Суть | Зависит от | Статус |
|---|---|---|---|
| M0 | Каркас | — | готово |
| M1 | Проп-MC | M0 | в работе |

## M0 — каркас

| Тред | Задача |
|---|---|
| T1 | docs \\| спеки |

## M1 — проп-MC

Текст.

| Трек | Задача |
|---|---|
| MC | симулятор |
"""

IDEAS = """# IDEAS

## 3. Карточка

```yaml
id: IDEA-9999
title: шаблон
```

## 5. Очередь

```yaml
id: IDEA-0001
title: "Range bot"
status: Новая
created: 2026-10-28
author: Nikita
source: {channel: chat, ref: null, attachments: []}
summary: s
markets: {asset_classes: [index_cfd], instruments: [NAS100]}
timeframes: [M1]
sessions:
  - {name: ny, et: "09:30-10:00", msk: "16:30-17:00", server_fn: "16:30-17:00"}
entry: e
stop: s
take: t
history:
  - {date: 2026-10-28, status: Новая, note: n}
```

## 6. Дальше
"""

CSV = "key,firm,label,p1,ev1,p1_lo_sample,fail_daily\nfn/6k,fundednext,FN 6K,0.39,2.5,nan,0.0\n"
SUMMARY = """# Проп-MC: test

Выборка: **WF** · сделок 3

## Рекомендация

Лучше всего — **FN 6K**.

## Издержки стратегии

Пояснение.

| Статья | $ |
|---|---|
| Спред | -1 |

## Стрессы

| Программа | Базовый |
|---|---|
| FN 6K | 39% |
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "ROADMAP.md").write_text(ROADMAP, encoding="utf-8")
    (tmp_path / "docs" / "IDEAS.md").write_text(IDEAS, encoding="utf-8")
    run = tmp_path / "reports" / "run1"
    run.mkdir(parents=True)
    (run / "prop_table.csv").write_text(CSV, encoding="utf-8")
    (run / "summary.md").write_text(SUMMARY, encoding="utf-8")
    (run / "manifest.json").write_text('{"seed": 42}', encoding="utf-8")
    (tmp_path / "reports" / "other.md").write_text("# Другой отчёт\n", encoding="utf-8")
    return tmp_path


def test_parse_md_tables_handles_escaped_pipe_and_bold() -> None:
    (table,) = parse_md_tables("## H\n\n| a | b |\n|---|---|\n| **x** \\| y | 2 |\n")
    assert table == {"heading": "H", "header": ["a", "b"], "rows": [["x | y", "2"]]}


def test_build_site_data(repo: Path) -> None:
    data = build_site_data(repo, repo="o/r", sha="abc", built_at="2026-01-01T00:00:00+00:00")
    assert data["meta"] == {"repo": "o/r", "sha": "abc", "built_at": "2026-01-01T00:00:00+00:00"}
    m0, m1 = data["milestones"]
    assert (m0["id"], m0["status"], m0["title"]) == ("M0", "готово", "каркас")
    assert m0["tasks"]["rows"] == [["T1", "docs | спеки"]]
    assert m1["tasks"]["rows"] == [["MC", "симулятор"]]
    (idea,) = data["ideas"]
    assert idea["id"] == "IDEA-0001"
    assert idea["sessions"] == [
        {"name": "ny", "et": "09:30-10:00", "msk": "16:30-17:00", "server": "16:30-17:00"}
    ]
    (run,) = data["reports"]["prop"]
    assert run["id"] == "run1"
    assert run["recommendation"] == "Лучше всего — FN 6K."
    assert run["costs"]["rows"] == [["Спред", "-1"]]
    assert run["stresses"]["header"] == ["Программа", "Базовый"]
    assert run["manifest"] == {"seed": 42}
    assert run["programs"][0]["p1"] == 0.39
    assert run["programs"][0]["p1_lo_sample"] is None
    assert [d["title"] for d in data["reports"]["documents"]] == ["Другой отчёт", "Проп-MC: test"]


def test_registry_card_wins_over_queue(repo: Path) -> None:
    registry = repo / "research" / "registry"
    registry.mkdir(parents=True)
    card = IDEAS.split("## 5. Очередь")[1].split("```yaml\n")[1].split("```")[0]
    (registry / "IDEA-0001-range.yaml").write_text(
        card.replace("status: Новая\ncreated", "status: Формализована\ncreated"),
        encoding="utf-8",
    )
    (idea,) = build_site_data(repo)["ideas"]
    assert idea["status"] == "Формализована"


def test_status_must_be_known(repo: Path) -> None:
    roadmap = repo / "docs" / "ROADMAP.md"
    roadmap.write_text(ROADMAP.replace("в работе", "почти"), encoding="utf-8")
    with pytest.raises(ValueError, match="статус"):
        build_site_data(repo)


def test_output_is_deterministic_json(repo: Path, tmp_path: Path) -> None:
    kwargs = {"repo": "o/r", "sha": "abc", "built_at": "t"}
    a = write_site_data(tmp_path / "a", build_site_data(repo, **kwargs))
    b = write_site_data(tmp_path / "b", build_site_data(repo, **kwargs))
    assert a.read_bytes() == b.read_bytes()
    json.loads(a.read_text(encoding="utf-8"))


def test_real_repository_builds() -> None:
    data = build_site_data(REPO_ROOT)
    assert [m["id"] for m in data["milestones"]] == [f"M{i}" for i in range(8)]
    assert data["ideas"], "очередь идей пуста"
    assert data["reports"]["prop"], "нет отчёта проп-MC"
    for run in data["reports"]["prop"]:
        assert run["programs"]


def test_cli_site_data(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["site-data", "--out", str(tmp_path), "--sha", "abc"])
    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / "site.json").read_text(encoding="utf-8"))["meta"]["sha"] == "abc"
