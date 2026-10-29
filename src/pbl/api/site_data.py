"""Данные статичного сайта дашборда: вехи, идеи и отчёты проп-MC из репозитория → один JSON.

Сайт публичный (GitHub Pages), поэтому берём только то, что и так лежит в git:
`docs/ROADMAP.md`, карточки идей и `reports/`. Папка `data/` и конфиги фирм не читаются.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

from pbl.config import load_ideas
from pbl.schemas import IdeaCard

SCHEMA_VERSION = 1
MILESTONE_STATUSES = ("готово", "в работе", "впереди")

_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_SEPARATOR = re.compile(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$")
_CELL_SPLIT = re.compile(r"(?<!\\)\|")
_MILESTONE_HEADING = re.compile(r"^## (M\d+) — (.+)$")
_YAML_FENCE = re.compile(r"```yaml\n(.*?)```", re.DOTALL)
_PROGRAM_FLOATS = (
    "size_usd",
    "cost_usd",
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
    "fail_daily",
    "fail_max",
    "fail_guard_total",
    "fail_timeout",
    "fail_inactivity",
    "fail_rule",
)


def _cells(line: str) -> list[str]:
    parts = _CELL_SPLIT.split(line.strip())[1:-1]
    return [p.strip().replace("\\|", "|").replace("**", "") for p in parts]


def parse_md_tables(text: str) -> list[dict[str, Any]]:
    """Все markdown-таблицы текста: `{heading, intro, header, rows}`; `heading` — ближайший `##`."""
    lines = text.splitlines()
    out: list[dict[str, Any]] = []
    heading = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
        if _TABLE_ROW.match(line) and i + 1 < len(lines) and _SEPARATOR.match(lines[i + 1]):
            header = _cells(line)
            rows: list[list[str]] = []
            i += 2
            while i < len(lines) and _TABLE_ROW.match(lines[i]):
                rows.append(_cells(lines[i]))
                i += 1
            out.append({"heading": heading, "header": header, "rows": rows})
            continue
        i += 1
    return out


def _section(text: str, prefix: str) -> str | None:
    """Тело `## <prefix>…`-раздела до следующего заголовка того же уровня."""
    match = re.search(rf"^## {re.escape(prefix)}[^\n]*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    return match.group(1).strip() if match else None


def _first_h1(text: str, fallback: str) -> str:
    match = re.search(r"^# (.+)$", text, re.M)
    return match.group(1).strip() if match else fallback


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("**", "")).strip()


def build_milestones(roadmap: str) -> list[dict[str, Any]]:
    """Вехи из таблицы «Обзор» (с колонкой «Статус») и первой таблицы каждой секции `## Mx`."""
    tables = parse_md_tables(roadmap)
    overview = next((t for t in tables if t["heading"] == "Обзор"), None)
    if overview is None:
        raise ValueError("ROADMAP.md: нет таблицы «Обзор»")
    header: list[str] = overview["header"]
    if "Статус" not in header:
        raise ValueError("ROADMAP.md: в таблице «Обзор» нет колонки «Статус»")
    status_ix = header.index("Статус")
    titles = {
        m.group(1): m.group(2)
        for line in roadmap.splitlines()
        if (m := _MILESTONE_HEADING.match(line))
    }
    out: list[dict[str, Any]] = []
    for row in overview["rows"]:
        mid, status = row[0], row[status_ix]
        if status not in MILESTONE_STATUSES:
            raise ValueError(
                f"ROADMAP.md: {mid}: статус «{status}», нужен один из {MILESTONE_STATUSES}"
            )
        task_table = next((t for t in tables if t["heading"].startswith(f"{mid} — ")), None)
        out.append(
            {
                "id": mid,
                "title": titles.get(mid, row[1]),
                "summary": row[1],
                "depends": row[2],
                "status": status,
                "tasks": (
                    {"header": task_table["header"], "rows": task_table["rows"]}
                    if task_table
                    else None
                ),
            }
        )
    return out


def _idea_to_json(card: IdeaCard) -> dict[str, Any]:
    data: dict[str, Any] = card.model_dump(mode="json")
    data["sessions"] = [
        {"name": s.name, "et": s.et, "msk": s.msk, "server": s.server_fn or "—"}
        for s in card.sessions
    ]
    return data


def _queue_cards(ideas_md: str) -> list[IdeaCard]:
    """Карточки из очереди `docs/IDEAS.md` §5 (до переноса в `research/registry/`)."""
    queue = _section(ideas_md, "5. Очередь")
    if queue is None:
        return []
    cards: list[IdeaCard] = []
    for block in _YAML_FENCE.findall(queue):
        data = yaml.safe_load(block)
        if isinstance(data, dict) and str(data.get("id", "")).startswith("IDEA-"):
            cards.append(IdeaCard.model_validate(data))
    return cards


def build_ideas(root: Path) -> list[dict[str, Any]]:
    """Реестр + очередь; при совпадении id побеждает реестр. Порядок — по id."""
    by_id: dict[str, IdeaCard] = {}
    ideas_md = root / "docs" / "IDEAS.md"
    if ideas_md.is_file():
        for card in _queue_cards(ideas_md.read_text(encoding="utf-8")):
            by_id[card.id] = card
    registry = root / "research" / "registry"
    if registry.is_dir():
        for card in load_ideas(registry):
            by_id[card.id] = card
    return [_idea_to_json(by_id[k]) for k in sorted(by_id)]


def _num(value: str) -> float | None:
    if value == "":
        return None
    x = float(value)
    return None if math.isnan(x) else round(x, 6)


def _read_program_table(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {k: (_num(v) if k in _PROGRAM_FLOATS else v) for k, v in row.items()}
        out.append(item)
    return out


def build_prop_report(directory: Path, root: Path) -> dict[str, Any]:
    summary_path = directory / "summary.md"
    summary = summary_path.read_text(encoding="utf-8") if summary_path.is_file() else ""
    manifest_path = directory / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    tables = parse_md_tables(summary)

    def table_in(prefix: str) -> dict[str, Any] | None:
        t = next((t for t in tables if t["heading"].startswith(prefix)), None)
        return {"header": t["header"], "rows": t["rows"]} if t else None

    recommendation = _section(summary, "Рекомендация")
    costs_body = _section(summary, "Издержки")
    sample = next((ln for ln in summary.splitlines() if ln.startswith("Выборка:")), None)
    return {
        "id": directory.name,
        "title": _first_h1(summary, directory.name),
        "path": directory.relative_to(root).as_posix(),
        "sample": _plain(sample) if sample else None,
        "recommendation": _plain(recommendation) if recommendation else None,
        "costs_note": _plain(costs_body.split("\n\n", 1)[0]) if costs_body else None,
        "costs": table_in("Издержки"),
        "stresses": table_in("Стрессы"),
        "manifest": manifest,
        "programs": _read_program_table(directory / "prop_table.csv"),
    }


def build_reports(root: Path) -> dict[str, Any]:
    reports = root / "reports"
    prop: list[dict[str, Any]] = []
    documents: list[dict[str, str]] = []
    if reports.is_dir():
        for directory in sorted(p for p in reports.iterdir() if p.is_dir()):
            if (directory / "prop_table.csv").is_file():
                prop.append(build_prop_report(directory, root))
        for path in sorted(reports.rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            documents.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "title": _first_h1(text, path.stem),
                }
            )
    return {"prop": prop, "documents": documents}


def build_site_data(
    root: Path, *, repo: str | None = None, sha: str | None = None, built_at: str | None = None
) -> dict[str, Any]:
    """Весь JSON сайта. `built_at`/`sha` передаёт вызывающий — сама функция детерминирована."""
    roadmap = (root / "docs" / "ROADMAP.md").read_text(encoding="utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "meta": {"repo": repo, "sha": sha, "built_at": built_at},
        "milestones": build_milestones(roadmap),
        "ideas": build_ideas(root),
        "reports": build_reports(root),
    }


def write_site_data(out_dir: Path, data: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "site.json"
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=1, allow_nan=False) + "\n", encoding="utf-8"
    )
    return path
