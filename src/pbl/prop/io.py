"""Чтение и запись списка сделок (CSV по полям схемы `Trade`).

Parquet от движка появится в M2; до тех пор вход MC — CSV с колонками `Trade`.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Sequence
from pathlib import Path

from pbl.schemas import Trade

FIELDS = list(Trade.model_fields)


def write_trades_csv(trades: Sequence[Trade], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for t in trades:
            row = t.model_dump(mode="json")
            w.writerow({k: "" if row[k] is None else row[k] for k in FIELDS})


def read_trades_csv(path: Path) -> list[Trade]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out: list[Trade] = []
    for row in rows:
        data = {k: (None if v == "" else v) for k, v in row.items() if k in FIELDS}
        out.append(Trade.model_validate(data))
    return out


def trades_hash(trades: Sequence[Trade]) -> str:
    h = hashlib.sha256()
    for t in trades:
        h.update(t.model_dump_json().encode())
    return h.hexdigest()[:16]
