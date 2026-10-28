"""Манифест прогона: воспроизводимость (SPEC §8, MONTE_CARLO §13)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from pbl.schemas.common import StrictModel


class RunManifest(StrictModel):
    run_id: str
    git_sha: str = Field(min_length=7)
    config_hash: str
    data_hash: str
    seed: int
    sample: Literal["IS", "OOS", "WF", "DEMO"]
    created_at: datetime
