from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from pbl.cli import app
from pbl.config import validate_all

runner = CliRunner()


def test_validate_configs_ok() -> None:
    result = runner.invoke(app, ["validate-configs"])
    assert result.exit_code == 0, result.output
    assert "валидны" in result.output


def test_validate_configs_reports_errors(tmp_path: Path) -> None:
    bad = tmp_path / "configs" / "prop_types"
    bad.mkdir(parents=True)
    (bad / "broken.yaml").write_text(
        yaml.safe_dump({"id": "x", "name": "x", "style": "cfd", "typo_field": 1}), encoding="utf-8"
    )
    errors = validate_all(tmp_path)
    assert len(errors) == 1
    assert "broken.yaml" in errors[0]
    result = runner.invoke(app, ["validate-configs", "--root", str(tmp_path)])
    assert result.exit_code == 1
