.PHONY: sync check lint typecheck test fmt validate-configs

sync:
	uv sync --frozen

check: lint typecheck test

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest

fmt:
	uv run ruff check --fix .
	uv run ruff format .

validate-configs:
	uv run pbl validate-configs
