# CLAUDE.md — quantitive-prop

Конвейер стратегий и ботов для CFD-пропов. Метрика успеха — P(1-я выплата) на живых данных. Спеки — `docs/` (читай SPEC, ARCHITECTURE, MONTE_CARLO перед работой над ядром). Общаемся по-русски, термины и код — на английском.

## Команды

- `make sync` — `uv sync --frozen` (Python 3.12).
- `make check` — ruff + ruff format --check + mypy (strict) + pytest. **Перед «готово» зелёный, итоговые строки — в финальное сообщение.**
- `make fmt` — автоправки ruff. `make validate-configs` — проверка всех YAML по схемам.

## Правила

- Одна задача = ветка = PR. Коммить и пушь часто.
- Тяжёлые прогоны — кодом, результаты в `reports/`, в чат — сводка до 30 строк. `data/` в git не попадает.
- Нет доступа (домен, ключ, данные) — скажи, чего именно нет, и остановись. Данные не выдумываем и не мокаем.
- Конфиги: числа правил фирм — только из `PROP_RULES_FundedNext_FundingPips.md` (D-005). Неизвестное — `null` + `open_items: [O-xx]` (O-xx — в `docs/DECISIONS.md`). Новые решения и вопросы пишем в `docs/DECISIONS.md`.
- Проценты в схемах и YAML — в «человеческих» единицах (5 значит 5%).
- Время хранится в UTC. Торговый день для проп-правил — по серверу фирмы.
- CI < 3 мин, детерминированный: фиксированные seed, без сети, маленькие фикстуры.

## Только с ОК Nikita

Пути `src/pbl/execution/`, `src/pbl/risk/`, `configs/firms/`, пороги гейтов и предохранители бота; реальные деньги и проп-аккаунты; платные данные; удаление данных.

## Структура

`src/pbl/schemas/` — pydantic-схемы (Firm/Program, PropTypeTemplate, InstrumentSpec, Strategy, Trade/DayRecord, IdeaCard, RunManifest). `src/pbl/config.py` — загрузка и валидация YAML. `src/pbl/cli.py` — `pbl ...`. Остальные модули — по `docs/ARCHITECTURE.md`; зависимости только «вправо».
