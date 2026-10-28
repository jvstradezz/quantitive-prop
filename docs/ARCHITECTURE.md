# ARCHITECTURE — Quantitive Prop

Статус: черновик M0 · 2026-10-28

## 1. Стек

| Слой | Выбор |
|---|---|
| Язык | Python 3.12, менеджер `uv` |
| Данные | polars, numpy, Parquet + duckdb |
| Ускорение | numba (MC, событийный бэктест) |
| Схемы и конфиги | pydantic v2, YAML |
| CLI | typer (`pbl ...`) |
| Тесты и качество | pytest + hypothesis, ruff, mypy; `make check` |
| API | FastAPI |
| Web | Vite + React + TypeScript + Tailwind + Lightweight Charts |
| Исполнение | MT5 (FundedNext), на VPS Nikita; способ (MQL5 EA или Python-пакет `MetaTrader5`) — решение M7 |

## 2. Структура репозитория

```
quantitive-prop/
├── docs/                    # спеки (этот каталог)
├── configs/
│   ├── prop_types/          # шаблоны типов пропов (1/2/3-step, instant, CFD- и futures-стиль)
│   ├── firms/               # реальные программы фирм — только из текста правил Nikita (мерж — с ОК)
│   └── instruments/
│       ├── fundednext/      # спеки инструментов FN (<symbol>.yaml)
│       └── fundingpips/
├── src/pbl/
│   ├── data/                # загрузка Dukascopy, импорт MT5, Parquet, duckdb
│   ├── instruments/         # модель инструмента, издержки, время сервера
│   ├── strategies/          # стратегии и семейства; без доступа к будущему
│   ├── engine/              # векторный скан + событийный бэктест, эквити, MAE/MFE
│   ├── prop/                # проп-симулятор Monte Carlo, оптимизатор риска
│   ├── robustness/          # WF, OOS, чувствительность, DSR/PBO
│   ├── portfolio/           # несколько стратегий на аккаунте
│   ├── research/            # реестр, очередь идей, автопоиск, счётчик вариантов
│   ├── risk/                # предохранители (мерж — с ОК)
│   ├── execution/           # адаптер MT5, демо-форвард (мерж — с ОК)
│   └── api/                 # FastAPI
├── web/                     # дашборд
├── research/registry/       # карточки идей и стратегий (YAML)
├── reports/                 # результаты прогонов: сводки, manifest.json, графики
├── data/                    # сырые и обработанные данные (в git не попадают)
└── tests/
```

Пакет называется `pbl`. Пути `src/pbl/execution/`, `src/pbl/risk/`, `configs/firms/` и пороги гейтов мержатся только с ОК Nikita.

## 3. Зависимости модулей

```
data ──► instruments ──► strategies ──► engine ──► prop ──► robustness ──► research
                                          │          ▲            │
                                          │          └─ portfolio ◄┘
                                          └────────────────────────────► execution ◄── risk
api ──► (чтение) research, reports, execution-статусы          web ──► api
```

Правила:
- Зависимости только «вправо». `prop` не знает о стратегиях: на входе — сделки и дневные записи.
- `execution` использует тот же код стратегий и тот же расчёт объёма, что и `engine` (для parity ≥ 99%).
- `risk` — отдельный модуль без зависимостей от `research` и `web`; он может только запретить действие, но не открыть сделку.
- `api` и `web` ничего не считают сами — только читают артефакты и статусы.

## 4. Главные сущности (pydantic, T2)

| Сущность | Ключевые поля |
|---|---|
| `Firm` | name, server_tz_rule, day_reset, programs[] |
| `Program` (тип пропа) | phases[], daily_loss, max_loss, payout, consistency, news, weekend, risk_rule, ea_policy, fee, sizes |
| `Phase` | target_pct, min_days, max_days, leverage |
| `InstrumentSpec` | symbol, asset_class, contract_size, point, point_value, min_lot, lot_step, commission, spread_by_hour (typical, worst), swap_long/short, triple_swap_day, sessions, server_tz |
| `Strategy` | id, family, params, instruments, timeframe, sessions, exits, filters, firm_profile |
| `Trade` | см. [MONTE_CARLO §2.1](MONTE_CARLO.md#21-сделка) |
| `DayRecord` | см. [MONTE_CARLO §2.2](MONTE_CARLO.md#22-дневная-запись) |
| `IdeaCard` | см. [IDEAS §3](IDEAS.md#3-карточка-идеи) |
| `RunManifest` | run_id, git_sha, config_hash, data_hash, seed, sample (IS/OOS/WF/DEMO), created_at |

## 5. Время

- Всё хранится в **UTC**.
- Торговый день для проп-правил считается **по серверу фирмы**:
  - FundedNext: GMT+2 зимой / GMT+3 летом, сброс 00:00 сервера. По какому календарю переход (США или ЕС) — открыто ([DECISIONS O-10](DECISIONS.md#открытые-вопросы)). Если по США, то сервер = ET + 7 ч круглый год, и 00:00 сервера = 17:00 ET.
  - FundingPips: 00:00 UTC+3 без перехода (не проверено).
- Поэтому дневные записи строятся **отдельно под каждый профиль фирмы** из одних и тех же сделок и поминутной эквити.
- В отчётах и UI сессии показываем в ET и МСК, время сервера — явной колонкой. МСК = ET + 7 ч, пока в США летнее время, и ET + 8 ч зимой.

## 6. Два класса правил фирмы

Ключевое архитектурное решение ([DECISIONS D-012](DECISIONS.md#принятые-решения)).

| Класс | Примеры | Где | Почему |
|---|---|---|---|
| Правила, меняющие сделки | окно 00:00–02:00, новостной фильтр, закрытие в пятницу (FP funded), мин. удержание ≥ 30 с, троттлинг модификаций, лимит лота | `engine` (вариант бэктеста под профиль фирмы) | без цен нельзя знать, как изменится сделка |
| Правила счёта | дневной и общий лимиты, трейлинг, цели, мин. дни, consistency, риск-правило, выплаты, предохранители бота | `prop` (Monte Carlo) | зависят от риска и последовательности, а не от цен |

## 7. Конвейер данных и прогона

```
Dukascopy ticks ─► M1 bid/ask (Parquet) ─┐
MT5 export/spec ─► калибровка спредов ────┼─► engine (стратегия × инструмент × профиль фирмы)
configs/instruments ─────────────────────┘         │
                                                   ▼
                       trades.parquet + equity_R.parquet (поминутно, в R)
                                                   │
                         ┌─────────────────────────┼──────────────────────┐
                         ▼                         ▼                      ▼
                   robustness (S1–S2)      day records по фирме      отчёт издержек
                         │                         │
                         └──────────► prop MC (S3) ◄┘
                                          │
                                          ▼
                   reports/<run_id>/ (summary.md, prop_table.parquet, manifest.json)
                                          │
                                          ▼
                            registry update ─► api ─► web
```

## 8. Хранение

| Что | Где | Формат |
|---|---|---|
| Тики и бары | `data/raw/dukascopy/`, `data/bars/<symbol>/<year>/` | Parquet, партиции по символу и году |
| Выгрузки MT5 | `data/mt5/` | исходник + Parquet |
| Сделки и эквити прогона | `data/runs/<run_id>/` | Parquet (в git не попадает) |
| Сводки прогонов | `reports/<run_id>/` | Markdown + маленький Parquet + `manifest.json` (в git) |
| Реестр | `research/registry/` | YAML-карточки (в git) |
| Счётчик вариантов DSR | `research/registry/variants.parquet` | Parquet (в git, маленький) |

В чат и контекст — только сводки до 30 строк. Большие таблицы и логи — в файлы.

## 9. Движок бэктеста

- **Векторный скан** (numpy/numba): сигнал → вход по следующему бару, стоп/тейк по high/low M1. Для автопоиска; результаты не идут в решения без событийного прогона.
- **Событийный бэктест**: bid/ask, спред по часам из спеки (худший — в стрессе), комиссия, своп с тройным днём, проскальзывание на стопах и рыночных ордерах, гэпы по цене открытия. Внутри бара путь цены неизвестен → если в одном баре задеты и стоп, и тейк, считаем **стоп**.
- **Внутридневная эквити**: поминутная плавающая прибыль каждой открытой позиции по худшей цене бара (low для лонга по bid, high для шорта по ask), в R при номинальном риске 1R на сделку.
- **Без доступа к будущему**: стратегия видит окно данных до текущего бара; тест hypothesis — подмена будущих баров не меняет прошлых сигналов.

## 10. Исполнение (M6–M7, кратко)

```
strategy (тот же код) ─► signal ─► risk guard ─► MT5 adapter ─► broker
                                       ▲               │
           monitor (heartbeat, P&L дня, лимиты) ◄──────┘──► Telegram, лог
```

- Монитор — **отдельный процесс** вне EA: уведомления в Telegram не должны исходить из EA ([DECISIONS O-07](DECISIONS.md#открытые-вопросы)).
- Kill switch закрывает всё и блокирует новые ордера на уровне `risk`.
- FundedNext: свой VPS с выделенным IP. FundingPips: VPS запрещён — решение в M6–M7.

## 11. CI

- GitHub Actions, `make check` (ruff + mypy + pytest), < 3 мин, без сети.
- Тесты на маленьких фикстурах; MC в тестах — 2 000 путей, фиксированный seed.
- Тяжёлые прогоны не в CI.
