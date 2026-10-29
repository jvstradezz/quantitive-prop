# web — дашборд

Дашборд. Экраны «Статус», «Идеи» и «Prop simulator» читают реальные данные репозитория из `public/data/site.json`; Leaderboard, Стратегия и Bots — макет на **фейковых данных** (`src/data/fake.ts`, помечены «макет»). Дизайн — `docs/DESIGN.md`. Стек: Vite, React, TypeScript, Tailwind v4, Lightweight Charts.

```bash
make site-data   # из корня репо: собирает web/public/data/site.json (в git не хранится)
npm ci
npm run dev      # http://localhost:5173
npm run check    # lint + typecheck + test + build (то же, что в CI)
```

Экраны: Идеи, Leaderboard, Стратегия, Prop simulator, Bots. Роутинг через hash (`#/ideas`), поэтому собранный `dist/` открывается с любого статического хоста.

- Токены — CSS-переменные в `src/index.css`; графики читают их через `src/lib/tokens.ts`.
- Тема: системная по умолчанию, ручной выбор в `localStorage` (`qp-theme`).
- Скриншоты макета — `screenshots/`.

## Публикация

`.github/workflows/pages.yml` после каждого мержа в `main` собирает `site.json` (`pbl site-data`: вехи из `docs/ROADMAP.md`, идеи из `research/registry/` и очереди `docs/IDEAS.md`, отчёты проп-MC из `reports/`), собирает `web/` и выкладывает на GitHub Pages. Сайт публичный: в `site.json` попадает только то, что и так лежит в git (`data/` и `configs/` не читаются). Чтобы статус вехи изменился на сайте, поменяйте колонку «Статус» в таблице «Обзор» `docs/ROADMAP.md` (готово / в работе / впереди).
