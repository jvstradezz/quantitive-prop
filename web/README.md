# web — дашборд

Статичный макет M0 на **фейковых данных** (`src/data/fake.ts`). Дизайн — `docs/DESIGN.md`. Стек: Vite, React, TypeScript, Tailwind v4, Lightweight Charts.

```bash
npm ci
npm run dev      # http://localhost:5173
npm run check    # lint + typecheck + test + build (то же, что в CI)
```

Экраны: Идеи, Leaderboard, Стратегия, Prop simulator, Bots. Роутинг через hash (`#/ideas`), поэтому собранный `dist/` открывается с любого статического хоста.

- Токены — CSS-переменные в `src/index.css`; графики читают их через `src/lib/tokens.ts`.
- Тема: системная по умолчанию, ручной выбор в `localStorage` (`qp-theme`).
- Скриншоты макета — `screenshots/`.
