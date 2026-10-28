import type { ReactNode } from 'react'
import { useThemeCtx } from '../lib/tokens'
import type { ThemeChoice } from '../lib/theme'

const icons: Record<ThemeChoice, ReactNode> = {
  system: <path d="M4 5h16v11H4zM9 20h6M12 16v4" />,
  light: (
    <>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </>
  ),
  dark: <path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z" />,
}

const titles: Record<ThemeChoice, string> = { system: 'Системная тема', light: 'Светлая тема', dark: 'Тёмная тема' }

export function ThemeToggle() {
  const { choice, setChoice } = useThemeCtx()
  return (
    <div role="group" aria-label="Тема оформления" className="inline-flex h-10 items-center gap-1 rounded-[10px] border border-line p-1">
      {(['system', 'light', 'dark'] as const).map((c) => (
        <button
          key={c}
          type="button"
          title={titles[c]}
          aria-label={titles[c]}
          aria-pressed={choice === c}
          onClick={() => setChoice(c)}
          className={`grid h-8 w-8 cursor-pointer place-items-center rounded-md border-0 ${choice === c ? 'bg-ink text-surface' : 'bg-transparent text-ink-2 hover:text-ink'}`}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            {icons[c]}
          </svg>
        </button>
      ))}
    </div>
  )
}
