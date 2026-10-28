import { useCallback, useEffect, useState } from 'react'

export type ThemeChoice = 'system' | 'light' | 'dark'
export type ThemeResolved = 'light' | 'dark'

const KEY = 'qp-theme'

function readChoice(): ThemeChoice {
  try {
    const v = localStorage.getItem(KEY)
    return v === 'light' || v === 'dark' ? v : 'system'
  } catch {
    return 'system'
  }
}

function systemDark(): boolean {
  return window.matchMedia('(prefers-color-scheme: dark)').matches
}

function apply(choice: ThemeChoice) {
  const root = document.documentElement
  if (choice === 'system') delete root.dataset.theme
  else root.dataset.theme = choice
}

/** Тема: по умолчанию системная, ручной выбор запоминается (docs/DESIGN.md §4). */
export function useTheme() {
  const [choice, setChoice] = useState<ThemeChoice>(readChoice)
  const [sysDark, setSysDark] = useState(systemDark)

  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const on = () => setSysDark(mq.matches)
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])

  const set = useCallback((next: ThemeChoice) => {
    setChoice(next)
    apply(next)
    try {
      if (next === 'system') localStorage.removeItem(KEY)
      else localStorage.setItem(KEY, next)
    } catch {
      /* без хранилища выбор живёт до перезагрузки */
    }
  }, [])

  const resolved: ThemeResolved = choice === 'system' ? (sysDark ? 'dark' : 'light') : choice
  return { choice, resolved, set }
}
