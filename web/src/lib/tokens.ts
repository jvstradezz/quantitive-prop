import { createContext, useContext } from 'react'
import type { ThemeChoice, ThemeResolved } from './theme'

export interface Tokens {
  theme: ThemeResolved
  bg: string
  surface: string
  text: string
  text2: string
  line: string
  accent: string
  pos: string
  neg: string
  sans: string
  mono: string
}

export interface ThemeCtx {
  choice: ThemeChoice
  resolved: ThemeResolved
  setChoice: (c: ThemeChoice) => void
  tokens: Tokens
}

export const Ctx = createContext<ThemeCtx | null>(null)

export function readTokens(theme: ThemeResolved): Tokens {
  const cs = getComputedStyle(document.documentElement)
  const v = (n: string) => cs.getPropertyValue(n).trim()
  return {
    theme,
    bg: v('--bg'),
    surface: v('--surface'),
    text: v('--text'),
    text2: v('--text-2'),
    line: v('--line'),
    accent: v('--accent'),
    pos: v('--pos'),
    neg: v('--neg'),
    sans: v('--font-sans'),
    mono: v('--font-mono'),
  }
}

export function useThemeCtx(): ThemeCtx {
  const v = useContext(Ctx)
  if (!v) throw new Error('ThemeProvider отсутствует')
  return v
}

export const useTokens = () => useThemeCtx().tokens
