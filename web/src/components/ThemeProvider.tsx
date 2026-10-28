import { useMemo, type ReactNode } from 'react'
import { useTheme } from '../lib/theme'
import { Ctx, readTokens } from '../lib/tokens'

/** Значения токенов нужны графикам (canvas не читает CSS-переменные), поэтому раздаём их через контекст. */
export function ThemeProvider({ children }: { children: ReactNode }) {
  const { choice, resolved, set } = useTheme()
  const tokens = useMemo(() => readTokens(resolved), [resolved])
  const value = useMemo(() => ({ choice, resolved, setChoice: set, tokens }), [choice, resolved, set, tokens])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}
