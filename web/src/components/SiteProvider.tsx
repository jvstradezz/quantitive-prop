import { useEffect, useState, type ReactNode } from 'react'
import { SUPPORTED_SCHEMA, SiteContext, type SiteData, type SiteState } from '../lib/site'

export function SiteProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<SiteState>({ status: 'loading' })

  useEffect(() => {
    const ctl = new AbortController()
    fetch('./data/site.json', { signal: ctl.signal })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json() as Promise<SiteData>
      })
      .then((data) => {
        if (data.schema_version !== SUPPORTED_SCHEMA) throw new Error(`схема данных v${data.schema_version}, сайт ждёт v${SUPPORTED_SCHEMA}`)
        setState({ status: 'ready', data })
      })
      .catch((e: unknown) => {
        if (ctl.signal.aborted) return
        setState({ status: 'error', message: e instanceof Error ? e.message : String(e) })
      })
    return () => ctl.abort()
  }, [])

  return <SiteContext.Provider value={state}>{children}</SiteContext.Provider>
}
