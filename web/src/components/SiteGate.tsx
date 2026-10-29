import type { ReactNode } from 'react'
import { useSite, type SiteData } from '../lib/site'

/** Рисует страницу, когда `site.json` загружен; иначе — заглушку загрузки или ошибки. */
export function SiteGate({ children }: { children: (data: SiteData) => ReactNode }) {
  const site = useSite()
  if (site.status === 'loading') return <div className="card p-8 text-center text-ink-2" role="status">Загружаю данные…</div>
  if (site.status === 'error') {
    return (
      <div className="card p-8 text-neg" role="alert">
        Не удалось загрузить данные сайта: {site.message}. Локально сначала выполните <span className="num">uv run pbl site-data</span>.
      </div>
    )
  }
  return <>{children(site.data)}</>
}
