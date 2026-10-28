import type { ReactNode } from 'react'

export function PageHeader({ title, lead, right }: { title: string; lead?: ReactNode; right?: ReactNode }) {
  return (
    <header className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div className="max-w-2xl">
        <h1>{title}</h1>
        {lead && <p className="mt-2 text-ink-2">{lead}</p>}
      </div>
      {right}
    </header>
  )
}

export function Card({ title, note, children, className = '', pad = true }: { title?: string; note?: ReactNode; children: ReactNode; className?: string; pad?: boolean }) {
  return (
    <section className={`card ${className}`}>
      {(title || note) && (
        <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 px-4 pt-4 sm:px-6 sm:pt-6">
          {title && <h3>{title}</h3>}
          {note && <div className="text-[13px] leading-5 text-ink-2">{note}</div>}
        </div>
      )}
      <div className={pad ? 'p-4 sm:p-6' : ''}>{children}</div>
    </section>
  )
}

export function Stat({ label, value, sub, tone }: { label: string; value: ReactNode; sub?: ReactNode; tone?: 'pos' | 'neg' }) {
  const color = tone === 'pos' ? 'text-pos' : tone === 'neg' ? 'text-neg' : 'text-ink'
  return (
    <div className="card p-4">
      <div className="label">{label}</div>
      <div className={`num mt-2 text-[24px] leading-8 ${color}`}>{value}</div>
      {sub && <div className="mt-1 text-[13px] leading-5 text-ink-2">{sub}</div>}
    </div>
  )
}

/** Таблица, которая скроллится внутри контейнера, а не ломает страницу на телефоне. */
export function TableWrap({ children }: { children: ReactNode }) {
  return <div className="overflow-x-auto">{children}</div>
}
