import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { ThemeToggle } from './ThemeToggle'

/** `mock` — экран на вымышленных данных (src/data/fake.ts), пока нет движка и бот-раннера. */
const nav = [
  { to: '/status', label: 'Статус', mock: false },
  { to: '/ideas', label: 'Идеи', mock: false },
  { to: '/prop', label: 'Prop simulator', mock: false },
  { to: '/leaderboard', label: 'Leaderboard', mock: true },
  { to: '/strategy', label: 'Стратегия', mock: true },
  { to: '/bots', label: 'Bots', mock: true },
]

export function Layout() {
  const { pathname } = useLocation()
  const mock = nav.find((n) => pathname.startsWith(n.to))?.mock ?? false
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-10 border-b border-line bg-bg">
        <div className="mx-auto flex max-w-[1200px] flex-wrap items-center justify-between gap-x-6 gap-y-2 px-4 py-2 sm:px-6 lg:flex-nowrap">
          <NavLink to="/" className="font-serif text-[24px] leading-8 tracking-tight text-ink no-underline">
            Quantitive <span className="text-accent">Prop</span>
          </NavLink>
          <nav aria-label="Разделы" className="order-3 -mx-4 flex w-[calc(100%+32px)] gap-1 overflow-x-auto px-4 pb-1 sm:-mx-6 sm:w-[calc(100%+48px)] sm:px-6 lg:order-none lg:m-0 lg:w-auto lg:flex-1 lg:overflow-visible lg:p-0">
            {nav.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                className={({ isActive }) =>
                  `whitespace-nowrap rounded-lg px-3 py-2 text-[14px] leading-5 font-medium no-underline ${isActive ? 'bg-surface text-ink shadow-[inset_0_0_0_1px_var(--line)]' : 'text-ink-2 hover:text-ink'}`
                }
              >
                {n.label}
                {n.mock && <span className="ml-2 text-[11px] leading-4 font-normal text-ink-2">макет</span>}
              </NavLink>
            ))}
          </nav>
          <ThemeToggle />
        </div>
      </header>
      <div className="border-b border-line bg-surface">
        <p className="mx-auto max-w-[1200px] px-4 py-2 text-[13px] leading-5 text-ink-2">
          {mock ? (
            <>
              <span className="font-medium text-accent">Макет.</span> Все цифры на этом экране вымышлены: это набросок, не результаты движка. Реальные данные появятся с M4–M6.
            </>
          ) : (
            <>
              <span className="font-medium text-pos">Данные из репозитория.</span> Страница читает файлы git, ничего не вымышлено.
            </>
          )}{' '}
          Время: ET и МСК; сервер FundedNext принят как ET + 7 ч (допущение O-10).
        </p>
      </div>
      <main className="mx-auto max-w-[1200px] px-4 py-8 sm:px-6 sm:py-12">
        <Outlet />
      </main>
      <footer className="mx-auto max-w-[1200px] px-4 pb-12 text-[12px] leading-4 text-ink-2 sm:px-6">
        Графики: <a href="https://www.tradingview.com/" className="underline">TradingView Lightweight Charts™</a>.
      </footer>
    </div>
  )
}
