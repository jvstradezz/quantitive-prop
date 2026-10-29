import { HashRouter, Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { SiteProvider } from './components/SiteProvider'
import { ThemeProvider } from './components/ThemeProvider'
import Bots from './pages/Bots'
import Ideas from './pages/Ideas'
import Leaderboard from './pages/Leaderboard'
import PropSimulator from './pages/PropSimulator'
import Status from './pages/Status'
import Strategy from './pages/Strategy'

export default function App() {
  return (
    <ThemeProvider>
      <SiteProvider>
        <HashRouter>
          <Routes>
            <Route element={<Layout />}>
              <Route index element={<Navigate to="/status" replace />} />
              <Route path="status" element={<Status />} />
              <Route path="ideas" element={<Ideas />} />
              <Route path="leaderboard" element={<Leaderboard />} />
              <Route path="strategy" element={<Strategy />} />
              <Route path="prop" element={<PropSimulator />} />
              <Route path="bots" element={<Bots />} />
              <Route path="*" element={<Navigate to="/status" replace />} />
            </Route>
          </Routes>
        </HashRouter>
      </SiteProvider>
    </ThemeProvider>
  )
}
