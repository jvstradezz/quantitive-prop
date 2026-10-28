import { HashRouter, Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { ThemeProvider } from './components/ThemeProvider'
import Bots from './pages/Bots'
import Ideas from './pages/Ideas'
import Leaderboard from './pages/Leaderboard'
import PropSimulator from './pages/PropSimulator'
import Strategy from './pages/Strategy'

export default function App() {
  return (
    <ThemeProvider>
      <HashRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Navigate to="/ideas" replace />} />
            <Route path="ideas" element={<Ideas />} />
            <Route path="leaderboard" element={<Leaderboard />} />
            <Route path="strategy" element={<Strategy />} />
            <Route path="prop" element={<PropSimulator />} />
            <Route path="bots" element={<Bots />} />
            <Route path="*" element={<Navigate to="/ideas" replace />} />
          </Route>
        </Routes>
      </HashRouter>
    </ThemeProvider>
  )
}
