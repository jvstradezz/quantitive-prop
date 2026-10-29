import { createContext, useContext } from 'react'
import type { IdeaStatus } from '../data/fake'

/** Форма `web/public/data/site.json`; собирает `pbl site-data` (src/pbl/api/site_data.py). */
export type MilestoneStatus = 'готово' | 'в работе' | 'впереди'

export interface MdTable {
  header: string[]
  rows: string[][]
}

export interface Milestone {
  id: string
  title: string
  summary: string
  depends: string
  status: MilestoneStatus
  tasks: MdTable | null
}

export interface SiteIdea {
  id: string
  title: string
  status: IdeaStatus
  created: string
  author: string
  summary: string
  markets: { asset_classes: string[]; instruments: string[] }
  timeframes: string[]
  sessions: { name: string; et: string; msk: string; server: string }[]
  entry: string
  stop: string
  take: string
  exits: string[]
  assumptions: string[]
  questions: string[]
  variants_tested: number
  verdict: string | null
  history: { date: string; status: IdeaStatus; note: string }[]
}

export interface ProgramRow {
  key: string
  firm: string
  label: string
  size_usd: number | null
  cost_usd: number | null
  risk: string
  p_phase1: number | null
  p_phase2: number | null
  p_funded: number | null
  p_payout_given_funded: number | null
  p1: number | null
  p1_lo_mc: number | null
  p1_hi_mc: number | null
  p1_lo_sample: number | null
  p1_hi_sample: number | null
  days_funded_median: number | null
  days_funded_p90: number | null
  days_payout_median: number | null
  ev1: number | null
  ev180: number | null
  breakeven_p: number | null
  fail_daily: number | null
  fail_max: number | null
  fail_guard_total: number | null
  fail_timeout: number | null
  fail_inactivity: number | null
  fail_rule: number | null
}

export interface PropReport {
  id: string
  title: string
  path: string
  sample: string | null
  recommendation: string | null
  costs_note: string | null
  costs: MdTable | null
  stresses: MdTable | null
  manifest: Record<string, unknown>
  programs: ProgramRow[]
}

export interface SiteData {
  schema_version: number
  meta: { repo: string | null; sha: string | null; built_at: string | null }
  milestones: Milestone[]
  ideas: SiteIdea[]
  reports: { prop: PropReport[]; documents: { path: string; title: string }[] }
}

export type SiteState = { status: 'loading' } | { status: 'error'; message: string } | { status: 'ready'; data: SiteData }

export const SiteContext = createContext<SiteState>({ status: 'loading' })

export const useSite = () => useContext(SiteContext)

export const SUPPORTED_SCHEMA = 1

/** Ссылка на файл репозитория; без `repo` (локальная сборка) возвращает null. */
export function repoFileUrl(meta: SiteData['meta'], path: string): string | null {
  return meta.repo ? `https://github.com/${meta.repo}/blob/main/${path}` : null
}
