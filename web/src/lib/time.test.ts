import { describe, expect, it } from 'vitest'
import { etOffsetHours, serverFn, timeEt, timeMsk } from './time'

describe('time', () => {
  it('летом ET = UTC−4, сервер FN = GMT+3', () => {
    const d = new Date('2026-07-15T13:30:00Z')
    expect(etOffsetHours(d)).toBe(-4)
    expect(timeEt(d)).toBe('09:30')
    expect(timeMsk(d)).toBe('16:30')
    expect(serverFn(d)).toBe('16:30')
  })
  it('зимой ET = UTC−5, сервер FN = GMT+2', () => {
    const d = new Date('2026-01-15T14:30:00Z')
    expect(etOffsetHours(d)).toBe(-5)
    expect(timeEt(d)).toBe('09:30')
    expect(timeMsk(d)).toBe('17:30')
    expect(serverFn(d)).toBe('16:30')
  })
})
