import { describe, expect, it } from 'vitest'
import { mix, parseColor, withAlpha } from './color'

describe('color', () => {
  it('парсит hex и rgb', () => {
    expect(parseColor('#C8643F')).toEqual([200, 100, 63])
    expect(parseColor('#fff')).toEqual([255, 255, 255])
    expect(parseColor('rgb(1, 2, 3)')).toEqual([1, 2, 3])
  })
  it('смешивает цвета', () => {
    expect(mix('#000000', '#ffffff', 0)).toBe('#000000')
    expect(mix('#000000', '#ffffff', 1)).toBe('#ffffff')
    expect(mix('#000000', '#ffffff', 0.5)).toBe('#808080')
  })
  it('добавляет прозрачность', () => {
    expect(withAlpha('#C8643F', 0.12)).toBe('rgba(200, 100, 63, 0.12)')
  })
})
