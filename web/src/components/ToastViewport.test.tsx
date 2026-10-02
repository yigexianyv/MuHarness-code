







import { afterEach, describe, expect, it, vi } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'

import { useToastsStore } from '../stores/toasts'
import { ToastViewport } from './ToastViewport'

describe('toasts store', () => {
  afterEach(() => {
    vi.useRealTimers()

    useToastsStore.setState({ toasts: [] })
  })

  it('push adds a toast with tone and message', () => {
    useToastsStore.getState().push('success', '已批准')
    const toasts = useToastsStore.getState().toasts
    expect(toasts).toHaveLength(1)
    expect(toasts[0].tone).toBe('success')
    expect(toasts[0].message).toBe('已批准')
  })

  it('dismiss removes the toast from the store', () => {
    useToastsStore.getState().push('info', '提示')
    const id = useToastsStore.getState().toasts[0].id
    useToastsStore.getState().dismiss(id)
    expect(useToastsStore.getState().toasts).toHaveLength(0)
  })

  it('auto-dismisses after the timeout', () => {
    vi.useFakeTimers()
    useToastsStore.getState().push('error', '失败')
    expect(useToastsStore.getState().toasts).toHaveLength(1)
    vi.advanceTimersByTime(4000)
    expect(useToastsStore.getState().toasts).toHaveLength(0)
  })
})

describe('ToastViewport', () => {
  afterEach(() => {
    useToastsStore.setState({ toasts: [] })
  })

  it('renders the viewport container with aria-live', () => {
    const html = renderToStaticMarkup(<ToastViewport />)
    expect(html).toContain('toast-viewport')
    expect(html).toContain('aria-live="polite"')
  })
})
