








import { useEffect, useRef, useState } from 'react'
import type { ReactElement } from 'react'

import { formatDuration } from '../agent/turnPresentation'

export default function AssistantReasoning({
  text,
  autoExpand = false,
  busy = false,
  durationMs = null,
}: {
  text: string

  autoExpand?: boolean

  busy?: boolean

  durationMs?: number | null
}): ReactElement | null {
  const [open, setOpen] = useState(autoExpand)
  const [userPinned, setUserPinned] = useState(false)
  const prevAutoExpand = useRef(autoExpand)

  useEffect(() => {
    const wasAuto = prevAutoExpand.current
    prevAutoExpand.current = autoExpand
    if (autoExpand) {

      setUserPinned(false)
      setOpen(true)
    } else if (wasAuto && !autoExpand && !userPinned) {

      setOpen(false)
    }
  }, [autoExpand, userPinned])

  if (!text) return null

  const label = busy
    ? 'Thinking'
    : durationMs !== null
      ? `Thought for ${formatDuration(durationMs)}`
      : 'Thinking'

  return (
    <div
      className={`assistant-reasoning${open ? ' assistant-reasoning--open' : ''}`}
      data-testid="assistant-reasoning"
    >
      <button
        type="button"
        className="assistant-reasoning__toggle"
        aria-expanded={open}
        aria-label={open ? '收起思考过程' : '展示思考过程'}
        title={label}
        onClick={() => {
          setUserPinned(true)
          setOpen((value) => !value)
        }}
      >
        {busy ? (
          <span className="assistant-reasoning__spinner" aria-hidden="true" />
        ) : null}
        <span className="assistant-reasoning__chevron" aria-hidden="true" />
      </button>
      <div className="assistant-reasoning__wrap">
        <div className="assistant-reasoning__body">{text}</div>
      </div>
    </div>
  )
}
