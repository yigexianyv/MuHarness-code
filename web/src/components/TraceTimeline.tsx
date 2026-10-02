import type { CSSProperties, ReactElement } from 'react'

import type { AgentEvent } from '../api/types'

interface TimelineEntry {
  id: string
  sequence: number
  timeLabel: string
  dateTime: string | undefined
  description: string
}

const ROW_STYLE: CSSProperties = {
  display: 'flex',
  gap: 10,
  padding: '4px 0',
  fontSize: 12.5,
  alignItems: 'baseline',
}

const TIME_STYLE: CSSProperties = { width: 76, flexShrink: 0 }
const SEQUENCE_STYLE: CSSProperties = { width: 40, flexShrink: 0 }

function presentEvent(event: AgentEvent): TimelineEntry {
  const date = new Date(event.event_time)
  const validTime = !Number.isNaN(date.getTime())
  const metadata = [
    event.step != null ? `step=${event.step}` : null,
    event.tool_call ? `tool=${event.tool_call.name}` : null,
    event.tool_result ? `success=${event.tool_result.success ? 'true' : 'false'}` : null,
    event.approval_decision ? `decision=${event.approval_decision}` : null,
    event.usage ? `tokens=${event.usage.total_tokens}` : null,
  ].filter((value) => value !== null)

  return {
    id: event.event_id,
    sequence: event.sequence,
    timeLabel: validTime ? date.toLocaleTimeString() : event.event_time,
    dateTime: validTime ? event.event_time : undefined,
    description: metadata.length > 0
      ? `${event.type} · ${metadata.join(' ')}`
      : event.type,
  }
}

function TimelineRow({ entry }: { entry: TimelineEntry }): ReactElement {
  return (
    <div role="listitem" style={ROW_STYLE}>
      <time className="text-muted" dateTime={entry.dateTime} style={TIME_STYLE}>
        {entry.timeLabel}
      </time>
      <span className="text-muted" style={SEQUENCE_STYLE}>#{entry.sequence}</span>
      <span>{entry.description}</span>
    </div>
  )
}

export default function TraceTimeline({ events }: { events: AgentEvent[] }): ReactElement {
  if (events.length === 0) {
    return <div className="empty">该 Run 暂无 Trace 事件。</div>
  }

  const entries = events.map(presentEvent)
  return (
    <div role="list" aria-label="运行事件">
      {entries.map((entry) => <TimelineRow key={entry.id} entry={entry} />)}
    </div>
  )
}
