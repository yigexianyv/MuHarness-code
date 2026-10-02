import { describe, expect, it } from 'vitest'

import type { AgentEvent } from '../api/types'
import { buildTurnView, formatDuration, toolActiveLabel, toolDoneLabel } from './turnPresentation'

function event(partial: Partial<AgentEvent>): AgentEvent {
  return {
    event_id: 'event-1', run_id: 'run-1', conversation_id: 'conversation-1',
    sequence: 1, type: 'model_started', event_time: '2026-08-22T00:00:00Z',
    step: 1, provider: 'fake', model: 'fake-model', message: null,
    tool_call: null, tool_result: null, usage: null, stop_reason: null,
    approval_decision: null, ...partial,
  }
}

describe('turn presentation', () => {
  it('renders generic file and shell labels', () => {
    expect(toolActiveLabel('read_file', { path: 'README.md' })).toBe('读取 README.md')
    expect(toolDoneLabel('run_shell_command', {}, true)).toBe('命令已执行')
  })

  it('builds a completed tool turn', () => {
    const view = buildTurnView([
      event({ type: 'tool_started', tool_call: { id: 'call-1', name: 'read_file', arguments: { path: 'README.md' } } }),
      event({ event_id: 'event-2', sequence: 2, type: 'tool_completed', tool_result: {
        tool_call_id: 'call-1', tool_name: 'read_file', success: true,
        output: 'ok', error: null, duration_ms: 5,
      } }),
      event({ event_id: 'event-3', sequence: 3, type: 'agent_completed', stop_reason: 'final_answer' }),
    ])
    expect(view.status).toBe('completed')
    expect(view.toolCount).toBe(1)
    expect(view.tools[0]?.label).toBe('已读取文件')
  })

  it('formats durations', () => {
    expect(formatDuration(1_500)).toBe('1.5s')
  })
})
