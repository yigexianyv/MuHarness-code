import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { handlers } = vi.hoisted(() => ({
  handlers: new Map<string, Set<(params: unknown) => void>>(),
}))

vi.mock('../rpc', () => ({
  rpcClient: {
    on: (event: string, handler: (params: unknown) => void) => {
      let set = handlers.get(event)
      if (!set) {
        set = new Set()
        handlers.set(event, set)
      }
      set.add(handler)
      return () => set.delete(handler)
    },
    setStatusListener: () => () => {},
    connect: vi.fn(),
    disconnect: vi.fn(),
  },
}))

import { useEventsStore } from './events'

function emit(method: string, params: unknown): void {
  handlers.get(method)?.forEach((handler) => handler(params))
}

const EMPTY = {
  meaById: {},
  meaVersion: {},
  meaLive: {},
  conversationVersion: {},
}

describe('events store · 长任务推送', () => {
  beforeEach(() => {
    useEventsStore.getState().disconnect()
    useEventsStore.setState(EMPTY)
    useEventsStore.getState().connect()
  })

  afterEach(() => {
    useEventsStore.getState().disconnect()
    useEventsStore.setState(EMPTY)
  })

  it('mea.status 保存最新状态并递增版本；mea.round 只递增版本', () => {
    emit('mea.status', { mea: { id: 'mea-1', status: 'running', conversation_id: 'conv-1' } })
    emit('mea.round', { mea_id: 'mea-1', round: { index: 1, phase: 'executing' } })
    emit('mea.status', { mea: { id: 'mea-1', status: 'waiting_user', conversation_id: 'conv-1' } })

    const state = useEventsStore.getState()
    expect(state.meaById['mea-1'].status).toBe('waiting_user')
    expect(state.meaVersion['mea-1']).toBe(3)
  })

  it('mea.agent_event 只记录结构化事件，不把角色输出放进会话事件', () => {
    emit('mea.agent_event', {
      mea_id: 'mea-1',
      role: 'executor',
      event: { type: 'model_output_delta', run_id: 'run-e', event_id: 'd1', step: 1, delta: 'x' },
    })
    expect(useEventsStore.getState().meaLive['mea-1']).toBeUndefined()

    emit('mea.agent_event', {
      mea_id: 'mea-1',
      role: 'executor',
      event: {
        type: 'tool_started',
        run_id: 'run-e',
        event_id: 't1',
        event_time: '2026-09-29T01:00:00+00:00',
        tool_call: { id: 'c1', name: 'write_file', arguments: {} },
      },
    })
    const state = useEventsStore.getState()
    expect(state.meaLive['mea-1']).toMatchObject({ role: 'executor', toolName: 'write_file', runId: 'run-e' })
    expect(state.eventsByRun['run-e']).toBeUndefined()
  })

  it('conversation.appended 递增会话版本', () => {
    emit('conversation.appended', { conversation_id: 'conv-1', key: 'mea-1/final' })
    emit('conversation.appended', { conversation_id: 'conv-1', key: 'mea-2/final' })
    expect(useEventsStore.getState().conversationVersion['conv-1']).toBe(2)
  })
})
