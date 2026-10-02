






import { create } from 'zustand'

import { liveActivityFrom } from '../agent/meaPresentation'
import type { AgentEvent, MeaLiveActivity, MeaRun } from '../api/types'
import { rpcClient } from '../rpc'

interface EventsState {
  connected: boolean
  eventsByRun: Record<string, AgentEvent[]>
  streamTextByRun: Record<string, Record<number, string>>

  reasoningByRun: Record<string, Record<number, string>>
  runStatuses: Record<string, string>

  /** 长任务：mea.status 推送的最新状态。 */
  meaById: Record<string, MeaRun>
  /** 长任务或它的轮次每写一次库加一，用来触发详情重新读取。 */
  meaVersion: Record<string, number>
  /** 长任务子 Run 的最近一条结构化事件（“谁在做什么”）。 */
  meaLive: Record<string, MeaLiveActivity>
  /** 会话在对话之外追加了消息（长任务的最终回复），加一以触发重新读取。 */
  conversationVersion: Record<string, number>
  connect: () => void
  disconnect: () => void

  syncRunStatuses: (updates: Record<string, string>) => void
}

const MAX_EVENTS_PER_RUN = 500








const STREAM_FLUSH_MS = 33

interface PendingStreamDelta {
  runId: string
  step: number
  delta: string
}

let pendingText: PendingStreamDelta[] = []
let pendingReasoning: PendingStreamDelta[] = []
let flushTimer: ReturnType<typeof setTimeout> | null = null

type StoreApi = EventsState

/** 合并流式文本增量，批量更新状态以控制渲染频率。 */
function flushPending(api: {
  getState: () => StoreApi
  setState: (partial: Partial<EventsState>) => void
}): void {
  if (flushTimer !== null) {
    clearTimeout(flushTimer)
    flushTimer = null
  }
  if (pendingText.length === 0 && pendingReasoning.length === 0) return

  const state = api.getState()
  const streamTextByRun: EventsState['streamTextByRun'] = { ...state.streamTextByRun }
  const reasoningByRun: EventsState['reasoningByRun'] = { ...state.reasoningByRun }

  for (const item of pendingText) {
    const runText = { ...(streamTextByRun[item.runId] ?? {}) }
    runText[item.step] = `${runText[item.step] ?? ''}${item.delta}`
    streamTextByRun[item.runId] = runText
  }
  for (const item of pendingReasoning) {
    const runReasoning = { ...(reasoningByRun[item.runId] ?? {}) }
    runReasoning[item.step] = `${runReasoning[item.step] ?? ''}${item.delta}`
    reasoningByRun[item.runId] = runReasoning
  }

  pendingText = []
  pendingReasoning = []
  api.setState({ streamTextByRun, reasoningByRun })
}

/** 将同一短时间窗口内的增量合并为一次状态更新。 */
function scheduleFlush(api: {
  getState: () => StoreApi
  setState: (partial: Partial<EventsState>) => void
}): void {
  if (flushTimer !== null) return
  flushTimer = setTimeout(() => flushPending(api), STREAM_FLUSH_MS)
}

/** 维护运行事件、流式输出和连接状态；用 event_id 去重。 */
export const useEventsStore = create<EventsState>((set, get) => {
  let unsubscribeStatus: (() => void) | null = null
  const unsubscribeHandlers: Array<() => void> = []


  const flushApi = {
    getState: () => get(),
    setState: (partial: Partial<EventsState>) => set(partial),
  }

  const flushNow = (): void => flushPending(flushApi)

  const handleAgentEvent = (params: unknown): void => {
    const agentEvent = params as AgentEvent
    const runId = agentEvent.run_id
    if (
      agentEvent.type === 'model_output_delta' &&
      agentEvent.step !== null &&
      agentEvent.delta
    ) {
      pendingText.push({ runId, step: agentEvent.step, delta: agentEvent.delta })
      scheduleFlush(flushApi)
      return
    }
    if (
      agentEvent.type === 'model_reasoning_delta' &&
      agentEvent.step !== null &&
      agentEvent.reasoning_delta
    ) {
      pendingReasoning.push({
        runId,
        step: agentEvent.step,
        delta: agentEvent.reasoning_delta,
      })
      scheduleFlush(flushApi)
      return
    }


    flushNow()

    const existing = get().eventsByRun[runId] ?? []
    if (existing.some((item) => item.event_id === agentEvent.event_id)) {
      return
    }
    const next = [...existing, agentEvent].slice(-MAX_EVENTS_PER_RUN)
    set({ eventsByRun: { ...get().eventsByRun, [runId]: next } })
  }

  const handleRunStatus = (params: unknown): void => {
    const data = params as { run_id: string; status: string }
    set({ runStatuses: { ...get().runStatuses, [data.run_id]: data.status } })
  }

  const bump = (map: Record<string, number>, key: string): Record<string, number> => ({
    ...map,
    [key]: (map[key] ?? 0) + 1,
  })

  const handleMeaStatus = (params: unknown): void => {
    const mea = (params as { mea?: MeaRun }).mea
    if (!mea?.id) return
    const state = get()
    set({
      meaById: { ...state.meaById, [mea.id]: mea },
      meaVersion: bump(state.meaVersion, mea.id),
    })
  }

  const handleMeaRound = (params: unknown): void => {
    const meaId = (params as { mea_id?: string }).mea_id
    if (!meaId) return
    set({ meaVersion: bump(get().meaVersion, meaId) })
  }

  const handleMeaAgentEvent = (params: unknown): void => {
    const data = params as { mea_id?: string | null; role?: string | null; event?: AgentEvent }
    if (!data.mea_id || !data.event) return
    const activity = liveActivityFrom(data.role ?? null, data.event)
    if (!activity) return
    set({ meaLive: { ...get().meaLive, [data.mea_id]: activity } })
  }

  const handleConversationAppended = (params: unknown): void => {
    const conversationId = (params as { conversation_id?: string }).conversation_id
    if (!conversationId) return
    set({ conversationVersion: bump(get().conversationVersion, conversationId) })
  }

  const syncRunStatuses = (updates: Record<string, string>): void => {
    if (!updates || Object.keys(updates).length === 0) return
    set({ runStatuses: { ...get().runStatuses, ...updates } })
  }

  return {
    connected: false,
    eventsByRun: {},
    streamTextByRun: {},
    reasoningByRun: {},
    runStatuses: {},
    meaById: {},
    meaVersion: {},
    meaLive: {},
    conversationVersion: {},
    connect: () => {
      if (unsubscribeStatus) return
      unsubscribeStatus = rpcClient.setStatusListener((connected) =>
        set({ connected }),
      )
      unsubscribeHandlers.push(rpcClient.on('agent.event', handleAgentEvent))
      unsubscribeHandlers.push(rpcClient.on('run.status', handleRunStatus))
      unsubscribeHandlers.push(rpcClient.on('mea.status', handleMeaStatus))
      unsubscribeHandlers.push(rpcClient.on('mea.round', handleMeaRound))
      unsubscribeHandlers.push(rpcClient.on('mea.agent_event', handleMeaAgentEvent))
      unsubscribeHandlers.push(rpcClient.on('conversation.appended', handleConversationAppended))
      rpcClient.connect()
    },
    syncRunStatuses,
    disconnect: () => {
      flushNow()
      unsubscribeStatus?.()
      unsubscribeStatus = null
      while (unsubscribeHandlers.length > 0) {
        unsubscribeHandlers.pop()?.()
      }
      rpcClient.disconnect()
      set({ connected: false })
    },
  }
})
