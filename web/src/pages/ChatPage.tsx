import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { approveApproval, denyApproval, listApprovals } from '../api/approvals'
import { listArtifacts } from '../api/artifacts'
import {
  createConversation,
  deleteConversation,
  getConversation,
  listConversations,
  renameConversation,
  sendMessage,
} from '../api/conversations'
import {
  answerMea,
  cancelMea,
  getMea,
  listMeas,
  listMeaTools,
  noteMea,
  pauseMea,
  resumeMea,
  startMea,
} from '../api/mea'
import type { MeaNoteKind } from '../api/mea'
import { cancelRun, interruptRun, listRuns, recoverRun } from '../api/runs'
import { listTasks, planAccept, planReject } from '../api/tasks'
import type { AgentMode, MeaAmendResult, Message, Task } from '../api/types'
import { createConversationDraftStore } from '../agent/conversationDraft'
import {
  isMeaBusy,
  isMeaOpen,
  isMeaRun,
  mergeMeas,
  pickPanelMea,
} from '../agent/meaPresentation'
import { latestRunId } from '../agent/runAnalysis'
import { buildTurnView } from '../agent/turnPresentation'
import ApprovalCard from '../components/ApprovalCard'
import RunStatusBar from '../components/RunStatusBar'
import Composer from '../components/Composer'
import type { ComposerCommand } from '../components/Composer'
import ConversationList from '../components/ConversationList'
import CurrentTaskPanel from '../components/CurrentTaskPanel'
import ForkBanner from '../components/ForkBanner'
import LiveAgentTurn from '../components/LiveAgentTurn'
import MeaPanel from '../components/MeaPanel'
import MessageList from '../components/MessageList'
import PlanCard from '../components/PlanCard'
import ResultCard from '../components/ResultCard'
import RunActivity from '../components/RunActivity'
import { Icon } from '../components/Icon'
import { SectionHeader } from '../components/ui'
import { useEventsStore } from '../stores/events'
import type { PageKey } from '../App'

export default function ChatPage({
  onNavigate,
  onOpenRun,
  initialConversationId,
  onConversationChange,
  initialDraft,
  onDraftConsumed,
  onNewWork,
}: {
  onNavigate?: (page: PageKey) => void
  onOpenRun?: (runId: string) => void
  initialConversationId?: string | null
  onConversationChange?: (conversationId: string | null) => void
  initialDraft?: { conversationId: string; content: string; mode: 'normal' | 'plan' } | null
  onDraftConsumed?: () => void
  onNewWork?: () => void
}): React.JSX.Element {
  const queryClient = useQueryClient()
  const eventsByRun = useEventsStore((state) => state.eventsByRun)
  const runStatuses = useEventsStore((state) => state.runStatuses)
  const connected = useEventsStore((state) => state.connected)
  const meaById = useEventsStore((state) => state.meaById)
  const meaVersion = useEventsStore((state) => state.meaVersion)
  const meaLive = useEventsStore((state) => state.meaLive)
  const conversationVersion = useEventsStore((state) => state.conversationVersion)
  const [selectedId, setSelectedId] = useState<string | null>(
    initialDraft?.conversationId ?? initialConversationId ?? null,
  )
  const selectedIdRef = useRef(selectedId)
  const [lastRunId, setLastRunId] = useState<string | null>(null)
  const [draftStore] = useState(() => createConversationDraftStore())
  const [, setDraftVersion] = useState(0)
  const appliedDraft = useRef<string | null>(null)
  const draftState = initialDraft && selectedId === initialDraft.conversationId && appliedDraft.current !== initialDraft.conversationId
    ? initialDraft : draftStore.get(selectedId)
  const draft = draftState.content
  const mode = draftState.mode
  const refreshDraft = useCallback(() => setDraftVersion((version) => version + 1), [])
  const setDraft = useCallback((content: string) => {
    draftStore.update(selectedId, { content })
    refreshDraft()
  }, [draftStore, selectedId, refreshDraft])
  const setMode = useCallback((next: AgentMode) => {
    draftStore.update(selectedId, { mode: next === 'plan' ? 'plan' : 'normal' })
    refreshDraft()
  }, [draftStore, selectedId, refreshDraft])
  const [conversationSidebarOpen, setConversationSidebarOpen] = useState(
    () => typeof window === 'undefined' || window.innerWidth > 760,
  )
  const [runInspectorOpen, setRunInspectorOpen] = useState(false)
  const [planResolved, setPlanResolved] = useState<string | null>(null)
  const [optimisticMessage, setOptimisticMessage] = useState<{
    conversationId: string
    message: Message
  } | null>(null)
  const [sendError, setSendError] = useState<string | null>(null)
  const [meaNotice, setMeaNotice] = useState<string | null>(null)

  const [liveTurnActive, setLiveTurnActive] = useState(false)

  const selectConversation = useCallback((conversationId: string | null): void => {
    if (selectedIdRef.current !== conversationId) {
      if (selectedIdRef.current === null && conversationId) draftStore.adoptUnassigned(conversationId)
      selectedIdRef.current = conversationId
      setLastRunId(null)
      setPlanResolved(null)
      setLiveTurnActive(false)
      setRunInspectorOpen(false)
      setOptimisticMessage(null)
      setSendError(null)
      setMeaNotice(null)
    }
    setSelectedId(conversationId)
    if (typeof window !== 'undefined' && window.innerWidth <= 760) setConversationSidebarOpen(false)
    onConversationChange?.(conversationId)
  }, [draftStore, onConversationChange])

  useEffect(() => {
    if (!initialDraft || appliedDraft.current === initialDraft.conversationId) return
    appliedDraft.current = initialDraft.conversationId
    draftStore.set(initialDraft.conversationId, { content: initialDraft.content, mode: initialDraft.mode })
    selectConversation(initialDraft.conversationId)
    refreshDraft()
    onDraftConsumed?.()
  }, [initialDraft, draftStore, selectConversation, refreshDraft, onDraftConsumed])

  const conversationsQuery = useQuery({
    queryKey: ['conversations'],
    queryFn: () => listConversations(),
  })
  const conversations = conversationsQuery.data ?? []

  useEffect(() => {
    if (selectedId === null && conversations.length > 0) {
      selectConversation(conversations[0].id)
    }
  }, [conversations, selectedId, selectConversation])

  useEffect(() => {
    if (initialConversationId && initialConversationId !== selectedId) {
      selectConversation(initialConversationId)
    }
  }, [initialConversationId, selectedId, selectConversation])

  const conversationQuery = useQuery({
    queryKey: ['conversation', selectedId],
    queryFn: () => (selectedId ? getConversation(selectedId) : Promise.resolve(null)),
    enabled: selectedId !== null,
  })


  const liveRunId = useMemo(() => {
    if (!selectedId) return null
    let candidate: { id: string; time: string } | null = null
    for (const [runId, events] of Object.entries(eventsByRun)) {
      const latest = events.at(-1)
      if (!latest || latest.conversation_id !== selectedId) continue
      if (!['pending', 'running'].includes(runStatuses[runId] ?? '')) continue
      if (!candidate || latest.event_time > candidate.time) {
        candidate = { id: runId, time: latest.event_time }
      }
    }
    return candidate?.id ?? null
  }, [eventsByRun, runStatuses, selectedId])
  const activeRunId = liveRunId ?? lastRunId
  const activeRunStatus = activeRunId ? runStatuses[activeRunId] : undefined
  const isRunning = Boolean(activeRunId) &&
    (activeRunStatus === 'running' || activeRunStatus === 'pending')

  const tasksQuery = useQuery({
    queryKey: ['tasks', selectedId],
    queryFn: () => listTasks(selectedId!),
    enabled: selectedId !== null,
    refetchInterval: isRunning ? 1500 : 5000,
  })
  const conversationTasks = tasksQuery.data ?? []
  // 从持久化任务恢复待确认计划，刷新或切换会话后仍保留执行入口。
  const planTask = conversationTasks
    .filter((task) => task.owner_conversation_id === selectedId && task.status === 'pending')
    .sort((left, right) => right.updated_at.localeCompare(left.updated_at))[0] ?? null

  // ---------------------------------------------------------------- 长任务
  const measQuery = useQuery({
    queryKey: ['meas', selectedId],
    queryFn: () => listMeas(selectedId!, { limit: 10 }),
    enabled: selectedId !== null,
    refetchInterval: 15000,
  })
  const conversationMeas = useMemo(
    () => mergeMeas(
      measQuery.data ?? [],
      Object.values(meaById).filter((mea) => mea.conversation_id === selectedId),
    ),
    [measQuery.data, meaById, selectedId],
  )
  const panelMea = pickPanelMea(conversationMeas)
  const panelMeaId = panelMea?.id ?? null
  const meaBusy = panelMea !== null && isMeaBusy(panelMea.status)
  const meaWaiting = panelMea?.status === 'waiting_user'
  const lockedTaskIds = conversationMeas
    .filter((mea) => isMeaOpen(mea.status))
    .map((mea) => mea.task_id)

  const meaDetailQuery = useQuery({
    queryKey: ['mea', panelMeaId],
    queryFn: () => getMea(panelMeaId!),
    enabled: panelMeaId !== null,
  })
  const panelVersion = panelMeaId ? meaVersion[panelMeaId] ?? 0 : 0
  useEffect(() => {
    if (!panelMeaId || panelVersion === 0) return
    void queryClient.invalidateQueries({ queryKey: ['mea', panelMeaId] })
    void queryClient.invalidateQueries({ queryKey: ['tasks', selectedId] })
  }, [panelMeaId, panelVersion, selectedId, queryClient])

  const meaToolsQuery = useQuery({
    queryKey: ['mea-tools'],
    queryFn: () => listMeaTools(),
    staleTime: 60_000,
  })

  // 长任务的最终回复在对话之外追加到会话
  const appendedVersion = selectedId ? conversationVersion[selectedId] ?? 0 : 0
  useEffect(() => {
    if (!selectedId || appendedVersion === 0) return
    void queryClient.invalidateQueries({ queryKey: ['conversation', selectedId] })
    void queryClient.invalidateQueries({ queryKey: ['conversations'] })
    void queryClient.invalidateQueries({ queryKey: ['tasks', selectedId] })
  }, [selectedId, appendedVersion, queryClient])

  useEffect(() => {
    setMeaNotice(null)
  }, [selectedId])



  useEffect(() => {
    if (liveRunId) setLastRunId(liveRunId)
  }, [liveRunId])




  useEffect(() => {
    if (!selectedId) return
    let cancelled = false
    void listRuns({ conversationId: selectedId, limit: 50 })
      .then((runs) => {
        if (cancelled) return
        const map: Record<string, string> = {}
        for (const run of runs) map[run.id] = run.status
        useEventsStore.getState().syncRunStatuses(map)
        // 长任务的子 Run 由长任务面板展示，不作为会话的当前运行
        const persistedRunId = latestRunId(runs.filter((run) => !isMeaRun(run)))
        setLastRunId((current) => current ?? persistedRunId)
      })
      .catch(() => {})
    if (connected) {
      void queryClient.invalidateQueries({ queryKey: ['conversation', selectedId] })
      void queryClient.invalidateQueries({ queryKey: ['conversations'] })
      void queryClient.invalidateQueries({ queryKey: ['runs'] })
    }
    return () => {
      cancelled = true
    }
  }, [selectedId, connected, queryClient])


  const { conversationStatus, conversationActivity } = useMemo(() => {
    const map: Record<string, string> = {}
    const activity: Record<string, string> = {}
    const latestTimes: Record<string, string> = {}
    for (const [runId, events] of Object.entries(eventsByRun)) {
      const latest = events.at(-1)
      const conv = latest?.conversation_id
      if (!conv) continue
      if (latestTimes[conv] && latestTimes[conv] > latest.event_time) continue
      latestTimes[conv] = latest.event_time
      const status = runStatuses[runId]
      if (status) map[conv] = status
      const turn = buildTurnView(events, { now: Date.now() })
      if (turn.currentAction) activity[conv] = turn.currentAction
    }
    return { conversationStatus: map, conversationActivity: activity }
  }, [eventsByRun, runStatuses])

  const approvalsQuery = useQuery({
    queryKey: ['chat-approvals', activeRunId, meaBusy],
    queryFn: () => listApprovals('pending'),
    refetchInterval: 2000,
    enabled: activeRunId !== null || meaBusy,
  })
  // 长任务子 Run 的审批也在这里处理：它们属于同一个会话
  const pendingApproval =
    approvalsQuery.data?.find(
      (approval) => approval.run_id === activeRunId,
    ) ??
    (meaBusy
      ? approvalsQuery.data?.find((approval) => approval.conversation_id === selectedId)
      : undefined) ??
    null

  const artifactsQuery = useQuery({
    queryKey: ['chat-artifacts', activeRunId],
    queryFn: () =>
      activeRunId ? listArtifacts({ runId: activeRunId }) : Promise.resolve([]),
    refetchInterval: 3000,
    enabled: activeRunId !== null,
  })
  const artifacts = artifactsQuery.data ?? []

  const resolveApprovalMutation = useMutation({
    mutationFn: (action: { id: string; decision: 'approve' | 'deny' }) =>
      action.decision === 'approve'
        ? approveApproval(action.id)
        : denyApproval(action.id),
    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: ['chat-approvals'] })
      void queryClient.invalidateQueries({ queryKey: ['approvals'] })
    },
  })

  const newConversationMutation = useMutation({
    mutationFn: () => createConversation(),
    onSuccess: (conversation) => {
      selectConversation(conversation.id)
      setLastRunId(null)
      setPlanResolved(null)
      setLiveTurnActive(false)
      void queryClient.invalidateQueries({ queryKey: ['conversations'] })
    },
  })


  const renameConversationAction = async (id: string, title: string): Promise<void> => {
    try {
      await renameConversation(id, title)
      void queryClient.invalidateQueries({ queryKey: ['conversations'] })
      void queryClient.invalidateQueries({ queryKey: ['conversation', id] })
    } catch (error) {
      setSendError(error instanceof Error ? error.message : String(error))
    }
  }


  const deleteConversationAction = async (id: string): Promise<void> => {
    try {
      await deleteConversation(id)
      if (id === selectedId) {
        const next = conversations.find((c) => c.id !== id)?.id ?? null
        selectConversation(next)
        setLastRunId(null)
        setPlanResolved(null)
        setLiveTurnActive(false)
        setRunInspectorOpen(false)
      }
      void queryClient.invalidateQueries({ queryKey: ['conversations'] })
      void queryClient.invalidateQueries({ queryKey: ['conversation', id] })
    } catch (error) {
      setSendError(error instanceof Error ? error.message : String(error))
    }
  }

  const sendMutation = useMutation({
    mutationFn: ({
      conversationId,
      content,
      sendMode,
    }: {
      conversationId: string
      content: string
      sendMode: AgentMode
    }) => sendMessage(conversationId, content, sendMode),
    onSuccess: async (data, variables) => {
      if (selectedIdRef.current === variables.conversationId) {
        setLastRunId(data.run.id)
        setPlanResolved(null)
      }
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['conversation', variables.conversationId] }),
        queryClient.invalidateQueries({ queryKey: ['conversations'] }),
        queryClient.invalidateQueries({ queryKey: ['runs'] }),
        queryClient.invalidateQueries({ queryKey: ['chat-artifacts'] }),
        queryClient.invalidateQueries({ queryKey: ['tasks', variables.conversationId] }),
      ])
    },
    onError: (error: unknown, variables) => {
      if (selectedIdRef.current === variables.conversationId) {
        setSendError(error instanceof Error ? error.message : String(error))
        setLiveTurnActive(false)
      }
    },
    onSettled: (_data, _error, variables) => {
      setOptimisticMessage((current) => current?.conversationId === variables.conversationId ? null : current)
    },
  })



  const prevConnectedRef = useRef(connected)
  useEffect(() => {
    const reconnected = !prevConnectedRef.current && connected
    prevConnectedRef.current = connected
    if (reconnected && sendMutation.isPending) {
      setOptimisticMessage(null)
      sendMutation.reset()
    }
  }, [connected, sendMutation])

  const resolvePlanMutation = useMutation({
    mutationFn: (action: { taskId: string; conversationId: string; decision: 'accept' | 'reject' }) =>
      action.decision === 'accept'
        ? planAccept(action.taskId)
        : planReject(action.taskId),
    onSuccess: (task, variables) => {
      queryClient.setQueryData<Task[]>(['tasks', task.owner_conversation_id], (tasks) =>
        tasks?.map((existing) => existing.id === task.id ? task : existing),
      )
      void queryClient.invalidateQueries({ queryKey: ['tasks'] })
      if (variables.decision === 'accept') {
        draftStore.update(task.owner_conversation_id, { mode: 'normal' })
        refreshDraft()
        if (selectedIdRef.current === task.owner_conversation_id) setPlanResolved('计划已接受，正在普通执行…')
        sendMutation.mutate({
          conversationId: task.owner_conversation_id,
          content: '执行已接受的计划。按当前任务的步骤实施、验证并更新任务状态。已有计划无需重复规划；没有依赖的文件操作可在同一轮提交。修改前检查拟写入内容，避免无必要地重复整文件重写。验证命令保留失败退出码；已有有效验证结果可复用，不要为了更新状态重复验证。',
          sendMode: 'normal',
        })
      } else {
        if (selectedIdRef.current === task.owner_conversation_id) setPlanResolved('计划已拒绝')
      }
    },
    onError: (error: unknown, variables) => {
      if (selectedIdRef.current === variables.conversationId) setPlanResolved(error instanceof Error ? error.message : String(error))
    },
  })

  const refreshMea = (meaId?: string | null, conversationId = selectedId): void => {
    void queryClient.invalidateQueries({ queryKey: ['meas', conversationId] })
    if (meaId) void queryClient.invalidateQueries({ queryKey: ['mea', meaId] })
    void queryClient.invalidateQueries({ queryKey: ['tasks', conversationId] })
  }

  const startMeaMutation = useMutation({
    mutationFn: (input: {
      conversationId: string
      taskId: string
      roundBudget?: number
      extraTools?: string[]
      autoApproveSandbox?: boolean
    }) =>
      startMea(input.conversationId, input.taskId, {
        roundBudget: input.roundBudget,
        extraTools: input.extraTools,
        autoApproveSandbox: input.autoApproveSandbox,
      }),
    onSuccess: ({ mea, task }, variables) => {
      queryClient.setQueryData<Task[]>(['tasks', task.owner_conversation_id], (tasks) =>
        tasks?.map((existing) => existing.id === task.id ? task : existing),
      )
      draftStore.update(task.owner_conversation_id, { mode: 'normal' })
      refreshDraft()
      if (selectedIdRef.current === variables.conversationId) {
        setPlanResolved('计划已接受，长任务已启动：每一轮都会先执行、再由审计者独立验收。')
        setMeaNotice(null)
      }
      refreshMea(mea.id, variables.conversationId)
    },
    onError: (error: unknown, variables) => {
      if (selectedIdRef.current === variables.conversationId) setPlanResolved(error instanceof Error ? error.message : String(error))
    },
  })

  const describeAmend = (result: MeaAmendResult, verb: string): string =>
    result.accepted
      ? `${verb}已记录${result.amendment_id ? `为 ${result.amendment_id}` : ''}，长任务会在下一步读到。`
      : result.reason === 'mea_finalized'
        ? '长任务已经在收尾，这条补充没有被采纳。'
        : `没有采纳：${result.reason ?? '未知原因'}`

  const meaActionMutation = useMutation({
    mutationFn: async (
      action:
        | { type: 'answer'; meaId: string; text: string }
        | { type: 'note'; meaId: string; text: string; kind: MeaNoteKind; immediate: boolean }
        | { type: 'pause' | 'cancel'; meaId: string }
        | { type: 'resume'; meaId: string; extraRounds: number },
    ): Promise<string> => {
      switch (action.type) {
        case 'answer':
          return describeAmend(await answerMea(action.meaId, action.text), '回答')
        case 'note':
          return describeAmend(
            await noteMea(action.meaId, action.text, { kind: action.kind, immediate: action.immediate }),
            action.kind === 'once' ? '一次性指令' : '补充要求',
          )
        case 'pause':
          await pauseMea(action.meaId)
          return '已请求暂停，当前这一轮结束后停下。'
        case 'resume':
          await resumeMea(action.meaId, action.extraRounds)
          return action.extraRounds > 0 ? `已追加 ${action.extraRounds} 轮并继续。` : '已继续。'
        case 'cancel':
          await cancelMea(action.meaId)
          return '长任务已取消。'
      }
    },
    onSuccess: (notice, action) => {
      setMeaNotice(notice)
      refreshMea(action.meaId)
    },
    onError: (error: unknown) => {
      setMeaNotice(error instanceof Error ? error.message : String(error))
    },
  })

  const stopRun = async (): Promise<void> => {
    if (!activeRunId) return
    try {
      const updated = await cancelRun(activeRunId)

      useEventsStore
        .getState()
        .syncRunStatuses({ [activeRunId]: updated.status })
      void queryClient.invalidateQueries({ queryKey: ['runs'] })
      void queryClient.invalidateQueries({ queryKey: ['conversation', selectedId] })
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      const stateMatch = /cannot cancel run in state (\w+)/.exec(message)
      if (activeRunId && stateMatch) {


        useEventsStore.getState().syncRunStatuses({ [activeRunId]: stateMatch[1] })
        void queryClient.invalidateQueries({ queryKey: ['runs'] })
        void queryClient.invalidateQueries({ queryKey: ['conversation', selectedId] })
      } else {
        setSendError(message)
      }
    }
  }

  const recoverRunAction = async (): Promise<void> => {
    if (!activeRunId) return
    try {
      const result = await recoverRun(activeRunId)
      setLastRunId(result.run.id)
      void queryClient.invalidateQueries({ queryKey: ['runs'] })
      void queryClient.invalidateQueries({ queryKey: ['conversation', selectedId] })
    } catch (error) {
      setSendError(error instanceof Error ? error.message : String(error))
    }
  }


  const pauseRun = async (): Promise<void> => {
    if (!activeRunId) return
    try {
      const updated = await interruptRun(activeRunId)

      useEventsStore
        .getState()
        .syncRunStatuses({ [activeRunId]: updated.status })
      void queryClient.invalidateQueries({ queryKey: ['runs'] })
      void queryClient.invalidateQueries({ queryKey: ['conversation', selectedId] })
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      const stateMatch = /cannot interrupt run in state (\w+)/.exec(message)
      if (activeRunId && stateMatch) {

        useEventsStore.getState().syncRunStatuses({ [activeRunId]: stateMatch[1] })
        void queryClient.invalidateQueries({ queryKey: ['runs'] })
        void queryClient.invalidateQueries({ queryKey: ['conversation', selectedId] })
      } else {
        setSendError(message)
      }
    }
  }


  const composerCommands: ComposerCommand[] = [
    { id: 'new', label: '新建会话', icon: 'plus', onSelect: () => newConversationMutation.mutate() },
    {
      id: 'plan',
      label: mode === 'plan' ? '切换到普通模式' : '切换到规划模式',
      icon: 'check',
      onSelect: () => setMode(mode === 'plan' ? 'normal' : 'plan'),
    },
    {
      id: 'runs',
      label: '查看当前运行',
      icon: 'runs',
      onSelect: () => {
        if (activeRunId) onOpenRun?.(activeRunId)
      },
    },
    { id: 'stop', label: '停止运行', icon: 'close', onSelect: () => void stopRun() },
    { id: 'artifacts', label: '查看交付物', icon: 'artifacts', onSelect: () => onNavigate?.('ledger') },
    { id: 'settings', label: '打开设置', icon: 'settings', onSelect: () => onNavigate?.('settings') },
  ]

  const storedMessages = conversationQuery.data?.messages ?? []
  const messages =
    optimisticMessage?.conversationId === selectedId
      ? [...storedMessages, optimisticMessage.message]
      : storedMessages
  const showAgentTurn = liveTurnActive || liveRunId !== null

  const displayMessages =
    showAgentTurn && messages.at(-1)?.role === 'assistant'
      ? messages.slice(0, -1)
      : messages
  const selectedConversation = conversations.find((item) => item.id === selectedId)
  const showNewConversationHome = selectedId !== null && messages.length === 0
  const progressRunId = activeRunId
  const activeEvents = progressRunId ? (eventsByRun[progressRunId] ?? []) : []
  const latestModelStep = [...activeEvents]
    .reverse()
    .find((event) => event.type === 'model_started')?.step



  const turnView = buildTurnView(activeEvents, { now: Date.now() })
  const currentAction = turnView.currentAction
  const startedEvent = activeEvents.find((event) => event.type === 'agent_started')
  const startedAt = startedEvent ? Date.parse(startedEvent.event_time) : null
  const failedEvent = [...activeEvents]
    .reverse()
    .find((event) => event.type === 'agent_failed')
  const stopReason = failedEvent?.stop_reason ?? null





  const conversationScrollRef = useRef<HTMLDivElement>(null)
  const stickToBottomRef = useRef(true)
  const scrollFrameRef = useRef<number | null>(null)

  const handleConversationScroll = (): void => {
    const el = conversationScrollRef.current
    if (!el) return
    stickToBottomRef.current =
      el.scrollHeight - el.scrollTop - el.clientHeight < 120
  }

  const scheduleScroll = (): void => {
    if (scrollFrameRef.current !== null) return
    scrollFrameRef.current = requestAnimationFrame(() => {
      scrollFrameRef.current = null
      const el = conversationScrollRef.current
      if (!el) return
      if (messages.length === 0 && !showAgentTurn) {
        el.scrollTop = 0
        stickToBottomRef.current = true
      } else if (stickToBottomRef.current) {
        el.scrollTop = el.scrollHeight
      }
    })
  }



  const autoScrollKey = [
    messages.length,
    messages.at(-1)?.content?.length ?? 0,
    activeEvents.length,
    sendMutation.isPending,
    artifacts.length,
    pendingApproval?.id ?? null,
    planTask?.id ?? null,
    sendError,
    selectedId,
    showAgentTurn,
  ].join('|')
  useEffect(() => {
    scheduleScroll()
    return () => {
      if (scrollFrameRef.current !== null) {
        cancelAnimationFrame(scrollFrameRef.current)
        scrollFrameRef.current = null
      }
    }
  }, [autoScrollKey])

  return (
    <div className="chat-workspace">
      <aside
        className={`conversation-sidebar ${conversationSidebarOpen ? 'open' : 'collapsed'}`}
        aria-hidden={!conversationSidebarOpen}
      >
        <ConversationList
          conversations={conversations}
          selectedId={selectedId}
          statusByConversation={conversationStatus}
          activityByConversation={conversationActivity}
          onSelect={selectConversation}
          onNew={() => newConversationMutation.mutate()}
          onRename={(id, title) => void renameConversationAction(id, title)}
          onDelete={(id) => void deleteConversationAction(id)}
        />
      </aside>

      <div className="chat-right">
        <RunStatusBar
          title={selectedConversation?.title || '新会话'}
          conversationSidebarOpen={conversationSidebarOpen}
          onToggleConversationSidebar={() => setConversationSidebarOpen((open) => !open)}
          runStatus={activeRunStatus}
          step={latestModelStep ?? turnView.steps}
          toolCount={turnView.toolCount}
          totalTokens={turnView.usage?.totalTokens ?? null}
          durationMs={turnView.durationMs}
          startedAt={startedAt}
          currentAction={currentAction}
          stopReason={stopReason}
          mode={mode}
          turnState={activeEvents.length > 0 ? turnView.status : undefined}
          activityOpen={runInspectorOpen}
          onToggleActivity={() => setRunInspectorOpen((open) => !open)}
          onStop={() => void stopRun()}
          onRecover={() => void recoverRunAction()}
        />
        <CurrentTaskPanel
          tasks={conversationTasks}
          busy={sendMutation.isPending || startMeaMutation.isPending}
          lockedTaskIds={lockedTaskIds}
          onExecute={(task) => {
            setMode('normal')
            sendMutation.mutate({
              conversationId: task.owner_conversation_id,
              content: '继续执行当前任务。按剩余步骤逐项实施、验证并更新任务状态。',
              sendMode: 'normal',
            })
          }}
          onExecuteLong={(task) =>
            startMeaMutation.mutate({
              conversationId: task.owner_conversation_id,
              taskId: task.id,
            })
          }
        />
        {panelMea ? (
          <MeaPanel
            key={panelMea.id}
            mea={panelMea}
            detail={meaDetailQuery.data ?? null}
            live={meaLive[panelMea.id] ?? null}
            busy={meaActionMutation.isPending}
            notice={meaNotice}
            onAnswer={(text) =>
              meaActionMutation.mutate({ type: 'answer', meaId: panelMea.id, text })
            }
            onNote={(text, options) =>
              meaActionMutation.mutate({ type: 'note', meaId: panelMea.id, text, ...options })
            }
            onPause={() => meaActionMutation.mutate({ type: 'pause', meaId: panelMea.id })}
            onResume={(extraRounds) =>
              meaActionMutation.mutate({ type: 'resume', meaId: panelMea.id, extraRounds })
            }
            onCancel={() => meaActionMutation.mutate({ type: 'cancel', meaId: panelMea.id })}
          />
        ) : null}
        <div className="chat-right__body">
          <main className="conversation-main">
            <div
              ref={conversationScrollRef}
          onScroll={handleConversationScroll}
          className={`conversation-scroll ${showNewConversationHome ? 'conversation-scroll--empty' : ''}`}
        >
          <div className="message-thread">
            {selectedId === null || showNewConversationHome ? (
              <section className="desk-chat-empty">
                <span className="desk-chat-empty__index" aria-hidden="true">01 — WORK SESSION</span>
                <h1>{draft ? '工作说明已就位。' : '为这次工作写下第一条指令。'}</h1>
                <p>{draft
                  ? '检查下方的目标、范围与验收标准。发送后，Agent 才会开始工作。'
                  : '这里保存一项工作的对话、步骤和结果。你可以直接输入，也可以先整理完整的工作说明。'}</p>
                <div className="desk-chat-empty__guide"><span>明确目标</span><i aria-hidden="true">→</i><span>执行与确认</span><i aria-hidden="true">→</i><span>检查交付</span></div>
                {onNewWork ? <button type="button" className="desk-text-button" onClick={onNewWork}><Icon name="pencil" size={15} />准备一份工作说明</button>
                  : selectedId === null ? <button type="button" className="btn btn-primary" disabled={newConversationMutation.isPending} onClick={() => newConversationMutation.mutate()}>创建会话</button> : null}
              </section>
            ) : (
              <>
                {conversationQuery.data?.fork && conversationQuery.data.fork.conversation_id === selectedId ? (
                  <ForkBanner
                    fork={conversationQuery.data.fork}
                    sourceTitle={conversations.find((item) => item.id === conversationQuery.data?.fork?.source_conversation_id)?.title ?? null}
                    onOpenSource={(conversationId) => {
                      selectConversation(conversationId)
                      onConversationChange?.(conversationId)
                    }}
                  />
                ) : null}
                <MessageList messages={displayMessages} />
              </>
            )}

            {showAgentTurn ? (
              <LiveAgentTurn
                runId={progressRunId}
                step={latestModelStep ?? null}
                events={activeEvents}
                onRecover={() => void recoverRunAction()}
                onInspect={() => setRunInspectorOpen(true)}
              />
            ) : null}

            {sendError ? (
              <div className="inline-notice inline-notice--error">{sendError}</div>
            ) : null}

            {planTask && !isRunning && !sendMutation.isPending ? (
              <PlanCard
                task={planTask}
                busy={meaBusy || resolvePlanMutation.isPending || startMeaMutation.isPending}
                longTaskTools={meaToolsQuery.data ?? []}
                onAcceptLongTask={(taskId, options) =>
                  startMeaMutation.mutate({
                    conversationId: planTask.owner_conversation_id,
                    taskId,
                    roundBudget: options.roundBudget,
                    extraTools: options.extraTools,
                    autoApproveSandbox: options.autoApproveSandbox,
                  })
                }
                onAccept={(taskId) =>
                  resolvePlanMutation.mutate({ taskId, conversationId: planTask.owner_conversation_id, decision: 'accept' })
                }
                onReject={(taskId) =>
                  resolvePlanMutation.mutate({ taskId, conversationId: planTask.owner_conversation_id, decision: 'reject' })
                }
              />
            ) : null}
            {planResolved ? <div className="inline-notice">{planResolved}</div> : null}

            {pendingApproval ? (
              <ApprovalCard
                approval={pendingApproval}
                busy={resolveApprovalMutation.isPending}
                onApprove={(id) => resolveApprovalMutation.mutate({ id, decision: 'approve' })}
                onDeny={(id) => resolveApprovalMutation.mutate({ id, decision: 'deny' })}
              />
            ) : null}

            {artifacts.length > 0 ? (
              <section className="results-section">
                <SectionHeader title="交付结果" hint={`${artifacts.length} 项已发布`} />
                <div className="results-list">
                  {artifacts.map((artifact) => (
                    <ResultCard key={artifact.id} artifact={artifact} />
                  ))}
                </div>
              </section>
            ) : null}
          </div>
        </div>

        <Composer
          disabled={selectedId === null}
          sending={sendMutation.isPending}
          running={isRunning}
          onStop={() => void pauseRun()}
          mode={mode}
          onModeChange={setMode}
          value={draft}
          onValueChange={setDraft}
          commands={composerCommands}
          placeholder={
            meaWaiting
              ? '回答长任务的问题…'
              : meaBusy
                ? '长任务运行中：这里输入的内容会作为补充要求交给它'
                : undefined
          }
          onSend={async (content) => {
            if (!selectedId) return
            const conversationId = selectedId
            const sentDraft = { content: draft, mode }
            // Composer clears its input before awaiting onSend. Keep the durable
            // copy until the host confirms receipt, including after navigation.
            draftStore.set(conversationId, sentDraft)
            refreshDraft()
            setSendError(null)
            try {
              // 长任务推进或等待回答时，输入交给长任务；它已收尾则按普通新消息发送
              if (panelMea && (meaBusy || meaWaiting)) {
                let result: MeaAmendResult
                try {
                  result = meaWaiting
                    ? await answerMea(panelMea.id, content)
                    : await noteMea(panelMea.id, content)
                } catch (error) {
                  if (selectedIdRef.current === conversationId) setMeaNotice(error instanceof Error ? error.message : String(error))
                  throw error
                }
                if (result.accepted || result.reason !== 'mea_finalized') {
                  if (selectedIdRef.current === conversationId) setMeaNotice(describeAmend(result, meaWaiting ? '回答' : '补充要求'))
                  refreshMea(panelMea.id, conversationId)
                  if (result.accepted) draftStore.clearSent(conversationId, sentDraft)
                  refreshDraft()
                  return
                }
              }
              if (selectedIdRef.current === conversationId) {
                setLastRunId(null)
                setLiveTurnActive(true)
                setOptimisticMessage({
                  conversationId,
                  message: { role: 'user', content },
                })
              }

              requestAnimationFrame(() => {
                const el = conversationScrollRef.current
                if (el && selectedIdRef.current === conversationId) el.scrollTop = el.scrollHeight
              })
              await sendMutation.mutateAsync({
                conversationId,
                content,
                sendMode: mode,
              })
              draftStore.clearSent(conversationId, sentDraft)
              refreshDraft()
            } catch {
              // Mutation/MEA handlers report the error. The saved copy already
              // contains the input; avoid Composer restoring it into newer edits.
              refreshDraft()
            }
          }}
          />
          </main>
          {runInspectorOpen ? (
            <div className="chat-panels">
              <div className="activity-drawer">
                <RunActivity
                  runId={activeRunId}
                  onClose={() => setRunInspectorOpen(false)}
                  onStop={() => void stopRun()}
                  onRecover={() => void recoverRunAction()}
                  onOpenFullDetail={onOpenRun}
                />
              </div>
            </div>
          ) : null}
      </div>
      </div>
    </div>
  )
}
