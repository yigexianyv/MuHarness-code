import type { Artifact } from '../api/artifacts'
import type { Conversation, MeaRun, Run } from '../api/types'
import type { InboxItem } from './hub'
import { isMeaRun, MEA_STATUS_LABEL, meaReason } from './meaPresentation'

export type WorkLane = 'ready' | 'running' | 'attention' | 'finished'
export type WorkKind = 'conversation' | 'run' | 'mea'
export type WorkFilter = 'all' | 'normal' | 'long'

/** A work item represents a conversation, not each internal agent invocation. */
export interface WorkItem {
  id: string
  title: string
  kind: WorkKind
  lane: WorkLane
  status: string
  detail: string
  updatedAt: string
  conversationId: string | null
  run: Run | null
  mea: MeaRun | null
  decisions: InboxItem[]
  artifacts: Artifact[]
}

export const WORK_LANES: Array<{ id: WorkLane; title: string; hint: string; empty: string }> = [
  { id: 'ready', title: '待开始', hint: '会话已建，近期还没有执行', empty: '新建工作后，从这里开始。' },
  { id: 'running', title: '执行中', hint: '正在推进或等待调度', empty: '当前列表没有执行中的工作。' },
  { id: 'attention', title: '待你决定', hint: '审批、回答或恢复', empty: '当前列表没有等待你决定的工作。' },
  { id: 'finished', title: '近期结束', hint: '查看结果与验收状态', empty: '执行结束后，在这里回看结果。' },
]

export function inboxConversationId(item: InboxItem, runs: Run[]): string | null {
  if (item.kind === 'mea') return item.mea.conversation_id
  if (item.kind === 'run') return item.run.conversation_id
  return item.approval.conversation_id
    ?? runs.find((run) => run.id === item.approval.run_id)?.conversation_id
    ?? null
}

export function decisionTitle(item: InboxItem): string {
  if (item.kind === 'approval') return `批准 ${item.approval.tool_name}？`
  if (item.kind === 'mea') return item.reason
  return item.run.error || '执行意外中断，是否从检查点继续？'
}

function laneForRun(run: Run, decisions: InboxItem[], recovered: Set<string>): WorkLane {
  if (decisions.length > 0) return 'attention'
  if (run.status === 'pending' || run.status === 'running') return 'running'
  if (run.status === 'interrupted' && !recovered.has(run.id)) return 'attention'
  return 'finished'
}

function laneForMea(mea: MeaRun, decisions: InboxItem[]): WorkLane {
  if (decisions.length > 0 || ['waiting_user', 'paused', 'blocked'].includes(mea.status)) return 'attention'
  if (mea.status === 'running' || mea.status === 'finalizing') return 'running'
  return 'finished'
}

function runStatus(run: Run, recovered: Set<string>): string {
  switch (run.status) {
    case 'completed': return '执行结束 · 未独立审计'
    case 'failed': return '执行失败'
    case 'cancelled': return '已取消'
    case 'interrupted': return recovered.has(run.id) ? '中断已恢复' : '中断 · 可恢复'
    case 'pending': return '等待执行'
    default: return '正在执行'
  }
}

function meaStatus(mea: MeaRun): string {
  if (mea.status !== 'completed') return MEA_STATUS_LABEL[mea.status]
  const decision = mea.completion_decision
  const confirmed = decision?.outcome === 'completed'
    && Boolean(decision.auditor_run_id)
    && decision.requirements_revision === mea.requirements_revision
  return confirmed ? '最终验收通过' : '已完成 · 验收信息未确认'
}

const LANE_PRIORITY: Record<WorkLane, number> = { attention: 0, running: 1, ready: 2, finished: 3 }

/**
 * The API supplies bounded recent lists. We never infer all-time counts or parent
 * MEA completion from an executor/auditor child run's status.
 */
export function buildWorkOverview({ conversations, runs, meas, inbox, artifacts }: {
  conversations: Conversation[]
  runs: Run[]
  meas: MeaRun[]
  inbox: InboxItem[]
  artifacts: Artifact[]
}): WorkItem[] {
  const conversationsById = new Map(conversations.map((conversation) => [conversation.id, conversation]))
  const recovered = new Set(runs.flatMap((run) => run.recovered_from_run_id ? [run.recovered_from_run_id] : []))
  const workByConversation = new Map<string, WorkItem>()
  const candidates: WorkItem[] = []

  for (const mea of meas) {
    const decisions = inbox.filter((item) =>
      item.kind === 'mea' ? item.mea.id === mea.id
        : item.kind === 'approval' && (
          runs.some((run) => run.id === item.approval.run_id && isMeaRun(run) && run.source_id === mea.id)
          || (!item.approval.run_id && item.approval.conversation_id === mea.conversation_id)
        ),
    )
    candidates.push({
      id: `mea:${mea.id}`,
      title: conversationsById.get(mea.conversation_id)?.title || `长任务 ${mea.id.slice(0, 8)}`,
      kind: 'mea', lane: laneForMea(mea, decisions), status: meaStatus(mea),
      detail: mea.pending_question || meaReason(mea) || mea.final_response || '管理、执行与审计协同推进。',
      updatedAt: mea.updated_at, conversationId: mea.conversation_id, run: null, mea, decisions,
      artifacts: artifacts.filter((artifact) => artifact.task_id === mea.task_id || artifact.conversation_id === mea.conversation_id),
    })
  }

  for (const run of runs) {
    // A completed child is a single role's output, never a completed user goal.
    if (isMeaRun(run) || ['manage', 'execute', 'audit'].includes(run.mode)) continue
    const decisions = inbox.filter((item) => item.kind === 'run'
      ? item.run.id === run.id && !recovered.has(run.id)
      : item.kind === 'approval' && (
        item.approval.run_id === run.id
        || (!item.approval.run_id && Boolean(run.conversation_id) && item.approval.conversation_id === run.conversation_id)
      ))
    candidates.push({
      id: `run:${run.id}`,
      title: (run.conversation_id ? conversationsById.get(run.conversation_id)?.title : null) || run.user_message || '未命名执行',
      kind: 'run', lane: laneForRun(run, decisions, recovered),
      status: decisions.some((item) => item.kind === 'approval') ? '等待工具审批' : runStatus(run, recovered),
      detail: run.error || run.user_message || '打开执行记录查看详细过程。',
      updatedAt: run.updated_at, conversationId: run.conversation_id, run, mea: null, decisions,
      artifacts: artifacts.filter((artifact) => artifact.run_id === run.id || (
        Boolean(run.conversation_id) && artifact.conversation_id === run.conversation_id
      )),
    })
  }

  // An unresolved decision has priority over newer historical completions.
  candidates.sort((a, b) => LANE_PRIORITY[a.lane] - LANE_PRIORITY[b.lane]
    || b.updatedAt.localeCompare(a.updatedAt) || a.id.localeCompare(b.id))
  for (const candidate of candidates) {
    const key = candidate.conversationId ? `conversation:${candidate.conversationId}` : candidate.id
    if (!workByConversation.has(key)) workByConversation.set(key, candidate)
  }

  for (const conversation of conversations) {
    const key = `conversation:${conversation.id}`
    if (workByConversation.has(key)) continue
    // A recent child run with an unavailable parent must not masquerade as an empty conversation.
    if (runs.some((run) => run.conversation_id === conversation.id && (isMeaRun(run) || ['manage', 'execute', 'audit'].includes(run.mode)))) continue
    const decisions = inbox.filter((item) => inboxConversationId(item, runs) === conversation.id)
    workByConversation.set(key, {
      id: key, title: conversation.title || '未命名会话', kind: 'conversation',
      lane: decisions.length > 0 ? 'attention' : 'ready',
      status: conversation.message_count === 0 ? '等待输入目标' : '近期无执行记录',
      detail: conversation.message_count === 0 ? '写下目标，选择执行方式，再开始工作。' : '会话已有消息；当前查询范围内没有执行记录。',
      updatedAt: conversation.updated_at, conversationId: conversation.id, run: null, mea: null, decisions,
      artifacts: artifacts.filter((artifact) => artifact.conversation_id === conversation.id),
    })
  }

  return [...workByConversation.values()].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt) || a.id.localeCompare(b.id))
}

export function filterWorkOverview(items: WorkItem[], filter: WorkFilter, query: string): WorkItem[] {
  const needle = query.trim().toLocaleLowerCase()
  return items.filter((item) => (filter === 'all' || (filter === 'long' ? item.kind === 'mea' : item.kind !== 'mea'))
    && (!needle || `${item.title}\n${item.detail}`.toLocaleLowerCase().includes(needle)))
}
