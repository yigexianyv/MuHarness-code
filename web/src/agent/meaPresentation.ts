import type {
  AgentEvent,
  MeaLiveActivity,
  MeaRole,
  MeaRound,
  MeaRoundKind,
  MeaRoundPhase,
  MeaRoundSummary,
  MeaRun,
  MeaStatus,
  Run,
  TaskStep,
} from '../api/types'

export const MEA_STATUS_LABEL: Record<MeaStatus, string> = {
  running: '运行中',
  waiting_user: '等待你回答',
  paused: '已暂停',
  finalizing: '收尾中',
  completed: '已完成',
  blocked: '已阻塞',
  failed: '失败',
  cancelled: '已取消',
}

export const ROUND_KIND_LABEL: Record<MeaRoundKind, string> = {
  normal: '执行',
  audit_only: '仅审计',
  final_audit: '最终验收',
  recovery_audit: '中断核查',
}

export const ROUND_PHASE_LABEL: Record<MeaRoundPhase, string> = {
  managing: '规划中',
  planned: '已规划',
  executing: '执行中',
  executed: '待审计',
  auditing: '审计中',
  audited: '已审计',
  applied: '已完成',
  interrupted: '已中断',
  abandoned: '已作废',
}

export const ROLE_LABEL: Record<MeaRole, string> = {
  manager: '管理者',
  executor: '执行者',
  auditor: '审计者',
}

const OPEN_STATUSES: ReadonlySet<MeaStatus> = new Set([
  'running',
  'waiting_user',
  'paused',
  'finalizing',
])
const BUSY_STATUSES: ReadonlySet<MeaStatus> = new Set(['running', 'finalizing'])

/** 还没结束（含等待回答、暂停和收尾）。 */
export function isMeaOpen(status: MeaStatus): boolean {
  return OPEN_STATUSES.has(status)
}

/** 正在推进：这时会话不接受普通消息，输入框的内容作为补充交给长任务。 */
export function isMeaBusy(status: MeaStatus): boolean {
  return BUSY_STATUSES.has(status)
}

/** 面板显示哪个长任务：优先最近的未结束任务，否则最近的一个。 */
export function pickPanelMea(meas: MeaRun[]): MeaRun | null {
  const sorted = [...meas].sort((a, b) => b.created_at.localeCompare(a.created_at))
  return sorted.find((mea) => isMeaOpen(mea.status)) ?? sorted[0] ?? null
}

/** 合并列表查询和推送：同一个长任务取 updated_at 更新的那份。 */
export function mergeMeas(listed: MeaRun[], pushed: MeaRun[]): MeaRun[] {
  const byId = new Map<string, MeaRun>()
  for (const mea of [...listed, ...pushed]) {
    const current = byId.get(mea.id)
    if (!current || mea.updated_at >= current.updated_at) byId.set(mea.id, mea)
  }
  return [...byId.values()]
}

/** 计入轮次预算的轮数（作废的轮不算）。 */
export function countedRounds(rounds: Array<Pick<MeaRound, 'phase'>>): number {
  return rounds.filter((round) => round.phase !== 'abandoned').length
}

export function stepProgress(steps: TaskStep[]): { done: number; total: number; percent: number } {
  const active = steps.filter((step) => step.status !== 'superseded')
  const done = active.filter((step) => step.status === 'done').length
  const total = active.length
  return { done, total, percent: total > 0 ? Math.round((done / total) * 100) : 0 }
}

export function stepSymbol(status: TaskStep['status']): string {
  switch (status) {
    case 'done':
      return '✓'
    case 'in_progress':
      return '→'
    case 'blocked':
      return '!'
    case 'superseded':
      return '×'
    default:
      return '·'
  }
}

export type VerdictTone = 'ok' | 'bad' | 'warn' | 'muted' | 'pending'

export interface RoundVerdict {
  label: string
  tone: VerdictTone
}

type VerdictInput = Pick<
  MeaRoundSummary,
  | 'kind'
  | 'phase'
  | 'route'
  | 'audit_status'
  | 'integrity_status'
  | 'contract_audit_status'
  | 'step_acceptance'
  | 'stale_requirements'
  | 'interrupt_reason'
> & Partial<Pick<MeaRoundSummary, 'audit_output_truncated'>>

/** 一轮的结论：代码按审计控制头判定，这里只是把判定翻译成一句话。 */
export function roundVerdict(round: VerdictInput): RoundVerdict {
  if (round.phase === 'abandoned') return { label: '计划作废', tone: 'muted' }
  if (round.phase === 'interrupted') return { label: '已中断', tone: 'warn' }
  if (round.route === 'ask') return { label: '请示用户', tone: 'warn' }
  if (round.route === 'blocked') return { label: '判定阻塞', tone: 'bad' }
  if (round.route === 'invalid') return { label: '计划无效', tone: 'bad' }
  if (round.phase !== 'applied') {
    return { label: ROUND_PHASE_LABEL[round.phase], tone: 'pending' }
  }
  if (round.interrupt_reason) return { label: '已中断', tone: 'warn' }
  if (round.audit_output_truncated) return { label: '审计输出被截断', tone: 'warn' }
  if (round.stale_requirements) return { label: '要求已更新，结论作废', tone: 'warn' }
  const clean = round.integrity_status === 'clean' && round.contract_audit_status === 'aligned'
  if (round.kind === 'final_audit') {
    return clean && round.audit_status === 'complete'
      ? { label: '最终验收通过', tone: 'ok' }
      : { label: '最终验收未通过', tone: 'bad' }
  }
  if (!round.audit_status) return { label: ROUND_PHASE_LABEL[round.phase], tone: 'muted' }
  return clean && round.step_acceptance === 'satisfied'
    ? { label: '步骤通过', tone: 'ok' }
    : { label: '步骤未通过', tone: 'bad' }
}

/** 暂停、失败等状态的原因说明。 */
export function meaReason(mea: Pick<MeaRun, 'status' | 'abort_reason' | 'round_budget'>): string | null {
  if (mea.abort_reason === 'max_rounds') {
    return `${mea.round_budget} 轮预算已用完，暂停在当前进度。可以追加轮次后继续。`
  }
  return mea.abort_reason
}

const LIVE_EVENT_TYPES: ReadonlySet<string> = new Set([
  'agent_started',
  'model_started',
  'tool_started',
  'tool_completed',
  'tool_approval_required',
  'tool_approval_completed',
  'run_budget_finalizing',
  'agent_completed',
  'agent_failed',
  'agent_cancelled',
])

/** 从 mea.agent_event 推送提取“谁在做什么”；流式增量等细碎事件返回 null。 */
export function liveActivityFrom(
  role: string | null,
  event: AgentEvent,
): MeaLiveActivity | null {
  if (!LIVE_EVENT_TYPES.has(event.type)) return null
  const knownRole = role === 'manager' || role === 'executor' || role === 'auditor' ? role : null
  return {
    role: knownRole,
    runId: event.run_id,
    type: event.type,
    toolName: event.tool_call?.name ?? event.tool_result?.tool_name ?? null,
    eventTime: event.event_time,
  }
}

export function liveActivityText(activity: MeaLiveActivity | null): string | null {
  if (!activity) return null
  const who = activity.role ? ROLE_LABEL[activity.role] : '子任务'
  switch (activity.type) {
    case 'tool_started':
      return `${who}正在调用 ${activity.toolName ?? '工具'}`
    case 'tool_approval_required':
      return `${who}在等你批准 ${activity.toolName ?? '工具调用'}`
    case 'tool_completed':
    case 'tool_approval_completed':
    case 'model_started':
    case 'agent_started':
      return `${who}正在思考`
    case 'run_budget_finalizing':
      return `${who}快用完步数，正在收尾`
    case 'agent_completed':
      return `${who}已交付`
    case 'agent_failed':
      return `${who}运行出错`
    case 'agent_cancelled':
      return `${who}已停止`
    default:
      return null
  }
}

// ---------------------------------------------------------------- 运行列表分组

export function isMeaRun(run: Pick<Run, 'source'>): boolean {
  return (run.source ?? '').startsWith('mea:')
}

export function meaRunRole(run: Pick<Run, 'source'>): MeaRole | null {
  const role = (run.source ?? '').slice('mea:'.length)
  return role === 'manager' || role === 'executor' || role === 'auditor' ? role : null
}

export type RunListEntry =
  | { kind: 'run'; run: Run }
  | {
      kind: 'mea'
      meaId: string
      runs: Run[]
      status: Run['status']
      createdAt: string
      latestAt: string
      roleCounts: Record<MeaRole, number>
    }

/** 长任务的子 Run 按 source_id 合成一组，其余 Run 原样保留；顺序由调用方决定。 */
export function groupRuns(runs: Run[]): RunListEntry[] {
  const entries: RunListEntry[] = []
  const groups = new Map<string, Run[]>()
  for (const run of runs) {
    if (isMeaRun(run) && run.source_id) {
      const list = groups.get(run.source_id)
      if (list) list.push(run)
      else {
        const created: Run[] = [run]
        groups.set(run.source_id, created)
        entries.push({
          kind: 'mea',
          meaId: run.source_id,
          runs: created,
          status: run.status,
          createdAt: run.created_at,
          latestAt: run.created_at,
          roleCounts: { manager: 0, executor: 0, auditor: 0 },
        })
      }
    } else {
      entries.push({ kind: 'run', run })
    }
  }
  return entries.map((entry) => {
    if (entry.kind !== 'mea') return entry
    const children = [...entry.runs].sort((a, b) => a.created_at.localeCompare(b.created_at))
    const roleCounts: Record<MeaRole, number> = { manager: 0, executor: 0, auditor: 0 }
    for (const child of children) {
      const role = meaRunRole(child)
      if (role) roleCounts[role] += 1
    }
    // 有子 Run 在跑就算运行中；否则看最近一个子 Run（早先被取消的审计之类不影响整体）
    const active = children.find((child) => child.status === 'running' || child.status === 'pending')
    const status = active ? active.status : children[children.length - 1].status
    return {
      ...entry,
      runs: children,
      status,
      createdAt: children[0].created_at,
      latestAt: children[children.length - 1].created_at,
      roleCounts,
    }
  })
}

export function entryStatus(entry: RunListEntry): Run['status'] {
  return entry.kind === 'run' ? entry.run.status : entry.status
}

export function entryTime(entry: RunListEntry): string {
  return entry.kind === 'run' ? entry.run.created_at : entry.latestAt
}
