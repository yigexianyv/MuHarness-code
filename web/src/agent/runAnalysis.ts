

import type { AgentEvent, ConversationSummarySnapshot, RequestToolView, Run } from '../api/types'

export interface ContextBreakdownItem {
  key: 'messages' | 'tool_schemas' | 'tool_results' | 'skills' | 'other'
  label: string
  tokens: number
  ratio: number
}

/**
 * 一次工具输出在上下文里发生的变化：
 * - shortened：本次请求里被上下文层截短（保留开头和结尾）；
 * - stored_truncated：执行器保存时截短，本次请求原样发出；
 * - omitted：之前在请求里，这一步起不再包含（已被摘要替代或移出请求）；
 * - stored_only：只知道执行器截短过，无法确认模型实际收到的版本（旧记录，或之后没有再请求模型）。
 */
export interface ToolNoticeVM {
  toolCallId: string
  toolName: string
  kind: 'shortened' | 'stored_truncated' | 'omitted' | 'stored_only'
  originalChars: number | null
  storedChars: number | null
  requestChars: number | null
  /** 用哪一步的请求记录查看"模型收到的内容"；null 表示没有请求记录 */
  viewStep: number | null
  /** stored_only 时执行器保存的版本 */
  storedOutput: string | null
}

export interface ContextStepVM {
  step: number
  eventTime: string
  originalInputTokens: number
  preparedInputTokens: number
  contextWindow: number
  inputBudget: number
  workingInputBudget: number
  triggerTokens: number
  targetTokens: number
  windowUsageRatio: number
  budgetUsageRatio: number
  messageTokensBefore: number
  messageTokensAfter: number
  toolSchemaTokens: number
  toolResultTokensBefore: number
  toolResultTokensAfter: number
  skillTokens: number
  compactionStage: string
  compactedToolResults: number
  removedToolRounds: number
  summaryUpdated: boolean
  summaryError: string | null
  summaryProvider: string | null
  summaryModel: string | null
  summaryDurationMs: number | null
  summaryUsage: number
  reachedTarget: boolean | null
  prefixDecision: string | null
  prefixRebuildReason: string | null
  localPrefixReused: boolean | null
  prefixMessageCount: number | null
  modelCompleted: boolean
  upstreamInputTokens: number | null
  upstreamCachedTokens: number | null
  upstreamCacheRatio: number | null
  breakdown: ContextBreakdownItem[]
  /** 本步请求前的原始消息总数（继承历史 + 本次运行）。 */
  sourceMessageCount: number | null
  /** 摘要覆盖水位：[0, coveredAfter) 的原始消息已由摘要替代。 */
  coveredBefore: number | null
  coveredAfter: number | null
  summarySnapshot: ConversationSummarySnapshot | null
  summaryPreviousSnapshot: ConversationSummarySnapshot | null
  constraintsPossiblyDropped: string[]
  /** 本步请求实际采用的"必须记住的事项"版本；null 表示未提供。 */
  constraintsRevision: number | null
  /** 本步请求里每次工具调用的三层长度；旧记录为 null。 */
  requestTools: RequestToolView[] | null
  /** 本步新出现的工具输出变化（同一次调用每种变化只提示一次）。 */
  toolNotices: ToolNoticeVM[]
}

export interface TraceGroupVM {
  id: string
  label: string
  events: AgentEvent[]
}

function numberOrZero(value: number | null | undefined): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function numberOrNull(value: number | null | undefined): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null
}


export function latestRunId(runs: Run[]): string | null {
  return runs.reduce<Run | null>((latest, run) => {
    if (!latest || run.created_at > latest.created_at) return run
    return latest
  }, null)?.id ?? null
}

/** 合并持久化和实时事件，按事件 ID 去重并恢复时间顺序。 */
export function mergeRunEvents(
  durable: AgentEvent[],
  live: AgentEvent[],
): AgentEvent[] {
  const byId = new Map<string, AgentEvent>()
  for (const event of [...durable, ...live]) byId.set(event.event_id, event)
  return [...byId.values()].sort((a, b) => {
    if (a.sequence !== b.sequence) return a.sequence - b.sequence
    return a.event_time.localeCompare(b.event_time)
  })
}

/** 从模型启动事件提取每一步的上下文用量信息。 */
export function buildContextSteps(events: AgentEvent[]): ContextStepVM[] {
  return attachToolNotices(buildContextStepsWithoutNotices(events), events)
}

function buildContextStepsWithoutNotices(events: AgentEvent[]): ContextStepVM[] {
  const pendingByStep = new Map<string, AgentEvent>()
  const completionByStart = new Map<AgentEvent, AgentEvent>()
  for (const event of [...events].sort((a, b) => a.sequence - b.sequence)) {
    if (event.step == null) continue
    const key = `${event.run_id}:${event.step}`
    if (event.type === 'model_started') {
      pendingByStep.set(key, event)
    } else if (event.type === 'model_completed') {
      const started = pendingByStep.get(key)
      if (started && event.sequence >= started.sequence) {
        completionByStart.set(started, event)
        pendingByStep.delete(key)
      }
    }
  }
  return events
    .filter((event) => event.type === 'model_started' && event.step != null)
    .map((event) => {
      const prepared = numberOrZero(
        event.prepared_input_tokens ?? event.estimated_input_tokens,
      )
      const original = numberOrZero(
        event.original_estimated_input_tokens ?? prepared,
      )
      const contextWindow = numberOrZero(event.context_window)
      const workingInputBudget = numberOrZero(event.working_input_budget)
      const allMessageTokens = numberOrZero(event.message_tokens_after)
      const schemas = numberOrZero(event.tool_schema_tokens)
      const toolResults = numberOrZero(event.tool_result_tokens_after)
      const skills = numberOrZero(event.skill_catalog_tokens)
        + numberOrZero(event.active_skill_tokens)
      const matchingCompletion = completionByStart.get(event) ?? null
      const upstreamInput = numberOrNull(matchingCompletion?.usage?.input_tokens)
      const upstreamCached = numberOrNull(matchingCompletion?.usage?.cached_input_tokens)

      const messages = Math.max(0, allMessageTokens - toolResults - skills)
      const known = messages + schemas + toolResults + skills
      const other = Math.max(0, prepared - known)
      const rawBreakdown = [
        { key: 'messages' as const, label: 'Messages & injected', tokens: messages },
        { key: 'tool_schemas' as const, label: 'Tool schemas', tokens: schemas },
        { key: 'tool_results' as const, label: 'Tool results', tokens: toolResults },
        { key: 'skills' as const, label: 'Skills', tokens: skills },
        { key: 'other' as const, label: 'Request overhead', tokens: other },
      ]
      return {
        step: event.step!,
        eventTime: event.event_time,
        originalInputTokens: original,
        preparedInputTokens: prepared,
        contextWindow,
        inputBudget: numberOrZero(event.input_budget),
        workingInputBudget,
        triggerTokens: numberOrZero(event.trigger_tokens),
        targetTokens: numberOrZero(event.target_tokens),
        windowUsageRatio: contextWindow > 0 ? prepared / contextWindow : 0,
        budgetUsageRatio: workingInputBudget > 0
          ? prepared / workingInputBudget
          : (event.prepared_usage_ratio ?? event.usage_ratio ?? 0),
        messageTokensBefore: numberOrZero(event.message_tokens_before),
        messageTokensAfter: allMessageTokens,
        toolSchemaTokens: schemas,
        toolResultTokensBefore: numberOrZero(event.tool_result_tokens_before),
        toolResultTokensAfter: toolResults,
        skillTokens: skills,
        compactionStage: event.compaction_stage ?? 'none',
        compactedToolResults: numberOrZero(event.compacted_tool_results),
        removedToolRounds: numberOrZero(event.removed_tool_rounds),
        summaryUpdated: event.summary_updated === true,
        summaryError: event.summary_error ?? null,
        summaryProvider: event.summary_provider ?? null,
        summaryModel: event.summary_model ?? null,
        summaryDurationMs: event.summary_duration_ms ?? null,
        summaryUsage: numberOrZero(event.summary_usage?.total_tokens),
        reachedTarget: event.reached_target ?? null,
        prefixDecision: event.prefix_decision ?? null,
        prefixRebuildReason: event.prefix_rebuild_reason ?? null,
        localPrefixReused: event.cache_prefix_reused ?? null,
        prefixMessageCount: numberOrNull(event.cache_prefix_message_count),
        modelCompleted: matchingCompletion !== null,
        upstreamInputTokens: upstreamInput,
        upstreamCachedTokens: upstreamCached,
        upstreamCacheRatio: upstreamCached !== null && upstreamInput !== null
          && upstreamInput > 0 && upstreamCached <= upstreamInput
          ? Math.min(1, upstreamCached / upstreamInput)
          : null,
        breakdown: rawBreakdown.map((item) => ({
          ...item,
          ratio: prepared > 0 ? item.tokens / prepared : 0,
        })),
        sourceMessageCount: numberOrNull(event.source_message_count),
        coveredBefore: numberOrNull(event.summary_covered_before),
        coveredAfter: numberOrNull(event.summary_covered_after),
        summarySnapshot: event.summary_snapshot ?? null,
        summaryPreviousSnapshot: event.summary_previous_snapshot ?? null,
        constraintsPossiblyDropped: event.constraints_possibly_dropped ?? [],
        constraintsRevision: numberOrNull(event.constraints_revision),
        requestTools: Array.isArray(event.request_tool_views) ? event.request_tool_views : null,
        toolNotices: [],
      }
    })
}

/** 根据每步请求记录，标出工具输出第一次被截短、第一次不再被包含的步骤。 */
function attachToolNotices(steps: ContextStepVM[], events: AgentEvent[]): ContextStepVM[] {
  const notified = new Set<string>()
  const included = new Set<string>()
  const requested = new Set<string>()
  // 同一步可能有多次请求（重试），按位置而不是步骤号回填
  const order = steps.map((step, index) => ({ step, index })).sort((a, b) => a.step.step - b.step.step || a.index - b.index)
  const withNotices: ContextStepVM[] = new Array(steps.length)
  for (const { step, index } of order) {
    const notices: ToolNoticeVM[] = []
    for (const view of step.requestTools ?? []) {
      requested.add(view.tool_call_id)
      const base = {
        toolCallId: view.tool_call_id,
        toolName: view.tool_name ?? '工具',
        originalChars: view.original_chars,
        storedChars: view.stored_chars,
        requestChars: view.request_chars ?? null,
        viewStep: step.step,
        storedOutput: null,
      }
      if (view.included) {
        included.add(view.tool_call_id)
        const kind = view.request_shortened ? 'shortened' : view.stored_truncated ? 'stored_truncated' : null
        if (kind && !notified.has(`${view.tool_call_id}:changed`)) {
          notified.add(`${view.tool_call_id}:changed`)
          notices.push({ ...base, kind })
        }
      } else if (included.has(view.tool_call_id) && !notified.has(`${view.tool_call_id}:omitted`)) {
        notified.add(`${view.tool_call_id}:omitted`)
        notices.push({ ...base, kind: 'omitted', requestChars: null, viewStep: null })
      }
    }
    withNotices[index] = { ...step, toolNotices: notices }
  }
  // 执行器截短过、但没有任何请求记录的调用（旧记录，或之后没有再请求模型）
  const byStep = new Map<number, ContextStepVM>()
  for (const step of withNotices) if (!byStep.has(step.step)) byStep.set(step.step, step)
  const seen = new Set<string>()
  for (const event of [...events].sort((a, b) => a.sequence - b.sequence)) {
    const result = event.tool_result
    if (event.type !== 'tool_completed' || event.step == null || !result?.output_truncated) continue
    if (requested.has(result.tool_call_id) || seen.has(result.tool_call_id)) continue
    seen.add(result.tool_call_id)
    byStep.get(event.step)?.toolNotices.push({
      toolCallId: result.tool_call_id,
      toolName: result.tool_name,
      kind: 'stored_only',
      originalChars: null,
      storedChars: result.output?.length ?? null,
      requestChars: null,
      viewStep: null,
      storedOutput: result.output ?? '',
    })
  }
  return withNotices
}

export type SummaryFieldKey = keyof ConversationSummarySnapshot

export const SUMMARY_FIELDS: { key: SummaryFieldKey; label: string }[] = [
  { key: 'current_objective', label: '目标' },
  { key: 'user_constraints', label: '用户约束' },
  { key: 'key_decisions', label: '关键决策' },
  { key: 'completed_work', label: '已完成' },
  { key: 'current_state', label: '当前状态' },
  { key: 'pending_work', label: '待办' },
  { key: 'important_facts', label: '重要事实' },
]

export function summaryEntries(
  snapshot: ConversationSummarySnapshot | null,
  key: SummaryFieldKey,
): string[] {
  if (!snapshot) return []
  const value = snapshot[key]
  if (Array.isArray(value)) return value
  return typeof value === 'string' && value ? [value] : []
}

/** 所选步骤生效的摘要：本步或之前最近一次附带的快照。 */
export function effectiveSummary(
  steps: ContextStepVM[],
  step: number,
): { snapshot: ConversationSummarySnapshot; fromStep: number } | null {
  for (let index = steps.length - 1; index >= 0; index -= 1) {
    const candidate = steps[index]
    if (candidate.step > step || !candidate.summarySnapshot) continue
    return { snapshot: candidate.summarySnapshot, fromStep: candidate.step }
  }
  return null
}

export interface SummaryFieldDiff {
  key: SummaryFieldKey
  label: string
  kept: string[]
  added: string[]
  removed: string[]
}

/** 按字段对比两版摘要；文字有任何改写都会显示为一删一增。 */
export function diffSummary(
  previous: ConversationSummarySnapshot | null,
  next: ConversationSummarySnapshot | null,
): SummaryFieldDiff[] {
  return SUMMARY_FIELDS.map(({ key, label }) => {
    const before = summaryEntries(previous, key)
    const after = summaryEntries(next, key)
    return {
      key,
      label,
      kept: after.filter((entry) => before.includes(entry)),
      added: after.filter((entry) => !before.includes(entry)),
      removed: before.filter((entry) => !after.includes(entry)),
    }
  })
}

/** 本步新被摘要替代的原始消息范围 [from, to)；没有新替代时返回 null。 */
export function newlyCoveredRange(step: ContextStepVM): { from: number; to: number } | null {
  if (step.coveredBefore === null || step.coveredAfter === null) return null
  if (step.coveredAfter <= step.coveredBefore) return null
  return { from: step.coveredBefore, to: step.coveredAfter }
}

/** 将事件序列整理为可展开的运行轨迹分组。 */
export function buildTraceGroups(events: AgentEvent[]): TraceGroupVM[] {
  const groups = new Map<string, AgentEvent[]>()
  for (const event of events) {
    const key = event.step == null ? 'run' : `step-${event.step}`
    const list = groups.get(key) ?? []
    list.push(event)
    groups.set(key, list)
  }
  return [...groups.entries()].map(([id, grouped]) => ({
    id,
    label: id === 'run' ? 'Run lifecycle' : `Step ${id.slice(5)}`,
    events: grouped,
  }))
}
