

import type { AgentEvent, ConversationSummarySnapshot, Run } from '../api/types'

export interface ContextBreakdownItem {
  key: 'messages' | 'tool_schemas' | 'tool_results' | 'skills' | 'other'
  label: string
  tokens: number
  ratio: number
}

/** 输出过长、模型只收到截短版本的一次工具调用。 */
export interface TruncatedToolVM {
  toolCallId: string
  toolName: string
  /** 模型实际收到的（截短后的）输出 */
  modelOutput: string
  evidenceId: string | null
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
  /** 本步工具输出过长被截短的调用（来自实际的工具完成事件）。 */
  truncatedTools: TruncatedToolVM[]
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
  const truncatedByStep = collectTruncatedTools(events)
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
        truncatedTools: truncatedByStep.get(`${event.run_id}:${event.step}`) ?? [],
      }
    })
}

/** 按"运行:步骤"汇总被截短的工具输出；同一次工具调用只算一次。 */
export function collectTruncatedTools(events: AgentEvent[]): Map<string, TruncatedToolVM[]> {
  const seen = new Set<string>()
  const byStep = new Map<string, TruncatedToolVM[]>()
  for (const event of [...events].sort((a, b) => a.sequence - b.sequence)) {
    const result = event.tool_result
    if (event.type !== 'tool_completed' || event.step == null || !result?.output_truncated) continue
    const key = `${event.run_id}:${result.tool_call_id}`
    if (seen.has(key)) continue
    seen.add(key)
    const stepKey = `${event.run_id}:${event.step}`
    const list = byStep.get(stepKey) ?? []
    list.push({
      toolCallId: result.tool_call_id,
      toolName: result.tool_name,
      modelOutput: result.output ?? '',
      evidenceId: result.evidence_id ?? null,
    })
    byStep.set(stepKey, list)
  }
  return byStep
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
