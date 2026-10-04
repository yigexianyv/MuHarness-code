

import { describe, expect, it } from 'vitest'

import type { AgentEvent, ConversationSummarySnapshot, Run } from '../api/types'
import {
  buildContextSteps,
  buildTraceGroups,
  diffSummary,
  effectiveSummary,
  latestRunId,
  mergeRunEvents,
  newlyCoveredRange,
} from './runAnalysis'

function event(partial: Partial<AgentEvent>): AgentEvent {
  return {
    event_id: 'e1',
    run_id: 'r1',
    conversation_id: 'c1',
    sequence: 1,
    type: 'model_started',
    event_time: '2026-08-22T00:00:00Z',
    step: 1,
    provider: 'fake',
    model: 'fake',
    message: null,
    tool_call: null,
    tool_result: null,
    usage: null,
    stop_reason: null,
    approval_decision: null,
    ...partial,
  }
}

describe('buildContextSteps', () => {
  it('按 Model Step 构建输入变化、窗口占比与不重复计数的 breakdown', () => {
    const [step] = buildContextSteps([
      event({
        original_estimated_input_tokens: 12_400,
        prepared_input_tokens: 8_100,
        context_window: 128_000,
        input_budget: 24_000,
        working_input_budget: 16_000,
        trigger_tokens: 12_000,
        target_tokens: 8_000,
        prepared_usage_ratio: 0.3375,
        message_tokens_before: 8_700,
        message_tokens_after: 4_700,
        tool_schema_tokens: 3_200,
        tool_result_tokens_before: 6_800,
        tool_result_tokens_after: 2_100,
        skill_catalog_tokens: 300,
        active_skill_tokens: 500,
        compaction_stage: 'tool_results_and_rounds',
        compacted_tool_results: 4,
        removed_tool_rounds: 3,
        summary_updated: true,
      }),
    ])
    expect(step.originalInputTokens).toBe(12_400)
    expect(step.preparedInputTokens).toBe(8_100)
    expect(step.windowUsageRatio).toBeCloseTo(8_100 / 128_000)
    expect(step.budgetUsageRatio).toBeCloseTo(8_100 / 16_000)
    expect(step.breakdown.find((item) => item.key === 'messages')?.tokens).toBe(1_800)
    expect(step.breakdown.reduce((sum, item) => sum + item.tokens, 0)).toBe(8_100)
    expect(step.removedToolRounds).toBe(3)
    expect(step.summaryUpdated).toBe(true)
    expect(step.prefixDecision).toBeNull()
    expect(step.localPrefixReused).toBeNull()
    expect(step.upstreamCachedTokens).toBeNull()
  })

  it('本地追加和真实上游缓存统计独立，按 run 与 step 关联', () => {
    const [step] = buildContextSteps([
      event({ prefix_decision: 'append', cache_prefix_reused: true, cache_prefix_message_count: 18 }),
      event({ event_id: 'other', type: 'model_completed', run_id: 'r2', sequence: 2,
        usage: { input_tokens: 10, output_tokens: 1, total_tokens: 11, cached_input_tokens: 9 } }),
      event({ event_id: 'completed', type: 'model_completed', sequence: 3,
        usage: { input_tokens: 100, output_tokens: 1, total_tokens: 101, cached_input_tokens: 0 } }),
    ])
    expect(step.prefixDecision).toBe('append')
    expect(step.localPrefixReused).toBe(true)
    expect(step.prefixMessageCount).toBe(18)
    expect(step.modelCompleted).toBe(true)
    expect(step.upstreamCachedTokens).toBe(0)
    expect(step.upstreamCacheRatio).toBe(0)
  })

  it('压缩后的缓存统计保留真实数值，缺失统计不推断为零', () => {
    const [compacted, unknown] = buildContextSteps([
      event({ prefix_decision: 'compact', prefix_rebuild_reason: 'input_budget', cache_prefix_reused: false }),
      event({ event_id: 'complete-1', type: 'model_completed', sequence: 2,
        usage: { input_tokens: 100, output_tokens: 1, total_tokens: 101, cached_input_tokens: 40 } }),
      event({ event_id: 'start-2', sequence: 3, step: 2, prefix_decision: 'append', cache_prefix_reused: true }),
      event({ event_id: 'complete-2', type: 'model_completed', sequence: 4, step: 2,
        usage: { input_tokens: 150, output_tokens: 1, total_tokens: 151 } }),
    ])
    expect(compacted.prefixRebuildReason).toBe('input_budget')
    expect(compacted.localPrefixReused).toBe(false)
    expect(compacted.upstreamCacheRatio).toBe(0.4)
    expect(unknown.modelCompleted).toBe(true)
    expect(unknown.upstreamInputTokens).toBe(150)
    expect(unknown.upstreamCachedTokens).toBeNull()
    expect(unknown.upstreamCacheRatio).toBeNull()
  })

  it('同一步重试不把后一次完成结果分配给已中断的请求', () => {
    const abandoned = event({ event_id: 'abandoned', sequence: 1 })
    const resumed = event({ event_id: 'resumed', sequence: 2 })
    const completed = event({ event_id: 'completed', type: 'model_completed', sequence: 3,
      usage: { input_tokens: 100, output_tokens: 1, total_tokens: 101, cached_input_tokens: 80 } })
    const events = Object.freeze([completed, abandoned, resumed])
    const steps = buildContextSteps([...events])
    expect(steps[0].modelCompleted).toBe(false)
    expect(steps[0].upstreamCachedTokens).toBeNull()
    expect(steps[1].upstreamCacheRatio).toBe(0.8)
    expect(events.map((item) => item.event_id)).toEqual(['completed', 'abandoned', 'resumed'])
  })

  it('无输入 token 或无效上游计数不生成虚假的百分比', () => {
    for (const [input, cached] of [[0, 0], [10, 20], [100, Number.NaN]]) {
      const [step] = buildContextSteps([
        event({}),
        event({ event_id: 'completed', type: 'model_completed', sequence: 2,
          usage: { input_tokens: input, output_tokens: 1, total_tokens: input + 1, cached_input_tokens: cached } }),
      ])
      expect(step.upstreamCacheRatio).toBeNull()
    }
  })

  it('旧 trace 全部使用默认 sequence 时沿已记录顺序关联结果', () => {
    const [step] = buildContextSteps([
      event({ sequence: 0 }),
      event({ event_id: 'completed', type: 'model_completed', sequence: 0,
        usage: { input_tokens: 100, output_tokens: 1, total_tokens: 101 } }),
    ])
    expect(step.modelCompleted).toBe(true)
    expect(step.upstreamCachedTokens).toBeNull()
  })
})

describe('trace', () => {
  it('durable 与 live 按 event_id 去重并按 sequence 排序', () => {
    const first = event({ event_id: 'e1', sequence: 1 })
    const second = event({ event_id: 'e2', sequence: 2, step: 2 })
    expect(mergeRunEvents([first], [second, first]).map((item) => item.event_id)).toEqual(['e1', 'e2'])
    expect(buildTraceGroups([first, second]).map((group) => group.label)).toEqual(['Step 1', 'Step 2'])
  })

  it('从持久化列表恢复最新Run，不依赖返回顺序', () => {
    const base: Run = {
      id: 'old',
      conversation_id: 'c1',
      status: 'completed',
      user_message: 'old',
      created_at: '2026-08-21T00:00:00Z',
      started_at: null,
      updated_at: '2026-08-21T00:00:00Z',
      completed_at: null,
      error: null,
      stop_reason: 'final_answer',
      recovered_from_run_id: null,
      source: null,
      source_id: null,
      scheduled_for: null,
      triggered_at: null,
      mode: 'normal',
    }
    const newest = {
      ...base,
      id: 'newest',
      created_at: '2026-08-22T00:00:00Z',
    }
    expect(latestRunId([newest, base])).toBe('newest')
    expect(latestRunId([])).toBeNull()
  })
})

function snapshot(partial: Partial<ConversationSummarySnapshot>): ConversationSummarySnapshot {
  return {
    current_objective: null,
    user_constraints: [],
    key_decisions: [],
    completed_work: [],
    current_state: [],
    pending_work: [],
    important_facts: [],
    ...partial,
  }
}

describe('上下文透明度', () => {
  const baseline = snapshot({ current_objective: '修复分页', user_constraints: ['只改 backend', '不要改数据库'] })
  const compacted = snapshot({ current_objective: '修复分页', user_constraints: ['只改 backend'], completed_work: ['定位到 export.py'] })
  const steps = buildContextSteps([
    event({ event_id: 'a', sequence: 1, step: 1, summary_snapshot: baseline, summary_covered_before: 4, summary_covered_after: 4, constraints_revision: 1, source_message_count: 9 }),
    event({ event_id: 'b', sequence: 2, step: 2, summary_covered_before: 4, summary_covered_after: 4, constraints_revision: 1 }),
    event({
      event_id: 'c',
      sequence: 3,
      step: 3,
      summary_updated: true,
      summary_snapshot: compacted,
      summary_previous_snapshot: baseline,
      summary_covered_before: 4,
      summary_covered_after: 28,
      constraints_possibly_dropped: ['不要改数据库'],
      constraints_revision: 2,
    }),
  ])

  it('解析覆盖范围、约束版本和可能丢失的约束', () => {
    expect(steps.map((step) => step.constraintsRevision)).toEqual([1, 1, 2])
    expect(steps[0].sourceMessageCount).toBe(9)
    expect(newlyCoveredRange(steps[0])).toBeNull()
    expect(newlyCoveredRange(steps[2])).toEqual({ from: 4, to: 28 })
    expect(steps[2].constraintsPossiblyDropped).toEqual(['不要改数据库'])
  })

  it('所选步骤沿用之前最近一次的摘要快照', () => {
    expect(effectiveSummary(steps, 2)).toEqual({ snapshot: baseline, fromStep: 1 })
    expect(effectiveSummary(steps, 3)?.fromStep).toBe(3)
    expect(effectiveSummary(buildContextSteps([event({})]), 1)).toBeNull()
  })

  it('按字段对比两版摘要', () => {
    const diff = diffSummary(baseline, compacted)
    const constraints = diff.find((field) => field.key === 'user_constraints')!
    expect(constraints.kept).toEqual(['只改 backend'])
    expect(constraints.removed).toEqual(['不要改数据库'])
    expect(diff.find((field) => field.key === 'completed_work')!.added).toEqual(['定位到 export.py'])
    expect(diff.find((field) => field.key === 'current_objective')!.kept).toEqual(['修复分页'])
  })

  it('旧事件没有新字段时保持兼容', () => {
    const [legacy] = buildContextSteps([event({})])
    expect(legacy.coveredBefore).toBeNull()
    expect(legacy.summarySnapshot).toBeNull()
    expect(legacy.constraintsPossiblyDropped).toEqual([])
    expect(legacy.constraintsRevision).toBeNull()
  })
})
