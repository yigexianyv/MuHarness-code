

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'

import type { AgentEvent, ConversationSummarySnapshot, RequestToolView } from '../api/types'
import ContextInspector, { ToolOutputViewer } from './ContextInspector'

function contextEvent(): AgentEvent {
  return {
    event_id: 'context-1',
    run_id: 'run-1',
    conversation_id: 'conversation-1',
    sequence: 1,
    type: 'model_started',
    event_time: '2026-08-22T00:00:00Z',
    step: 2,
    provider: 'fake',
    model: 'fake-model',
    message: null,
    tool_call: null,
    tool_result: null,
    usage: null,
    stop_reason: null,
    approval_decision: null,
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
    summary_provider: 'qwen',
    summary_model: 'qwen-turbo',
    summary_duration_ms: 810,
    summary_usage: {
      input_tokens: 900,
      output_tokens: 100,
      total_tokens: 1_000,
      model_calls: 1,
    },
  }
}

describe('ContextInspector', () => {
  it('展示窗口、预算与压缩动作', () => {
    const html = renderToStaticMarkup(<ContextInspector events={[contextEvent()]} />)
    expect(html).toContain('Model step')
    expect(html).toContain('128k')
    expect(html).toContain('6.3%')
    expect(html).toContain('50.6%')
    expect(html).toContain('Working budget')
    expect(html).toContain('12k / 8k')
    expect(html).toContain('4 个工具结果已压缩')
    expect(html).toContain('3 个旧工具轮已移除')
    expect(html).toContain('Conversation summary 已更新')
    expect(html).toContain('qwen / qwen-turbo')
    expect(html).toContain('810 ms')
    expect(html).toContain('1k')
    expect(html).toContain('Memory、Task 与系统消息当前没有独立计数字段')
    expect(html).not.toContain('为什么这轮可能更贵')
  })

  it('没有模型请求时显示明确空状态', () => {
    const html = renderToStaticMarkup(<ContextInspector events={[]} />)
    expect(html).toContain('暂无 Context 数据')
  })

  it('分开显示本地追加与上游报告的零命中', () => {
    const started = { ...contextEvent(), prefix_decision: 'append', cache_prefix_reused: true, cache_prefix_message_count: 18 }
    const completed: AgentEvent = {
      ...started, event_id: 'completed', type: 'model_completed', sequence: 2,
      usage: { input_tokens: 100, output_tokens: 5, total_tokens: 105, cached_input_tokens: 0 },
    }
    const html = renderToStaticMarkup(<ContextInspector events={[started, completed]} />)
    expect(html).toContain('追加')
    expect(html).toContain('已复用 18 条消息')
    expect(html).toContain('0 cached / 100 input · 0%')
    expect(html).toContain('本地前缀复用不代表上游 KV cache 命中')
    expect(html).not.toContain('上游未返回缓存统计')
  })

  it('旧 trace 缺字段时显示未知，已完成但无缓存统计不是零', () => {
    const started = contextEvent()
    const completed: AgentEvent = {
      ...started, event_id: 'completed', type: 'model_completed', sequence: 2,
      usage: { input_tokens: 100, output_tokens: 5, total_tokens: 105 },
    }
    const html = renderToStaticMarkup(<ContextInspector events={[started, completed]} />)
    expect(html).toContain('历史记录未提供')
    expect(html).toContain('上游未返回缓存统计')
    expect(html).not.toContain('0 cached')
    expect(html).not.toContain('模型请求尚未完成')
  })

  it('阶段压缩显示原因和上游统计，不误报缓存故障', () => {
    const started = { ...contextEvent(), prefix_decision: 'compact', prefix_rebuild_reason: 'input_budget', cache_prefix_reused: false }
    const completed: AgentEvent = {
      ...started, event_id: 'completed', type: 'model_completed', sequence: 2,
      usage: { input_tokens: 100, output_tokens: 5, total_tokens: 105, cached_input_tokens: 40 },
    }
    const html = renderToStaticMarkup(<ContextInspector events={[started, completed]} />)
    expect(html).toContain('阶段压缩')
    expect(html).toContain('未复用')
    expect(html).toContain('input_budget')
    expect(html).toContain('40 cached / 100 input · 40%')
    expect(html).not.toContain('缓存故障')
    expect(html).not.toContain('缓存失效')
  })

  it('请求进行中不预测上游命中结果', () => {
    const html = renderToStaticMarkup(<ContextInspector events={[contextEvent()]} />)
    expect(html).toContain('模型请求尚未完成')
    expect(html).not.toContain('0 cached')
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

function started(partial: Partial<AgentEvent>): AgentEvent {
  return {
    event_id: `e${partial.step}`,
    run_id: 'r1',
    conversation_id: 'c1',
    sequence: partial.step ?? 1,
    type: 'model_started',
    event_time: '2026-10-04T00:00:00Z',
    step: 1,
    provider: 'fake',
    model: 'fake',
    message: null,
    tool_call: null,
    tool_result: null,
    usage: null,
    stop_reason: null,
    approval_decision: null,
    context_window: 128_000,
    working_input_budget: 128_000,
    ...partial,
  }
}

function render(events: AgentEvent[], conversationId: string | null): string {
  const client = new QueryClient()
  return renderToStaticMarkup(
    <QueryClientProvider client={client}>
      <ContextInspector events={events} runId="r1" conversationId={conversationId} />
    </QueryClientProvider>,
  )
}

describe('ContextInspector 上下文面板', () => {
  const previous = snapshot({ current_objective: '修复分页', user_constraints: ['只改 backend', '不要改数据库'] })
  const events = [
    started({ step: 1, prepared_input_tokens: 12_000, summary_snapshot: previous, summary_covered_before: 0, summary_covered_after: 0 }),
    started({
      step: 2,
      original_estimated_input_tokens: 131_000,
      prepared_input_tokens: 58_000,
      compaction_stage: 'conversation_summary',
      summary_updated: true,
      summary_snapshot: snapshot({ current_objective: '修复分页', user_constraints: ['只改 backend'] }),
      summary_previous_snapshot: previous,
      summary_covered_before: 0,
      summary_covered_after: 24,
      compacted_tool_results: 3,
      source_message_count: 40,
      constraints_possibly_dropped: ['不要改数据库'],
      constraints_revision: 2,
    }),
  ]

  it('展示压缩时间线、被替代范围、摘要和约束提醒', () => {
    const html = render(events, 'c1')
    expect(html).toContain('时间线')
    expect(html).toContain('──压缩──▶')
    expect(html).toContain('第 1~24 条消息已由摘要替代')
    expect(html).toContain('3 个旧工具结果在请求中被压缩')
    expect(html).toContain('⚠ 1 条约束可能丢失')
    expect(html).toContain('当前摘要（步骤 2 生成）')
    expect(html).toContain('和上一版对比')
    expect(html).toContain('&quot;不要改数据库&quot;')
    expect(html).toContain('必须记住的事项')
  })

  it('长任务子运行不显示必须记住的事项编辑框', () => {
    const html = render(events, null)
    expect(html).not.toContain('必须记住的事项')
  })
})

describe('工具输出截短', () => {
  const toolCompleted = (partial: { sequence: number; step: number; id: string; truncated: boolean }): AgentEvent => ({
    ...started({ step: partial.step }),
    event_id: `tool-${partial.sequence}`,
    sequence: partial.sequence,
    type: 'tool_completed',
    tool_result: {
      tool_call_id: partial.id,
      tool_name: 'read_file',
      success: true,
      output: partial.truncated ? 'HEAD…[truncated]' : 'short',
      error: null,
      duration_ms: 1,
      evidence_id: partial.truncated ? 'abcd1234' : null,
      output_truncated: partial.truncated ? true : null,
    },
  })
  const view = (id: string, partial: Partial<RequestToolView>): RequestToolView => ({
    tool_call_id: id,
    tool_name: 'read_file',
    original_chars: 133_021,
    stored_chars: 20_000,
    stored_truncated: true,
    included: true,
    request_chars: 6_067,
    request_shortened: true,
    ...partial,
  })

  it('三层长度：工具返回 → 执行器保存 → 本次请求', () => {
    const events = [
      started({ step: 1, sequence: 1, request_tool_views: [] }),
      toolCompleted({ sequence: 2, step: 1, id: 'call-long', truncated: true }),
      started({ step: 2, sequence: 3, request_tool_views: [view('call-long', {})] }),
      // 之后每步都带着同一段摘录，只在第一次出现时提示
      started({ step: 3, sequence: 4, request_tool_views: [view('call-long', {})] }),
    ]
    const html = render(events, 'c1')
    // 只在第一次出现（步骤 2）时提示一次
    expect(html.match(/工具返回 133,021 → 执行器保存 20,000 → 本次请求 6,067 字符/g)).toHaveLength(1)
    expect(html).toContain('上下文层保留开头和结尾')
    expect(html).toContain('查看模型收到的内容')
    expect(html).toContain('查看完整工具原文')
    // 选中的步骤有工具输出表
    expect(html).toContain('本次请求中的工具输出')
    expect(html).toContain('上下文层截短')
  })

  it('执行器没截短、但上下文层截短的输出也会提示', () => {
    const events = [
      started({ step: 1, sequence: 1 }),
      toolCompleted({ sequence: 2, step: 1, id: 'call-10k', truncated: false }),
      started({
        step: 2,
        sequence: 3,
        request_tool_views: [view('call-10k', { original_chars: 10_000, stored_chars: 10_000, stored_truncated: false, request_chars: 6_066 })],
      }),
    ]
    const html = render(events, 'c1')
    expect(html).toContain('工具返回 10,000 → 执行器保存 10,000 → 本次请求 6,066 字符')
  })

  it('被摘要替代后提示本次请求不包含', () => {
    const events = [
      started({ step: 1, sequence: 1, request_tool_views: [view('call-a', { request_shortened: false, stored_truncated: false, original_chars: 50, stored_chars: 50, request_chars: 50 })] }),
      started({ step: 2, sequence: 2, request_tool_views: [view('call-a', { included: false, request_chars: null, request_shortened: false, stored_truncated: false })] }),
    ]
    const html = render(events, 'c1')
    expect(html).toContain('的输出从这一步起不在请求里')
    expect(html).toContain('本次请求不包含')
  })

  it('旧记录只能显示执行器保存的版本，不冒充模型收到的内容', () => {
    const events = [
      started({ step: 1, sequence: 1, prepared_input_tokens: 9_000 }),
      toolCompleted({ sequence: 2, step: 1, id: 'call-long', truncated: true }),
      toolCompleted({ sequence: 3, step: 1, id: 'call-long', truncated: true }),
    ]
    const html = render(events, 'c1')
    expect(html.match(/执行器保存了截短版本/g)).toHaveLength(1)
    expect(html).toContain('查看执行器保存的版本')
    expect(html).not.toContain('查看模型收到的内容')
  })

  it('短输出不提示', () => {
    const html = render(
      [
        started({ step: 1, sequence: 1 }),
        toolCompleted({ sequence: 2, step: 1, id: 'call-short', truncated: false }),
        started({ step: 2, sequence: 3, request_tool_views: [view('call-short', { original_chars: 5, stored_chars: 5, stored_truncated: false, request_chars: 5, request_shortened: false })] }),
      ],
      'c1',
    )
    expect(html).not.toContain('工具返回 5')
    expect(html).not.toContain('执行器保存了截短版本')
  })
})

describe('工具输出查看器', () => {
  const renderViewer = (node: React.ReactNode, seed: (client: QueryClient) => void = () => {}): string => {
    const client = new QueryClient({ defaultOptions: { queries: { staleTime: Infinity, retry: false } } })
    seed(client)
    return renderToStaticMarkup(<QueryClientProvider client={client}>{node}</QueryClientProvider>)
  }

  it('完整原文按页显示总长度和下一页', () => {
    const html = renderViewer(
      <ToolOutputViewer runId="r1" target={{ kind: 'full', toolCallId: 'call-long', toolName: 'read_file' }} onClose={() => {}} />,
      (client) => client.setQueryData(['run-tool-evidence', 'r1', 'call-long', 0], {
        run_id: 'r1', tool_call_id: 'call-long', tool_name: 'read_file', evidence_id: 'abcd1234ef',
        total_chars: 133_021, offset: 0, content: 'HEAD_MARKER', next_offset: 12_000,
      }),
    )
    expect(html).toContain('read_file：完整工具原文')
    expect(html).toContain('133,021')
    expect(html).toContain('HEAD_MARKER')
    expect(html).toMatch(/<button[^>]*>下一页/)
    expect(html).not.toMatch(/<button[^>]*disabled=""[^>]*>下一页/)
  })

  it('模型收到的内容来自这一步的请求记录，逐字显示', () => {
    const content = '{"output":"HEAD\\n\\n[tool output excerpt: 14000 characters omitted from the middle]\\n\\nTAIL"}'
    const html = renderViewer(
      <ToolOutputViewer runId="r1" target={{ kind: 'model', step: 2, toolCallId: 'c', toolName: 'read_file' }} onClose={() => {}} />,
      (client) => client.setQueryData(['run-tool-view', 'r1', 2, 'c'], {
        run_id: 'r1', step: 2, tool_call_id: 'c', tool_name: 'read_file',
        original_chars: 133_021, stored_chars: 20_000, stored_truncated: true,
        included: true, request_chars: 6_067, request_shortened: true, content,
      }),
    )
    expect(html).toContain('第 2 步模型收到的内容')
    expect(html).toContain('工具返回 133,021 → 执行器保存 20,000 → 本次请求 6,067 字符')
    expect(html).toContain('tool output excerpt: 14000 characters omitted from the middle')
    expect(html).not.toContain('完整工具原文')
  })
})
