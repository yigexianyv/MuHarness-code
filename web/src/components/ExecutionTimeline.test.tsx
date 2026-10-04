import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { AgentEvent, ConversationFork, RewindPreview, RewindStepInfo } from '../api/types'
import ExecutionTimeline, { buildTimelineSteps, RewindDialog } from './ExecutionTimeline'
import ForkBanner from './ForkBanner'

function event(partial: Partial<AgentEvent>): AgentEvent {
  return {
    event_id: `e${partial.sequence}`,
    run_id: 'r1',
    conversation_id: 'c1',
    sequence: 1,
    type: 'model_completed',
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
    ...partial,
  }
}

const toolResult = (success: boolean) => ({
  tool_call_id: 'c', tool_name: 't', success, output: null, error: null, duration_ms: 1,
})

const events: AgentEvent[] = [
  event({ sequence: 1, step: 1, message: { role: 'assistant', content: null, tool_calls: [{ id: 'a', name: 'read_file', arguments: { path: 'src/export.py' } }] } }),
  event({ sequence: 2, step: 1, type: 'tool_completed', tool_result: toolResult(true) }),
  event({ sequence: 3, step: 2, message: { role: 'assistant', content: null, tool_calls: [{ id: 'b', name: 'run_shell_command', arguments: { command: 'pytest' } }] } }),
  event({ sequence: 4, step: 2, type: 'tool_completed', tool_result: toolResult(false) }),
  event({ sequence: 5, step: 3, message: { role: 'assistant', content: '已修复分页' } }),
]

function render(node: React.ReactNode, seed: (client: QueryClient) => void): string {
  const client = new QueryClient({ defaultOptions: { queries: { staleTime: Infinity } } })
  seed(client)
  return renderToStaticMarkup(<QueryClientProvider client={client}>{node}</QueryClientProvider>)
}

describe('执行时间线', () => {
  it('按步骤整理工具调用、失败数和最终回答', () => {
    const steps = buildTimelineSteps(events)
    expect(steps.map((step) => step.outcome)).toEqual(['ok', 'failed', 'answer'])
    expect(steps[0].label).toBe('read_file({"path":"src/export.py"})')
    expect(steps[1].failedTools).toBe(1)
    expect(steps[2].label).toBe('回答：已修复分页')
  })

  it('可重做的步骤按钮可用，不可重做的显示原因', () => {
    const infos: RewindStepInfo[] = [
      { step: 1, message_count: 3, has_snapshot: true, snapshot_error: null, rewindable: true, reason: null },
      { step: 2, message_count: 5, has_snapshot: false, snapshot_error: '工作区文件超过 20 MiB', rewindable: false, reason: '这一步没有文件快照（工作区文件超过 20 MiB）' },
    ]
    const html = render(
      <ExecutionTimeline runId="r1" events={events} onRewound={() => {}} />,
      (client) => client.setQueryData(['run-steps', 'r1'], infos),
    )
    expect(html.match(/重做此步/g)).toHaveLength(3)
    expect(html).toContain('title="回到这一步之前，改个决策重新执行"')
    expect(html).toContain('这一步没有文件快照')
    expect(html).toContain('title="没有检查点记录"')
    expect(html).toContain('✗ 1 失败')
  })

  it('确认弹窗列出文件变化、不会回退的操作和纠正输入框', () => {
    const preview: RewindPreview = {
      preview_id: 'p',
      run_id: 'r1',
      step: 3,
      files: [
        { path: 'src/db/schema.py', action: 'restore', changed_after_run: false },
        { path: 'migrations/0042.py', action: 'delete', changed_after_run: false },
        { path: 'src/export.py', action: 'restore', changed_after_run: true },
      ],
      skipped_files: [],
      irreversible: [{ step: 4, tool: 'run_shell_command', summary: '{"command":"alembic upgrade"}', note: '命令对工作区文件的修改会被恢复，但对数据库、网络、工作区之外文件的影响不会回退' }],
      blocked_reason: null,
    }
    const html = render(
      <RewindDialog runId="r1" step={3} onClose={() => {}} onRewound={() => {}} />,
      (client) => client.setQueryData(['rewind-preview', 'r1', 3], preview),
    )
    expect(html).toContain('回到第 3 步之前')
    expect(html).toContain('第 2 步完成后')
    expect(html).toContain('文件变化（共 3 个）')
    expect(html).toContain('恢复为第 3 步之前的内容')
    expect(html).toContain('将被删除（第 3 步之后新建）')
    expect(html).toContain('执行结束后又被改过')
    expect(html).toContain('以下操作不会回退')
    expect(html).toContain('alembic upgrade')
    expect(html).toContain('你的纠正')
    // 没写纠正时不能确认
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>确认并回到这一步/)
  })

  it('有执行在进行时提示原因', () => {
    const preview: RewindPreview = {
      preview_id: 'p', run_id: 'r1', step: 1, files: [], skipped_files: [], irreversible: [],
      blocked_reason: '有执行正在进行，请等它结束（或停止）后再回退',
    }
    const html = render(
      <RewindDialog runId="r1" step={1} onClose={() => {}} onRewound={() => {}} />,
      (client) => client.setQueryData(['rewind-preview', 'r1', 1], preview),
    )
    expect(html).toContain('执行开始时的状态')
    expect(html).toContain('有执行正在进行')
    expect(html).toContain('无需恢复')
  })
})

describe('分支会话横幅', () => {
  const fork: ConversationFork = {
    conversation_id: 'c2', source_conversation_id: 'c1', source_run_id: 'r1', source_step: 3, rewind_key: 'k', undone: false,
  }

  it('显示来源并提供返回与撤销', () => {
    const html = render(<ForkBanner fork={fork} sourceTitle="修复分页" onOpenSource={() => {}} />, () => {})
    expect(html).toContain('从《修复分页》第 3 步之前重新执行')
    expect(html).toContain('返回来源')
    expect(html).toContain('撤销文件回退')
  })

  it('撤销后不再提供撤销按钮', () => {
    const html = render(<ForkBanner fork={{ ...fork, undone: true }} sourceTitle={null} onOpenSource={() => {}} />, () => {})
    expect(html).toContain('（文件回退已撤销）')
    expect(html).not.toContain('>撤销文件回退<')
  })
})
