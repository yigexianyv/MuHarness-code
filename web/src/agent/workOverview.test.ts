import { describe, expect, it } from 'vitest'

import type { Artifact } from '../api/artifacts'
import type { ApprovalRequest, Conversation, MeaRun, Run } from '../api/types'
import { buildInbox } from './hub'
import { buildWorkOverview, filterWorkOverview, inboxConversationId } from './workOverview'

const time = '2026-09-01T00:00:00Z'
const conversation = (id: string, patch: Partial<Conversation> = {}): Conversation => ({ id, title: id, created_at: time, updated_at: time, message_count: 0, ...patch })
const run = (id: string, patch: Partial<Run> = {}): Run => ({
  id, conversation_id: 'c1', status: 'completed', user_message: id, created_at: time,
  started_at: time, updated_at: time, completed_at: time, error: null, stop_reason: null,
  recovered_from_run_id: null, source: null, source_id: null, scheduled_for: null, triggered_at: null,
  mode: 'normal', ...patch,
})
const mea = (id: string, patch: Partial<MeaRun> = {}): MeaRun => ({
  id, task_id: 't1', conversation_id: 'c1', status: 'running', round_budget: 25,
  requirements_revision: 1, completion_decision: null, extra_tools: [], once_notes: [], pending_question: null,
  pending_choices: [], pause_requested: false, abort_reason: null, final_response: null, created_at: time, updated_at: time, ...patch,
})
const approval = (patch: Partial<ApprovalRequest> = {}): ApprovalRequest => ({
  id: 'a1', run_id: 'r1', conversation_id: 'c1', tool_name: 'run_shell_command', tool_call_id: 'call1',
  arguments: {}, reason: '需要写入文件', status: 'pending', created_at: time, resolved_at: null, ...patch,
})
const artifact = (id: string, patch: Partial<Artifact> = {}): Artifact => ({
  id, kind: 'file', title: id, description: null, filename: id, mime_type: 'text/plain', size_bytes: 12,
  sha256: 'sha', run_id: 'r1', conversation_id: 'c1', task_id: null, source_url: null, created_at: time, ...patch,
})

function build(patch: Partial<Parameters<typeof buildWorkOverview>[0]> = {}) {
  return buildWorkOverview({ conversations: [], runs: [], meas: [], inbox: [], artifacts: [], ...patch })
}

describe('work overview projection', () => {
  it('merges executions by conversation and preserves unstarted conversations', () => {
    const items = build({ conversations: [conversation('c1'), conversation('c2')], runs: [run('old'), run('new', { updated_at: '2026-09-02T00:00:00Z' })] })
    expect(items).toHaveLength(2)
    expect(items.find((item) => item.conversationId === 'c1')?.run?.id).toBe('new')
    expect(items.find((item) => item.conversationId === 'c2')).toMatchObject({ kind: 'conversation', lane: 'ready', status: '等待输入目标' })
  })

  it('keeps unresolved approvals ahead of newer completed history', () => {
    const runs = [run('r1', { status: 'running' }), run('new', { updated_at: '2026-09-04T00:00:00Z' })]
    const inbox = buildInbox({ approvals: [approval()], meas: [], runs })
    expect(build({ runs, inbox })[0]).toMatchObject({ lane: 'attention', status: '等待工具审批', run: { id: 'r1' }, decisions: [{ kind: 'approval' }] })
  })

  it('does not show a recovered interruption as awaiting a decision', () => {
    const runs = [run('old', { status: 'interrupted' }), run('new', { recovered_from_run_id: 'old', updated_at: '2026-09-02T00:00:00Z' })]
    const inbox = buildInbox({ approvals: [], meas: [], runs })
    expect(build({ runs, inbox })).toEqual([expect.objectContaining({ lane: 'finished', run: expect.objectContaining({ id: 'new' }), decisions: [] })])
  })

  it('shows a parent MEA once and never treats child completion as goal completion', () => {
    const runs = [run('e1', { source: 'mea:executor', source_id: 'm1', mode: 'execute' }), run('a1', { source: 'mea:auditor', source_id: 'm1', mode: 'audit' })]
    const items = build({ conversations: [conversation('c1')], runs, meas: [mea('m1')] })
    expect(items).toHaveLength(1)
    expect(items[0]).toMatchObject({ kind: 'mea', lane: 'running', status: '运行中', run: null })
    expect(build({ conversations: [conversation('c1')], runs })).toEqual([])
  })

  it('never labels a normal completed run as independently audited', () => {
    expect(build({ runs: [run('r1')] })[0].status).toBe('执行结束 · 未独立审计')
  })

  it('requires a matching completion decision and auditor for the final audit label', () => {
    const completed = mea('m1', { status: 'completed' })
    expect(build({ meas: [completed] })[0].status).toBe('已完成 · 验收信息未确认')
    const confirmed = { ...completed, completion_decision: {
      outcome: 'completed' as const, round_index: 2, auditor_run_id: 'audit1', requirements_revision: 1,
      reason: null, decided_at: time,
    } }
    expect(build({ meas: [confirmed] })[0].status).toBe('最终验收通过')
    expect(build({ meas: [{ ...confirmed, requirements_revision: 2 }] })[0].status).toBe('已完成 · 验收信息未确认')
  })

  it('associates child-run approvals with the MEA and relates published artifacts', () => {
    const runs = [run('e1', { source: 'mea:executor', source_id: 'm1', mode: 'execute' })]
    const meas = [mea('m1')]
    const inbox = buildInbox({ approvals: [approval({ run_id: 'e1', conversation_id: null })], runs, meas })
    const items = build({ runs, meas, inbox, artifacts: [artifact('file1', { conversation_id: null, task_id: 't1', run_id: 'e1' }), artifact('elsewhere', { task_id: 't2', conversation_id: 'c2' })] })
    expect(items[0]).toMatchObject({ lane: 'attention', decisions: [{ kind: 'approval' }], artifacts: [{ id: 'file1' }] })
    expect(inboxConversationId(inbox[0], runs)).toBe('c1')
  })

  it('filters real names and goals without mutating the source list', () => {
    const items = build({ conversations: [conversation('c1', { title: '设计报告' }), conversation('c2', { title: 'API 接口' })], runs: [run('r2', { conversation_id: 'c2' })], meas: [mea('m1')] })
    expect(filterWorkOverview(items, 'long', '报告').map((item) => item.kind)).toEqual(['mea'])
    expect(filterWorkOverview(items, 'normal', ' api ').map((item) => item.run?.id)).toEqual(['r2'])
    expect(filterWorkOverview(items, 'all', '不存在')).toEqual([])
    expect(items).toHaveLength(2)
  })

  it('uses the conversation lane only when no execution is in the recent query', () => {
    const items = build({ conversations: [conversation('historic', { message_count: 10 })] })
    expect(items[0].status).toBe('近期无执行记录')
    expect(items[0].detail).toContain('当前查询范围')
  })
})
