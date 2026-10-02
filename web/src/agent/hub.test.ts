import { describe, expect, it } from 'vitest'

import type { ApprovalRequest, MeaRun, Run } from '../api/types'
import { buildInbox, ledgerSummary, meaVerdict, runVerdict } from './hub'
import { groupRuns } from './meaPresentation'

const run = (id: string, patch: Partial<Run> = {}): Run => ({
  id,
  conversation_id: 'c1',
  status: 'completed',
  user_message: id,
  created_at: '2026-09-01T00:00:00Z',
  started_at: null,
  updated_at: '2026-09-01T00:00:00Z',
  completed_at: null,
  error: null,
  stop_reason: null,
  recovered_from_run_id: null,
  source: null,
  source_id: null,
  scheduled_for: null,
  triggered_at: null,
  mode: 'normal',
  ...patch,
})

const mea = (id: string, patch: Partial<MeaRun> = {}): MeaRun => ({
  id,
  task_id: 't1',
  conversation_id: 'c1',
  status: 'running',
  round_budget: 25,
  requirements_revision: 1,
  completion_decision: null,
  extra_tools: [],
  once_notes: [],
  pending_question: null,
  pending_choices: [],
  pause_requested: false,
  abort_reason: null,
  final_response: null,
  created_at: '2026-09-01T00:00:00Z',
  updated_at: '2026-09-01T00:00:00Z',
  ...patch,
})

const approval = (id: string, patch: Partial<ApprovalRequest> = {}): ApprovalRequest => ({
  id,
  run_id: 'r1',
  conversation_id: 'c1',
  tool_name: 'run_shell_command',
  tool_call_id: 'call',
  arguments: {},
  reason: '',
  status: 'pending',
  created_at: '2026-09-01T00:00:00Z',
  resolved_at: null,
  ...patch,
})

describe('待处理队列', () => {
  it('汇总审批、需要介入的长任务和可恢复的中断，审批排最前', () => {
    const items = buildInbox({
      approvals: [approval('a1'), approval('a2', { status: 'approved' })],
      meas: [
        mea('m-ask', { status: 'waiting_user', pending_question: '用哪个数据库？', updated_at: '2026-09-03T00:00:00Z' }),
        mea('m-paused', { status: 'paused', abort_reason: 'max_rounds', updated_at: '2026-09-02T00:00:00Z' }),
        mea('m-running'),
        mea('m-done', { status: 'completed' }),
      ],
      runs: [
        run('r-int', { status: 'interrupted' }),
        run('r-fixed', { status: 'interrupted' }),
        run('r-recovery', { recovered_from_run_id: 'r-fixed' }),
        run('r-child', { status: 'interrupted', source: 'mea:executor', source_id: 'm-running' }),
        run('r-failed', { status: 'failed' }),
      ],
    })

    expect(items.map((item) => item.id)).toEqual(['a1', 'm-ask', 'm-paused', 'r-int'])
    const ask = items[1]
    expect(ask.kind === 'mea' && ask.reason).toBe('用哪个数据库？')
    const paused = items[2]
    expect(paused.kind === 'mea' && paused.reason).toContain('25 轮预算已用完')
  })
})

describe('账本', () => {
  it('长任务给审计结论，普通执行标成未经审计', () => {
    expect(meaVerdict(mea('m', { status: 'completed' }))).toEqual({ label: '审计通过', tone: 'ok' })
    expect(meaVerdict(mea('m', { status: 'blocked' })).tone).toBe('bad')
    expect(meaVerdict(undefined).tone).toBe('muted')
    expect(runVerdict(run('r'))).toEqual({ label: '未经审计', tone: 'muted' })
    expect(runVerdict(run('r', { status: 'interrupted' })).label).toBe('可恢复')
  })

  it('汇总：长任务按审计结论计数，普通执行单独计', () => {
    const entries = groupRuns([
      run('p1'),
      run('p2', { status: 'failed' }),
      run('e1', { source: 'mea:executor', source_id: 'm-ok' }),
      run('a1', { source: 'mea:auditor', source_id: 'm-ok' }),
      run('e2', { source: 'mea:executor', source_id: 'm-bad' }),
      run('e3', { source: 'mea:executor', source_id: 'm-unknown' }),
    ])
    const meas = new Map([
      ['m-ok', mea('m-ok', { status: 'completed' })],
      ['m-bad', mea('m-bad', { status: 'blocked' })],
    ])
    expect(ledgerSummary(entries, meas)).toEqual({ audited: 3, passed: 1, rejected: 1, unaudited: 2 })
  })
})
