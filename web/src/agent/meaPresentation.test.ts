import { describe, expect, it } from 'vitest'

import type { AgentEvent, MeaRun, Run } from '../api/types'
import {
  groupRuns,
  isMeaBusy,
  liveActivityFrom,
  liveActivityText,
  mergeMeas,
  pickPanelMea,
  roundVerdict,
  stepProgress,
} from './meaPresentation'

function run(id: string, overrides: Partial<Run> = {}): Run {
  return {
    id,
    conversation_id: 'conv-1',
    status: 'completed',
    user_message: 'prompt',
    created_at: `2026-09-29T01:00:0${id.length % 10}+00:00`,
    started_at: null,
    updated_at: '2026-09-29T01:00:00+00:00',
    completed_at: null,
    error: null,
    stop_reason: null,
    recovered_from_run_id: null,
    source: 'manual',
    source_id: null,
    scheduled_for: null,
    triggered_at: null,
    mode: 'normal',
    ...overrides,
  }
}

function meaRun(id: string, status: MeaRun['status'], createdAt: string, updatedAt = createdAt): MeaRun {
  return {
    id,
    task_id: `task-${id}`,
    conversation_id: 'conv-1',
    status,
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
    created_at: createdAt,
    updated_at: updatedAt,
  }
}

describe('groupRuns', () => {
  it('长任务的子 Run 按 source_id 合成一组，普通 Run 保持原样', () => {
    const entries = groupRuns([
      run('chat-1', { created_at: '2026-09-29T01:00:00+00:00' }),
      run('m1', { source: 'mea:manager', source_id: 'mea-a', created_at: '2026-09-29T01:01:00+00:00' }),
      run('e1', { source: 'mea:executor', source_id: 'mea-a', status: 'cancelled', created_at: '2026-09-29T01:02:00+00:00' }),
      run('a1', { source: 'mea:auditor', source_id: 'mea-a', created_at: '2026-09-29T01:03:00+00:00' }),
      run('m2', { source: 'mea:manager', source_id: 'mea-b', status: 'running', created_at: '2026-09-29T01:04:00+00:00' }),
    ])
    expect(entries.map((entry) => entry.kind)).toEqual(['run', 'mea', 'mea'])
    const first = entries[1]
    if (first.kind !== 'mea') throw new Error('expected group')
    expect(first.meaId).toBe('mea-a')
    expect(first.runs.map((child) => child.id)).toEqual(['m1', 'e1', 'a1'])
    expect(first.roleCounts).toEqual({ manager: 1, executor: 1, auditor: 1 })
    expect(first.status).toBe('completed') // 早先被取消的子 Run 不决定整体状态
    expect(first.latestAt).toBe('2026-09-29T01:03:00+00:00')
    const second = entries[2]
    if (second.kind !== 'mea') throw new Error('expected group')
    expect(second.status).toBe('running')
  })
})

describe('roundVerdict', () => {
  const base = {
    kind: 'normal' as const,
    phase: 'applied' as const,
    route: 'execute',
    audit_status: 'complete',
    integrity_status: 'clean',
    contract_audit_status: 'aligned',
    step_acceptance: 'satisfied',
    stale_requirements: false,
    interrupt_reason: null,
  }

  it('按控制头判定步骤和最终验收', () => {
    expect(roundVerdict(base)).toEqual({ label: '步骤通过', tone: 'ok' })
    expect(roundVerdict({ ...base, step_acceptance: 'not_satisfied' }).label).toBe('步骤未通过')
    expect(roundVerdict({ ...base, integrity_status: 'violation' }).label).toBe('步骤未通过')
    expect(roundVerdict({ ...base, kind: 'final_audit' }).label).toBe('最终验收通过')
    expect(roundVerdict({ ...base, kind: 'final_audit', audit_status: 'incomplete' }).label).toBe('最终验收未通过')
    expect(roundVerdict({ ...base, stale_requirements: true }).tone).toBe('warn')
    expect(roundVerdict({ ...base, audit_output_truncated: true, step_acceptance: 'not_satisfied' }))
      .toEqual({ label: '审计输出被截断', tone: 'warn' })
  })

  it('作废、中断、请示和进行中的轮次', () => {
    expect(roundVerdict({ ...base, phase: 'abandoned' }).label).toBe('计划作废')
    expect(roundVerdict({ ...base, phase: 'interrupted' }).label).toBe('已中断')
    expect(roundVerdict({ ...base, route: 'ask' }).label).toBe('请示用户')
    expect(roundVerdict({ ...base, phase: 'auditing', audit_status: null })).toEqual({ label: '审计中', tone: 'pending' })
  })
})

describe('panel selection', () => {
  it('优先显示未结束的长任务，推送的状态覆盖较旧的列表结果', () => {
    const old = meaRun('old', 'completed', '2026-09-29T01:00:00+00:00')
    const open = meaRun('open', 'paused', '2026-09-28T01:00:00+00:00')
    expect(pickPanelMea([old, open])?.id).toBe('open')
    expect(pickPanelMea([old])?.id).toBe('old')
    expect(pickPanelMea([])).toBeNull()

    const pushed = { ...open, status: 'running' as const, updated_at: '2026-09-29T02:00:00+00:00' }
    const merged = mergeMeas([old, open], [pushed])
    expect(merged.find((item) => item.id === 'open')?.status).toBe('running')
    expect(isMeaBusy('running') && isMeaBusy('finalizing') && !isMeaBusy('waiting_user')).toBe(true)
  })
})

describe('live activity', () => {
  function event(type: string, extra: Partial<AgentEvent> = {}): AgentEvent {
    return {
      event_id: 'ev',
      run_id: 'run-e',
      conversation_id: 'conv-1',
      sequence: 1,
      type,
      event_time: '2026-09-29T01:00:00+00:00',
      step: 1,
      provider: null,
      model: null,
      message: null,
      tool_call: null,
      tool_result: null,
      usage: null,
      stop_reason: null,
      approval_decision: null,
      ...extra,
    }
  }

  it('只保留结构化事件，并说清是哪个角色在做什么', () => {
    expect(liveActivityFrom('executor', event('model_output_delta'))).toBeNull()
    const calling = liveActivityFrom('executor', event('tool_started', {
      tool_call: { id: 'c1', name: 'run_shell_command', arguments: {} },
    }))
    expect(liveActivityText(calling)).toBe('执行者正在调用 run_shell_command')
    const approval = liveActivityFrom('auditor', event('tool_approval_required', {
      tool_call: { id: 'c2', name: 'run_shell_command', arguments: {} },
    }))
    expect(liveActivityText(approval)).toBe('审计者在等你批准 run_shell_command')
    expect(liveActivityText(liveActivityFrom('manager', event('model_started')))).toBe('管理者正在思考')
  })

  it('步骤进度不计被取代的步骤', () => {
    expect(stepProgress([
      { id: 's1', title: 'a', status: 'done', note: null },
      { id: 's2', title: 'b', status: 'superseded', note: null },
      { id: 's3', title: 'c', status: 'todo', note: null },
    ])).toEqual({ done: 1, total: 2, percent: 50 })
  })
})
