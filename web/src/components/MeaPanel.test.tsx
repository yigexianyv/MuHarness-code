import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'

import type { MeaDetail, MeaRound, MeaRun, MeaStatus, Task } from '../api/types'
import MeaPanel from './MeaPanel'

const noop = (): void => {}

function mea(status: MeaStatus, overrides: Partial<MeaRun> = {}): MeaRun {
  return {
    id: 'mea-1',
    task_id: 'task-1',
    conversation_id: 'conv-1',
    status,
    round_budget: 25,
    requirements_revision: 2,
    completion_decision: null,
    extra_tools: [],
    once_notes: [],
    pending_question: null,
    pending_choices: [],
    pause_requested: false,
    abort_reason: null,
    final_response: null,
    created_at: '2026-09-29T01:00:00+00:00',
    updated_at: '2026-09-29T01:10:00+00:00',
    ...overrides,
  }
}

function round(index: number, overrides: Partial<MeaRound> = {}): MeaRound {
  return {
    mea_run_id: 'mea-1',
    index,
    kind: 'normal',
    phase: 'applied',
    route: 'execute',
    step_id: 's1',
    subtask: '读取 users.csv 的表头',
    focus: null,
    manager_run_id: 'm',
    executor_run_id: 'e',
    auditor_run_id: 'a',
    audit_status: 'complete',
    integrity_status: 'clean',
    contract_audit_status: 'aligned',
    step_acceptance: 'satisfied',
    stale_requirements: false,
    abandon_reason: null,
    interrupt_reason: null,
    created_at: '2026-09-29T01:00:00+00:00',
    updated_at: '2026-09-29T01:05:00+00:00',
    plan_text: '下一步: 执行任务',
    executor_output: '表头是 id,name',
    auditor_report: '状态: complete',
    harness_feedback: null,
    related_refs: [],
    executor_rejections: {},
    ...overrides,
  }
}

const task: Task = {
  id: 'task-1',
  title: '导入用户',
  description: null,
  goal: 'users 表包含 CSV 全部数据',
  status: 'active',
  priority: 'normal',
  constraints: ['不要修改数据库结构'],
  state: [],
  key_facts: [],
  steps: [
    { id: 's1', title: '读取 CSV', status: 'done', note: null, acceptance: '读到表头和数据' },
    { id: 's2', title: '导出 JSON', status: 'superseded', note: null, superseded_by: 'A1', superseded_reason: '改为导出 CSV' },
    { id: 's3', title: '导出 CSV', status: 'todo', note: null, acceptance: '文件存在且行数正确' },
  ],
  owner_conversation_id: 'conv-1',
  run_ids: [],
  created_at: '2026-09-29T00:00:00+00:00',
  updated_at: '2026-09-29T01:00:00+00:00',
  completed_at: null,
  revision: 5,
}

function detail(run: MeaRun, rounds: MeaRound[] = [round(1), round(2, { phase: 'abandoned', abandon_reason: '要求已更新' })]): MeaDetail {
  return {
    mea: run,
    rounds,
    task,
    requirements: {
      original_request: '把 users.csv 导入数据库',
      goal: 'users 表包含 CSV 全部数据',
      description: null,
      constraints: ['不要修改数据库结构'],
      amendments: [{ id: 'A1', kind: 'note', text: '改为导出 CSV', revision: 2, at: '2026-09-29T01:02:00+00:00' }],
      revision: 2,
    },
  }
}

function render(run: MeaRun, extra: Partial<Parameters<typeof MeaPanel>[0]> = {}): string {
  return renderToStaticMarkup(
    <MeaPanel
      mea={run}
      detail={detail(run)}
      onAnswer={noop}
      onNote={noop}
      onPause={noop}
      onResume={noop}
      onCancel={noop}
      {...extra}
    />,
  )
}

describe('MeaPanel', () => {
  it('运行中：显示当前活动、轮次和步骤进度，可以暂停、取消和补充', () => {
    const html = render(mea('running'), {
      defaultOpen: true,
      live: { role: 'executor', runId: 'e', type: 'tool_started', toolName: 'run_shell_command', eventTime: '' },
    })
    expect(html).toContain('长任务 · 运行中')
    expect(html).toContain('执行者正在调用 run_shell_command')
    expect(html).toContain('第 1/25 轮 · 步骤 1/2') // 作废的轮和被取代的步骤都不计入
    expect(html).toContain('>暂停</button>')
    expect(html).toContain('取消长任务')
    expect(html).toContain('补充指令')
    expect(html).toContain('已被修订 A1 取代：改为导出 CSV')
    expect(html).toContain('验收：读到表头和数据')
    expect(html).toContain('步骤通过')
    expect(html).toContain('计划作废')
  })

  it('等待回答：自动展开并显示问题、选项和回答框', () => {
    const html = render(mea('waiting_user', {
      pending_question: '目标库用生产库还是测试库？',
      pending_choices: ['生产库', '测试库'],
    }))
    expect(html).toContain('长任务 · 等待你回答')
    expect(html).toContain('<details open=""')
    expect(html).toContain('目标库用生产库还是测试库？')
    expect(html).toContain('>生产库</button>')
    expect(html).toContain('>测试库</button>')
    expect(html).toContain('回答并继续')
    expect(html).toContain('不回答，直接继续')
  })

  it('已完成：显示最终验收结论和最终回复，不再提供补充入口', () => {
    const html = render(mea('completed', { final_response: '长任务已完成：CSV 已导入。' }), { defaultOpen: true })
    expect(html).toContain('长任务 · 已完成')
    expect(html).toContain('已通过最终验收')
    expect(html).toContain('长任务已完成：CSV 已导入。')
    expect(html).not.toContain('补充指令')
    expect(html).not.toContain('>暂停</button>')
  })

  it('已阻塞：显示阻塞原因', () => {
    const html = render(mea('blocked', {
      completion_decision: {
        outcome: 'blocked',
        round_index: 3,
        auditor_run_id: null,
        requirements_revision: 2,
        reason: '数据库连接被拒绝，需要凭据',
        decided_at: '2026-09-29T01:20:00+00:00',
      },
      final_response: '长任务无法继续：缺少数据库凭据。',
    }), { defaultOpen: true })
    expect(html).toContain('长任务 · 已阻塞')
    expect(html).toContain('长任务无法继续推进，已停止')
    expect(html).toContain('数据库连接被拒绝，需要凭据')
    expect(html).not.toContain('补充指令')
  })

  it('已暂停：轮次用完时给出说明，并默认追加 10 轮', () => {
    const html = render(mea('paused', { abort_reason: 'max_rounds' }))
    expect(html).toContain('长任务 · 已暂停')
    expect(html).toContain('<details open=""')
    expect(html).toContain('25 轮预算已用完')
    expect(html).toContain('追加轮次')
    expect(html).toContain('value="10"')
    expect(html).toContain('>继续</button>')
    expect(html).toContain('补充指令')
  })

  it('详情还没读到时只显示状态', () => {
    const html = renderToStaticMarkup(
      <MeaPanel mea={mea('running')} detail={null} onAnswer={noop} onNote={noop}
        onPause={noop} onResume={noop} onCancel={noop} />,
    )
    expect(html).toContain('长任务 · 运行中')
    expect(html).toContain('第 0/25 轮')
    expect(html).not.toContain('轮次记录')
  })
})
