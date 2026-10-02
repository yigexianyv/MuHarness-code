

import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'

import type { Task } from '../api/types'
import PlanCard from './PlanCard'

const task: Task = {
  id: 'task-1',
  title: 'Refactor Web UI',
  description: null,
  goal: 'Create a focused agent workspace.',
  status: 'pending',
  priority: 'normal',
  constraints: [],
  state: [],
  key_facts: [],
  steps: [
    { id: 'step-1', title: 'Inspect the shell', status: 'done', note: null },
    { id: 'step-2', title: 'Build the chat workspace', status: 'todo', note: null },
  ],
  owner_conversation_id: 'conv-1',
  run_ids: [],
  created_at: '2026-08-20T00:00:00+00:00',
  updated_at: '2026-08-20T00:00:00+00:00',
  completed_at: null,
  revision: 1,
}

describe('PlanCard', () => {
  it('按序展示计划并与权限审批保持独立视觉', () => {
    const html = renderToStaticMarkup(
      <PlanCard task={task} onAccept={() => {}} onReject={() => {}} />,
    )
    expect(html).toContain('Refactor Web UI')
    expect(html).toContain('Create a focused agent workspace.')
    expect(html).toContain('Inspect the shell')
    expect(html).toContain('Build the chat workspace')
    expect(html).toContain('接受并普通执行')
    expect(html).toContain('拒绝计划')
    expect(html).not.toContain('Approval required')
    expect(html).not.toContain('接受并长任务执行')
  })

  it('提供长任务入口时，步骤缺验收标准就不能按长任务执行', () => {
    const html = renderToStaticMarkup(
      <PlanCard task={task} onAccept={() => {}} onReject={() => {}} onAcceptLongTask={() => {}} />,
    )
    expect(html).toContain('接受并长任务执行')
    expect(html).toContain('2 个步骤缺少验收标准')
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>接受并长任务执行<\/button>/)
  })

  it('步骤都有验收标准时显示验收并允许长任务执行', () => {
    const planned: Task = {
      ...task,
      steps: task.steps.map((step) => ({ ...step, acceptance: `${step.title} 已验证` })),
    }
    const html = renderToStaticMarkup(
      <PlanCard task={planned} onAccept={() => {}} onReject={() => {}} onAcceptLongTask={() => {}} />,
    )
    expect(html).toContain('验收：Inspect the shell 已验证')
    expect(html).not.toContain('缺少验收标准')
    expect(html).not.toMatch(/<button[^>]*disabled=""[^>]*>接受并长任务执行<\/button>/)
  })
})
