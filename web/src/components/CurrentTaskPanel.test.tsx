

import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'

import type { Task } from '../api/types'
import CurrentTaskPanel, { orderConversationTasks } from './CurrentTaskPanel'

function task(status: Task['status'], title: string, updatedAt: string): Task {
  return {
    id: `${title}-id`,
    title,
    description: null,
    goal: `${title}目标`,
    status,
    priority: 'normal',
    constraints: [],
    state: [],
    key_facts: [],
    steps: [
      { id: 's1', title: '检查现状', status: 'done', note: '已完成' },
      { id: 's2', title: '继续实现', status: 'in_progress', note: null },
    ],
    owner_conversation_id: 'conv-1',
    run_ids: [],
    created_at: updatedAt,
    updated_at: updatedAt,
    completed_at: null,
    revision: 1,
  }
}

describe('CurrentTaskPanel', () => {
  it('优先展示活动任务与当前步骤', () => {
    const completed = task('completed', '旧任务', '2026-08-22T10:00:00Z')
    const active = task('active', '当前任务', '2026-08-22T09:00:00Z')
    expect(orderConversationTasks([completed, active])[0]).toBe(active)

    const html = renderToStaticMarkup(
      <CurrentTaskPanel tasks={[completed, active]} onExecute={() => {}} />,
    )
    expect(html).toContain('当前任务 · 进行中')
    expect(html).toContain('当前步骤：继续实现')
    expect(html).toContain('1/2')
    expect(html).toContain('2 个任务')
    expect(html).toContain('继续剩余阶段')
  })

  it('没有任务时不渲染面板', () => {
    expect(renderToStaticMarkup(<CurrentTaskPanel tasks={[]} />)).toBe('')
  })
})


it('展示两个明确入口和覆盖提示，已完成任务只提供重做', () => {
  const active = task('active', '任务', '2026-10-05')
  const html = renderToStaticMarkup(
    <CurrentTaskPanel tasks={[active]} onExecute={() => {}} onRestart={() => {}} />,
  )
  expect(html).toContain('继续剩余阶段')
  expect(html).toContain('从阶段 1 重新执行')
  expect(html).toContain('可能覆盖工作区同名文件')
  const completed = renderToStaticMarkup(
    <CurrentTaskPanel tasks={[{ ...active, status: 'completed' }]}
      onExecute={() => {}} onRestart={() => {}} />,
  )
  expect(completed).not.toContain('继续剩余阶段')
  expect(completed).toContain('从阶段 1 重新执行')
})

it('有未结束长任务时禁止重复启动重做', () => {
  const active = task('active', '任务', '2026-10-05')
  active.steps = active.steps.map((step) => ({ ...step, acceptance: 'verified' }))
  const html = renderToStaticMarkup(
    <CurrentTaskPanel tasks={[active]} lockedTaskIds={[active.id]}
      onExecute={() => {}} onRestart={() => {}} />,
  )
  expect(html).not.toContain('继续剩余阶段')
  expect(html).toMatch(/disabled=""[^>]*>从阶段 1 重新执行/)
})
