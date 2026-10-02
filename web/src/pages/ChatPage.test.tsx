import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import { createConversationDraftStore } from '../agent/conversationDraft'
import type { Task } from '../api/types'
import ChatPage from './ChatPage'

const pendingTask: Task = {
  id: 'wordcount-plan',
  title: '创建 wordcount',
  description: null,
  goal: '只在 wordcount 内创建工具并通过 unittest',
  status: 'pending',
  priority: 'normal',
  constraints: [],
  state: [],
  key_facts: [],
  steps: [{ id: 's1', title: '编写工具', status: 'todo', note: null, acceptance: '测试通过' }],
  owner_conversation_id: 'conversation-1',
  run_ids: [],
  created_at: '2026-09-29T01:00:00Z',
  updated_at: '2026-09-29T01:00:00Z',
  completed_at: null,
  revision: 1,
}

function renderConversation(tasks: Task[], conversationId = 'conversation-1'): string {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: Infinity } } })
  client.setQueryData(['tasks', conversationId], tasks)
  try {
    return renderToStaticMarkup(
      <QueryClientProvider client={client}>
        <ChatPage initialConversationId={conversationId} />
      </QueryClientProvider>,
    )
  } finally {
    client.clear()
  }
}

describe('ChatPage persisted plan', () => {
  it('重新打开会话时从已保存的 pending 任务恢复两个接受入口', () => {
    const html = renderConversation([pendingTask])
    expect(html).toContain('data-testid="plan-card"')
    expect(html).toContain('接受并普通执行')
    expect(html).toContain('接受并长任务执行')
    expect(html).not.toMatch(/<button[^>]*disabled=""[^>]*>接受并长任务执行<\/button>/)
  })

  it('已经接受或拒绝的计划不再显示接受卡片', () => {
    for (const status of ['active', 'completed', 'cancelled'] as const) {
      expect(renderConversation([{ ...pendingTask, status }])).not.toContain('data-testid="plan-card"')
    }
  })

  it('不会显示其他会话的待确认计划', () => {
    expect(renderConversation([pendingTask], 'conversation-2')).not.toContain('data-testid="plan-card"')
  })
})

describe('ChatPage conversation drafts', () => {
  it('reopens the matching saved draft and mode without leaking another conversation input', () => {
    const values = new Map<string, string>()
    const storage = {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => { values.set(key, value) },
      removeItem: (key: string) => { values.delete(key) },
    }
    vi.stubGlobal('window', { localStorage: storage, innerWidth: 1024 })
    const drafts = createConversationDraftStore(storage)
    drafts.set('conversation-1', { content: '保留这项工作说明', mode: 'plan' })
    drafts.set('conversation-2', { content: '另一项工作的输入', mode: 'normal' })
    const first = renderConversation([], 'conversation-1')
    expect(first).toContain('>保留这项工作说明</textarea>')
    expect(first).toContain('先给出方案，确认后执行')
    expect(first).not.toContain('另一项工作的输入')
    const second = renderConversation([], 'conversation-2')
    expect(second).toContain('>另一项工作的输入</textarea>')
    expect(second).not.toContain('保留这项工作说明')
  })
})
