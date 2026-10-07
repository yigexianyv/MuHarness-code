

import { afterEach, describe, expect, it, vi } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'

import type { Conversation } from '../api/types'
import ConversationList from './ConversationList'

const conversation: Conversation = {
  id: 'conv-1',
  title: 'Web redesign',
  message_count: 12,
  created_at: '2026-08-20T00:00:00+00:00',
  updated_at: '2026-08-20T01:00:00+00:00',
}

describe('ConversationList', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('渲染新建入口、标题、状态与选中态', () => {
    const html = renderToStaticMarkup(
      <ConversationList
        conversations={[conversation]}
        selectedId="conv-1"
        statusByConversation={{ 'conv-1': 'running' }}
        onSelect={() => {}}
        onNew={() => {}}
      />,
    )
    expect(html).toContain('工作')
    expect(html).toContain('Web redesign')
    expect(html).toContain('工作中')
    expect(html).toContain('conversation-item__status--running')
    expect(html).toContain('conversation-item active')
    expect(html).not.toContain('12 messages')
  })

  it('空列表提供轻量空状态', () => {
    const html = renderToStaticMarkup(
      <ConversationList conversations={[]} selectedId={null} onSelect={() => {}} onNew={() => {}} />,
    )
    expect(html).toContain('暂无对话')
    expect(html).toContain('0 个会话')
    expect(html).not.toContain('没有匹配的会话')
  })

  it('提供带标签的标题检索和根据实际列表生成的数量', () => {
    const html = renderToStaticMarkup(
      <ConversationList
        conversations={[conversation, { ...conversation, id: 'conv-2', title: '接口调试' }]}
        selectedId={null}
        onSelect={() => {}}
        onNew={() => {}}
      />,
    )
    expect(html).toMatch(/<label[^>]+for="[^"]+">检索会话<\/label>/)
    expect(html).toContain('type="search"')
    expect(html).toContain('placeholder="搜索会话标题"')
    expect(html).toContain('aria-describedby=')
    expect(html).toContain('role="status" aria-live="polite">2 个会话')
    expect(html).toContain('Web redesign')
    expect(html).toContain('接口调试')
    expect(html).not.toContain('清空检索')
  })

  it('突出 working / approval / failed，弱化 completed，并展示当前动作', () => {
    const conversations = ['running', 'pending', 'failed', 'completed'].map((status) => ({
      ...conversation,
      id: status,
      title: status,
    }))
    const html = renderToStaticMarkup(
      <ConversationList
        conversations={conversations}
        selectedId="running"
        statusByConversation={{ running: 'running', pending: 'pending', failed: 'failed', completed: 'completed' }}
        activityByConversation={{ running: 'Typing in Notes', pending: 'Waiting for approval' }}
        onSelect={() => {}}
        onNew={() => {}}
      />,
    )
    expect(html).toContain('Typing in Notes')
    expect(html).toContain('Waiting for approval')
    expect(html).toContain('工作中')
    expect(html).toContain('等待中')
    expect(html).toContain('失败')
    expect(html).toContain('conversation-item__status--completed">✓')
    expect(html).not.toContain('>已完成</span>')
  })

  it('仅从 MuHarness 存储键恢复置顶', () => {
    const getItem = vi.fn((key: string) =>
      key === 'muharness.pinnedConversations' ? '["pinned"]' : null,
    )
    const setItem = vi.fn()
    vi.stubGlobal('localStorage', { getItem, setItem })

    const html = renderToStaticMarkup(
      <ConversationList
        conversations={[
          { ...conversation, id: 'normal', title: '普通对话' },
          { ...conversation, id: 'pinned', title: '置顶对话' },
        ]}
        selectedId={null}
        onSelect={() => {}}
        onNew={() => {}}
      />,
    )

    expect(html.indexOf('置顶对话')).toBeLessThan(html.indexOf('普通对话'))
    expect(getItem).toHaveBeenCalledWith('muharness.pinnedConversations')
    expect(getItem).toHaveBeenCalledTimes(1)
    expect(setItem).not.toHaveBeenCalled()
  })
})
