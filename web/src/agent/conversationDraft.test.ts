import { describe, expect, it, vi } from 'vitest'

import { CONVERSATION_DRAFT_PREFIX, createConversationDraftStore } from './conversationDraft'

function memoryStorage() {
  const values = new Map<string, string>()
  return {
    values,
    getItem: vi.fn((key: string) => values.get(key) ?? null),
    setItem: vi.fn((key: string, value: string) => { values.set(key, value) }),
    removeItem: vi.fn((key: string) => { values.delete(key) }),
  }
}

describe('conversation drafts', () => {
  it('restores each conversation content and mode after leaving or reloading the workspace', () => {
    const storage = memoryStorage()
    const firstVisit = createConversationDraftStore(storage)
    firstVisit.set('work-a', { content: '未发送的说明\n继续编辑', mode: 'plan' })
    firstVisit.set('work-b', { content: '另一份说明', mode: 'normal' })
    firstVisit.update('work-a', { content: '修改后的说明' })

    const reopened = createConversationDraftStore(storage)
    expect(reopened.get('work-a')).toEqual({ content: '修改后的说明', mode: 'plan' })
    expect(reopened.get('work-b')).toEqual({ content: '另一份说明', mode: 'normal' })
    expect(reopened.get('new-work')).toEqual({ content: '', mode: 'normal' })
  })

  it('stores only user content and mode under a conversation-specific key', () => {
    const storage = memoryStorage()
    const drafts = createConversationDraftStore(storage)
    drafts.set('work:a/b', { content: '说明', mode: 'plan', internalState: 'ignored' } as { content: string; mode: 'plan' })
    expect(storage.setItem).toHaveBeenCalledWith(`${CONVERSATION_DRAFT_PREFIX}work%3Aa%2Fb`, JSON.stringify({ content: '说明', mode: 'plan' }))
  })

  it('keeps the draft available during a send and after failure, and clears only the successful conversation', () => {
    const storage = memoryStorage()
    const drafts = createConversationDraftStore(storage)
    const sent = { content: '第一项工作', mode: 'normal' as const }
    drafts.set('work-a', sent)
    drafts.set('work-b', { content: '第二项工作', mode: 'plan' })
    expect(createConversationDraftStore(storage).get('work-a')).toEqual(sent)

    drafts.clearSent('work-a', sent)
    const reopened = createConversationDraftStore(storage)
    expect(reopened.get('work-a').content).toBe('')
    expect(reopened.get('work-b')).toEqual({ content: '第二项工作', mode: 'plan' })
  })

  it('does not erase edits or mode changes made while an earlier request was in flight', () => {
    const storage = memoryStorage()
    const drafts = createConversationDraftStore(storage)
    const sent = { content: '已发送', mode: 'normal' as const }
    drafts.set('work-a', sent)
    drafts.update('work-a', { content: '新的编辑' })
    drafts.clearSent('work-a', sent)
    expect(drafts.get('work-a').content).toBe('新的编辑')

    drafts.set('work-a', sent)
    drafts.update('work-a', { mode: 'plan' })
    drafts.clearSent('work-a', sent)
    expect(drafts.get('work-a')).toEqual({ content: '已发送', mode: 'plan' })
  })

  it('retains the selected mode after a successfully sent plan', () => {
    const storage = memoryStorage()
    const drafts = createConversationDraftStore(storage)
    const sent = { content: '制定方案', mode: 'plan' as const }
    drafts.set('work-a', sent)
    drafts.clearSent('work-a', sent)
    expect(createConversationDraftStore(storage).get('work-a')).toEqual({ content: '', mode: 'plan' })
  })

  it('migrates an unassigned draft once without writing a null conversation key', () => {
    const storage = memoryStorage()
    const drafts = createConversationDraftStore(storage)
    drafts.set(null, { content: '尚未创建会话的输入', mode: 'plan' })
    expect(storage.setItem).not.toHaveBeenCalled()
    drafts.adoptUnassigned('new-work')
    expect(drafts.get('new-work')).toEqual({ content: '尚未创建会话的输入', mode: 'plan' })
    expect(drafts.get(null)).toEqual({ content: '', mode: 'normal' })
    drafts.adoptUnassigned('another-work')
    expect(drafts.get('another-work').content).toBe('')
  })

  it('never overwrites a saved conversation when adopting unassigned input', () => {
    const drafts = createConversationDraftStore(memoryStorage())
    drafts.set(null, { content: '未分配输入', mode: 'normal' })
    drafts.set('existing-work', { content: '已经保存的草稿', mode: 'plan' })
    drafts.adoptUnassigned('existing-work')
    expect(drafts.get('existing-work')).toEqual({ content: '已经保存的草稿', mode: 'plan' })
  })

  it('falls back safely for invalid storage and unsupported fields', () => {
    const storage = memoryStorage()
    for (const raw of ['{', 'null', '[]', '"string"']) {
      storage.values.set(`${CONVERSATION_DRAFT_PREFIX}work-a`, raw)
      expect(createConversationDraftStore(storage).get('work-a')).toEqual({ content: '', mode: 'normal' })
    }
    storage.values.set(`${CONVERSATION_DRAFT_PREFIX}work-a`, JSON.stringify({ content: 3, mode: 'audit' }))
    expect(createConversationDraftStore(storage).get('work-a')).toEqual({ content: '', mode: 'normal' })
  })

  it('continues editing in memory if browser storage is denied', () => {
    const storage = {
      getItem: vi.fn(() => { throw new Error('denied') }),
      setItem: vi.fn(() => { throw new Error('denied') }),
      removeItem: vi.fn(() => { throw new Error('denied') }),
    }
    const drafts = createConversationDraftStore(storage)
    expect(drafts.get('work-a')).toEqual({ content: '', mode: 'normal' })
    drafts.set('work-a', { content: '当前窗口继续编辑', mode: 'plan' })
    expect(drafts.get('work-a')).toEqual({ content: '当前窗口继续编辑', mode: 'plan' })
    expect(() => drafts.clearSent('work-a', drafts.get('work-a'))).not.toThrow()
  })
})
