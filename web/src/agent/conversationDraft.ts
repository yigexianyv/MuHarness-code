export interface ConversationDraft {
  content: string
  mode: 'normal' | 'plan'
}

export type ConversationDraftStorage = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>

export const CONVERSATION_DRAFT_PREFIX = 'muharness.conversation-draft.v1:'

function browserStorage(): ConversationDraftStorage | undefined {
  try {
    return typeof window === 'undefined' ? undefined : window.localStorage
  } catch {
    return undefined
  }
}

const emptyDraft = (): ConversationDraft => ({ content: '', mode: 'normal' })

/** Keep a separate copy per conversation, including while a send is in flight. */
export function createConversationDraftStore(storage = browserStorage()) {
  const drafts = new Map<string | null, ConversationDraft>()
  const key = (conversationId: string): string => `${CONVERSATION_DRAFT_PREFIX}${encodeURIComponent(conversationId)}`

  function get(conversationId: string | null): ConversationDraft {
    const cached = drafts.get(conversationId)
    if (cached) return cached
    let draft = emptyDraft()
    try {
      const raw = conversationId ? storage?.getItem(key(conversationId)) : null
      const saved: unknown = raw ? JSON.parse(raw) : null
      if (saved && typeof saved === 'object' && !Array.isArray(saved)) {
        const values = saved as Record<string, unknown>
        draft = {
          content: typeof values.content === 'string' ? values.content : '',
          mode: values.mode === 'plan' ? 'plan' : 'normal',
        }
      }
    } catch {
      // Storage denial or an invalid saved draft must not prevent editing.
    }
    drafts.set(conversationId, draft)
    return draft
  }

  function set(conversationId: string | null, draft: ConversationDraft): void {
    const next = { content: draft.content, mode: draft.mode }
    drafts.set(conversationId, next)
    if (!conversationId || !storage) return
    try {
      if (!next.content && next.mode === 'normal') storage.removeItem(key(conversationId))
      else storage.setItem(key(conversationId), JSON.stringify(next))
    } catch {
      // The in-memory copy remains usable when browser persistence is unavailable.
    }
  }

  function update(conversationId: string | null, patch: Partial<ConversationDraft>): void {
    set(conversationId, { ...get(conversationId), ...patch })
  }

  function clearSent(conversationId: string, sent: ConversationDraft): void {
    const current = get(conversationId)
    // An earlier send completion must not erase newer edits in this conversation.
    if (current.content === sent.content && current.mode === sent.mode) {
      set(conversationId, { content: '', mode: current.mode })
    }
  }

  function adoptUnassigned(conversationId: string): void {
    const unassigned = get(null)
    const target = get(conversationId)
    if (!target.content && (unassigned.content || unassigned.mode !== 'normal')) {
      set(conversationId, unassigned)
      set(null, emptyDraft())
    }
  }

  return { get, set, update, clearSent, adoptUnassigned }
}
