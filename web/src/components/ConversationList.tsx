import { useEffect, useId, useRef, useState } from 'react'
import type { Conversation } from '../api/types'
import { Icon } from './Icon'

const PINNED_KEY = 'muharness.pinnedConversations'

function loadPinned(): string[] {
  try {
    const raw = localStorage.getItem(PINNED_KEY)
    const parsed = raw ? JSON.parse(raw) : []
    const pinned = Array.isArray(parsed)
      ? parsed.filter((x): x is string => typeof x === 'string')
      : []
    return pinned
  } catch {
    return []
  }
}

const STATUS_META: Record<string, { label: string; tone: string }> = {
  running: { label: '工作中', tone: 'running' },
  pending: { label: '等待中', tone: 'waiting' },
  completed: { label: '已完成', tone: 'completed' },
  failed: { label: '失败', tone: 'failed' },
  cancelled: { label: '已取消', tone: 'cancelled' },
  interrupted: { label: '已停止', tone: 'failed' },
}

function relativeTime(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return ''
  const diff = Date.now() - date.getTime()
  const minutes = Math.floor(diff / 60_000)
  if (minutes < 1) return '刚刚'
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h`
  const days = Math.floor(hours / 24)
  return `${days}d`
}

export default function ConversationList({
  conversations,
  selectedId,
  onSelect,
  onNew,
  onRename,
  onDelete,
  statusByConversation = {},
  activityByConversation = {},
}: {
  conversations: Conversation[]
  selectedId: string | null
  onSelect: (id: string) => void
  onNew: () => void

  onRename?: (id: string, title: string) => void | Promise<void>

  onDelete?: (id: string) => void | Promise<void>

  statusByConversation?: Record<string, string>

  activityByConversation?: Record<string, string>
}): React.JSX.Element {
  const [pinned, setPinned] = useState<string[]>(loadPinned)
  const [searchQuery, setSearchQuery] = useState('')
  const [menuFor, setMenuFor] = useState<string | null>(null)
  const menuRef = useRef<HTMLDivElement | null>(null)
  const searchRef = useRef<HTMLInputElement | null>(null)
  const searchId = useId()
  const summaryId = useId()


  useEffect(() => {
    if (menuFor === null) return
    const onPointer = (event: MouseEvent): void => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setMenuFor(null)
      }
    }
    document.addEventListener('mousedown', onPointer)
    return () => document.removeEventListener('mousedown', onPointer)
  }, [menuFor])

  const togglePin = (id: string): void => {
    setPinned((prev) => {
      const next = prev.includes(id)
        ? prev.filter((p) => p !== id)
        : [id, ...prev]
      try {
        localStorage.setItem(PINNED_KEY, JSON.stringify(next))
      } catch {

      }
      return next
    })
  }


  const ordered = [...conversations].sort((a, b) => {
    const ai = pinned.includes(a.id) ? 0 : 1
    const bi = pinned.includes(b.id) ? 0 : 1
    return ai - bi
  })

  const query = searchQuery.trim().toLocaleLowerCase()
  const visible = query
    ? ordered.filter((conversation) =>
      (conversation.title || '未命名对话').toLocaleLowerCase().includes(query),
    )
    : ordered
  const firstUnpinnedIndex = visible.findIndex((c) => !pinned.includes(c.id))


  const [editingId, setEditingId] = useState<string | null>(null)
  const [editingValue, setEditingValue] = useState('')

  const startRename = (id: string, current: string): void => {
    setEditingValue(current)
    setEditingId(id)
    setMenuFor(null)
  }

  const commitRename = (id: string, original: string): void => {
    setEditingId(null)
    const trimmed = editingValue.trim()
    if (!trimmed || trimmed === original) return
    void onRename?.(id, trimmed)
  }

  const runDelete = (id: string): void => {

    void onDelete?.(id)
    setMenuFor(null)
  }

  return (
    <div className="conversation-sidebar__content">
      <div className="conversation-sidebar__top">
        <div className="conversation-sidebar__label">工作</div>
        <button
          className="new-conversation"
          type="button"
          onClick={() => {
            setSearchQuery('')
            setMenuFor(null)
            onNew()
          }}
        >
          <Icon name="plus" size={14} />
          新建
        </button>
      </div>
      <div className="conversation-search">
        <label className="conversation-search__label" htmlFor={searchId}>检索会话</label>
        <div className="conversation-search__field">
          <svg className="conversation-search__icon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true" focusable="false">
            <circle cx="10.5" cy="10.5" r="6.5" />
            <path d="m16 16 4.5 4.5" />
          </svg>
          <input
            ref={searchRef}
            id={searchId}
            type="search"
            placeholder="搜索会话标题"
            value={searchQuery}
            aria-describedby={summaryId}
            autoComplete="off"
            onChange={(event) => {
              setSearchQuery(event.target.value)
              setMenuFor(null)
            }}
          />
          {searchQuery ? (
            <button
              type="button"
              className="conversation-search__clear"
              aria-label="清空检索"
              title="清空检索"
              onClick={() => {
                setSearchQuery('')
                searchRef.current?.focus()
              }}
            >
              <Icon name="close" size={13} />
            </button>
          ) : null}
        </div>
      </div>
      <div className="conversation-list__summary" id={summaryId} role="status" aria-live="polite">
        {query ? `找到 ${visible.length} 个会话` : `${conversations.length} 个会话`}
      </div>
      <div className="conversation-list">
        {conversations.length === 0 ? (
          <div className="conversation-list__empty">暂无对话</div>
        ) : visible.length === 0 ? (
          <div className="conversation-list__empty">
            <span className="conversation-list__empty-title">没有匹配的会话</span>
            <span className="conversation-list__empty-detail">试试其他标题关键词</span>
          </div>
        ) : null}
        {visible.map((conversation, index) => {
          const status = statusByConversation[conversation.id]
          const meta = status ? STATUS_META[status] : undefined
          const activity = activityByConversation[conversation.id]
          const isPinned = pinned.includes(conversation.id)
          const showDivider = index === firstUnpinnedIndex && firstUnpinnedIndex > 0
          return (
            <div key={conversation.id} className="conversation-list__row">
              {showDivider ? (
                <div className="conversation-list__divider" aria-hidden="true" />
              ) : null}
              <div
                className={`conversation-item ${selectedId === conversation.id ? 'active' : ''}`}
              >
              <button
                type="button"
                className="conversation-item__main"
                onClick={() => onSelect(conversation.id)}
                aria-current={selectedId === conversation.id ? 'true' : undefined}
              >
                {editingId === conversation.id ? (
                  <input
                    className="conversation-item__edit"
                    value={editingValue}
                    autoFocus
                    aria-label="编辑会话标题"
                    onClick={(event) => event.stopPropagation()}
                    onChange={(event) => setEditingValue(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter') {
                        event.preventDefault()
                        commitRename(conversation.id, conversation.title || '')
                      } else if (event.key === 'Escape') {
                        setEditingId(null)
                      }
                    }}
                    onBlur={() => {
                      if (editingId === conversation.id) {
                        commitRename(conversation.id, conversation.title || '')
                      }
                    }}
                  />
                ) : (
                  <span className="conversation-item__title">
                    {isPinned ? <Icon name="pin" size={11} /> : null}
                    {conversation.title || '未命名对话'}
                  </span>
                )}
                <span className="conversation-item__meta">
                  {meta ? (
                    <span
                      className={`conversation-item__status conversation-item__status--${meta.tone}`}
                    >
                      {status === 'completed' ? '✓' : meta.label}
                    </span>
                  ) : null}
                  <span>{relativeTime(conversation.updated_at)}</span>
                </span>
                {activity && ['running', 'pending'].includes(status ?? '') ? (
                  <span className="conversation-item__activity">{activity}</span>
                ) : null}
              </button>
              <button
                type="button"
                className="conversation-item__more"
                aria-label="更多操作"
                title="更多操作"
                onClick={(event) => {
                  event.stopPropagation()
                  setMenuFor((cur) => (cur === conversation.id ? null : conversation.id))
                }}
              >
                <Icon name="more" size={15} />
              </button>
              {menuFor === conversation.id ? (
                <div className="conversation-menu" ref={menuRef} role="menu">
                  <button
                    type="button"
                    role="menuitem"
                    onClick={() => togglePin(conversation.id)}
                  >
                    <Icon name="pin" size={13} />
                    {isPinned ? '取消置顶' : '置顶'}
                  </button>
                  <button
                    type="button"
                    role="menuitem"
                    onClick={() => startRename(conversation.id, conversation.title || '')}
                  >
                    <Icon name="pencil" size={13} />
                    编辑
                  </button>
                  <button
                    type="button"
                    role="menuitem"
                    className="conversation-menu__danger"
                    onClick={() => runDelete(conversation.id)}
                  >
                    <Icon name="trash" size={13} />
                    删除
                  </button>
                </div>
              ) : null}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
