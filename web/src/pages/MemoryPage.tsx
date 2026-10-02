

import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { listMemories } from '../api/memories'
import type { LongTermMemory } from '../api/types'
import { EmptyState, ErrorState, LoadingState } from '../components/PageStates'
import { PageShell } from '../components/PageShell'

type MemoryView = 'active' | 'archived'

interface CoreMemoryDisplayItem {
  label: string
  content: string
}

function formatTime(value: string): string {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN')
}

function coreCategory(key: string): string {
  const normalized = key.toLowerCase()
  if (normalized.includes('preference')) return '偏好'
  if (normalized.includes('identity') || normalized.includes('profile')) return '身份信息'
  if (normalized.includes('constraint') || normalized.includes('rule')) return '长期约束'
  if (normalized.includes('communication') || normalized.includes('language')) return '沟通方式'
  if (normalized.includes('location') || normalized.includes('timezone')) return '常用环境'
  return '核心信息'
}

function cleanCoreLine(line: string): string {
  return line
    .replace(/^\s*[-*+]\s+/, '')
    .replace(/^\s*>\s?/, '')
    .trim()
}


export function parseCoreMemory(content: string): CoreMemoryDisplayItem[] {
  const items: CoreMemoryDisplayItem[] = []
  const legacy: string[] = []
  let key: string | null = null
  let values: string[] = []

  const flush = (): void => {
    const value = values.map(cleanCoreLine).filter(Boolean).join('\n')
    if (key && value) items.push({ label: coreCategory(key), content: value })
    key = null
    values = []
  }

  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim()
    if (!line || line === '```') continue
    if (line === '# Core Memory' || line === '## Managed Core Entries') continue
    const managedHeading = line.match(/^###\s+(.+)$/)
    if (managedHeading) {
      flush()
      key = managedHeading[1].trim()
      continue
    }
    if (key) values.push(rawLine)
    else {
      const readable = cleanCoreLine(rawLine.replace(/^#{1,6}\s+/, ''))
      if (readable) legacy.push(readable)
    }
  }
  flush()
  if (legacy.length > 0) {
    items.unshift({ label: '固定信息', content: legacy.join('\n') })
  }
  return items
}

export function CoreMemoryView({ content }: { content: string }): React.JSX.Element {
  const items = parseCoreMemory(content)
  return (
    <div className="memory-core__content">
      <div className="memory-core__visual" aria-hidden="true"><span /></div>
      <div className="memory-core__body">
        <div className="memory-core__summary">
          <strong>始终随身携带</strong>
          <span>{items.length} 条信息会进入每次运行</span>
        </div>
        <ol className="memory-core__entries">
          {items.map((item, index) => (
            <li key={`${item.label}-${index}`}>
              <span className="memory-core__node" aria-hidden="true" />
              <div><small>{item.label}</small><p>{item.content}</p></div>
            </li>
          ))}
        </ol>
      </div>
    </div>
  )
}

export function MemoryCard({ memory }: { memory: LongTermMemory }): React.JSX.Element {
  return (
    <article className={`memory-card memory-card--${memory.status}`}>
      <div className="memory-card__identity">
        <span className="memory-card__mark" aria-hidden="true" />
        <div>
          <header className="memory-card__header">
            <span className="mono">{memory.id}</span>
            <span>{memory.status === 'active' ? '可检索' : '已归档'}</span>
          </header>
          <h3>{memory.title}</h3>
          <p>{memory.summary}</p>
        </div>
      </div>
      <dl className="memory-card__meta">
        <div><dt>版本</dt><dd>r{memory.revision}</dd></div>
        <div title="Agent 成功调用 memory_read 并读取完整正文的次数">
          <dt>正文读取</dt><dd>{memory.access_count} 次</dd>
        </div>
        <div><dt>更新</dt><dd>{formatTime(memory.updated_at)}</dd></div>
      </dl>
      <details className="memory-card__details">
        <summary>查看完整内容</summary>
        <div className="memory-card__content">{memory.content}</div>
        {memory.last_update_reason ? <small>更新原因：{memory.last_update_reason}</small> : null}
        {memory.archive_reason ? <small>归档原因：{memory.archive_reason}</small> : null}
      </details>
    </article>
  )
}

export default function MemoryPage(): React.JSX.Element {
  const [view, setView] = useState<MemoryView>('active')
  const [search, setSearch] = useState('')
  const query = useQuery({
    queryKey: ['memories'],
    queryFn: listMemories,
    refetchInterval: 10_000,
  })
  const data = query.data
  const memories = view === 'active' ? (data?.active ?? []) : (data?.archived ?? [])
  const searchText = search.trim().toLocaleLowerCase()
  const visibleMemories = memories.filter((memory) => (
    !searchText || `${memory.id}\n${memory.title}\n${memory.summary}\n${memory.content}`
      .toLocaleLowerCase().includes(searchText)
  ))

  return (
    <PageShell
      title="长期记忆"
      subtitle="让偏好与经验跨越会话。查看常驻信息，以及需要时才读取的记忆。"
      maxWidth={1360}
      actions={
        <div className="segmented-control" aria-label="记忆筛选">
          <button type="button" aria-pressed={view === 'active'} className={view === 'active' ? 'active' : ''} onClick={() => setView('active')}>
            使用中 {data ? `(${data.active_count})` : ''}
          </button>
          <button type="button" aria-pressed={view === 'archived'} className={view === 'archived' ? 'active' : ''} onClick={() => setView('archived')}>
            已归档 {data ? `(${data.archived.length})` : ''}
          </button>
        </div>
      }
    >
      {query.isPending ? <LoadingState label="正在加载长期记忆…" />
        : query.isError ? <ErrorState message={String(query.error)} onRetry={() => void query.refetch()} />
          : data ? (
            <div className="memory-page">
              <section className="memory-overview" aria-label="长期记忆概览">
                <div className="memory-overview__intro">
                  <span className="memory-overview__eyebrow">跨会话知识</span>
                  <strong>少量常驻，按需回忆</strong>
                  <p>核心记忆随每次运行进入上下文；普通记忆只提供索引，由模型在需要时读取。</p>
                </div>
                <dl className="memory-overview__stats">
                  <div><dt>使用中</dt><dd>{data.active_count}</dd></div>
                  <div><dt>容量</dt><dd>{data.max_active || '—'}</dd></div>
                  <div><dt>已归档</dt><dd>{data.archived.length}</dd></div>
                </dl>
              </section>

              <section className="memory-core">
                <div className="section-heading">
                  <div><h2>核心记忆</h2><p>稳定偏好与长期约束，每次运行都会携带</p></div>
                  <span className="memory-core__badge">常驻上下文</span>
                </div>
                {data.core.trim() ? (
                  <CoreMemoryView content={data.core} />
                ) : (
                  <p className="memory-empty-hint">暂无核心记忆。当用户明确表达稳定偏好或长期约束后，系统会在这里保留。</p>
                )}
              </section>

              <section className="memory-list-section">
                <div className="section-heading">
                  <div>
                    <h2>{view === 'active' ? '普通记忆' : '归档记忆'}</h2>
                    <p>{view === 'active' ? `${memories.length} 条可通过索引按需读取` : `${memories.length} 条已退出索引但仍完整保留`}</p>
                  </div>
                </div>
                <div className="page-toolbar">
                  <span className="page-toolbar__summary memory-list-count">
                    {searchText ? `找到 ${visibleMemories.length} 条匹配记忆` : '展开一条记忆，查看完整内容与更新原因。'}
                  </span>
                  <label className="page-search">
                    <span className="sr-only">搜索{view === 'active' ? '普通' : '归档'}记忆</span>
                    <input
                      type="search"
                      className="input"
                      value={search}
                      placeholder="搜索标题、内容或编号"
                      onChange={(event) => setSearch(event.target.value)}
                    />
                  </label>
                </div>
                {memories.length === 0 ? (
                  <EmptyState
                    title={view === 'active' ? '暂无普通记忆' : '暂无归档记忆'}
                    hint={view === 'active' ? '有长期价值的信息会在 Run 完成后由系统整理。' : '过期或被替代的记忆会保留在这里。'}
                    icon="memory"
                  />
                ) : visibleMemories.length === 0 ? (
                  <EmptyState title="没有匹配的记忆" hint="尝试其他关键词，或切换记忆分组。" icon="memory" />
                ) : (
                  <div className="memory-grid">
                    {visibleMemories.map((memory) => <MemoryCard key={memory.id} memory={memory} />)}
                  </div>
                )}
              </section>
            </div>
          ) : null}
    </PageShell>
  )
}
