import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'

import type { InboxItem } from '../agent/hub'
import { buildWorkOverview, decisionTitle, filterWorkOverview, WORK_LANES } from '../agent/workOverview'
import type { WorkFilter, WorkItem } from '../agent/workOverview'
import { buildArtifactDownloadUrl, listArtifacts } from '../api/artifacts'
import type { Artifact } from '../api/artifacts'
import { listAutomations } from '../api/automations'
import { listConversations } from '../api/conversations'
import { listRuns } from '../api/runs'
import type { MeaRun } from '../api/types'
import { Icon } from '../components/Icon'

interface WorkOverviewProps {
  onOpenConversation: (id: string) => void
  onOpenRun: (id: string) => void
  onNewWork: () => void
  onNavigate: (page: 'inbox' | 'automations' | 'ledger') => void
  onOpenArtifacts: () => void
  inboxItems: InboxItem[]
  meas: MeaRun[]
  connected: boolean
  inboxPending?: boolean
  inboxError?: string | null
  onRefreshInbox?: () => void
}

function formatTime(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '时间未知'
  return new Intl.DateTimeFormat('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' }).format(date)
}

function artifactSize(artifact: Artifact): string {
  if (artifact.kind === 'url') return '网页链接'
  return artifact.size_bytes >= 1024 ? `${(artifact.size_bytes / 1024).toFixed(1)} KB` : `${artifact.size_bytes} B`
}

function ArtifactRow({ artifact }: { artifact: Artifact }): React.JSX.Element {
  const href = artifact.kind === 'file' ? buildArtifactDownloadUrl(artifact.id) : artifact.source_url
  const safeHref = href && (artifact.kind === 'file' || /^https?:\/\//i.test(href)) ? href : null
  return (
    <div className="desk-focus__artifact">
      <Icon name={artifact.kind === 'file' ? 'file' : 'external'} size={17} />
      <div>
        {safeHref ? <a href={safeHref} target="_blank" rel="noreferrer">{artifact.title || artifact.filename || '未命名成果'}</a>
          : <strong>{artifact.title || artifact.filename || '未命名成果'}</strong>}
        <span>{artifactSize(artifact)} · {formatTime(artifact.created_at)}</span>
      </div>
    </div>
  )
}

function WorkCard({ item, selected, onSelect }: { item: WorkItem; selected: boolean; onSelect: (element: HTMLButtonElement) => void }): React.JSX.Element {
  return (
    <button
      type="button"
      className={`desk-work-card desk-work-card--${item.lane}${selected ? ' is-selected' : ''}`}
      aria-pressed={selected}
      aria-label={`预览 ${item.title}`}
      onClick={(event) => onSelect(event.currentTarget)}
    >
      <span className="desk-work-card__top"><span>{item.kind === 'mea' ? '长任务' : item.kind === 'run' ? '普通执行' : '会话'}</span><Icon name={item.kind === 'mea' ? 'automations' : 'chat'} size={14} /></span>
      <strong className="desk-work-card__title">{item.title}</strong>
      <span className="desk-work-card__detail">{item.decisions[0] ? decisionTitle(item.decisions[0]) : item.detail}</span>
      <span className="desk-work-card__status"><i aria-hidden="true" />{item.status}</span>
      <span className="desk-work-card__bottom"><time dateTime={item.updatedAt}>{formatTime(item.updatedAt)}</time>{item.artifacts.length > 0 ? <span><Icon name="file" size={12} />{item.artifacts.length} 项近期成果</span> : null}</span>
    </button>
  )
}

export default function WorkOverviewPage({
  onOpenConversation, onOpenRun, onNewWork, onNavigate, onOpenArtifacts,
  inboxItems, meas, connected, inboxPending = false, inboxError = null, onRefreshInbox,
}: WorkOverviewProps): React.JSX.Element {
  const [filter, setFilter] = useState<WorkFilter>('all')
  const [query, setQuery] = useState('')
  const [layout, setLayout] = useState<'board' | 'list'>('board')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const previewRef = useRef<HTMLElement | null>(null)
  const selectionTrigger = useRef<HTMLButtonElement | null>(null)
  const conversations = useQuery({ queryKey: ['conversations'], queryFn: () => listConversations(50), enabled: connected, refetchInterval: 10000 })
  const runs = useQuery({ queryKey: ['runs'], queryFn: () => listRuns({ limit: 100 }), enabled: connected, refetchInterval: 4000 })
  const artifacts = useQuery({ queryKey: ['artifacts', 'overview'], queryFn: () => listArtifacts({ limit: 20 }), enabled: connected, refetchInterval: 10000 })
  const automations = useQuery({ queryKey: ['automations'], queryFn: listAutomations, enabled: connected, refetchInterval: 10000 })
  const work = useMemo(() => buildWorkOverview({
    conversations: conversations.data ?? [], runs: runs.data ?? [], meas, inbox: inboxItems, artifacts: artifacts.data ?? [],
  }), [conversations.data, runs.data, meas, inboxItems, artifacts.data])
  const visible = useMemo(() => filterWorkOverview(work, filter, query), [work, filter, query])
  const selected = visible.find((item) => item.id === selectedId) ?? null
  useEffect(() => {
    if (!selectedId || !previewRef.current) return
    const frame = window.requestAnimationFrame(() => {
      const preview = previewRef.current
      if (!preview) return
      preview.focus({ preventScroll: true })
      if (window.matchMedia('(max-width: 760px)').matches) {
        const overview = preview.closest<HTMLElement>('.desk-overview')
        if (overview) overview.scrollTop += preview.getBoundingClientRect().top - overview.getBoundingClientRect().top
      }
    })
    return () => window.cancelAnimationFrame(frame)
  }, [selectedId])
  const closePreview = (): void => {
    setSelectedId(null)
    selectionTrigger.current?.focus()
  }
  const upcoming = (automations.data ?? []).filter((automation) => automation.status === 'active' && automation.next_run_at)
    .sort((a, b) => (a.next_run_at ?? '').localeCompare(b.next_run_at ?? '')).slice(0, 2)
  // Wait for the initial state sources together; a conversation arriving first
  // must not briefly appear unstarted while its run or decision is still loading.
  const loading = connected && (conversations.isPending || runs.isPending || inboxPending)
  const failures = [conversations.isError && '会话', runs.isError && '执行记录', artifacts.isError && '成果', automations.isError && '自动化', inboxError && '待处理']
    .filter(Boolean)
  const refresh = (): void => {
    void conversations.refetch(); void runs.refetch(); void artifacts.refetch(); void automations.refetch()
    onRefreshInbox?.()
  }
  const openDecision = (item: InboxItem): void => {
    if (item.kind === 'run') onOpenRun(item.run.id)
    else if (item.kind === 'mea') onOpenConversation(item.mea.conversation_id)
    else onNavigate('inbox')
  }

  return (
    <section className="desk-overview" aria-label="工作总览">
      <header className="desk-overview__header">
        <div><span className="desk-overview__eyebrow">YOUR WORK, IN ONE PLACE</span><h1>工作总览<span>从目标到交付，看到每一步。</span></h1></div>
      </header>
      {!connected ? <div className="desk-overview__notice" role="status"><Icon name="alert" size={16} />本地服务离线。已有内容来自缓存，连接恢复后会刷新。</div> : null}
      {failures.length > 0 ? <div className="desk-overview__notice desk-overview__notice--error" role="alert"><span>{failures.join('、')}读取失败，以下仅显示已获取的数据。</span><button type="button" onClick={refresh}>重试读取</button></div> : null}
      <div className={`desk-overview__layout${selected ? ' has-preview' : ''}`}>
        <div className="desk-overview__work">
          <div className="desk-overview__toolbar">
            <div className="desk-overview__filters" role="group" aria-label="工作类型筛选">
              {([['all', '全部工作'], ['long', '长任务'], ['normal', '普通工作']] as const).map(([value, label]) => <button key={value} type="button" aria-pressed={filter === value} className={filter === value ? 'active' : ''} onClick={() => setFilter(value)}>{label}</button>)}
            </div>
            <div className="desk-overview__tools">
              <input type="search" aria-label="搜索工作" placeholder="搜索名称或目标…" value={query} onChange={(event) => setQuery(event.target.value)} />
              <div className="desk-overview__view" role="group" aria-label="总览布局">
                <button type="button" aria-pressed={layout === 'board'} className={layout === 'board' ? 'active' : ''} onClick={() => setLayout('board')}>看板</button>
                <button type="button" aria-pressed={layout === 'list'} className={layout === 'list' ? 'active' : ''} onClick={() => setLayout('list')}>列表</button>
              </div>
            </div>
          </div>
          <p className="desk-overview__scope">最近 50 个会话、100 次执行与已获取的长任务，按会话合并。列内数量来自当前列表，非全库统计。</p>
          {loading ? <div className="desk-overview__loading" role="status"><Icon name="activity" size={24} /><span>正在汇总工作与决定…</span></div> : (
            <div className={`desk-board desk-board--${layout}`}>
              {WORK_LANES.map((lane) => {
                const laneItems = visible.filter((item) => item.lane === lane.id)
                return <section className={`desk-board__lane desk-board__lane--${lane.id}`} key={lane.id} aria-label={lane.title}>
                  <header className="desk-board__heading"><h2><i aria-hidden="true" />{lane.title}<span>{laneItems.length}</span></h2><p>{lane.hint}</p></header>
                  <div className="desk-board__items">{laneItems.length ? laneItems.map((item) => <WorkCard key={item.id} item={item} selected={selected?.id === item.id} onSelect={(element) => { selectionTrigger.current = element; setSelectedId(item.id) }} />)
                    : <p className="desk-board__empty">{query.trim() || filter !== 'all' ? '没有符合筛选的工作。' : lane.empty}</p>}</div>
                </section>
              })}
            </div>
          )}
          {work.length === 0 && !loading && connected && failures.length === 0 ? <div className="desk-overview__first"><h2>还没有工作，先给 Agent 一个目标。</h2><p>普通执行适合快速协作；长任务适合多轮推进并独立验收。</p><button type="button" onClick={onNewWork}>创建第一项工作 <Icon name="plus" size={15} /></button></div> : null}
        </div>
        <aside ref={previewRef} tabIndex={selected ? -1 : undefined} className={selected ? 'desk-preview' : 'desk-focus'} aria-label={selected ? '工作预览' : '工作焦点'}>
          {selected ? <>
            <header className="desk-preview__heading"><span>工作预览</span><button type="button" aria-label="关闭工作预览" onClick={closePreview}><Icon name="close" size={17} /></button></header>
            <span className={`desk-preview__status desk-preview__status--${selected.lane}`}>{selected.status}</span>
            <h2>{selected.title}</h2><p className="desk-preview__detail">{selected.detail}</p>
            <dl className="desk-preview__meta"><div><dt>执行方式</dt><dd>{selected.kind === 'mea' ? '长任务 · 管理 / 执行 / 审计' : selected.run?.mode === 'plan' ? '普通执行 · 规划模式' : selected.kind === 'run' ? '普通执行' : '尚未执行'}</dd></div><div><dt>最近更新</dt><dd>{formatTime(selected.updatedAt)}</dd></div></dl>
            <div className="desk-preview__actions">{selected.conversationId ? <button type="button" onClick={() => onOpenConversation(selected.conversationId!)}><Icon name="chat" size={15} />进入会话</button> : null}{selected.run ? <button type="button" onClick={() => onOpenRun(selected.run!.id)}><Icon name="runs" size={15} />查看执行</button> : null}</div>
            {selected.decisions.length > 0 ? <section className="desk-preview__section"><h3>等待你的决定</h3>{selected.decisions.map((item) => <button className="desk-focus__decision" type="button" key={`${item.kind}:${item.id}`} onClick={() => openDecision(item)}><Icon name="alert" size={15} /><span>{decisionTitle(item)}<small>{item.kind === 'approval' ? '查看调用后审批' : item.kind === 'mea' ? '去会话处理' : '查看并恢复'}</small></span><span aria-hidden="true">↗</span></button>)}</section> : null}
            <section className="desk-preview__section"><h3>近期相关成果</h3>{artifacts.isError ? <p>成果读取失败。</p> : artifacts.isPending && connected ? <p>正在读取成果…</p> : selected.artifacts.length ? selected.artifacts.slice(0, 3).map((artifact) => <ArtifactRow key={artifact.id} artifact={artifact} />) : <p>最近 20 项成果中没有关联记录。</p>}<button type="button" className="desk-focus__link" onClick={onOpenArtifacts}>打开全部成果 <span aria-hidden="true">→</span></button></section>
            <button type="button" className="desk-preview__return" onClick={closePreview}>返回工作焦点</button>
          </> : <>
            <section className="desk-focus__section desk-focus__section--decisions"><header><h2>现在需要你决定</h2><span>{inboxPending ? '…' : inboxError ? '—' : inboxItems.length}</span></header>
              {inboxError ? <p className="desk-focus__empty">待处理读取失败，请进入待处理页重试。</p> : inboxPending && inboxItems.length === 0 ? <p className="desk-focus__empty">正在读取待处理…</p> : inboxItems.length ? inboxItems.slice(0, 3).map((item) => <button type="button" className="desk-focus__decision" key={`${item.kind}:${item.id}`} onClick={() => openDecision(item)}><Icon name={item.kind === 'approval' ? 'approvals' : item.kind === 'mea' ? 'chat' : 'runs'} size={16} /><span>{decisionTitle(item)}<small>{item.kind === 'approval' ? '工具审批' : item.kind === 'mea' ? '长任务介入' : '中断恢复'}</small></span><span aria-hidden="true">↗</span></button>) : <p className="desk-focus__empty">当前没有等待你处理的事项。</p>}
              <button type="button" className="desk-focus__link" onClick={() => onNavigate('inbox')}>进入处理中心 <span aria-hidden="true">→</span></button>
            </section>
            <section className="desk-focus__section"><header><h2>刚刚交付</h2><Icon name="artifacts" size={16} /></header>{artifacts.isError ? <p className="desk-focus__empty">成果读取失败。</p> : artifacts.isPending && connected ? <p className="desk-focus__empty">正在读取成果…</p> : artifacts.data?.length ? artifacts.data.slice(0, 3).map((artifact) => <ArtifactRow key={artifact.id} artifact={artifact} />) : <p className="desk-focus__empty">发布后的文件和链接会出现在这里。</p>}<button type="button" className="desk-focus__link" onClick={onOpenArtifacts}>全部成果 <span aria-hidden="true">→</span></button></section>
            <section className="desk-focus__section"><header><h2>接下来自动运行</h2><Icon name="automations" size={16} /></header>{automations.isError ? <p className="desk-focus__empty">自动化读取失败。</p> : automations.isPending && connected ? <p className="desk-focus__empty">正在读取日程…</p> : upcoming.length ? upcoming.map((automation) => <button type="button" className="desk-focus__schedule" key={automation.id} onClick={() => onNavigate('automations')}><span>{automation.title}</span><time dateTime={automation.next_run_at!}>{formatTime(automation.next_run_at!)}</time></button>) : <p className="desk-focus__empty">没有已安排的下一次执行。</p>}<button type="button" className="desk-focus__link" onClick={() => onNavigate('automations')}>管理自动化 <span aria-hidden="true">→</span></button></section>
          </>}
        </aside>
      </div>
    </section>
  )
}
