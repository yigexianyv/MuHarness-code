

import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'

import { listRuns } from '../api/runs'
import type { MeaRun, Run, RunStatus } from '../api/types'
import { ledgerSummary, meaVerdict, runVerdict } from '../agent/hub'
import {
  ROLE_LABEL,
  entryStatus,
  entryTime,
  groupRuns,
  meaRunRole,
} from '../agent/meaPresentation'
import type { RunListEntry } from '../agent/meaPresentation'
import { humanizeRunError } from '../agent/turnPresentation'
import { Icon } from '../components/Icon'
import { EmptyState, ErrorState, LoadingState } from '../components/PageStates'
import RunBadge from '../components/RunBadge'
import { useEventsStore } from '../stores/events'

type RunFilter = 'all' | 'audited' | 'running' | 'attention'

const NO_MEAS: Map<string, MeaRun> = new Map()

const FILTER_LABEL: Record<RunFilter, string> = {
  all: '全部',
  audited: '已审计',
  running: '运行中',
  attention: '需要关注',
}

/**
 * 账本里的“执行记录”分页。长任务整组显示独立审计的结论；
 * 普通执行只有 Agent 自己的结论，明确标成“未经审计”。
 */
export default function RunsLedger({
  openRun,
  meas = NO_MEAS,
}: {
  openRun: (runId: string) => void
  /** 长任务详情（按 id），用来显示审计结论 */
  meas?: Map<string, MeaRun>
}): React.JSX.Element {
  const [filter, setFilter] = useState<RunFilter>('all')
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set())
  const runStatuses = useEventsStore((state) => state.runStatuses)
  const query = useQuery({
    queryKey: ['runs'],
    queryFn: () => listRuns({ limit: 100 }),
    refetchInterval: 4000,
  })
  // 长任务的子 Run 按长任务合成一组，避免几十个角色子 Run 把列表淹没
  const grouped = useMemo(() => {
    const effective = (query.data ?? []).map((run) => ({
      ...run,
      status: (runStatuses[run.id] ?? run.status) as RunStatus,
    }))
    return groupRuns(effective)
  }, [query.data, runStatuses])
  const summary = useMemo(() => ledgerSummary(grouped, meas), [grouped, meas])
  const entries = useMemo(() => {
    return grouped
      .filter((entry) => {
        const status = entryStatus(entry)
        return filter === 'all'
          || (filter === 'audited' && entry.kind === 'mea')
          || (filter === 'running' && ['running', 'pending'].includes(status))
          || (filter === 'attention' && ['failed', 'interrupted'].includes(status))
      })
      .sort((a, b) => {
        const activeA = ['running', 'pending', 'interrupted'].includes(entryStatus(a)) ? 1 : 0
        const activeB = ['running', 'pending', 'interrupted'].includes(entryStatus(b)) ? 1 : 0
        return activeB - activeA || entryTime(b).localeCompare(entryTime(a))
      })
  }, [filter, grouped])

  const toggleGroup = (meaId: string): void => {
    setExpanded((current) => {
      const next = new Set(current)
      if (next.has(meaId)) next.delete(meaId)
      else next.add(meaId)
      return next
    })
  }

  return (
    <section className="ledger-runs">
      <div className="ledger-toolbar">
        <dl className="ledger-summary" aria-label="账本汇总">
          <div className="ledger-summary__item ledger-summary__item--audited">
            <dt>独立审计</dt>
            <dd>{summary.audited}</dd>
            <small>通过 {summary.passed} · 未通过 {summary.rejected}</small>
          </div>
          <div className="ledger-summary__item">
            <dt>未经审计</dt>
            <dd>{summary.unaudited}</dd>
            <small>普通执行，只有 Agent 自评</small>
          </div>
        </dl>
        <div className="segmented-control" aria-label="Run filter">
          {(['all', 'audited', 'running', 'attention'] as const).map((item) => (
            <button
              key={item}
              className={filter === item ? 'active' : ''}
              onClick={() => setFilter(item)}
            >
              {FILTER_LABEL[item]}
            </button>
          ))}
        </div>
      </div>
      {query.isPending ? <LoadingState label="正在加载执行历史…" />
        : query.isError ? <ErrorState message={String(query.error)} onRetry={() => void query.refetch()} />
          : entries.length === 0 ? (
            <EmptyState title="当前没有执行记录" hint="MuHarness 开始处理工作后，Run 会出现在这里。" icon="runs" />
          ) : (
            <div className="run-history">
              {entries.map((entry) => {
                if (entry.kind === 'mea') {
                  return (
                    <MeaRunGroup
                      key={`mea-${entry.meaId}`}
                      entry={entry}
                      mea={meas.get(entry.meaId)}
                      expanded={expanded.has(entry.meaId)}
                      onToggle={() => toggleGroup(entry.meaId)}
                      openRun={openRun}
                    />
                  )
                }
                const run = entry.run
                const failed = ['failed', 'interrupted'].includes(run.status)
                const reason = failed ? humanizeRunError(run.stop_reason, run.error) : null
                const verdict = runVerdict(run)
                return (
                  <article key={run.id} className={`run-card run-card--${run.status}`}>
                    <button className="run-card__button" onClick={() => openRun(run.id)}>
                      <header className="run-card__header">
                        <RunBadge status={run.status} />
                        <span className={`ledger-verdict ledger-verdict--${verdict.tone}`}>{verdict.label}</span>
                        <span className="mono">{run.id.slice(0, 8)}</span>
                      </header>
                      <div className="run-card__content">
                        <strong className="run-card__title">{run.user_message || '未命名执行'}</strong>
                        {reason ? <small className="run-card__error">{reason.message}</small> : null}
                      </div>
                      <div className="run-card__meta">
                        <span>{run.mode === 'plan' ? '计划模式' : '普通模式'}</span>
                        <span>{run.source === 'automation' ? '自动化触发' : '会话触发'}</span>
                      </div>
                      <footer className="run-card__footer">
                        <time>{relativeTime(run.created_at)}</time>
                        <span>{run.status === 'interrupted' ? '查看恢复' : '查看详情'} <Icon name="chevronDown" size={14} /></span>
                      </footer>
                    </button>
                  </article>
                )
              })}
            </div>
          )}
    </section>
  )
}

function MeaRunGroup({
  entry,
  mea,
  expanded,
  onToggle,
  openRun,
}: {
  entry: Extract<RunListEntry, { kind: 'mea' }>
  mea: MeaRun | undefined
  expanded: boolean
  onToggle: () => void
  openRun: (runId: string) => void
}): React.JSX.Element {
  const counts = entry.roleCounts
  const verdict = meaVerdict(mea)
  return (
    <article className={`run-card run-card--mea run-card--${entry.status}`}>
      <button className="run-card__button" onClick={onToggle} aria-expanded={expanded}>
        <header className="run-card__header">
          <RunBadge status={entry.status} />
          <span className={`ledger-verdict ledger-verdict--${verdict.tone}`}>
            {verdict.tone === 'ok' ? '✓ ' : ''}{verdict.label}
          </span>
          <span className="mono">{entry.meaId.slice(0, 8)}</span>
        </header>
        <div className="run-card__content">
          <strong className="run-card__title">长任务 · {entry.runs.length} 个子运行</strong>
          <small className="run-card__subtitle">
            {ROLE_LABEL.manager} {counts.manager} · {ROLE_LABEL.executor} {counts.executor} · {ROLE_LABEL.auditor} {counts.auditor}
          </small>
        </div>
        <div className="run-card__meta">
          <span>长任务</span>
          <span>每步独立审计</span>
        </div>
        <footer className="run-card__footer">
          <time>{relativeTime(entry.latestAt)}</time>
          <span>{expanded ? '收起' : '展开'} <Icon name="chevronDown" size={14} /></span>
        </footer>
      </button>
      {expanded ? (
        <ol className="run-group">
          {entry.runs.map((child: Run, index) => {
            const role = meaRunRole(child)
            return (
              <li key={child.id}>
                <button type="button" className="run-group__item" onClick={() => openRun(child.id)}>
                  <span className="run-group__index mono">{String(index + 1).padStart(2, '0')}</span>
                  <span className="run-group__role">{role ? ROLE_LABEL[role] : '子运行'}</span>
                  <RunBadge status={child.status} />
                  <span className="mono run-group__id">{child.id.slice(0, 8)}</span>
                  <time>{relativeTime(child.created_at)}</time>
                </button>
              </li>
            )
          })}
        </ol>
      ) : null}
    </article>
  )
}

function relativeTime(iso: string): string {
  const time = new Date(iso).getTime()
  if (Number.isNaN(time)) return iso
  const minutes = Math.max(0, Math.floor((Date.now() - time) / 60_000))
  if (minutes < 1) return '刚刚'
  if (minutes < 60) return `${minutes} 分钟前`
  if (minutes < 1440) return `${Math.floor(minutes / 60)} 小时前`
  return `${Math.floor(minutes / 1440)} 天前`
}
