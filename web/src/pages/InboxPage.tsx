import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useState } from 'react'

import { buildInbox } from '../agent/hub'
import type { InboxItem } from '../agent/hub'
import { MEA_STATUS_LABEL, mergeMeas } from '../agent/meaPresentation'
import { listApprovals } from '../api/approvals'
import { answerMea, listAllMeas } from '../api/mea'
import { listRuns } from '../api/runs'
import type { MeaRun } from '../api/types'
import { Icon } from '../components/Icon'
import { PageShell } from '../components/PageShell'
import { EmptyState, ErrorState, LoadingState } from '../components/PageStates'
import { Button } from '../components/ui'
import { useEventsStore } from '../stores/events'
import { toast } from '../stores/toasts'
import {
  ApprovalHistory,
  ApprovalItem,
  useApprovalRefresh,
  useApprovalResolver,
} from './ApprovalsPage'

/** 待处理队列的数据；App 用它算侧栏角标，InboxPage 用它渲染列表。 */
export function useInbox(): {
  items: InboxItem[]
  meas: MeaRun[]
  pending: boolean
  error: string | null
  refetch: () => void
} {
  const approvalsQuery = useQuery({
    queryKey: ['approvals', 'pending'],
    queryFn: () => listApprovals('pending'),
    refetchInterval: 3000,
  })
  const measQuery = useQuery({
    queryKey: ['inbox', 'meas'],
    queryFn: () => listAllMeas({ limit: 100 }),
    refetchInterval: 5000,
  })
  const runsQuery = useQuery({
    queryKey: ['runs'],
    queryFn: () => listRuns({ limit: 100 }),
    refetchInterval: 4000,
  })
  const pushedMeas = useEventsStore((state) => state.meaById)
  const meas = useMemo(
    () => mergeMeas(measQuery.data ?? [], Object.values(pushedMeas)),
    [measQuery.data, pushedMeas],
  )
  const items = useMemo(
    () => buildInbox({ approvals: approvalsQuery.data ?? [], meas, runs: runsQuery.data ?? [] }),
    [approvalsQuery.data, meas, runsQuery.data],
  )
  const failed = [approvalsQuery, measQuery, runsQuery].find((query) => query.isError)
  return {
    items,
    meas,
    pending: approvalsQuery.isPending || measQuery.isPending || runsQuery.isPending,
    error: failed ? String(failed.error) : null,
    refetch: () => {
      void approvalsQuery.refetch()
      void measQuery.refetch()
      void runsQuery.refetch()
    },
  }
}

const SECTION_META: Record<InboxItem['kind'], { title: string; hint: string }> = {
  approval: { title: '工具审批', hint: '检查调用内容与原因，再决定是否允许执行。' },
  mea: { title: '长任务介入', hint: '回答问题或打开会话，处理暂停与阻塞。' },
  run: { title: '中断恢复', hint: '查看中断位置，从保存的检查点继续。' },
}

type InboxFilter = 'all' | InboxItem['kind']

export default function InboxPage({
  onOpenConversation,
  onOpenRun,
}: {
  onOpenConversation: (conversationId: string) => void
  onOpenRun: (runId: string) => void
}): React.JSX.Element {
  const inbox = useInbox()
  const approvals = useApprovalResolver()
  const [filter, setFilter] = useState<InboxFilter>('all')
  useApprovalRefresh()

  const sections = (['approval', 'mea', 'run'] as const)
    .map((kind) => ({ kind, items: inbox.items.filter((item) => item.kind === kind) }))
    .filter((section) => section.items.length > 0 && (filter === 'all' || section.kind === filter))

  return (
    <PageShell
      title="待处理"
      subtitle={
        inbox.items.length > 0
          ? `${inbox.items.length} 项需要你介入。审批、回答与恢复集中处理。`
          : '需要你的决定时，任务会在这里等待。'
      }
      maxWidth={1100}
    >
      {!inbox.pending && !inbox.error ? (
        <div className="page-toolbar">
          <div className="page-filter-tabs" role="group" aria-label="待处理筛选">
            {(['all', 'approval', 'mea', 'run'] as const).map((kind) => (
              <button
                key={kind}
                type="button"
                aria-pressed={filter === kind}
                className={filter === kind ? 'active' : ''}
                onClick={() => setFilter(kind)}
              >
                {kind === 'all' ? '全部' : SECTION_META[kind].title}
                <span>{kind === 'all' ? inbox.items.length : inbox.items.filter((item) => item.kind === kind).length}</span>
              </button>
            ))}
          </div>
          <Button size="sm" variant="ghost" onClick={inbox.refetch}>
            <Icon name="activity" size={14} />刷新
          </Button>
        </div>
      ) : null}
      {inbox.pending ? (
        <LoadingState label="正在汇总待处理事项…" />
      ) : inbox.error ? (
        <ErrorState message={inbox.error} onRetry={inbox.refetch} />
      ) : sections.length === 0 ? (
        <EmptyState
          title={filter === 'all' ? '没有需要你处理的事' : `暂无${SECTION_META[filter].title}`}
          hint="你可以继续在工作台发起任务；需要审批、补充信息或恢复时，我们会在这里提示。"
          icon="inbox"
        />
      ) : (
        <div className="inbox">
          {sections.map((section) => (
            <section key={section.kind} className={`inbox-section inbox-section--${section.kind}`}>
              <header className="inbox-section__header">
                <h2>
                  {SECTION_META[section.kind].title}
                  <span className="inbox-section__count">{section.items.length}</span>
                </h2>
                <p>{SECTION_META[section.kind].hint}</p>
              </header>
              <div className="inbox-section__items">
                {section.items.map((item) =>
                  item.kind === 'approval' ? (
                    <ApprovalItem
                      key={item.id}
                      approval={item.approval}
                      busy={approvals.busy}
                      onApprove={(id) => approvals.resolve(id, 'approve')}
                      onDeny={(id) => approvals.resolve(id, 'deny')}
                    />
                  ) : item.kind === 'mea' ? (
                    <MeaInboxCard
                      key={item.id}
                      mea={item.mea}
                      reason={item.reason}
                      onOpen={() => onOpenConversation(item.mea.conversation_id)}
                    />
                  ) : (
                    <article key={item.id} className="inbox-card inbox-card--run">
                      <div className="inbox-card__body">
                        <span className="inbox-card__kind">执行中断</span>
                        <strong>{item.run.user_message || '未命名执行'}</strong>
                        <small className="mono">{item.run.id.slice(0, 8)}</small>
                      </div>
                      <Button size="sm" onClick={() => onOpenRun(item.run.id)}>
                        查看并恢复
                      </Button>
                    </article>
                  ),
                )}
              </div>
            </section>
          ))}
        </div>
      )}

      <details className="inbox-history">
        <summary>
          最近处理过的审批 <Icon name="chevronDown" size={14} />
        </summary>
        <ApprovalHistory />
      </details>
    </PageShell>
  )
}

/** 长任务提问可以直接在这里回答；暂停和阻塞跳回对应会话处理。 */
function MeaInboxCard({
  mea,
  reason,
  onOpen,
}: {
  mea: MeaRun
  reason: string
  onOpen: () => void
}): React.JSX.Element {
  const queryClient = useQueryClient()
  const [text, setText] = useState('')
  const answer = useMutation({
    mutationFn: (value: string) => answerMea(mea.id, value),
    onSuccess: (result) => {
      if (result.accepted) {
        toast.success('已回答，长任务继续推进')
        setText('')
      } else {
        toast.error(result.reason ?? '长任务没有接受这个回答')
      }
      void queryClient.invalidateQueries({ queryKey: ['inbox'] })
      void queryClient.invalidateQueries({ queryKey: ['mea'] })
      void queryClient.invalidateQueries({ queryKey: ['meas'] })
    },
    onError: (err: unknown) => toast.error(err instanceof Error ? err.message : String(err)),
  })
  const asking = mea.status === 'waiting_user'

  return (
    <article className={`inbox-card inbox-card--mea inbox-card--${mea.status}`}>
      <div className="inbox-card__body">
        <span className="inbox-card__kind">长任务 · {MEA_STATUS_LABEL[mea.status]}</span>
        <strong>{reason}</strong>
        <small className="mono">{mea.id.slice(0, 8)}</small>
        {asking ? (
          <div className="inbox-card__answer">
            {mea.pending_choices.length > 0 ? (
              <div className="mea-choices">
                {mea.pending_choices.map((choice) => (
                  <Button
                    key={choice}
                    size="sm"
                    disabled={answer.isPending}
                    onClick={() => answer.mutate(choice)}
                  >
                    {choice}
                  </Button>
                ))}
              </div>
            ) : null}
            <form
              className="inbox-card__answer-form"
              onSubmit={(event) => {
                event.preventDefault()
                if (text.trim()) answer.mutate(text.trim())
              }}
            >
              <input
                className="input"
                value={text}
                aria-label="回答长任务问题"
                placeholder="直接在这里回答…"
                onChange={(event) => setText(event.target.value)}
              />
              <Button size="sm" variant="primary" type="submit" disabled={!text.trim() || answer.isPending}>
                回答
              </Button>
            </form>
          </div>
        ) : null}
      </div>
      <Button size="sm" onClick={onOpen}>
        去会话处理
      </Button>
    </article>
  )
}
