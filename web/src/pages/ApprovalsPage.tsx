

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect } from 'react'

import {
  approveApproval,
  denyApproval,
  listApprovals,
} from '../api/approvals'
import type { ApprovalRequest, ApprovalStatus } from '../api/types'
import { ErrorState, LoadingState } from '../components/PageStates'
import { Badge, Button } from '../components/ui'
import { rpcClient } from '../rpc'
import { toast } from '../stores/toasts'

function formatTime(iso: string | null): string {
  if (!iso) return '-'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleString()
}

function formatArguments(argumentsValue: Record<string, unknown>): string {
  try {
    return JSON.stringify(argumentsValue, null, 2)
  } catch {
    return String(argumentsValue)
  }
}

const STATUS_TONE: Record<ApprovalStatus, 'warning' | 'success' | 'danger'> = {
  pending: 'warning',
  approved: 'success',
  denied: 'danger',
}

const STATUS_LABEL: Record<ApprovalStatus, string> = {
  pending: '待处理',
  approved: '已批准',
  denied: '已拒绝',
}

export function ApprovalItem({
  approval,
  busy,
  onApprove,
  onDeny,
}: {
  approval: ApprovalRequest
  busy: boolean
  onApprove: (id: string) => void
  onDeny: (id: string) => void
}): React.JSX.Element {
  return (
    <div className="approval-card">
      <div className="approval-card__heading">
        <div className="approval-card__icon">
          <span className="approval-card__icon-label">
            ›
          </span>
        </div>
        <div className="approval-card__content">
          <div className="approval-card__title-row">
            <strong className="approval-card__title">{approval.tool_name}</strong>
            <Badge tone={STATUS_TONE[approval.status]}>{STATUS_LABEL[approval.status]}</Badge>
          </div>
          <details className="approval-card__details">
            <summary>查看详情</summary>
            <div className="approval-card__detail-body">
              {approval.reason ? (
                <div className="approval-card__reason">{approval.reason}</div>
              ) : null}
              <div className="approval-card__meta">
                {approval.run_id ? (
                  <span className="text-muted">
                    Run：{approval.run_id.slice(0, 8)}
                  </span>
                ) : null}
                <span className="text-muted">{formatTime(approval.created_at)}</span>
              </div>
              <div className="approval-card__arguments">
                <span>调用参数</span>
                <pre>{formatArguments(approval.arguments)}</pre>
              </div>
            </div>
          </details>
        </div>
      </div>

      {approval.status === 'pending' ? (
        <div className="approval-card__actions">
          <Button
            variant="primary"
            size="sm"
            disabled={busy}
            onClick={() => onApprove(approval.id)}
          >
            批准
          </Button>
          <Button
            variant="danger"
            size="sm"
            disabled={busy}
            onClick={() => onDeny(approval.id)}
          >
            拒绝
          </Button>
        </div>
      ) : null}
    </div>
  )
}


/** 批准 / 拒绝一条审批，成功后刷新审批和执行列表。 */
export function useApprovalResolver(): {
  resolve: (id: string, decision: 'approve' | 'deny') => void
  busy: boolean
} {
  const queryClient = useQueryClient()
  const mutation = useMutation({
    mutationFn: (action: { id: string; decision: 'approve' | 'deny' }) =>
      action.decision === 'approve'
        ? approveApproval(action.id)
        : denyApproval(action.id),
    onSuccess: (approval, action) => {
      toast.success(
        action.decision === 'approve'
          ? `已批准 ${approval.tool_name}`
          : `已拒绝 ${approval.tool_name}`,
      )
      void queryClient.invalidateQueries({ queryKey: ['approvals'] })
      void queryClient.invalidateQueries({ queryKey: ['runs'] })
      void queryClient.invalidateQueries({ queryKey: ['inbox'] })
    },
    onError: (err: unknown) => {
      toast.error(err instanceof Error ? err.message : String(err))
    },
  })
  return {
    resolve: (id, decision) => mutation.mutate({ id, decision }),
    busy: mutation.isPending,
  }
}

/** 审批推送到达时刷新相关查询。 */
export function useApprovalRefresh(): void {
  const queryClient = useQueryClient()
  useEffect(() => {
    const refresh = (): void => {
      void queryClient.invalidateQueries({ queryKey: ['approvals'] })
      void queryClient.invalidateQueries({ queryKey: ['runs'] })
      void queryClient.invalidateQueries({ queryKey: ['inbox'] })
    }
    const offRequired = rpcClient.on('approval.required', refresh)
    const offResolved = rpcClient.on('approval.resolved', refresh)
    return () => {
      offRequired()
      offResolved()
    }
  }, [queryClient])
}

/** 最近处理过的审批（“待处理”页底部的折叠记录）。 */
export function ApprovalHistory(): React.JSX.Element {
  const historyQuery = useQuery({
    queryKey: ['approvals', 'history'],
    queryFn: () => listApprovals(undefined, 50),
    refetchInterval: 5000,
  })
  const history = (historyQuery.data ?? [])
    .filter((approval) => approval.status !== 'pending')
    .slice(0, 20)
  if (historyQuery.isPending) return <LoadingState label="正在加载审批记录…" />
  if (historyQuery.isError) {
    return <ErrorState message={String(historyQuery.error)} onRetry={() => void historyQuery.refetch()} />
  }
  if (history.length === 0) {
    return <div className="approval-history-empty">暂无最近审批记录。</div>
  }
  return (
    <div className="approvals-list">
      {history.map((approval) => (
        <ApprovalItem
          key={approval.id}
          approval={approval}
          busy={false}
          onApprove={() => {}}
          onDeny={() => {}}
        />
      ))}
    </div>
  )
}
