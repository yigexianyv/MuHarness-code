import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

import { listApprovals } from '../api/approvals'
import {
  cancelWorkspaceSetup,
  getWorkspaceSetup,
  startWorkspaceSetup,
} from '../api/workspace'
import type { WorkspaceSetupCheck, WorkspaceSetupRecord } from '../api/workspace'
import { useApprovalResolver } from '../pages/ApprovalsPage'
import { toast } from '../stores/toasts'
import ApprovalCard from './ApprovalCard'
import { Button, Input } from './ui'

const STATUS_LABEL: Record<WorkspaceSetupRecord['status'], string> = {
  running: '进行中',
  succeeded: '已准备',
  failed: '失败',
  denied: '已拒绝',
  cancelled: '已取消',
}

function checkText(check: WorkspaceSetupCheck | null): string {
  if (!check) return '—'
  const state = check.passed === null ? '跳过' : check.passed ? '通过' : '未通过'
  return `${state} · ${check.detail}`
}

/**
 * 工作区环境准备：用户触发，审批后在联网容器中执行准备脚本。
 * Agent 自己的 Shell 始终断网，只通过工作区里准备好的依赖文件使用它们。
 */
export default function WorkspaceSetupPanel(): React.JSX.Element {
  const queryClient = useQueryClient()
  const [scriptPath, setScriptPath] = useState('setup.sh')
  const [verifyCommand, setVerifyCommand] = useState('')
  const { resolve, busy: resolving } = useApprovalResolver()

  const statusQuery = useQuery({
    queryKey: ['workspace-setup'],
    queryFn: () => getWorkspaceSetup(),
    refetchInterval: (query) => (query.state.data?.current ? 2000 : false),
  })
  const current = statusQuery.data?.current ?? null
  const last = statusQuery.data?.last ?? null

  // 只展示属于本次准备的审批，脚本全文在卡片里
  const approvalQuery = useQuery({
    queryKey: ['approvals', 'workspace-setup', current?.id],
    queryFn: () => listApprovals('pending', 50),
    enabled: current !== null,
    refetchInterval: 2000,
  })
  const pendingApproval = approvalQuery.data?.find(
    (approval) => approval.tool_name === 'workspace_setup' && approval.run_id === current?.id,
  )

  const refresh = (): void => {
    void queryClient.invalidateQueries({ queryKey: ['workspace-setup'] })
    void queryClient.invalidateQueries({ queryKey: ['approvals'] })
  }
  const startMutation = useMutation({
    mutationFn: () => startWorkspaceSetup(scriptPath.trim(), verifyCommand.trim() || undefined),
    onSuccess: refresh,
    onError: (err: unknown) => toast.error(err instanceof Error ? err.message : String(err)),
  })
  const cancelMutation = useMutation({
    mutationFn: () => cancelWorkspaceSetup(),
    onSuccess: refresh,
    onError: (err: unknown) => toast.error(err instanceof Error ? err.message : String(err)),
  })

  const shown = current ?? last
  return (
    <section className="settings-group" data-testid="workspace-setup">
      <header className="settings-group__header">
        <div>
          <h3>工作区环境</h3>
          <p>Agent 的终端始终断网。需要的依赖由准备脚本在独立的联网容器中安装，每次都需要你确认脚本内容。</p>
        </div>
      </header>

      {current ? (
        <div className="settings-loading">
          <span className="spinner" />
          {pendingApproval ? '等待你确认准备脚本…' : '正在准备环境，完成前不能开始新的对话或任务…'}
          <Button size="sm" variant="ghost" disabled={cancelMutation.isPending} onClick={() => cancelMutation.mutate()}>
            取消
          </Button>
        </div>
      ) : (
        <div style={{ display: 'grid', gap: 8, maxWidth: 520 }}>
          <label style={{ display: 'grid', gap: 4 }}>
            准备脚本（工作区内路径）
            <Input value={scriptPath} onChange={(event) => setScriptPath(event.target.value)} />
          </label>
          <label style={{ display: 'grid', gap: 4 }}>
            验证命令（可选，在断网容器中运行）
            <Input
              value={verifyCommand}
              placeholder=".venv/bin/python -c 'import pytest'"
              onChange={(event) => setVerifyCommand(event.target.value)}
            />
          </label>
          <Button
            size="sm"
            variant="primary"
            style={{ justifySelf: 'start' }}
            disabled={startMutation.isPending || !scriptPath.trim()}
            onClick={() => startMutation.mutate()}
          >
            准备环境
          </Button>
        </div>
      )}

      {pendingApproval ? (
        <ApprovalCard
          approval={pendingApproval}
          busy={resolving}
          onApprove={(id) => resolve(id, 'approve')}
          onDeny={(id) => resolve(id, 'deny')}
        />
      ) : null}

      {shown ? (
        <dl className="settings-info-list">
          <div><dt>状态</dt><dd>{STATUS_LABEL[shown.status]}</dd></div>
          <div><dt>脚本</dt><dd className="mono">{shown.script_path} · {shown.script_sha256.slice(0, 12)}</dd></div>
          {shown.exit_code !== null ? (
            <div><dt>退出码</dt><dd>{shown.exit_code}{shown.timed_out ? '（超时）' : ''}</dd></div>
          ) : null}
          <div><dt>网络隔离</dt><dd>{checkText(shown.network_check)}</dd></div>
          <div><dt>环境验证</dt><dd>{checkText(shown.environment_check)}</dd></div>
          {shown.error ? <div><dt>错误</dt><dd>{shown.error}</dd></div> : null}
          {shown.log_path ? <div><dt>日志</dt><dd className="mono">{shown.log_path}</dd></div> : null}
        </dl>
      ) : null}
      {shown?.output_tail ? (
        <details>
          <summary>输出结尾</summary>
          <pre>{shown.output_tail}</pre>
        </details>
      ) : null}
    </section>
  )
}
