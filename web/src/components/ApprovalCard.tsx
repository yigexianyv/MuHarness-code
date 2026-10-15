

import type { ReactElement } from 'react'
import type { ApprovalRequest } from '../api/types'
import { Icon } from './Icon'
import { Button } from './ui'

export interface ApprovalCardProps {
  approval: ApprovalRequest
  busy?: boolean
  onApprove: (id: string) => void
  onDeny: (id: string) => void
}

const WORKSPACE_SETUP_TOOL = 'workspace_setup'

function sandboxApprovalLabel(approval: ApprovalRequest): string {
  if (approval.tool_name === WORKSPACE_SETUP_TOOL) return '联网执行环境准备脚本'
  if (['run_shell_command', 'run_command'].includes(approval.tool_name)) {
    return '执行终端命令'
  }
  if (approval.tool_name === 'http_request') return '访问外部网络'
  return '执行敏感操作'
}

function sandboxApprovalDescription(approval: ApprovalRequest): string {
  if (approval.tool_name === WORKSPACE_SETUP_TOOL) {
    return '脚本会在独立容器中运行，可以访问外网，并能读写工作区（.env、.git 等受保护路径除外）。请逐行确认脚本内容；执行结束后 Agent 仍只使用断网容器。'
  }
  if (['run_shell_command', 'run_command'].includes(approval.tool_name)) {
    return 'MuHarness 需要在本地终端执行以下命令。请确认命令内容和影响后再允许。'
  }
  if (approval.tool_name === 'http_request') {
    return 'MuHarness 需要向外部服务发送网络请求。请确认目标和参数后再允许。'
  }
  return '这项操作可能修改本地环境或访问外部资源，请确认后再继续。'
}

interface SetupScript {
  path: string
  sha256: string
  script: string
}

/** 准备脚本必须完整展示，不能藏在折叠的技术详情里。 */
function setupScript(approval: ApprovalRequest): SetupScript | null {
  if (approval.tool_name !== WORKSPACE_SETUP_TOOL) return null
  const { script_path: path, sha256, script } = approval.arguments ?? {}
  return {
    path: typeof path === 'string' ? path : '',
    sha256: typeof sha256 === 'string' ? sha256 : '',
    script: typeof script === 'string' ? script : '',
  }
}

function commandPreview(approval: ApprovalRequest): string | null {
  if (!['run_shell_command', 'run_command'].includes(approval.tool_name)) {
    return null
  }
  const command = approval.arguments?.command
  return typeof command === 'string' && command.trim() ? command.trim() : null
}

export default function ApprovalCard({
  approval,
  busy = false,
  onApprove,
  onDeny,
}: ApprovalCardProps): ReactElement {
  const title = sandboxApprovalLabel(approval)
  const description = sandboxApprovalDescription(approval)
  const command = commandPreview(approval)
  const setup = setupScript(approval)
  const argsText = (() => {
    try {
      return JSON.stringify(approval.arguments, null, 2)
    } catch {
      return String(approval.arguments)
    }
  })()

  return (
    <div className="approval-card" data-testid="approval-card">
      <div className="approval-card__heading">
        <span className="approval-card__icon"><Icon name="approvals" size={17} /></span>
        <div>
          <div className="approval-card__eyebrow">需要你的确认</div>
          <div className="approval-card__title">允许“{title}”吗？</div>
          <div className="approval-card__desc">{description}</div>
        </div>
      </div>
      {command ? (
        <div className="approval-card__command">
          <span>即将执行</span>
          <code>{command}</code>
        </div>
      ) : null}
      {setup ? (
        <div className="approval-card__command" data-testid="setup-script">
          <span>即将联网执行 {setup.path}（SHA-256 {setup.sha256.slice(0, 12)}）</span>
          <pre>{setup.script}</pre>
        </div>
      ) : null}
      {approval.reason && !setup ? (
        <div className="approval-card__reason">{approval.reason}</div>
      ) : null}
      {argsText && !setup ? (
        <details className="approval-card__details">
          <summary>查看技术详情</summary>
          <div className="approval-card__tech">工具：{approval.tool_name}</div>
          <pre>{argsText}</pre>
        </details>
      ) : null}
      <div className="approval-card__actions">
        <Button
          variant="danger"
          size="sm"
          disabled={busy}
          onClick={() => onDeny(approval.id)}
        >
          拒绝
        </Button>
        <Button
          variant="primary"
          size="sm"
          disabled={busy}
          onClick={() => onApprove(approval.id)}
        >
          允许执行
        </Button>
      </div>
    </div>
  )
}
