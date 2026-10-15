

import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'

import type { ApprovalRequest } from '../api/types'
import ApprovalCard from './ApprovalCard'

const approval: ApprovalRequest = {
  id: 'appr-123456',
  run_id: 'run-1',
  conversation_id: 'conv-1',
  tool_name: 'run_command',
  tool_call_id: 'call-abc',
  arguments: { command: 'ls -la' },
  reason: '需要执行 shell 命令',
  status: 'pending',
  created_at: '2026-08-20T00:00:00+00:00',
  resolved_at: null,
}

describe('ApprovalCard', () => {
  it('展示工具名与 reason', () => {
    const html = renderToStaticMarkup(
      <ApprovalCard approval={approval} onApprove={() => {}} onDeny={() => {}} />,
    )
    expect(html).toContain('run_command')
    expect(html).toContain('需要执行 shell 命令')
    expect(html).toContain('approval-card')
  })

  it('提供中文允许 / 拒绝按钮', () => {
    const html = renderToStaticMarkup(
      <ApprovalCard approval={approval} onApprove={() => {}} onDeny={() => {}} />,
    )
    expect(html).toContain('允许执行')
    expect(html).toContain('拒绝')
    expect(html).toContain('即将执行')
    expect(html).toContain('ls -la')
  })

  it('准备脚本完整展示脚本和哈希，不放进折叠详情', () => {
    const script = 'python -m venv --copies .venv\n.venv/bin/pip install -r requirements.txt\n'
    const html = renderToStaticMarkup(
      <ApprovalCard
        approval={{
          ...approval,
          tool_name: 'workspace_setup',
          arguments: { script_path: 'setup.sh', script, sha256: 'abcdef1234567890' },
        }}
        onApprove={() => {}}
        onDeny={() => {}}
      />,
    )
    expect(html).toContain('setup-script')
    expect(html).toContain('pip install -r requirements.txt')
    expect(html).toContain('abcdef123456')
    expect(html).toContain('访问外网')
    expect(html).not.toContain('查看技术详情')
  })

  it('busy 时按钮禁用', () => {
    const html = renderToStaticMarkup(
      <ApprovalCard approval={approval} busy onApprove={() => {}} onDeny={() => {}} />,
    )
    expect(html).toContain('disabled=""')
  })

})
