/**
 * 「待处理」和「账本」两个页面的纯逻辑。
 *
 * 待处理：把所有需要人介入的事汇成一个队列——工具审批、长任务的提问/暂停/阻塞、
 * 中断后可以恢复的执行。账本：执行记录按“是否经过独立审计”分账，长任务带审计结论。
 */

import type { ApprovalRequest, MeaRun, MeaStatus, Run } from '../api/types'
import { MEA_STATUS_LABEL, isMeaRun, meaReason } from './meaPresentation'
import type { RunListEntry } from './meaPresentation'

// ---------------------------------------------------------------- 待处理

export type InboxItem =
  | { kind: 'approval'; id: string; at: string; approval: ApprovalRequest }
  | { kind: 'mea'; id: string; at: string; mea: MeaRun; reason: string }
  | { kind: 'run'; id: string; at: string; run: Run }

/** 需要人介入的长任务状态：提问要回答，暂停要决定是否继续，阻塞要补信息。 */
const MEA_ATTENTION: ReadonlySet<MeaStatus> = new Set(['waiting_user', 'paused', 'blocked'])

export function meaNeedsAttention(mea: Pick<MeaRun, 'status'>): boolean {
  return MEA_ATTENTION.has(mea.status)
}

export function meaAttentionText(mea: MeaRun): string {
  if (mea.status === 'waiting_user' && mea.pending_question) return mea.pending_question
  return meaReason(mea) ?? MEA_STATUS_LABEL[mea.status]
}

/**
 * 汇总待处理队列。长任务的子 Run 中断后由长任务自己核查恢复，不单独列出；
 * 已经被恢复过的中断 Run（有 Run 的 recovered_from_run_id 指向它）也不再列出。
 */
export function buildInbox({
  approvals,
  meas,
  runs,
}: {
  approvals: ApprovalRequest[]
  meas: MeaRun[]
  runs: Run[]
}): InboxItem[] {
  const recovered = new Set(runs.map((run) => run.recovered_from_run_id).filter(Boolean))
  const items: InboxItem[] = [
    ...approvals
      .filter((approval) => approval.status === 'pending')
      .map((approval): InboxItem => ({
        kind: 'approval', id: approval.id, at: approval.created_at, approval,
      })),
    ...meas
      .filter(meaNeedsAttention)
      .map((mea): InboxItem => ({
        kind: 'mea', id: mea.id, at: mea.updated_at, mea, reason: meaAttentionText(mea),
      })),
    ...runs
      .filter((run) => run.status === 'interrupted' && !isMeaRun(run) && !recovered.has(run.id))
      .map((run): InboxItem => ({ kind: 'run', id: run.id, at: run.updated_at, run })),
  ]
  // 审批会卡住正在跑的任务，排最前；其余按时间倒序
  const weight = (item: InboxItem): number => (item.kind === 'approval' ? 0 : item.kind === 'mea' ? 1 : 2)
  return items.sort((a, b) => weight(a) - weight(b) || b.at.localeCompare(a.at))
}

// ---------------------------------------------------------------- 账本

export type AuditTone = 'ok' | 'bad' | 'warn' | 'muted' | 'pending'

export interface AuditVerdict {
  label: string
  tone: AuditTone
}

/** 长任务的整体审计结论；拿不到长任务详情时退回“审计中/未知”。 */
export function meaVerdict(mea: Pick<MeaRun, 'status'> | undefined): AuditVerdict {
  switch (mea?.status) {
    case 'completed':
      return { label: '审计通过', tone: 'ok' }
    case 'blocked':
    case 'failed':
      return { label: '未通过审计', tone: 'bad' }
    case 'cancelled':
      return { label: '已取消', tone: 'muted' }
    case 'waiting_user':
    case 'paused':
      return { label: '等你处理', tone: 'warn' }
    case 'running':
    case 'finalizing':
      return { label: '审计中', tone: 'pending' }
    default:
      return { label: '长任务', tone: 'muted' }
  }
}

/** 普通执行没有独立审计：只标注它自己报告的结果。 */
export function runVerdict(run: Pick<Run, 'status'>): AuditVerdict {
  switch (run.status) {
    case 'completed':
      return { label: '未经审计', tone: 'muted' }
    case 'failed':
      return { label: '失败', tone: 'bad' }
    case 'interrupted':
      return { label: '可恢复', tone: 'warn' }
    case 'running':
    case 'pending':
      return { label: '运行中', tone: 'pending' }
    default:
      return { label: '已停止', tone: 'muted' }
  }
}

export interface LedgerSummary {
  /** 长任务：每一步都经过独立审计 */
  audited: number
  passed: number
  rejected: number
  /** 普通执行：只有 Agent 自己的结论 */
  unaudited: number
}

export function ledgerSummary(entries: RunListEntry[], meas: Map<string, MeaRun>): LedgerSummary {
  const summary: LedgerSummary = { audited: 0, passed: 0, rejected: 0, unaudited: 0 }
  for (const entry of entries) {
    if (entry.kind === 'run') {
      summary.unaudited += 1
      continue
    }
    summary.audited += 1
    const tone = meaVerdict(meas.get(entry.meaId)).tone
    if (tone === 'ok') summary.passed += 1
    else if (tone === 'bad') summary.rejected += 1
  }
  return summary
}
