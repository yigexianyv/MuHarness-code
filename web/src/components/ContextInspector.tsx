

import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'

import { getConversationConstraints, setConversationConstraints } from '../api/conversations'
import { getRunContextMessages, getRunToolEvidence } from '../api/runs'
import type { AgentEvent, Message } from '../api/types'
import {
  buildContextSteps,
  diffSummary,
  effectiveSummary,
  newlyCoveredRange,
  SUMMARY_FIELDS,
  summaryEntries,
  type ContextStepVM,
  type TruncatedToolVM,
} from '../agent/runAnalysis'
import { formatCacheHitRate, formatTokens } from '../agent/turnPresentation'
import { toast } from '../stores/toasts'
import { EmptyState } from './ui'

/** 打开某次工具调用的输出：模型实际收到的（截短版），或证据库里的完整原文。 */
type OpenToolOutput =
  | { kind: 'model'; tool: TruncatedToolVM }
  | { kind: 'full'; toolCallId: string; toolName: string }

interface MessageRange {
  from: number
  to: number
  title: string
}

/** 面向用户的消息编号从 1 开始。 */
function rangeLabel(from: number, to: number): string {
  return to - from <= 1 ? `第 ${from + 1} 条` : `第 ${from + 1}~${to} 条`
}

function UsageBar({ step }: { step: ContextStepVM }): React.JSX.Element {
  const limit = step.workingInputBudget || step.inputBudget || step.contextWindow
  const remaining = Math.max(0, limit - step.preparedInputTokens)
  return (
    <section className="context-section">
      <div className="context-section__heading">
        <h3>当前用量（估算）</h3>
        <span className="context-stage mono">
          {formatTokens(step.preparedInputTokens)} / {formatTokens(limit)} · 剩 {formatTokens(remaining)}
        </span>
      </div>
      <progress
        className="context-window-progress"
        max={Math.max(1, limit)}
        value={Math.min(step.preparedInputTokens, Math.max(1, limit))}
        aria-label="当前上下文用量"
      />
    </section>
  )
}

function ContextTimeline({
  steps,
  selected,
  onSelect,
  onOpenRange,
  onOpenTool,
}: {
  steps: ContextStepVM[]
  selected: number
  onSelect: (step: number) => void
  onOpenRange: (range: MessageRange) => void
  onOpenTool?: (target: OpenToolOutput) => void
}): React.JSX.Element {
  return (
    <section className="context-section">
      <h3>时间线</h3>
      <ol className="context-timeline">
        {steps.map((step) => {
          const covered = newlyCoveredRange(step)
          const compacted = step.compactionStage !== 'none' || step.summaryUpdated || covered !== null
          return (
            <li key={step.step} className={step.step === selected ? 'active' : ''}>
              <button type="button" className="context-timeline__head" onClick={() => onSelect(step.step)}>
                <span>步骤 {step.step}</span>
                <span className="mono">
                  {compacted && step.originalInputTokens !== step.preparedInputTokens
                    ? <>{formatTokens(step.originalInputTokens)} <em>──压缩──▶</em> {formatTokens(step.preparedInputTokens)}</>
                    : formatTokens(step.preparedInputTokens)}
                </span>
                {step.constraintsPossiblyDropped.length > 0
                  ? <span className="context-timeline__alert">⚠ {step.constraintsPossiblyDropped.length} 条约束可能丢失</span>
                  : null}
              </button>
              {covered || step.truncatedTools.length > 0 || step.compactedToolResults > 0 || step.removedToolRounds > 0 || step.summaryError ? (
                <ul className="context-timeline__details">
                  {covered ? (
                    <li>
                      · {rangeLabel(covered.from, covered.to)}消息已由摘要替代{' '}
                      <button
                        type="button"
                        className="context-link"
                        onClick={() => onOpenRange({
                          ...covered,
                          title: `步骤 ${step.step}：${rangeLabel(covered.from, covered.to)}已由摘要替代的消息`,
                        })}
                      >
                        查看原文
                      </button>
                    </li>
                  ) : null}
                  {step.truncatedTools.map((tool) => (
                    <li key={tool.toolCallId}>
                      · {tool.toolName} 输出过长，模型收到截短版本{' '}
                      {onOpenTool ? (
                        <>
                          <button type="button" className="context-link" onClick={() => onOpenTool({ kind: 'model', tool })}>
                            查看模型收到的内容
                          </button>
                          {' · '}
                          <button
                            type="button"
                            className="context-link"
                            onClick={() => onOpenTool({ kind: 'full', toolCallId: tool.toolCallId, toolName: tool.toolName })}
                          >
                            查看完整工具原文
                          </button>
                        </>
                      ) : null}
                    </li>
                  ))}
                  {step.compactedToolResults > 0 ? (
                    <li>
                      · {step.compactedToolResults} 个旧工具结果在请求中被压缩
                      {step.sourceMessageCount !== null ? (
                        <>
                          {' '}
                          <button
                            type="button"
                            className="context-link"
                            onClick={() => onOpenRange({
                              from: 0,
                              to: step.sourceMessageCount!,
                              title: `步骤 ${step.step}：请求前的全部消息`,
                            })}
                          >
                            查看原文
                          </button>
                        </>
                      ) : null}
                    </li>
                  ) : null}
                  {step.removedToolRounds > 0 ? <li>· {step.removedToolRounds} 个旧工具轮已移出请求</li> : null}
                  {step.summaryError ? <li className="context-warning">· 摘要更新失败：{step.summaryError}</li> : null}
                </ul>
              ) : null}
            </li>
          )
        })}
      </ol>
      <p className="context-note">压缩只影响发给模型的请求，原始消息仍完整保存，可随时查看原文。</p>
    </section>
  )
}

function SummaryPanel({
  steps,
  step,
  canPin,
}: {
  steps: ContextStepVM[]
  step: ContextStepVM
  canPin: boolean
}): React.JSX.Element {
  const [compare, setCompare] = useState(false)
  const effective = effectiveSummary(steps, step.step)
  const canCompare = step.summaryPreviousSnapshot !== null && step.summarySnapshot !== null
  const showDiff = compare && canCompare
  const diffs = showDiff ? diffSummary(step.summaryPreviousSnapshot, step.summarySnapshot) : []

  return (
    <section className="context-section">
      <div className="context-section__heading">
        <h3>{effective ? `当前摘要（步骤 ${effective.fromStep} ${effective.fromStep === step.step && step.summaryUpdated ? '生成' : '采用'}）` : '当前摘要'}</h3>
        {canCompare ? (
          <button type="button" className="context-link" onClick={() => setCompare((value) => !value)}>
            {showDiff ? '只看新版' : '和上一版对比'}
          </button>
        ) : null}
      </div>
      {!effective ? (
        <p className="context-muted">还没有摘要：所有原始消息都直接发给模型。</p>
      ) : showDiff ? (
        <dl className="context-summary-fields">
          {diffs.map((field) => (
            <div key={field.key}>
              <dt>{field.label}</dt>
              <dd>
                {field.kept.length + field.added.length + field.removed.length === 0 ? <span className="context-muted">（暂无）</span> : null}
                {field.kept.map((entry) => <div key={`k-${entry}`}>{entry}</div>)}
                {field.added.map((entry) => <div key={`a-${entry}`} className="context-diff-added">+ {entry}</div>)}
                {field.removed.map((entry) => <div key={`r-${entry}`} className="context-diff-removed">− {entry}</div>)}
              </dd>
            </div>
          ))}
        </dl>
      ) : (
        <dl className="context-summary-fields">
          {SUMMARY_FIELDS.map((field) => {
            const entries = summaryEntries(effective.snapshot, field.key)
            return (
              <div key={field.key}>
                <dt>{field.label}</dt>
                <dd>
                  {entries.length === 0 ? <span className="context-muted">（暂无）</span> : entries.map((entry) => <div key={entry}>{entry}</div>)}
                </dd>
              </div>
            )
          })}
        </dl>
      )}
      {step.constraintsPossiblyDropped.length > 0 ? (
        <div className="context-alert" role="alert">
          <strong>⚠ 压缩后，这些约束文字不在新摘要里（可能遗漏或被改写），请核对：</strong>
          <ul>
            {step.constraintsPossiblyDropped.map((entry) => <li key={entry}>"{entry}"</li>)}
          </ul>
          {canPin ? <p>需要一直生效的约定，请写进下方"必须记住的事项"。</p> : null}
        </div>
      ) : null}
    </section>
  )
}

function messagePreview(message: Message): string {
  const parts: string[] = []
  if (message.content) parts.push(message.content)
  for (const call of message.tool_calls ?? []) {
    const args = typeof call.arguments === 'string' ? call.arguments : JSON.stringify(call.arguments)
    parts.push(`→ 调用 ${call.name}(${args})`)
  }
  return parts.join('\n') || '（空）'
}

const RAW_PAGE_SIZE = 20
const EVIDENCE_PAGE_CHARS = 12_000

export function ToolOutputViewer({
  runId,
  target,
  onClose,
}: {
  runId: string
  target: OpenToolOutput
  onClose: () => void
}): React.JSX.Element {
  const toolCallId = target.kind === 'model' ? target.tool.toolCallId : target.toolCallId
  const toolName = target.kind === 'model' ? target.tool.toolName : target.toolName
  const [offset, setOffset] = useState(0)
  useEffect(() => setOffset(0), [toolCallId, target.kind])
  const query = useQuery({
    queryKey: ['run-tool-evidence', runId, toolCallId, offset],
    queryFn: () => getRunToolEvidence(runId, toolCallId, offset),
    enabled: target.kind === 'full',
    retry: false,
  })
  const page = query.data
  return (
    <section className="context-section context-raw" aria-label="工具输出">
      <div className="context-section__heading">
        <h3>{target.kind === 'model' ? `${toolName}：模型收到的内容（已截短）` : `${toolName}：完整工具原文`}</h3>
        <button type="button" className="context-link" onClick={onClose}>收起</button>
      </div>
      {target.kind === 'model' ? (
        <>
          <p className="context-note">模型实际看到的是下面这段截短后的输出（{target.tool.modelOutput.length.toLocaleString()} 字符）。</p>
          <pre className="context-raw__pre">{target.tool.modelOutput || '（空）'}</pre>
        </>
      ) : query.isPending ? <p className="context-muted">正在读取完整原文…</p>
        : query.isError || !page ? (
          <p className="context-warning">{query.error instanceof Error ? query.error.message : '完整原文不可用'}</p>
        ) : (
          <>
            <p className="context-note">
              证据 {page.evidence_id.slice(0, 8)} · 共 {page.total_chars.toLocaleString()} 字符。这是工具实际返回的全文，模型收到的可能只是其中一部分。
            </p>
            <pre className="context-raw__pre">{page.content}</pre>
            <div className="context-raw__pager">
              <button type="button" className="btn btn-sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - EVIDENCE_PAGE_CHARS))}>上一页</button>
              <span className="mono">{(page.offset + 1).toLocaleString()}–{(page.offset + page.content.length).toLocaleString()} / {page.total_chars.toLocaleString()}</span>
              <button type="button" className="btn btn-sm" disabled={page.next_offset === null} onClick={() => page.next_offset !== null && setOffset(page.next_offset)}>下一页</button>
            </div>
          </>
        )}
    </section>
  )
}

function RawMessagesViewer({
  runId,
  range,
  onClose,
  onOpenTool,
}: {
  runId: string
  range: MessageRange
  onClose: () => void
  onOpenTool: (target: OpenToolOutput) => void
}): React.JSX.Element {
  const [offset, setOffset] = useState(range.from)
  useEffect(() => setOffset(range.from), [range.from, range.to])
  const limit = Math.max(1, Math.min(RAW_PAGE_SIZE, range.to - offset))
  const query = useQuery({
    queryKey: ['run-context-messages', runId, offset, limit],
    queryFn: () => getRunContextMessages(runId, offset, limit),
  })
  const items = query.data?.messages ?? []
  const pageEnd = offset + items.length
  return (
    <section className="context-section context-raw" aria-label="原文">
      <div className="context-section__heading">
        <h3>{range.title}</h3>
        <button type="button" className="context-link" onClick={onClose}>收起</button>
      </div>
      <p className="context-note">这里是会话记录里的消息；过长的工具输出只存了截短版，模型请求里可能只看到其中开头和结尾，完整内容点"查看完整工具原文"。</p>
      {query.isPending ? <p className="context-muted">正在读取原文…</p>
        : query.isError ? <p className="context-warning">{query.error instanceof Error ? query.error.message : String(query.error)}</p>
          : items.length === 0 ? <p className="context-muted">这一段没有记录到原文。</p>
            : (
              <ol className="context-raw__list">
                {items.map((item) => (
                  <li key={item.index}>
                    <div className="context-raw__meta mono">
                      #{item.index + 1} · {item.message.role}
                      {item.message.name ? ` · ${item.message.name}` : ''}
                      {item.inherited ? ' · 来自之前的会话' : ''}
                      {item.message.role === 'tool' && item.message.tool_call_id ? (
                        <>
                          {' · '}
                          <button
                            type="button"
                            className="context-link"
                            onClick={() => onOpenTool({
                              kind: 'full',
                              toolCallId: item.message.tool_call_id!,
                              toolName: item.message.name ?? '工具',
                            })}
                          >
                            查看完整工具原文
                          </button>
                        </>
                      ) : null}
                    </div>
                    <pre>{messagePreview(item.message)}</pre>
                  </li>
                ))}
              </ol>
            )}
      <div className="context-raw__pager">
        <button type="button" className="btn btn-sm" disabled={offset <= range.from} onClick={() => setOffset(Math.max(range.from, offset - RAW_PAGE_SIZE))}>上一页</button>
        <span className="mono">{items.length > 0 ? `${offset + 1}–${pageEnd} / ${range.to}` : ''}</span>
        <button type="button" className="btn btn-sm" disabled={pageEnd >= range.to || items.length === 0} onClick={() => setOffset(pageEnd)}>下一页</button>
      </div>
    </section>
  )
}

function PinnedConstraintsEditor({
  conversationId,
  adoptedRevision,
}: {
  conversationId: string
  adoptedRevision: number | null
}): React.JSX.Element {
  const queryClient = useQueryClient()
  const queryKey = ['conversation-constraints', conversationId]
  const query = useQuery({ queryKey, queryFn: () => getConversationConstraints(conversationId) })
  const [draft, setDraft] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const current = query.data
  const text = draft ?? current?.text ?? ''
  const dirty = current !== undefined && draft !== null && draft.trim() !== current.text

  const save = async (): Promise<void> => {
    if (!current) return
    setSaving(true)
    try {
      const saved = await setConversationConstraints(conversationId, text, current.revision)
      queryClient.setQueryData(queryKey, saved)
      setDraft(null)
      toast.success(`已保存第 ${saved.revision} 版，从下一次模型请求开始生效`)
    } catch (cause) {
      toast.error(cause instanceof Error ? cause.message : String(cause))
      void queryClient.invalidateQueries({ queryKey })
    } finally {
      setSaving(false)
    }
  }

  return (
    <section className="context-section">
      <div className="context-section__heading">
        <h3>必须记住的事项</h3>
        <span className="context-stage mono">{current ? `v${current.revision}` : ''}</span>
      </div>
      <p className="context-note">每次请求都会带上，不会被压缩；与较早的历史冲突时以此为准。</p>
      {query.isError ? <p className="context-warning">{query.error instanceof Error ? query.error.message : String(query.error)}</p> : null}
      <textarea
        className="textarea context-constraints__input"
        rows={4}
        value={text}
        disabled={!current || saving}
        placeholder={'每行一条，例如：\n只修改 backend 目录\n不要修改数据库结构'}
        onChange={(event) => setDraft(event.target.value)}
        aria-label="必须记住的事项"
      />
      <div className="context-constraints__footer">
        <button type="button" className="btn btn-primary btn-sm" disabled={!dirty || saving} onClick={() => void save()}>
          {saving ? '保存中…' : '保存'}
        </button>
        <span className="context-note">保存后从下一次模型请求开始生效</span>
      </div>
      {current ? (
        <p className="context-note">
          所选步骤实际采用：{adoptedRevision === null ? '未记录' : adoptedRevision === 0 ? '无（v0）' : `v${adoptedRevision}`}
          （当前最新：v{current.revision}）
        </p>
      ) : null}
    </section>
  )
}

function TokenTransition({ before, after }: { before: number; after: number }): React.JSX.Element {
  return (
    <span className="context-token-transition mono">
      {formatTokens(before)}{before !== after ? ` → ${formatTokens(after)}` : ''}
    </span>
  )
}

function RequestCache({ step }: { step: ContextStepVM }): React.JSX.Element {
  const decisions: Record<string, string> = {
    append: '追加',
    rebuild: '重建',
    compact: '阶段压缩',
    reuse: '追加（旧记录）',
    defer: '追加（旧记录：延后压缩）',
  }
  const localPrefix = step.localPrefixReused === null
    ? '历史记录未提供'
    : step.localPrefixReused
      ? `已复用${step.prefixMessageCount === null ? '' : ` ${step.prefixMessageCount} 条消息`}`
      : '未复用'
  return (
    <section className="context-section">
      <h3>Request &amp; cache</h3>
      <dl className="context-metrics">
        <div><dt>请求策略</dt><dd>{step.prefixDecision ? decisions[step.prefixDecision] ?? step.prefixDecision : '历史记录未提供'}</dd></div>
        <div><dt>本地前缀</dt><dd>{localPrefix}</dd></div>
        <div><dt>重建原因</dt><dd>{step.prefixRebuildReason ?? '—'}</dd></div>
        <div>
          <dt>上游缓存</dt>
          <dd className="mono">
            {!step.modelCompleted
              ? '模型请求尚未完成'
              : step.upstreamCachedTokens === null
                ? '上游未返回缓存统计'
                : <>{formatTokens(step.upstreamCachedTokens)} cached
                  {step.upstreamInputTokens === null ? '' : ` / ${formatTokens(step.upstreamInputTokens)} input`}
                  {step.upstreamCacheRatio === null ? '' : ` · ${formatCacheHitRate(step.upstreamCacheRatio * 100)}`}
                </>}
          </dd>
        </div>
      </dl>
      <p className="context-note">本地前缀复用不代表上游 KV cache 命中；缓存统计缺失不等于 0。</p>
    </section>
  )
}

function CompactionList({ step }: { step: ContextStepVM }): React.JSX.Element {
  const actions = [
    step.compactedToolResults > 0
      ? `${step.compactedToolResults} 个工具结果已压缩`
      : null,
    step.removedToolRounds > 0
      ? `${step.removedToolRounds} 个旧工具轮已移除`
      : null,
    step.summaryUpdated ? 'Conversation summary 已更新' : null,
  ].filter((item): item is string => item !== null)

  return (
    <section className="context-section">
      <div className="context-section__heading">
        <h3>Compaction</h3>
        <span className={`context-stage ${step.compactionStage === 'none' ? '' : 'active'}`}>
          {step.compactionStage === 'none' ? '未触发' : step.compactionStage.replaceAll('_', ' ')}
        </span>
      </div>
      {actions.length > 0 ? (
        <ul className="context-checks">
          {actions.map((action) => <li key={action}>✓ {action}</li>)}
        </ul>
      ) : (
        <p className="context-muted">本步骤不需要压缩。</p>
      )}
      {step.summaryError ? <p className="context-warning">摘要更新失败：{step.summaryError}</p> : null}
      {step.summaryProvider || step.summaryUsage > 0 || step.summaryError ? (
        <dl className="context-summary-details">
          <div><dt>摘要模型</dt><dd>{step.summaryProvider ? `${step.summaryProvider} / ${step.summaryModel ?? 'default'}` : '历史记录未提供'}</dd></div>
          <div><dt>摘要用量</dt><dd className="mono">{formatTokens(step.summaryUsage)}</dd></div>
          <div><dt>耗时</dt><dd className="mono">{step.summaryDurationMs === null ? '—' : `${Math.round(step.summaryDurationMs)} ms`}</dd></div>
          <div><dt>原始历史</dt><dd>{step.summaryError ? '已保留' : '摘要成功后仍由数据库完整保存'}</dd></div>
        </dl>
      ) : null}
    </section>
  )
}

export default function ContextInspector({
  events,
  runId,
  conversationId,
}: {
  events: AgentEvent[]
  /** 提供后可查看原文。 */
  runId?: string
  /** 提供后显示"必须记住的事项"编辑框；长任务的子运行不传。 */
  conversationId?: string | null
}): React.JSX.Element {
  const steps = useMemo(() => buildContextSteps(events), [events])
  const [selectedStep, setSelectedStep] = useState<number | null>(null)
  const [openRange, setOpenRange] = useState<MessageRange | null>(null)
  const [openTool, setOpenTool] = useState<OpenToolOutput | null>(null)
  const selected = steps.find((step) => step.step === selectedStep) ?? steps.at(-1)

  useEffect(() => {
    if (selectedStep === null && steps.length > 0) setSelectedStep(steps.at(-1)!.step)
    if (selectedStep !== null && !steps.some((step) => step.step === selectedStep)) {
      setSelectedStep(steps.at(-1)?.step ?? null)
    }
  }, [selectedStep, steps])

  if (!selected) {
    return (
      <div className="context-inspector">
        <EmptyState title="暂无 Context 数据" hint="模型请求开始后会记录上下文构成。" />
        {conversationId ? <PinnedConstraintsEditor conversationId={conversationId} adoptedRevision={null} /> : null}
      </div>
    )
  }

  return (
    <div className="context-inspector">
      <div className="context-step-picker" aria-label="Context model step">
        <span>Model step</span>
        <div>
          {steps.map((step) => (
            <button
              key={step.step}
              type="button"
              className={step.step === selected.step ? 'active' : ''}
              aria-pressed={step.step === selected.step}
              onClick={() => setSelectedStep(step.step)}
            >
              {step.step}
            </button>
          ))}
        </div>
      </div>

      <UsageBar step={selected} />
      <ContextTimeline
        steps={steps}
        selected={selected.step}
        onSelect={setSelectedStep}
        onOpenRange={setOpenRange}
        onOpenTool={runId ? setOpenTool : undefined}
      />
      {openRange && runId ? (
        <RawMessagesViewer runId={runId} range={openRange} onClose={() => setOpenRange(null)} onOpenTool={setOpenTool} />
      ) : null}
      {openTool && runId ? (
        <ToolOutputViewer runId={runId} target={openTool} onClose={() => setOpenTool(null)} />
      ) : null}
      <SummaryPanel key={selected.step} steps={steps} step={selected} canPin={Boolean(conversationId)} />
      {conversationId ? (
        <PinnedConstraintsEditor conversationId={conversationId} adoptedRevision={selected.constraintsRevision} />
      ) : null}

      <section className="context-section">
        <h3>Context</h3>
        <dl className="context-metrics">
          <div><dt>Input</dt><dd><TokenTransition before={selected.originalInputTokens} after={selected.preparedInputTokens} /></dd></div>
          <div><dt>Context window</dt><dd className="mono">{formatTokens(selected.contextWindow)}</dd></div>
          <div><dt>Working budget</dt><dd className="mono">{formatTokens(selected.workingInputBudget)}</dd></div>
          <div><dt>Model input limit</dt><dd className="mono">{formatTokens(selected.inputBudget)}</dd></div>
          <div><dt>Trigger / target</dt><dd className="mono">{formatTokens(selected.triggerTokens)} / {formatTokens(selected.targetTokens)}</dd></div>
          <div><dt>Window usage</dt><dd className="mono">{(selected.windowUsageRatio * 100).toFixed(1)}%</dd></div>
          <div><dt>Budget usage</dt><dd className="mono">{(selected.budgetUsageRatio * 100).toFixed(1)}%</dd></div>
          <div><dt>Tool schemas</dt><dd className="mono">{formatTokens(selected.toolSchemaTokens)}</dd></div>
          <div><dt>Tool results</dt><dd><TokenTransition before={selected.toolResultTokensBefore} after={selected.toolResultTokensAfter} /></dd></div>
          <div><dt>Messages</dt><dd><TokenTransition before={selected.messageTokensBefore} after={selected.messageTokensAfter} /></dd></div>
          <div><dt>Skills</dt><dd className="mono">{formatTokens(selected.skillTokens)}</dd></div>
        </dl>
        <progress
          className="context-window-progress"
          max={Math.max(1, selected.contextWindow)}
          value={Math.min(selected.preparedInputTokens, Math.max(1, selected.contextWindow))}
          aria-label="Context window usage"
        />
      </section>

      <section className="context-section">
        <h3>Context breakdown</h3>
        <div className="context-breakdown">
          {selected.breakdown.map((item) => (
            <div key={item.key} className="context-breakdown__row">
              <span>{item.label}</span>
              <progress max={1} value={item.ratio} aria-label={`${item.label} ratio`} />
              <strong className="mono">{formatTokens(item.tokens)}</strong>
              <small className="mono">{(item.ratio * 100).toFixed(0)}%</small>
            </div>
          ))}
        </div>
        <p className="context-note">Memory、Task 与系统消息当前没有独立计数字段，统一包含在 Messages &amp; injected。</p>
      </section>

      <RequestCache step={selected} />
      <CompactionList step={selected} />

    </div>
  )
}
