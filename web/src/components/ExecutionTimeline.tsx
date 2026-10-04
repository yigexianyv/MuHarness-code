import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useState } from 'react'

import { applyRewind, listRunSteps, previewRewind } from '../api/runs'
import type { AgentEvent, RewindStepInfo, ToolCall } from '../api/types'
import { toast } from '../stores/toasts'

export interface TimelineStepVM {
  step: number
  /** 这一步模型做了什么：工具调用或最终回答。 */
  label: string
  outcome: 'ok' | 'failed' | 'pending' | 'answer'
  failedTools: number
}

function argumentsPreview(call: ToolCall): string {
  const raw = typeof call.arguments === 'string' ? call.arguments : JSON.stringify(call.arguments)
  return raw.length > 60 ? `${raw.slice(0, 59)}…` : raw
}

/** 从执行事件整理每一步的动作和结果。 */
export function buildTimelineSteps(events: AgentEvent[]): TimelineStepVM[] {
  const steps = new Map<number, TimelineStepVM>()
  for (const event of [...events].sort((a, b) => a.sequence - b.sequence)) {
    if (event.step == null) continue
    if (event.type === 'model_completed' && event.message) {
      const calls = event.message.tool_calls ?? []
      const label = calls.length > 0
        ? calls.map((call) => `${call.name}(${argumentsPreview(call)})`).join('，')
        : `回答：${(event.message.content ?? '').trim().slice(0, 60) || '（空）'}`
      steps.set(event.step, {
        step: event.step,
        label,
        outcome: calls.length > 0 ? 'pending' : 'answer',
        failedTools: 0,
      })
    } else if (event.type === 'tool_completed' && event.tool_result) {
      const current = steps.get(event.step)
      if (!current) continue
      const failed = current.failedTools + (event.tool_result.success ? 0 : 1)
      steps.set(event.step, { ...current, failedTools: failed, outcome: failed > 0 ? 'failed' : 'ok' })
    }
  }
  return [...steps.values()].sort((a, b) => a.step - b.step)
}

function outcomeLabel(step: TimelineStepVM): string {
  if (step.outcome === 'answer') return ''
  if (step.outcome === 'pending') return '…'
  return step.failedTools > 0 ? `✗ ${step.failedTools} 失败` : '✓'
}

function newRewindKey(): string {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export function RewindDialog({
  runId,
  step,
  onClose,
  onRewound,
}: {
  runId: string
  step: number
  onClose: () => void
  onRewound: (conversationId: string, draft: string) => void
}): React.JSX.Element {
  const queryClient = useQueryClient()
  const queryKey = ['rewind-preview', runId, step]
  const preview = useQuery({ queryKey, queryFn: () => previewRewind(runId, step), retry: false })
  const [correction, setCorrection] = useState('')
  const [busy, setBusy] = useState(false)
  // 同一次确认重复提交（双击、网络重试）只生效一次
  const [rewindKey] = useState(newRewindKey)
  const data = preview.data
  const blocked = data?.blocked_reason ?? null

  const confirm = async (): Promise<void> => {
    if (!data) return
    setBusy(true)
    try {
      const result = await applyRewind({
        runId,
        step,
        previewId: data.preview_id,
        correction,
        rewindKey,
      })
      toast.success(`已回到第 ${step} 步之前：恢复 ${result.files_restored} 个文件，删除 ${result.files_deleted} 个`)
      void queryClient.invalidateQueries({ queryKey: ['conversations'] })
      onRewound(result.conversation_id, result.draft)
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause)
      toast.error(message)
      void queryClient.invalidateQueries({ queryKey })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="dialog-overlay" role="presentation">
      <div className="dialog rewind-dialog" role="dialog" aria-modal="true" aria-label={`回到第 ${step} 步之前`}>
        <h3 className="dialog__title">回到第 {step} 步之前</h3>
        <p className="rewind-dialog__lead">
          {step > 1 ? `将恢复到"第 ${step - 1} 步完成后"的状态，` : '将恢复到执行开始时的状态，'}
          前面的步骤不会重跑；原来的执行记录会保留。
        </p>
        {preview.isPending ? <p className="context-muted">正在计算文件变化…</p>
          : preview.isError && !data ? <p className="context-warning">{preview.error instanceof Error ? preview.error.message : String(preview.error)}</p>
            : data ? (
              <>
                <section>
                  <h4>文件变化（共 {data.files.length} 个）</h4>
                  {data.files.length === 0 ? <p className="context-muted">工作区文件与第 {step} 步之前一致，无需恢复。</p> : (
                    <ul className="rewind-dialog__files">
                      {data.files.map((file) => (
                        <li key={file.path}>
                          <span className="mono">{file.action === 'restore' ? '~' : '+'} {file.path}</span>
                          <span>
                            {file.action === 'restore' ? `恢复为第 ${step} 步之前的内容` : `将被删除（第 ${step} 步之后新建）`}
                            {file.changed_after_run ? <em className="rewind-dialog__warn">（⚠ 执行结束后又被改过，这些修改也会被覆盖）</em> : null}
                          </span>
                        </li>
                      ))}
                    </ul>
                  )}
                  {data.skipped_files.length > 0 ? (
                    <p className="context-note">{data.skipped_files.length} 个过大文件或链接不在快照内，保持现状：{data.skipped_files.slice(0, 5).join('、')}{data.skipped_files.length > 5 ? ' 等' : ''}</p>
                  ) : null}
                </section>
                {data.irreversible.length > 0 ? (
                  <section>
                    <h4>以下操作不会回退</h4>
                    <ul className="rewind-dialog__ops">
                      {data.irreversible.map((op, index) => (
                        <li key={`${op.tool}-${index}`}>
                          <span className="mono">第 {op.step ?? '?'} 步 {op.tool}({op.summary})</span>
                          <small>{op.note}</small>
                        </li>
                      ))}
                    </ul>
                  </section>
                ) : null}
                {blocked ? <p className="context-warning" role="alert">{blocked}</p> : null}
                <section>
                  <h4>你的纠正</h4>
                  <textarea
                    className="textarea"
                    rows={3}
                    value={correction}
                    onChange={(event) => setCorrection(event.target.value)}
                    placeholder="例如：不要改 schema，问题在 export.py 的分页逻辑"
                    aria-label="你的纠正"
                  />
                  <p className="context-note">确认后会恢复文件并打开一个新会话，纠正内容已填好，检查后点发送即重新执行。</p>
                </section>
              </>
            ) : null}
        <div className="dialog__actions">
          <button type="button" className="btn btn-ghost btn-sm" disabled={busy} onClick={onClose}>取消</button>
          <button
            type="button"
            className="btn btn-primary btn-sm"
            disabled={busy || !data || Boolean(blocked) || !correction.trim()}
            onClick={() => void confirm()}
          >
            {busy ? '正在回退…' : '确认并回到这一步'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default function ExecutionTimeline({
  runId,
  events,
  onRewound,
}: {
  runId: string
  events: AgentEvent[]
  onRewound?: (conversationId: string, draft: string) => void
}): React.JSX.Element | null {
  const steps = useMemo(() => buildTimelineSteps(events), [events])
  const checkpoints = useQuery({
    queryKey: ['run-steps', runId],
    queryFn: () => listRunSteps(runId),
    refetchInterval: 5000,
  })
  const [rewindStep, setRewindStep] = useState<number | null>(null)
  const byStep = new Map<number, RewindStepInfo>((checkpoints.data ?? []).map((info) => [info.step, info]))

  if (steps.length === 0) return null
  return (
    <>
      <ol className="exec-timeline">
        {steps.map((step) => {
          const info = byStep.get(step.step)
          return (
            <li key={step.step}>
              <span className="exec-timeline__step">步骤 {step.step}</span>
              <span className="exec-timeline__label mono" title={step.label}>{step.label}</span>
              <span className={`exec-timeline__outcome ${step.outcome}`}>{outcomeLabel(step)}</span>
              {onRewound ? (
                <button
                  type="button"
                  className="btn btn-sm"
                  disabled={!info?.rewindable}
                  title={info?.reason ?? (info ? '回到这一步之前，改个决策重新执行' : '没有检查点记录')}
                  onClick={() => setRewindStep(step.step)}
                >
                  重做此步
                </button>
              ) : null}
            </li>
          )
        })}
      </ol>
      {rewindStep !== null && onRewound ? (
        <RewindDialog
          runId={runId}
          step={rewindStep}
          onClose={() => setRewindStep(null)}
          onRewound={(conversationId, draft) => {
            setRewindStep(null)
            onRewound(conversationId, draft)
          }}
        />
      ) : null}
    </>
  )
}
