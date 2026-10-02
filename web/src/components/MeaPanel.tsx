import { useEffect, useState } from 'react'
import type { ReactElement, ReactNode } from 'react'

import {
  MEA_STATUS_LABEL,
  ROUND_KIND_LABEL,
  countedRounds,
  isMeaOpen,
  liveActivityText,
  meaReason,
  roundVerdict,
  stepProgress,
  stepSymbol,
} from '../agent/meaPresentation'
import type { MeaNoteKind } from '../api/mea'
import type { MeaDetail, MeaLiveActivity, MeaRound, MeaRun } from '../api/types'
import { Icon } from './Icon'

export interface MeaPanelProps {
  mea: MeaRun
  /** mea.get 的结果；还没读到时为 null，面板只显示状态。 */
  detail: MeaDetail | null
  live?: MeaLiveActivity | null
  busy?: boolean
  /** 最近一次操作的结果提示，例如“已作为补充要求 A2 交给长任务”。 */
  notice?: string | null
  defaultOpen?: boolean
  onAnswer: (text: string) => void
  onNote: (text: string, options: { kind: MeaNoteKind; immediate: boolean }) => void
  onPause: () => void
  onResume: (extraRounds: number) => void
  onCancel: () => void
}

/** 长任务面板：状态、需要你处理的事（回答 / 继续）、步骤、每轮的执行与审计记录。 */
export default function MeaPanel({
  mea,
  detail,
  live = null,
  busy = false,
  notice = null,
  defaultOpen = false,
  onAnswer,
  onNote,
  onPause,
  onResume,
  onCancel,
}: MeaPanelProps): ReactElement {
  const rounds = detail?.rounds ?? []
  const steps = detail?.task?.steps ?? []
  const progress = stepProgress(steps)
  const used = countedRounds(rounds)
  const liveText = mea.status === 'running' ? liveActivityText(live) : null
  const headline = mea.status === 'waiting_user'
    ? mea.pending_question ?? '需要你确认后继续'
    : liveText ?? meaReason(mea) ?? detail?.task?.title ?? null
  // 需要你处理（回答、继续）时自动展开；其余时候保持你上次的展开状态
  const needsAttention = mea.status === 'waiting_user' || mea.status === 'paused'
  const [open, setOpen] = useState(defaultOpen || needsAttention)
  useEffect(() => {
    if (needsAttention) setOpen(true)
  }, [needsAttention])

  return (
    <section className={`mea-panel mea-panel--${mea.status}`} aria-label="长任务" data-testid="mea-panel">
      <details
        open={open}
        onToggle={(event) => setOpen((event.currentTarget as HTMLDetailsElement).open)}
      >
        <summary>
          <span className={`mea-panel__icon mea-panel__icon--${mea.status}`}>
            <Icon name={mea.status === 'running' || mea.status === 'finalizing' ? 'activity' : 'runs'} size={15} />
          </span>
          <div className="mea-panel__identity">
            <small>长任务 · {MEA_STATUS_LABEL[mea.status]}</small>
            <strong>{detail?.task?.title ?? '长任务'}</strong>
            {headline ? <span>{headline}</span> : null}
          </div>
          <div className="mea-panel__progress">
            <span>第 {used}/{mea.round_budget} 轮{progress.total > 0 ? ` · 步骤 ${progress.done}/${progress.total}` : ''}</span>
            <i><b style={{ width: `${progress.percent}%` }} /></i>
          </div>
          <Icon name="chevronDown" size={15} />
        </summary>

        <div className="mea-panel__body">
          <StatusCallout
            mea={mea}
            busy={busy}
            liveText={liveText}
            onAnswer={onAnswer}
            onPause={onPause}
            onResume={onResume}
            onCancel={onCancel}
          />
          {notice ? <p className="mea-panel__notice" role="status">{notice}</p> : null}

          {steps.length > 0 ? (
            <MeaSection title="步骤">
              <ol className="mea-steps">
                {steps.map((step) => (
                  <li key={step.id} className={`mea-step mea-step--${step.status}`}>
                    <span aria-hidden="true">{stepSymbol(step.status)}</span>
                    <div>
                      <strong><code>{step.id}</code> {step.title}</strong>
                      {step.status === 'superseded' ? (
                        <small>
                          已被修订 {step.superseded_by ?? '?'} 取代
                          {step.superseded_reason ? `：${step.superseded_reason}` : ''}
                        </small>
                      ) : step.acceptance ? (
                        <small>验收：{step.acceptance}</small>
                      ) : null}
                    </div>
                  </li>
                ))}
              </ol>
            </MeaSection>
          ) : null}

          {detail && detail.requirements.amendments.length > 0 ? (
            <MeaSection title={`你的补充（要求 v${detail.requirements.revision}）`}>
              <ul className="mea-amendments">
                {detail.requirements.amendments.map((amendment) => (
                  <li key={amendment.id}>
                    <code>{amendment.id}</code>
                    <span>{amendment.kind === 'answer' ? '回答' : '补充'}</span>
                    <p>{amendment.text}</p>
                  </li>
                ))}
              </ul>
            </MeaSection>
          ) : null}

          {rounds.length > 0 ? (
            <MeaSection title={`轮次记录（${rounds.length}）`}>
              <div className="mea-rounds">
                {[...rounds].reverse().map((round) => (
                  <RoundItem key={round.index} round={round} />
                ))}
              </div>
            </MeaSection>
          ) : detail ? (
            <p className="empty-inline">还没有开始第一轮。</p>
          ) : null}

          {isMeaOpen(mea.status) && mea.status !== 'finalizing' ? (
            <NoteForm busy={busy} onNote={onNote} />
          ) : null}
        </div>
      </details>
    </section>
  )
}

function StatusCallout({
  mea,
  busy,
  liveText,
  onAnswer,
  onPause,
  onResume,
  onCancel,
}: {
  mea: MeaRun
  busy: boolean
  liveText: string | null
  onAnswer: (text: string) => void
  onPause: () => void
  onResume: (extraRounds: number) => void
  onCancel: () => void
}): ReactElement | null {
  const [answer, setAnswer] = useState('')
  const [extraRounds, setExtraRounds] = useState(mea.abort_reason === 'max_rounds' ? 10 : 0)
  const reason = meaReason(mea)

  switch (mea.status) {
    case 'running':
      return (
        <div className="mea-callout">
          <p>{mea.pause_requested ? '已请求暂停，当前这一轮结束后停下。' : liveText ?? '正在推进下一轮。'}</p>
          <div className="mea-callout__actions">
            <button type="button" className="btn" disabled={busy || mea.pause_requested} onClick={onPause}>
              暂停
            </button>
            <button type="button" className="btn btn-text-danger" disabled={busy} onClick={onCancel}>
              取消长任务
            </button>
          </div>
        </div>
      )
    case 'waiting_user':
      return (
        <div className="mea-callout mea-callout--question">
          <strong>{mea.pending_question ?? '需要你确认后继续'}</strong>
          {mea.pending_choices.length > 0 ? (
            <div className="mea-choices">
              {mea.pending_choices.map((choice) => (
                <button key={choice} type="button" className="btn" disabled={busy} onClick={() => onAnswer(choice)}>
                  {choice}
                </button>
              ))}
            </div>
          ) : null}
          <textarea
            className="textarea mea-callout__input"
            value={answer}
            onChange={(event) => setAnswer(event.target.value)}
            placeholder="写下你的回答，它会作为新的要求交给长任务"
            rows={2}
            disabled={busy}
            aria-label="回答长任务的问题"
          />
          <div className="mea-callout__actions">
            <button
              type="button"
              className="btn btn-primary"
              disabled={busy || answer.trim() === ''}
              onClick={() => {
                onAnswer(answer.trim())
                setAnswer('')
              }}
            >
              回答并继续
            </button>
            <button type="button" className="btn" disabled={busy} onClick={() => onResume(0)}>
              不回答，直接继续
            </button>
            <button type="button" className="btn btn-text-danger" disabled={busy} onClick={onCancel}>
              取消长任务
            </button>
          </div>
        </div>
      )
    case 'paused':
      return (
        <div className="mea-callout">
          <p>{reason ?? '长任务已暂停。点继续后会先检查中断时的状态，再接着推进。'}</p>
          <div className="mea-callout__actions">
            <label className="mea-callout__rounds">
              追加轮次
              <input
                className="input"
                type="number"
                min={0}
                max={200}
                value={extraRounds}
                onChange={(event) => setExtraRounds(Math.max(0, Number(event.target.value) || 0))}
                disabled={busy}
              />
            </label>
            <button type="button" className="btn btn-primary" disabled={busy} onClick={() => onResume(extraRounds)}>
              继续
            </button>
            <button type="button" className="btn btn-text-danger" disabled={busy} onClick={onCancel}>
              取消长任务
            </button>
          </div>
        </div>
      )
    case 'finalizing':
      return (
        <div className="mea-callout">
          <p>终结决定已记录，正在生成最终回复。这时不再接受补充，新的要求会作为普通消息发送。</p>
        </div>
      )
    case 'completed':
    case 'blocked':
      return (
        <div className={`mea-callout mea-callout--${mea.status}`}>
          <strong>{mea.status === 'completed' ? '已通过最终验收' : '长任务无法继续推进，已停止'}</strong>
          {mea.status === 'blocked' && mea.completion_decision?.reason ? (
            <pre className="mea-text">{mea.completion_decision.reason}</pre>
          ) : null}
          {mea.final_response ? <pre className="mea-text">{mea.final_response}</pre> : null}
        </div>
      )
    case 'failed':
    case 'cancelled':
      return (
        <div className={`mea-callout mea-callout--${mea.status}`}>
          <strong>{mea.status === 'failed' ? '长任务出错停止' : '长任务已取消'}</strong>
          {reason ? <p>{reason}</p> : null}
        </div>
      )
    default:
      return null
  }
}

function RoundItem({ round }: { round: MeaRound }): ReactElement {
  const [open, setOpen] = useState(false)
  const verdict = roundVerdict(round)
  const ref = `round_${String(round.index).padStart(3, '0')}`
  return (
    <details
      className={`mea-round mea-round--${round.phase}`}
      onToggle={(event) => setOpen((event.currentTarget as HTMLDetailsElement).open)}
    >
      <summary>
        <span className="mea-round__ref mono">{ref}</span>
        <span className="mea-round__kind">
          {ROUND_KIND_LABEL[round.kind]}
          {round.step_id ? <code>{round.step_id}</code> : null}
        </span>
        <span className={`mea-verdict mea-verdict--${verdict.tone}`}>{verdict.label}</span>
        <Icon name="chevronDown" size={14} />
      </summary>
      {open ? (
        <div className="mea-round__body">
          {round.subtask ? <RoundText title="子任务" text={round.subtask} /> : null}
          {round.focus && round.kind !== 'normal' ? <RoundText title="核实重点" text={round.focus} /> : null}
          {round.plan_text ? <RoundText title="管理者的判断" text={round.plan_text} /> : null}
          {round.executor_output ? <RoundText title="执行报告" text={round.executor_output} /> : null}
          {round.auditor_report ? <RoundText title="审计报告" text={round.auditor_report} /> : null}
          {round.harness_feedback ? <RoundText title="系统反馈" text={round.harness_feedback} /> : null}
          {round.abandon_reason ? <RoundText title="作废原因" text={round.abandon_reason} /> : null}
          {round.interrupt_reason ? <RoundText title="中断原因" text={round.interrupt_reason} /> : null}
        </div>
      ) : null}
    </details>
  )
}

function RoundText({ title, text }: { title: string; text: string }): ReactElement {
  return (
    <div className="mea-round__section">
      <h4>{title}</h4>
      <pre className="mea-text">{text}</pre>
    </div>
  )
}

function MeaSection({ title, children }: { title: string; children: ReactNode }): ReactElement {
  return (
    <div className="mea-section">
      <h3>{title}</h3>
      {children}
    </div>
  )
}

function NoteForm({
  busy,
  onNote,
}: {
  busy: boolean
  onNote: (text: string, options: { kind: MeaNoteKind; immediate: boolean }) => void
}): ReactElement {
  const [text, setText] = useState('')
  const [kind, setKind] = useState<MeaNoteKind>('persistent')
  const [immediate, setImmediateFlag] = useState(false)
  return (
    <div className="mea-note">
      <h3>补充指令</h3>
      <textarea
        className="textarea"
        value={text}
        onChange={(event) => setText(event.target.value)}
        placeholder="例如：数据库改用测试库；不要动 migrations 目录"
        rows={2}
        disabled={busy}
        aria-label="长任务补充指令"
      />
      <div className="mea-note__options">
        <label>
          <input
            type="radio"
            name="mea-note-kind"
            checked={kind === 'persistent'}
            onChange={() => setKind('persistent')}
            disabled={busy}
          />
          持续有效（写进要求）
        </label>
        <label>
          <input
            type="radio"
            name="mea-note-kind"
            checked={kind === 'once'}
            onChange={() => setKind('once')}
            disabled={busy}
          />
          只给下一轮
        </label>
        <label>
          <input
            type="checkbox"
            checked={immediate}
            onChange={(event) => setImmediateFlag(event.target.checked)}
            disabled={busy}
          />
          立即生效（中断当前子任务）
        </label>
        <button
          type="button"
          className="btn"
          disabled={busy || text.trim() === ''}
          onClick={() => {
            onNote(text.trim(), { kind, immediate })
            setText('')
            setImmediateFlag(false)
          }}
        >
          发送补充
        </button>
      </div>
    </div>
  )
}
