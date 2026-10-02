import { useState } from 'react'
import type { ReactElement } from 'react'

import type { Task } from '../api/types'
import { Icon } from './Icon'

export interface LongTaskOptions {
  roundBudget: number
  extraTools: string[]
  /** 执行者在沙箱内的 shell 命令自动批准（审计者的只读命令总是自动批准）。 */
  autoApproveSandbox: boolean
}

const DEFAULT_ROUND_BUDGET = 25

export default function PlanCard({
  task,
  busy = false,
  onAccept,
  onReject,
  onAcceptLongTask,
  longTaskTools = [],
}: {
  task: Task
  busy?: boolean
  onAccept: (id: string) => void
  onReject: (id: string) => void
  /** 提供时显示“接受并长任务执行”：按轮次 Manager → Executor → Auditor 推进并逐步审计。 */
  onAcceptLongTask?: (id: string, options: LongTaskOptions) => void
  /** 可以额外授权给长任务 Executor 的工具（来自 mea.tools）。 */
  longTaskTools?: string[]
}): ReactElement {
  const [optionsOpen, setOptionsOpen] = useState(false)
  const [roundBudget, setRoundBudget] = useState(DEFAULT_ROUND_BUDGET)
  const [extraTools, setExtraTools] = useState<string[]>([])
  const [autoApproveSandbox, setAutoApproveSandbox] = useState(true)
  const missingAcceptance = task.steps.filter((step) => !step.acceptance?.trim())

  const toggleTool = (name: string): void => {
    setExtraTools((current) =>
      current.includes(name) ? current.filter((item) => item !== name) : [...current, name],
    )
  }

  return (
    <section className="plan-card" data-testid="plan-card">
      <div className="plan-card__eyebrow">
        <Icon name="runs" size={14} />
        Plan
      </div>
      <h2>{task.title}</h2>
      {task.goal ? <p className="plan-card__goal">{task.goal}</p> : null}
      <ol className="plan-card__steps">
        {task.steps.map((step, index) => (
          <li key={step.id ?? index}>
            <span>{index + 1}</span>
            <div>
              {step.title}
              {step.acceptance ? (
                <small className="plan-card__acceptance">验收：{step.acceptance}</small>
              ) : null}
            </div>
          </li>
        ))}
      </ol>
      {onAcceptLongTask ? (
        <p className="plan-card__hint">普通执行由同一个 Agent 推进；长任务执行按管理、执行、独立审计循环推进。</p>
      ) : null}
      <div className="plan-card__actions">
        <button
          type="button"
          className="btn btn-primary"
          disabled={busy}
          onClick={() => onAccept(task.id)}
        >
          接受并普通执行
        </button>
        {onAcceptLongTask ? (
          <button
            type="button"
            className="btn"
            disabled={busy || missingAcceptance.length > 0}
            title={
              missingAcceptance.length > 0
                ? '有步骤缺少验收标准，长任务无法逐步审计；请让它重新规划。'
                : '按轮次执行，每一步都由只读的审计者独立验收。'
            }
            onClick={() => onAcceptLongTask(task.id, { roundBudget, extraTools, autoApproveSandbox })}
          >
            接受并长任务执行
          </button>
        ) : null}
        <button
          type="button"
          className="btn btn-text-danger"
          disabled={busy}
          onClick={() => onReject(task.id)}
        >
          拒绝计划
        </button>
        {onAcceptLongTask ? (
          <button
            type="button"
            className="btn btn-ghost plan-card__options-toggle"
            aria-expanded={optionsOpen}
            onClick={() => setOptionsOpen((open) => !open)}
          >
            长任务选项
          </button>
        ) : null}
      </div>
      {onAcceptLongTask && missingAcceptance.length > 0 ? (
        <p className="plan-card__hint">
          {missingAcceptance.length} 个步骤缺少验收标准，不能按长任务执行。
        </p>
      ) : null}
      {onAcceptLongTask && optionsOpen ? (
        <div className="plan-card__options">
          <label>
            轮次预算
            <input
              className="input"
              type="number"
              min={1}
              max={200}
              value={roundBudget}
              onChange={(event) =>
                setRoundBudget(Math.min(200, Math.max(1, Number(event.target.value) || 1)))
              }
            />
          </label>
          <label className="plan-card__option-check">
            <input
              type="checkbox"
              checked={autoApproveSandbox}
              onChange={(event) => setAutoApproveSandbox(event.target.checked)}
            />
            自动批准执行者在沙箱内的命令（无网络，只能改工作区；网络类工具仍需批准）
          </label>
          {longTaskTools.length > 0 ? (
            <fieldset>
              <legend>额外授权给执行者的工具</legend>
              {longTaskTools.map((name) => (
                <label key={name}>
                  <input
                    type="checkbox"
                    checked={extraTools.includes(name)}
                    onChange={() => toggleTool(name)}
                  />
                  <code>{name}</code>
                </label>
              ))}
            </fieldset>
          ) : null}
        </div>
      ) : null}
    </section>
  )
}
