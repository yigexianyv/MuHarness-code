

import { stepProgress, stepSymbol } from '../agent/meaPresentation'
import type { Task, TaskStatus } from '../api/types'
import { Icon } from './Icon'

const STATUS_LABEL: Record<TaskStatus, string> = {
  pending: '等待确认',
  active: '进行中',
  paused: '已暂停',
  completed: '已完成',
  failed: '失败',
  cancelled: '已取消',
}

const STATUS_ORDER: Record<TaskStatus, number> = {
  active: 0,
  pending: 1,
  paused: 2,
  failed: 3,
  completed: 4,
  cancelled: 5,
}

// 被用户修订取代的步骤不计入进度
function progress(task: Task): { done: number; total: number; percent: number } {
  return stepProgress(task.steps)
}

function currentStep(task: Task): string | null {
  return task.steps.find((step) => step.status === 'in_progress')?.title
    ?? task.steps.find((step) => step.status === 'blocked')?.title
    ?? task.steps.find((step) => step.status === 'todo')?.title
    ?? null
}

export function orderConversationTasks(tasks: Task[]): Task[] {
  return [...tasks].sort((left, right) => {
    const status = STATUS_ORDER[left.status] - STATUS_ORDER[right.status]
    return status || right.updated_at.localeCompare(left.updated_at)
  })
}

export default function CurrentTaskPanel({
  tasks,
  busy = false,
  onExecute,
  onExecuteLong,
  onRestart,
  lockedTaskIds = [],
}: {
  tasks: Task[]
  busy?: boolean
  onExecute?: (task: Task) => void
  onRestart?: (task: Task) => void
  /** 用长任务（逐轮执行 + 审计）继续这个任务。 */
  onExecuteLong?: (task: Task) => void
  /** 这些任务有未结束的长任务：不再提供普通执行入口。 */
  lockedTaskIds?: string[]
}): React.JSX.Element | null {
  const ordered = orderConversationTasks(tasks)
  const current = ordered[0]
  if (!current) return null
  const currentProgress = progress(current)
  const step = currentStep(current)

  return (
    <section className="current-task-panel" aria-label="当前会话任务">
      <details open>
        <summary>
          <span className={`current-task-panel__icon current-task-panel__icon--${current.status}`}>
            <Icon name="runs" size={15} />
          </span>
          <div className="current-task-panel__identity">
            <small>当前任务 · {STATUS_LABEL[current.status]}</small>
            <strong>{current.title}</strong>
            {step ? <span>当前步骤：{step}</span> : null}
          </div>
          <div className="current-task-panel__progress">
            <span>{currentProgress.total > 0 ? `${currentProgress.done}/${currentProgress.total}` : '暂无步骤'}</span>
            <i><b style={{ width: `${currentProgress.percent}%` }} /></i>
          </div>
          {ordered.length > 1 ? <span className="current-task-panel__count">{ordered.length} 个任务</span> : null}
          <Icon name="chevronDown" size={15} />
        </summary>

        <div className="current-task-panel__body">
          {ordered.map((task) => {
            const taskProgress = progress(task)
            const steps = task.steps.filter((step) => step.status !== 'superseded')
            const missingAcceptance = steps.length === 0 || steps.some((step) => !step.acceptance?.trim())
            return (
              <details key={task.id} className="current-task-item" open={task.id === current.id}>
                <summary>
                  <div><strong>{task.title}</strong><span>{STATUS_LABEL[task.status]}</span></div>
                  <span>{taskProgress.total > 0 ? `${taskProgress.done}/${taskProgress.total}` : '无步骤'}</span>
                  <Icon name="chevronDown" size={14} />
                </summary>
                <div className="current-task-item__details">
                  {task.goal ? <p>{task.goal}</p> : task.description ? <p>{task.description}</p> : null}
                  {task.status === 'active' && lockedTaskIds.includes(task.id) ? (
                    <span className="empty-inline">长任务正在处理这个任务，进度见上方的长任务面板。</span>
                  ) : task.status === 'active' && (onExecute || onExecuteLong) ? (
                    <div className="current-task-item__actions">
                      {onExecute ? (
                        <button
                          type="button"
                          className="btn btn-primary"
                          disabled={busy || missingAcceptance}
                          title="直接执行所选任务的剩余步骤，已完成步骤保持不变。"
                          onClick={() => onExecute(task)}
                        >
                          {busy ? '正在启动…' : '继续剩余阶段'}
                        </button>
                      ) : null}
                      {onExecuteLong ? (
                        <button
                          type="button"
                          className="btn"
                          disabled={busy || task.steps.some((step) => step.status !== 'superseded' && !step.acceptance?.trim())}
                          title="逐轮执行，每一步都由只读的审计者独立验收；步骤必须都有验收标准。"
                          onClick={() => onExecuteLong(task)}
                        >
                          长任务执行
                        </button>
                      ) : null}
                    </div>
                  ) : null}
                  {onRestart && task.status !== 'pending' ? (
                    <div className="current-task-item__actions">
                      <button
                        type="button"
                        className="btn"
                        disabled={busy || lockedTaskIds.includes(task.id) || missingAcceptance}
                        onClick={() => onRestart(task)}
                      >
                        从阶段 1 重新执行
                      </button>
                      <small>创建新任务并逐步执行，保留旧记录；可能覆盖工作区同名文件。</small>
                    </div>
                  ) : null}
                  {task.steps.length > 0 ? (
                    <ol>
                      {task.steps.map((taskStep) => (
                        <li key={taskStep.id} className={`current-task-step current-task-step--${taskStep.status}`}>
                          <span>{stepSymbol(taskStep.status)}</span>
                          <div>
                            <strong>{taskStep.title}</strong>
                            {taskStep.status === 'superseded' ? (
                              <small>已被修订 {taskStep.superseded_by ?? '?'} 取代{taskStep.superseded_reason ? `：${taskStep.superseded_reason}` : ''}</small>
                            ) : taskStep.note ? <small>{taskStep.note}</small> : null}
                          </div>
                        </li>
                      ))}
                    </ol>
                  ) : <span className="empty-inline">该任务还没有拆分步骤。</span>}
                </div>
              </details>
            )
          })}
        </div>
      </details>
    </section>
  )
}
