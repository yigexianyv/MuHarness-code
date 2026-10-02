import type { ReactElement } from 'react'

import type { RunStatus } from '../api/types'

const STATUS_LABELS = {
  pending: '准备中',
  running: '运行中',
  completed: '已完成',
  failed: '失败',
  cancelled: '已取消',
  interrupted: '已中断',
} satisfies Record<RunStatus, string>

const ORDER: RunStatus[] = Object.keys(STATUS_LABELS) as RunStatus[]

export default function RunBadge({ status }: { status: RunStatus }): ReactElement {
  return <span className={`badge badge-${status}`}>{STATUS_LABELS[status]}</span>
}

export { ORDER }
