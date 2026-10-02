

import type { AgentEvent, ApprovalRequest } from '../api/types'

export interface AgentEventNotificationParams {

  jsonrpc?: '2.0'
  method?: string
  params: AgentEvent
}

export interface RunStatusNotificationParams {
  jsonrpc?: '2.0'
  method?: string
  params: { run_id: string; status: string }
}

export interface ApprovalNotificationParams {
  jsonrpc?: '2.0'
  method?: string
  params: { approval: ApprovalRequest }
}
