

import { WS_URL } from '../api/config'
import { RpcClient } from './client'


export const rpcClient = new RpcClient({
  url: `${WS_URL}/rpc`,
})

export * from './client'
export * from './errors'
export * from './protocol'
export type {
  AgentEventNotificationParams,
  ApprovalNotificationParams,
  RunStatusNotificationParams,
} from './notifications'
