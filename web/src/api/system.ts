

import { rpcClient } from '../rpc'
import { RpcMethods } from '../rpc/methods'

export interface SystemInfo {
  status: string
  provider: string
  model: string
  version: string
  database: string
}

export async function getSystemInfo(): Promise<SystemInfo> {
  return rpcClient.call(RpcMethods.systemInfo, {})
}
