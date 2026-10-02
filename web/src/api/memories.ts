

import { rpcClient } from '../rpc'
import { RpcMethods } from '../rpc/methods'
import type { LongTermMemoryOverview } from './types'

export async function listMemories(): Promise<LongTermMemoryOverview> {
  return rpcClient.call<LongTermMemoryOverview>(RpcMethods.memoryList, {})
}
