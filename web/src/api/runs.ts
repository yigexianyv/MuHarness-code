

import { rpcClient } from '../rpc'
import { RpcMethods } from '../rpc/methods'
import type {
  AgentEvent,
  AgentRunTrace,
  AgentResult,
  Run,
  RunUsageSummary,
} from './types'

export interface RunListQuery {
  conversationId?: string
  status?: string
  limit?: number
}

export async function listRuns(query: RunListQuery = {}): Promise<Run[]> {
  const params: Record<string, unknown> = {}
  if (query.conversationId) params.conversation_id = query.conversationId
  if (query.status) params.status = query.status
  params.limit = query.limit ?? 50
  const response = await rpcClient.call<{ runs: Run[] }>(RpcMethods.runList, params)
  return response.runs
}

export async function getRun(runId: string): Promise<Run> {
  const response = await rpcClient.call<{ run: Run }>(RpcMethods.runGet, {
    run_id: runId,
  })
  return response.run
}

export async function cancelRun(runId: string): Promise<Run> {
  const response = await rpcClient.call<{ run: Run }>(RpcMethods.runCancel, {
    run_id: runId,
  })
  return response.run
}


export async function interruptRun(runId: string): Promise<Run> {
  const response = await rpcClient.call<{ run: Run }>(RpcMethods.runInterrupt, {
    run_id: runId,
  })
  return response.run
}

export async function recoverRun(
  runId: string,
): Promise<{ recovered_from_run_id: string; run: Run; result: AgentResult | null }> {
  return rpcClient.call(RpcMethods.runRecover, { run_id: runId })
}

export async function getRunTrace(
  runId: string,
): Promise<{ run: AgentRunTrace; events: AgentEvent[]; usage: RunUsageSummary }> {
  return rpcClient.call(RpcMethods.traceGet, { run_id: runId })
}
