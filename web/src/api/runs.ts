

import { rpcClient } from '../rpc'
import { RpcMethods } from '../rpc/methods'
import type {
  AgentEvent,
  AgentRunTrace,
  AgentResult,
  Run,
  RequestToolViewDetail,
  RewindPreview,
  RewindResult,
  RewindStepInfo,
  RunContextMessagesPage,
  RunUsageSummary,
  ToolEvidencePage,
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

/** 分页读取运行看过的原始消息（继承的会话历史 + 本次运行新增的消息）。 */
export async function getRunContextMessages(
  runId: string,
  offset: number,
  limit: number,
): Promise<RunContextMessagesPage> {
  return rpcClient.call(RpcMethods.runContextMessages, {
    run_id: runId,
    offset,
    limit,
  })
}

export async function listRunSteps(runId: string): Promise<RewindStepInfo[]> {
  const response = await rpcClient.call<{ steps: RewindStepInfo[] }>(
    RpcMethods.runStepsList,
    { run_id: runId },
  )
  return response.steps
}

/** 只读：回到第 step 步之前会恢复、删除哪些文件，哪些操作不会回退。 */
export async function previewRewind(runId: string, step: number): Promise<RewindPreview> {
  const response = await rpcClient.call<{ preview: RewindPreview }>(
    RpcMethods.runRewindPreview,
    { run_id: runId, step },
  )
  return response.preview
}

/** rewindKey 在一次确认中保持不变，重复提交只会生效一次。 */
export async function applyRewind(input: {
  runId: string
  step: number
  previewId: string
  correction: string
  rewindKey: string
}): Promise<RewindResult> {
  const response = await rpcClient.call<{ rewind: RewindResult }>(
    RpcMethods.runRewindApply,
    {
      run_id: input.runId,
      step: input.step,
      preview_id: input.previewId,
      correction: input.correction,
      rewind_key: input.rewindKey,
    },
  )
  return response.rewind
}

export async function undoRewind(rewindKey: string): Promise<RewindResult> {
  const response = await rpcClient.call<{ rewind: RewindResult }>(
    RpcMethods.runRewindUndo,
    { rewind_key: rewindKey },
  )
  return response.rewind
}

/** 分页读取工具调用的完整原文（模型收到的可能是截短版本）。 */
export async function getRunToolEvidence(
  runId: string,
  toolCallId: string,
  offset: number,
): Promise<ToolEvidencePage> {
  return rpcClient.call(RpcMethods.runContextEvidence, {
    run_id: runId,
    tool_call_id: toolCallId,
    offset,
  })
}

/** 第 step 步请求里模型实际收到的这次工具输出（逐字）。 */
export async function getRunToolView(
  runId: string,
  step: number,
  toolCallId: string,
): Promise<RequestToolViewDetail> {
  return rpcClient.call(RpcMethods.runContextToolView, {
    run_id: runId,
    step,
    tool_call_id: toolCallId,
  })
}
