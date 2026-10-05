import { rpcClient } from '../rpc'
import { RpcMethods } from '../rpc/methods'
import type { MeaAmendResult, MeaDetail, MeaRun, MeaStatus, Task } from './types'

export interface StartMeaOptions {
  /** 创建全新任务，从第一个步骤执行，保留旧任务记录。 */
  restart?: boolean
  /** 轮次预算，默认 25。 */
  roundBudget?: number
  /** 额外授权给 Executor 的工具（只能是 mea.tools 列出的）。 */
  extraTools?: string[]
  /** 原始请求；不传时后端取生成计划的那条用户消息。 */
  originalRequest?: string
  /** 执行者在沙箱内的 shell 命令是否自动批准；后端默认 true。 */
  autoApproveSandbox?: boolean
}

/** 为计划启动长任务。计划还在待确认时，后端先检查能否启动，再接受计划。 */
export async function startMea(
  conversationId: string,
  taskId: string,
  options: StartMeaOptions = {},
): Promise<{ mea: MeaRun; task: Task }> {
  const params: Record<string, unknown> = {
    conversation_id: conversationId,
    task_id: taskId,
  }
  if (options.restart !== undefined) params.restart = options.restart
  if (options.roundBudget !== undefined) params.round_budget = options.roundBudget
  if (options.extraTools && options.extraTools.length > 0) params.extra_tools = options.extraTools
  if (options.originalRequest) params.original_request = options.originalRequest
  if (options.autoApproveSandbox !== undefined) params.auto_approve_sandbox = options.autoApproveSandbox
  return rpcClient.call(RpcMethods.meaStart, params)
}

export async function getMea(meaId: string): Promise<MeaDetail> {
  return rpcClient.call(RpcMethods.meaGet, { mea_id: meaId })
}

export async function listMeas(
  conversationId: string,
  options: { status?: MeaStatus; limit?: number } = {},
): Promise<MeaRun[]> {
  const params: Record<string, unknown> = {
    conversation_id: conversationId,
    limit: options.limit ?? 20,
  }
  if (options.status) params.status = options.status
  const response = await rpcClient.call<{ meas: MeaRun[] }>(RpcMethods.meaList, params)
  return response.meas
}

/** 所有会话的长任务（“待处理”和“账本”用），按创建时间倒序。 */
export async function listAllMeas(
  options: { status?: MeaStatus; limit?: number } = {},
): Promise<MeaRun[]> {
  const params: Record<string, unknown> = { limit: options.limit ?? 100 }
  if (options.status) params.status = options.status
  const response = await rpcClient.call<{ meas: MeaRun[] }>(RpcMethods.meaList, params)
  return response.meas
}

/** 启动时可以勾选授权给 Executor 的额外工具。 */
export async function listMeaTools(): Promise<string[]> {
  const response = await rpcClient.call<{ tools: string[] }>(RpcMethods.meaTools, {})
  return response.tools
}

export async function answerMea(meaId: string, text: string): Promise<MeaAmendResult> {
  return rpcClient.call(RpcMethods.meaAnswer, { mea_id: meaId, text })
}

export type MeaNoteKind = 'persistent' | 'once'

/**
 * 补充指令。persistent 生成新的要求版本，once 只进入下一轮 Manager；
 * immediate 会取消当前子 Run。长任务已收尾时返回 accepted=false、reason=mea_finalized。
 */
export async function noteMea(
  meaId: string,
  text: string,
  options: { kind?: MeaNoteKind; immediate?: boolean } = {},
): Promise<MeaAmendResult> {
  return rpcClient.call(RpcMethods.meaNote, {
    mea_id: meaId,
    text,
    kind: options.kind ?? 'persistent',
    immediate: options.immediate ?? false,
  })
}

export async function pauseMea(meaId: string): Promise<MeaRun> {
  const response = await rpcClient.call<{ mea: MeaRun }>(RpcMethods.meaPause, { mea_id: meaId })
  return response.mea
}

export async function resumeMea(meaId: string, extraRounds = 0): Promise<MeaRun> {
  const response = await rpcClient.call<{ mea: MeaRun }>(RpcMethods.meaResume, {
    mea_id: meaId,
    extra_rounds: extraRounds,
  })
  return response.mea
}

export async function cancelMea(meaId: string): Promise<MeaRun> {
  const response = await rpcClient.call<{ mea: MeaRun }>(RpcMethods.meaCancel, { mea_id: meaId })
  return response.mea
}
