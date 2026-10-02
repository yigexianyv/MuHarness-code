

import { rpcClient } from '../rpc'
import { RpcMethods } from '../rpc/methods'
import type { Automation, AutomationKind } from './types'

export interface CreateAutomationInput {
  title: string
  prompt: string
  kind: AutomationKind
  run_at?: string
  interval_seconds?: number
  cron_expr?: string
  timezone?: string
  conversation_id?: string
}

export async function listAutomations(): Promise<Automation[]> {
  const response = await rpcClient.call<{ automations: Automation[] }>(
    RpcMethods.automationList,
    {},
  )
  return response.automations
}

export async function getAutomation(automationId: string): Promise<Automation> {
  const response = await rpcClient.call<{ automation: Automation }>(
    RpcMethods.automationGet,
    { automation_id: automationId },
  )
  return response.automation
}

export async function createAutomation(
  input: CreateAutomationInput,
): Promise<Automation> {
  const response = await rpcClient.call<{ automation: Automation }>(
    RpcMethods.automationCreate,
    { ...input },
  )
  return response.automation
}

export async function pauseAutomation(automationId: string): Promise<Automation> {
  const response = await rpcClient.call<{ automation: Automation }>(
    RpcMethods.automationPause,
    { automation_id: automationId },
  )
  return response.automation
}

export async function resumeAutomation(automationId: string): Promise<Automation> {
  const response = await rpcClient.call<{ automation: Automation }>(
    RpcMethods.automationResume,
    { automation_id: automationId },
  )
  return response.automation
}

export async function cancelAutomation(automationId: string): Promise<Automation> {
  const response = await rpcClient.call<{ automation: Automation }>(
    RpcMethods.automationCancel,
    { automation_id: automationId },
  )
  return response.automation
}
