import { rpcClient } from '../rpc'
import { RpcMethods } from '../rpc/methods'

export type WorkspaceSetupStatus = 'running' | 'succeeded' | 'failed' | 'denied' | 'cancelled'

export interface WorkspaceSetupCheck {
  /** null 表示跳过 */
  passed: boolean | null
  detail: string
}

export interface WorkspaceSetupRecord {
  id: string
  status: WorkspaceSetupStatus
  script_path: string
  script_sha256: string
  verify_command: string | null
  started_at: string
  finished_at: string | null
  exit_code: number | null
  timed_out: boolean
  log_path: string | null
  output_head: string
  output_tail: string
  network_check: WorkspaceSetupCheck | null
  environment_check: WorkspaceSetupCheck | null
  error: string | null
}

export interface WorkspaceSetupState {
  current: WorkspaceSetupRecord | null
  last: WorkspaceSetupRecord | null
}

export async function getWorkspaceSetup(): Promise<WorkspaceSetupState> {
  return rpcClient.call(RpcMethods.workspaceSetupStatus, {})
}

export async function startWorkspaceSetup(
  scriptPath: string,
  verifyCommand?: string,
): Promise<WorkspaceSetupRecord> {
  const response = await rpcClient.call<{ setup: WorkspaceSetupRecord }>(
    RpcMethods.workspaceSetupStart,
    verifyCommand
      ? { script_path: scriptPath, verify_command: verifyCommand }
      : { script_path: scriptPath },
  )
  return response.setup
}

export async function cancelWorkspaceSetup(): Promise<boolean> {
  const response = await rpcClient.call<{ cancelled: boolean }>(
    RpcMethods.workspaceSetupCancel,
    {},
  )
  return response.cancelled
}
