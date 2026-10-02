

import { rpcClient } from '../rpc'
import { RpcMethods } from '../rpc/methods'
import type { ApprovalRequest, ApprovalStatus } from './types'

export async function listApprovals(
  status?: ApprovalStatus,
  limit = 50,
): Promise<ApprovalRequest[]> {
  const response = await rpcClient.call<{ approvals: ApprovalRequest[] }>(
    RpcMethods.approvalList,
    status ? { status, limit } : { limit },
  )
  return response.approvals
}

export async function getApproval(approvalId: string): Promise<ApprovalRequest> {
  const response = await rpcClient.call<{ approval: ApprovalRequest }>(
    RpcMethods.approvalGet,
    { approval_id: approvalId },
  )
  return response.approval
}

export async function approveApproval(approvalId: string): Promise<ApprovalRequest> {
  const response = await rpcClient.call<{ approval: ApprovalRequest }>(
    RpcMethods.approvalApprove,
    { approval_id: approvalId },
  )
  return response.approval
}

export async function denyApproval(approvalId: string): Promise<ApprovalRequest> {
  const response = await rpcClient.call<{ approval: ApprovalRequest }>(
    RpcMethods.approvalDeny,
    { approval_id: approvalId },
  )
  return response.approval
}
