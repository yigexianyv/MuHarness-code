

import { rpcClient } from '../rpc'
import { RpcMethods } from '../rpc/methods'
import type { Task } from './types'

export async function listTasks(conversationId: string, limit = 20): Promise<Task[]> {
  const response = await rpcClient.call<{ tasks: Task[] }>(RpcMethods.taskList, {
    conversation_id: conversationId,
    limit,
  })
  return response.tasks
}

export async function getTask(taskId: string): Promise<Task> {
  const response = await rpcClient.call<{ task: Task }>(RpcMethods.taskGet, {
    task_id: taskId,
  })
  return response.task
}

export async function planAccept(taskId: string): Promise<Task> {
  const response = await rpcClient.call<{ task: Task }>(RpcMethods.taskPlanAccept, {
    task_id: taskId,
  })
  return response.task
}

export async function planReject(taskId: string): Promise<Task> {
  const response = await rpcClient.call<{ task: Task }>(RpcMethods.taskPlanReject, {
    task_id: taskId,
  })
  return response.task
}
