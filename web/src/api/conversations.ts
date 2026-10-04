

import { rpcClient } from '../rpc'
import { RpcMethods } from '../rpc/methods'
import type {
  AgentMode,
  Conversation,
  ConversationConstraints,
  Message,
  SendMessageResponse,
} from './types'

export async function listConversations(limit = 50): Promise<Conversation[]> {
  const response = await rpcClient.call<{ conversations: Conversation[] }>(
    RpcMethods.conversationList,
    { limit },
  )
  return response.conversations
}

export async function getConversation(
  conversationId: string,
): Promise<{ conversation: Conversation; messages: Message[] }> {
  return rpcClient.call(RpcMethods.conversationGet, {
    conversation_id: conversationId,
  })
}

export async function createConversation(): Promise<Conversation> {
  const response = await rpcClient.call<{ conversation: Conversation }>(
    RpcMethods.conversationCreate,
    {},
  )
  return response.conversation
}

export async function renameConversation(
  conversationId: string,
  title: string,
): Promise<Conversation> {
  const response = await rpcClient.call<{ conversation: Conversation }>(
    RpcMethods.conversationRename,
    { conversation_id: conversationId, title },
  )
  return response.conversation
}

export async function deleteConversation(conversationId: string): Promise<boolean> {
  const response = await rpcClient.call<{ deleted: boolean }>(
    RpcMethods.conversationDelete,
    { conversation_id: conversationId },
  )
  return response.deleted
}

export async function sendMessage(
  conversationId: string,
  content: string,
  mode: AgentMode = 'normal',
): Promise<SendMessageResponse> {


  return rpcClient.call<SendMessageResponse>(
    RpcMethods.conversationSend,
    {
      conversation_id: conversationId,
      content,
      mode,
    },
    { timeoutMs: 0 },
  )
}

export async function getConversationConstraints(
  conversationId: string,
): Promise<ConversationConstraints> {
  const response = await rpcClient.call<{ constraints: ConversationConstraints }>(
    RpcMethods.conversationConstraintsGet,
    { conversation_id: conversationId },
  )
  return response.constraints
}

/** 保存新版本；expectedRevision 用于发现别处的并发修改。 */
export async function setConversationConstraints(
  conversationId: string,
  text: string,
  expectedRevision: number,
): Promise<ConversationConstraints> {
  const response = await rpcClient.call<{ constraints: ConversationConstraints }>(
    RpcMethods.conversationConstraintsSet,
    { conversation_id: conversationId, text, expected_revision: expectedRevision },
  )
  return response.constraints
}
