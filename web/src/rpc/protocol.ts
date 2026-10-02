

export interface RpcRequestMessage {
  jsonrpc: '2.0'
  id: number
  method: string
  params?: Record<string, unknown>
}

export interface RpcResponseMessage {
  jsonrpc: '2.0'
  id: number
  result?: unknown
  error?: { code: number; message: string; data?: unknown }
}

export interface RpcNotificationMessage {
  jsonrpc: '2.0'
  method: string
  params?: unknown
}

export type RpcIncomingMessage = RpcResponseMessage | RpcNotificationMessage

/** 编码带请求 ID 的 JSON-RPC 调用。 */
export function encodeRequest(
  id: number,
  method: string,
  params?: Record<string, unknown>,
): string {
  return JSON.stringify({
    jsonrpc: '2.0',
    id,
    method,
    params: params ?? {},
  })
}

/** 解析并校验服务端响应或通知的协议字段。 */
export function parseMessage(text: string): RpcIncomingMessage {
  return JSON.parse(text) as RpcIncomingMessage
}
