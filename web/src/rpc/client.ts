








import { RpcError, RpcErrorCode } from './errors'
import { encodeRequest, parseMessage } from './protocol'

export type NotificationHandler = (params: unknown) => void
export type StatusListener = (connected: boolean) => void

export interface RpcClientOptions {
  url: string
  reconnectDelayMs?: number
  requestTimeoutMs?: number
  socketFactory?: () => WebSocket
}

export interface RpcCallOptions {

  timeoutMs?: number
}

interface PendingRequest {
  resolve: (value: unknown) => void
  reject: (error: RpcError) => void
  timer: ReturnType<typeof setTimeout> | null
}

interface OpenWaiter {
  resolve: () => void
  reject: (error: RpcError) => void
  timer: ReturnType<typeof setTimeout>
}

const WS_OPEN = 1

export class RpcClient {
  private socket: WebSocket | null = null
  private nextId = 1
  private pending = new Map<number, PendingRequest>()
  private handlers = new Map<string, Set<NotificationHandler>>()
  private statusListeners = new Set<StatusListener>()
  private openWaiters: OpenWaiter[] = []
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private shouldReconnect = false
  private connectPromise: Promise<void> | null = null

  private connectToken = 0

  private readonly url: string
  private readonly reconnectDelayMs: number
  private readonly requestTimeoutMs: number
  private readonly openTimeoutMs: number
  private readonly socketFactory: () => WebSocket

  constructor(options: RpcClientOptions) {
    this.url = options.url
    this.reconnectDelayMs = options.reconnectDelayMs ?? 2000
    this.requestTimeoutMs = options.requestTimeoutMs ?? 60_000
    this.openTimeoutMs = 8000
    this.socketFactory = options.socketFactory ?? (() => new WebSocket(this.url))
  }

  get connected(): boolean {
    return this.socket !== null && this.socket.readyState === WS_OPEN
  }





  /** 启动 WebSocket 连接；断线后按配置延迟重连。 */
  connect(): void {
    this.shouldReconnect = true
    if (this.socket && this.socket.readyState !== 3) {
      return
    }
    if (this.connectPromise) return
    void this.openSocket()
  }

  /** 主动断开连接，并拒绝所有未完成的 RPC 请求。 */
  disconnect(): void {
    this.shouldReconnect = false
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }


    this.connectPromise = null
    const socket = this.socket
    this.socket = null
    if (socket) {
      socket.onclose = null
      socket.onmessage = null
      socket.close()
    }
    this.rejectAllPending(
      new RpcError(RpcErrorCode.NotConnected, 'client disconnected'),
    )
    this.rejectOpenWaiters(
      new RpcError(RpcErrorCode.NotConnected, 'client disconnected'),
    )
    this.emitStatus(false)
  }

  /** 建立连接；连接令牌防止旧连接的回调覆盖新连接状态。 */
  private async openSocket(): Promise<void> {
    const token = ++this.connectToken
    const socket = this.socketFactory()
    this.socket = socket

    socket.onmessage = (event: MessageEvent<string>) => this.handleMessage(event.data)
    socket.onclose = () => this.handleClose(socket)
    const opener = new Promise<void>((resolve, reject) => {
      socket.onopen = () => resolve()
      socket.onerror = () =>
        reject(new RpcError(RpcErrorCode.NotConnected, 'websocket error'))
    })


    this.connectPromise = (async () => {
      try {
        await opener
        if (token !== this.connectToken) return
        this.flushOpenWaiters()
        this.emitStatus(true)
      } catch {
        if (token === this.connectToken) {
          this.handleClose(socket)
        }
      } finally {
        if (token === this.connectToken) {
          this.connectPromise = null
        }
      }
    })()
  }

  /** 清理已关闭连接，并在允许重连时安排下一次尝试。 */
  private handleClose(socket: WebSocket): void {
    if (this.socket !== socket) return
    this.socket = null
    this.rejectAllPending(
      new RpcError(RpcErrorCode.NotConnected, 'connection closed'),
    )
    this.rejectOpenWaiters(
      new RpcError(RpcErrorCode.NotConnected, 'connection closed'),
    )
    this.emitStatus(false)
    if (this.shouldReconnect && this.reconnectTimer === null) {

      const delay = this.reconnectDelayMs + Math.floor(Math.random() * 1000)
      this.reconnectTimer = setTimeout(() => {
        this.reconnectTimer = null
        this.connect()
      }, delay)
    }
  }





  /** 发送 RPC 请求并等待响应；连接尚未打开时先等待握手完成。 */
  async call<T>(
    method: string,
    params?: Record<string, unknown>,
    options?: RpcCallOptions,
  ): Promise<T> {
    if (!this.socket) {
      throw new RpcError(RpcErrorCode.NotConnected, 'not connected')
    }
    if (this.socket.readyState !== WS_OPEN) {
      await this.waitForOpen()
    }
    return this.sendRequest<T>(method, params, options?.timeoutMs)
  }

  private waitForOpen(): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.openWaiters = this.openWaiters.filter((waiter) => waiter.timer !== timer)
        reject(
          new RpcError(RpcErrorCode.NotConnected, 'connection open timeout'),
        )
      }, this.openTimeoutMs)
      this.openWaiters.push({
        resolve: () => {
          clearTimeout(timer)
          resolve()
        },
        reject: (error) => {
          clearTimeout(timer)
          reject(error)
        },
        timer,
      })
    })
  }

  /** 分配请求 ID，登记超时处理，再发送 JSON-RPC 消息。 */
  private sendRequest<T>(
    method: string,
    params?: Record<string, unknown>,
    timeoutMs?: number,
  ): Promise<T> {
    const id = this.nextId++
    const socket = this.socket
    if (!socket) {
      return Promise.reject(new RpcError(RpcErrorCode.NotConnected, 'not connected'))
    }
    return new Promise<T>((resolve, reject) => {

      const effectiveTimeout = timeoutMs ?? this.requestTimeoutMs
      let timer: ReturnType<typeof setTimeout> | null = null
      if (effectiveTimeout !== 0) {
        timer = setTimeout(() => {
          this.pending.delete(id)
          reject(new RpcError(RpcErrorCode.RequestTimeout, `request timeout: ${method}`))
        }, effectiveTimeout)
      }
      this.pending.set(id, {
        resolve: resolve as (value: unknown) => void,
        reject,
        timer,
      })
      socket.send(encodeRequest(id, method, params))
    })
  }

  /** 按请求 ID 完成响应，或将服务端通知分发给订阅者。 */
  private handleMessage(text: string): void {
    let message: {
      id?: number
      method?: string
      params?: unknown
      result?: unknown
      error?: { code: number; message: string; data?: unknown }
    }
    try {
      message = parseMessage(text)
    } catch {
      return
    }

    if (message.id !== undefined) {
      const pending = this.pending.get(message.id)
      if (!pending) return
      if (pending.timer !== null) clearTimeout(pending.timer)
      this.pending.delete(message.id)
      if (message.error !== undefined) {
        pending.reject(
          new RpcError(message.error.code, message.error.message, message.error.data),
        )
      } else {
        pending.resolve(message.result)
      }
      return
    }

    if (message.method !== undefined) {
      const handlers = this.handlers.get(message.method)
      if (handlers) {
        for (const handler of [...handlers]) {
          try {
            handler(message.params)
          } catch {

          }
        }
      }
    }
  }





  /** 订阅通知并返回对应的取消订阅函数。 */
  on(method: string, handler: NotificationHandler): () => void {
    let set = this.handlers.get(method)
    if (!set) {
      set = new Set()
      this.handlers.set(method, set)
    }
    set.add(handler)
    return () => this.off(method, handler)
  }

  off(method: string, handler: NotificationHandler): void {
    const set = this.handlers.get(method)
    if (!set) return
    set.delete(handler)
    if (set.size === 0) this.handlers.delete(method)
  }

  setStatusListener(listener: StatusListener): () => void {
    this.statusListeners.add(listener)
    return () => this.statusListeners.delete(listener)
  }





  private flushOpenWaiters(): void {
    const waiters = this.openWaiters
    this.openWaiters = []
    for (const waiter of waiters) {
      clearTimeout(waiter.timer)
      waiter.resolve()
    }
  }

  private rejectOpenWaiters(error: RpcError): void {
    const waiters = this.openWaiters
    this.openWaiters = []
    for (const waiter of waiters) {
      clearTimeout(waiter.timer)
      waiter.reject(error)
    }
  }

  private rejectAllPending(error: RpcError): void {
    const entries = [...this.pending.values()]
    this.pending.clear()
    for (const entry of entries) {
      if (entry.timer !== null) clearTimeout(entry.timer)
      entry.reject(error)
    }
  }

  private emitStatus(connected: boolean): void {
    for (const listener of this.statusListeners) {
      try {
        listener(connected)
      } catch {

      }
    }
  }
}
