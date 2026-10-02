

export const SERVER_URL: string =
  (import.meta.env.VITE_AGENT_SERVER_URL as string | undefined) ??
  'http://127.0.0.1:8000'

export const WS_URL: string = SERVER_URL.replace(/^http/, 'ws')


export const RPC_URL: string = `${WS_URL}/rpc`
