






import type { AgentEvent, ModelUsage } from '../api/types'

export type ToolState = 'active' | 'done' | 'failed' | 'waiting'

export interface ToolStepVM {
  id: string
  name: string

  label: string
  state: ToolState

  details: string
  resultDetails?: string
  durationMs?: number
  errorCode?: string | null
  approval?: 'pending' | 'approved' | 'denied'
}

export interface UsageVM {
  inputTokens: number
  outputTokens: number
  totalTokens: number
  cachedInputTokens: number | null
  cacheHitRate: number | null
}

export interface TurnView {
  tools: ToolStepVM[]

  toolCount: number

  steps: number
  usage: UsageVM | null
  durationMs: number | null
  status:
    | 'thinking'
    | 'working'
    | 'waiting_approval'
    | 'verifying'
    | 'completed'
    | 'failed'
    | 'cancelled'
    | 'interrupted'
  finalText: string
  currentAction: string | null
  capability: 'Files' | 'Web' | 'Memory' | 'Task' | 'Artifact' | null
  error: { title: string; message: string; technical: string | null } | null
}


function argValue(args: unknown, key: string): string | null {
  if (!args) return null
  if (typeof args === 'string') {
    try {
      return argValue(JSON.parse(args), key)
    } catch {
      return null
    }
  }
  if (typeof args === 'object') {
    const value = (args as Record<string, unknown>)[key]
    return typeof value === 'string' && value ? value : null
  }
  return null
}


export function toolActiveLabel(name: string, args: unknown): string {
  switch (name) {
    case 'read_file': {
      const path = argValue(args, 'path')
      return path ? `读取 ${path}` : '读取文件'
    }
    case 'write_file': {
      const path = argValue(args, 'path')
      return path ? `写入 ${path}` : '写入文件'
    }
    case 'list_files': return '查看文件'
    case 'artifact_publish': return '准备结果'
    case 'memory_get': return '读取记忆'
    case 'memory_search': return '搜索记忆'
    case 'task_create': return '创建计划'
    case 'task_update': return '更新计划'
    case 'task_get': return '查看计划'
    case 'task_list': return '查看任务'
    case 'mcp_status': return '查看 MCP 工具'
    case 'run_shell_command': return '运行命令'
    case 'web_search': {
      const query = argValue(args, 'query')
      return query ? `搜索 “${query.slice(0, 40)}”` : '联网搜索'
    }
    default: return `运行 ${humanizeToolName(name)}`
  }
}

export function toolDoneLabel(name: string, _args: unknown, ok: boolean): string {
  if (!ok) {
    switch (name) {
      case 'read_file': return '无法读取文件'
      case 'write_file': return '无法写入文件'
      case 'run_shell_command': return '命令失败'
      default: return `失败 ${humanizeToolName(name)}`
    }
  }
  switch (name) {
    case 'read_file': return '已读取文件'
    case 'write_file': return '已写入文件'
    case 'list_files': return '已查看文件'
    case 'artifact_publish': return '已生成结果'
    case 'memory_get': return '已读取记忆'
    case 'memory_search': return '已搜索记忆'
    case 'task_create': return '已创建计划'
    case 'task_update': return '已更新计划'
    case 'task_get': return '已查看计划'
    case 'task_list': return '已查看任务'
    case 'mcp_status': return '已读取 MCP 工具清单'
    case 'run_shell_command': return '命令已执行'
    case 'web_search': return '已完成搜索'
    default: return `完成 ${humanizeToolName(name)}`
  }
}

export function humanizeToolName(name: string): string {
  return name.replace(/^mcp__[^_]+__/, '').replaceAll('_', ' ')
}

function detailsText(args: unknown): string {
  if (!args) return ''
  if (typeof args === 'string') return args
  try {
    return JSON.stringify(args, null, 2)
  } catch {
    return String(args)
  }
}

function capabilityForTool(name: string): TurnView['capability'] {
  if (name.includes('artifact')) return 'Artifact'
  if (name.startsWith('memory_')) return 'Memory'
  if (name.startsWith('task_')) return 'Task'
  if (name === 'web_search') return 'Web'
  if (name.includes('file') || name.includes('shell')) return 'Files'
  return null
}

/** 把运行错误转换成适合界面展示的文案，同时保留技术详情。 */
export function humanizeRunError(
  stopReason: string | null,
  rawError: string | null = null,
): { title: string; message: string; technical: string | null } {
  const known: Record<string, { title: string; message: string }> = {
    max_steps: {
      title: '本轮执行已暂停',
      message: '执行步骤已达到上限，任务可能尚未完全完成。你可以继续发送消息，让 MuHarness 接着处理。',
    },
    repeated_tool_call: {
      title: '执行遇到循环',
      message: 'MuHarness 连续尝试了相同操作但没有取得进展。请补充信息，或换一种方式继续。',
    },
    permission_denied: {
      title: '操作未获允许',
      message: '这项操作没有执行。你可以调整要求后重新尝试。',
    },
    model_error: {
      title: '模型暂时无法响应',
      message: '模型服务没有完成本轮请求。请稍后重试；已完成的工具结果不会因此被伪装成成功。',
    },
    context_error: {
      title: '上下文整理失败',
      message: '当前内容超过了本次请求可安全处理的范围。可以新建会话，或缩小本次任务范围。',
    },
    run_budget: {
      title: '本轮用量已达上限',
      message: 'MuHarness 已停止继续消耗模型用量。已完成的结果仍会保留，你可以在下一条消息中继续。',
    },
    interrupted: {
      title: '执行已中断',
      message: '本轮已安全停止，可以从恢复点继续执行。',
    },
    cancelled: {
      title: '执行已取消',
      message: '本轮操作已经取消，没有继续执行后续动作。',
    },
  }
  const presentation = known[stopReason ?? ''] ?? {
    title: '本轮未能完成',
    message: 'MuHarness 已停止本轮执行。你可以查看技术详情，或调整要求后重试。',
  }
  return {
    ...presentation,
    technical: rawError || stopReason || null,
  }
}

/** 将原始 Agent 事件归并为单次对话的工具、用量与最终答复视图。 */
export function buildTurnView(
  events: AgentEvent[],
  opts: { now?: number } = {},
): TurnView {
  const tools: ToolStepVM[] = []
  const toolIndexes = new Map<string, number>()
  let steps = 0
  const usageParts: ModelUsage[] = []
  let finalUsage: ModelUsage | null = null
  let startedAt: number | null = null
  let endedAt: number | null = null
  let finalText = ''
  let capability: TurnView['capability'] = null
  let stopReason: string | null = null
  let rawError: string | null = null
  const modelSteps = new Set<number>()

  const upsertTool = (
    id: string,
    name: string,
    args: unknown,
    state: ToolState,
  ): number => {
    const existing = toolIndexes.get(id)
    if (existing === undefined) {
      toolIndexes.set(id, tools.length)
      tools.push({
        id,
        name,
        label: toolActiveLabel(name, args),
        state,
        details: detailsText(args),
      })
      return tools.length - 1
    }
    tools[existing] = { ...tools[existing], name, state }
    return existing
  }

  for (const event of events) {
    if (event.event_time) {
      const t = Date.parse(event.event_time)
      if (!Number.isNaN(t)) {
        if (event.type === 'agent_started') startedAt = t
        if (
          event.type === 'agent_completed' ||
          event.type === 'agent_failed' ||
          event.type === 'agent_cancelled'
        ) {
          endedAt = t
        }
      }
    }

    switch (event.type) {
      case 'model_started': {
        if (event.step !== null && event.step !== undefined) modelSteps.add(event.step)
        break
      }
      case 'model_completed': {
        if (event.step !== null && event.step !== undefined) modelSteps.add(event.step)
        if (event.usage) usageParts.push(event.usage)
        break
      }
      case 'tool_started': {
        if (event.tool_call) {
          const name = event.tool_call.name
          capability = capabilityForTool(name) ?? capability
          const idx = upsertTool(
            event.tool_call.id,
            name,
            event.tool_call.arguments,
            'active',
          )
          tools[idx].label = toolActiveLabel(name, event.tool_call.arguments)
        }
        break
      }
      case 'tool_approval_required': {
        if (event.tool_call) {
          const idx = upsertTool(
            event.tool_call.id,
            event.tool_call.name,
            event.tool_call.arguments,
            'waiting',
          )
          tools[idx].approval = 'pending'
          tools[idx].state = 'waiting'
        }
        break
      }
      case 'tool_approval_completed': {
        if (event.tool_call && event.approval_decision) {
          const idx = toolIndexes.get(event.tool_call.id)
          if (idx !== undefined) {
            const decision = event.approval_decision === 'approved' ? 'approved' : 'denied'
            tools[idx].approval = decision

            tools[idx].state = decision === 'approved' ? 'active' : 'failed'
          }
        }
        break
      }
      case 'tool_completed': {
        if (event.tool_result) {
          const idx = toolIndexes.get(event.tool_result.tool_call_id)
          const name = event.tool_result.tool_name
          const ok = event.tool_result.success
          if (idx !== undefined) {
            tools[idx].name = name
            tools[idx].state = ok ? 'done' : 'failed'
            tools[idx].label = toolDoneLabel(name, tools[idx].details, ok)
            tools[idx].durationMs = event.tool_result.duration_ms
            tools[idx].resultDetails = event.tool_result.output ?? undefined
            tools[idx].errorCode = event.tool_result.error
          }
        }
        break
      }
      case 'agent_completed':
      case 'agent_failed':
      case 'agent_cancelled': {
        finalUsage = event.result?.usage ?? event.usage ?? finalUsage
        if (event.result?.steps) steps = event.result.steps
        finalText = event.result?.final_message?.content ?? finalText
        stopReason = event.stop_reason ?? event.result?.stop_reason ?? stopReason
        rawError = event.result?.error?.message ?? rawError
        break
      }
      default:
        break
    }
  }


  let usage: UsageVM | null = null
  if (finalUsage) {
    const cachedInputTokens = finalUsage.cached_input_tokens ?? null
    usage = {
      inputTokens: finalUsage.input_tokens,
      outputTokens: finalUsage.output_tokens,
      totalTokens: finalUsage.total_tokens,
      cachedInputTokens,
      cacheHitRate: cacheHitRate(
        cachedInputTokens,
        finalUsage.input_tokens,
      ),
    }
  } else if (usageParts.length > 0) {
    const cachedInputTokens = sumOptionalUsage(
      usageParts,
      'cached_input_tokens',
    )
    const inputTokens = usageParts.reduce((sum, u) => sum + u.input_tokens, 0)
    usage = {
      inputTokens,
      outputTokens: usageParts.reduce((sum, u) => sum + u.output_tokens, 0),
      totalTokens: usageParts.reduce((sum, u) => sum + u.total_tokens, 0),
      cachedInputTokens,
      cacheHitRate: cacheHitRate(cachedInputTokens, inputTokens),
    }
  }


  const startedIds = new Set<string>()
  for (const event of events) {
    if (event.type === 'tool_started' && event.tool_call) startedIds.add(event.tool_call.id)
  }
  const toolCount = startedIds.size

  if (steps === 0) steps = modelSteps.size

  let durationMs: number | null = null
  if (startedAt !== null) {
    const end = endedAt ?? opts.now ?? Date.now()
    durationMs = Math.max(0, end - startedAt)
  }

  const hasPendingApproval = tools.some(
    (tool) => tool.approval === 'pending' && tool.state === 'waiting',
  )
  const hasActiveTool = tools.some((tool) => tool.state === 'active')
  const status: TurnView['status'] = events.some((e) => e.type === 'agent_failed')
    ? stopReason === 'interrupted' ? 'interrupted' : 'failed'
    : events.some((e) => e.type === 'agent_cancelled')
      ? 'cancelled'
      : events.some((e) => e.type === 'agent_completed')
        ? 'completed'
        : hasPendingApproval
          ? 'waiting_approval'
          : hasActiveTool
              ? 'working'
              : 'thinking'

  const currentTool = [...tools].reverse().find(
    (tool) => tool.state === 'active' || tool.state === 'waiting',
  )

  return {
    tools,
    toolCount,
    steps,
    usage,
    durationMs,
    status,
    finalText,
    currentAction: currentTool?.label ?? null,
    capability,
    error:
      status === 'failed' || status === 'interrupted'
        ? humanizeRunError(stopReason, rawError)
        : null,
  }
}

function sumOptionalUsage(
  usages: ModelUsage[],
  field: 'cached_input_tokens',
): number | null {
  if (usages.some((usage) => usage[field] === null || usage[field] === undefined)) {
    return null
  }
  return usages.reduce((sum, usage) => sum + (usage[field] ?? 0), 0)
}

function cacheHitRate(cachedInputTokens: number | null, inputTokens: number): number | null {
  if (cachedInputTokens === null || inputTokens <= 0) return null
  return Math.min(100, Math.max(0, (cachedInputTokens / inputTokens) * 100))
}


export function formatCacheHitRate(rate: number | null): string {
  if (rate === null) return '暂无'
  const rounded = Math.round(rate * 10) / 10
  return `${Number.isInteger(rounded) ? rounded.toFixed(0) : rounded.toFixed(1)}%`
}


export function formatTokens(n: number): string {
  if (n < 1000) return String(Math.round(n))
  return `${(n / 1000).toFixed(1).replace(/\.0$/, '')}k`
}


export function formatDuration(ms: number | null): string {
  if (ms === null) return ''
  if (ms < 1000) return `${Math.round(ms)}ms`
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`
  const m = Math.floor(ms / 60_000)
  const s = Math.round((ms % 60_000) / 1000)
  return `${m}m ${String(s).padStart(2, '0')}s`
}
