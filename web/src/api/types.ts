

export interface Conversation {
  id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

export type MessageRole = 'system' | 'user' | 'assistant' | 'tool'

export interface ToolCall {
  id: string
  name: string
  arguments: Record<string, unknown> | string
}

export interface Message {
  role: MessageRole
  content: string | null
  name?: string | null
  tool_call_id?: string | null
  tool_calls?: ToolCall[]

  reasoning?: string | null
}

export type RunStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'interrupted'

// manage / execute / audit 只出现在长任务的子 Run 上，前端不会用它们发送消息
export type AgentMode = 'normal' | 'plan' | 'manage' | 'execute' | 'audit'

export interface Run {
  id: string
  conversation_id: string | null
  status: RunStatus
  user_message: string
  created_at: string
  started_at: string | null
  updated_at: string
  completed_at: string | null
  error: string | null
  stop_reason: string | null
  recovered_from_run_id: string | null
  source: string | null
  source_id: string | null
  scheduled_for: string | null
  triggered_at: string | null
  mode: AgentMode
}

export interface ModelUsage {
  input_tokens: number
  output_tokens: number
  total_tokens: number
  cached_input_tokens?: number | null
  uncached_input_tokens?: number | null
  cache_read_input_tokens?: number | null
  cache_write_input_tokens?: number | null
  model_calls?: number
}

export interface RunUsageSummary {
  main_agent: ModelUsage
  context_summary: ModelUsage
  memory_reflection: ModelUsage
  memory_maintenance: ModelUsage
  provider_total: ModelUsage
  tool_schema_tokens_estimated: number
  memory_reflection_status: string
  memory_reflection_skip_reason: string | null
  context_summary_status: string
  context_summary_provider: string | null
  context_summary_model: string | null
  context_summary_duration_ms: number
  main_agent_chargeable_tokens: number
  run_budget_status: string
  run_budget_reason: string | null
  run_budget_warning_tokens: number | null
  run_budget_finalization_tokens: number | null
  run_budget_hard_tokens: number | null
  run_budget_warning_model_calls: number | null
  run_budget_finalization_model_calls: number | null
  run_budget_hard_model_calls: number | null
}

export interface LongTermMemory {
  id: string
  title: string
  summary: string
  content: string
  created_at: string
  updated_at: string
  last_accessed_at: string
  access_count: number
  revision: number
  status: 'active' | 'archived'
  last_update_reason: string | null
  archive_reason: string | null
}

export interface LongTermMemoryOverview {
  core: string
  active: LongTermMemory[]
  archived: LongTermMemory[]
  active_count: number
  max_active: number
}

export interface AgentResult {
  run_id: string
  final_message: Message
  messages: Message[]
  steps: number
  stop_reason: string
  usage: ModelUsage
  error: { type: string; message: string } | null
  plan_task_id: string | null
}

export interface SendMessageResponse {
  conversation_id: string
  content: string | null
  run: Run
  result: AgentResult
  plan_task_id: string | null
}

export type TaskStatus =
  | 'pending'
  | 'active'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled'

export type TaskStepStatus = 'todo' | 'in_progress' | 'done' | 'blocked' | 'superseded'

export interface TaskStep {
  id: string
  title: string
  status: TaskStepStatus
  note: string | null
  /** 步骤验收标准；长任务按它审计每个步骤。 */
  acceptance?: string | null
  origin_requirements_revision?: number
  /** 被哪条用户修订取代（例如 A1）；只在 superseded 时有值。 */
  superseded_by?: string | null
  superseded_reason?: string | null
}

export interface Task {
  id: string
  title: string
  description: string | null
  goal: string | null
  status: TaskStatus
  priority: string
  constraints: string[]
  state: string[]
  key_facts: string[]
  steps: TaskStep[]
  /** 长任务 Manager 维护的任务契约全文。 */
  contract?: string | null
  owner_conversation_id: string
  run_ids: string[]
  created_at: string
  updated_at: string
  completed_at: string | null
  revision: number
}

export interface AgentRunTrace {
  run_id: string
  conversation_id: string | null
  status: string
  started_at: string
  completed_at: string | null
  provider: string | null
  model: string | null
  steps: number
  stop_reason: string | null
  input_tokens: number
  output_tokens: number
  total_tokens: number
  event_count: number
}

export interface AgentEvent {
  event_id: string
  run_id: string
  conversation_id: string | null
  sequence: number
  type: string
  event_time: string
  step: number | null
  provider: string | null
  model: string | null
  delta?: string | null

  reasoning_delta?: string | null
  message: Message | null
  tool_call: ToolCall | null
  tool_result: {
    tool_call_id: string
    tool_name: string
    success: boolean
    output: string | null
    error: string | null
    duration_ms: number
    approval_wait_ms?: number | null
    execution_duration_ms?: number | null
    /** 完整输出存在证据库里的编号 */
    evidence_id?: string | null
    /** 输出过长，模型收到的是截短版本 */
    output_truncated?: boolean | null
  } | null
  usage: ModelUsage | null
  stop_reason: string | null
  approval_decision: string | null

  result?: AgentResult | null
  original_estimated_input_tokens?: number | null
  prepared_input_tokens?: number | null
  estimated_input_tokens?: number | null
  context_trimmed?: boolean | null
  context_window?: number | null
  input_budget?: number | null
  working_input_budget?: number | null
  trigger_tokens?: number | null
  target_tokens?: number | null
  usage_ratio?: number | null
  tool_result_budget_tokens?: number | null
  tool_result_tokens_before?: number | null
  tool_result_tokens_after?: number | null
  tool_schema_tokens?: number | null
  message_tokens_before?: number | null
  message_tokens_after?: number | null
  original_usage_ratio?: number | null
  prepared_usage_ratio?: number | null
  compaction_stage?: string | null
  compacted_tool_results?: number | null
  removed_tool_rounds?: number | null
  reached_target?: boolean | null
  summary_updated?: boolean | null
  summarized_conversation_blocks?: number | null
  summary_error?: string | null
  summary_usage?: ModelUsage | null
  summary_provider?: string | null
  summary_model?: string | null
  summary_duration_ms?: number | null
  cache_prefix_reused?: boolean | null
  cache_prefix_message_count?: number | null
  prefix_decision?: string | null
  prefix_rebuild_reason?: string | null
  available_skill_count?: number | null
  skill_catalog_tokens?: number | null
  active_skill_names?: string[]
  active_skill_tokens?: number | null
  source_message_count?: number | null
  summary_covered_before?: number | null
  summary_covered_after?: number | null
  summary_snapshot?: ConversationSummarySnapshot | null
  summary_previous_snapshot?: ConversationSummarySnapshot | null
  constraints_possibly_dropped?: string[]
  constraints_revision?: number | null
  [key: string]: unknown
}

/** 滚动摘要的 7 个字段，与后端 RollingConversationSummary 一致。 */
export interface ConversationSummarySnapshot {
  current_objective: string | null
  user_constraints: string[]
  key_decisions: string[]
  completed_work: string[]
  current_state: string[]
  pending_work: string[]
  important_facts: string[]
}

/** 会话"必须记住的事项"：每次请求都附带，不参与压缩。 */
export interface ConversationConstraints {
  conversation_id: string
  text: string
  revision: number
  updated_at: string | null
}

export interface RewindStepInfo {
  step: number
  message_count: number
  has_snapshot: boolean
  snapshot_error: string | null
  rewindable: boolean
  reason: string | null
}

export interface RewindFileChange {
  path: string
  /** restore：恢复为该步之前的内容；delete：该步之后新建，将被删除 */
  action: 'restore' | 'delete'
  changed_after_run: boolean
}

export interface IrreversibleOperation {
  step: number | null
  tool: string
  summary: string
  note: string
}

export interface RewindPreview {
  preview_id: string
  run_id: string
  step: number
  files: RewindFileChange[]
  skipped_files: string[]
  irreversible: IrreversibleOperation[]
  blocked_reason: string | null
}

export interface RewindResult {
  rewind_key: string
  run_id: string
  step: number
  conversation_id: string
  draft: string
  files_restored: number
  files_deleted: number
  status: string
}

/** 由"重做此步"创建的分支会话的来源。 */
export interface ConversationFork {
  conversation_id: string
  source_conversation_id: string
  source_run_id: string
  source_step: number
  rewind_key: string
  undone: boolean
}

/** 某次工具调用完整原文的一页（来自证据库）。 */
export interface ToolEvidencePage {
  run_id: string
  tool_call_id: string
  tool_name: string
  evidence_id: string
  total_chars: number
  offset: number
  content: string
  next_offset: number | null
}

export interface RunContextMessage {
  index: number
  message: Message
  inherited: boolean
}

export interface RunContextMessagesPage {
  run_id: string
  conversation_id: string | null
  inherited_count: number
  total: number
  offset: number
  messages: RunContextMessage[]
}

export type AutomationStatus = 'active' | 'paused' | 'completed' | 'cancelled'
export type AutomationKind = 'once' | 'interval' | 'cron'

export interface Schedule {
  kind: AutomationKind
  run_at: string | null
  interval_seconds: number | null
  cron_expr: string | null
  timezone: string
}

export interface Automation {
  id: string
  title: string
  prompt: string
  conversation_id: string | null
  status: AutomationStatus
  schedule: Schedule
  next_run_at: string | null
  last_run_at: string | null
  last_run_id: string | null
  created_at: string
  updated_at: string
}

export type ApprovalStatus = 'pending' | 'approved' | 'denied'

export interface ApprovalRequest {
  id: string
  run_id: string | null
  conversation_id: string | null
  tool_name: string
  tool_call_id: string
  arguments: Record<string, unknown>
  reason: string
  status: ApprovalStatus
  created_at: string
  resolved_at: string | null
}

export interface Health {
  status: string
  provider: string
  model: string
  version: string
}

export type WsMessage =
  | { type: 'agent_event'; data: AgentEvent }
  | { type: 'run_status'; data: { run_id: string; status: string } }


// ---------------------------------------------------------------- 长任务（MEA）

export type MeaStatus =
  | 'running'
  | 'waiting_user'
  | 'paused'
  | 'finalizing'
  | 'completed'
  | 'blocked'
  | 'failed'
  | 'cancelled'

export type MeaRoundKind = 'normal' | 'audit_only' | 'final_audit' | 'recovery_audit'

export type MeaRoundPhase =
  | 'managing'
  | 'planned'
  | 'executing'
  | 'executed'
  | 'auditing'
  | 'audited'
  | 'applied'
  | 'interrupted'
  | 'abandoned'

export type MeaRole = 'manager' | 'executor' | 'auditor'

export interface MeaOnceNote {
  text: string
  consumed_in_round: number | null
}

export interface MeaCompletionDecision {
  outcome: 'completed' | 'blocked'
  round_index: number
  auditor_run_id: string | null
  requirements_revision: number
  reason: string | null
  decided_at: string
}

export interface MeaRun {
  id: string
  task_id: string
  conversation_id: string
  status: MeaStatus
  round_budget: number
  requirements_revision: number
  completion_decision: MeaCompletionDecision | null
  extra_tools: string[]
  auto_approve_sandbox?: boolean
  once_notes: MeaOnceNote[]
  pending_question: string | null
  pending_choices: string[]
  pause_requested: boolean
  stall_guard_after?: number
  abort_reason: string | null
  final_response: string | null
  created_at: string
  updated_at: string
}

/** mea.round 推送里的轮次摘要：不含计划、执行报告和审计报告全文。 */
export interface MeaRoundSummary {
  index: number
  ref: string
  kind: MeaRoundKind
  phase: MeaRoundPhase
  route: string | null
  step_id: string | null
  subtask: string | null
  focus: string | null
  manager_run_id: string | null
  executor_run_id: string | null
  auditor_run_id: string | null
  audit_status: string | null
  integrity_status: string | null
  contract_audit_status: string | null
  step_acceptance: string | null
  stale_requirements: boolean
  /** 审计者这次输出被截断（输出预算耗尽）；这样的审计不会判通过。 */
  audit_output_truncated?: boolean
  abandon_reason: string | null
  interrupt_reason: string | null
  created_at: string
  updated_at: string
}

export interface MeaRound extends Omit<MeaRoundSummary, 'ref'> {
  mea_run_id: string
  plan_text: string | null
  executor_output: string | null
  auditor_report: string | null
  harness_feedback: string | null
  related_refs: string[]
  executor_rejections: Record<string, number>
}

export interface MeaAmendment {
  id: string
  kind: 'answer' | 'note'
  text: string
  revision: number
  at: string
}

export interface MeaRequirements {
  original_request: string
  goal: string
  description: string | null
  constraints: string[]
  amendments: MeaAmendment[]
  revision: number
}

export interface MeaDetail {
  mea: MeaRun
  rounds: MeaRound[]
  requirements: MeaRequirements
  task: Task | null
}

export interface MeaAmendResult {
  accepted: boolean
  reason: string | null
  revision?: number | null
  amendment_id?: string | null
  mea?: MeaRun
}

/** 子 Run 的最近一条事件，用来显示“谁正在做什么”。 */
export interface MeaLiveActivity {
  role: MeaRole | null
  runId: string
  type: string
  toolName: string | null
  eventTime: string
}
