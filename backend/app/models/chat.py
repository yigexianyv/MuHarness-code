
from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from app.application import (
    DEFAULT_SYSTEM_PROMPT,
    Application,
    title_from_content,
)
from app.cli_ui import print_banner, print_startup_status, run_setup
from app.domain.conversation import (
    DEFAULT_DATABASE_PATH,
    Conversation,
    ConversationSource,
    SQLiteConversationStore,
    TriggerContext,
)
from app.domain.conversation.service import ConversationService
from app.domain.memory import (
    MemoryRecord,
)
from app.domain.skill_learning import (
    SkillCandidate,
    SkillLearningService,
)
from app.domain.task import (
    DEFAULT_TASKS_DIR,
)
from app.integrations.mcp import (
    DEFAULT_MCP_CONFIG_PATH,
    MCPServerState,
    MCPServerStatus,
)
from app.records.trace import AgentRunTrace
from app.runtime.agent.events import (
    AgentEvent,
    AgentEventHandler,
    AgentEventType,
)
from app.runtime.checkpoint import RunCheckpoint
from app.runtime.run import Run, RunStatus
from app.tools import (
    ApprovalScope,
    PermissionRule,
    SQLitePermissionRuleStore,
    ToolRegistry,
)
from app.tools.builtin import WebSearchTool
from app.tools.search import SearchError

from .types import Message, MessageRole, ModelProvider

_COMMAND_OVERVIEW = (
    "常用命令：/new 新建会话 · /sessions 切换会话 · /runs 查看运行 · "
    "/help 查看全部命令"
)

_HELP_TEXT = (
    "\n会话\n"
    "  /new [标题] 新建会话\n"
    "  /sessions 查看最近会话\n"
    "  /use <会话ID> 切换会话\n"
    "  /clear 清空当前会话\n"
    "\n运行与恢复\n"
    "  /runs 查看最近 Run\n"
    "  /run <Run ID> 查看 Run 生命周期详情\n"
    "  /run cancel <Run ID> 取消正在执行的 Run\n"
    "  /run recover <Run ID> 恢复中断的 Run\n"
    "  /checkpoints 查看当前会话的运行恢复点\n"
    "  /trace <Run ID> 查看完整事件轨迹\n"
    "\n记忆与扩展\n"
    "  /memories 查看活跃长期记忆及 Recall Cue\n"
    "  /memory <记忆ID> 查看一条长期记忆的完整内容\n"
    "  /mcp 查看 MCP Server 连接状态和已注册工具\n"
    "  /skill-candidates 查看待人工评审的 Skill Learning 候选\n"
    "  /skill-candidate <ID> 查看候选详情\n"
    "  /skill-candidate <ID> accept [scope] 接受候选：CREATE 创建 / UPDATE 更新"
    "正式 Skill（默认 project）\n"
    "  /skill-candidate <ID> reject 拒绝候选\n"
    "\n自动化与权限\n"
    "  /automations 查看当前会话的自动化\n"
    "  /automation <ID> 查看自动化详情\n"
    "  /automation cancel|pause|resume <ID> 管理自动化\n"
    "  /permissions 查看当前会话的审批规则\n"
    "  /permission remove <规则ID> 删除一条审批规则\n"
    "  /permissions clear 清除当前会话的全部审批规则\n"
    "\n系统\n"
    "  /setup 查看模型设置入口\n"
    "  /help 查看全部命令\n"
    "  /exit 退出聊天"
)


# 函数说明：_initial_history
# 用途：处理模型请求与响应处理中的 `_initial_history` 数据；结果及边界条件见下方说明。
# 参数：
#   system_prompt：系统提示文本，类型 `str | None`。
# 返回：类型 `list[Message]`；按分支返回 `[]`；
# `[Message(role=MessageRole.SYSTEM, content=system_prompt)]`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Message`。
# 分支与异常：
#   当 `not system_prompt` 时，返回 `[]`。
def _initial_history(system_prompt: str | None) -> list[Message]:
    if not system_prompt:
        return []
    return [Message(role=MessageRole.SYSTEM, content=system_prompt)]


class _CliEventHandler(AgentEventHandler):

    # 函数说明：_CliEventHandler.emit
    # 用途：发出_CliEventHandler，供模型请求与响应处理使用。
    # 参数：
    #   event：待记录或转发的事件，类型 `AgentEvent`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`_print_agent_event`。
    async def emit(self, event: AgentEvent) -> None:
        _print_agent_event(event)


# 函数说明：_send_message
# 用途：发送消息，供模型请求与响应处理使用。
# 参数：
#   conversation_service：会话输入或配置值，类型 `ConversationService`。
#   conversation_store：会话历史存储，类型 `SQLiteConversationStore`。
#   conversation：会话输入或配置值，类型 `Conversation`。
#   provider：模型或搜索服务商，类型 `ModelProvider | str`。
#   history：原始会话历史，类型 `list[Message]`。
#   content：内容正文，类型 `str`。
#   model：模型名称，类型 `str`。
# 返回：类型 `tuple[bool, Conversation]`；按分支返回 `(False, conversation)`；
# `(result.ok, conversation)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`conversation_service.dispatch` →
# `TriggerContext` → `_CliEventHandler` → `conversation_store.rename` →
# `title_from_content`。
# 分支与异常：
#   捕获 `KeyboardInterrupt` 后，返回 `(False, conversation)`。
async def _send_message(
    *,
    conversation_service: ConversationService,
    conversation_store: SQLiteConversationStore,
    conversation: Conversation,
    provider: ModelProvider | str,
    history: list[Message],
    content: str,
    model: str,
) -> tuple[bool, Conversation]:
    print("MuHarness 正在思考...", flush=True)

    try:
        dispatch = await conversation_service.dispatch(
            conversation_id=conversation.id,
            content=content,
            trigger=TriggerContext(source=ConversationSource.MANUAL),
            event_handler=_CliEventHandler(),
        )
    except KeyboardInterrupt:
        print("\n[cancel] 已取消当前 Run。")
        return False, conversation

    result = dispatch.result
    history[:] = result.messages
    if conversation.title == "新会话":
        conversation = await conversation_store.rename(
            conversation.id,
            title_from_content(content),
        )
    answer = result.content or "<模型未返回文本>"
    print(f"\nMuHarness> {answer.strip()}")
    stop_reason = dispatch.run.stop_reason or result.stop_reason.value
    provider_name = provider.value if isinstance(provider, ModelProvider) else provider
    print(
        f"\n[{provider_name}/{model} · {result.steps} steps · "
        f"{result.usage.total_tokens} tokens · {stop_reason}]"
    )
    if result.tool_calls:
        tools = ", ".join(
            f"{record.tool_call.name}({'成功' if record.result.success else '失败'})"
            for record in result.tool_calls
        )
        print(f"[工具调用：{tools}]")
    if dispatch.run.status.value == "cancelled":
        print("[Run 已被取消]")
    return result.ok, conversation


# 函数说明：_print_agent_event
# 用途：打印事件，供模型请求与响应处理使用。
# 参数：
#   event：待记录或转发的事件，类型 `AgentEvent`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `event.event_time.astimezone().strftime` → `event.event_time.astimezone`。
def _print_agent_event(event: AgentEvent) -> None:

    event_time = event.event_time.astimezone().strftime("%H:%M:%S")
    prefix = f"[{event_time}]"
    if event.type is AgentEventType.AGENT_STARTED:
        print(f"{prefix} Agent 开始执行")
    elif event.type is AgentEventType.MODEL_STARTED:
        print(f"{prefix} 第 {event.step} 步：正在请求模型")
    elif event.type is AgentEventType.MODEL_COMPLETED:
        tool_count = len(event.message.tool_calls) if event.message else 0
        if tool_count:
            print(f"{prefix} 模型请求调用 {tool_count} 个工具")
        else:
            print(f"{prefix} 模型已返回回复")
    elif event.type is AgentEventType.RUN_BUDGET_WARNING:
        print(
            f"{prefix} Run 用量进入预警区："
            f"{event.run_budget_chargeable_tokens or 0} tokens · "
            f"{event.run_budget_model_calls or 0} calls"
        )
    elif event.type is AgentEventType.RUN_BUDGET_FINALIZING:
        print(f"{prefix} Run 用量达到收口线，正在生成最终答复")
    elif event.type is AgentEventType.RUN_BUDGET_EXCEEDED:
        print(f"{prefix} Run 用量达到硬上限，停止继续请求模型")
    elif event.type is AgentEventType.TOOL_STARTED and event.tool_call:
        print(f"{prefix} 开始执行工具：{event.tool_call.name}")
    elif event.type is AgentEventType.TOOL_COMPLETED and event.tool_result:
        status = "成功" if event.tool_result.success else "失败"
        print(
            f"{prefix} 工具 {event.tool_result.tool_name} {status} "
            f"({event.tool_result.duration_ms:.1f}ms)"
        )
    elif event.type is AgentEventType.TOOL_APPROVAL_REQUIRED and event.tool_call:
        print(f"{prefix} 工具等待人工审批：{event.tool_call.name}")
    elif event.type is AgentEventType.TOOL_APPROVAL_COMPLETED and event.tool_call:
        decision = (
            event.approval_decision.value
            if event.approval_decision is not None
            else "unknown"
        )
        rule_text = (
            f" · 规则：{event.rule_description}"
            if event.rule_description is not None
            else ""
        )
        print(
            f"{prefix} 工具权限检查完成：{event.tool_call.name} · {decision}{rule_text}"
        )
    elif event.type is AgentEventType.MEMORY_REFLECTION_STARTED:
        print(f"{prefix} 正在整理本轮长期记忆")
    elif event.type is AgentEventType.MEMORY_REFLECTION_COMPLETED:
        action = event.reflection_action or "none"
        suffix = (
            f" · {event.reflection_memory_id}"
            if event.reflection_memory_id is not None
            else ""
        )
        print(f"{prefix} 长期记忆整理完成：{action}{suffix}")
    elif event.type is AgentEventType.MEMORY_REFLECTION_FAILED:
        print(f"{prefix} 长期记忆整理失败，已跳过")
    elif event.type is AgentEventType.MEMORY_REFLECTION_SKIPPED:
        reason = event.reflection_skip_reason or "policy"
        print(f"{prefix} 长期记忆整理已跳过：{reason}")
    elif event.type is AgentEventType.MEMORY_MAINTENANCE_STARTED:
        print(f"{prefix} 长期记忆容量不足，正在选择可归档候选")
    elif event.type is AgentEventType.MEMORY_MAINTENANCE_COMPLETED:
        action = event.maintenance_action or "unknown"
        suffix = (
            f" · {event.maintenance_memory_id}"
            if event.maintenance_memory_id is not None
            else ""
        )
        print(f"{prefix} 长期记忆容量维护完成：{action}{suffix}")
    elif event.type is AgentEventType.MEMORY_MAINTENANCE_FAILED:
        print(f"{prefix} 长期记忆容量维护失败，未执行归档")
    elif event.type is AgentEventType.AGENT_COMPLETED:
        print(f"{prefix} Agent 执行完成")
    elif event.type is AgentEventType.AGENT_FAILED:
        reason = event.stop_reason.value if event.stop_reason else "unknown"
        print(f"{prefix} Agent 执行停止：{reason}")


# 函数说明：_load_or_create_conversation
# 用途：加载会话，供模型请求与响应处理使用。
# 参数：
#   store：持久化存储依赖，类型 `SQLiteConversationStore`。
#   identifier：待规范化的标识，类型 `str | None`。
#   force_new：`force_new`输入或配置值，类型 `bool`。
#   system_prompt：系统提示文本，类型 `str | None`。
# 返回：类型 `tuple[Conversation, list[Message], bool]`；按分支返回
# `(conversation, list(await store.load_messages(conversation.id)), False)`；
# `(conversation, history, True)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.create` → `_initial_history` →
# `store.load_messages` → `store.resolve` → `store.latest`。
# 分支与异常：
#   `force_new` 分支在完成前置处理后返回
# `(conversation, list(await store.load_messages(…`。
#   当 `conversation is None` 时，抛出 `ValueError(f'找不到会话：{identifier}')`。
#   `conversation is None` 分支在完成前置处理后返回
# `(conversation, list(await store.load_messages(…`。
async def _load_or_create_conversation(
    store: SQLiteConversationStore,
    *,
    identifier: str | None,
    force_new: bool,
    system_prompt: str | None,
) -> tuple[Conversation, list[Message], bool]:

    if force_new:
        conversation = await store.create(messages=_initial_history(system_prompt))
        return conversation, list(await store.load_messages(conversation.id)), False

    if identifier:
        conversation = await store.resolve(identifier)
        if conversation is None:
            raise ValueError(f"找不到会话：{identifier}")
    else:
        conversation = await store.latest()

    if conversation is None:
        conversation = await store.create(messages=_initial_history(system_prompt))
        return conversation, list(await store.load_messages(conversation.id)), False

    history = list(await store.load_messages(conversation.id))
    return conversation, history, True


# 函数说明：_print_conversations
# 用途：打印`conversations`，供模型请求与响应处理使用。
# 参数：
#   conversations：`conversations`输入或配置值，类型 `tuple[Conversation, ...]`。
#   current_id：`current`标识，类型 `str`。
# 返回：类型 `None`；无结果值，显式返回 None。
# 分支与异常：
#   `not conversations` 分支在完成前置处理后返回 `None`。
def _print_conversations(
    conversations: tuple[Conversation, ...],
    current_id: str,
) -> None:
    if not conversations:
        print("暂无会话。")
        return
    for conversation in conversations:
        marker = "*" if conversation.id == current_id else " "
        print(
            f"{marker} {conversation.id[:8]}  "
            f"{conversation.title}  ({conversation.message_count} 条消息)"
        )


# 函数说明：_print_runs
# 用途：打印运行集合，供模型请求与响应处理使用。
# 参数：
#   runs：运行集合输入或配置值，类型 `tuple[AgentRunTrace, ...]`。
#   current_conversation_id：会话标识，类型 `str`。
# 返回：类型 `None`；无结果值，显式返回 None。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`run.started_at.astimezone().strftime`
#  → `run.started_at.astimezone`。
# 分支与异常：
#   `not runs` 分支在完成前置处理后返回 `None`。
def _print_runs(runs: tuple[AgentRunTrace, ...], current_conversation_id: str) -> None:

    if not runs:
        print("暂无运行记录。")
        return
    for run in runs:
        marker = "*" if run.conversation_id == current_conversation_id else " "
        started_at = run.started_at.astimezone().strftime("%m-%d %H:%M:%S")
        reason = run.stop_reason.value if run.stop_reason else "-"
        print(
            f"{marker} {run.run_id[:8]}  {started_at}  {run.status.value:<9} "
            f"steps={run.steps} tokens={run.total_tokens} reason={reason}"
        )


# 函数说明：_print_run_lifecycle
# 用途：打印运行，供模型请求与响应处理使用。
# 参数：
#   runs：运行集合输入或配置值，类型 `tuple[Run, ...]`。
#   current_conversation_id：会话标识，类型 `str`。
# 返回：类型 `None`；无结果值，显式返回 None。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`run.created_at.astimezone().strftime`
#  → `run.created_at.astimezone`。
# 分支与异常：
#   `not runs` 分支在完成前置处理后返回 `None`。
def _print_run_lifecycle(runs: tuple[Run, ...], current_conversation_id: str) -> None:

    if not runs:
        print("暂无 Run 记录。")
        return
    for run in runs:
        marker = "*" if run.conversation_id == current_conversation_id else " "
        created_at = run.created_at.astimezone().strftime("%m-%d %H:%M:%S")
        stop_reason = run.stop_reason or "-"
        print(
            f"{marker} {run.id[:8]}  {created_at}  "
            f"{run.status.value:<11} reason={stop_reason}"
        )


# 函数说明：_print_run_detail
# 用途：打印运行，供模型请求与响应处理使用。
# 参数：
#   run：当前运行记录，类型 `Run`。
# 返回：类型 `None`；无结果值，显式返回 None。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`run.started_at.astimezone().strftime`
#  → `run.started_at.astimezone` → `run.completed_at.astimezone().strftime` →
# `run.completed_at.astimezone` → `run.created_at.astimezone().strftime` →
# `run.created_at.astimezone`。
# 分支与异常：
#   `run is None` 分支在完成前置处理后返回 `None`。
def _print_run_detail(run: Run) -> None:

    if run is None:
        print("找不到 Run。")
        return
    started = (
        run.started_at.astimezone().strftime("%m-%d %H:%M:%S")
        if run.started_at is not None
        else "-"
    )
    completed = (
        run.completed_at.astimezone().strftime("%m-%d %H:%M:%S")
        if run.completed_at is not None
        else "-"
    )
    print(f"Run {run.id}")
    print(f"  conversation_id: {run.conversation_id or '-'}")
    print(f"  status: {run.status.value}")
    print(f"  created_at: {run.created_at.astimezone().strftime('%m-%d %H:%M:%S')}")
    print(f"  started_at: {started}")
    print(f"  completed_at: {completed}")
    print(f"  stop_reason: {run.stop_reason or '-'}")
    if run.error:
        print(f"  error: {run.error}")
    if run.recovered_from_run_id:
        print(f"  recovered_from: {run.recovered_from_run_id[:8]}")
    print(f"  user_message: {(run.user_message or '')[:120]}")


# 函数说明：_print_automations
# 用途：打印自动化任务集合，供模型请求与响应处理使用。
# 参数：
#   automations：自动化任务集合输入或配置值，类型 `tuple[object, ...]`。
# 返回：类型 `None`；无结果值，显式返回 None。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `item.next_run_at.astimezone().strftime` → `item.next_run_at.astimezone`。
# 分支与异常：
#   `not automations` 分支在完成前置处理后返回 `None`。
def _print_automations(automations: tuple[object, ...]) -> None:

    if not automations:
        print("暂无 Automation。")
        return
    for item in automations:
        next_run = (
            item.next_run_at.astimezone().strftime("%m-%d %H:%M")
            if item.next_run_at is not None
            else "-"
        )
        print(
            f"{item.id[:8]}  {item.title[:24]:<24}  "
            f"{item.status.value:<9} next={next_run}"
        )


# 函数说明：_print_automation
# 用途：打印自动化任务，供模型请求与响应处理使用。
# 参数：
#   automation：自动化任务输入或配置值，类型 `object`。
# 返回：类型 `None`；无结果值，显式返回 None。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `automation.next_run_at.astimezone().strftime` → `automation.next_run_at.astimezone`。
# 分支与异常：
#   `automation is None` 分支在完成前置处理后返回 `None`。
def _print_automation(automation: object) -> None:

    if automation is None:
        print("找不到 Automation。")
        return
    schedule = automation.schedule
    print(f"Automation {automation.id}")
    print(f"  title: {automation.title}")
    print(f"  status: {automation.status.value}")
    print(f"  kind: {schedule.kind.value}")
    if schedule.kind.value == "once":
        print(f"  run_at: {schedule.run_at}")
    elif schedule.kind.value == "interval":
        print(f"  interval_seconds: {schedule.interval_seconds}")
    else:
        print(f"  cron_expr: {schedule.cron_expr}")
    print(f"  timezone: {schedule.timezone}")
    print(f"  conversation_id: {automation.conversation_id or '-'}")
    next_run = (
        automation.next_run_at.astimezone().strftime("%m-%d %H:%M:%S")
        if automation.next_run_at is not None
        else "-"
    )
    print(f"  next_run_at: {next_run}")
    print(f"  last_run_id: {automation.last_run_id or '-'}")
    print(f"  prompt: {(automation.prompt or '')[:160]}")


# 函数说明：_print_checkpoints
# 用途：打印检查点，供模型请求与响应处理使用。
# 参数：
#   checkpoints：检查点输入或配置值，类型 `tuple[RunCheckpoint, ...]`。
# 返回：类型 `None`；无结果值，显式返回 None。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `checkpoint.updated_at.astimezone().strftime` → `checkpoint.updated_at.astimezone`。
# 分支与异常：
#   `not checkpoints` 分支在完成前置处理后返回 `None`。
def _print_checkpoints(checkpoints: tuple[RunCheckpoint, ...]) -> None:

    if not checkpoints:
        print("当前会话暂无 Checkpoint。")
        return
    for checkpoint in checkpoints:
        updated_at = checkpoint.updated_at.astimezone().strftime("%m-%d %H:%M:%S")
        pending = len(checkpoint.pending_tool_calls)
        print(
            f"{checkpoint.run_id[:8]}  {updated_at}  "
            f"{checkpoint.status.value:<11} phase={checkpoint.phase.value} "
            f"step={checkpoint.step} pending_tools={pending}"
        )


# 函数说明：_print_mcp_statuses
# 用途：打印`mcp_statuses`，供模型请求与响应处理使用。
# 参数：
#   statuses：`statuses`输入或配置值，类型 `tuple[MCPServerStatus, ...]`。
# 返回：类型 `None`；无结果值，显式返回 None。
# 分支与异常：
#   `not statuses` 分支在完成前置处理后返回 `None`。
def _print_mcp_statuses(statuses: tuple[MCPServerStatus, ...]) -> None:

    if not statuses:
        print("尚未配置 MCP Server。")
        return
    for status in statuses:
        marker = "✓" if status.state is MCPServerState.RUNNING else "-"
        print(f"{marker} {status.name} · {status.state.value}")
        if status.error:
            print(f"  错误：{status.error}")
        for tool_name in status.tool_names:
            print(f"  - {tool_name}")


# 函数说明：_print_recovered_runs
# 用途：打印运行集合，供模型请求与响应处理使用。
# 参数：
#   runs：运行集合输入或配置值，类型 `tuple[Run, ...]`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
def _print_recovered_runs(runs: tuple[Run, ...]) -> None:

    for run in runs:
        if run.status is RunStatus.INTERRUPTED:
            print(
                f"检测到中断 Run：{run.id[:8]} · 可恢复 "
                f"（/run recover {run.id[:8]} 继续执行）"
            )
        else:
            print(
                f"Run 状态修正：{run.id[:8]} RUNNING → {run.status.value}"
                + (f" · {run.error}" if run.error else "")
            )


# 函数说明：_print_trace
# 用途：打印执行轨迹，供模型请求与响应处理使用。
# 参数：
#   events：事件集合，类型 `tuple[AgentEvent, ...]`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `event.event_time.astimezone().strftime` → `event.event_time.astimezone`。
def _print_trace(events: tuple[AgentEvent, ...]) -> None:

    for event in events:
        event_time = event.event_time.astimezone().strftime("%H:%M:%S.%f")[:-3]
        details: list[str] = []
        if event.step is not None:
            details.append(f"step={event.step}")
        if event.tool_call is not None:
            details.append(f"tool={event.tool_call.name}")
        if event.approval_decision is not None:
            details.append(f"decision={event.approval_decision.value}")
        if event.rule_id is not None:
            details.append(f"rule={event.rule_id[:8]}")
        if event.tool_result is not None:
            details.append(
                f"success={'true' if event.tool_result.success else 'false'}"
            )
        if event.type is AgentEventType.MODEL_STARTED:
            if event.prepared_input_tokens is not None:
                details.append(f"input≈{event.prepared_input_tokens}")
            if event.tool_schema_tokens is not None:
                details.append(f"schemas≈{event.tool_schema_tokens}")
            if (
                event.tool_result_tokens_before is not None
                and event.tool_result_tokens_after is not None
            ):
                details.append(
                    "tool_results≈"
                    f"{event.tool_result_tokens_before}→"
                    f"{event.tool_result_tokens_after}"
                )
            if event.compaction_stage not in (None, "none"):
                details.append(f"context={event.compaction_stage}")
            if event.summary_provider is not None:
                details.append(
                    "summary="
                    f"{event.summary_provider}/{event.summary_model or 'default'}"
                )
            if event.summary_duration_ms is not None:
                details.append(f"summary_ms={event.summary_duration_ms:.1f}")
        if event.run_budget_status is not None:
            details.append(
                f"budget={event.run_budget_status}:"
                f"{event.run_budget_chargeable_tokens or 0}t/"
                f"{event.run_budget_model_calls or 0}calls"
            )
        detail_text = f"  {' '.join(details)}" if details else ""
        print(f"{event.sequence:03d}  {event_time}  {event.type.value}{detail_text}")


# 函数说明：_print_skill_candidates
# 用途：打印技能候选集合，供模型请求与响应处理使用。
# 参数：
#   candidates：候选集合输入或配置值，类型 `tuple[SkillCandidate, ...]`。
# 返回：类型 `None`；无结果值，显式返回 None。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`candidate.action.value.upper`。
# 分支与异常：
#   `not candidates` 分支在完成前置处理后返回 `None`。
def _print_skill_candidates(candidates: tuple[SkillCandidate, ...]) -> None:

    if not candidates:
        print("没有 Skill Learning 候选。")
        return
    for candidate in candidates:
        print(
            f"{candidate.id}  [{candidate.status.value}] "
            f"{candidate.action.value.upper()} {candidate.proposed_name} "
            f"· 来源 {len(candidate.source_task_ids)} Task"
        )


# 函数说明：_maybe_run_skill_learning
# 用途：运行技能，供模型请求与响应处理使用。
# 参数：
#   skill_learning：技能输入或配置值，类型 `SkillLearningService`。
# 返回：类型 `None`；无结果值，显式返回 None。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`skill_learning.maybe_run_mining`。
# 分支与异常：
#   捕获 `Exception` 后，返回 `None`。
#   当 `not outcome.triggered` 时，返回 `None`。
async def _maybe_run_skill_learning(
    skill_learning: SkillLearningService,
) -> None:

    try:
        outcome = await skill_learning.maybe_run_mining()
    except Exception as exc:
        print(f"Skill Learning 失败：{type(exc).__name__}: {exc}")
        return
    if not outcome.triggered:
        return
    print("Skill Learning:")
    print(f"  tasks scanned: {outcome.scanned_task_count}")
    print(f"  clusters: {outcome.cluster_count}")
    print(f"  candidates: {outcome.candidate_count}")
    print(f"  model calls: {outcome.pattern_mining_calls + outcome.distillation_calls}")
    print(f"  input tokens: {outcome.input_tokens}")
    print(f"  output tokens: {outcome.output_tokens}")
    print(f"  total tokens: {outcome.total_tokens}")
    print(f"  latency: {outcome.total_duration_ms / 1000:.1f}s")
    if outcome.error:
        print(f"  error: {outcome.error}")
    if outcome.candidate_count:
        print("  运行 /skill-candidates 查看待评审候选")


# 函数说明：_mark_deferred_tools
# 用途：标记需要延后暴露的工具定义，供工具发现流程使用。
# 参数：
#   registry：工具、模型或能力注册表，类型 `ToolRegistry`。
#   names：`names`输入或配置值，类型 `frozenset[str]`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`registry.unregister` →
# `registry.register`。
def _mark_deferred_tools(
    registry: ToolRegistry,
    names: frozenset[str],
) -> None:

    for name in names:
        tool = registry.unregister(name)
        registry.register(tool, deferred=True)


# 函数说明：_print_permission_rules
# 用途：打印权限规则集合，供模型请求与响应处理使用。
# 参数：
#   rules：规则集合输入或配置值，类型 `tuple[PermissionRule, ...]`。
# 返回：类型 `None`；无结果值，显式返回 None。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `rule.created_at.astimezone().strftime` → `rule.created_at.astimezone`。
# 分支与异常：
#   `not rules` 分支在完成前置处理后返回 `None`。
def _print_permission_rules(rules: tuple[PermissionRule, ...]) -> None:

    if not rules:
        print("当前会话没有已记住的审批规则。")
        return
    for rule in rules:
        created_at = rule.created_at.astimezone().strftime("%m-%d %H:%M:%S")
        print(f"{rule.id[:8]}  {created_at}  {rule.tool_name}  {rule.description}")


# 函数说明：_remove_permission_rule
# 用途：移除权限规则，供模型请求与响应处理使用。
# 参数：
#   store：持久化存储依赖，类型 `SQLitePermissionRuleStore`。
#   conversation_id：目标会话标识，类型 `str`。
#   identifier：待规范化的标识，类型 `str`。
# 返回：类型 `bool`；按分支返回 `False`；`await store.remove(matched[0].id)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`store.list` → `rule.id.startswith` →
# `store.remove`。
# 分支与异常：
#   当 `not normalized` 时，返回 `False`。
#   当 `len(matched) > 1` 时，抛出 `ValueError(f'规则 ID 前缀不唯一：{identifier}')`。
#   当 `not matched` 时，返回 `False`。
async def _remove_permission_rule(
    store: SQLitePermissionRuleStore,
    conversation_id: str,
    identifier: str,
) -> bool:

    normalized = identifier.strip()
    if not normalized:
        return False
    rules = await store.list(scope_ids=(conversation_id,))
    matched = [rule for rule in rules if rule.id.startswith(normalized)]
    if len(matched) > 1:
        raise ValueError(f"规则 ID 前缀不唯一：{identifier}")
    if not matched:
        return False
    return await store.remove(matched[0].id)


# 函数说明：_print_memories
# 用途：打印记忆集合，供模型请求与响应处理使用。
# 参数：
#   memories：记忆集合输入或配置值，类型 `Sequence[MemoryRecord]`。
# 返回：类型 `None`；无结果值，显式返回 None。
# 分支与异常：
#   `not memories` 分支在完成前置处理后返回 `None`。
def _print_memories(memories: Sequence[MemoryRecord]) -> None:

    if not memories:
        print("没有长期记忆。")
        return
    for memory in memories:
        cue = " ".join(memory.summary.split())
        print(f"{memory.id}  [{memory.status.value}]  {memory.title}")
        print(f"  Cue: {cue}")


# 函数说明：_print_memory
# 用途：打印记忆，供模型请求与响应处理使用。
# 参数：
#   memory：记忆输入或配置值，类型 `MemoryRecord`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `memory.created_at.astimezone().isoformat` → `memory.created_at.astimezone` →
# `memory.render_full`。
def _print_memory(memory: MemoryRecord) -> None:

    print(
        f"ID: {memory.id}\n"
        f"状态: {memory.status.value}\n"
        f"标题: {memory.title}\n"
        f"访问次数: {memory.access_count}\n"
        f"创建时间: {memory.created_at.astimezone().isoformat()}\n"
        f"内容:\n{memory.render_full()}"
    )


# 函数说明：_run
# 用途：运行模型请求与响应处理，供模型请求与响应处理使用。
# 参数：
#   args：传给 `_run` 的输入，类型 `argparse.Namespace`。
#   offer_setup：`offer_setup`输入或配置值，类型 `bool`；默认 `True`。
# 返回：类型 `int`；按分支返回 `await _run(args, offer_setup=False)`；`0`；`2`；
# `0 if success else 1`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`print_banner` → `Application` →
# `sys.stdin.isatty` → `run_setup` → `_run` → `app.start`；另有 55 个调用点。
# 分支与异常：
#   捕获 `ValueError` 后，返回 `await _run(args, offer_setup=False)`。
#   `offer_setup and missing_provider and (args.message is None)…` 分支在完成前置处理后
# 返回 `0`。
#   当 `await run_setup()` 时，返回 `await _run(args, offer_setup=False)`。
#   捕获 `SearchError` 后，返回 `2`。
#   捕获 `ValueError` 后，返回 `2`。
#   捕获 `(EOFError, KeyboardInterrupt)` 后，返回 `0`。
#   捕获 `ValueError` 后，跳过当前循环项，继续处理后续项。
# 副作用与资源：
#   阻塞操作通过线程执行，调用方仍以 await 等待最终结果。
async def _run(
    args: argparse.Namespace,
    *,
    offer_setup: bool = True,
) -> int:
    print_banner()
    try:
        app = Application(
            provider=args.provider,
            model=args.model,
            system_prompt=args.system,
            database=args.database,
            tasks_dir=args.tasks_dir,
            mcp_config=args.mcp_config,
            max_steps=args.max_steps,
            max_tool_rounds=args.max_tool_rounds,
            max_output_tokens=args.max_output_tokens,
        )
    except ValueError as exc:
        missing_provider = "No model provider is configured" in str(exc)
        if (
            offer_setup
            and missing_provider
            and args.message is None
            and sys.stdin.isatty()
        ):
            print("尚未配置主模型，正在进入首次设置。")
            if await run_setup():
                return await _run(args, offer_setup=False)
            return 0
        print(f"启动失败：{exc}", file=sys.stderr)
        print(
            "请运行 `.venv/bin/python -m app --setup` 完成模型配置。",
            file=sys.stderr,
        )
        return 2
    provider = app.provider
    model = app.model

    try:
        await app.start()
    except SearchError as exc:
        print(f"搜索配置错误：{exc}", file=sys.stderr)
        return 2
    if app.mcp_error is not None:
        print(f"MCP 配置错误：{app.mcp_error}", file=sys.stderr)
        return 2

    conversation_store = app.conversation_store
    summary_store = app.summary_store
    trace_store = app.trace_store
    checkpoint_store = app.checkpoint_store
    rule_store = app.rule_store
    run_manager = app.run_manager
    conversation_service = app.conversation_service
    automation_scheduler = app.automation_scheduler
    memory_manager = app.memory_manager
    skill_learning = app.skill_learning
    mcp_manager = app.mcp_manager
    tool_registry = app.tool_registry
    try:
        conversation, history, resumed = await _load_or_create_conversation(
            conversation_store,
            identifier=args.conversation,
            force_new=args.new_conversation,
            system_prompt=args.system,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    action = "已恢复" if resumed else "已创建"
    search_tool = tool_registry.get("web_search")
    search_status = "未启用"
    if isinstance(search_tool, WebSearchTool):
        if search_tool.provider_name == "tavily":
            search_status = "Tavily · DuckDuckGo fallback"
        else:
            search_status = "DuckDuckGo · 配置 TAVILY_API_KEY 可启用 Tavily"
    learning_status = "关闭"
    if skill_learning.settings.skill_learning_enabled:
        learning_status = (
            f"每 {skill_learning.settings.skill_learning_batch_size} 个 "
            "Completed Task 扫描一次"
        )
    reflection_status = "启用" if app.memory_reflection_enabled else "关闭"
    reflection_model = app.memory_reflector.model_hint or "未解析"
    reflection_provider = app.memory_reflector.provider_hint or "未解析"
    connected_mcp = sum(
        status.state is MCPServerState.RUNNING for status in app.mcp_statuses
    )
    failed_mcp = sum(
        status.state is MCPServerState.FAILED for status in app.mcp_statuses
    )
    mcp_status = "未配置"
    if app.mcp_statuses:
        mcp_status = f"{connected_mcp} 已连接 · {failed_mcp} 启动失败"
    notices = []
    if failed_mcp:
        notices.append("存在 MCP 启动失败，输入 /mcp 查看详情")
    print_startup_status(
        (
            ("主模型", f"{provider}/{model}"),
            (
                "会话",
                f"{action} {conversation.id[:8]} · {conversation.title} · "
                f"{conversation.message_count} 条消息",
            ),
            ("搜索", search_status),
            ("MCP", mcp_status),
            (
                "长期记忆",
                f"Sparse Memory · 反思{reflection_status} "
                f"{reflection_provider}/{reflection_model}",
            ),
            ("技能学习", learning_status),
        ),
        notices=notices,
    )
    if app.reconciled_runs:
        _print_recovered_runs(app.reconciled_runs)
    try:
        if args.message is not None:
            success, conversation = await _send_message(
                conversation_service=conversation_service,
                conversation_store=conversation_store,
                conversation=conversation,
                provider=provider,
                history=history,
                content=args.message,
                model=model,
            )
            await _maybe_run_skill_learning(skill_learning)
            return 0 if success else 1

        print(_COMMAND_OVERVIEW)
        while True:
            try:
                content = (await asyncio.to_thread(input, "\n你> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n聊天已结束。")
                return 0

            if not content:
                continue
            if content in {"/exit", "/quit"}:
                print("聊天已结束。")
                return 0
            if content == "/new" or content.startswith("/new "):
                title = content.removeprefix("/new").strip() or "新会话"
                conversation = await conversation_store.create(
                    title=title,
                    messages=_initial_history(args.system),
                )
                history = list(await conversation_store.load_messages(conversation.id))
                print(f"已创建会话：{conversation.id[:8]} · {conversation.title}")
                continue
            if content == "/sessions":
                _print_conversations(
                    await conversation_store.list(),
                    conversation.id,
                )
                continue
            if content == "/memories" or content.startswith("/memories "):
                _print_memories(await memory_manager.list())
                continue
            if content == "/mcp":
                _print_mcp_statuses(mcp_manager.statuses())
                continue
            if content == "/memory" or content.startswith("/memory "):
                identifier = content.removeprefix("/memory").strip()
                if not identifier:
                    print("用法：/memory <记忆ID>")
                    continue
                memory = await memory_manager.read(identifier)
                if memory is None:
                    print(f"找不到记忆：{identifier}")
                    continue
                _print_memory(memory)
                continue
            if content == "/permissions":
                rules = await rule_store.list(scope_ids=(conversation.id,))
                _print_permission_rules(rules)
                continue
            if content == "/permissions clear":
                removed = await rule_store.remove_scope(
                    ApprovalScope.CONVERSATION,
                    conversation.id,
                )
                print(f"已清除当前会话的 {removed} 条审批规则。")
                continue
            if content == "/permission remove" or content.startswith(
                "/permission remove "
            ):
                identifier = content.removeprefix("/permission remove").strip()
                if not identifier:
                    print("用法：/permission remove <规则ID>")
                    continue
                try:
                    removed = await _remove_permission_rule(
                        rule_store,
                        conversation.id,
                        identifier,
                    )
                except ValueError as exc:
                    print(exc)
                    continue
                if removed:
                    print(f"已删除审批规则：{identifier}")
                else:
                    print(f"当前会话找不到审批规则：{identifier}")
                continue
            if content == "/runs":
                _print_run_lifecycle(
                    await run_manager.list_runs(),
                    conversation.id,
                )
                continue
            if content == "/run" or content.startswith("/run "):
                parts = content.split()
                if len(parts) == 1:
                    _print_run_lifecycle(
                        await run_manager.list_runs(
                            conversation_id=conversation.id,
                        ),
                        conversation.id,
                    )
                    continue
                action_word = parts[1].lower()
                if action_word in ("cancel", "recover"):
                    if len(parts) < 3:
                        print(f"用法：/run {action_word} <Run ID>")
                        continue
                    identifier = parts[2]
                    run = await run_manager.get_run(identifier)
                    if run is None:
                        print(f"找不到 Run：{identifier}")
                        continue
                    if action_word == "cancel":
                        try:
                            updated = await run_manager.cancel(run.id)
                        except (KeyError, ValueError) as exc:
                            print(exc)
                            continue
                        print(
                            f"Run {updated.id[:8]} 已取消：{updated.status.value}"
                        )
                    else:
                        try:
                            dispatch = await conversation_service.recover(
                                run.id,
                                event_handler=_CliEventHandler(),
                            )
                        except (KeyError, ValueError) as exc:
                            print(exc)
                            continue
                        recovered_run = dispatch.run
                        if dispatch.result is not None:
                            history[:] = dispatch.result.messages
                        print(
                            f"Run {recovered_run.id[:8]} 恢复自 "
                            f"{run.id[:8]} · {recovered_run.status.value}"
                        )
                    continue
                run = await run_manager.get_run(parts[1])
                if run is None:
                    print(f"找不到 Run：{parts[1]}")
                    continue
                _print_run_detail(run)
                continue
            if content == "/checkpoints":
                _print_checkpoints(
                    await checkpoint_store.list(
                        conversation_id=conversation.id,
                    )
                )
                continue
            if content == "/trace" or content.startswith("/trace "):
                identifier = content.removeprefix("/trace").strip()
                if not identifier:
                    print("用法：/trace <Run ID>")
                    continue
                try:
                    run = await trace_store.resolve(identifier)
                except ValueError as exc:
                    print(exc)
                    continue
                if run is None:
                    print(f"找不到 Run：{identifier}")
                    continue
                print(
                    f"Run {run.run_id} · {run.status.value} · {run.event_count} 个事件"
                )
                _print_trace(await trace_store.load_events(run.run_id))
                continue
            if content == "/automations":
                _print_automations(
                    await automation_scheduler.list(
                        conversation_id=conversation.id,
                    )
                )
                continue
            if content == "/automation" or content.startswith("/automation "):
                parts = content.split()
                if len(parts) < 2:
                    _print_automations(
                        await automation_scheduler.list(
                            conversation_id=conversation.id,
                        )
                    )
                    continue
                action_word = parts[1].lower()
                if action_word in ("cancel", "pause", "resume"):
                    if len(parts) < 3:
                        print(f"用法：/automation {action_word} <ID>")
                        continue
                    identifier = parts[2]
                    try:
                        automation = await automation_scheduler.resolve(
                            identifier
                        )
                    except ValueError as exc:
                        print(exc)
                        continue
                    if automation is None:
                        print(f"找不到 Automation：{identifier}")
                        continue
                    try:
                        if action_word == "cancel":
                            updated = await automation_scheduler.cancel(
                                automation.id
                            )
                        elif action_word == "pause":
                            updated = await automation_scheduler.pause(
                                automation.id
                            )
                        else:
                            updated = await automation_scheduler.resume(
                                automation.id
                            )
                    except ValueError as exc:
                        print(exc)
                        continue
                    print(
                        f"Automation {updated.id[:8]} "
                        f"{action_word} → {updated.status.value}"
                    )
                    continue
                automation = await automation_scheduler.get(parts[1])
                if automation is None:
                    print(f"找不到 Automation：{parts[1]}")
                    continue
                _print_automation(automation)
                continue
            if content == "/use" or content.startswith("/use "):
                identifier = content.removeprefix("/use").strip()
                if not identifier:
                    print("用法：/use <会话ID>")
                    continue
                try:
                    selected = await conversation_store.resolve(identifier)
                except ValueError as exc:
                    print(exc)
                    continue
                if selected is None:
                    print(f"找不到会话：{identifier}")
                    continue
                conversation = selected
                history = list(await conversation_store.load_messages(conversation.id))
                _print_recovered_runs(
                    await run_manager.list_runs(
                        conversation_id=conversation.id,
                        status=RunStatus.INTERRUPTED,
                    )
                )
                print(
                    f"已切换会话：{conversation.id[:8]} · "
                    f"{conversation.title} · {conversation.message_count} 条消息"
                )
                continue
            if content == "/skill-candidates":
                _print_skill_candidates(
                    await skill_learning.list_candidates()
                )
                continue
            if content == "/skill-candidate" or content.startswith(
                "/skill-candidate "
            ):
                parts = content.split()
                if len(parts) < 2:
                    print("用法：/skill-candidate <ID> [accept [scope] | reject]")
                    continue
                candidate = await skill_learning.get_candidate(parts[1])
                if candidate is None:
                    print(f"找不到候选：{parts[1]}")
                    continue
                if len(parts) == 2:
                    print(skill_learning.render_candidate_details(candidate))
                    continue
                action_word = parts[2].lower()
                if action_word == "accept":
                    scope = parts[3] if len(parts) > 3 else None
                    try:
                        updated, target = await skill_learning.accept(
                            candidate.id,
                            scope=scope,
                        )
                    except (KeyError, ValueError) as exc:
                        print(exc)
                        continue
                    if updated.action.value == "update":
                        print(
                            "已接受 UPDATE 候选，已应用更新到正式 Skill "
                            f"{updated.existing_skill_name}。"
                        )
                        print(f"Updated Skill: {updated.existing_skill_name}")
                    else:
                        print(
                            "已接受 CREATE 候选，已创建正式 Skill "
                            f"{updated.proposed_name}。"
                        )
                        print(f"Created Skill: {updated.proposed_name}")
                    if target is not None:
                        print(f"Path: {target}")
                    print(f"Status: {updated.status.value.upper()}")
                    continue
                if action_word == "reject":
                    try:
                        await skill_learning.reject(candidate.id)
                    except (KeyError, ValueError) as exc:
                        print(exc)
                        continue
                    print(f"已拒绝候选 {candidate.proposed_name}。")
                    continue
                print(f"未知操作：{action_word}（支持 accept / reject）")
                continue
            if content == "/clear":
                history = _initial_history(args.system)
                conversation = await conversation_store.replace_messages(
                    conversation.id,
                    history,
                )
                await summary_store.delete(conversation.id)
                print("上下文已清空。")
                continue
            if content == "/setup":
                print("模型设置在下次启动时生效。请退出后运行：")
                print("  .venv/bin/python -m app --setup")
                continue
            if content == "/help":
                print(_HELP_TEXT)
                continue

            _, conversation = await _send_message(
                conversation_service=conversation_service,
                conversation_store=conversation_store,
                conversation=conversation,
                provider=provider,
                history=history,
                content=content,
                model=model,
            )
            await _maybe_run_skill_learning(skill_learning)
    finally:
        await app.close()


# 函数说明：_parse_args
# 用途：解析`args`，供模型请求与响应处理使用。
# 返回：类型 `argparse.Namespace`；返回 `args`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`argparse.ArgumentParser` →
# `parser.add_argument` → `parser.parse_args` → `parser.error`。
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description="启动 MuHarness CLI，或完成首次模型设置。",
    )
    parser.add_argument(
        "-p",
        "--provider",
        choices=[provider.value for provider in ModelProvider],
        help="指定本次使用的模型 Provider；默认使用设置中的主模型。",
    )
    parser.add_argument("-m", "--model", help="临时覆盖模型名称。")
    parser.add_argument(
        "--message",
        help="发送一条消息后退出，不进入交互聊天。",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="启动交互式模型设置；Web 部署的密钥由服务端环境变量提供。",
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM_PROMPT,
        help="覆盖本次会话的系统提示词。",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="每次模型回复允许生成的最大 Token；默认使用 Provider 配置。",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=12,
        help="每条消息最多执行的模型/工具循环步数。",
    )
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=15,
        help="强制模型收口最终回答前允许的最大工具轮数。",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help="会话 SQLite 数据库路径。",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=DEFAULT_TASKS_DIR,
        help="持久化 Task JSON 文件目录。",
    )
    parser.add_argument(
        "--mcp-config",
        type=Path,
        default=DEFAULT_MCP_CONFIG_PATH,
        help="MCP Server JSON 配置文件路径。",
    )
    parser.add_argument(
        "--conversation",
        help="使用完整会话 ID 或唯一前缀恢复会话。",
    )
    parser.add_argument(
        "--new",
        "--new-conversation",
        dest="new_conversation",
        action="store_true",
        help="新建会话，不恢复最近会话。",
    )
    args = parser.parse_args()
    if args.max_output_tokens is not None and args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be greater than zero")
    if args.max_steps <= 0:
        parser.error("--max-steps must be greater than zero")
    if args.max_tool_rounds <= 0:
        parser.error("--max-tool-rounds must be greater than zero")
    if args.conversation and args.new_conversation:
        parser.error("--conversation and --new-conversation cannot be used together")
    if args.setup and (args.message or args.conversation or args.new_conversation):
        parser.error("--setup 不能和会话或单次消息参数同时使用")
    return args


# 函数说明：_main
# 用途：运行模型请求与响应处理入口，按照当前参数装配依赖并驱动主流程。
# 参数：
#   args：传给 `_run` 的输入，类型 `argparse.Namespace`。
# 返回：类型 `int`；按分支返回 `0`；`await _run(args)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`run_setup` → `_run`。
# 分支与异常：
#   当 `not should_start` 时，返回 `0`。
async def _main(args: argparse.Namespace) -> int:
    if args.setup:
        should_start = await run_setup()
        if not should_start:
            return 0
    return await _run(args)


# 函数说明：main
# 用途：运行模型请求与响应处理入口，按照当前参数装配依赖并驱动主流程。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SystemExit` → `asyncio.run` → `_main`
#  → `_parse_args`。
def main() -> None:
    raise SystemExit(asyncio.run(_main(_parse_args())))


if __name__ == "__main__":
    main()
