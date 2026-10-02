
from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from app.domain.skills import SkillStore
from app.models.types import ToolDefinition
from app.tools.base import BaseTool
from app.tools.hooks import ToolExecutionContext
from app.tools.registry import ToolRegistry

from .models import (
    SkillCandidate,
    SkillCandidateAction,
    SkillCandidateOrigin,
    SkillCandidateStatus,
)
from .store import SkillCandidateStore

SKILL_PROPOSE_TOOL_NAME = "skill_propose"


class SkillProposeTool(BaseTool):

    # 函数说明：SkillProposeTool.__init__
    # 用途：初始化 SkillProposeTool；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   candidate_store：候选持久化存储依赖，类型 `SkillCandidateStore`。
    #   skill_store：技能存储，类型 `SkillStore`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 副作用与资源：
    #   更新对象字段：`self._candidate_store`、`self._skill_store`。
    def __init__(
        self,
        candidate_store: SkillCandidateStore,
        skill_store: SkillStore,
    ) -> None:
        self._candidate_store = candidate_store
        self._skill_store = skill_store

    # 函数说明：SkillProposeTool.definition
    # 用途：提供 SkillProposeTool 的模型可见定义，包含名称、说明、参数结构及权限声明。
    # 返回：类型 `ToolDefinition`；返回 `ToolDefinition(…)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolDefinition`。
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=SKILL_PROPOSE_TOOL_NAME,
            record_output=False,
            description=(
                "为可复用操作流程保存创建/更新 Skill 的待审候选。仅当用户明确"
                "要求保存或更新已完成、已验证的方法时使用；不保存事实、偏好、"
                "密钥、未验证猜测或一次性任务内容。task_update 记录本次进展，"
                "skill_read 加载已有流程，本工具只提交新建议。已有正式同名 "
                "Skill 用 update，已有同名待审候选不要重复提出。首次保存返回 "
                "pending 候选及 requires_human_review=true，不修改或激活正式"
                "Skill；replayed=true 返回既有候选的当前状态，不是再次创建或审核通过。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "update"],
                        "description": "create 提议新增正式目录中不存在的名称；update 提议更新已存在的 Skill。两者都只生成待审候选。",
                    },
                    "name": {
                        "type": "string",
                        "description": (
                            "使用小写字母、数字和单连字符的 Skill 名称；update"
                            "须用已有精确名称，不以改名绕过同名待审候选检查。"
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": "非空的适用任务、触发条件与范围说明；描述可复用流程，不复述本次任务进度。",
                    },
                    "reason": {
                        "type": "string",
                        "description": "非空的复用价值和本轮验证依据；更新时说明已有流程为何需要修订，不编造经验。",
                    },
                    "procedure": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "至少一个按顺序执行的可复用步骤，包含前提和关键动作；仅提炼已验证的方法，不夹带密钥或一次性结果。",
                    },
                    "pitfalls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选的已观察到的错误、适用限制及规避方法；没有依据时省略，不补写猜测。",
                    },
                    "verification": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选的可执行验证方法和预期条件，便于复用后核对结果；提交候选不代表这些验证已执行。",
                    },
                },
                "required": [
                    "action",
                    "name",
                    "description",
                    "reason",
                    "procedure",
                ],
                "additionalProperties": False,
            },
            strict=False,
        )

    # 函数说明：SkillProposeTool.execute
    # 用途：执行SkillProposeTool，供技能候选提炼与审核使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`。
    # 返回：类型 `Any`；不返回结果值（隐式 None）。
    async def execute(self, arguments: dict[str, Any]) -> Any:
        raise ValueError("skill_propose requires run and conversation context")

    # 函数说明：SkillProposeTool.execute_with_context
    # 用途：执行上下文，供技能候选提炼与审核使用。
    # 参数：
    #   arguments：工具调用的参数对象或 JSON 文本，类型 `dict[str, Any]`；读取键
    # `action`、`name`、`description`、`reason`、`procedure`、`pitfalls`、`verification`
    # 。
    #   context：本次操作的上下文对象，类型 `ToolExecutionContext`。
    # 返回：类型 `dict[str, Any]`；按分支返回
    # `_candidate_output(existing_candidate, replayed=True)`；
    # `_candidate_output(candidate, replayed=False)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillCandidateAction` →
    # `self._skill_store.load` →
    # `sha256(f'agent-proposal:{run_id}:{tool_call_id}'.encode()).hexdigest` → `sha256`
    # → `f'agent-proposal:{run_id}:{tool_call_id}'.encode` → `_candidate_output`；另有 4
    #  个调用点。
    # 分支与异常：
    #   当 `not run_id or not conversation_id or (not tool_call_id)` 时，抛出
    # `ValueError(…)`。
    #   当 `action is SkillCandidateAction.CREATE and existing is not…` 时，抛出
    # `ValueError(…)`。
    #   当 `action is SkillCandidateAction.UPDATE and existing is None` 时，抛出
    # `ValueError(…)`。
    #   当 `existing_candidate is not None` 时，返回
    # `_candidate_output(existing_candidate, replayed=True)`。
    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        run_id = (context.run_id or "").strip()
        conversation_id = (context.conversation_id or "").strip()
        tool_call_id = context.tool_call.id.strip()
        if not run_id or not conversation_id or not tool_call_id:
            raise ValueError(
                "skill_propose requires run, conversation, and tool call context"
            )

        action = SkillCandidateAction(str(arguments.get("action", "")))
        name = str(arguments.get("name", "")).strip()
        existing = await self._skill_store.load(name)
        if action is SkillCandidateAction.CREATE and existing is not None:
            raise ValueError(f"skill '{name}' already exists; use action=update")
        if action is SkillCandidateAction.UPDATE and existing is None:
            raise ValueError(f"existing skill '{name}' not found; use action=create")

        candidate_id = sha256(
            f"agent-proposal:{run_id}:{tool_call_id}".encode()
        ).hexdigest()
        existing_candidate = await self._candidate_store.get(candidate_id)
        if existing_candidate is not None:
            return _candidate_output(existing_candidate, replayed=True)
        for pending in await self._candidate_store.list(
            status=SkillCandidateStatus.PENDING
        ):
            if pending.proposed_name == name:
                raise ValueError(
                    f"skill '{name}' already has pending candidate "
                    f"'{pending.id}'; review it before proposing another"
                )

        candidate = SkillCandidate(
            id=candidate_id,
            origin=SkillCandidateOrigin.AGENT_PROPOSAL,
            action=action,
            proposed_name=name,
            description=arguments.get("description"),
            reason=arguments.get("reason"),
            procedure=arguments.get("procedure"),
            pitfalls=arguments.get("pitfalls", ()),
            verification=arguments.get("verification", ()),
            source_run_ids=(run_id,),
            source_conversation_id=conversation_id,
            source_tool_call_id=tool_call_id,
            existing_skill_name=(
                name if action is SkillCandidateAction.UPDATE else None
            ),
            status=SkillCandidateStatus.PENDING,
            created_at=datetime.now(UTC),
            evidence_summary=(
                "由主 Agent 根据用户明确要求提出；正式生效前需要人工审核。"
            ),
        )
        await self._candidate_store.create(candidate)
        return _candidate_output(candidate, replayed=False)


# 函数说明：_candidate_output
# 用途：返回 `{'candidate_id': candidate.id, 'action': candidate.action.value, 'name':…`
# ，提供 技能候选提炼与审核 的派生值。
# 参数：
#   candidate：候选记录，类型 `SkillCandidate`。
#   replayed：`replayed`输入或配置值，类型 `bool`。
# 返回：类型 `dict[str, Any]`；字典，包含字段 `candidate_id`、`action`、`name`、`status`
# 、`requires_human_review`、`replayed`、`message`。
def _candidate_output(
    candidate: SkillCandidate,
    *,
    replayed: bool,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate.id,
        "action": candidate.action.value,
        "name": candidate.proposed_name,
        "status": candidate.status.value,
        "requires_human_review": True,
        "replayed": replayed,
        "message": "Skill 候选已保存，正式生效前需要用户审核。",
    }


# 函数说明：register_skill_learning_tools
# 用途：注册技能工具集合，供技能候选提炼与审核使用。
# 参数：
#   registry：工具、模型或能力注册表，类型 `ToolRegistry`。
#   candidate_store：候选持久化存储依赖，类型 `SkillCandidateStore`。
#   skill_store：技能存储，类型 `SkillStore`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`registry.register` →
# `SkillProposeTool`。
def register_skill_learning_tools(
    registry: ToolRegistry,
    candidate_store: SkillCandidateStore,
    skill_store: SkillStore,
) -> None:

    registry.register(SkillProposeTool(candidate_store, skill_store))


__all__ = [
    "SKILL_PROPOSE_TOOL_NAME",
    "SkillProposeTool",
    "register_skill_learning_tools",
]
