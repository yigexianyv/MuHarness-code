
from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.skill_learning import (
    SKILL_PROPOSE_TOOL_NAME,
    SkillCandidateOrigin,
    SkillCandidateStore,
    register_skill_learning_tools,
)
from app.domain.skills import SkillScope, SkillStore
from app.models.types import ToolCall
from app.tools.hooks import ToolExecutionContext
from app.tools.registry import ToolRegistry


# 函数说明：_environment
# 用途：在回归测试与测试辅助中处理 `_environment`，通过 `skill_store.initialize` 完成首
# 个内部处理步骤。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `tuple[ToolRegistry, SkillStore, SkillCandidateStore]`；返回
# `(registry, skill_store, candidate_store)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillStore` →
# `skill_store.initialize` → `SkillCandidateStore` → `candidate_store.initialize` →
# `ToolRegistry` → `register_skill_learning_tools`。
async def _environment(
    tmp_path: Path,
) -> tuple[ToolRegistry, SkillStore, SkillCandidateStore]:
    skill_store = SkillStore(
        tmp_path / "user-skills",
        tmp_path / "project-skills",
    )
    await skill_store.initialize()
    candidate_store = SkillCandidateStore(tmp_path / "learning")
    await candidate_store.initialize()
    registry = ToolRegistry()
    register_skill_learning_tools(registry, candidate_store, skill_store)
    return registry, skill_store, candidate_store


# 函数说明：_arguments
# 用途：返回 `{'action': action, 'name': name, 'description': '在 Python 修改完成后进行
# 结构化复查', 'reason': '用户明确要…`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   action：`action`输入或配置值，类型 `str`；默认 `'create'`。
#   name：目标对象、工具或配置项名称，类型 `str`；默认 `'review-python'`。
# 返回：类型 `dict`；字典，包含字段 `action`、`name`、`description`、`reason`、
# `procedure`、`pitfalls`、`verification`。
def _arguments(*, action: str = "create", name: str = "review-python") -> dict:
    return {
        "action": action,
        "name": name,
        "description": "在 Python 修改完成后进行结构化复查",
        "reason": "用户明确要求保存本轮已验证的复查方法",
        "procedure": ["检查变更范围", "运行针对性测试", "报告剩余风险"],
        "pitfalls": ["不要把测试通过等同于没有风险"],
        "verification": ["ruff 与 pytest 均通过"],
    }


# 函数说明：_context
# 用途：返回 `ToolExecutionContext(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   arguments：工具调用的参数对象或 JSON 文本，类型 `dict`。
# 返回：类型 `ToolExecutionContext`；返回 `ToolExecutionContext(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolExecutionContext` → `ToolCall`。
def _context(arguments: dict) -> ToolExecutionContext:
    return ToolExecutionContext(
        tool_call=ToolCall(
            id="call-skill-propose",
            name=SKILL_PROPOSE_TOOL_NAME,
            arguments=arguments,
        ),
        run_id="run-1",
        conversation_id="conversation-1",
    )


# 函数说明：test_skill_propose_creates_pending_candidate_only
# 用途：回归验证回归测试与测试辅助中的 `skill_propose_creates_pending_candidate_only` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_environment` → `_arguments` →
# `tool.execute_with_context` → `_context` → `skill_store.load`。
# 分支与异常：
#   验证条件：`output['status'] == 'pending'`。
#   验证条件：`output['requires_human_review'] is True`。
#   验证条件：`candidate is not None`。
#   验证条件：`candidate.origin is SkillCandidateOrigin.AGENT_PROPOSAL`。
@pytest.mark.asyncio
async def test_skill_propose_creates_pending_candidate_only(
    tmp_path: Path,
) -> None:
    registry, skill_store, candidate_store = await _environment(tmp_path)
    tool = registry.get(SKILL_PROPOSE_TOOL_NAME)
    arguments = _arguments()

    output = await tool.execute_with_context(arguments, _context(arguments))

    assert output["status"] == "pending"
    assert output["requires_human_review"] is True
    candidate = await candidate_store.get(output["candidate_id"])
    assert candidate is not None
    assert candidate.origin is SkillCandidateOrigin.AGENT_PROPOSAL
    assert candidate.source_run_ids == ("run-1",)
    assert candidate.source_conversation_id == "conversation-1"
    assert candidate.source_tool_call_id == "call-skill-propose"
    assert await skill_store.load("review-python") is None


# 函数说明：test_skill_propose_replay_is_idempotent
# 用途：回归验证回归测试与测试辅助中的 `skill_propose_replay_is_idempotent` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_environment` → `_arguments` →
# `_context` → `tool.execute_with_context` → `candidate_store.list`。
# 分支与异常：
#   验证条件：`second['candidate_id'] == first['candidate_id']`。
#   验证条件：`second['replayed'] is True`。
#   验证条件：`len(await candidate_store.list()) == 1`。
@pytest.mark.asyncio
async def test_skill_propose_replay_is_idempotent(tmp_path: Path) -> None:
    registry, _, candidate_store = await _environment(tmp_path)
    tool = registry.get(SKILL_PROPOSE_TOOL_NAME)
    arguments = _arguments()
    context = _context(arguments)

    first = await tool.execute_with_context(arguments, context)
    second = await tool.execute_with_context(arguments, context)

    assert second["candidate_id"] == first["candidate_id"]
    assert second["replayed"] is True
    assert len(await candidate_store.list()) == 1


# 函数说明：test_skill_propose_rejects_second_pending_target
# 用途：回归验证回归测试与测试辅助中的 `skill_propose_rejects_second_pending_target` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_environment` → `_arguments` →
# `tool.execute_with_context` → `_context` → `ToolExecutionContext` → `ToolCall`；另有 1
#  个调用点。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='already has pending candidate')`。
@pytest.mark.asyncio
async def test_skill_propose_rejects_second_pending_target(tmp_path: Path) -> None:
    registry, _, _ = await _environment(tmp_path)
    tool = registry.get(SKILL_PROPOSE_TOOL_NAME)
    arguments = _arguments()
    await tool.execute_with_context(arguments, _context(arguments))
    another_context = ToolExecutionContext(
        tool_call=ToolCall(
            id="another-call",
            name=SKILL_PROPOSE_TOOL_NAME,
            arguments=arguments,
        ),
        run_id="run-2",
        conversation_id="conversation-1",
    )

    with pytest.raises(ValueError, match="already has pending candidate"):
        await tool.execute_with_context(arguments, another_context)


# 函数说明：test_skill_propose_update_requires_existing_skill
# 用途：回归验证回归测试与测试辅助中的 `skill_propose_update_requires_existing_skill` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_environment` → `_arguments` →
# `pytest.raises` → `tool.execute_with_context` → `_context` → `skill_store.install`。
# 分支与异常：
#   验证条件：`output['action'] == 'update'`。
#   预期异常：`pytest.raises(ValueError, match='not found')`。
@pytest.mark.asyncio
async def test_skill_propose_update_requires_existing_skill(
    tmp_path: Path,
) -> None:
    registry, skill_store, _ = await _environment(tmp_path)
    tool = registry.get(SKILL_PROPOSE_TOOL_NAME)
    missing_arguments = _arguments(action="update")
    with pytest.raises(ValueError, match="not found"):
        await tool.execute_with_context(
            missing_arguments,
            _context(missing_arguments),
        )

    await skill_store.install(
        name="review-python",
        description="旧描述",
        instructions="# Review Python\n\n旧流程",
        scope=SkillScope.PROJECT,
    )
    output = await tool.execute_with_context(
        missing_arguments,
        _context(missing_arguments),
    )
    assert output["action"] == "update"


# 函数说明：test_skill_propose_requires_execution_context
# 用途：回归验证回归测试与测试辅助中的 `skill_propose_requires_execution_context` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_environment` → `pytest.raises` →
# `registry.get(SKILL_PROPOSE_TOOL_NAME).execute` → `_arguments`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='requires run and conversation')`。
@pytest.mark.asyncio
async def test_skill_propose_requires_execution_context(tmp_path: Path) -> None:
    registry, _, _ = await _environment(tmp_path)
    with pytest.raises(ValueError, match="requires run and conversation"):
        await registry.get(SKILL_PROPOSE_TOOL_NAME).execute(_arguments())


# 函数说明：test_skill_store_update_preserves_resources
# 用途：回归验证回归测试与测试辅助中的 `skill_store_update_preserves_resources` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_environment` → `skill_store.install`
#  → `references.mkdir` → `(references / 'checklist.md').write_text` →
# `skill_store.update` → `(updated.root / 'references' / 'checklist.md').read_text`。
# 分支与异常：
#   验证条件：`updated.metadata.description == '新描述'`。
#   验证条件：`'新流程' in updated.content`。
#   验证条件：`(updated.root / 'references' / 'checklist.md').read_text(encoding='utf-8'
# ) == '检查项'`。
# 副作用与资源：
#   文件或资源访问：`references.mkdir`、`(references / 'checklist.md').write_text`、
# `(updated.root / 'references' / 'checklist.md').read_text`。
@pytest.mark.asyncio
async def test_skill_store_update_preserves_resources(tmp_path: Path) -> None:
    _, skill_store, _ = await _environment(tmp_path)
    installed = await skill_store.install(
        name="review-python",
        description="旧描述",
        instructions="# Review Python\n\n旧流程",
    )
    references = installed.root / "references"
    references.mkdir()
    (references / "checklist.md").write_text("检查项", encoding="utf-8")

    updated = await skill_store.update(
        name="review-python",
        description="新描述",
        instructions="# Review Python\n\n新流程",
    )

    assert updated.metadata.description == "新描述"
    assert "新流程" in updated.content
    assert (updated.root / "references" / "checklist.md").read_text(
        encoding="utf-8"
    ) == "检查项"
