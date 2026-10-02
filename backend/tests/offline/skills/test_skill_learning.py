
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from app.domain.skill_learning import (
    SkillCandidate,
    SkillCandidateAction,
    SkillCandidateOrigin,
    SkillCandidateStatus,
    SkillCandidateStore,
    SkillLearningService,
    SkillLearningSettings,
    TaskCard,
    TraceEvidenceBuilder,
)
from app.domain.skills import SkillStore
from app.domain.task import FileTaskStore, TaskStatus, TaskStep, TaskStepStatus
from app.models.adapter import ModelAdapter
from app.models.config import ModelSettings, ProviderConfig
from app.models.registry import ModelAdapterRegistry
from app.models.types import (
    ApiStyle,
    Message,
    MessageRole,
    ModelResponse,
    ModelUsage,
    ToolCall,
    ToolResult,
)
from app.records.trace.store import SQLiteTraceStore
from app.runtime.agent.events import AgentEvent, AgentEventType


class _FakeAdapter(ModelAdapter):

    # 函数说明：_FakeAdapter.__init__
    # 用途：初始化 _FakeAdapter；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   config：运行配置，类型 `ProviderConfig`。
    #   responses：预设的模型或服务响应序列，类型 `list[ModelResponse | Exception]`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    # 副作用与资源：
    #   更新对象字段：`self.responses`、`self.requests`。
    def __init__(
        self,
        config: ProviderConfig,
        responses: list[ModelResponse | Exception],
    ) -> None:
        super().__init__(config)
        self.responses = list(responses)
        self.requests: list[Message] = []

    # 函数说明：_FakeAdapter.complete
    # 用途：完成_FakeAdapter，供回归测试与测试辅助使用。
    # 参数：
    #   request：待处理的请求对象。
    # 返回：类型 `ModelResponse`；返回 `response`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.responses.pop`。
    # 分支与异常：
    #   当 `isinstance(response, Exception)` 时，抛出 `response`。
    async def complete(self, request) -> ModelResponse:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    # 函数说明：_FakeAdapter.close
    # 用途：关闭_FakeAdapter，供回归测试与测试辅助使用。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def close(self) -> None:
        pass


# 函数说明：_model_response
# 用途：返回 `ModelResponse(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   content：内容正文，类型 `str`。
# 返回：类型 `ModelResponse`；返回 `ModelResponse(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelResponse` → `Message` →
# `ModelUsage`。
def _model_response(content: str) -> ModelResponse:
    return ModelResponse(
        id="fake-response",
        provider="fake",
        model="fake-model",
        message=Message(role=MessageRole.ASSISTANT, content=content),
        usage=ModelUsage(),
    )


# 函数说明：_fake_registry
# 用途：在回归测试与测试辅助中处理 `_fake_registry`，通过 `registry.register` 完成首个内
# 部处理步骤。
# 参数：
#   responses：预设的模型或服务响应序列，类型 `list[ModelResponse | Exception]`。
# 返回：类型 `tuple[ModelAdapterRegistry, _FakeAdapter]`；返回 `(registry, adapter)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ProviderConfig` → `SecretStr` →
# `_FakeAdapter` → `ModelAdapterRegistry` → `ModelSettings` → `registry.register`。
def _fake_registry(
    responses: list[ModelResponse | Exception],
) -> tuple[ModelAdapterRegistry, _FakeAdapter]:
    config = ProviderConfig(
        provider="fake",
        model="fake-model",
        api_key=SecretStr("offline-test-key"),
        api_style=ApiStyle.CHAT_COMPLETIONS,
    )
    adapter = _FakeAdapter(config, responses)
    registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
    registry.register("fake", lambda _: adapter, config=config)
    return registry, adapter




# 函数说明：_step
# 用途：返回 `TaskStep(id=f's{index}', title=title, status=TaskStepStatus.DONE, note='已
# 完成并验证')`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   index：当前位置或索引，类型 `int`。
#   title：面向用户的标题，类型 `str`。
# 返回：类型 `TaskStep`；返回 `TaskStep(id=f's{index}', title=title, status=
# TaskStepStatus.DONE, note='已完成并验证')`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`TaskStep`。
def _step(index: int, title: str) -> TaskStep:
    return TaskStep(
        id=f"s{index}",
        title=title,
        status=TaskStepStatus.DONE,
        note="已完成并验证",
    )


# 函数说明：_create_completed
# 用途：创建已完成项，供回归测试与测试辅助使用。
# 参数：
#   task_store：任务存储，类型 `FileTaskStore`。
#   title：面向用户的标题，类型 `str`。
#   goal：`goal`输入或配置值，类型 `str | None`；默认 `None`。
#   steps：任务步骤集合，类型 `tuple[str, ...]`；默认 `()`。
#   run_ids：待处理的运行标识集合，类型 `tuple[str, ...]`；默认 `()`。
# 返回：类型 `TaskStep`；返回
# `(await task_store.set_status(task.id, TaskStatus.COMPLETED)).id`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`task_store.create` → `_step` →
# `task_store.set_status`。
async def _create_completed(
    task_store: FileTaskStore,
    *,
    title: str,
    goal: str | None = None,
    steps: tuple[str, ...] = (),
    run_ids: tuple[str, ...] = (),
) -> TaskStep:
    task = await task_store.create(
        title=title,
        goal=goal,
        steps=tuple(
            _step(index, step_title) for index, step_title in enumerate(steps)
        ),
        owner_conversation_id="conv",
        run_ids=run_ids,
    )
    return (await task_store.set_status(task.id, TaskStatus.COMPLETED)).id  


# 函数说明：_tool_started
# 用途：返回 `AgentEvent(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   run_id：目标运行标识，类型 `str`。
#   seq：`seq`输入或配置值，类型 `int`。
#   name：目标对象、工具或配置项名称，类型 `str`。
#   arguments：工具调用的参数对象或 JSON 文本，类型 `dict`。
# 返回：类型 `AgentEvent`；返回 `AgentEvent(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`AgentEvent` → `ToolCall`。
def _tool_started(run_id: str, seq: int, name: str, arguments: dict) -> AgentEvent:
    return AgentEvent(
        run_id=run_id,
        conversation_id="conv",
        sequence=seq,
        type=AgentEventType.TOOL_STARTED,
        tool_call=ToolCall(id=f"c{seq}", name=name, arguments=arguments),
    )


# 函数说明：_tool_completed
# 用途：返回 `AgentEvent(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   run_id：目标运行标识，类型 `str`。
#   seq：`seq`输入或配置值，类型 `int`。
#   name：目标对象、工具或配置项名称，类型 `str`。
#   arguments：工具调用的参数对象或 JSON 文本，类型 `dict`。
#   success：执行是否成功，类型 `bool`；默认 `True`。
#   error：异常或错误信息，类型 `str | None`；默认 `None`。
# 返回：类型 `AgentEvent`；返回 `AgentEvent(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`AgentEvent` → `ToolCall` →
# `ToolResult`。
def _tool_completed(
    run_id: str,
    seq: int,
    name: str,
    arguments: dict,
    *,
    success: bool = True,
    error: str | None = None,
) -> AgentEvent:
    return AgentEvent(
        run_id=run_id,
        conversation_id="conv",
        sequence=seq,
        type=AgentEventType.TOOL_COMPLETED,
        tool_call=ToolCall(id=f"c{seq}", name=name, arguments=arguments),
        tool_result=ToolResult(
            tool_call_id=f"c{seq}",
            tool_name=name,
            success=success,
            error=error,
            duration_ms=0.0,
        ),
    )


# 函数说明：_record_trace
# 用途：记录执行轨迹，供回归测试与测试辅助使用。
# 参数：
#   trace_store：执行轨迹存储，类型 `SQLiteTraceStore`。
#   events：事件集合，类型 `tuple[AgentEvent, ...]`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`trace_store.record_event`。
async def _record_trace(
    trace_store: SQLiteTraceStore,
    events: tuple[AgentEvent, ...],
) -> None:
    for event in events:
        await trace_store.record_event(event)


# 函数说明：_make_env
# 用途：构造`env`，供回归测试与测试辅助使用。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
#   batch_size：`batch_size`输入或配置值，类型 `int`；默认 `20`。
# 返回：类型 `tuple[dict, Path]`；返回 `({'task_store': task_store, 'trace_store':
# trace_store, 'skill_store': skill_store, '…`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`FileTaskStore` →
# `task_store.initialize` → `SQLiteTraceStore` → `trace_store.initialize` → `SkillStore`
#  → `skill_store.initialize`；另有 2 个调用点。
async def _make_env(tmp_path: Path, *, batch_size: int = 20) -> tuple[dict, Path]:
    root = tmp_path / "env"
    task_store = FileTaskStore(root / "tasks")
    await task_store.initialize()
    trace_store = SQLiteTraceStore(root / "trace.db")
    await trace_store.initialize()
    skill_store = SkillStore(root / "user-skills", root / "project-skills")
    await skill_store.initialize()
    candidate_store = SkillCandidateStore(root / "data")
    await candidate_store.initialize()
    return {
        "task_store": task_store,
        "trace_store": trace_store,
        "skill_store": skill_store,
        "candidate_store": candidate_store,
    }, root


# 函数说明：_settings
# 用途：返回 `SkillLearningSettings(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
#   batch_size：`batch_size`输入或配置值，类型 `int`；默认 `20`。
#   min_cluster：`min_cluster`输入或配置值，类型 `int`；默认 `3`。
# 返回：类型 `SkillLearningSettings`；返回 `SkillLearningSettings(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillLearningSettings`。
def _settings(
    tmp_path: Path,
    *,
    batch_size: int = 20,
    min_cluster: int = 3,
) -> SkillLearningSettings:
    return SkillLearningSettings(
        _env_file=None,
        skill_learning_batch_size=batch_size,
        skill_learning_min_cluster_size=min_cluster,
        skill_learning_data_dir=tmp_path / "env" / "data",
    )




# 函数说明：test_task_card_from_completed_task
# 用途：回归验证回归测试与测试辅助中的 `task_card_from_completed_task` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `_create_completed` →
# `_to_card`。
# 分支与异常：
#   验证条件：`task is not None`。
#   验证条件：`card.task_id == task_id`。
#   验证条件：`card.title == '修复 Python ImportError'`。
#   验证条件：`card.goal == '恢复项目启动'`。
@pytest.mark.asyncio
async def test_task_card_from_completed_task(tmp_path: Path) -> None:
    env, _ = await _make_env(tmp_path)
    task_id = await _create_completed(
        env["task_store"],
        title="修复 Python ImportError",
        goal="恢复项目启动",
        steps=("复现", "读 traceback", "修复", "跑 pytest"),
        run_ids=("r1", "r2"),
    )
    task = await env["task_store"].get(task_id)
    assert task is not None

    from app.domain.skill_learning.service import _to_card

    card: TaskCard = _to_card(task)
    assert card.task_id == task_id
    assert card.title == "修复 Python ImportError"
    assert card.goal == "恢复项目启动"
    assert card.final_steps == ("复现", "读 traceback", "修复", "跑 pytest")
    assert card.run_count == 2
    assert not hasattr(card, "steps")


# 函数说明：test_task_card_only_built_from_completed
# 用途：回归验证回归测试与测试辅助中的 `task_card_only_built_from_completed` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` →
# `env['task_store'].create`。
# 分支与异常：
#   验证条件：`task.status is not TaskStatus.COMPLETED`。
@pytest.mark.asyncio
async def test_task_card_only_built_from_completed(tmp_path: Path) -> None:
    env, _ = await _make_env(tmp_path)
    task = await env["task_store"].create(
        title="进行中的任务",
        goal="目标",
        owner_conversation_id="conv",
    )
    assert task.status is not TaskStatus.COMPLETED




# 函数说明：test_trigger_requires_batch_size
# 用途：回归验证回归测试与测试辅助中的 `trigger_requires_batch_size` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `_fake_registry` →
# `_model_response` → `SkillLearningService` → `_settings` → `_create_completed`；另有 1
#  个调用点。
# 分支与异常：
#   验证条件：`outcome.triggered is False`。
#   验证条件：`outcome.skipped_reason == 'batch_not_ready'`。
#   验证条件：`outcome.pending_count == 19`。
#   验证条件：`outcome.triggered is True`。
@pytest.mark.asyncio
async def test_trigger_requires_batch_size(tmp_path: Path) -> None:
    env, root = await _make_env(tmp_path, batch_size=20)
    registry, _ = _fake_registry([_model_response('{"clusters": []}')])
    service = SkillLearningService(
        env["task_store"],
        env["trace_store"],
        env["skill_store"],
        env["candidate_store"],
        registry,
        settings=_settings(root, batch_size=20),
        default_provider="fake",
    )
    for index in range(19):
        await _create_completed(env["task_store"], title=f"任务{index}")
    outcome = await service.maybe_run_mining()
    assert outcome.triggered is False
    assert outcome.skipped_reason == "batch_not_ready"
    assert outcome.pending_count == 19
    await _create_completed(env["task_store"], title="任务20")
    outcome = await service.maybe_run_mining()
    assert outcome.triggered is True
    assert outcome.scanned_task_count == 20
    assert outcome.cluster_count == 0


# 函数说明：test_processed_tasks_not_counted_again
# 用途：回归验证回归测试与测试辅助中的 `processed_tasks_not_counted_again` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `_fake_registry` →
# `_model_response` → `SkillLearningService` → `_settings` → `_create_completed`；另有 1
#  个调用点。
# 分支与异常：
#   验证条件：`first.triggered is True`。
#   验证条件：`first.pending_count == 0`。
#   验证条件：`second.triggered is False`。
#   验证条件：`second.pending_count == 4`。
@pytest.mark.asyncio
async def test_processed_tasks_not_counted_again(tmp_path: Path) -> None:
    env, root = await _make_env(tmp_path, batch_size=5)
    registry, _ = _fake_registry([_model_response('{"clusters": []}')])
    service = SkillLearningService(
        env["task_store"],
        env["trace_store"],
        env["skill_store"],
        env["candidate_store"],
        registry,
        settings=_settings(root, batch_size=5),
        default_provider="fake",
    )
    for index in range(5):
        await _create_completed(env["task_store"], title=f"任务{index}")
    first = await service.maybe_run_mining()
    assert first.triggered is True
    assert first.pending_count == 0

    for index in range(4):
        await _create_completed(env["task_store"], title=f"新任务{index}")
    second = await service.maybe_run_mining()
    assert second.triggered is False
    assert second.pending_count == 4


# 函数说明：test_watermark_survives_restart
# 用途：回归验证回归测试与测试辅助中的 `watermark_survives_restart` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `_fake_registry` →
# `_model_response` → `SkillLearningService` → `_settings` → `_create_completed`；另有 3
#  个调用点。
# 分支与异常：
#   验证条件：`len(reloaded.processed_task_ids) == 5`。
#   验证条件：`outcome.triggered is False`。
#   验证条件：`outcome.pending_count == 0`。
@pytest.mark.asyncio
async def test_watermark_survives_restart(tmp_path: Path) -> None:
    env, root = await _make_env(tmp_path, batch_size=5)
    registry, _ = _fake_registry([_model_response('{"clusters": []}')])
    service = SkillLearningService(
        env["task_store"],
        env["trace_store"],
        env["skill_store"],
        env["candidate_store"],
        registry,
        settings=_settings(root, batch_size=5),
        default_provider="fake",
    )
    for index in range(5):
        await _create_completed(env["task_store"], title=f"任务{index}")
    await service.maybe_run_mining()

    reloaded = await env["candidate_store"].load_watermark()
    assert len(reloaded.processed_task_ids) == 5
    service2 = SkillLearningService(
        env["task_store"],
        env["trace_store"],
        env["skill_store"],
        env["candidate_store"],
        registry,
        settings=_settings(root, batch_size=5),
        default_provider="fake",
    )
    outcome = await service2.maybe_run_mining()
    assert outcome.triggered is False
    assert outcome.pending_count == 0




# 函数说明：test_no_cluster_no_candidate
# 用途：回归验证回归测试与测试辅助中的 `no_cluster_no_candidate` 场景，下方断言说明列出
# 实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `_fake_registry` →
# `_model_response` → `SkillLearningService` → `_settings` → `_create_completed`；另有 2
#  个调用点。
# 分支与异常：
#   验证条件：`outcome.triggered is True`。
#   验证条件：`outcome.cluster_count == 0`。
#   验证条件：`outcome.candidate_count == 0`。
#   验证条件：`outcome.pattern_mining_raw_output == '{"clusters": []}'`。
@pytest.mark.asyncio
async def test_no_cluster_no_candidate(tmp_path: Path) -> None:
    env, root = await _make_env(tmp_path, batch_size=3)
    registry, _ = _fake_registry([_model_response('{"clusters": []}')])
    service = SkillLearningService(
        env["task_store"],
        env["trace_store"],
        env["skill_store"],
        env["candidate_store"],
        registry,
        settings=_settings(root, batch_size=3),
        default_provider="fake",
    )
    for index in range(20):
        await _create_completed(env["task_store"], title=f"任务{index}")
    outcome = await service.maybe_run_mining()
    assert outcome.triggered is True
    assert outcome.cluster_count == 0
    assert outcome.candidate_count == 0
    assert outcome.pattern_mining_raw_output == '{"clusters": []}'
    assert await service.list_candidates() == ()


# 函数说明：test_similar_tasks_produce_cluster_and_candidate
# 用途：回归验证回归测试与测试辅助中的 `similar_tasks_produce_cluster_and_candidate` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `json.dumps` →
# `_fake_registry` → `_model_response` → `SkillLearningService` → `_settings`；另有 3 个
# 调用点。
# 分支与异常：
#   验证条件：`outcome.triggered is True`。
#   验证条件：`outcome.cluster_count == 1`。
#   验证条件：`outcome.candidate_count == 1`。
#   验证条件：`len(candidates) == 1`。
@pytest.mark.asyncio
async def test_similar_tasks_produce_cluster_and_candidate(tmp_path: Path) -> None:
    env, root = await _make_env(tmp_path, batch_size=3)
    cluster_json = json.dumps(
        {
            "clusters": [
                {
                    "id": "python-runtime-debug",
                    "task_ids": [],  
                    "pattern_name": "Python runtime debugging",
                    "description": "修复 Python 运行时错误",
                    "similarity_reason": "都是 ImportError/TypeError 排查",
                    "reusable_value": "多步骤排查流程可复用",
                }
            ]
        },
        ensure_ascii=False,
    )
    distill_json = json.dumps(
        {
            "action": "create",
            "proposed_name": "python-runtime-debug",
            "description": "排查 Python 运行时错误的标准流程",
            "reason": "多个相似任务证明该流程稳定",
            "procedure": ["复现", "读 traceback", "定位根因", "修复", "跑 pytest"],
            "pitfalls": ["不要跳过复现"],
            "verification": ["pytest 通过"],
        },
        ensure_ascii=False,
    )
    registry, _ = _fake_registry(
        [_model_response(cluster_json), _model_response(distill_json)]
    )
    service = SkillLearningService(
        env["task_store"],
        env["trace_store"],
        env["skill_store"],
        env["candidate_store"],
        registry,
        settings=_settings(root, batch_size=3),
        default_provider="fake",
    )
    task_ids: list[str] = []
    for index in range(3):
        task_id = await _create_completed(
            env["task_store"],
            title=f"修复 Python 报错{index}",
            goal="恢复运行",
            steps=("复现", "读 traceback", "修复", "跑 pytest"),
            run_ids=(f"r{index}",),
        )
        task_ids.append(task_id)
    registry, adapter = _fake_registry(
        [
            _model_response(
                json.dumps(
                    {
                        "clusters": [
                            {
                                "id": "python-runtime-debug",
                                "task_ids": task_ids,
                                "pattern_name": "Python runtime debugging",
                                "description": "修复 Python 运行时错误",
                                "similarity_reason": "都是 Python 报错排查",
                                "reusable_value": "多步骤排查流程可复用",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            ),
            _model_response(distill_json),
        ]
    )
    service = SkillLearningService(
        env["task_store"],
        env["trace_store"],
        env["skill_store"],
        env["candidate_store"],
        registry,
        settings=_settings(root, batch_size=3),
        default_provider="fake",
    )
    outcome = await service.maybe_run_mining()
    assert outcome.triggered is True
    assert outcome.cluster_count == 1
    assert outcome.candidate_count == 1
    candidates = await service.list_candidates()
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.proposed_name == "python-runtime-debug"
    assert candidate.source_task_ids == tuple(task_ids)
    assert candidate.action is SkillCandidateAction.CREATE
    assert candidate.reason
    assert candidate.procedure
    assert outcome.pattern_mining_raw_output is not None
    assert '"clusters"' in outcome.pattern_mining_raw_output
    assert len(outcome.distillations) == 1
    assert outcome.distillations[0].raw_output == distill_json
    assert len(adapter.requests) == 2




# 函数说明：test_evidence_builds_from_trace
# 用途：回归验证回归测试与测试辅助中的 `evidence_builds_from_trace` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `_create_completed` →
# `_record_trace` → `_tool_started` → `_tool_completed` →
# `env['trace_store'].load_events`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`task is not None`。
#   验证条件：`'read_file' in text`。
#   验证条件：`'失败工具调用' in text`。
#   验证条件：`'run_pytest' in text`。
@pytest.mark.asyncio
async def test_evidence_builds_from_trace(tmp_path: Path) -> None:
    env, _ = await _make_env(tmp_path)
    task_id = await _create_completed(
        env["task_store"],
        title="修复报错",
        steps=("复现", "修复"),
        run_ids=("r1",),
    )
    task = await env["task_store"].get(task_id)
    assert task is not None
    await _record_trace(
        env["trace_store"],
        (
            _tool_started("r1", 1, "read_file", {"path": "x.py"}),
            _tool_completed("r1", 2, "read_file", {"path": "x.py"}),
            _tool_started("r1", 3, "run_pytest", {}),
            _tool_completed(
                "r1",
                4,
                "run_pytest",
                {},
                success=False,
                error="AssertionError",
            ),
            _tool_started("r1", 5, "task_update", {"goal": "新目标"}),
            _tool_completed("r1", 6, "task_update", {"goal": "新目标"}),
        ),
    )
    events = await env["trace_store"].load_events("r1")
    builder = TraceEvidenceBuilder(_settings(tmp_path))
    text = builder.build(task, events)
    assert "read_file" in text
    assert "失败工具调用" in text
    assert "run_pytest" in text
    assert "task_update" in text


# 函数说明：test_evidence_degrades_gracefully_when_trace_missing
# 用途：回归验证回归测试与测试辅助中的 `evidence_degrades_gracefully_when_trace_missing`
#  场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `_create_completed` →
# `service_probe_load_events` → `TraceEvidenceBuilder` → `_settings` → `builder.build`。
# 分支与异常：
#   验证条件：`task is not None`。
#   验证条件：`events == ()`。
#   验证条件：`'没有可用的 Trace 事件' in text`。
#   验证条件：`task.title in text`。
@pytest.mark.asyncio
async def test_evidence_degrades_gracefully_when_trace_missing(
    tmp_path: Path,
) -> None:
    env, _ = await _make_env(tmp_path)
    task_id = await _create_completed(
        env["task_store"],
        title="无 Trace 的任务",
        steps=("步骤A",),
        run_ids=("missing-run",),
    )
    task = await env["task_store"].get(task_id)
    assert task is not None
    events = await service_probe_load_events(env["trace_store"], task)
    assert events == ()
    builder = TraceEvidenceBuilder(_settings(tmp_path))
    text = builder.build(task, events)
    assert "没有可用的 Trace 事件" in text
    assert task.title in text


# 函数说明：service_probe_load_events
# 用途：加载事件序列，供回归测试与测试辅助使用。
# 参数：
#   trace_store：执行轨迹存储，类型 `SQLiteTraceStore`。
#   task：当前任务记录。
# 返回：类型 `tuple`；返回 `tuple(events)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`trace_store.load_events`。
# 分支与异常：
#   捕获 `(KeyError, ValueError, OSError)` 后，跳过当前循环项，继续处理后续项。
async def service_probe_load_events(
    trace_store: SQLiteTraceStore,
    task,
) -> tuple:

    events: list = []
    for run_id in task.run_ids:
        try:
            loaded = await trace_store.load_events(run_id)
        except (KeyError, ValueError, OSError):
            continue
        events.extend(loaded)
    return tuple(events)




# 函数说明：test_candidate_fields_and_duplicate_suppression
# 用途：回归验证回归测试与测试辅助中的 `candidate_fields_and_duplicate_suppression` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `_fake_registry` →
# `SkillLearningService` → `_settings` → `SkillCandidate` → `uuid4`；另有 4 个调用点。
# 分支与异常：
#   验证条件：`dup is not None and dup.id == candidate.id`。
#   验证条件：`await service.candidate_store.find_duplicate_source(('x', 'y')) is None`
# 。
#   预期异常：`pytest.raises(ValueError)`。
@pytest.mark.asyncio
async def test_candidate_fields_and_duplicate_suppression(tmp_path: Path) -> None:
    env, root = await _make_env(tmp_path)
    registry, _ = _fake_registry([])
    service = SkillLearningService(
        env["task_store"],
        env["trace_store"],
        env["skill_store"],
        env["candidate_store"],
        registry,
        settings=_settings(root),
        default_provider="fake",
    )
    from datetime import UTC, datetime
    from uuid import uuid4

    candidate = SkillCandidate(
        id=uuid4().hex,
        action=SkillCandidateAction.CREATE,
        proposed_name="python-runtime-debug",
        description="描述",
        reason="原因",
        procedure=("复现", "修复"),
        source_task_ids=("a", "b", "c"),
        source_run_ids=("r1", "r2"),
        created_at=datetime.now(UTC),
    )
    await service.candidate_store.create(candidate)

    dup = await service.candidate_store.find_duplicate_source(("c", "a", "b"))
    assert dup is not None and dup.id == candidate.id
    assert await service.candidate_store.find_duplicate_source(("x", "y")) is None

    with pytest.raises(ValueError):
        SkillCandidate(
            id=uuid4().hex,
            action=SkillCandidateAction.UPDATE,
            proposed_name="debug-python",
            description="描述",
            reason="原因",
            procedure=("步骤",),
            source_task_ids=("a",),
            created_at=datetime.now(UTC),
        )




# 函数说明：test_pending_candidate_not_visible_to_skill_runtime
# 用途：回归验证回归测试与测试辅助中的 `pending_candidate_not_visible_to_skill_runtime`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `_fake_registry` →
# `SkillLearningService` → `_settings` → `SkillCandidate` → `uuid4`；另有 4 个调用点。
# 分支与异常：
#   验证条件：`all((item.name != 'pending-skill' for item in catalog))`。
#   验证条件：`await env['skill_store'].load('pending-skill') is None`。
@pytest.mark.asyncio
async def test_pending_candidate_not_visible_to_skill_runtime(
    tmp_path: Path,
) -> None:
    env, root = await _make_env(tmp_path)
    registry, _ = _fake_registry([])
    service = SkillLearningService(
        env["task_store"],
        env["trace_store"],
        env["skill_store"],
        env["candidate_store"],
        registry,
        settings=_settings(root),
        default_provider="fake",
    )
    from datetime import UTC, datetime
    from uuid import uuid4

    candidate = SkillCandidate(
        id=uuid4().hex,
        action=SkillCandidateAction.CREATE,
        proposed_name="pending-skill",
        description="描述",
        reason="原因",
        procedure=("步骤",),
        source_task_ids=("a", "b", "c"),
        created_at=datetime.now(UTC),
    )
    await service.candidate_store.create(candidate)
    catalog = await env["skill_store"].catalog()
    assert all(item.name != "pending-skill" for item in catalog)
    assert await env["skill_store"].load("pending-skill") is None


# 函数说明：test_accept_creates_skill_and_reject_does_not
# 用途：回归验证回归测试与测试辅助中的 `accept_creates_skill_and_reject_does_not` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `_fake_registry` →
# `SkillLearningService` → `_settings` → `SkillCandidate` → `uuid4`；另有 7 个调用点。
# 分支与异常：
#   验证条件：`await env['skill_store'].load('rejected-skill') is None`。
#   验证条件：
# `rejected is not None and rejected.status is SkillCandidateStatus.REJECTED`。
#   验证条件：`updated.status is SkillCandidateStatus.ACCEPTED`。
#   验证条件：`target is not None and target.name == 'SKILL.md'`。
@pytest.mark.asyncio
async def test_accept_creates_skill_and_reject_does_not(
    tmp_path: Path,
) -> None:
    env, root = await _make_env(tmp_path)
    registry, _ = _fake_registry([])
    service = SkillLearningService(
        env["task_store"],
        env["trace_store"],
        env["skill_store"],
        env["candidate_store"],
        registry,
        settings=_settings(root),
        default_provider="fake",
    )
    from datetime import UTC, datetime
    from uuid import uuid4

    accept_candidate = SkillCandidate(
        id=uuid4().hex,
        action=SkillCandidateAction.CREATE,
        proposed_name="accepted-skill",
        description="被接受的技能",
        reason="原因",
        procedure=("步骤A", "步骤B"),
        source_task_ids=("a", "b", "c"),
        created_at=datetime.now(UTC),
    )
    reject_candidate = SkillCandidate(
        id=uuid4().hex,
        action=SkillCandidateAction.CREATE,
        proposed_name="rejected-skill",
        description="被拒绝的技能",
        reason="原因",
        procedure=("步骤A",),
        source_task_ids=("d", "e", "f"),
        created_at=datetime.now(UTC),
    )
    await service.candidate_store.create(accept_candidate)
    await service.candidate_store.create(reject_candidate)

    await service.reject(reject_candidate.id)
    assert await env["skill_store"].load("rejected-skill") is None
    rejected = await service.get_candidate(reject_candidate.id)
    assert rejected is not None and rejected.status is SkillCandidateStatus.REJECTED

    updated, target = await service.accept(accept_candidate.id)
    assert updated.status is SkillCandidateStatus.ACCEPTED
    assert target is not None and target.name == "SKILL.md"
    skill = await env["skill_store"].load("accepted-skill")
    assert skill is not None
    assert "步骤A" in skill.content
    catalog = await env["skill_store"].catalog()
    assert any(item.name == "accepted-skill" for item in catalog)




# 函数说明：_task_update_events
# 用途：更新事件序列，供回归测试与测试辅助使用。
# 参数：
#   run_id：目标运行标识，类型 `str`。
#   arguments：工具调用的参数对象或 JSON 文本，类型 `dict`。
# 返回：类型 `tuple[AgentEvent, ...]`；返回
# `(_tool_started(run_id, 1, 'task_update', arguments), _tool_completed(run_id, 2, '…`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tool_started` → `_tool_completed`。
def _task_update_events(run_id: str, arguments: dict) -> tuple[AgentEvent, ...]:
    return (
        _tool_started(run_id, 1, "task_update", arguments),
        _tool_completed(run_id, 2, "task_update", arguments),
    )


# 函数说明：_build_evidence
# 用途：构建原始证据，供回归测试与测试辅助使用。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
#   arguments：工具调用的参数对象或 JSON 文本，类型 `dict`。
# 返回：类型 `str`；返回 `builder.build(task, events)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `_create_completed` →
# `_task_update_events` → `TraceEvidenceBuilder` → `_settings` → `builder.build`。
# 分支与异常：
#   验证条件：`task is not None`。
async def _build_evidence(tmp_path: Path, arguments: dict) -> str:
    env, _ = await _make_env(tmp_path)
    task_id = await _create_completed(
        env["task_store"],
        title="修复报错",
        run_ids=("r1",),
    )
    task = await env["task_store"].get(task_id)
    assert task is not None
    events = _task_update_events("r1", arguments)
    builder = TraceEvidenceBuilder(_settings(tmp_path))
    return builder.build(task, events)


# 函数说明：test_evidence_sees_constraints_content
# 用途：回归验证回归测试与测试辅助中的 `evidence_sees_constraints_content` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_build_evidence`。
# 分支与异常：
#   验证条件：`'constraints added: 不要重装全部依赖' in text`。
#   验证条件：`'使用现有 .venv' in text`。
@pytest.mark.asyncio
async def test_evidence_sees_constraints_content(tmp_path: Path) -> None:
    text = await _build_evidence(
        tmp_path,
        {"task_id": "x", "constraints": ["不要重装全部依赖", "使用现有 .venv"]},
    )
    assert "constraints added: 不要重装全部依赖" in text
    assert "使用现有 .venv" in text


# 函数说明：test_evidence_sees_facts_content
# 用途：回归验证回归测试与测试辅助中的 `evidence_sees_facts_content` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_build_evidence`。
# 分支与异常：
#   验证条件：`'facts added: 项目实际使用 .venv' in text`。
#   验证条件：`'CI 依赖缓存' in text`。
@pytest.mark.asyncio
async def test_evidence_sees_facts_content(tmp_path: Path) -> None:
    text = await _build_evidence(
        tmp_path,
        {"task_id": "x", "facts": ["项目实际使用 .venv", "CI 依赖缓存"]},
    )
    assert "facts added: 项目实际使用 .venv" in text
    assert "CI 依赖缓存" in text


# 函数说明：test_evidence_sees_state_content
# 用途：回归验证回归测试与测试辅助中的 `evidence_sees_state_content` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_build_evidence`。
# 分支与异常：
#   验证条件：`'state replaced: 已定位 import path' in text`。
#   验证条件：`'等待运行 pytest' in text`。
@pytest.mark.asyncio
async def test_evidence_sees_state_content(tmp_path: Path) -> None:
    text = await _build_evidence(
        tmp_path,
        {"task_id": "x", "state": ["已定位 import path", "等待运行 pytest"]},
    )
    assert "state replaced: 已定位 import path" in text
    assert "等待运行 pytest" in text


# 函数说明：test_evidence_sees_replacement_steps
# 用途：回归验证回归测试与测试辅助中的 `evidence_sees_replacement_steps` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_build_evidence`。
# 分支与异常：
#   验证条件：`'plan replaced:' in text`。
#   验证条件：`'- 检查 virtualenv' in text`。
#   验证条件：`'- 读取 traceback' in text`。
#   验证条件：`'- 运行 pytest' in text`。
@pytest.mark.asyncio
async def test_evidence_sees_replacement_steps(tmp_path: Path) -> None:
    text = await _build_evidence(
        tmp_path,
        {
            "task_id": "x",
            "steps": [
                {"title": "检查 virtualenv"},
                {"title": "读取 traceback"},
                {"title": "定位 import path"},
                {"title": "运行 pytest"},
            ],
        },
    )
    assert "plan replaced:" in text
    assert "- 检查 virtualenv" in text
    assert "- 读取 traceback" in text
    assert "- 运行 pytest" in text


# 函数说明：test_evidence_sees_step_progress_and_note
# 用途：回归验证回归测试与测试辅助中的 `evidence_sees_step_progress_and_note` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_build_evidence`。
# 分支与异常：
#   验证条件：`'step s2 -> done: 已确认 .venv 存在，pytest 通过' in text`。
@pytest.mark.asyncio
async def test_evidence_sees_step_progress_and_note(tmp_path: Path) -> None:
    text = await _build_evidence(
        tmp_path,
        {
            "task_id": "x",
            "step_id": "s2",
            "step_status": "done",
            "step_note": "已确认 .venv 存在，pytest 通过",
        },
    )
    assert "step s2 -> done: 已确认 .venv 存在，pytest 通过" in text


# 函数说明：test_evidence_combined_update_keeps_key_fields
# 用途：回归验证回归测试与测试辅助中的 `evidence_combined_update_keeps_key_fields` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_build_evidence`。
# 分支与异常：
#   验证条件：`'goal: 恢复 CI 全绿' in text`。
#   验证条件：`'constraints added: 不要跳过 virtualenv 确认' in text`。
#   验证条件：`'facts added: 项目使用 .venv' in text`。
@pytest.mark.asyncio
async def test_evidence_combined_update_keeps_key_fields(tmp_path: Path) -> None:
    text = await _build_evidence(
        tmp_path,
        {
            "task_id": "x",
            "goal": "恢复 CI 全绿",
            "constraints": ["不要跳过 virtualenv 确认"],
            "facts": ["项目使用 .venv"],
            "steps": [{"title": "复现"}, {"title": "修复"}],
            "step_id": "s1",
            "step_status": "done",
            "step_note": "已复现",
        },
    )
    assert "goal: 恢复 CI 全绿" in text
    assert "constraints added: 不要跳过 virtualenv 确认" in text
    assert "facts added: 项目使用 .venv" in text




# 函数说明：test_mining_failure_keeps_batch_inflight
# 用途：回归验证回归测试与测试辅助中的 `mining_failure_keeps_batch_inflight` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `_fake_registry` →
# `SkillLearningService` → `_settings` → `_create_completed` →
# `service.maybe_run_mining`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`outcome.triggered is True`。
#   验证条件：`outcome.error is not None`。
#   验证条件：`watermark.inflight is not None`。
#   验证条件：`len(watermark.inflight.task_ids) == 20`。
@pytest.mark.asyncio
async def test_mining_failure_keeps_batch_inflight(tmp_path: Path) -> None:
    env, root = await _make_env(tmp_path, batch_size=20)
    registry, _ = _fake_registry([TimeoutError("model timeout")])
    service = SkillLearningService(
        env["task_store"],
        env["trace_store"],
        env["skill_store"],
        env["candidate_store"],
        registry,
        settings=_settings(root, batch_size=20),
        default_provider="fake",
    )
    for index in range(20):
        await _create_completed(env["task_store"], title=f"任务{index}")

    outcome = await service.maybe_run_mining()
    assert outcome.triggered is True
    assert outcome.error is not None

    watermark = await env["candidate_store"].load_watermark()
    assert watermark.inflight is not None
    assert len(watermark.inflight.task_ids) == 20
    assert not set(watermark.inflight.task_ids) & set(watermark.processed_task_ids)


# 函数说明：test_inflight_survives_restart_and_retry_succeeds
# 用途：回归验证回归测试与测试辅助中的 `inflight_survives_restart_and_retry_succeeds` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `_fake_registry` →
# `SkillLearningService` → `_settings` → `_create_completed` →
# `service.maybe_run_mining`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`watermark.inflight is not None`。
#   验证条件：`watermark.inflight.attempt == 1`。
#   验证条件：`reloaded.inflight is not None`。
#   验证条件：`outcome.triggered is True`。
@pytest.mark.asyncio
async def test_inflight_survives_restart_and_retry_succeeds(tmp_path: Path) -> None:
    env, root = await _make_env(tmp_path, batch_size=5)
    registry_fail, _ = _fake_registry([TimeoutError("timeout")])
    service = SkillLearningService(
        env["task_store"],
        env["trace_store"],
        env["skill_store"],
        env["candidate_store"],
        registry_fail,
        settings=_settings(root, batch_size=5),
        default_provider="fake",
    )
    for index in range(5):
        await _create_completed(env["task_store"], title=f"任务{index}")
    await service.maybe_run_mining()
    watermark = await env["candidate_store"].load_watermark()
    assert watermark.inflight is not None
    assert watermark.inflight.attempt == 1

    reloaded = await env["candidate_store"].load_watermark()
    assert reloaded.inflight is not None

    registry_ok, _ = _fake_registry([_model_response('{"clusters": []}')])
    service2 = SkillLearningService(
        env["task_store"],
        env["trace_store"],
        env["skill_store"],
        env["candidate_store"],
        registry_ok,
        settings=_settings(root, batch_size=5),
        default_provider="fake",
    )
    outcome = await service2.maybe_run_mining()
    assert outcome.triggered is True
    assert outcome.error is None
    watermark2 = await env["candidate_store"].load_watermark()
    assert watermark2.inflight is None
    assert len(watermark2.processed_task_ids) == 5
    outcome_again = await service2.maybe_run_mining()
    assert outcome_again.triggered is False


# 函数说明：test_invalid_json_mining_keeps_batch
# 用途：回归验证回归测试与测试辅助中的 `invalid_json_mining_keeps_batch` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `_fake_registry` →
# `_model_response` → `SkillLearningService` → `_settings` → `_create_completed`；另有 2
#  个调用点。
# 分支与异常：
#   验证条件：`outcome.triggered is True`。
#   验证条件：`outcome.error is not None`。
#   验证条件：`watermark.inflight is not None`。
#   验证条件：`not set(watermark.inflight.task_ids) & set(watermark.processed_task_ids)`
# 。
@pytest.mark.asyncio
async def test_invalid_json_mining_keeps_batch(tmp_path: Path) -> None:
    env, root = await _make_env(tmp_path, batch_size=5)
    registry, _ = _fake_registry([_model_response("not a json payload")])
    service = SkillLearningService(
        env["task_store"],
        env["trace_store"],
        env["skill_store"],
        env["candidate_store"],
        registry,
        settings=_settings(root, batch_size=5),
        default_provider="fake",
    )
    for index in range(5):
        await _create_completed(env["task_store"], title=f"任务{index}")
    outcome = await service.maybe_run_mining()
    assert outcome.triggered is True
    assert outcome.error is not None
    watermark = await env["candidate_store"].load_watermark()
    assert watermark.inflight is not None
    assert not set(watermark.inflight.task_ids) & set(watermark.processed_task_ids)


# 函数说明：test_mining_gives_up_after_max_attempts
# 用途：回归验证回归测试与测试辅助中的 `mining_gives_up_after_max_attempts` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `_fake_registry` →
# `_settings` → `SkillLearningSettings` → `SkillLearningService` → `_create_completed`；
# 另有 2 个调用点。
# 分支与异常：
#   验证条件：`first.error is not None`。
#   验证条件：`watermark.inflight is not None and watermark.inflight.attempt == 1`。
#   验证条件：`second.error is not None`。
#   验证条件：`watermark2.inflight is None`。
@pytest.mark.asyncio
async def test_mining_gives_up_after_max_attempts(tmp_path: Path) -> None:
    env, root = await _make_env(tmp_path, batch_size=5)
    registry, _ = _fake_registry([TimeoutError("t1")])
    settings = _settings(root, batch_size=5)
    settings = SkillLearningSettings(
        _env_file=None,
        skill_learning_batch_size=5,
        skill_learning_min_cluster_size=3,
        skill_learning_max_attempts=2,
        skill_learning_data_dir=root / "env" / "data",
    )
    service = SkillLearningService(
        env["task_store"],
        env["trace_store"],
        env["skill_store"],
        env["candidate_store"],
        registry,
        settings=settings,
        default_provider="fake",
    )
    for index in range(5):
        await _create_completed(env["task_store"], title=f"任务{index}")
    first = await service.maybe_run_mining()
    assert first.error is not None
    watermark = await env["candidate_store"].load_watermark()
    assert watermark.inflight is not None and watermark.inflight.attempt == 1

    second = await service.maybe_run_mining()
    assert second.error is not None
    watermark2 = await env["candidate_store"].load_watermark()
    assert watermark2.inflight is None
    assert len(watermark2.processed_task_ids) == 5




# 函数说明：_mining_cluster_json
# 用途：返回 `json.dumps(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   task_ids：任务输入或配置值，类型 `list[str]`。
# 返回：类型 `str`；返回 `json.dumps(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.dumps`。
def _mining_cluster_json(task_ids: list[str]) -> str:
    return json.dumps(
        {
            "clusters": [
                {
                    "id": "python-debug",
                    "task_ids": task_ids,
                    "pattern_name": "Python runtime debugging",
                    "description": "修复 Python 运行时错误",
                    "similarity_reason": "均为 Python 报错排查",
                    "reusable_value": "多步骤流程可复用",
                }
            ]
        },
        ensure_ascii=False,
    )


_DISTILL_CREATE_JSON = json.dumps(
    {
        "action": "create",
        "proposed_name": "python-runtime-debug",
        "description": "排查 Python 运行时错误",
        "reason": "多个相似任务证明流程稳定",
        "procedure": ["复现", "读 traceback", "修复", "验证"],
        "pitfalls": ["不要跳过复现"],
        "verification": ["pytest 通过"],
    },
    ensure_ascii=False,
)
_DISTILL_NONE_JSON = json.dumps(
    {"action": "none", "reason": "被 pending candidate 覆盖"},
    ensure_ascii=False,
)
_DISTILL_UPDATE_JSON = json.dumps(
    {
        "action": "update",
        "proposed_name": None,
        "description": "补充 virtualenv 确认",
        "reason": "新证据证明应先确认 virtualenv",
        "procedure": ["复现", "确认 virtualenv", "修复", "验证"],
        "pitfalls": [],
        "verification": ["pytest 通过"],
        "existing_skill_name": "debug-python",
    },
    ensure_ascii=False,
)


# 函数说明：_run_batch
# 用途：运行`batch`，供回归测试与测试辅助使用。
# 参数：
#   env：环境变量映射；读取键 `task_store`、`skill_store`、`trace_store`、
# `candidate_store`。
#   root：当前操作的根目录。
#   task_count：传给 `range` 的输入，类型 `int`；默认 `3`。
#   batch_size：`batch_size`输入或配置值，类型 `int`；默认 `3`。
#   distill_response：传给 `_model_response` 的输入，类型 `str | None`；默认 `None`。
# 返回：返回 `(service, outcome, task_ids)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_create_completed` →
# `_model_response` → `_mining_cluster_json` → `env['skill_store'].catalog` →
# `_fake_registry` → `SkillLearningService`；另有 2 个调用点。
async def _run_batch(
    env,
    root,
    *,
    task_count: int = 3,
    batch_size: int = 3,
    distill_response: str | None = None,
):

    task_ids: list[str] = []
    for index in range(task_count):
        task_id = await _create_completed(
            env["task_store"],
            title=f"Python 报错{index}",
            steps=("复现", "读 traceback", "修复"),
            run_ids=(f"r{index}",),
        )
        task_ids.append(task_id)
    responses: list = [_model_response(_mining_cluster_json(task_ids))]
    if distill_response is not None:
        catalog = await env["skill_store"].catalog()
        if catalog:
            responses.append(
                _model_response('{"related_skills": ["debug-python"]}')
            )
        responses.append(_model_response(distill_response))
    registry, _ = _fake_registry(responses)
    service = SkillLearningService(
        env["task_store"],
        env["trace_store"],
        env["skill_store"],
        env["candidate_store"],
        registry,
        settings=_settings(root, batch_size=batch_size),
        default_provider="fake",
    )
    outcome = await service.maybe_run_mining()
    return service, outcome, task_ids


# 函数说明：service_candidates
# 用途：返回 `await env['candidate_store'].list()`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   env：环境变量映射；读取键 `candidate_store`。
# 返回：类型 `tuple`；返回 `await env['candidate_store'].list()`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`env['candidate_store'].list`。
async def service_candidates(env) -> tuple:
    return await env["candidate_store"].list()


# 函数说明：test_pending_candidate_blocks_exact_name_duplicate
# 用途：回归验证回归测试与测试辅助中的 `pending_candidate_blocks_exact_name_duplicate`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `SkillCandidate` →
# `uuid4` → `datetime.now` → `env['candidate_store'].create` → `_run_batch`；另有 1 个调
# 用点。
# 分支与异常：
#   验证条件：`outcome.candidate_count == 0`。
#   验证条件：`len(await service_candidates(env)) == 1`。
@pytest.mark.asyncio
async def test_pending_candidate_blocks_exact_name_duplicate(tmp_path: Path) -> None:
    env, root = await _make_env(tmp_path)
    from datetime import UTC, datetime
    from uuid import uuid4

    pending = SkillCandidate(
        id=uuid4().hex,
        action=SkillCandidateAction.CREATE,
        proposed_name="python-runtime-debug",
        description="已待评审",
        reason="原因",
        procedure=("复现",),
        source_task_ids=("old-a", "old-b", "old-c"),
        created_at=datetime.now(UTC),
    )
    await env["candidate_store"].create(pending)

    _, outcome, _ = await _run_batch(
        env,
        root,
        distill_response=_DISTILL_CREATE_JSON,
    )
    assert outcome.candidate_count == 0
    assert len(await service_candidates(env)) == 1


# 函数说明：test_pending_candidate_semantic_cover_returns_none
# 用途：回归验证回归测试与测试辅助中的 `pending_candidate_semantic_cover_returns_none`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `SkillCandidate` →
# `uuid4` → `datetime.now` → `env['candidate_store'].create` → `_run_batch`；另有 1 个调
# 用点。
# 分支与异常：
#   验证条件：`outcome.candidate_count == 0`。
#   验证条件：`len(await service_candidates(env)) == 1`。
@pytest.mark.asyncio
async def test_pending_candidate_semantic_cover_returns_none(tmp_path: Path) -> None:
    env, root = await _make_env(tmp_path)
    from datetime import UTC, datetime
    from uuid import uuid4

    pending = SkillCandidate(
        id=uuid4().hex,
        action=SkillCandidateAction.CREATE,
        proposed_name="debug-python-runtime",
        description="待评审",
        reason="原因",
        procedure=("复现",),
        source_task_ids=("x", "y", "z"),
        created_at=datetime.now(UTC),
    )
    await env["candidate_store"].create(pending)

    _, outcome, _ = await _run_batch(
        env,
        root,
        distill_response=_DISTILL_NONE_JSON,
    )
    assert outcome.candidate_count == 0
    assert len(await service_candidates(env)) == 1


# 函数说明：test_rejected_candidate_does_not_block_future
# 用途：回归验证回归测试与测试辅助中的 `rejected_candidate_does_not_block_future` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `SkillCandidate` →
# `uuid4` → `datetime.now` → `env['candidate_store'].create` →
# `env['candidate_store'].update`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`outcome.candidate_count == 1`。
@pytest.mark.asyncio
async def test_rejected_candidate_does_not_block_future(tmp_path: Path) -> None:
    env, root = await _make_env(tmp_path)
    from datetime import UTC, datetime
    from uuid import uuid4

    pending = SkillCandidate(
        id=uuid4().hex,
        action=SkillCandidateAction.CREATE,
        proposed_name="python-runtime-debug",
        description="待评审",
        reason="原因",
        procedure=("复现",),
        source_task_ids=("x", "y", "z"),
        created_at=datetime.now(UTC),
    )
    await env["candidate_store"].create(pending)
    await env["candidate_store"].update(
        pending.model_copy(
            update={
                "status": SkillCandidateStatus.REJECTED,
                "reviewed_at": datetime.now(UTC),
            }
        )
    )
    _, outcome, _ = await _run_batch(
        env,
        root,
        distill_response=_DISTILL_CREATE_JSON,
    )
    assert outcome.candidate_count == 1


# 函数说明：test_accepted_skill_in_catalog_drives_update
# 用途：回归验证回归测试与测试辅助中的 `accepted_skill_in_catalog_drives_update` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `skill_root.mkdir` →
# `(skill_root / 'SKILL.md').write_text` → `_run_batch` → `service_candidates`。
# 分支与异常：
#   验证条件：`outcome.candidate_count == 1`。
#   验证条件：`candidates[0].action is SkillCandidateAction.UPDATE`。
#   验证条件：`candidates[0].existing_skill_name == 'debug-python'`。
#   验证条件：`candidates[0].proposed_name == 'debug-python'`。
# 副作用与资源：
#   文件或资源访问：`skill_root.mkdir`、`(skill_root / 'SKILL.md').write_text`。
@pytest.mark.asyncio
async def test_accepted_skill_in_catalog_drives_update(tmp_path: Path) -> None:
    env, root = await _make_env(tmp_path)
    skill_root = env["skill_store"].project_dir / "debug-python"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "---\nname: debug-python\ndescription: 排查 Python 报错\n---\n\n"
        "# Debug\n\n1. 复现",
        encoding="utf-8",
    )
    _, outcome, _ = await _run_batch(
        env,
        root,
        distill_response=_DISTILL_UPDATE_JSON,
    )
    assert outcome.candidate_count == 1
    candidates = await service_candidates(env)
    assert candidates[0].action is SkillCandidateAction.UPDATE
    assert candidates[0].existing_skill_name == "debug-python"
    assert candidates[0].proposed_name == "debug-python"




# 函数说明：_run_progressive_disclosure
# 用途：运行`progressive_disclosure`，供回归测试与测试辅助使用。
# 参数：
#   env：环境变量映射；读取键 `task_store`、`trace_store`、`skill_store`、
# `candidate_store`。
#   root：当前操作的根目录。
#   relevance：`relevance`输入或配置值，类型 `list[str]`。
#   distill：传给 `_model_response` 的输入，类型 `str`。
#   adjudication：传给 `_model_response` 的输入，类型 `str | None`；默认 `None`。
# 返回：返回 `(service, outcome, adapter)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_create_completed` →
# `_model_response` → `_mining_cluster_json` → `json.dumps` → `_fake_registry` →
# `SkillLearningService`；另有 2 个调用点。
async def _run_progressive_disclosure(
    env,
    root,
    *,
    relevance: list[str],
    distill: str,
    adjudication: str | None = None,
):

    task_ids: list[str] = []
    for index in range(3):
        task_id = await _create_completed(
            env["task_store"],
            title=f"排查 Python 环境报错{index}",
            steps=("复现", "确认 virtualenv", "修复", "验证"),
            run_ids=(f"r{index}",),
        )
        task_ids.append(task_id)
    responses = [
        _model_response(_mining_cluster_json(task_ids)),
        _model_response(
            json.dumps({"related_skills": relevance}, ensure_ascii=False)
        ),
        _model_response(distill),
    ]
    if adjudication is not None:
        responses.append(_model_response(adjudication))
    registry, adapter = _fake_registry(responses)
    service = SkillLearningService(
        env["task_store"],
        env["trace_store"],
        env["skill_store"],
        env["candidate_store"],
        registry,
        settings=_settings(root, batch_size=3),
        default_provider="fake",
    )
    outcome = await service.maybe_run_mining()
    return service, outcome, adapter


# 函数说明：test_distiller_returns_none_when_skill_body_covers_procedure
# 用途：回归验证回归测试与测试辅助中的
# `distiller_returns_none_when_skill_body_covers_procedure` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `skill_root.mkdir` →
# `(skill_root / 'SKILL.md').write_text` → `_run_progressive_disclosure` → `json.dumps`
# 。
# 分支与异常：
#   验证条件：`outcome.candidate_count == 0`。
#   验证条件：`'内置步骤' in final_content`。
#   验证条件：`'"related_skills"' in final_content`。
# 副作用与资源：
#   文件或资源访问：`skill_root.mkdir`、`(skill_root / 'SKILL.md').write_text`。
@pytest.mark.asyncio
async def test_distiller_returns_none_when_skill_body_covers_procedure(
    tmp_path: Path,
) -> None:
    env, root = await _make_env(tmp_path)
    skill_root = env["skill_store"].project_dir / "debug-python"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "---\nname: debug-python\ndescription: 排查 Python 虚拟环境报错\n---\n\n"
        "# Debug Python\n\n1. 复现\n2. 确认 virtualenv（内置步骤）\n3. 修复并验证",
        encoding="utf-8",
    )
    _, outcome, adapter = await _run_progressive_disclosure(
        env,
        root,
        relevance=["debug-python"],
        distill=json.dumps(
            {"action": "none", "reason": "正文已包含确认 virtualenv 步骤"},
            ensure_ascii=False,
        ),
    )
    assert outcome.candidate_count == 0
    final_content = adapter.requests[-1].messages[-1].content or ""
    assert "内置步骤" in final_content
    assert '"related_skills"' in final_content


# 函数说明：test_distiller_loads_related_skill_body_for_update
# 用途：回归验证回归测试与测试辅助中的 `distiller_loads_related_skill_body_for_update`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `skill_root.mkdir` →
# `(skill_root / 'SKILL.md').write_text` → `_run_progressive_disclosure` →
# `service_candidates`。
# 分支与异常：
#   验证条件：`outcome.candidate_count == 1`。
#   验证条件：`candidates[0].action is SkillCandidateAction.UPDATE`。
#   验证条件：`candidates[0].existing_skill_name == 'debug-python'`。
#   验证条件：`'读 traceback' in final_content`。
# 副作用与资源：
#   文件或资源访问：`skill_root.mkdir`、`(skill_root / 'SKILL.md').write_text`。
@pytest.mark.asyncio
async def test_distiller_loads_related_skill_body_for_update(
    tmp_path: Path,
) -> None:
    env, root = await _make_env(tmp_path)
    skill_root = env["skill_store"].project_dir / "debug-python"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "---\nname: debug-python\ndescription: 排查 Python 虚拟环境报错\n---\n\n"
        "# Debug Python\n\n1. 复现\n2. 读 traceback\n3. 修复并验证",
        encoding="utf-8",
    )
    _, outcome, adapter = await _run_progressive_disclosure(
        env,
        root,
        relevance=["debug-python"],
        distill=_DISTILL_UPDATE_JSON,
    )
    assert outcome.candidate_count == 1
    candidates = await service_candidates(env)
    assert candidates[0].action is SkillCandidateAction.UPDATE
    assert candidates[0].existing_skill_name == "debug-python"
    final_content = adapter.requests[-1].messages[-1].content or ""
    assert "读 traceback" in final_content


# 函数说明：test_distiller_creates_when_related_skill_unrelated
# 用途：回归验证回归测试与测试辅助中的 `distiller_creates_when_related_skill_unrelated`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `skill_root.mkdir` →
# `(skill_root / 'SKILL.md').write_text` → `_run_progressive_disclosure` →
# `service_candidates`。
# 分支与异常：
#   验证条件：`outcome.candidate_count == 1`。
#   验证条件：`candidates[0].action is SkillCandidateAction.CREATE`。
#   验证条件：`'"related_skills":[]' in final_content`。
#   验证条件：`'写总结' not in final_content`。
# 副作用与资源：
#   文件或资源访问：`skill_root.mkdir`、`(skill_root / 'SKILL.md').write_text`。
@pytest.mark.asyncio
async def test_distiller_creates_when_related_skill_unrelated(
    tmp_path: Path,
) -> None:
    env, root = await _make_env(tmp_path)
    skill_root = env["skill_store"].project_dir / "weekly-report"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "---\nname: weekly-report\ndescription: 撰写每周工作周报\n---\n\n"
        "# 周报\n\n1. 收集信息\n2. 写总结",
        encoding="utf-8",
    )
    _, outcome, adapter = await _run_progressive_disclosure(
        env,
        root,
        relevance=[],
        distill=_DISTILL_CREATE_JSON,
    )
    assert outcome.candidate_count == 1
    candidates = await service_candidates(env)
    assert candidates[0].action is SkillCandidateAction.CREATE
    final_content = adapter.requests[-1].messages[-1].content or ""
    assert '"related_skills":[]' in final_content
    assert "写总结" not in final_content


# 函数说明：test_create_with_related_skill_gets_overlap_adjudication
# 用途：回归验证回归测试与测试辅助中的
# `create_with_related_skill_gets_overlap_adjudication` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `skill_root.mkdir` →
# `(skill_root / 'SKILL.md').write_text` → `_run_progressive_disclosure` → `json.dumps`
# → `service_candidates`。
# 分支与异常：
#   验证条件：`outcome.error is None`。
#   验证条件：`outcome.candidate_count == 1`。
#   验证条件：`outcome.distillation_calls == 3`。
#   验证条件：`outcome.distillations[0].adjudication_raw_output is not None`。
# 副作用与资源：
#   文件或资源访问：`skill_root.mkdir`、`(skill_root / 'SKILL.md').write_text`。
@pytest.mark.asyncio
async def test_create_with_related_skill_gets_overlap_adjudication(
    tmp_path: Path,
) -> None:

    env, root = await _make_env(tmp_path)
    skill_root = env["skill_store"].project_dir / "debug-python"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "---\nname: debug-python\ndescription: 排查 Python 报错\n---\n\n"
        "# Debug Python\n\n1. 复现\n2. 读 traceback\n3. 修复并验证",
        encoding="utf-8",
    )
    _, outcome, adapter = await _run_progressive_disclosure(
        env,
        root,
        relevance=["debug-python"],
        distill=_DISTILL_CREATE_JSON,
        adjudication=json.dumps(
            {
                "relationship": "same",
                "existing_skill_name": "debug-python",
                "reason": "解释器错配是 Python 运行时排错的子场景",
            },
            ensure_ascii=False,
        ),
    )

    assert outcome.error is None
    assert outcome.candidate_count == 1
    assert outcome.distillation_calls == 3
    assert outcome.distillations[0].adjudication_raw_output is not None
    candidates = await service_candidates(env)
    assert candidates[0].action is SkillCandidateAction.UPDATE
    assert candidates[0].existing_skill_name == "debug-python"
    assert candidates[0].proposed_name == "debug-python"
    assert "技能家族裁决" in (
        adapter.requests[-1].messages[0].content or ""
    )


# 函数说明：test_update_candidate_inherits_description_when_model_omits
# 用途：回归验证回归测试与测试辅助中的
# `update_candidate_inherits_description_when_model_omits` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `skill_root.mkdir` →
# `(skill_root / 'SKILL.md').write_text` → `json.dumps` → `_run_progressive_disclosure`
# → `service_candidates`。
# 分支与异常：
#   验证条件：`outcome.candidate_count == 1`。
#   验证条件：`candidates[0].action is SkillCandidateAction.UPDATE`。
#   验证条件：`candidates[0].existing_skill_name == 'debug-python'`。
#   验证条件：`candidates[0].description == description`。
# 副作用与资源：
#   文件或资源访问：`skill_root.mkdir`、`(skill_root / 'SKILL.md').write_text`。
@pytest.mark.asyncio
async def test_update_candidate_inherits_description_when_model_omits(
    tmp_path: Path,
) -> None:
    env, root = await _make_env(tmp_path)
    skill_root = env["skill_store"].project_dir / "debug-python"
    skill_root.mkdir(parents=True)
    description = "排查 Python 虚拟环境报错的标准流程"
    (skill_root / "SKILL.md").write_text(
        "---\nname: debug-python\n"
        f"description: {description}\n---\n\n"
        "# Debug Python\n\n1. 复现\n2. 读 traceback\n3. 修复并验证",
        encoding="utf-8",
    )
    update_no_desc = json.dumps(
        {
            "action": "update",
            "proposed_name": None,
            "description": None,
            "reason": "正文缺 virtualenv 确认步骤",
            "procedure": ["复现", "确认 virtualenv", "修复", "验证"],
            "pitfalls": [],
            "verification": ["pytest 通过"],
            "existing_skill_name": "debug-python",
        },
        ensure_ascii=False,
    )
    _, outcome, _ = await _run_progressive_disclosure(
        env,
        root,
        relevance=["debug-python"],
        distill=update_no_desc,
    )
    assert outcome.candidate_count == 1
    candidates = await service_candidates(env)
    assert candidates[0].action is SkillCandidateAction.UPDATE
    assert candidates[0].existing_skill_name == "debug-python"
    assert candidates[0].description == description


# 函数说明：test_update_candidate_uses_model_description_when_provided
# 用途：回归验证回归测试与测试辅助中的
# `update_candidate_uses_model_description_when_provided` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `skill_root.mkdir` →
# `(skill_root / 'SKILL.md').write_text` → `json.dumps` → `_run_progressive_disclosure`
# → `service_candidates`。
# 分支与异常：
#   验证条件：`outcome.candidate_count == 1`。
#   验证条件：`candidates[0].action is SkillCandidateAction.UPDATE`。
#   验证条件：`candidates[0].description == model_desc`。
# 副作用与资源：
#   文件或资源访问：`skill_root.mkdir`、`(skill_root / 'SKILL.md').write_text`。
@pytest.mark.asyncio
async def test_update_candidate_uses_model_description_when_provided(
    tmp_path: Path,
) -> None:
    env, root = await _make_env(tmp_path)
    skill_root = env["skill_store"].project_dir / "debug-python"
    skill_root.mkdir(parents=True)
    catalog_desc = "排查 Python 虚拟环境报错的标准流程"
    (skill_root / "SKILL.md").write_text(
        "---\nname: debug-python\n"
        f"description: {catalog_desc}\n---\n\n"
        "# Debug Python\n\n1. 复现\n2. 读 traceback\n3. 修复并验证",
        encoding="utf-8",
    )
    model_desc = "模型给的新描述：先确认 virtualenv"
    update_with_desc = json.dumps(
        {
            "action": "update",
            "proposed_name": None,
            "description": model_desc,
            "reason": "补充 virtualenv 步骤",
            "procedure": ["复现", "确认 virtualenv", "修复", "验证"],
            "pitfalls": [],
            "verification": ["pytest 通过"],
            "existing_skill_name": "debug-python",
        },
        ensure_ascii=False,
    )
    _, outcome, _ = await _run_progressive_disclosure(
        env,
        root,
        relevance=["debug-python"],
        distill=update_with_desc,
    )
    assert outcome.candidate_count == 1
    candidates = await service_candidates(env)
    assert candidates[0].action is SkillCandidateAction.UPDATE
    assert candidates[0].description == model_desc


# 函数说明：test_update_candidate_repairs_unique_missing_existing_skill_name
# 用途：回归验证回归测试与测试辅助中的
# `update_candidate_repairs_unique_missing_existing_skill_name` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `skill_root.mkdir` →
# `(skill_root / 'SKILL.md').write_text` → `json.dumps` → `_run_progressive_disclosure`
# → `service_candidates`。
# 分支与异常：
#   验证条件：`outcome.candidate_count == 1`。
#   验证条件：`candidates[0].action is SkillCandidateAction.UPDATE`。
#   验证条件：`candidates[0].existing_skill_name == 'debug-python'`。
#   验证条件：`candidates[0].description == description`。
# 副作用与资源：
#   文件或资源访问：`skill_root.mkdir`、`(skill_root / 'SKILL.md').write_text`。
@pytest.mark.asyncio
async def test_update_candidate_repairs_unique_missing_existing_skill_name(
    tmp_path: Path,
) -> None:

    env, root = await _make_env(tmp_path)
    skill_root = env["skill_store"].project_dir / "debug-python"
    skill_root.mkdir(parents=True)
    description = "排查 Python 报错或异常的标准流程"
    (skill_root / "SKILL.md").write_text(
        "---\nname: debug-python\n"
        f"description: {description}\n---\n\n"
        "# Debug Python\n\n1. 复现\n2. 读 traceback\n3. 修复并验证",
        encoding="utf-8",
    )
    update_without_target = json.dumps(
        {
            "action": "update",
            "proposed_name": None,
            "description": None,
            "reason": "补充稳定的 virtualenv 检查步骤",
            "procedure": ["复现", "确认 virtualenv", "修复", "验证"],
            "pitfalls": [],
            "verification": ["pytest 通过"],
            "existing_skill_name": None,
        },
        ensure_ascii=False,
    )

    _, outcome, _ = await _run_progressive_disclosure(
        env,
        root,
        relevance=["debug-python"],
        distill=update_without_target,
    )

    assert outcome.candidate_count == 1
    candidates = await service_candidates(env)
    assert candidates[0].action is SkillCandidateAction.UPDATE
    assert candidates[0].existing_skill_name == "debug-python"
    assert candidates[0].description == description


# 函数说明：test_update_candidate_fails_when_existing_skill_missing
# 用途：回归验证回归测试与测试辅助中的
# `update_candidate_fails_when_existing_skill_missing` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `json.dumps` →
# `_run_batch`。
# 分支与异常：
#   验证条件：`outcome.candidate_count == 0`。
#   验证条件：`outcome.error is not None`。
#   验证条件：`'ghost-skill' in outcome.error`。
#   验证条件：`'not found' in outcome.error`。
@pytest.mark.asyncio
async def test_update_candidate_fails_when_existing_skill_missing(
    tmp_path: Path,
) -> None:
    env, root = await _make_env(tmp_path)
    update_ghost = json.dumps(
        {
            "action": "update",
            "proposed_name": None,
            "description": None,
            "reason": "指向不存在的 skill",
            "procedure": ["复现", "修复"],
            "pitfalls": [],
            "verification": ["验证"],
            "existing_skill_name": "ghost-skill",
        },
        ensure_ascii=False,
    )
    _, outcome, _ = await _run_batch(
        env,
        root,
        distill_response=update_ghost,
    )
    assert outcome.candidate_count == 0
    assert outcome.error is not None
    assert "ghost-skill" in outcome.error
    assert "not found" in outcome.error


# 函数说明：test_create_candidate_still_requires_description
# 用途：回归验证回归测试与测试辅助中的 `create_candidate_still_requires_description` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `json.dumps` →
# `_run_batch`。
# 分支与异常：
#   验证条件：`outcome.candidate_count == 0`。
#   验证条件：`outcome.error is not None`。
#   验证条件：`'create candidate requires' in outcome.error`。
@pytest.mark.asyncio
async def test_create_candidate_still_requires_description(tmp_path: Path) -> None:
    env, root = await _make_env(tmp_path)
    create_no_desc = json.dumps(
        {
            "action": "create",
            "proposed_name": "new-skill",
            "description": None,
            "reason": "缺少 description",
            "procedure": ["步骤"],
            "pitfalls": [],
            "verification": [],
        },
        ensure_ascii=False,
    )
    _, outcome, _ = await _run_batch(
        env,
        root,
        distill_response=create_no_desc,
    )
    assert outcome.candidate_count == 0
    assert outcome.error is not None
    assert "create candidate requires" in outcome.error




# 函数说明：_tool_event_step
# 用途：返回 `AgentEvent(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   run_id：目标运行标识，类型 `str`。
#   sequence：事件或记录顺序号，类型 `int`。
#   step：当前任务步骤，类型 `int`。
#   name：目标对象、工具或配置项名称，类型 `str`。
#   arguments：工具调用的参数对象或 JSON 文本，类型 `dict`。
#   success：执行是否成功，类型 `bool`；默认 `True`。
# 返回：类型 `AgentEvent`；返回 `AgentEvent(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`AgentEvent` → `ToolCall` →
# `ToolResult`。
def _tool_event_step(
    run_id: str,
    sequence: int,
    step: int,
    name: str,
    arguments: dict,
    *,
    success: bool = True,
) -> AgentEvent:

    return AgentEvent(
        run_id=run_id,
        conversation_id="conv",
        sequence=sequence,
        step=step,
        type=AgentEventType.TOOL_COMPLETED,
        tool_call=ToolCall(id=f"c{sequence}", name=name, arguments=arguments),
        tool_result=ToolResult(
            tool_call_id=f"c{sequence}",
            tool_name=name,
            success=success,
            duration_ms=0.0,
        ),
    )


# 函数说明：test_service_loads_anchor_bounded_events
# 用途：回归验证回归测试与测试辅助中的 `service_loads_anchor_bounded_events` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `_fake_registry` →
# `SkillLearningService` → `_settings` → `_create_completed` → `_record_trace`；另有 2
# 个调用点。
# 分支与异常：
#   验证条件：`task is not None`。
#   验证条件：`[event.step for event in events] == [2, 3, 4]`。
@pytest.mark.asyncio
async def test_service_loads_anchor_bounded_events(tmp_path: Path) -> None:
    env, root = await _make_env(tmp_path, batch_size=3)
    registry, _ = _fake_registry([])
    service = SkillLearningService(
        env["task_store"],
        env["trace_store"],
        env["skill_store"],
        env["candidate_store"],
        registry,
        settings=_settings(root, batch_size=3),
        default_provider="fake",
    )
    task_id = await _create_completed(
        env["task_store"],
        title="修复报错",
        run_ids=("r1",),
    )
    task = await env["task_store"].get(task_id)
    assert task is not None
    await _record_trace(
        env["trace_store"],
        (
            _tool_event_step("r1", 1, 1, "read_file", {}),
            _tool_event_step(
                "r1", 2, 2, "task_update",
                {"task_id": task_id, "step_id": "s1",
                 "step_status": "in_progress"},
            ),
            _tool_event_step(
                "r1", 3, 3, "run_pytest", {}, success=False
            ),
            _tool_event_step(
                "r1", 4, 4, "task_update",
                {"task_id": task_id, "step_id": "s1",
                 "step_status": "done", "step_note": "ok"},
            ),
            _tool_event_step("r1", 5, 5, "read_file", {}),
        ),
    )
    events = await service._load_task_events(task)
    assert [event.step for event in events] == [2, 3, 4]




# 函数说明：_seed_skill
# 用途：在回归测试与测试辅助中处理 `_seed_skill`，通过 `skill_root.mkdir` 完成首个内部处
# 理步骤。
# 参数：
#   env：环境变量映射；读取键 `skill_store`。
#   name：目标对象、工具或配置项名称，类型 `str`；默认 `'debug-python'`。
#   description：补充描述，类型 `str`；默认 `'old'`。
#   body：请求正文或内容主体，类型 `str`；默认 `'# Debug Python\n\n1. Procedure A'`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`skill_root.mkdir` →
# `(skill_root / 'SKILL.md').write_text`。
# 副作用与资源：
#   文件或资源访问：`skill_root.mkdir`、`(skill_root / 'SKILL.md').write_text`。
def _seed_skill(
    env,
    name: str = "debug-python",
    description: str = "old",
    body: str = "# Debug Python\n\n1. Procedure A",
) -> None:
    skill_root = env["skill_store"].project_dir / name
    skill_root.mkdir(parents=True, exist_ok=True)
    (skill_root / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}",
        encoding="utf-8",
    )


# 函数说明：_update_candidate
# 用途：更新候选，供回归测试与测试辅助使用。
# 参数：
#   existing：已经存在的值或记录，类型 `str`；默认 `'debug-python'`。
#   proposed：`proposed`输入或配置值，类型 `str`；默认 `'debug-python'`。
#   description：补充描述，类型 `str`；默认 `'new'`。
#   procedure：`procedure`输入或配置值，类型 `tuple[str, ...]`；默认 `('Procedure B',)`
# 。
# 返回：类型 `SkillCandidate`；返回 `SkillCandidate(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillCandidate` → `uuid4` →
# `datetime.now`。
def _update_candidate(
    *,
    existing: str = "debug-python",
    proposed: str = "debug-python",
    description: str = "new",
    procedure: tuple[str, ...] = ("Procedure B",),
) -> SkillCandidate:
    from datetime import UTC, datetime
    from uuid import uuid4

    return SkillCandidate(
        id=uuid4().hex,
        action=SkillCandidateAction.UPDATE,
        proposed_name=proposed,
        description=description,
        reason="补充 Procedure B",
        procedure=procedure,
        pitfalls=("不要跳过验证",),
        verification=("pytest 通过",),
        source_task_ids=("a", "b", "c"),
        existing_skill_name=existing,
        created_at=datetime.now(UTC),
    )


# 函数说明：_learning_service
# 用途：处理回归测试与测试辅助中的 `_learning_service` 数据；结果及边界条件见下方说明。
# 参数：
#   env：环境变量映射；读取键 `task_store`、`trace_store`、`skill_store`、
# `candidate_store`。
#   root：当前操作的根目录。
# 返回：类型 `SkillLearningService`；返回 `SkillLearningService(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_fake_registry` →
# `SkillLearningService` → `_settings`。
def _learning_service(env, root) -> SkillLearningService:
    registry, _ = _fake_registry([])
    return SkillLearningService(
        env["task_store"],
        env["trace_store"],
        env["skill_store"],
        env["candidate_store"],
        registry,
        settings=_settings(root, batch_size=3),
        default_provider="fake",
    )


# 函数说明：test_accept_update_overwrites_real_skill
# 用途：回归验证回归测试与测试辅助中的 `accept_update_overwrites_real_skill` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `_seed_skill` →
# `env['candidate_store'].create` → `_update_candidate` → `_learning_service` →
# `service.accept`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`updated.status is SkillCandidateStatus.ACCEPTED`。
#   验证条件：`target is not None and target.name == 'SKILL.md'`。
#   验证条件：`skill is not None`。
#   验证条件：`'Procedure B' in skill.content`。
@pytest.mark.asyncio
async def test_accept_update_overwrites_real_skill(tmp_path: Path) -> None:
    env, root = await _make_env(tmp_path)
    _seed_skill(env, body="# Debug Python\n\n1. Procedure A")
    await env["candidate_store"].create(_update_candidate(procedure=("Procedure B",)))
    service = _learning_service(env, root)

    updated, target = await service.accept((await service_candidates(env))[0].id)
    assert updated.status is SkillCandidateStatus.ACCEPTED
    assert target is not None and target.name == "SKILL.md"
    skill = await env["skill_store"].load("debug-python")
    assert skill is not None
    assert "Procedure B" in skill.content
    assert "提案" not in skill.content
    assert "来源 Task" not in skill.content


# 函数说明：test_accept_update_keeps_skill_name
# 用途：回归验证回归测试与测试辅助中的 `accept_update_keeps_skill_name` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `_seed_skill` →
# `env['candidate_store'].create` → `_update_candidate` → `_learning_service` →
# `service.accept`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`skill is not None and skill.metadata.name == 'debug-python'`。
#   验证条件：`await env['skill_store'].load('evil-name') is None`。
#   验证条件：`'debug-python' in str(target)`。
@pytest.mark.asyncio
async def test_accept_update_keeps_skill_name(tmp_path: Path) -> None:
    env, root = await _make_env(tmp_path)
    _seed_skill(env)
    await env["candidate_store"].create(
        _update_candidate(proposed="evil-name")
    )
    service = _learning_service(env, root)

    _, target = await service.accept((await service_candidates(env))[0].id)
    skill = await env["skill_store"].load("debug-python")
    assert skill is not None and skill.metadata.name == "debug-python"
    assert await env["skill_store"].load("evil-name") is None
    assert "debug-python" in str(target)


# 函数说明：test_accept_update_uses_candidate_description
# 用途：回归验证回归测试与测试辅助中的 `accept_update_uses_candidate_description` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `_seed_skill` →
# `env['candidate_store'].create` → `_update_candidate` → `_learning_service` →
# `service.accept`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`skill is not None`。
#   验证条件：`skill.metadata.description == 'new'`。
@pytest.mark.asyncio
async def test_accept_update_uses_candidate_description(tmp_path: Path) -> None:
    env, root = await _make_env(tmp_path)
    _seed_skill(env, description="old")
    await env["candidate_store"].create(_update_candidate(description="new"))
    service = _learning_service(env, root)

    await service.accept((await service_candidates(env))[0].id)
    skill = await env["skill_store"].load("debug-python")
    assert skill is not None
    assert skill.metadata.description == "new"


# 函数说明：test_accept_update_missing_skill_fails
# 用途：回归验证回归测试与测试辅助中的 `accept_update_missing_skill_fails` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` →
# `env['candidate_store'].create` → `_update_candidate` → `_learning_service` →
# `pytest.raises` → `service.accept`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`after.status is SkillCandidateStatus.PENDING`。
#   验证条件：`await env['skill_store'].load('ghost-skill') is None`。
#   预期异常：`pytest.raises(ValueError)`。
@pytest.mark.asyncio
async def test_accept_update_missing_skill_fails(tmp_path: Path) -> None:
    env, root = await _make_env(tmp_path)
    await env["candidate_store"].create(
        _update_candidate(existing="ghost-skill")
    )
    service = _learning_service(env, root)

    with pytest.raises(ValueError):
        await service.accept((await service_candidates(env))[0].id)
    after = (await service_candidates(env))[0]
    assert after.status is SkillCandidateStatus.PENDING
    assert await env["skill_store"].load("ghost-skill") is None


# 函数说明：test_accept_update_write_failure_keeps_pending
# 用途：回归验证回归测试与测试辅助中的 `accept_update_write_failure_keeps_pending` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
#   monkeypatch：pytest 提供的临时替换依赖夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `_seed_skill` →
# `env['candidate_store'].create` → `_update_candidate` → `_learning_service` →
# `monkeypatch.setattr`；另有 4 个调用点。
# 分支与异常：
#   验证条件：`after.status is SkillCandidateStatus.PENDING`。
#   验证条件：`skill is not None and 'Procedure A' in skill.content`。
#   预期异常：`pytest.raises(OSError)`。
@pytest.mark.asyncio
async def test_accept_update_write_failure_keeps_pending(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env, root = await _make_env(tmp_path)
    _seed_skill(env, body="# Debug Python\n\n1. Procedure A")
    await env["candidate_store"].create(_update_candidate())
    service = _learning_service(env, root)

    # 函数说明：test_accept_update_write_failure_keeps_pending.boom
    # 用途：处理回归测试与测试辅助中的 `boom` 数据；结果及边界条件见下方说明。
    # 参数：
    #   **kwargs：额外关键字参数，按实现处理或转交。
    # 返回：不返回结果值（隐式 None）。
    async def boom(**kwargs):  
        raise OSError("disk full")

    monkeypatch.setattr(env["skill_store"], "update", boom)
    with pytest.raises(OSError):
        await service.accept((await service_candidates(env))[0].id)
    after = (await service_candidates(env))[0]
    assert after.status is SkillCandidateStatus.PENDING
    skill = await env["skill_store"].load("debug-python")
    assert skill is not None and "Procedure A" in skill.content


# 函数说明：test_reject_update_keeps_skill
# 用途：回归验证回归测试与测试辅助中的 `reject_update_keeps_skill` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `_seed_skill` →
# `env['candidate_store'].create` → `_update_candidate` → `_learning_service` →
# `service.reject`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`skill is not None and 'Procedure A' in skill.content`。
#   验证条件：`after.status is SkillCandidateStatus.REJECTED`。
@pytest.mark.asyncio
async def test_reject_update_keeps_skill(tmp_path: Path) -> None:
    env, root = await _make_env(tmp_path)
    _seed_skill(env, body="# Debug Python\n\n1. Procedure A")
    await env["candidate_store"].create(_update_candidate())
    service = _learning_service(env, root)

    await service.reject((await service_candidates(env))[0].id)
    skill = await env["skill_store"].load("debug-python")
    assert skill is not None and "Procedure A" in skill.content
    after = (await service_candidates(env))[0]
    assert after.status is SkillCandidateStatus.REJECTED


# 函数说明：test_accept_create_regression
# 用途：回归验证回归测试与测试辅助中的 `accept_create_regression` 场景，下方断言说明列出
# 实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `SkillCandidate` →
# `uuid4` → `datetime.now` → `env['candidate_store'].create` → `_learning_service`；另有
#  2 个调用点。
# 分支与异常：
#   验证条件：`updated.status is SkillCandidateStatus.ACCEPTED`。
#   验证条件：`target is not None and target.name == 'SKILL.md'`。
#   验证条件：`skill is not None`。
#   验证条件：`'步骤' in skill.content`。
@pytest.mark.asyncio
async def test_accept_create_regression(tmp_path: Path) -> None:
    from datetime import UTC, datetime
    from uuid import uuid4

    env, root = await _make_env(tmp_path)
    candidate = SkillCandidate(
        id=uuid4().hex,
        action=SkillCandidateAction.CREATE,
        proposed_name="new-skill",
        description="描述",
        reason="原因",
        procedure=("步骤",),
        source_task_ids=("a", "b", "c"),
        created_at=datetime.now(UTC),
    )
    await env["candidate_store"].create(candidate)
    service = _learning_service(env, root)

    updated, target = await service.accept(candidate.id)
    assert updated.status is SkillCandidateStatus.ACCEPTED
    assert target is not None and target.name == "SKILL.md"
    skill = await env["skill_store"].load("new-skill")
    assert skill is not None
    assert "步骤" in skill.content


# 函数说明：test_accept_agent_proposal_creates_formal_skill
# 用途：回归验证回归测试与测试辅助中的 `accept_agent_proposal_creates_formal_skill` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `SkillCandidate` →
# `uuid4` → `datetime.now` → `env['candidate_store'].create` → `_learning_service`；另有
#  2 个调用点。
# 分支与异常：
#   验证条件：`accepted.status is SkillCandidateStatus.ACCEPTED`。
#   验证条件：`target == expected`。
#   验证条件：`skill is not None`。
#   验证条件：`'识别入口' in skill.content`。
@pytest.mark.asyncio
async def test_accept_agent_proposal_creates_formal_skill(tmp_path: Path) -> None:

    from datetime import UTC, datetime
    from uuid import uuid4

    env, root = await _make_env(tmp_path)
    candidate = SkillCandidate(
        id=uuid4().hex,
        origin=SkillCandidateOrigin.AGENT_PROPOSAL,
        action=SkillCandidateAction.CREATE,
        proposed_name="workspace-explainer",
        description="解释工作区结构",
        reason="用户明确要求沉淀已验证流程",
        procedure=("扫描目录", "识别入口", "解释调用关系"),
        source_run_ids=("run-1",),
        source_conversation_id="conversation-1",
        source_tool_call_id="call-1",
        created_at=datetime.now(UTC),
    )
    await env["candidate_store"].create(candidate)
    service = _learning_service(env, root)

    accepted, target = await service.accept(candidate.id)

    assert accepted.status is SkillCandidateStatus.ACCEPTED
    expected = (
        env["skill_store"].project_dir
        / candidate.proposed_name
        / "SKILL.md"
    )
    assert target == expected
    skill = await env["skill_store"].load(candidate.proposed_name)
    assert skill is not None
    assert "识别入口" in skill.content




# 函数说明：test_distillation_failure_keeps_batch_inflight
# 用途：回归验证回归测试与测试辅助中的 `distillation_failure_keeps_batch_inflight` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `_create_completed` →
# `_mining_cluster_json` → `_fake_registry` → `_model_response` → `SkillLearningService`
# ；另有 4 个调用点。
# 分支与异常：
#   验证条件：`outcome1.cluster_count == 1`。
#   验证条件：`outcome1.candidate_count == 0`。
#   验证条件：`outcome1.error`。
#   验证条件：`watermark1.inflight is not None`。
@pytest.mark.asyncio
async def test_distillation_failure_keeps_batch_inflight(tmp_path: Path) -> None:
    env, root = await _make_env(tmp_path, batch_size=3)
    task_ids: list[str] = []
    for index in range(3):
        task_ids.append(
            await _create_completed(
                env["task_store"],
                title=f"Python 报错{index}",
                steps=("复现", "读 traceback", "修复"),
                run_ids=(f"r{index}",),
            )
        )
    cluster_json = _mining_cluster_json(tuple(task_ids))
    registry, adapter = _fake_registry(
        [
            _model_response(cluster_json),
            RuntimeError("model down"),
            _model_response(cluster_json),
            _model_response(_DISTILL_CREATE_JSON),
        ]
    )
    service = SkillLearningService(
        env["task_store"],
        env["trace_store"],
        env["skill_store"],
        env["candidate_store"],
        registry,
        settings=_settings(root, batch_size=3),
        default_provider="fake",
    )

    outcome1 = await service.maybe_run_mining()
    assert outcome1.cluster_count == 1
    assert outcome1.candidate_count == 0
    assert outcome1.error
    watermark1 = await env["candidate_store"].load_watermark()
    assert watermark1.inflight is not None
    assert watermark1.inflight.attempt == 1
    assert not set(task_ids).issubset(set(watermark1.processed_task_ids))

    outcome2 = await service.maybe_run_mining()
    assert outcome2.candidate_count == 1
    watermark2 = await env["candidate_store"].load_watermark()
    assert watermark2.inflight is None
    assert set(task_ids).issubset(set(watermark2.processed_task_ids))




# 函数说明：test_failed_task_update_not_in_task_changes
# 用途：回归验证回归测试与测试辅助中的 `failed_task_update_not_in_task_changes` 场景，下
# 方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_env` → `_create_completed` →
# `_tool_started` → `_tool_completed` → `TraceEvidenceBuilder` → `_settings`；另有 1 个
# 调用点。
# 分支与异常：
#   验证条件：`task is not None`。
#   验证条件：`'goal: 新目标' in text`。
#   验证条件：`'失败工具调用' in text`。
#   验证条件：`'boom' in text`。
@pytest.mark.asyncio
async def test_failed_task_update_not_in_task_changes(tmp_path: Path) -> None:
    env, _ = await _make_env(tmp_path)
    task_id = await _create_completed(
        env["task_store"],
        title="修复报错",
        run_ids=("r1",),
    )
    task = await env["task_store"].get(task_id)
    assert task is not None
    events = (
        _tool_started("r1", 1, "task_update", {"task_id": task_id, "goal": "新目标"}),
        _tool_completed("r1", 2, "task_update", {"task_id": task_id, "goal": "新目标"}),
        _tool_started(
            "r1", 3, "task_update",
            {"task_id": task_id, "constraints": ["不要重装"]},
        ),
        _tool_completed(
            "r1", 4, "task_update",
            {"task_id": task_id, "constraints": ["不要重装"]},
            success=False,
            error="boom",
        ),
    )
    builder = TraceEvidenceBuilder(_settings(tmp_path))
    text = builder.build(task, events)
    assert "goal: 新目标" in text
    assert "失败工具调用" in text
    assert "boom" in text
    assert "constraints added: 不要重装" not in text
