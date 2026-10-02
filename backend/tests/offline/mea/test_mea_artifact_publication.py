"""Regression for executable and independently auditable s6 publication."""

import hashlib
import json

import pytest

from app.domain.artifact import (
    ArtifactService,
    SQLiteArtifactStore,
    register_artifact_tools,
)
from app.models.types import AgentMode, ToolCall
from app.records.evidence import EvidenceRecorder, SQLiteEvidenceStore
from app.runtime.agent.runtime import AgentRuntime
from app.runtime.checkpoint import SQLiteCheckpointStore
from app.runtime.run import RunManager, RunStatus, SQLiteRunStore
from app.tools.executor import ToolExecutor
from app.tools.hooks import ToolExecutionContext
from app.tools.registry import ToolRegistry


# 函数说明：environment
# 用途：在回归测试与测试辅助中处理 `environment`，通过 `workspace.mkdir` 完成首个内部处
# 理步骤。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：返回 `(workspace, service, evidence, registry, executor)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`workspace.mkdir` →
# `SQLiteArtifactStore` → `artifacts.initialize` → `ArtifactService` →
# `SQLiteEvidenceStore` → `evidence.initialize`；另有 4 个调用点。
# 副作用与资源：
#   文件或资源访问：`workspace.mkdir`。
async def environment(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifacts = SQLiteArtifactStore(tmp_path / "test.db")
    await artifacts.initialize()
    service = ArtifactService(artifacts, workspace, managed_dir=tmp_path / "published")
    evidence = SQLiteEvidenceStore(tmp_path / "test.db")
    await evidence.initialize()
    registry = ToolRegistry()
    register_artifact_tools(registry, service)
    executor = ToolExecutor(registry, output_recorder=EvidenceRecorder(evidence))
    return workspace, service, evidence, registry, executor


# 函数说明：context
# 用途：返回 `ToolExecutionContext(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   call：调用输入或配置值。
#   mode：Agent 执行模式或检索模式。
# 返回：返回 `ToolExecutionContext(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolExecutionContext`。
def context(call, mode):
    return ToolExecutionContext(
        tool_call=call, run_id="executor-run", conversation_id="conversation", mode=mode
    )


# 函数说明：test_executor_sees_publication_in_normal_and_closing_tools
# 用途：回归验证回归测试与测试辅助中的
# `executor_sees_publication_in_normal_and_closing_tools` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`environment` →
# `registry.model_definitions_for_mode` → `registry.closing_definitions_for_mode`。
# 分支与异常：
#   验证条件：`'artifact_publish' in {d.name for d in definitions}`。
async def test_executor_sees_publication_in_normal_and_closing_tools(tmp_path):
    _, _, _, registry, _ = await environment(tmp_path)
    for definitions in (
        registry.model_definitions_for_mode(AgentMode.EXECUTE),
        registry.closing_definitions_for_mode(AgentMode.EXECUTE),
    ):
        assert "artifact_publish" in {d.name for d in definitions}


# 函数说明：test_executor_publishes_two_files_with_real_hashes
# 用途：回归验证回归测试与测试辅助中的 `executor_publishes_two_files_with_real_hashes`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`environment` →
# `(workspace / name).write_bytes` → `ToolCall` → `executor.execute` → `context` →
# `json.loads`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`result.success`。
#   验证条件：`receipt['size_bytes'] == len(content)`。
#   验证条件：`receipt['sha256'] == hashlib.sha256(content).hexdigest()`。
#   验证条件：`receipt['run_id'] == 'executor-run'`。
# 副作用与资源：
#   文件或资源访问：`(workspace / name).write_bytes`、`copied.read_bytes`。
async def test_executor_publishes_two_files_with_real_hashes(tmp_path):
    workspace, service, _, _, executor = await environment(tmp_path)
    for index, (name, content) in enumerate(
        (("wc.py", b"print('words')\n"), ("sample.txt", b"one two two\n"))
    ):
        (workspace / name).write_bytes(content)
        call = ToolCall(
            id=f"publish-{index}", name="artifact_publish", arguments={"path": name}
        )
        result = await executor.execute(call, context=context(call, AgentMode.EXECUTE))
        assert result.success, result.error
        receipt = json.loads(result.output)
        assert receipt["size_bytes"] == len(content)
        assert receipt["sha256"] == hashlib.sha256(content).hexdigest()
        assert receipt["run_id"] == "executor-run"
        copied = await service.file_path(receipt["id"])
        assert copied.read_bytes() == content
    assert len(await service.store.list()) == 2


# 函数说明：test_publication_receipt_is_durable_conversation_private_evidence
# 用途：回归验证回归测试与测试辅助中的
# `publication_receipt_is_durable_conversation_private_evidence` 场景，下方断言说明列出
# 实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`environment` →
# `(workspace / 'sample.txt').write_bytes` → `ToolCall` → `executor.execute` → `context`
#  → `evidence.resolve`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`result.success`。
#   验证条件：`result.evidence_id is not None`。
#   验证条件：`json.loads(document.content) == json.loads(result.output)`。
#   验证条件：
# `await evidence.resolve(result.evidence_id, conversation_id='other') is None`。
# 副作用与资源：
#   文件或资源访问：`(workspace / 'sample.txt').write_bytes`。
async def test_publication_receipt_is_durable_conversation_private_evidence(tmp_path):
    workspace, _, evidence, _, executor = await environment(tmp_path)
    (workspace / "sample.txt").write_bytes(b"independent evidence")
    call = ToolCall(
        id="receipt", name="artifact_publish", arguments={"path": "sample.txt"}
    )
    # NORMAL isolates the second defect from the EXECUTE whitelist defect.
    result = await executor.execute(call, context=context(call, AgentMode.NORMAL))
    assert result.success, result.error
    assert result.evidence_id is not None
    document = await evidence.resolve(
        result.evidence_id, conversation_id="conversation"
    )
    assert json.loads(document.content) == json.loads(result.output)
    assert await evidence.resolve(result.evidence_id, conversation_id="other") is None


# 函数说明：test_task_attribution_failure_does_not_block_publication
# 用途：回归验证回归测试与测试辅助中的
# `task_attribution_failure_does_not_block_publication` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`workspace.mkdir` →
# `(workspace / 'sample.txt').write_bytes` → `SQLiteArtifactStore` → `store.initialize`
# → `SQLiteEvidenceStore` → `evidence.initialize`；另有 11 个调用点。
# 分支与异常：
#   验证条件：`result.success`。
#   验证条件：`receipt['task_id'] is None`。
#   验证条件：`result.evidence_id is not None`。
#   验证条件：`resolver.calls == 1`。
# 副作用与资源：
#   文件或资源访问：`workspace.mkdir`、`(workspace / 'sample.txt').write_bytes`。
async def test_task_attribution_failure_does_not_block_publication(tmp_path):
    class BrokenResolver:
        # 函数说明：test_task_attribution_failure_does_not_block_publication.
        # BrokenResolver.__init__
        # 用途：初始化 BrokenResolver；参数及实际保存的实例字段见下方说明。
        # 返回：不返回结果值（隐式 None）。
        # 副作用与资源：
        #   更新对象字段：`self.calls`。
        def __init__(self):
            self.calls = 0

        # 函数说明：test_task_attribution_failure_does_not_block_publication.
        # BrokenResolver.resolve
        # 用途：解析或定位BrokenResolver，供回归测试与测试辅助使用。
        # 参数：
        #   conversation_id：目标会话标识。
        # 返回：不返回结果值（隐式 None）。
        # 副作用与资源：
        #   更新对象字段：`self.calls`。
        async def resolve(self, conversation_id):
            self.calls += 1
            raise OSError("task store unavailable")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "sample.txt").write_bytes(b"still publish")
    store = SQLiteArtifactStore(tmp_path / "test.db")
    await store.initialize()
    evidence = SQLiteEvidenceStore(tmp_path / "test.db")
    await evidence.initialize()
    service = ArtifactService(store, workspace, managed_dir=tmp_path / "published")
    registry = ToolRegistry()
    resolver = BrokenResolver()
    register_artifact_tools(
        registry,
        service,
        attribution_resolver=resolver,
    )
    executor = ToolExecutor(
        registry,
        output_recorder=EvidenceRecorder(
            evidence,
            attribution_resolver=resolver,
        ),
    )
    call = ToolCall(
        id="publish-without-attribution",
        name="artifact_publish",
        arguments={"path": "sample.txt"},
    )
    result = await executor.execute(call, context=context(call, AgentMode.EXECUTE))
    assert result.success, result.error
    receipt = json.loads(result.output)
    assert receipt["task_id"] is None
    assert result.evidence_id is not None
    assert resolver.calls == 1
    assert len(await service.store.list()) == 1


# 函数说明：test_mea_task_metadata_stays_exact_through_run_manager
# 用途：回归验证回归测试与测试辅助中的
# `mea_task_metadata_stays_exact_through_run_manager` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`workspace.mkdir` →
# `(workspace / 'sample.txt').write_bytes` → `SQLiteArtifactStore` →
# `SQLiteEvidenceStore` → `artifacts.initialize` → `evidence.initialize`；另有 26 个调用
# 点。
# 分支与异常：
#   验证条件：`fallback.task_id == unrelated_task.id`。
#   验证条件：`run.status is RunStatus.COMPLETED`。
#   验证条件：`result is not None`。
#   验证条件：`record.result.success`。
# 副作用与资源：
#   文件或资源访问：`workspace.mkdir`、`(workspace / 'sample.txt').write_bytes`。
async def test_mea_task_metadata_stays_exact_through_run_manager(tmp_path):
    from app.domain.task import (
        FileTaskStore,
        TaskStatus,
        TaskToolOutputAttributionResolver,
    )
    from tests.offline.tools.test_role_boundary import _model_registry, _response

    class CountingResolver:
        # 函数说明：test_mea_task_metadata_stays_exact_through_run_manager.
        # CountingResolver.__init__
        # 用途：初始化 CountingResolver；参数及实际保存的实例字段见下方说明。
        # 参数：
        #   delegate：`delegate`输入或配置值。
        # 返回：不返回结果值（隐式 None）。
        # 副作用与资源：
        #   更新对象字段：`self.delegate`、`self.calls`。
        def __init__(self, delegate):
            self.delegate = delegate
            self.calls = 0

        # 函数说明：test_mea_task_metadata_stays_exact_through_run_manager.
        # CountingResolver.resolve
        # 用途：解析或定位CountingResolver，供回归测试与测试辅助使用。
        # 参数：
        #   conversation_id：目标会话标识。
        # 返回：返回 `await self.delegate.resolve(conversation_id)`。
        # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.delegate.resolve`。
        # 副作用与资源：
        #   更新对象字段：`self.calls`。
        async def resolve(self, conversation_id):
            self.calls += 1
            return await self.delegate.resolve(conversation_id)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "sample.txt").write_bytes(b"exact MEA attribution")
    database = tmp_path / "test.db"
    artifacts = SQLiteArtifactStore(database)
    evidence = SQLiteEvidenceStore(database)
    await artifacts.initialize()
    await evidence.initialize()
    service = ArtifactService(
        artifacts,
        workspace,
        managed_dir=tmp_path / "published",
    )
    tasks = FileTaskStore(tmp_path / "tasks")
    await tasks.initialize()
    intended_task = await tasks.create(
        title="MEA target",
        owner_conversation_id="conversation",
    )
    await tasks.set_status(intended_task.id, TaskStatus.ACTIVE)
    unrelated_task = await tasks.create(
        title="Newer task in the same conversation",
        owner_conversation_id="conversation",
    )
    await tasks.set_status(unrelated_task.id, TaskStatus.ACTIVE)
    fallback_resolver = TaskToolOutputAttributionResolver(tasks)
    fallback = await fallback_resolver.resolve("conversation")
    assert fallback.task_id == unrelated_task.id
    resolver = CountingResolver(fallback_resolver)
    tools = ToolRegistry()
    register_artifact_tools(tools, service, attribution_resolver=resolver)
    models, _ = _model_registry(
        [
            _response(
                tool_calls=(
                    ToolCall(
                        id="publish-exact",
                        name="artifact_publish",
                        arguments={"path": "sample.txt"},
                    ),
                )
            ),
            _response(content="Finished"),
        ]
    )
    checkpoints = SQLiteCheckpointStore(tmp_path / "runs.db")
    await checkpoints.initialize()
    runtime = AgentRuntime(
        models,
        tools,
        provider="fake",
        checkpoint_store=checkpoints,
        tool_output_recorder=EvidenceRecorder(
            evidence,
            attribution_resolver=resolver,
        ),
    )
    manager = RunManager(
        SQLiteRunStore(tmp_path / "runs.db"),
        checkpoints,
        runtime,
    )
    await manager.initialize()

    run_id, _ = await manager.start(
        "Publish sample.txt",
        conversation_id="conversation",
        mode=AgentMode.EXECUTE,
        tool_context_metadata={
            "task_id": intended_task.id,
            "task_step_id": "s6",
            "mea_id": "mea-1",
        },
    )
    run = await manager.wait(run_id)

    assert run.status is RunStatus.COMPLETED
    result = manager.result(run_id)
    assert result is not None
    [record] = result.tool_calls
    assert record.result.success, record.result.error
    receipt = json.loads(record.result.output)
    assert receipt["task_id"] == intended_task.id
    stored_artifact = await service.store.get(receipt["id"])
    assert stored_artifact is not None
    assert stored_artifact.task_id == intended_task.id
    stored_evidence = await evidence.resolve(
        record.result.evidence_id,
        conversation_id="conversation",
    )
    assert stored_evidence is not None
    assert stored_evidence.record.task_id == intended_task.id
    assert stored_evidence.record.task_step_id == "s6"
    assert resolver.calls == 0
    await models.close()


# 函数说明：test_readonly_roles_cannot_publish_or_create_artifacts
# 用途：回归验证回归测试与测试辅助中的
# `readonly_roles_cannot_publish_or_create_artifacts` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
#   mode：Agent 执行模式或检索模式。
# 返回：不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`environment` →
# `(workspace / 'sample.txt').write_bytes` → `registry.model_definitions_for_mode` →
# `ToolCall` → `executor.execute` → `context`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`'artifact_publish' not in {d.name for d in registry.
# model_definitions_for_mode(mode)}`。
#   验证条件：`not result.success`。
#   验证条件：`result.error.startswith('Role boundary:')`。
#   验证条件：`not await service.store.list()`。
# 副作用与资源：
#   文件或资源访问：`(workspace / 'sample.txt').write_bytes`。
@pytest.mark.parametrize("mode", [AgentMode.MANAGE, AgentMode.AUDIT])
async def test_readonly_roles_cannot_publish_or_create_artifacts(tmp_path, mode):
    workspace, service, _, registry, executor = await environment(tmp_path)
    (workspace / "sample.txt").write_bytes(b"read only")
    assert "artifact_publish" not in {
        d.name for d in registry.model_definitions_for_mode(mode)
    }
    call = ToolCall(
        id="denied", name="artifact_publish", arguments={"path": "sample.txt"}
    )
    result = await executor.execute(call, context=context(call, mode))
    assert not result.success
    assert result.error.startswith("Role boundary:")
    assert not await service.store.list()
    assert not service.managed_dir.exists()


# 函数说明：test_publication_does_not_bypass_path_or_identity_checks
# 用途：回归验证回归测试与测试辅助中的
# `publication_does_not_bypass_path_or_identity_checks` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
#   arguments：工具调用的参数对象或 JSON 文本。
# 返回：不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`environment` →
# `(workspace / 'sample.txt').write_bytes` → `(tmp_path / 'outside.txt').write_bytes` →
# `ToolCall` → `executor.execute` → `context`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`not result.success`。
#   验证条件：`not await service.store.list()`。
# 副作用与资源：
#   文件或资源访问：`(workspace / 'sample.txt').write_bytes`、
# `(tmp_path / 'outside.txt').write_bytes`。
@pytest.mark.parametrize(
    "arguments",
    [
        {"path": "../outside.txt"},
        {"path": "missing.txt"},
        {"path": "sample.txt", "run_id": "forged"},
        {"path": "sample.txt", "task_id": "forged"},
    ],
)
async def test_publication_does_not_bypass_path_or_identity_checks(tmp_path, arguments):
    workspace, service, _, _, executor = await environment(tmp_path)
    (workspace / "sample.txt").write_bytes(b"inside")
    (tmp_path / "outside.txt").write_bytes(b"outside")
    call = ToolCall(id="bad", name="artifact_publish", arguments=arguments)
    result = await executor.execute(call, context=context(call, AgentMode.NORMAL))
    assert not result.success
    assert not await service.store.list()


# 函数说明：test_auditor_reads_executor_receipt_by_active_task
# 用途：回归验证回归测试与测试辅助中的 `auditor_reads_executor_receipt_by_active_task`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`environment` → `FileTaskStore` →
# `tasks.initialize` → `tasks.create` → `TaskStep` → `tasks.apply_patch`；另有 16 个调用
# 点。
# 分支与异常：
#   验证条件：`published.success`。
#   验证条件：`receipt['task_id'] == task.id`。
#   验证条件：`(await service.store.get(receipt['id'])).task_id == task.id`。
#   验证条件：`found.success`。
# 副作用与资源：
#   文件或资源访问：`(workspace / 'sample.txt').write_bytes`。
async def test_auditor_reads_executor_receipt_by_active_task(tmp_path):
    from app.domain.task import (
        FileTaskStore,
        TaskPatch,
        TaskStatus,
        TaskStep,
        TaskStepStatus,
        TaskToolOutputAttributionResolver,
    )
    from app.records.evidence import EvidenceReadTool, EvidenceSearchTool

    workspace, service, evidence, _, _ = await environment(tmp_path)
    tasks = FileTaskStore(tmp_path / "tasks")
    await tasks.initialize()
    task = await tasks.create(
        title="Deliver",
        owner_conversation_id="conversation",
        steps=(TaskStep(id="s6", title="Publish"),),
    )
    await tasks.apply_patch(task.id, TaskPatch(status=TaskStatus.ACTIVE))
    await tasks.apply_patch(
        task.id, TaskPatch(step_id="s6", step_status=TaskStepStatus.IN_PROGRESS)
    )
    class CountingResolver:
        # 函数说明：
        # test_auditor_reads_executor_receipt_by_active_task.CountingResolver.__init__
        # 用途：初始化 CountingResolver；参数及实际保存的实例字段见下方说明。
        # 参数：
        #   delegate：`delegate`输入或配置值。
        # 返回：不返回结果值（隐式 None）。
        # 副作用与资源：
        #   更新对象字段：`self.delegate`、`self.calls`。
        def __init__(self, delegate):
            self.delegate = delegate
            self.calls = 0

        # 函数说明：
        # test_auditor_reads_executor_receipt_by_active_task.CountingResolver.resolve
        # 用途：解析或定位CountingResolver，供回归测试与测试辅助使用。
        # 参数：
        #   conversation_id：目标会话标识。
        # 返回：返回 `await self.delegate.resolve(conversation_id)`。
        # 关键调用（按源码出现顺序，实际执行取决于分支）：`self.delegate.resolve`。
        # 副作用与资源：
        #   更新对象字段：`self.calls`。
        async def resolve(self, conversation_id):
            self.calls += 1
            return await self.delegate.resolve(conversation_id)

    resolver = CountingResolver(TaskToolOutputAttributionResolver(tasks))
    registry = ToolRegistry()
    register_artifact_tools(registry, service, attribution_resolver=resolver)
    registry.register(EvidenceSearchTool(evidence))
    registry.register(EvidenceReadTool(evidence))
    executor = ToolExecutor(
        registry,
        output_recorder=EvidenceRecorder(evidence, attribution_resolver=resolver),
    )
    (workspace / "sample.txt").write_bytes(b"published content")
    call = ToolCall(
        id="publish", name="artifact_publish", arguments={"path": "sample.txt"}
    )
    published = await executor.execute(call, context=context(call, AgentMode.EXECUTE))
    assert published.success, published.error
    receipt = json.loads(published.output)
    assert receipt["task_id"] == task.id
    assert (await service.store.get(receipt["id"])).task_id == task.id
    search = ToolCall(
        id="search",
        name="evidence_search",
        arguments={
            "query": task.id,
            "task_id": task.id,
            "tool_name": "artifact_publish",
        },
    )
    found = await executor.execute(search, context=context(search, AgentMode.AUDIT))
    assert found.success, found.error
    hits = json.loads(found.output)
    assert hits["count"] == 1
    assert hits["results"][0]["task_step_id"] == "s6"
    read = ToolCall(
        id="read",
        name="evidence_read",
        arguments={"evidence_id": published.evidence_id},
    )
    audited = await executor.execute(read, context=context(read, AgentMode.AUDIT))
    assert audited.success, audited.error
    assert json.loads(json.loads(audited.output)["content"]) == receipt
    assert len(await service.store.list()) == 1
    assert resolver.calls == 1


# 函数说明：test_model_tool_round_obeys_publication_boundary
# 用途：回归验证回归测试与测试辅助中的 `model_tool_round_obeys_publication_boundary` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
#   mode：Agent 执行模式或检索模式。
# 返回：不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`environment` →
# `(workspace / 'sample.txt').write_bytes` → `ToolCall` → `_model_registry` →
# `_response` → `AgentRuntime`；另有 5 个调用点。
# 分支与异常：
#   验证条件：`'artifact_publish' in visible`。
#   验证条件：`record.result.success`。
#   验证条件：`record.result.evidence_id is not None`。
#   验证条件：`len(await service.store.list()) == 1`。
# 副作用与资源：
#   文件或资源访问：`(workspace / 'sample.txt').write_bytes`。
@pytest.mark.parametrize("mode", [AgentMode.EXECUTE, AgentMode.MANAGE, AgentMode.AUDIT])
async def test_model_tool_round_obeys_publication_boundary(tmp_path, mode):
    from app.runtime.agent.runtime import AgentRuntime
    from tests.offline.tools.test_role_boundary import _model_registry, _response

    workspace, service, evidence, tools, _ = await environment(tmp_path)
    (workspace / "sample.txt").write_bytes(b"model tool round")
    call = ToolCall(
        id="publish", name="artifact_publish", arguments={"path": "sample.txt"}
    )
    models, adapter = _model_registry(
        [_response(tool_calls=(call,)), _response(content="Finished")]
    )
    runtime = AgentRuntime(
        models, tools, provider="fake", tool_output_recorder=EvidenceRecorder(evidence)
    )
    result = await runtime.run(
        "Publish sample.txt",
        conversation_id="conversation",
        run_id="role-run",
        mode=mode,
    )
    visible = {definition.name for definition in adapter.requests[0].tools}
    [record] = result.tool_calls
    if mode is AgentMode.EXECUTE:
        assert "artifact_publish" in visible
        assert record.result.success, record.result.error
        assert record.result.evidence_id is not None
        assert len(await service.store.list()) == 1
    else:
        assert "artifact_publish" not in visible
        assert not record.result.success
        assert record.result.error.startswith("Role boundary:")
        assert not await service.store.list()
    await models.close()


# 函数说明：test_recovery_audit_can_read_committed_publication
# 用途：回归验证回归测试与测试辅助中的 `recovery_audit_can_read_committed_publication`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
#   monkeypatch：pytest 提供的临时替换依赖夹具。
#   window：`window`输入或配置值。
# 返回：不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`environment` →
# `(workspace / 'sample.txt').write_bytes` → `ToolCall` → `_model_registry` →
# `_response` → `SQLiteCheckpointStore`；另有 20 个调用点。
# 分支与异常：
#   验证条件：`result.tool_calls[0].result.success`。
#   验证条件：`result.tool_calls[0].result.evidence_error`。
#   验证条件：`checkpoint is not None`。
#   验证条件：`[pending.id for pending in checkpoint.pending_tool_calls] == [call.id]`。
#   预期异常：`pytest.raises(SimulatedPowerLoss)`。
# 副作用与资源：
#   文件或资源访问：`(workspace / 'sample.txt').write_bytes`、`copied.read_bytes`。
@pytest.mark.parametrize("window", ["after_commit", "after_evidence", "evidence_failure"])
async def test_recovery_audit_can_read_committed_publication(tmp_path, monkeypatch, window):
    from tests.offline.tools.test_role_boundary import _model_registry, _response

    class SimulatedPowerLoss(BaseException):
        pass

    # 函数说明：test_recovery_audit_can_read_committed_publication.lose_power
    # 用途：处理回归测试与测试辅助中的 `lose_power` 数据；结果及边界条件见下方说明。
    # 参数：
    #   *args：额外位置参数，按实现向内部调用传递。
    #   **kwargs：额外关键字参数，按实现处理或转交。
    # 返回：不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`SimulatedPowerLoss`。
    async def lose_power(*args, **kwargs):
        raise SimulatedPowerLoss()

    # 函数说明：test_recovery_audit_can_read_committed_publication.evidence_unavailable
    # 用途：处理回归测试与测试辅助中的 `evidence_unavailable` 数据；结果及边界条件见下方
    # 说明。
    # 参数：
    #   *args：额外位置参数，按实现向内部调用传递。
    #   **kwargs：额外关键字参数，按实现处理或转交。
    # 返回：不返回结果值（隐式 None）。
    async def evidence_unavailable(*args, **kwargs):
        raise OSError("evidence store unavailable")

    workspace, service, evidence, tools, _ = await environment(tmp_path)
    content = b"committed publication survives interrupted tool round"
    (workspace / "sample.txt").write_bytes(content)
    call = ToolCall(
        id="interrupted-publish", name="artifact_publish",
        arguments={"path": "sample.txt"},
    )
    models, _ = _model_registry([
        _response(tool_calls=(call,)), _response(content="Finished"),
    ])
    checkpoints = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    await checkpoints.initialize()
    runtime = AgentRuntime(
        models, tools, provider="fake", checkpoint_store=checkpoints,
        tool_output_recorder=EvidenceRecorder(evidence),
    )
    if window == "after_commit":
        monkeypatch.setattr(service, "_notify", lose_power)
    elif window == "after_evidence":
        monkeypatch.setattr(checkpoints, "complete_tool", lose_power)
    else:
        monkeypatch.setattr(evidence, "create", evidence_unavailable)
    kwargs = {
        "conversation_id": "conversation", "run_id": "interrupted-run",
        "mode": AgentMode.EXECUTE,
        "tool_context_metadata": {"task_id": "task-1", "task_step_id": "s6"},
    }
    if window == "evidence_failure":
        result = await runtime.run("Publish sample.txt", **kwargs)
        assert result.tool_calls[0].result.success
        assert result.tool_calls[0].result.evidence_error
    else:
        with pytest.raises(SimulatedPowerLoss):
            await runtime.run("Publish sample.txt", **kwargs)
        checkpoint = await checkpoints.get("interrupted-run")
        assert checkpoint is not None
        assert [pending.id for pending in checkpoint.pending_tool_calls] == [call.id]

    # Reconstruct storage and tools like a new process. No replay or new publish.
    restarted_service = ArtifactService(
        SQLiteArtifactStore(tmp_path / "test.db"), workspace,
        managed_dir=tmp_path / "published",
    )
    restarted_tools = ToolRegistry()
    register_artifact_tools(restarted_tools, restarted_service)
    audit_executor = ToolExecutor(restarted_tools)
    query = ToolCall(
        id="recover-receipt", name="artifact_list",
        arguments={"task_id": "task-1", "run_id": "interrupted-run"},
    )
    result = await audit_executor.execute(query, context=context(query, AgentMode.AUDIT))
    assert result.success, result.error
    payload = json.loads(result.output)
    assert payload["record_source"] == "committed_artifact_store"
    assert payload["count"] == 1
    [receipt] = payload["artifacts"]
    assert receipt["size_bytes"] == len(content)
    assert receipt["sha256"] == hashlib.sha256(content).hexdigest()
    copied = await restarted_service.file_path(receipt["id"])
    assert copied.read_bytes() == content
    assert len(await restarted_service.store.list()) == 1
    await models.close()


# 函数说明：test_publication_queries_are_conversation_private_and_readonly
# 用途：回归验证回归测试与测试辅助中的
# `publication_queries_are_conversation_private_and_readonly` 场景，下方断言说明列出实际
# 通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`environment` →
# `(workspace / 'sample.txt').write_bytes` → `service.publish_file` → `ToolCall` →
# `ToolExecutionContext` → `executor.execute`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`hidden.success`。
#   验证条件：`json.loads(hidden.output)['count'] == 0`。
#   验证条件：`'artifact_list' not in {definition.name for definition in tools.
# model_definitions_for_mode(AgentMode.MANAGE)}`。
#   验证条件：`len(await service.store.list()) == 1`。
# 副作用与资源：
#   文件或资源访问：`(workspace / 'sample.txt').write_bytes`。
async def test_publication_queries_are_conversation_private_and_readonly(tmp_path):
    workspace, service, _, tools, executor = await environment(tmp_path)
    (workspace / "sample.txt").write_bytes(b"private receipt")
    await service.publish_file(
        path="sample.txt", conversation_id="conversation", task_id="task-1",
    )
    query = ToolCall(id="query", name="artifact_list", arguments={"task_id": "task-1"})
    private_context = ToolExecutionContext(
        tool_call=query, conversation_id="other", mode=AgentMode.AUDIT,
    )
    hidden = await executor.execute(query, context=private_context)
    assert hidden.success, hidden.error
    assert json.loads(hidden.output)["count"] == 0
    assert "artifact_list" not in {
        definition.name for definition in tools.model_definitions_for_mode(AgentMode.MANAGE)
    }
    assert len(await service.store.list()) == 1
