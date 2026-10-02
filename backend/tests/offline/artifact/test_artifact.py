
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.domain.artifact import (
    Artifact,
    ArtifactKind,
    ArtifactPublishTool,
    ArtifactService,
    ArtifactTooLargeError,
    SQLiteArtifactStore,
    register_artifact_tools,
)
from app.models.types import ToolCall
from app.server.app import artifact_content
from app.server.rpc.dispatcher import RpcContext
from app.server.rpc.methods import artifacts as artifacts_rpc
from app.server.rpc.protocol import JsonRpcError, RpcErrorCode
from app.tools.hooks import ToolExecutionContext
from app.tools.registry import ToolRegistry


# 函数说明：_make_service
# 用途：构造`service`，供回归测试与测试辅助使用。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
#   workspace_name：工作区名称输入或配置值；默认 `'workspace'`。
# 返回：类型 `ArtifactService`；返回 `service`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteArtifactStore` →
# `store.initialize` → `workspace.mkdir` → `ArtifactService`。
# 副作用与资源：
#   文件或资源访问：`workspace.mkdir`。
async def _make_service(tmp_path, *, workspace_name="workspace") -> ArtifactService:
    store = SQLiteArtifactStore(tmp_path / "artifacts.db")
    await store.initialize()
    workspace = tmp_path / workspace_name
    workspace.mkdir(parents=True, exist_ok=True)
    service = ArtifactService(
        store, workspace, managed_dir=tmp_path / "managed-artifacts"
    )
    return service


# 函数说明：_write
# 用途：写入回归测试与测试辅助，供回归测试与测试辅助使用。
# 参数：
#   path：目标文件或目录路径，类型 `Path`。
#   content：内容正文，类型 `bytes | str`。
# 返回：类型 `Path`；返回 `path`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`path.parent.mkdir` →
# `path.write_bytes` → `content.encode`。
# 副作用与资源：
#   文件或资源访问：`path.parent.mkdir`、`path.write_bytes`。
def _write(path: Path, content: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    return path


# 函数说明：_ctx
# 用途：返回 `RpcContext(application=app, connection=None)`，提供 回归测试与测试辅助 的
# 派生值。
# 参数：
#   app：应用实例。
# 返回：类型 `RpcContext`；返回 `RpcContext(application=app, connection=None)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`RpcContext`。
def _ctx(app) -> RpcContext:
    return RpcContext(application=app, connection=None)  


# 函数说明：_tool_context
# 用途：返回 `ToolExecutionContext(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   run_id：目标运行标识；默认 `'run-1'`。
#   conversation_id：目标会话标识；默认 `'conv-1'`。
# 返回：类型 `ToolExecutionContext`；返回 `ToolExecutionContext(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ToolExecutionContext` → `ToolCall`。
def _tool_context(run_id="run-1", conversation_id="conv-1") -> ToolExecutionContext:
    return ToolExecutionContext(
        tool_call=ToolCall(id="t1", name="artifact_publish", arguments={}),
        run_id=run_id,
        conversation_id=conversation_id,
    )




# 函数说明：test_store_create_get_list
# 用途：回归验证回归测试与测试辅助中的 `store_create_get_list` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteArtifactStore` →
# `store.initialize` → `Artifact` → `store.create` → `store.list`。
# 分支与异常：
#   验证条件：`await store.get(a.id) is not None`。
#   验证条件：`len(items) == 1`。
#   验证条件：`items[0].id == a.id`。
async def test_store_create_get_list(tmp_path) -> None:
    store = SQLiteArtifactStore(tmp_path / "a.db")
    await store.initialize()
    a = Artifact(
        kind=ArtifactKind.FILE,
        title="Report",
        filename="report.md",
        run_id="run-1",
        conversation_id="conv-1",
        sha256="abc",
        size_bytes=3,
    )
    await store.create(a)
    assert (await store.get(a.id)) is not None
    items = await store.list()
    assert len(items) == 1
    assert items[0].id == a.id


# 函数说明：test_store_filters_and_durable
# 用途：回归验证回归测试与测试辅助中的 `store_filters_and_durable` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteArtifactStore` →
# `store.initialize` → `Artifact` → `store.create` → `store.list` → `store2.initialize`
# ；另有 1 个调用点。
# 分支与异常：
#   验证条件：`{a.id for a in await store.list(run_id='run-1')} == {a1.id, a2.id}`。
#   验证条件：
# `{a.id for a in await store.list(conversation_id='conv-1')} == {a1.id, a3.id}`。
#   验证条件：`{a.id for a in await store.list(run_id='run-1', conversation_id='conv-1')
# } == {a1.id}`。
#   验证条件：`await store2.get(a1.id) is not None`。
async def test_store_filters_and_durable(tmp_path) -> None:
    store = SQLiteArtifactStore(tmp_path / "a.db")
    await store.initialize()
    a1 = Artifact(
        kind=ArtifactKind.FILE, title="A", run_id="run-1", conversation_id="conv-1"
    )
    a2 = Artifact(
        kind=ArtifactKind.URL, title="B", run_id="run-1", conversation_id="conv-2"
    )
    a3 = Artifact(
        kind=ArtifactKind.FILE, title="C", run_id="run-2", conversation_id="conv-1"
    )
    for a in (a1, a2, a3):
        await store.create(a)

    assert {a.id for a in await store.list(run_id="run-1")} == {a1.id, a2.id}
    assert {a.id for a in await store.list(conversation_id="conv-1")} == {
        a1.id,
        a3.id,
    }
    assert {
        a.id for a in await store.list(run_id="run-1", conversation_id="conv-1")
    } == {a1.id}

    store2 = SQLiteArtifactStore(tmp_path / "a.db")
    await store2.initialize()
    assert (await store2.get(a1.id)) is not None
    assert len(await store2.list()) == 3




# 函数说明：test_publish_file_ok_and_immutable
# 用途：回归验证回归测试与测试辅助中的 `publish_file_ok_and_immutable` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_service` → `_write` →
# `service.publish_file` → `service.file_path` → `copied.read_text` →
# `str(copied).startswith`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`artifact.kind is ArtifactKind.FILE`。
#   验证条件：`artifact.title == 'Report'`。
#   验证条件：`artifact.filename == 'report.md'`。
#   验证条件：`artifact.run_id == 'run-1'`。
# 副作用与资源：
#   文件或资源访问：`copied.read_text`、`source.write_text`。
async def test_publish_file_ok_and_immutable(tmp_path) -> None:
    service = await _make_service(tmp_path)
    source = _write(
        tmp_path / "workspace" / "reports" / "report.md", "hello artifact"
    )
    artifact = await service.publish_file(
        path="reports/report.md",
        title="Report",
        run_id="run-1",
        conversation_id="conv-1",
    )
    assert artifact.kind is ArtifactKind.FILE
    assert artifact.title == "Report"
    assert artifact.filename == "report.md"
    assert artifact.run_id == "run-1"
    assert artifact.size_bytes == len("hello artifact")

    copied = await service.file_path(artifact.id)
    assert copied is not None
    assert copied.read_text() == "hello artifact"
    assert str(copied).startswith(str(service.managed_dir))

    source.write_text("changed later")
    assert copied.read_text() == "hello artifact"


# 函数说明：test_publish_file_rejects_bad_paths
# 用途：回归验证回归测试与测试辅助中的 `publish_file_rejects_bad_paths` 场景，下方断言说
# 明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_service` → `_write` →
# `pytest.raises` → `service.publish_file` → `link.symlink_to` → `pytest.skip`。
# 分支与异常：
#   捕获 `OSError` 后，执行异常处理调用 `pytest.skip`。
#   预期异常：`pytest.raises(ValueError, match='relative')`。
#   预期异常：`pytest.raises(ValueError, match='escapes')`。
#   预期异常：`pytest.raises(ValueError)`。
#   预期异常：`pytest.raises(ValueError, match='file')`。
async def test_publish_file_rejects_bad_paths(tmp_path) -> None:
    service = await _make_service(tmp_path)
    _write(tmp_path / "workspace" / "ok.txt", "ok")
    _write(tmp_path / "outside.txt", "outside")  

    with pytest.raises(ValueError, match="relative"):
        await service.publish_file(path=str(tmp_path / "workspace" / "ok.txt"))
    with pytest.raises(ValueError, match="escapes"):
        await service.publish_file(path="../outside.txt")
    with pytest.raises(ValueError):
        await service.publish_file(path=".")
    with pytest.raises(ValueError, match="file"):
        await service.publish_file(path="reports")  

    link = tmp_path / "workspace" / "escape.txt"
    try:
        link.symlink_to(tmp_path / "outside.txt")
    except OSError:
        pytest.skip("当前 Windows 账户没有创建符号链接的权限")
    with pytest.raises(ValueError, match="escapes"):
        await service.publish_file(path="escape.txt")


# 函数说明：test_publish_file_sha256_and_mime
# 用途：回归验证回归测试与测试辅助中的 `publish_file_sha256_and_mime` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_service` → `_write` →
# `service.publish_file` → `hashlib.sha256(b'a,b\n1,2\n').hexdigest` → `hashlib.sha256`
# 。
# 分支与异常：
#   验证条件：`artifact.sha256 == hashlib.sha256(b'a,b\n1,2\n').hexdigest()`。
#   验证条件：`artifact.mime_type == 'text/csv'`。
async def test_publish_file_sha256_and_mime(tmp_path) -> None:
    service = await _make_service(tmp_path)
    _write(tmp_path / "workspace" / "data.csv", "a,b\n1,2\n")
    artifact = await service.publish_file(path="data.csv", run_id="run-1")
    import hashlib

    assert artifact.sha256 == hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    assert artifact.mime_type == "text/csv"


# 函数说明：test_publish_file_too_large
# 用途：回归验证回归测试与测试辅助中的 `publish_file_too_large` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   monkeypatch：pytest 提供的临时替换依赖夹具。
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`monkeypatch.setattr` →
# `_make_service` → `_write` → `pytest.raises` → `service.publish_file`。
# 分支与异常：
#   预期异常：`pytest.raises(ArtifactTooLargeError)`。
async def test_publish_file_too_large(monkeypatch, tmp_path) -> None:
    import app.domain.artifact.service as service_module

    monkeypatch.setattr(service_module, "MAX_ARTIFACT_BYTES", 10)
    service = await _make_service(tmp_path)
    _write(tmp_path / "workspace" / "big.bin", "x" * 20)
    with pytest.raises(ArtifactTooLargeError):
        await service.publish_file(path="big.bin")




# 函数说明：test_publish_url_ok
# 用途：回归验证回归测试与测试辅助中的 `publish_url_ok` 场景，下方断言说明列出实际通过条
# 件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_service` →
# `service.publish_url` → `service.file_path`。
# 分支与异常：
#   验证条件：`a.kind is ArtifactKind.URL`。
#   验证条件：`a.source_url == 'https://example.com/result'`。
#   验证条件：`a.size_bytes == 0`。
#   验证条件：`await service.file_path(a.id) is None`。
async def test_publish_url_ok(tmp_path) -> None:
    service = await _make_service(tmp_path)
    a = await service.publish_url(
        url="https://example.com/result", title="Result", run_id="run-1"
    )
    assert a.kind is ArtifactKind.URL
    assert a.source_url == "https://example.com/result"
    assert a.size_bytes == 0
    assert await service.file_path(a.id) is None


# 函数说明：test_publish_url_rejects_bad_schemes
# 用途：回归验证回归测试与测试辅助中的 `publish_url_rejects_bad_schemes` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
#   url：目标 HTTP 地址，类型 `str`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_service` → `pytest.raises` →
# `service.publish_url`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='http')`。
@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "javascript:alert(1)", "data:text/plain,hi"]
)
async def test_publish_url_rejects_bad_schemes(tmp_path, url: str) -> None:
    service = await _make_service(tmp_path)
    with pytest.raises(ValueError, match="http"):
        await service.publish_url(url=url)




# 函数说明：test_tool_binds_context_and_rejects_forged_ids
# 用途：回归验证回归测试与测试辅助中的 `tool_binds_context_and_rejects_forged_ids` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_service` →
# `ArtifactPublishTool` → `_write` → `tool.execute_with_context` → `_tool_context` →
# `pytest.raises`。
# 分支与异常：
#   验证条件：`result['run_id'] == 'real-run'`。
#   验证条件：`result['conversation_id'] == 'real-conv'`。
#   验证条件：`result['kind'] == 'file'`。
#   预期异常：`pytest.raises(ValueError, match='unsupported')`。
async def test_tool_binds_context_and_rejects_forged_ids(tmp_path) -> None:
    service = await _make_service(tmp_path)
    tool = ArtifactPublishTool(service)
    _write(tmp_path / "workspace" / "f.txt", "data")

    result = await tool.execute_with_context(
        {"path": "f.txt"},
        _tool_context(run_id="real-run", conversation_id="real-conv"),
    )
    assert result["run_id"] == "real-run"
    assert result["conversation_id"] == "real-conv"
    assert result["kind"] == "file"

    with pytest.raises(ValueError, match="unsupported"):
        await tool.execute_with_context(
            {"path": "f.txt", "run_id": "forged-run"},
            _tool_context(run_id="real-run", conversation_id="real-conv"),
        )


# 函数说明：test_tool_missing_run_context_rejected
# 用途：回归验证回归测试与测试辅助中的 `tool_missing_run_context_rejected` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_service` →
# `ArtifactPublishTool` → `_write` → `pytest.raises` → `tool.execute` →
# `tool.execute_with_context`；另有 1 个调用点。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='run context')`。
async def test_tool_missing_run_context_rejected(tmp_path) -> None:
    service = await _make_service(tmp_path)
    tool = ArtifactPublishTool(service)
    _write(tmp_path / "workspace" / "f.txt", "data")
    with pytest.raises(ValueError, match="run context"):
        await tool.execute({"path": "f.txt"})
    with pytest.raises(ValueError, match="run context"):
        await tool.execute_with_context({"path": "f.txt"}, _tool_context(run_id=None))


# 函数说明：test_tool_path_xor_url
# 用途：回归验证回归测试与测试辅助中的 `tool_path_xor_url` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_service` →
# `ArtifactPublishTool` → `_write` → `pytest.raises` → `tool.execute_with_context` →
# `_tool_context`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='exactly one')`。
async def test_tool_path_xor_url(tmp_path) -> None:
    service = await _make_service(tmp_path)
    tool = ArtifactPublishTool(service)
    _write(tmp_path / "workspace" / "f.txt", "data")
    with pytest.raises(ValueError, match="exactly one"):
        await tool.execute_with_context({}, _tool_context())
    with pytest.raises(ValueError, match="exactly one"):
        await tool.execute_with_context(
            {"path": "f.txt", "url": "https://x"}, _tool_context()
        )


# 函数说明：test_register_artifact_tools
# 用途：回归验证回归测试与测试辅助中的 `register_artifact_tools` 场景，下方断言说明列出
# 实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_service` → `ToolRegistry` →
# `register_artifact_tools` → `registry.names`。
# 分支与异常：
#   验证条件：`'artifact_publish' in registry.names()`。
async def test_register_artifact_tools(tmp_path) -> None:
    service = await _make_service(tmp_path)
    registry = ToolRegistry()
    register_artifact_tools(registry, service)
    assert "artifact_publish" in registry.names()




# 函数说明：test_publish_broadcasts_artifact_created
# 用途：回归验证回归测试与测试辅助中的 `publish_broadcasts_artifact_created` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_service` →
# `service.set_broadcaster` → `_write` → `service.publish_file`。
# 分支与异常：
#   验证条件：`events and events[0][0] == 'artifact.created'`。
#   验证条件：`events[0][1]['artifact']['id'] == artifact.id`。
#   验证条件：`'storage_path' not in events[0][1]['artifact']`。
async def test_publish_broadcasts_artifact_created(tmp_path) -> None:
    service = await _make_service(tmp_path)
    events: list[tuple[str, dict]] = []

    # 函数说明：test_publish_broadcasts_artifact_created.broadcaster
    # 用途：在回归测试与测试辅助中处理 `broadcaster`，通过 `events.append` 完成首个内部
    # 处理步骤。
    # 参数：
    #   method：HTTP 或 RPC 方法名称，类型 `str`。
    #   params：JSON-RPC 方法参数，类型 `dict`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 闭包依赖：从外层读取 `events`。
    async def broadcaster(method: str, params: dict) -> None:
        events.append((method, params))

    service.set_broadcaster(broadcaster)
    _write(tmp_path / "workspace" / "f.txt", "data")
    artifact = await service.publish_file(path="f.txt", run_id="run-1")
    assert events and events[0][0] == "artifact.created"
    assert events[0][1]["artifact"]["id"] == artifact.id
    assert "storage_path" not in events[0][1]["artifact"]


# 函数说明：test_publish_without_broadcaster_still_saves
# 用途：回归验证回归测试与测试辅助中的 `publish_without_broadcaster_still_saves` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_service` → `_write` →
# `service.publish_file`。
# 分支与异常：
#   验证条件：`await service.store.get(artifact.id) is not None`。
async def test_publish_without_broadcaster_still_saves(tmp_path) -> None:
    service = await _make_service(tmp_path)
    _write(tmp_path / "workspace" / "f.txt", "data")
    artifact = await service.publish_file(path="f.txt", run_id="run-1")
    assert (await service.store.get(artifact.id)) is not None


# 函数说明：test_broadcast_failure_does_not_rollback_artifact
# 用途：回归验证回归测试与测试辅助中的 `broadcast_failure_does_not_rollback_artifact` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_service` →
# `service.set_broadcaster` → `_write` → `service.publish_file`。
# 分支与异常：
#   验证条件：`await service.store.get(artifact.id) is not None`。
async def test_broadcast_failure_does_not_rollback_artifact(tmp_path) -> None:
    service = await _make_service(tmp_path)

    # 函数说明：test_broadcast_failure_does_not_rollback_artifact.broken_broadcaster
    # 用途：处理回归测试与测试辅助中的 `broken_broadcaster` 数据；结果及边界条件见下方说
    # 明。
    # 参数：
    #   method：HTTP 或 RPC 方法名称，类型 `str`。
    #   params：JSON-RPC 方法参数，类型 `dict`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    async def broken_broadcaster(method: str, params: dict) -> None:
        raise RuntimeError("offline")

    service.set_broadcaster(broken_broadcaster)
    _write(tmp_path / "workspace" / "f.txt", "data")
    artifact = await service.publish_file(path="f.txt", run_id="run-1")
    assert (await service.store.get(artifact.id)) is not None


# 函数说明：test_store_failure_cleans_managed_copy
# 用途：回归验证回归测试与测试辅助中的 `store_failure_cleans_managed_copy` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
#   monkeypatch：pytest 提供的临时替换依赖夹具。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_service` → `_write` →
# `monkeypatch.setattr` → `pytest.raises` → `service.publish_file` →
# `service.managed_dir.glob`。
# 分支与异常：
#   验证条件：`list(service.managed_dir.glob('*')) == []`。
#   预期异常：`pytest.raises(RuntimeError, match='database unavailable')`。
async def test_store_failure_cleans_managed_copy(tmp_path, monkeypatch) -> None:
    service = await _make_service(tmp_path)
    _write(tmp_path / "workspace" / "f.txt", "data")

    # 函数说明：test_store_failure_cleans_managed_copy.broken_create
    # 用途：创建`broken`，供回归测试与测试辅助使用。
    # 参数：
    #   artifact：交付物输入或配置值，类型 `Artifact`。
    # 返回：类型 `Artifact`；不返回结果值（隐式 None）。
    async def broken_create(artifact: Artifact) -> Artifact:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(service.store, "create", broken_create)
    with pytest.raises(RuntimeError, match="database unavailable"):
        await service.publish_file(path="f.txt", run_id="run-1")
    assert list(service.managed_dir.glob("*")) == []




# 函数说明：test_rpc_artifact_list_and_get
# 用途：回归验证回归测试与测试辅助中的 `rpc_artifact_list_and_get` 场景，下方断言说明列
# 出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SQLiteArtifactStore` →
# `store.initialize` → `Artifact` → `store.create` → `SimpleNamespace` →
# `artifacts_rpc.artifact_list`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`result['count'] == 1`。
#   验证条件：`result['artifacts'][0]['id'] == a.id`。
#   验证条件：`'storage_path' not in result['artifacts'][0]`。
#   验证条件：`got['artifact']['title'] == 'R'`。
#   预期异常：`pytest.raises(JsonRpcError)`。
async def test_rpc_artifact_list_and_get(tmp_path) -> None:
    store = SQLiteArtifactStore(tmp_path / "a.db")
    await store.initialize()
    a = Artifact(kind=ArtifactKind.FILE, title="R", filename="r.md", run_id="run-1")
    await store.create(a)
    app = SimpleNamespace(artifact_store=store)

    result = await artifacts_rpc.artifact_list({"run_id": "run-1"}, _ctx(app))
    assert result["count"] == 1
    assert result["artifacts"][0]["id"] == a.id
    assert "storage_path" not in result["artifacts"][0]

    got = await artifacts_rpc.artifact_get({"id": a.id}, _ctx(app))
    assert got["artifact"]["title"] == "R"

    with pytest.raises(JsonRpcError) as exc:
        await artifacts_rpc.artifact_get({}, _ctx(app))
    assert exc.value.code == RpcErrorCode.INVALID_PARAMS
    with pytest.raises(JsonRpcError) as exc:
        await artifacts_rpc.artifact_get({"id": "missing"}, _ctx(app))
    assert exc.value.code == RpcErrorCode.INVALID_PARAMS
    with pytest.raises(JsonRpcError) as exc:
        await artifacts_rpc.artifact_get({"id": "f" * 32}, _ctx(app))
    assert exc.value.code == -32000




# 函数说明：_fake_request
# 用途：返回 `SimpleNamespace(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   fake_app：`fake_app`输入或配置值。
#   host：`host`输入或配置值；默认 `'127.0.0.1'`。
# 返回：返回 `SimpleNamespace(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SimpleNamespace`。
def _fake_request(fake_app, host="127.0.0.1"):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(application=fake_app)),
        client=SimpleNamespace(host=host),
    )


# 函数说明：test_media_endpoint_file_ok
# 用途：回归验证回归测试与测试辅助中的 `media_endpoint_file_ok` 场景，下方断言说明列出实
# 际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_service` → `_write` →
# `service.publish_file` → `SimpleNamespace` → `artifact_content` → `_fake_request`；另
# 有 2 个调用点。
# 分支与异常：
#   验证条件：`response.status_code == 200`。
#   验证条件：`Path(response.path).read_bytes() == b'artifact body'`。
#   验证条件：`response.headers['cache-control'] == 'no-store'`。
#   验证条件：
# `response.headers['content-disposition'] == 'attachment; filename="notes.txt"'`。
# 副作用与资源：
#   文件或资源访问：`Path(response.path).read_bytes`。
async def test_media_endpoint_file_ok(tmp_path) -> None:
    service = await _make_service(tmp_path)
    _write(tmp_path / "workspace" / "notes.txt", "artifact body")
    a = await service.publish_file(path="notes.txt")
    fake_app = SimpleNamespace(artifact_service=service)

    response = await artifact_content(a.id, _fake_request(fake_app))
    assert response.status_code == 200
    assert Path(response.path).read_bytes() == b"artifact body"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-disposition"] == 'attachment; filename="notes.txt"'
    assert response.media_type in ("text/plain", "text/plain; charset=utf-8")


# 函数说明：test_media_endpoint_url_rejected
# 用途：回归验证回归测试与测试辅助中的 `media_endpoint_url_rejected` 场景，下方断言说明
# 列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_service` →
# `service.publish_url` → `SimpleNamespace` → `pytest.raises` → `artifact_content` →
# `_fake_request`。
# 分支与异常：
#   验证条件：`exc.value.status_code == 404`。
#   预期异常：`pytest.raises(HTTPException)`。
async def test_media_endpoint_url_rejected(tmp_path) -> None:
    service = await _make_service(tmp_path)
    a = await service.publish_url(url="https://example.com/x")
    fake_app = SimpleNamespace(artifact_service=service)
    with pytest.raises(HTTPException) as exc:
        await artifact_content(a.id, _fake_request(fake_app))
    assert exc.value.status_code == 404


# 函数说明：test_media_endpoint_missing_and_invalid
# 用途：回归验证回归测试与测试辅助中的 `media_endpoint_missing_and_invalid` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_service` → `SimpleNamespace` →
# `pytest.raises` → `artifact_content` → `_fake_request`。
# 分支与异常：
#   验证条件：`exc.value.status_code == 404`。
#   验证条件：`exc.value.status_code in (403, 404)`。
#   预期异常：`pytest.raises(HTTPException)`。
async def test_media_endpoint_missing_and_invalid(tmp_path) -> None:
    service = await _make_service(tmp_path)
    fake_app = SimpleNamespace(artifact_service=service)
    with pytest.raises(HTTPException) as exc:
        await artifact_content("f" * 32, _fake_request(fake_app))
    assert exc.value.status_code == 404
    with pytest.raises(HTTPException) as exc:
        await artifact_content("../evil", _fake_request(fake_app))
    assert exc.value.status_code in (403, 404)


# 函数说明：test_media_endpoint_non_loopback_rejected
# 用途：回归验证回归测试与测试辅助中的 `media_endpoint_non_loopback_rejected` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_service` → `_write` →
# `service.publish_file` → `SimpleNamespace` → `pytest.raises` → `artifact_content`；另
# 有 1 个调用点。
# 分支与异常：
#   验证条件：`exc.value.status_code == 403`。
#   预期异常：`pytest.raises(HTTPException)`。
async def test_media_endpoint_non_loopback_rejected(tmp_path) -> None:
    service = await _make_service(tmp_path)
    _write(tmp_path / "workspace" / "f.txt", "data")
    a = await service.publish_file(path="f.txt")
    fake_app = SimpleNamespace(artifact_service=service)
    with pytest.raises(HTTPException) as exc:
        await artifact_content(a.id, _fake_request(fake_app, host="10.0.0.5"))
    assert exc.value.status_code == 403


# 函数说明：test_media_endpoint_unavailable_service
# 用途：回归验证回归测试与测试辅助中的 `media_endpoint_unavailable_service` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SimpleNamespace` → `pytest.raises` →
# `artifact_content` → `_fake_request`。
# 分支与异常：
#   验证条件：`exc.value.status_code == 404`。
#   预期异常：`pytest.raises(HTTPException)`。
async def test_media_endpoint_unavailable_service(tmp_path) -> None:
    fake_app = SimpleNamespace(artifact_service=None)
    with pytest.raises(HTTPException) as exc:
        await artifact_content("a" * 32, _fake_request(fake_app))
    assert exc.value.status_code == 404
