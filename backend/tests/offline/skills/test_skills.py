
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from app.domain.skills import (
    ACTIVE_SKILL_MESSAGE_NAME,
    SKILL_CATALOG_MESSAGE_NAME,
    SKILL_READ_TOOL_NAME,
    Skill,
    SkillContextProvider,
    SkillMetadata,
    SkillResources,
    SkillScope,
    SkillStore,
    parse_skill_document,
    register_skill_tools,
    safe_skill_dir,
    safe_skill_file,
    safe_skill_resource,
    valid_skill_name,
    validate_skill_name,
)
from app.domain.skills.parser import SkillParseError
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
)
from app.runtime.agent.events import (
    AgentEventType,
    InMemoryEventHandler,
)
from app.runtime.agent.runtime import AgentRuntime
from app.tools.hooks import ToolExecutionContext
from app.tools.registry import ToolRegistry


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
#   content：内容正文，类型 `str | None`；默认 `None`。
#   tool_calls：待执行的结构化工具调用，类型 `tuple[ToolCall, ...]`；默认 `()`。
# 返回：类型 `ModelResponse`；返回 `ModelResponse(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelResponse` → `Message` →
# `ModelUsage`。
def _model_response(
    *,
    content: str | None = None,
    tool_calls: tuple[ToolCall, ...] = (),
) -> ModelResponse:
    return ModelResponse(
        id="fake-response",
        provider="fake",
        model="fake-model",
        message=Message(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
        ),
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



# 函数说明：test_validate_skill_name_accepts_valid_names
# 用途：回归验证回归测试与测试辅助中的 `validate_skill_name_accepts_valid_names` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   name：目标对象、工具或配置项名称，类型 `str`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`validate_skill_name` →
# `valid_skill_name`。
# 分支与异常：
#   验证条件：`validate_skill_name(name) == name`。
#   验证条件：`valid_skill_name(name)`。
@pytest.mark.parametrize(
    "name",
    [
        "debug-python",
        "code-review",
        "skill1",
        "a",
        "a-b-c",
    ],
)
def test_validate_skill_name_accepts_valid_names(name: str) -> None:
    assert validate_skill_name(name) == name
    assert valid_skill_name(name)


# 函数说明：test_validate_skill_name_rejects_invalid_names
# 用途：回归验证回归测试与测试辅助中的 `validate_skill_name_rejects_invalid_names` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   name：目标对象、工具或配置项名称，类型 `str`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`valid_skill_name` → `pytest.raises` →
#  `validate_skill_name`。
# 分支与异常：
#   验证条件：`not valid_skill_name(name)`。
#   预期异常：`pytest.raises(ValueError)`。
@pytest.mark.parametrize(
    "name",
    [
        "",
        "Debug-Python",  
        "debug_python",  
        "-debug",  
        "debug-",  
        "debug--python",  
        "debug python",  
        "x" * 65,  
    ],
)
def test_validate_skill_name_rejects_invalid_names(name: str) -> None:
    assert not valid_skill_name(name)
    with pytest.raises(ValueError):
        validate_skill_name(name)




# 函数说明：_skill_text
# 用途：在回归测试与测试辅助中处理 `_skill_text`，通过 `extra.items` 完成首个内部处理步
# 骤。
# 参数：
#   name：目标对象、工具或配置项名称，类型 `str`。
#   description：补充描述，类型 `str`。
#   body：请求正文或内容主体，类型 `str`；默认 `'内容'`。
#   **extra：额外关键字参数，按实现处理或转交。
# 返回：类型 `str`；返回 `'\n'.join(lines)`。
def _skill_text(
    name: str,
    description: str,
    body: str = "内容",
    **extra: object,
) -> str:
    lines = ["---", f"name: {name}", f"description: {description}"]
    for key, value in extra.items():
        if isinstance(value, str):
            lines.append(f"{key}: {value}")
        elif isinstance(value, list):
            items = "\n".join(f"  - {item}" for item in value)
            lines.append(f"{key}:\n{items}")
        elif isinstance(value, dict):
            items = "\n".join(f"  {k}: {v}" for k, v in value.items())
            lines.append(f"{key}:\n{items}")
    lines.append("---")
    lines.append("")
    lines.append(body)
    return "\n".join(lines)


# 函数说明：test_parse_skill_document_reads_full_metadata
# 用途：回归验证回归测试与测试辅助中的 `parse_skill_document_reads_full_metadata` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`parse_skill_document` → `_skill_text`
#  → `parsed.body.startswith`。
# 分支与异常：
#   验证条件：`parsed.name == 'code-review'`。
#   验证条件：`parsed.description == '评审代码'`。
#   验证条件：`parsed.license == 'MIT'`。
#   验证条件：`parsed.compatibility == '>=3.10'`。
def test_parse_skill_document_reads_full_metadata() -> None:
    parsed = parse_skill_document(
        _skill_text(
            "code-review",
            "评审代码",
            body="# 评审\n\n1. 正确性",
            license="MIT",
            compatibility='">=3.10"',
            metadata={"audience": "developer"},
            **{"allowed-tools": ["read_file", "write_file"]},
        ),
        expected_name="code-review",
    )

    assert parsed.name == "code-review"
    assert parsed.description == "评审代码"
    assert parsed.license == "MIT"
    assert parsed.compatibility == ">=3.10"
    assert parsed.metadata == {"audience": "developer"}
    assert parsed.allowed_tools == ("read_file", "write_file")
    assert parsed.body.startswith("# 评审")


# 函数说明：test_parse_skill_document_trims_description_and_body
# 用途：回归验证回归测试与测试辅助中的 `parse_skill_document_trims_description_and_body`
#  场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`parse_skill_document`。
# 分支与异常：
#   验证条件：`parsed.description == '描述'`。
#   验证条件：`parsed.body == '正文'`。
def test_parse_skill_document_trims_description_and_body() -> None:
    parsed = parse_skill_document(
        "---\nname: demo\ndescription:   描述  \n---\n\n  正文  \n",
        expected_name="demo",
    )
    assert parsed.description == "描述"
    assert parsed.body == "正文"


# 函数说明：test_parse_skill_document_rejects_bad_structure
# 用途：回归验证回归测试与测试辅助中的 `parse_skill_document_rejects_bad_structure` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   text：待处理的文本，类型 `str`。
#   reason：状态变化、拒绝或降级原因，类型 `str`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` →
# `parse_skill_document`。
# 分支与异常：
#   预期异常：`pytest.raises(SkillParseError, match=reason)`。
@pytest.mark.parametrize(
    "text, reason",
    [
        ("# 没有 front matter", "missing YAML front matter"),
        ("---\nname: demo\ndescription: 描述\n---", "empty skill body"),
    ],
)
def test_parse_skill_document_rejects_bad_structure(
    text: str,
    reason: str,
) -> None:
    with pytest.raises(SkillParseError, match=reason):
        parse_skill_document(text, expected_name="demo")


# 函数说明：test_parse_skill_document_rejects_invalid_yaml
# 用途：回归验证回归测试与测试辅助中的 `parse_skill_document_rejects_invalid_yaml` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` →
# `parse_skill_document`。
# 分支与异常：
#   预期异常：`pytest.raises(SkillParseError, match='invalid YAML')`。
def test_parse_skill_document_rejects_invalid_yaml() -> None:
    with pytest.raises(SkillParseError, match="invalid YAML"):
        parse_skill_document(
            "---\nname: [unclosed\ndescription: 描述\n---\n\n正文",
            expected_name="demo",
        )


# 函数说明：test_parse_skill_document_rejects_name_mismatch
# 用途：回归验证回归测试与测试辅助中的 `parse_skill_document_rejects_name_mismatch` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` →
# `parse_skill_document` → `_skill_text`。
# 分支与异常：
#   预期异常：`pytest.raises(SkillParseError, match='does not match')`。
def test_parse_skill_document_rejects_name_mismatch() -> None:
    with pytest.raises(SkillParseError, match="does not match"):
        parse_skill_document(
            _skill_text("other-name", "描述"),
            expected_name="demo",
        )


# 函数说明：test_parse_skill_document_rejects_invalid_name
# 用途：回归验证回归测试与测试辅助中的 `parse_skill_document_rejects_invalid_name` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` →
# `parse_skill_document` → `_skill_text`。
# 分支与异常：
#   预期异常：`pytest.raises(SkillParseError, match='hyphens')`。
def test_parse_skill_document_rejects_invalid_name() -> None:
    with pytest.raises(SkillParseError, match="hyphens"):
        parse_skill_document(
            _skill_text("debug_python", "描述"),
            expected_name="debug_python",
        )


# 函数说明：test_parse_skill_document_rejects_missing_or_empty_description
# 用途：回归验证回归测试与测试辅助中的
# `parse_skill_document_rejects_missing_or_empty_description` 场景，下方断言说明列出实际
# 通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` →
# `parse_skill_document`。
# 分支与异常：
#   预期异常：`pytest.raises(SkillParseError, match='description')`。
#   预期异常：`pytest.raises(SkillParseError, match='non-string')`。
def test_parse_skill_document_rejects_missing_or_empty_description() -> None:
    with pytest.raises(SkillParseError, match="description"):
        parse_skill_document(
            "---\nname: demo\n---\n\n正文",
            expected_name="demo",
        )
    with pytest.raises(SkillParseError, match="non-string"):
        parse_skill_document(
            "---\nname: demo\ndescription:   \n---\n\n正文",
            expected_name="demo",
        )


# 函数说明：test_parse_skill_document_rejects_overlong_description
# 用途：回归验证回归测试与测试辅助中的
# `parse_skill_document_rejects_overlong_description` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` →
# `parse_skill_document` → `_skill_text`。
# 分支与异常：
#   预期异常：`pytest.raises(SkillParseError, match='1024')`。
def test_parse_skill_document_rejects_overlong_description() -> None:
    with pytest.raises(SkillParseError, match="1024"):
        parse_skill_document(
            _skill_text("demo", "x" * 1025),
            expected_name="demo",
        )


# 函数说明：test_parse_skill_document_rejects_non_mapping_metadata
# 用途：回归验证回归测试与测试辅助中的
# `parse_skill_document_rejects_non_mapping_metadata` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` →
# `parse_skill_document`。
# 分支与异常：
#   预期异常：`pytest.raises(SkillParseError, match='metadata')`。
def test_parse_skill_document_rejects_non_mapping_metadata() -> None:
    with pytest.raises(SkillParseError, match="metadata"):
        parse_skill_document(
            "---\nname: demo\ndescription: 描述\nmetadata: [1, 2]\n---\n\n正文",
            expected_name="demo",
        )


# 函数说明：test_parse_skill_document_rejects_bad_allowed_tools
# 用途：回归验证回归测试与测试辅助中的 `parse_skill_document_rejects_bad_allowed_tools`
# 场景，下方断言说明列出实际通过条件。
# 参数：
#   allowed：`allowed`输入或配置值，类型 `object`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` →
# `parse_skill_document`。
# 分支与异常：
#   预期异常：`pytest.raises(SkillParseError, match='allowed-tools')`。
@pytest.mark.parametrize(
    "allowed",
    [
        "read_file",  
        ["read_file", ""],  
        [1, 2],  
    ],
)
def test_parse_skill_document_rejects_bad_allowed_tools(allowed: object) -> None:
    with pytest.raises(SkillParseError, match="allowed-tools"):
        parse_skill_document(
            "---\nname: demo\ndescription: 描述\n"
            f"allowed-tools: {allowed}\n---\n\n正文",
            expected_name="demo",
        )


# 函数说明：test_parse_skill_document_rejects_unknown_top_level_field
# 用途：回归验证回归测试与测试辅助中的
# `parse_skill_document_rejects_unknown_top_level_field` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` →
# `parse_skill_document`。
# 分支与异常：
#   预期异常：`pytest.raises(SkillParseError, match='unknown front matter field')`。
def test_parse_skill_document_rejects_unknown_top_level_field() -> None:
    with pytest.raises(SkillParseError, match="unknown front matter field"):
        parse_skill_document(
            "---\nname: demo\ndescription: 描述\nlicenceeeee: MIT\n---\n\n正文",
            expected_name="demo",
        )


# 函数说明：test_parse_skill_document_rejects_conflicting_allowed_tools_keys
# 用途：回归验证回归测试与测试辅助中的
# `parse_skill_document_rejects_conflicting_allowed_tools_keys` 场景，下方断言说明列出实
# 际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`pytest.raises` →
# `parse_skill_document`。
# 分支与异常：
#   预期异常：`pytest.raises(SkillParseError, match='cannot both be present')`。
def test_parse_skill_document_rejects_conflicting_allowed_tools_keys() -> None:
    with pytest.raises(SkillParseError, match="cannot both be present"):
        parse_skill_document(
            "---\nname: demo\ndescription: 描述\n"
            "allowed-tools: [read_file]\nallowed_tools: [read_file]\n---\n\n正文",
            expected_name="demo",
        )




# 函数说明：test_safe_skill_dir_accepts_valid_dir
# 用途：回归验证回归测试与测试辅助中的 `safe_skill_dir_accepts_valid_dir` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`(tmp_path / 'demo').mkdir` →
# `safe_skill_dir` → `(tmp_path / 'demo').resolve`。
# 分支与异常：
#   验证条件：`result is not None`。
#   验证条件：`result == (tmp_path / 'demo').resolve()`。
# 副作用与资源：
#   文件或资源访问：`(tmp_path / 'demo').mkdir`。
def test_safe_skill_dir_accepts_valid_dir(tmp_path: Path) -> None:
    (tmp_path / "demo").mkdir()
    result = safe_skill_dir(tmp_path, "demo")
    assert result is not None
    assert result == (tmp_path / "demo").resolve()


# 函数说明：test_safe_skill_dir_rejects_escape_and_symlink
# 用途：回归验证回归测试与测试辅助中的 `safe_skill_dir_rejects_escape_and_symlink` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`outside.mkdir` →
# `(tmp_path / 'link').symlink_to` → `pytest.skip` → `safe_skill_dir`。
# 分支与异常：
#   捕获 `OSError` 后，执行异常处理调用 `pytest.skip`。
#   验证条件：`safe_skill_dir(tmp_path, '..') is None`。
#   验证条件：`safe_skill_dir(tmp_path, 'link') is None`。
#   验证条件：`safe_skill_dir(tmp_path, 'missing') is None`。
# 副作用与资源：
#   文件或资源访问：`outside.mkdir`。
def test_safe_skill_dir_rejects_escape_and_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("当前 Windows 账户没有创建符号链接的权限")

    assert safe_skill_dir(tmp_path, "..") is None
    assert safe_skill_dir(tmp_path, "link") is None
    assert safe_skill_dir(tmp_path, "missing") is None


# 函数说明：test_safe_skill_file_rejects_symlink
# 用途：回归验证回归测试与测试辅助中的 `safe_skill_file_rejects_symlink` 场景，下方断言
# 说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`skill_dir.mkdir` →
# `target.write_text` → `(skill_dir / 'SKILL.md').symlink_to` → `pytest.skip` →
# `safe_skill_file`。
# 分支与异常：
#   捕获 `OSError` 后，执行异常处理调用 `pytest.skip`。
#   验证条件：`safe_skill_file(skill_dir) is None`。
# 副作用与资源：
#   文件或资源访问：`skill_dir.mkdir`、`target.write_text`。
def test_safe_skill_file_rejects_symlink(tmp_path: Path) -> None:
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    target = tmp_path / "elsewhere.txt"
    target.write_text("x", encoding="utf-8")
    try:
        (skill_dir / "SKILL.md").symlink_to(target)
    except OSError:
        pytest.skip("当前 Windows 账户没有创建符号链接的权限")

    assert safe_skill_file(skill_dir) is None


# 函数说明：test_safe_skill_resource_accepts_internal_file
# 用途：回归验证回归测试与测试辅助中的 `safe_skill_resource_accepts_internal_file` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`(skill_dir / 'references').mkdir` →
# `(skill_dir / 'references' / 'a.md').write_text` → `safe_skill_resource` →
# `(skill_dir / 'references' / 'a.md').resolve`。
# 分支与异常：
#   验证条件：`result == (skill_dir / 'references' / 'a.md').resolve()`。
# 副作用与资源：
#   文件或资源访问：`(skill_dir / 'references').mkdir`、
# `(skill_dir / 'references' / 'a.md').write_text`。
def test_safe_skill_resource_accepts_internal_file(tmp_path: Path) -> None:
    skill_dir = tmp_path / "demo"
    (skill_dir / "references").mkdir(parents=True)
    (skill_dir / "references" / "a.md").write_text("x", encoding="utf-8")

    result = safe_skill_resource(skill_dir, "references/a.md")
    assert result == (skill_dir / "references" / "a.md").resolve()


# 函数说明：test_safe_skill_resource_rejects_escape
# 用途：回归验证回归测试与测试辅助中的 `safe_skill_resource_rejects_escape` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
#   relative：相对路径，类型 `str`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`skill_dir.mkdir` →
# `safe_skill_resource`。
# 分支与异常：
#   验证条件：`safe_skill_resource(skill_dir, relative) is None`。
# 副作用与资源：
#   文件或资源访问：`skill_dir.mkdir`。
@pytest.mark.parametrize(
    "relative",
    ["../secret.txt", "/etc/passwd", "a/../../secret.txt", "a/../b"],
)
def test_safe_skill_resource_rejects_escape(
    tmp_path: Path,
    relative: str,
) -> None:
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    assert safe_skill_resource(skill_dir, relative) is None


# 函数说明：test_safe_skill_resource_rejects_symlink_and_directory
# 用途：回归验证回归测试与测试辅助中的
# `safe_skill_resource_rejects_symlink_and_directory` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`(skill_dir / 'refs').mkdir` →
# `target.write_text` → `(skill_dir / 'refs' / 'link.txt').symlink_to` → `pytest.skip` →
#  `(skill_dir / 'refs' / 'subdir').mkdir` → `safe_skill_resource`。
# 分支与异常：
#   捕获 `OSError` 后，执行异常处理调用 `pytest.skip`。
#   验证条件：`safe_skill_resource(skill_dir, 'refs/link.txt') is None`。
#   验证条件：`safe_skill_resource(skill_dir, 'refs/subdir') is None`。
#   验证条件：`safe_skill_resource(skill_dir, 'refs/missing.txt') is None`。
# 副作用与资源：
#   文件或资源访问：`(skill_dir / 'refs').mkdir`、`target.write_text`、
# `(skill_dir / 'refs' / 'subdir').mkdir`。
def test_safe_skill_resource_rejects_symlink_and_directory(
    tmp_path: Path,
) -> None:
    skill_dir = tmp_path / "demo"
    (skill_dir / "refs").mkdir(parents=True)
    target = tmp_path / "outside.txt"
    target.write_text("x", encoding="utf-8")
    try:
        (skill_dir / "refs" / "link.txt").symlink_to(target)
    except OSError:
        pytest.skip("当前 Windows 账户没有创建符号链接的权限")
    (skill_dir / "refs" / "subdir").mkdir()

    assert safe_skill_resource(skill_dir, "refs/link.txt") is None
    assert safe_skill_resource(skill_dir, "refs/subdir") is None
    assert safe_skill_resource(skill_dir, "refs/missing.txt") is None




# 函数说明：_write_skill_dir
# 用途：写入技能，供回归测试与测试辅助使用。
# 参数：
#   root：当前操作的根目录，类型 `Path`。
#   name：目标对象、工具或配置项名称，类型 `str`。
#   description：补充描述，类型 `str`；默认 `'描述'`。
#   body：请求正文或内容主体，类型 `str`；默认 `'# 正文\n\n步骤 1'`。
# 返回：类型 `Path`；返回 `skill_dir`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`skill_dir.mkdir` →
# `(skill_dir / 'SKILL.md').write_text`。
# 副作用与资源：
#   文件或资源访问：`skill_dir.mkdir`、`(skill_dir / 'SKILL.md').write_text`。
def _write_skill_dir(
    root: Path,
    name: str,
    *,
    description: str = "描述",
    body: str = "# 正文\n\n步骤 1",
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = (
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}"
    )
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


# 函数说明：_make_store
# 用途：构造`store`，供回归测试与测试辅助使用。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `SkillStore`；返回 `store`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillStore` → `store.initialize`。
async def _make_store(tmp_path: Path) -> SkillStore:
    store = SkillStore(tmp_path / "user", tmp_path / "project")
    await store.initialize()
    return store


# 函数说明：test_store_catalog_sorts_and_merges_project_over_user
# 用途：回归验证回归测试与测试辅助中的
# `store_catalog_sorts_and_merges_project_over_user` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_store` → `_write_skill_dir` →
# `store.catalog` → `next`。
# 分支与异常：
#   验证条件：`[m.name for m in catalog] == ['alpha', 'beta']`。
#   验证条件：`alpha.scope is SkillScope.PROJECT`。
#   验证条件：`alpha.description == '项目版 alpha'`。
#   验证条件：`beta.scope is SkillScope.USER`。
@pytest.mark.asyncio
async def test_store_catalog_sorts_and_merges_project_over_user(
    tmp_path: Path,
) -> None:
    store = await _make_store(tmp_path)
    _write_skill_dir(store.user_dir, "alpha", description="用户版 alpha")
    _write_skill_dir(store.user_dir, "beta", description="用户版 beta")
    _write_skill_dir(store.project_dir, "alpha", description="项目版 alpha")

    catalog = await store.catalog()

    assert [m.name for m in catalog] == ["alpha", "beta"]
    alpha = next(m for m in catalog if m.name == "alpha")
    assert alpha.scope is SkillScope.PROJECT
    assert alpha.description == "项目版 alpha"
    beta = next(m for m in catalog if m.name == "beta")
    assert beta.scope is SkillScope.USER


# 函数说明：test_store_catalog_skips_bad_skills_with_diagnostics
# 用途：回归验证回归测试与测试辅助中的 `store_catalog_skips_bad_skills_with_diagnostics`
#  场景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_store` → `bad_dir.mkdir` →
# `(bad_dir / 'SKILL.md').write_text` → `_write_skill_dir` → `broken_dir.mkdir` →
# `(broken_dir / 'SKILL.md').write_text`；另有 3 个调用点。
# 分支与异常：
#   验证条件：`[m.name for m in catalog] == ['good']`。
#   验证条件：`any(('bad_name' in d.render() for d in diagnostics))`。
#   验证条件：`any(('broken' in d.render() for d in diagnostics))`。
# 副作用与资源：
#   文件或资源访问：`bad_dir.mkdir`、`(bad_dir / 'SKILL.md').write_text`、
# `broken_dir.mkdir`、`(broken_dir / 'SKILL.md').write_text`。
@pytest.mark.asyncio
async def test_store_catalog_skips_bad_skills_with_diagnostics(
    tmp_path: Path,
) -> None:
    store = await _make_store(tmp_path)
    bad_dir = store.project_dir / "bad_name"
    bad_dir.mkdir(parents=True)
    (bad_dir / "SKILL.md").write_text(
        "---\nname: bad_name\ndescription: 坏\n---\n\n正文",
        encoding="utf-8",
    )
    _write_skill_dir(store.project_dir, "good", description="好")
    broken_dir = store.project_dir / "broken"
    broken_dir.mkdir()
    (broken_dir / "SKILL.md").write_text(
        "---\nname: broken\n---\n\n正文",
        encoding="utf-8",
    )

    catalog = await store.catalog()
    diagnostics = store.diagnostics()

    assert [m.name for m in catalog] == ["good"]
    assert any("bad_name" in d.render() for d in diagnostics)
    assert any("broken" in d.render() for d in diagnostics)


# 函数说明：test_store_load_returns_none_when_missing_or_bad
# 用途：回归验证回归测试与测试辅助中的 `store_load_returns_none_when_missing_or_bad` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_store` → `store.load` →
# `_write_skill_dir`。
# 分支与异常：
#   验证条件：`await store.load('missing') is None`。
#   验证条件：`await store.load('bad') is None`。
@pytest.mark.asyncio
async def test_store_load_returns_none_when_missing_or_bad(
    tmp_path: Path,
) -> None:
    store = await _make_store(tmp_path)
    assert await store.load("missing") is None

    _write_skill_dir(store.project_dir, "bad", description="")
    assert await store.load("bad") is None


# 函数说明：test_store_load_returns_full_skill_with_resources
# 用途：回归验证回归测试与测试辅助中的 `store_load_returns_full_skill_with_resources` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_store` → `_write_skill_dir` →
# `(skill_dir / 'references').mkdir` →
# `(skill_dir / 'references' / 'tpl.md').write_text` → `(skill_dir / 'scripts').mkdir` →
#  `(skill_dir / 'scripts' / 'run.sh').write_text`；另有 2 个调用点。
# 分支与异常：
#   验证条件：`skill is not None`。
#   验证条件：`isinstance(skill, Skill)`。
#   验证条件：`skill.metadata.name == 'research'`。
#   验证条件：`skill.root == skill_dir.resolve()`。
# 副作用与资源：
#   文件或资源访问：`(skill_dir / 'references').mkdir`、
# `(skill_dir / 'references' / 'tpl.md').write_text`、`(skill_dir / 'scripts').mkdir`、
# `(skill_dir / 'scripts' / 'run.sh').write_text`。
@pytest.mark.asyncio
async def test_store_load_returns_full_skill_with_resources(
    tmp_path: Path,
) -> None:
    store = await _make_store(tmp_path)
    skill_dir = _write_skill_dir(
        store.project_dir,
        "research",
        description="研究",
        body="# 研究\n\n步骤",
    )
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "tpl.md").write_text("模板", encoding="utf-8")
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "run.sh").write_text("#!/bin/sh", encoding="utf-8")

    skill = await store.load("research")

    assert skill is not None
    assert isinstance(skill, Skill)
    assert skill.metadata.name == "research"
    assert skill.root == skill_dir.resolve()
    assert skill.resources.references == ("references/tpl.md",)
    assert skill.resources.scripts == ("scripts/run.sh",)
    assert skill.resources.assets == ()
    assert "步骤" in skill.content


# 函数说明：test_skill_render_instructions_lists_resources
# 用途：回归验证回归测试与测试辅助中的 `skill_render_instructions_lists_resources` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillMetadata` → `Path` → `Skill` →
# `SkillResources` → `skill.render_instructions` → `rendered.startswith`。
# 分支与异常：
#   验证条件：`rendered.startswith('# Skill: demo')`。
#   验证条件：`'references/a.md' in rendered`。
#   验证条件：`'skill_resource_read' in rendered`。
def test_skill_render_instructions_lists_resources() -> None:
    metadata = SkillMetadata(
        name="demo",
        description="描述",
        scope=SkillScope.PROJECT,
        location=Path("/tmp/demo/SKILL.md"),
    )
    skill = Skill(
        metadata=metadata,
        content="# 正文",
        root=Path("/tmp/demo"),
        resources=SkillResources(references=("references/a.md",)),
    )
    rendered = skill.render_instructions()
    assert rendered.startswith("# Skill: demo")
    assert "references/a.md" in rendered
    assert "skill_resource_read" in rendered




# 函数说明：_metadata
# 用途：返回 `SkillMetadata(…)`，提供 回归测试与测试辅助 的派生值。
# 参数：
#   name：目标对象、工具或配置项名称，类型 `str`。
#   description：补充描述，类型 `str`；默认 `'描述'`。
# 返回：类型 `SkillMetadata`；返回 `SkillMetadata(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillMetadata` → `Path`。
def _metadata(name: str, description: str = "描述") -> SkillMetadata:
    return SkillMetadata(
        name=name,
        description=description,
        scope=SkillScope.PROJECT,
        location=Path(f"/tmp/{name}/SKILL.md"),
    )


# 函数说明：_skill
# 用途：返回
# `Skill(metadata=_metadata(name), content=content, root=Path(f'/tmp/{name}'))`，提供 回
# 归测试与测试辅助 的派生值。
# 参数：
#   name：目标对象、工具或配置项名称，类型 `str`。
#   content：内容正文，类型 `str`；默认 `'# 正文\n\n步骤'`。
# 返回：类型 `Skill`；返回
# `Skill(metadata=_metadata(name), content=content, root=Path(f'/tmp/{name}'))`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Skill` → `_metadata` → `Path`。
def _skill(name: str, content: str = "# 正文\n\n步骤") -> Skill:
    return Skill(
        metadata=_metadata(name),
        content=content,
        root=Path(f"/tmp/{name}"),
    )


# 函数说明：test_catalog_message_injects_system_message_with_entries
# 用途：回归验证回归测试与测试辅助中的
# `catalog_message_injects_system_message_with_entries` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillContextProvider` →
# `provider.catalog_message` → `_metadata`。
# 分支与异常：
#   验证条件：`message is not None`。
#   验证条件：`message.role is MessageRole.SYSTEM`。
#   验证条件：`message.name == SKILL_CATALOG_MESSAGE_NAME`。
#   验证条件：`'[a] 描述' in (message.content or '')`。
def test_catalog_message_injects_system_message_with_entries() -> None:
    provider = SkillContextProvider(max_tokens=4096, max_active=4)
    message = provider.catalog_message((_metadata("a"), _metadata("b")))

    assert message is not None
    assert message.role is MessageRole.SYSTEM
    assert message.name == SKILL_CATALOG_MESSAGE_NAME
    assert "[a] 描述" in (message.content or "")
    assert "[b] 描述" in (message.content or "")


# 函数说明：test_catalog_message_handles_empty_catalog
# 用途：回归验证回归测试与测试辅助中的 `catalog_message_handles_empty_catalog` 场景，下
# 方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillContextProvider` →
# `provider.catalog_message`。
# 分支与异常：
#   验证条件：`message is not None`。
#   验证条件：`'No skills available' in (message.content or '')`。
def test_catalog_message_handles_empty_catalog() -> None:
    provider = SkillContextProvider(max_tokens=4096, max_active=4)
    message = provider.catalog_message(())
    assert message is not None
    assert "No skills available" in (message.content or "")


# 函数说明：test_catalog_budget_keeps_small_catalog
# 用途：回归验证回归测试与测试辅助中的 `catalog_budget_keeps_small_catalog` 场景，下方断
# 言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillContextProvider` → `_metadata` →
#  `provider.render_catalog` → `provider.catalog_tokens`。
# 分支与异常：
#   验证条件：`f'[{item.name}]' in text`。
#   验证条件：`'not shown' not in text`。
#   验证条件：`provider.catalog_tokens(metadata) <= provider.catalog_max_tokens`。
def test_catalog_budget_keeps_small_catalog() -> None:
    provider = SkillContextProvider(
        max_tokens=4096,
        max_active=4,
        catalog_max_tokens=2048,
    )
    metadata = tuple(_metadata(f"skill-{index}") for index in range(5))
    text = provider.render_catalog(metadata)
    for item in metadata:
        assert f"[{item.name}]" in text
    assert "not shown" not in text
    assert provider.catalog_tokens(metadata) <= provider.catalog_max_tokens


# 函数说明：test_catalog_budget_truncates_large_catalog
# 用途：回归验证回归测试与测试辅助中的 `catalog_budget_truncates_large_catalog` 场景，下
# 方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillContextProvider` → `_metadata` →
#  `provider.render_catalog` → `provider.catalog_tokens` → `text.startswith`。
# 分支与异常：
#   验证条件：`provider.catalog_tokens(metadata) <= provider.catalog_max_tokens`。
#   验证条件：`'not shown' in text`。
#   验证条件：`text.startswith('# Available Skills')`。
#   验证条件：`provider.render_catalog(metadata) == text`。
def test_catalog_budget_truncates_large_catalog() -> None:
    provider = SkillContextProvider(
        max_tokens=4096,
        max_active=4,
        catalog_max_tokens=200,
    )
    metadata = tuple(
        _metadata(f"skill-{index}", description="很长的描述" * 30)
        for index in range(50)
    )
    text = provider.render_catalog(metadata)
    assert provider.catalog_tokens(metadata) <= provider.catalog_max_tokens
    assert "not shown" in text
    assert text.startswith("# Available Skills")
    assert provider.render_catalog(metadata) == text


# 函数说明：test_catalog_budget_tokens_within_limit
# 用途：回归验证回归测试与测试辅助中的 `catalog_budget_tokens_within_limit` 场景，下方断
# 言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillContextProvider` → `_metadata` →
#  `provider.catalog_message` → `provider.catalog_tokens`。
# 分支与异常：
#   验证条件：`message is not None`。
#   验证条件：`provider.catalog_tokens(metadata) <= provider.catalog_max_tokens`。
def test_catalog_budget_tokens_within_limit() -> None:
    provider = SkillContextProvider(
        max_tokens=4096,
        max_active=4,
        catalog_max_tokens=64,
    )
    metadata = tuple(
        _metadata(f"skill-{index}", description="x" * 200) for index in range(20)
    )
    message = provider.catalog_message(metadata)
    assert message is not None
    assert provider.catalog_tokens(metadata) <= provider.catalog_max_tokens


# 函数说明：test_active_messages_dedupes_and_preserves_order
# 用途：回归验证回归测试与测试辅助中的 `active_messages_dedupes_and_preserves_order` 场
# 景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillContextProvider` →
# `provider.active_messages` → `_skill`。
# 分支与异常：
#   验证条件：`[m.name for m in messages] == [ACTIVE_SKILL_MESSAGE_NAME] * 2`。
#   验证条件：`'Skill: a' in (messages[0].content or '')`。
#   验证条件：`'Skill: b' in (messages[1].content or '')`。
def test_active_messages_dedupes_and_preserves_order() -> None:
    provider = SkillContextProvider(max_tokens=4096, max_active=4)
    messages = provider.active_messages((_skill("a"), _skill("b"), _skill("a")))

    assert [m.name for m in messages] == [ACTIVE_SKILL_MESSAGE_NAME] * 2
    assert "Skill: a" in (messages[0].content or "")
    assert "Skill: b" in (messages[1].content or "")


# 函数说明：test_active_tokens_counts_instructions
# 用途：回归验证回归测试与测试辅助中的 `active_tokens_counts_instructions` 场景，下方断
# 言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillContextProvider` →
# `provider.active_tokens` → `_skill`。
# 分支与异常：
#   验证条件：`single > 0`。
#   验证条件：`both > single`。
def test_active_tokens_counts_instructions() -> None:
    provider = SkillContextProvider(max_tokens=4096, max_active=4)
    single = provider.active_tokens((_skill("a"),))
    both = provider.active_tokens((_skill("a"), _skill("b")))
    assert single > 0
    assert both > single


# 函数说明：test_would_exceed_budget_respects_max_active
# 用途：回归验证回归测试与测试辅助中的 `would_exceed_budget_respects_max_active` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillContextProvider` →
# `provider.would_exceed_budget` → `_skill`。
# 分支与异常：
#   验证条件：`not provider.would_exceed_budget((_skill('a'),), _skill('b'))`。
#   验证条件：`provider.would_exceed_budget((_skill('a'), _skill('b')), _skill('c'))`。
def test_would_exceed_budget_respects_max_active() -> None:
    provider = SkillContextProvider(max_tokens=1_000_000, max_active=2)
    assert not provider.would_exceed_budget((_skill("a"),), _skill("b"))
    assert provider.would_exceed_budget((_skill("a"), _skill("b")), _skill("c"))


# 函数说明：test_would_exceed_budget_respects_token_limit
# 用途：回归验证回归测试与测试辅助中的 `would_exceed_budget_respects_token_limit` 场景，
# 下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillContextProvider` →
# `provider.would_exceed_budget` → `_skill`。
# 分支与异常：
#   验证条件：`provider.would_exceed_budget((), _skill('a'))`。
def test_would_exceed_budget_respects_token_limit() -> None:
    provider = SkillContextProvider(max_tokens=10, max_active=4)
    assert provider.would_exceed_budget((), _skill("a"))


# 函数说明：test_would_exceed_budget_dedupes_existing_skill
# 用途：回归验证回归测试与测试辅助中的 `would_exceed_budget_dedupes_existing_skill` 场景
# ，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillContextProvider` → `_skill` →
# `provider.would_exceed_budget`。
# 分支与异常：
#   验证条件：`not provider.would_exceed_budget((existing,), existing)`。
def test_would_exceed_budget_dedupes_existing_skill() -> None:
    provider = SkillContextProvider(max_tokens=100, max_active=4)
    existing = _skill("a")
    assert not provider.would_exceed_budget((existing,), existing)




# 函数说明：_tool_registry_with_store
# 用途：保存工具，供回归测试与测试辅助使用。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `tuple[ToolRegistry, SkillStore]`；返回 `(registry, store)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_make_store` → `ToolRegistry` →
# `register_skill_tools`。
async def _tool_registry_with_store(tmp_path: Path) -> tuple[ToolRegistry, SkillStore]:
    store = await _make_store(tmp_path)
    registry = ToolRegistry()
    register_skill_tools(registry, store)
    return registry, store


# 函数说明：test_skill_read_returns_skill_and_missing
# 用途：回归验证回归测试与测试辅助中的 `skill_read_returns_skill_and_missing` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tool_registry_with_store` →
# `_write_skill_dir` → `registry.get(SKILL_READ_TOOL_NAME).execute`。
# 分支与异常：
#   验证条件：`found['found'] is True`。
#   验证条件：`found['name'] == 'demo'`。
#   验证条件：`found['scope'] == 'project'`。
#   验证条件：`'content' not in found`。
@pytest.mark.asyncio
async def test_skill_read_returns_skill_and_missing(tmp_path: Path) -> None:
    registry, _ = await _tool_registry_with_store(tmp_path)
    _write_skill_dir(tmp_path / "project", "demo", description="演示")

    found = await registry.get(SKILL_READ_TOOL_NAME).execute({"name": "demo"})
    assert found["found"] is True
    assert found["name"] == "demo"
    assert found["scope"] == "project"
    assert "content" not in found
    assert "resources" in found
    assert found["resources"] == {"references": (), "scripts": (), "assets": ()}

    missing = await registry.get(SKILL_READ_TOOL_NAME).execute(
        {"name": "nope"}
    )
    assert missing["found"] is False


# 函数说明：test_skill_read_validates_name_argument
# 用途：回归验证回归测试与测试辅助中的 `skill_read_validates_name_argument` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tool_registry_with_store` →
# `pytest.raises` → `registry.get(SKILL_READ_TOOL_NAME).execute`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match="'name'")`。
@pytest.mark.asyncio
async def test_skill_read_validates_name_argument(tmp_path: Path) -> None:
    registry, _ = await _tool_registry_with_store(tmp_path)
    with pytest.raises(ValueError, match="'name'"):
        await registry.get(SKILL_READ_TOOL_NAME).execute({})


# 函数说明：test_skill_resource_read_reads_managed_resource
# 用途：回归验证回归测试与测试辅助中的 `skill_resource_read_reads_managed_resource` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tool_registry_with_store` →
# `_write_skill_dir` → `(skill_dir / 'references').mkdir` →
# `(skill_dir / 'references' / 'tpl.md').write_text` →
# `registry.get('skill_resource_read').execute`。
# 分支与异常：
#   验证条件：`result['found'] is True`。
#   验证条件：`result['content'] == '模板内容'`。
# 副作用与资源：
#   文件或资源访问：`(skill_dir / 'references').mkdir`、
# `(skill_dir / 'references' / 'tpl.md').write_text`。
@pytest.mark.asyncio
async def test_skill_resource_read_reads_managed_resource(
    tmp_path: Path,
) -> None:
    registry, _ = await _tool_registry_with_store(tmp_path)
    skill_dir = _write_skill_dir(
        tmp_path / "project",
        "research",
        description="研究",
    )
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "tpl.md").write_text("模板内容", encoding="utf-8")

    result = await registry.get("skill_resource_read").execute(
        {"name": "research", "path": "references/tpl.md"}
    )
    assert result["found"] is True
    assert result["content"] == "模板内容"


# 函数说明：test_skill_resource_read_rejects_escape_path
# 用途：回归验证回归测试与测试辅助中的 `skill_resource_read_rejects_escape_path` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tool_registry_with_store` →
# `_write_skill_dir` → `registry.get('skill_resource_read').execute`。
# 分支与异常：
#   验证条件：`result['found'] is False`。
#   验证条件：`'escape' in result['error']`。
@pytest.mark.asyncio
async def test_skill_resource_read_rejects_escape_path(
    tmp_path: Path,
) -> None:
    registry, _ = await _tool_registry_with_store(tmp_path)
    _write_skill_dir(tmp_path / "project", "research", description="研究")

    result = await registry.get("skill_resource_read").execute(
        {"name": "research", "path": "../SKILL.md"}
    )
    assert result["found"] is False
    assert "escape" in result["error"]


# 函数说明：test_skill_resource_read_requires_active_skill
# 用途：回归验证回归测试与测试辅助中的 `skill_resource_read_requires_active_skill` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_tool_registry_with_store` →
# `_write_skill_dir` → `(skill_dir / 'references').mkdir` →
# `(skill_dir / 'references' / 'tpl.md').write_text` → `ToolExecutionContext` →
# `ToolCall`；另有 1 个调用点。
# 分支与异常：
#   验证条件：`rejected['found'] is False`。
#   验证条件：`'not active' in rejected['error']`。
#   验证条件：`ok['found'] is True`。
#   验证条件：`ok['content'] == '模板内容'`。
# 副作用与资源：
#   文件或资源访问：`(skill_dir / 'references').mkdir`、
# `(skill_dir / 'references' / 'tpl.md').write_text`。
@pytest.mark.asyncio
async def test_skill_resource_read_requires_active_skill(
    tmp_path: Path,
) -> None:

    registry, _ = await _tool_registry_with_store(tmp_path)
    skill_dir = _write_skill_dir(
        tmp_path / "project",
        "research",
        description="研究",
    )
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "tpl.md").write_text("模板内容", encoding="utf-8")
    tool = registry.get("skill_resource_read")

    inactive_context = ToolExecutionContext(
        tool_call=ToolCall(
            id="c1",
            name="skill_resource_read",
            arguments={"name": "research", "path": "references/tpl.md"},
        ),
        metadata={"active_skill_names": ()},
    )
    rejected = await tool.execute_with_context(
        {"name": "research", "path": "references/tpl.md"},
        inactive_context,
    )
    assert rejected["found"] is False
    assert "not active" in rejected["error"]

    active_context = ToolExecutionContext(
        tool_call=ToolCall(
            id="c1",
            name="skill_resource_read",
            arguments={"name": "research", "path": "references/tpl.md"},
        ),
        metadata={"active_skill_names": ("research",)},
    )
    ok = await tool.execute_with_context(
        {"name": "research", "path": "references/tpl.md"},
        active_context,
    )
    assert ok["found"] is True
    assert ok["content"] == "模板内容"




# 函数说明：test_runtime_activates_skill_and_injects_instructions
# 用途：回归验证回归测试与测试辅助中的
# `runtime_activates_skill_and_injects_instructions` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`skill_dir.mkdir` →
# `(skill_dir / 'SKILL.md').write_text` → `SkillStore` → `store.initialize` →
# `ToolRegistry` → `register_skill_tools`；另有 8 个调用点。
# 分支与异常：
#   验证条件：`len(activated) == 1`。
#   验证条件：`activated[0].skill_name == 'demo'`。
#   验证条件：`activated[0].skill_scope == 'project'`。
#   验证条件：`activated[0].active_skill_names == ('demo',)`。
# 副作用与资源：
#   文件或资源访问：`skill_dir.mkdir`、`(skill_dir / 'SKILL.md').write_text`。
@pytest.mark.asyncio
async def test_runtime_activates_skill_and_injects_instructions(
    tmp_path: Path,
) -> None:
    skill_dir = tmp_path / "project" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: 演示\n---\n\n# Demo\n\n按演示流程操作",
        encoding="utf-8",
    )
    store = SkillStore(tmp_path / "user", tmp_path / "project")
    await store.initialize()
    registry = ToolRegistry()
    register_skill_tools(registry, store)
    provider = SkillContextProvider(max_tokens=4096, max_active=4)

    responses = [
        _model_response(
            tool_calls=(
                ToolCall(
                    id="skill-1",
                    name=SKILL_READ_TOOL_NAME,
                    arguments={"name": "demo"},
                ),
            )
        ),
        _model_response(content="已完成"),
    ]
    model_registry, adapter = _fake_registry(responses)
    events = InMemoryEventHandler()
    runtime = AgentRuntime(
        model_registry,
        registry,
        provider="fake",
        skill_store=store,
        skill_context_provider=provider,
    )

    await runtime.run(
        "请处理演示任务",
        event_handler=events,
    )

    activated = [
        e for e in events.events if e.type is AgentEventType.SKILL_ACTIVATED
    ]
    assert len(activated) == 1
    assert activated[0].skill_name == "demo"
    assert activated[0].skill_scope == "project"
    assert activated[0].active_skill_names == ("demo",)
    assert activated[0].active_skill_tokens and activated[0].active_skill_tokens > 0

    started = [
        e for e in events.events if e.type is AgentEventType.MODEL_STARTED
    ]
    assert started[-1].active_skill_names == ("demo",)
    assert started[-1].available_skill_count == 1
    assert started[-1].skill_catalog_tokens and started[-1].skill_catalog_tokens > 0
    assert started[-1].active_skill_tokens and started[-1].active_skill_tokens > 0
    assert started[-1].active_skill_message_names == ("demo",)

    second_request = adapter.requests[1]
    tool_result_messages = [
        m for m in second_request.messages if m.role is MessageRole.TOOL
    ]
    assert tool_result_messages
    assert all(
        "按演示流程操作" not in (m.content or "") for m in tool_result_messages
    )

    injected = [
        m
        for m in second_request.messages
        if m.name in (ACTIVE_SKILL_MESSAGE_NAME, SKILL_CATALOG_MESSAGE_NAME)
    ]
    assert any(m.name == ACTIVE_SKILL_MESSAGE_NAME for m in injected)
    assert any("Skill: demo" in (m.content or "") for m in injected)
    assert any(m.name == SKILL_CATALOG_MESSAGE_NAME for m in injected)
    occurrences = sum(
        (m.content or "").count("按演示流程操作")
        for m in second_request.messages
    )
    assert occurrences == 1


# 函数说明：test_runtime_emits_activation_failed_when_budget_exceeded
# 用途：回归验证回归测试与测试辅助中的
# `runtime_emits_activation_failed_when_budget_exceeded` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`skill_dir.mkdir` →
# `(skill_dir / 'SKILL.md').write_text` → `SkillStore` → `store.initialize` →
# `ToolRegistry` → `register_skill_tools`；另有 7 个调用点。
# 分支与异常：
#   验证条件：`len(failed) == 1`。
#   验证条件：`failed[0].skill_name == 'big'`。
#   验证条件：`'budget' in failed[0].skill_error`。
#   验证条件：
# `not any((e.type is AgentEventType.SKILL_ACTIVATED for e in events.events))`。
# 副作用与资源：
#   文件或资源访问：`skill_dir.mkdir`、`(skill_dir / 'SKILL.md').write_text`。
@pytest.mark.asyncio
async def test_runtime_emits_activation_failed_when_budget_exceeded(
    tmp_path: Path,
) -> None:
    skill_dir = tmp_path / "project" / "big"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: big\ndescription: 大\n---\n\n# Big\n\n" + "x" * 200,
        encoding="utf-8",
    )
    store = SkillStore(tmp_path / "user", tmp_path / "project")
    await store.initialize()
    registry = ToolRegistry()
    register_skill_tools(registry, store)
    provider = SkillContextProvider(max_tokens=5, max_active=4)

    tc = ToolCall(
        id="skill-1",
        name=SKILL_READ_TOOL_NAME,
        arguments={"name": "big"},
    )
    model_registry, adapter = _fake_registry(
        [
            _model_response(tool_calls=(tc,)),
            _model_response(content="done"),
        ]
    )
    events = InMemoryEventHandler()
    runtime = AgentRuntime(
        model_registry,
        registry,
        provider="fake",
        skill_store=store,
        skill_context_provider=provider,
    )

    await runtime.run("处理任务", event_handler=events)

    failed = [
        e for e in events.events if e.type is AgentEventType.SKILL_ACTIVATION_FAILED
    ]
    assert len(failed) == 1
    assert failed[0].skill_name == "big"
    assert "budget" in failed[0].skill_error
    assert not any(
        e.type is AgentEventType.SKILL_ACTIVATED for e in events.events
    )

    for request in adapter.requests:
        assert all(
            "x" * 40 not in (m.content or "") for m in request.messages
        )


# 函数说明：test_runtime_emits_activation_failed_when_skill_is_missing
# 用途：回归验证回归测试与测试辅助中的
# `runtime_emits_activation_failed_when_skill_is_missing` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`SkillStore` → `store.initialize` →
# `ToolRegistry` → `register_skill_tools` → `SkillContextProvider` → `_fake_registry`；
# 另有 5 个调用点。
# 分支与异常：
#   验证条件：`result.ok is True`。
#   验证条件：`result.tool_calls[0].result.success is True`。
#   验证条件：`len(failed) == 1`。
#   验证条件：`failed[0].skill_name == 'write-notes'`。
@pytest.mark.asyncio
async def test_runtime_emits_activation_failed_when_skill_is_missing(
    tmp_path: Path,
) -> None:
    store = SkillStore(tmp_path / "user", tmp_path / "project")
    await store.initialize()
    registry = ToolRegistry()
    register_skill_tools(registry, store)
    provider = SkillContextProvider(max_tokens=4096, max_active=4)
    model_registry, _ = _fake_registry(
        [
            _model_response(
                tool_calls=(
                    ToolCall(
                        id="missing-skill",
                        name=SKILL_READ_TOOL_NAME,
                        arguments={"name": "write-notes"},
                    ),
                )
            ),
            _model_response(content="write-notes 不存在。"),
        ]
    )
    events = InMemoryEventHandler()
    runtime = AgentRuntime(
        model_registry,
        registry,
        provider="fake",
        skill_store=store,
        skill_context_provider=provider,
    )

    result = await runtime.run("请使用 write-notes", event_handler=events)

    assert result.ok is True
    assert result.tool_calls[0].result.success is True
    failed = [
        event
        for event in events.events
        if event.type is AgentEventType.SKILL_ACTIVATION_FAILED
    ]
    assert len(failed) == 1
    assert failed[0].skill_name == "write-notes"
    assert failed[0].skill_error == "skill not found"
    assert not any(
        event.type is AgentEventType.SKILL_ACTIVATED for event in events.events
    )


# 函数说明：test_runtime_rejects_provider_without_store
# 用途：回归验证回归测试与测试辅助中的 `runtime_rejects_provider_without_store` 场景，下
# 方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ModelAdapterRegistry` →
# `ModelSettings` → `pytest.raises` → `AgentRuntime` → `ToolRegistry` →
# `SkillContextProvider`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='skill_context_provider requires')`。
def test_runtime_rejects_provider_without_store() -> None:
    registry = ModelAdapterRegistry(ModelSettings(_env_file=None))
    with pytest.raises(ValueError, match="skill_context_provider requires"):
        AgentRuntime(
            registry,
            ToolRegistry(),
            provider="fake",
            skill_context_provider=SkillContextProvider(
                max_tokens=4096,
                max_active=4,
            ),
        )
