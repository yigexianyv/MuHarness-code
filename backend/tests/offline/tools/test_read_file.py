from pathlib import Path

import pytest

from app.tools import ReadFileTool
from app.tools.builtin.read_file import MAX_RANGE_CHARS, MAX_READ_LINES


# 函数说明：test_path_only_keeps_complete_utf8_read
# 用途：回归验证回归测试与测试辅助中的 `path_only_keeps_complete_utf8_read` 场景，下方断
# 言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`(tmp_path / 'text.txt').write_text` →
#  `ReadFileTool(tmp_path).execute` → `ReadFileTool`。
# 分支与异常：
#   验证条件：`await ReadFileTool(tmp_path).execute({'path': 'text.txt'}) == content`。
# 副作用与资源：
#   文件或资源访问：`(tmp_path / 'text.txt').write_text`。
@pytest.mark.asyncio
async def test_path_only_keeps_complete_utf8_read(tmp_path: Path) -> None:
    content = "你好\n" + "x" * (MAX_RANGE_CHARS + 10)
    (tmp_path / "text.txt").write_text(content, encoding="utf-8")

    assert await ReadFileTool(tmp_path).execute({"path": "text.txt"}) == content


# 函数说明：test_ranged_read_returns_lines_and_continuation
# 用途：回归验证回归测试与测试辅助中的 `ranged_read_returns_lines_and_continuation` 场景
# ，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`(tmp_path / 'text.txt').write_text` →
#  `ReadFileTool` → `tool.execute`。
# 分支与异常：
#   验证条件：`first == {'path': 'text.txt', 'start_line': 2, 'end_line': 3, 'content':
# '二\n三\n', 'next_line': 4, 'truncated': True, '…`。
#   验证条件：`isinstance(last, dict)`。
#   验证条件：`last['content'] == '四'`。
#   验证条件：`last['end_line'] == 4`。
# 副作用与资源：
#   文件或资源访问：`(tmp_path / 'text.txt').write_text`。
@pytest.mark.asyncio
async def test_ranged_read_returns_lines_and_continuation(tmp_path: Path) -> None:
    (tmp_path / "text.txt").write_text("一\n二\n三\n四", encoding="utf-8")
    tool = ReadFileTool(tmp_path)

    first = await tool.execute({"path": "text.txt", "start_line": 2, "max_lines": 2})
    assert first == {
        "path": "text.txt", "start_line": 2, "end_line": 3,
        "content": "二\n三\n", "next_line": 4,
        "truncated": True, "line_truncated": None,
    }
    last = await tool.execute({"path": "text.txt", "start_line": 4})
    assert isinstance(last, dict)
    assert last["content"] == "四"
    assert last["end_line"] == 4
    assert last["next_line"] is None
    assert last["truncated"] is False


# 函数说明：test_ranged_read_defaults_to_first_line_and_200_lines
# 用途：回归验证回归测试与测试辅助中的
# `ranged_read_defaults_to_first_line_and_200_lines` 场景，下方断言说明列出实际通过条件
# 。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`(tmp_path / 'text.txt').write_text` →
#  `ReadFileTool` → `tool.execute`。
# 分支与异常：
#   验证条件：`isinstance(first, dict)`。
#   验证条件：`first['start_line'] == 1`。
#   验证条件：`first['end_line'] == 1`。
#   验证条件：`isinstance(default, dict)`。
# 副作用与资源：
#   文件或资源访问：`(tmp_path / 'text.txt').write_text`。
@pytest.mark.asyncio
async def test_ranged_read_defaults_to_first_line_and_200_lines(tmp_path: Path) -> None:
    (tmp_path / "text.txt").write_text("a\n" * 205, encoding="utf-8")
    tool = ReadFileTool(tmp_path)

    first = await tool.execute({"path": "text.txt", "max_lines": 1})
    assert isinstance(first, dict)
    assert first["start_line"] == 1
    assert first["end_line"] == 1
    default = await tool.execute({"path": "text.txt", "start_line": 1})
    assert isinstance(default, dict)
    assert default["end_line"] == 200
    assert default["next_line"] == 201


# 函数说明：test_empty_or_out_of_range_has_no_fake_next_line
# 用途：回归验证回归测试与测试辅助中的 `empty_or_out_of_range_has_no_fake_next_line` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
#   content：内容正文，类型 `str`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`(tmp_path / 'text.txt').write_text` →
#  `ReadFileTool(tmp_path).execute` → `ReadFileTool`。
# 分支与异常：
#   验证条件：`isinstance(result, dict)`。
#   验证条件：`result['content'] == ''`。
#   验证条件：`result['end_line'] is None`。
#   验证条件：`result['next_line'] is None`。
# 副作用与资源：
#   文件或资源访问：`(tmp_path / 'text.txt').write_text`。
@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", "one\ntwo\n"])
async def test_empty_or_out_of_range_has_no_fake_next_line(
    tmp_path: Path, content: str,
) -> None:
    (tmp_path / "text.txt").write_text(content, encoding="utf-8")

    result = await ReadFileTool(tmp_path).execute(
        {"path": "text.txt", "start_line": 10}
    )
    assert isinstance(result, dict)
    assert result["content"] == ""
    assert result["end_line"] is None
    assert result["next_line"] is None
    assert result["truncated"] is False


# 函数说明：test_single_long_line_is_bounded_and_explicit
# 用途：回归验证回归测试与测试辅助中的 `single_long_line_is_bounded_and_explicit` 场景，
# 下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`(tmp_path / 'text.txt').write_text` →
#  `ReadFileTool` → `tool.execute`。
# 分支与异常：
#   验证条件：`isinstance(result, dict)`。
#   验证条件：`result['content'] == '长' * MAX_RANGE_CHARS`。
#   验证条件：`result['line_truncated'] == 1`。
#   验证条件：`result['end_line'] == 1`。
# 副作用与资源：
#   文件或资源访问：`(tmp_path / 'text.txt').write_text`。
@pytest.mark.asyncio
async def test_single_long_line_is_bounded_and_explicit(tmp_path: Path) -> None:
    (tmp_path / "text.txt").write_text("长" * 100_000 + "\nnext", encoding="utf-8")
    tool = ReadFileTool(tmp_path)

    result = await tool.execute({"path": "text.txt", "max_lines": 2})
    assert isinstance(result, dict)
    assert result["content"] == "长" * MAX_RANGE_CHARS
    assert result["line_truncated"] == 1
    assert result["end_line"] == 1
    assert result["next_line"] == 2
    assert result["truncated"] is True
    next_page = await tool.execute({"path": "text.txt", "start_line": 2})
    assert isinstance(next_page, dict)
    assert next_page["content"] == "next"
    assert next_page["end_line"] == 2


# 函数说明：test_character_budget_covers_whole_page_and_final_long_line
# 用途：回归验证回归测试与测试辅助中的
# `character_budget_covers_whole_page_and_final_long_line` 场景，下方断言说明列出实际通
# 过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`(tmp_path / 'text.txt').write_text` →
#  `ReadFileTool(tmp_path).execute` → `ReadFileTool`。
# 分支与异常：
#   验证条件：`isinstance(result, dict)`。
#   验证条件：`len(result['content']) == MAX_RANGE_CHARS`。
#   验证条件：`result['line_truncated'] == 2`。
#   验证条件：`result['end_line'] == 2`。
# 副作用与资源：
#   文件或资源访问：`(tmp_path / 'text.txt').write_text`。
@pytest.mark.asyncio
async def test_character_budget_covers_whole_page_and_final_long_line(
    tmp_path: Path,
) -> None:
    (tmp_path / "text.txt").write_text(
        "head\n" + "x" * MAX_RANGE_CHARS, encoding="utf-8"
    )

    result = await ReadFileTool(tmp_path).execute({"path": "text.txt", "max_lines": 3})
    assert isinstance(result, dict)
    assert len(result["content"]) == MAX_RANGE_CHARS
    assert result["line_truncated"] == 2
    assert result["end_line"] == 2
    assert result["next_line"] is None
    assert result["truncated"] is True


# 函数说明：test_invalid_range_arguments_are_rejected
# 用途：回归验证回归测试与测试辅助中的 `invalid_range_arguments_are_rejected` 场景，下方
# 断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
#   field：待校验的字段名，类型 `str`。
#   value：待校验、规范化或转换的值，类型 `object`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`(tmp_path / 'text.txt').write_text` →
#  `pytest.raises` → `ReadFileTool(tmp_path).execute` → `ReadFileTool`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match=field)`。
# 副作用与资源：
#   文件或资源访问：`(tmp_path / 'text.txt').write_text`。
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start_line", 0), ("start_line", -1), ("start_line", True),
        ("start_line", "2"), ("start_line", 1.5), ("start_line", None),
        ("max_lines", 0), ("max_lines", False), ("max_lines", "3"),
        ("max_lines", MAX_READ_LINES + 1),
    ],
)
async def test_invalid_range_arguments_are_rejected(
    tmp_path: Path, field: str, value: object,
) -> None:
    (tmp_path / "text.txt").write_text("text", encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        await ReadFileTool(tmp_path).execute({"path": "text.txt", field: value})


# 函数说明：test_ranged_read_keeps_workspace_and_utf8_checks
# 用途：回归验证回归测试与测试辅助中的 `ranged_read_keeps_workspace_and_utf8_checks` 场
# 景，下方断言说明列出实际通过条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ReadFileTool` → `pytest.raises` →
# `tool.execute` → `(tmp_path / 'binary.txt').write_bytes`。
# 分支与异常：
#   预期异常：`pytest.raises(ValueError, match='escapes the workspace')`。
#   预期异常：`pytest.raises(FileNotFoundError)`。
#   预期异常：`pytest.raises(UnicodeDecodeError)`。
# 副作用与资源：
#   文件或资源访问：`(tmp_path / 'binary.txt').write_bytes`。
@pytest.mark.asyncio
async def test_ranged_read_keeps_workspace_and_utf8_checks(tmp_path: Path) -> None:
    tool = ReadFileTool(tmp_path)
    with pytest.raises(ValueError, match="escapes the workspace"):
        await tool.execute({"path": "../secret.txt", "max_lines": 2})
    with pytest.raises(FileNotFoundError):
        await tool.execute({"path": "missing.txt", "start_line": 1})
    (tmp_path / "binary.txt").write_bytes(b"\xff\xfe")
    with pytest.raises(UnicodeDecodeError):
        await tool.execute({"path": "binary.txt", "max_lines": 2})


# 函数说明：test_read_schema_preserves_path_only_and_documents_bounds
# 用途：回归验证回归测试与测试辅助中的
# `read_schema_preserves_path_only_and_documents_bounds` 场景，下方断言说明列出实际通过
# 条件。
# 参数：
#   tmp_path：pytest 提供的隔离临时目录，类型 `Path`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`ReadFileTool`。
# 分支与异常：
#   验证条件：`definition.parameters['required'] == ['path']`。
#   验证条件：`definition.parameters['properties']['start_line']['minimum'] == 1`。
#   验证条件：
# `definition.parameters['properties']['max_lines']['maximum'] == MAX_READ_LINES`。
#   验证条件：`definition.strict is False`。
def test_read_schema_preserves_path_only_and_documents_bounds(tmp_path: Path) -> None:
    definition = ReadFileTool(tmp_path).definition
    assert definition.parameters["required"] == ["path"]
    assert definition.parameters["properties"]["start_line"]["minimum"] == 1
    assert definition.parameters["properties"]["max_lines"]["maximum"] == MAX_READ_LINES
    assert definition.strict is False
