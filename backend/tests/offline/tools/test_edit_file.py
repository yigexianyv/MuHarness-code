"""edit_file：只提交改动片段修改已有文件，避免整文件重写导致单次输出过长被截断。"""

from __future__ import annotations

import pytest

from app.models.types import AgentMode
from app.tools.builtin import EditFileTool, build_builtin_tool_registry
from app.tools.role_boundary import AUDIT_ALLOWED_TOOLS, EXECUTE_ALLOWED_TOOLS


@pytest.fixture
def tool(tmp_path):
    (tmp_path / "pkg").mkdir()
    # 按字节写入，固定 LF 换行（Windows 上 write_text 会把 \n 转成 \r\n）
    (tmp_path / "pkg" / "mod.py").write_bytes(
        b"def a():\n    return 1\n\n\ndef b():\n    return 1\n"
    )
    return EditFileTool(tmp_path)


@pytest.mark.asyncio
async def test_replaces_unique_fragment_only(tool, tmp_path) -> None:
    result = await tool.execute(
        {
            "path": "pkg/mod.py",
            "old_string": "def a():\n    return 1",
            "new_string": "def a():\n    return 2",
        }
    )
    text = (tmp_path / "pkg" / "mod.py").read_bytes().decode("utf-8")
    assert text == "def a():\n    return 2\n\n\ndef b():\n    return 1\n"
    assert result == {"path": "pkg/mod.py", "replacements": 1, "characters": len(text)}


@pytest.mark.asyncio
async def test_rejects_ambiguous_fragment_unless_replace_all(tool, tmp_path) -> None:
    with pytest.raises(ValueError, match="occurs 2 times"):
        await tool.execute(
            {"path": "pkg/mod.py", "old_string": "return 1", "new_string": "return 3"}
        )
    # 失败时文件保持原样
    assert "return 3" not in (tmp_path / "pkg" / "mod.py").read_text(encoding="utf-8")

    result = await tool.execute(
        {
            "path": "pkg/mod.py",
            "old_string": "return 1",
            "new_string": "return 3",
            "replace_all": True,
        }
    )
    assert result["replacements"] == 2
    text = (tmp_path / "pkg" / "mod.py").read_text(encoding="utf-8")
    assert text.count("return 3") == 2


@pytest.mark.asyncio
async def test_missing_fragment_and_missing_file_give_actionable_errors(tool) -> None:
    with pytest.raises(ValueError, match="not found"):
        await tool.execute(
            {"path": "pkg/mod.py", "old_string": "def c():", "new_string": "x"}
        )
    with pytest.raises(ValueError, match="write_file"):
        await tool.execute(
            {"path": "pkg/new.py", "old_string": "x", "new_string": "y"}
        )
    with pytest.raises(ValueError, match="non-empty"):
        await tool.execute({"path": "pkg/mod.py", "old_string": "", "new_string": "y"})
    with pytest.raises(ValueError, match="identical"):
        await tool.execute(
            {"path": "pkg/mod.py", "old_string": "def a", "new_string": "def a"}
        )
    with pytest.raises(ValueError, match="relative|escapes"):
        await tool.execute(
            {"path": "/etc/passwd", "old_string": "root", "new_string": "x"}
        )
    with pytest.raises(ValueError, match="escapes"):
        await tool.execute(
            {"path": "../outside.py", "old_string": "x", "new_string": "y"}
        )


@pytest.mark.asyncio
async def test_append_by_anchoring_on_last_lines(tool, tmp_path) -> None:
    await tool.execute(
        {
            "path": "pkg/mod.py",
            "old_string": "def b():\n    return 1\n",
            "new_string": "def b():\n    return 1\n\n\ndef c():\n    return 3\n",
        }
    )
    assert (tmp_path / "pkg" / "mod.py").read_text(encoding="utf-8").endswith(
        "def c():\n    return 3\n"
    )


@pytest.mark.asyncio
async def test_crlf_file_matches_lf_fragment_and_keeps_crlf(tmp_path) -> None:
    target = tmp_path / "win.py"
    target.write_bytes(b"x = 1\r\ny = 2\r\n")
    tool = EditFileTool(tmp_path)
    await tool.execute(
        {"path": "win.py", "old_string": "x = 1\ny = 2", "new_string": "x = 1\ny = 3"}
    )
    assert target.read_bytes() == b"x = 1\r\ny = 3\r\n"


def test_registered_for_executor_and_closing_but_not_auditor(tmp_path) -> None:
    registry = build_builtin_tool_registry(tmp_path)
    names = {definition.name for definition in registry.definitions()}
    assert "edit_file" in names
    assert "edit_file" in EXECUTE_ALLOWED_TOOLS
    assert "edit_file" not in AUDIT_ALLOWED_TOOLS
    definition = EditFileTool(tmp_path).definition
    assert definition.closing_allowed is True
    assert definition.parameters["required"] == ["path", "old_string", "new_string"]


@pytest.mark.asyncio
async def test_read_only_roles_cannot_edit(tool) -> None:
    class Context:
        mode = AgentMode.AUDIT

    with pytest.raises(PermissionError, match="read-only"):
        await tool.execute_with_context(
            {"path": "pkg/mod.py", "old_string": "def a", "new_string": "def z"},
            Context(),
        )
