from app.models.providers.openai_compatible import _chat_tool, _responses_tool
from app.models.types import ToolDefinition


def test_optional_parameters_preserve_schema_and_disable_strict_mode():
    schema = {"type": "object", "properties": {"name": {"type": "string"},
              "count": {"type": "integer"}}, "required": ["name"],
              "additionalProperties": False}
    tool = ToolDefinition(name="example", description="example", parameters=schema, strict=True)
    for wire in (_responses_tool(tool), _chat_tool(tool)["function"]):
        assert wire["strict"] is False
        assert wire["parameters"] == schema
    assert schema["required"] == ["name"]


def test_strict_mode_is_kept_for_fully_required_schema():
    schema = {"type": "object", "properties": {"name": {"type": "string"}},
              "required": ["name"], "additionalProperties": False}
    tool = ToolDefinition(name="example", description="example", parameters=schema, strict=True)
    assert _responses_tool(tool)["strict"] is True


def test_nested_optional_schema_disables_strict_mode():
    child = {"type": "object", "properties": {"value": {"type": "string"}},
             "required": [], "additionalProperties": False}
    schema = {"type": "object", "properties": {"items": {"type": "array", "items": child}},
              "required": ["items"], "additionalProperties": False}
    tool = ToolDefinition(name="example", description="example", parameters=schema, strict=True)
    assert _responses_tool(tool)["strict"] is False
