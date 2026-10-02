from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from app.domain.memory.maintenance_models import MemoryMaintenanceDecision
from app.domain.memory.maintenance_reflection import _MAINTENANCE_PROMPT
from app.domain.memory.reflection import _REFLECTION_PROMPT
from app.domain.memory.reflection_models import ReflectionDecision
from app.domain.skill_learning.distiller import _Distilled, _OverlapDecision
from app.domain.skill_learning.models import PatternMiningResult
from app.domain.skill_learning.prompts import (
    _DISTILLATION_PROMPT,
    _OVERLAP_ADJUDICATION_PROMPT,
    _PATTERN_MINING_PROMPT,
)


# 函数说明：test_auxiliary_prompt_example_matches_runtime_parser
# 用途：回归验证回归测试与测试辅助中的 `auxiliary_prompt_example_matches_runtime_parser`
#  场景，下方断言说明列出实际通过条件。
# 参数：
#   prompt：本次调用使用的提示文本，类型 `str`。
#   parser：`parser`输入或配置值，类型 `type[BaseModel]`。
#   fields：`fields`输入或配置值，类型 `set[str]`。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`json.JSONDecoder().raw_decode` →
# `json.JSONDecoder` → `prompt.index` → `parser.model_validate`。
# 分支与异常：
#   验证条件：`set(payload) == fields`。
@pytest.mark.parametrize(
    ("prompt", "parser", "fields"),
    [
        pytest.param(
            _REFLECTION_PROMPT,
            ReflectionDecision,
            {"action", "memory_id", "title", "summary", "content", "reason"},
            id="memory-reflection",
        ),
        pytest.param(
            _MAINTENANCE_PROMPT,
            MemoryMaintenanceDecision,
            {"action", "memory_id", "reason"},
            id="memory-maintenance",
        ),
        pytest.param(
            _PATTERN_MINING_PROMPT,
            PatternMiningResult,
            {"clusters"},
            id="pattern-mining",
        ),
        pytest.param(
            _DISTILLATION_PROMPT,
            _Distilled,
            {
                "action", "proposed_name", "description", "reason", "procedure",
                "pitfalls", "verification", "existing_skill_name",
            },
            id="skill-distillation",
        ),
        pytest.param(
            _OVERLAP_ADJUDICATION_PROMPT,
            _OverlapDecision,
            {"relationship", "existing_skill_name", "reason"},
            id="skill-overlap",
        ),
    ],
)
def test_auxiliary_prompt_example_matches_runtime_parser(
    prompt: str,
    parser: type[BaseModel],
    fields: set[str],
) -> None:
    # Models may copy the displayed example. It must be accepted as-is,
    # including enum values and the null fields of a no-mutation decision.
    payload, _ = json.JSONDecoder().raw_decode(prompt[prompt.index("{"):])
    assert set(payload) == fields
    parser.model_validate(payload)
