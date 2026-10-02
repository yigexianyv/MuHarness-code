
from __future__ import annotations

from app.application import DEFAULT_SYSTEM_PROMPT, Application
from app.models.config import ModelSettings
from app.models.registry import ModelAdapterRegistry


# 函数说明：test_default_system_prompt_mentions_artifact_publish
# 用途：回归验证回归测试与测试辅助中的 `default_system_prompt_mentions_artifact_publish`
#  场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 分支与异常：
#   验证条件：`'artifact_publish' in DEFAULT_SYSTEM_PROMPT`。
def test_default_system_prompt_mentions_artifact_publish() -> None:
    assert "artifact_publish" in DEFAULT_SYSTEM_PROMPT


# 函数说明：test_default_system_prompt_publish_before_final_answer
# 用途：回归验证回归测试与测试辅助中的
# `default_system_prompt_publish_before_final_answer` 场景，下方断言说明列出实际通过条件
# 。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 分支与异常：
#   验证条件：`'最终回答前' in DEFAULT_SYSTEM_PROMPT`。
#   验证条件：`'artifact_publish' in DEFAULT_SYSTEM_PROMPT`。
def test_default_system_prompt_publish_before_final_answer() -> None:
    assert "最终回答前" in DEFAULT_SYSTEM_PROMPT
    assert "artifact_publish" in DEFAULT_SYSTEM_PROMPT


# 函数说明：test_default_system_prompt_excludes_non_deliverables
# 用途：回归验证回归测试与测试辅助中的 `default_system_prompt_excludes_non_deliverables`
#  场景，下方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 分支与异常：
#   验证条件：`'不作为交付物发布' in DEFAULT_SYSTEM_PROMPT`。
#   验证条件：`'Trace' in DEFAULT_SYSTEM_PROMPT`。
#   验证条件：`'运行日志' in DEFAULT_SYSTEM_PROMPT`。
def test_default_system_prompt_excludes_non_deliverables() -> None:
    assert "不作为交付物发布" in DEFAULT_SYSTEM_PROMPT
    assert "Trace" in DEFAULT_SYSTEM_PROMPT
    assert "运行日志" in DEFAULT_SYSTEM_PROMPT


# 函数说明：test_default_system_prompt_no_formal_publish_without_deliverable
# 用途：回归验证回归测试与测试辅助中的
# `default_system_prompt_no_formal_publish_without_deliverable` 场景，下方断言说明列出实
# 际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 分支与异常：
#   验证条件：`'没有实际交付物就不调用 artifact_publish' in DEFAULT_SYSTEM_PROMPT`。
def test_default_system_prompt_no_formal_publish_without_deliverable() -> None:
    assert "没有实际交付物就不调用 artifact_publish" in DEFAULT_SYSTEM_PROMPT


# 函数说明：test_system_prompt_defines_task_granularity
# 用途：回归验证回归测试与测试辅助中的 `system_prompt_defines_task_granularity` 场景，下
# 方断言说明列出实际通过条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 分支与异常：
#   验证条件：`'一个整体目标对应一个 Task' in DEFAULT_SYSTEM_PROMPT`。
#   验证条件：`'内部阶段用 Steps 表达' in DEFAULT_SYSTEM_PROMPT`。
#   验证条件：`'彼此独立' in DEFAULT_SYSTEM_PROMPT`。
def test_system_prompt_defines_task_granularity() -> None:
    assert "一个整体目标对应一个 Task" in DEFAULT_SYSTEM_PROMPT
    assert "内部阶段用 Steps 表达" in DEFAULT_SYSTEM_PROMPT
    assert "彼此独立" in DEFAULT_SYSTEM_PROMPT


# 函数说明：test_system_prompt_requires_a_real_need_before_using_tools
# 用途：回归验证回归测试与测试辅助中的
# `system_prompt_requires_a_real_need_before_using_tools` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 分支与异常：
#   验证条件：`'必要的实时或外部信息' in DEFAULT_SYSTEM_PROMPT`。
#   验证条件：`'普通知识问答和能力说明直接回答' in DEFAULT_SYSTEM_PROMPT`。
#   验证条件：`'所需工具确实不存在时如实说明' in DEFAULT_SYSTEM_PROMPT`。
#   验证条件：`'不用无关搜索或文件操作试探' in DEFAULT_SYSTEM_PROMPT`。
def test_system_prompt_requires_a_real_need_before_using_tools() -> None:
    assert "必要的实时或外部信息" in DEFAULT_SYSTEM_PROMPT
    assert "普通知识问答和能力说明直接回答" in DEFAULT_SYSTEM_PROMPT
    assert "所需工具确实不存在时如实说明" in DEFAULT_SYSTEM_PROMPT
    assert "不用无关搜索或文件操作试探" in DEFAULT_SYSTEM_PROMPT


# 函数说明：test_application_uses_default_system_prompt_when_omitted
# 用途：回归验证回归测试与测试辅助中的
# `application_uses_default_system_prompt_when_omitted` 场景，下方断言说明列出实际通过条
# 件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Application` → `ModelAdapterRegistry`
#  → `ModelSettings`。
# 分支与异常：
#   验证条件：`application.system_prompt == DEFAULT_SYSTEM_PROMPT`。
#   验证条件：`application.max_steps == 12`。
def test_application_uses_default_system_prompt_when_omitted() -> None:
    application = Application(
        provider="fake",
        model="fake-model",
        registry=ModelAdapterRegistry(ModelSettings(_env_file=None)),
    )

    assert application.system_prompt == DEFAULT_SYSTEM_PROMPT
    assert application.max_steps == 12


# 函数说明：test_application_allows_explicitly_disabling_system_prompt
# 用途：回归验证回归测试与测试辅助中的
# `application_allows_explicitly_disabling_system_prompt` 场景，下方断言说明列出实际通过
# 条件。
# 返回：类型 `None`；不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`Application` → `ModelAdapterRegistry`
#  → `ModelSettings`。
# 分支与异常：
#   验证条件：`application.system_prompt == ''`。
def test_application_allows_explicitly_disabling_system_prompt() -> None:
    application = Application(
        provider="fake",
        model="fake-model",
        system_prompt="",
        registry=ModelAdapterRegistry(ModelSettings(_env_file=None)),
    )

    assert application.system_prompt == ""
