

class AgentRuntimeError(RuntimeError):
    pass


class ModelInvocationError(AgentRuntimeError):

    # 函数说明：ModelInvocationError.__init__
    # 用途：初始化 ModelInvocationError；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   detail：`detail`输入或配置值，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    def __init__(self, detail: str) -> None:
        super().__init__(f"model invocation failed: {detail}")


class ContextPreparationError(AgentRuntimeError):

    # 函数说明：ContextPreparationError.__init__
    # 用途：初始化 ContextPreparationError；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   detail：`detail`输入或配置值，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    def __init__(self, detail: str) -> None:
        super().__init__(f"context preparation failed: {detail}")


class ContextWindowExceededError(ContextPreparationError):

    # 函数说明：ContextWindowExceededError.__init__
    # 用途：初始化 ContextWindowExceededError；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   estimated_tokens：Token 数量或 Token 预算，类型 `int`。
    #   input_budget：预算输入或配置值，类型 `int`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    def __init__(self, estimated_tokens: int, input_budget: int) -> None:
        super().__init__(
            f"estimated input tokens ({estimated_tokens}) exceed "
            f"input budget ({input_budget})"
        )


class MaxStepsExceededError(AgentRuntimeError):

    # 函数说明：MaxStepsExceededError.__init__
    # 用途：初始化 MaxStepsExceededError；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   max_steps：模型执行步数上限，类型 `int`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    def __init__(self, max_steps: int) -> None:
        super().__init__(f"maximum step limit ({max_steps}) reached")


class RunBudgetExceededError(AgentRuntimeError):

    # 函数说明：RunBudgetExceededError.__init__
    # 用途：初始化 RunBudgetExceededError；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   detail：`detail`输入或配置值，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    def __init__(self, detail: str) -> None:
        super().__init__(f"main agent run budget exceeded: {detail}")


class RepeatedToolCallError(AgentRuntimeError):

    # 函数说明：RepeatedToolCallError.__init__
    # 用途：初始化 RepeatedToolCallError；参数及实际保存的实例字段见下方说明。
    # 参数：
    #   tool_name：工具名称，类型 `str`。
    # 返回：类型 `None`；不返回结果值（隐式 None）。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`super().__init__` → `super`。
    def __init__(self, tool_name: str) -> None:
        super().__init__(
            f"tool {tool_name!r} was called with identical arguments "
            "3 consecutive times"
        )
