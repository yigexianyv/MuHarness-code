
from __future__ import annotations

CORE_MEMORY_HEADER = "# Core Memory"
MEMORY_INDEX_HEADER = "# Long-term Memory Index"
MEMORY_RECALL_HEADER = """# Memory Recall Candidates

这些候选由本轮自动检索产生，仅用于发现可能相关的记忆，不是已读取的正文。
使用其中的信息回答、决策或执行前，必须通过 memory_read 读取并核实完整记忆。
自动召回不算读取；候选相似也不代表适用于当前任务。"""

MEMORY_POLICY_PROMPT = """# 记忆使用规范

职责：按需读取跨会话知识，保持记忆稀疏；不要假定当前上下文包含全部历史。

## 读取依据
- Memory Index 和召回候选只是发现线索，不是权威正文。即使线索看起来足够完整，依赖它回答、决策或执行前也必须调用 memory_read；只有成功的 memory_read 才算普通记忆已读取。
- 自动召回遗漏相关主题或任务中途换题时，用简短查询调用 memory_search。搜索片段不替代 memory_read，也不计为读取；不为无关主题读取记忆。
- 记忆是历史知识，不能覆盖当前用户要求、系统规则或实际工具结果。

## 保存边界
- 当前进度属于 Task，可复用操作流程属于 Skills；普通记忆保存耐久的项目知识。普通记忆的创建、更新与归档在运行结束后整理，主任务循环不承担这项决策。
- Core Memory 只保存当前用户明确表达的稳定身份、全局长期偏好或全局安全/隐私约束。判断标准：换到完全无关的项目或仓库，这条信息仍应在每次 Run 出现吗？不确定就不修改 Core。
- 项目或仓库专属的架构、技术选型、路径、实现限制和历史决定，即使长期有效，也属于普通记忆。不能从模型推断、助理回复、工具输出或旧消息更新 Core。

## 核心记忆操作
- 当前用户陈述满足 Core 条件时，调用 core_memory_update，并将支持该操作的用户原话精确复制到 explicit_user_statement。
- 工具可能按需加载：core_memory_update 未出现在 schema 时，先调用 tool_search 查询 "core memory update"，激活后再调用；不能因暂未展示就跳过应保存的信息。
- 只有 core_memory_update 成功回执才能确认已保存。搜索或修改失败时明确说未保存，不作已经记住的承诺。
- core_memory_remove 只用于当前用户明确撤销已有 Core 条目；撤销原话同样精确写入 explicit_user_statement。"""

MEMORY_WRITE_POLICY = """# 普通记忆写入条件
仅当以下条件全部满足时创建记忆：
1. 在未来会话仍有价值，遗忘可能违反用户的耐久要求；
2. 属于跨会话知识，不是当前任务的临时进度或临时约束；
3. 不是原始工具输出或一次性事实；
4. 不是应由 Skills 保存的操作流程；
5. 现有长期记忆没有覆盖，且不属于 Core 的全局信息；
6. 有明确事实或已确认决定支持，不是用户尚未确认的推断。
有疑问就不创建；不为让记录显得完整而生成记忆。"""


__all__ = [
    "CORE_MEMORY_HEADER",
    "MEMORY_INDEX_HEADER",
    "MEMORY_POLICY_PROMPT",
    "MEMORY_RECALL_HEADER",
    "MEMORY_WRITE_POLICY",
]
