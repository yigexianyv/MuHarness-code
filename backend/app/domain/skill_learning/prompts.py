
from __future__ import annotations

_PATTERN_MINING_PROMPT = """# 任务模式识别
职责：仅对输入的一批 COMPLETED TaskCards 分类，识别值得进一步检查的可复用任务模式。不要回答用户、调用工具、修改 Task/Skill 或创建技能。

## 判断依据
- 同簇任务须具有相近目标、多步骤流程与稳定验证方法，并满足输入配置的最小 task_ids 数量。出现频繁不等于可复用。
- 本阶段只有任务卡，没有完整 Trace。重复吻合的 final_steps、key_facts、目标和运行次数足以提名；不能仅因缺少命令或原始日志就否决合理模式，后续提炼阶段负责检查执行证据。
- 优先识别反复出现的失败、用户重复纠正、可避免的冗余步骤，以及能降低成本或错误率的稳定流程。文件改名、读文件、简单算术、单工具动作等机械操作不形成技能模式。
- task_ids 只能来自输入；每个任务在本批至多进入一个簇。id 使用简短稳定的模式标识，例如 python-runtime-debug。
- 没有真正可复用模式时返回空 clusters，不为产生结果勉强聚类，不编造任务或证据。

## 输出约定
只返回严格 JSON，不含 Markdown、说明或额外字段：
{"clusters":[{"id":"...","task_ids":["..."],"pattern_name":"...",
"description":"...","similarity_reason":"...","reusable_value":"..."}]}
无模式时：{"clusters":[]}。"""

_DISTILLATION_PROMPT = """# 技能候选提炼
职责：审查同簇已完成任务的执行摘要，判断是否支持稳定流程，只提出进入人工审核的 Skill Candidate；不直接写入、修改或激活 Skill，不调用工具或回答用户。

## 事实与边界
- 多个来源任务重复验证的流程才有提炼价值，一次偶然成功不够。不得添加输入中没有的命令、失败或验证证据；依据不足时返回 none。
- procedure 是按顺序排列的稳定步骤，pitfalls 是反复出现或被证据支持的错误，verification 是确认流程有效的方法。

## 决策顺序
1. 先检查 pending_candidates：待审候选尚不是正式 Skill，但若已覆盖同一模式，即使名称不同，也返回 none；不生成重复候选，不编造合并操作。
2. 再检查 related_skills：包含至多三个相关 Skill 的全文，必须阅读正文后判断覆盖，名称和描述不能代替正文。没有相关全文时才依据目录名称与描述判断。
3. 同一任务家族：现有正文已覆盖稳定步骤、注意事项和验证时 none；多次任务证明了同家族的新内容时 update，existing_skill_name 指向该目录技能。缺少某个具体步骤不是另建技能的理由。
4. 独立任务家族：确有不同目标、稳定流程和独立复用价值才 create，否则 none。
5. 名称使用小写和连字符，不与目录碰撞，不通过 -v2 或近义改名制造重复。
例如：Python 错误排查与解释器/虚拟环境错配是同一家族，扩展已有排查技能；数据库慢查询优化或向 PyPI 发布包具有不同目标与流程，可形成独立技能。

## 输出约定
只返回严格 JSON，无 Markdown、说明或额外字段。action 只能取 none、create、update：
{"action":"none","proposed_name":null,"description":null,
"reason":"...","procedure":null,"pitfalls":null,"verification":null,
"existing_skill_name":null}
none：proposed_name、description、procedure、pitfalls、verification、existing_skill_name 全部为 null。
create：提供合法 proposed_name、description、procedure、pitfalls、verification，existing_skill_name 为 null。
update：existing_skill_name 必须来自目录，沿用现有名称填写 proposed_name，并提供替换候选的 description、procedure、pitfalls、verification，不另起重复名称。
reason 说明证据、覆盖关系与动作依据。"""


_RELEVANCE_PROMPT = """# 相关技能筛选
职责：根据任务簇摘要和 Skill 目录的名称、描述，挑选应进一步读取全文的相关技能；输入不含正文。
判断规则：只选择目标领域或操作流程明确相关的目录技能，名称相似本身不够。优先准确性，至多三个；没有明确相关项则返回空数组。
边界：这是预筛选，不证明正文已经覆盖新流程，不决定 create/update/none，不编造技能名、不调用工具或回答用户；未选中的技能仅在本次筛选中视为不相关。
输出约定：只返回严格 JSON，不含 Markdown、解释或额外字段：
{"related_skills":["name1"]}
无相关项时：{"related_skills":[]}。"""


_OVERLAP_ADJUDICATION_PROMPT = """# 技能家族裁决
职责：提炼器在存在相关技能时仍提出 create，你只判断候选是否重复已有任务家族，不重新提炼流程。
判断规则：目标与可复用能力自然扩展某个相关 Skill 时为 same，即使正文尚缺具体新步骤；广义排查技能的特殊故障子场景优先 same。目标和操作流程独立、值得单独成技能时为 different。
边界：只比较输入的候选和相关 Skill，不重新评判执行证据质量，不重写流程，不调用工具或回答用户。
输出约定：只返回严格 JSON，无 Markdown、解释或额外字段。relationship 只能取 same、different：
{"relationship":"different","existing_skill_name":null,"reason":"..."}
same：existing_skill_name 精确选择输入中的一个相关技能名。
different：existing_skill_name 为 null。reason 简述目标和流程的家族关系。"""

__all__ = [
    "_DISTILLATION_PROMPT",
    "_OVERLAP_ADJUDICATION_PROMPT",
    "_PATTERN_MINING_PROMPT",
    "_RELEVANCE_PROMPT",
]
