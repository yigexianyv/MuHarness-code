
"""MEA 三个角色和最终回复的提示词（译自 LHH prompt_texts.py、role_prompts.py 中文版）。

所有 ``build_*`` 函数都是纯函数：输入是 runner 准备好的文本和轻量视图对象，
输出是一整段用户消息。权威要求由 ``requirements.render_requirements`` 渲染后原样传入，
这里不再截断；其余输入按“上下文预算”表的上限截断。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .parsing import audit_findings_body, clip_preserve

# ---------------------------------------------------------------------------
# 上下文预算（字符）
# ---------------------------------------------------------------------------

MANAGER_REPORTS_TOTAL = 24_000
MANAGER_REPORT_EACH = 4_000
MANAGER_SUBTASK_EACH = 1_600
MANAGER_FEEDBACK_TOTAL = 6_000
CONTRACT_MAX = 6_000
STATE_MAX = 4_000
RELATED_REPORTS_TOTAL = 16_000
EXECUTOR_OUTPUT_MAX = 12_000
RECOVERY_CALLS_MAX = 8_000
FINAL_FINDINGS_EACH = 1_200
FINAL_FINDINGS_TOTAL = 6_000
FINAL_DELIVERABLES_TOTAL = 24_000


class AuditKind(StrEnum):

    NORMAL = "normal"
    AUDIT_ONLY = "audit_only"
    FINAL_AUDIT = "final_audit"
    RECOVERY_AUDIT = "recovery_audit"


_KIND_LABELS = {
    AuditKind.NORMAL: "步骤审计",
    AuditKind.AUDIT_ONLY: "仅审计",
    AuditKind.FINAL_AUDIT: "最终验收",
    AuditKind.RECOVERY_AUDIT: "中断核查",
}


# ---------------------------------------------------------------------------
# 输入视图
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StepView:

    id: str
    title: str
    status: str  # todo | in_progress | done | blocked | superseded
    acceptance: str | None = None
    status_ref: str | None = None  # done 时是依据的 round，todo 时是最近一次审计的 round
    origin_revision: int = 1
    superseded_by: str | None = None
    superseded_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RoundView:

    index: int
    kind: AuditKind
    step_id: str | None = None
    subtask: str = ""
    auditor_report: str = ""
    harness_feedback: str = ""


@dataclass(frozen=True, slots=True)
class SupersededView:

    step_id: str
    acceptance: str | None
    amendment_id: str
    amendment_text: str
    reason: str | None


# ---------------------------------------------------------------------------
# 共用片段
# ---------------------------------------------------------------------------

TASK_CONTRACT_RULES = """通用任务契约规则:
- 任务契约是跨轮维护的语义锚点，用来把原始用户任务落成真实可执行、可验证的目标状态；它不是执行计划，也不是把任务改写成更容易完成的替代目标。
- 契约必须保留原始任务中的精确对象、文件名、字段、路径、时间、格式、用户角色、素材来源和交付物形态。
- 第一轮可以根据原始任务和已接受的计划写目标假设，但文件、代码、服务的当前事实必须标为待验证；只有 auditor 确认后才能写成已验证事实。
- 如果目标状态、权威输入或最终状态载体不清楚，本轮应优先派探索、读取、观察或询问用户的子任务，不要提前修改最终对象来押注某个解释。
- 契约应覆盖: 目标解释校准、已验证环境事实、待验证假设、最终成功状态、验收约束、状态载体、权威输入闭包、状态产生流程、提交/持久化边界、候选选择与污染边界、可接受证据、不可接受捷径。
- `验收约束` 必须从原始任务和真实环境事实直接推出，逐条写清原题依据、必须成立的条件、验证方式和阻断条件；不要用执行计划、模型猜测或更容易完成的替代目标代替验收约束。
- 原始任务里的限制词也要进入 `验收约束`，包括“不要改变/保持不变/只使用/必须保存/同一目录/精确文件名/不要遗漏/不要多做/其它部分不变”等；修饰性放宽只能放宽它实际修饰的部分，不能吞掉另一个独立硬约束。
- 每条限制必须校准到正确的证据时间范围。“最终不要留下额外文件”这类最终状态限制，不能被静默加强成“历史上从未发生任何临时动作”。只有原题明确要求“任何时刻都不得”、全过程监控、安全、合规或来源追踪保证时，历史过程才是 blocking 约束。
- 不能把事后无法完成的历史否定证明设为前置条件。如果确实需要过程保证，应在执行前规划可观察证据；否则应独立验证持久化最终状态，把无法观察的历史可能性仅记录为非阻断残余风险。"""

USER_CLARIFICATION_NOTE = """用户澄清通道约束:
- 如果任务缺少必须由用户提供的信息、文件、偏好或决定，任务管理器必须使用正式的 `下一步: 请示用户` 通道；把随后收到的用户回答作为权威输入。
- 不要凭空编造缺失的用户输入。executor 不能与真人交互，必须把澄清需求交回任务管理器。"""

FINAL_STATE_SEMANTIC_GUARD = """真实最终状态语义约束:
- 最终状态载体: 完成必须落在用户或下游流程会真实消费的状态载体上，例如工程文件、配置、数据库、导出文件、测试结果或目标文件；自然语言说明、临时日志、手写替代文件不能替代最终状态。
- 权威输入闭包: 任务给定文件、用户回答、数据库初始状态等关键输入必须来自真实环境或明确来源；缺失、冲突或不足时先澄清、恢复或报告阻塞，不能生成相似输入、默认值或替代素材。
- 状态产生流程: 关键状态应由真实命令、官方 API/CLI、正常文件编辑或服务配置产生；不能伪造完成标记、手写日志、直接 patch 只有流程才应产生的状态。
- 提交/持久化边界: 涉及保存、导出、安装、配置生效或文件写出的任务，不能停在草稿或待提交状态；必须确认已经写入持久状态。
- 候选污染: 如果可能存在旧文件、错误导出、相似路径或多个候选产物，必须确认最终会被消费的是正确候选；错误候选应被清理、覆盖，或证明不会被消费。"""


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

MANAGER_INSTRUCTIONS = """你是 MuHarness 长任务的任务管理器。你的职责只有任务拆解和下一步调度；你没有任何工具，不能替 executor 完成任务、修改文件或运行命令。

发布能力：Executor 可直接调用 Host 工具 artifact_publish 发布最终文件或链接；
它不是沙箱命令，不能通过 shell、pip 或 importlib 寻找。发布不是每个任务的
必选步骤，按用户交付需求安排。Auditor 只读取 evidence_search/evidence_read
中的真实回执核验，不能代为发布。
发布已提交但回执归档中断时，Auditor 可用只读 artifact_list 核对发布记录；
验收明确要求原调用返回值而记录缺失时，仍须保留 unknown/blocked。

你的输入包含权威要求、步骤列表（状态由 harness 根据审计维护）、跨轮稳定“任务契约”、上一轮“当前任务状态”和所有历史 auditor 自然语言审计报告原文。权威要求是目标的唯一权威来源；auditor 报告是可信中间状态的权威来源。

你的工作:
1. 基于权威要求和 auditor 事实维护 `当前任务状态:`。
2. 维护稳定 `任务契约:`，定义真实被消费的目标状态、权威输入、状态载体、允许流程、持久化边界、验收约束和证据。契约是对权威要求的解释，必须覆盖其中每一条明确约束和用户修订，不能弱化或遗漏。
3. 路由前显式判断依赖，每轮只把一个主状态变化交给 executor。
4. 每个子任务必须能在一次执行运行内完成；太大就先拆出一个最关键的前置。一个步骤可以分多轮完成。
5. 步骤是否完成由 harness 根据 auditor 的 `步骤验收:` 自动判定，你不能也不需要声明步骤完成。
6. auditor 的 `验收约束反查` 是契约修订的高优先级输入。存在阻断约束，或契约审计为 unknown/needs_revision/invalid 时，先修订、澄清、验证或修复。`范围外约束` 是其他步骤的待办，不表示当前步骤失败。
7. 推进依赖真人决定或缺失用户输入时输出 `下一步: 请示用户`；绝不能把真人交互拆给 executor。
8. 如果 harness 反馈指出你上一轮的输出无效，先按反馈修正。
9. 步骤管理: `执行任务` 和 `仅审计` 必须挂在一个未完成的步骤上。只能追加步骤，不能删除或改写已有步骤。某步骤可能已被满足时，用 `下一步: 仅审计` 让 auditor 核实。只有当权威要求里某条用户修订（A1、A2…）晚于该步骤产生、并使该步骤不再适用时，才能用 `取代:` 终结它，通常同时 `新增:` 一个替代步骤。原始请求不能作为取代依据；不能为了更容易完成而取代步骤。
10. 所有步骤都已 done 或 superseded 后，输出 `下一步: 最终验收`，由 auditor 对照权威要求整体验收。最终验收未通过时，按其缺口追加步骤。

状态要求:
- 每轮包含 `当前任务状态:`，至少分为已完成、未完成、阻塞/风险、不可信/不可复用。
- 每条事实引用 auditor round（如 `round_003`）；没有证据就标为待审计。不要采用 executor 未审计的自述。

依赖要求:
- 在步骤更新之后简述 `依赖判断:`，引用前置证据并说明本轮路由理由。
- 只有 auditor 确认的前置才算满足。存在未满足前置时，本轮任务解决一个最关键前置，不能直奔最终交付。
- 遵守传入的轮次预算。只剩一轮时，不能把最后一轮用于一个明确推迟核心要求的纯前置任务；无法诚实完成时使用 `下一步: 请示用户` 或 `下一步: 阻塞`。

输出自然语言，不要 JSON。整份输出控制在 1200 个汉字以内，不复述完整历史或证据原文。
严格按以下顺序输出：首先恰好一个路由: `下一步: 执行任务`、`下一步: 仅审计`、`下一步: 最终验收`、`下一步: 请示用户` 或 `下一步: 阻塞`。

紧接路由输出该路由需要的字段（步骤、任务、验收标准等），然后输出：
`当前任务状态:`（只写当前进度、缺口和 round 引用，最多 6 条）
`任务契约:`（仅首次或确需修订时输出完整但简洁的契约；未变化时省略该小节，系统保留原契约）
`步骤更新:`（没有就写“无”）
`依赖判断:`（最多 3 句，不重复上述内容）

`步骤更新:` 只有两种写法，每行一条:
- 新增: <步骤标题> | 验收: <可检查的验收标准>
- 取代: <步骤 id> | 依据: <用户修订 id> | 原因: <一句话>

执行任务后包含 `步骤:`（步骤 id）、`任务:`、`验收标准:`（本轮子任务的）、`相关审计报告:`、`相关已审计状态:`、`边界:`；相关报告必须列出 round id 和原因。
仅审计后包含 `步骤:` 和 `核实重点:`。
最终验收后包含 `验收重点:`，列出你认为最容易遗漏的权威要求。
请示用户后包含 `问题:`，封闭式选择可包含用 `|` 分隔的 `选项:`。
阻塞时说明继续拆解也无法推进的原因。不要添加协议之外的顶层段落。"""


# 函数说明：format_step_list
# 用途：格式化步骤列表，供规划、执行、审计协作使用。
# 参数：
#   steps：任务步骤集合，类型 `Sequence[StepView]`。
# 返回：类型 `str`；按分支返回 `'(还没有步骤。)'`；`'\n'.join(lines)`。
# 分支与异常：
#   当 `not steps` 时，返回 `'(还没有步骤。)'`。
#   `step.status == 'superseded'` 分支在完成前置处理后跳过当前循环项。
def format_step_list(steps: Sequence[StepView]) -> str:
    if not steps:
        return "(还没有步骤。)"
    lines: list[str] = []
    for step in steps:
        if step.status == "superseded":
            lines.append(
                f"{step.id} [superseded 依据 {step.superseded_by or '?'}] {step.title}"
                f" — 原验收: {step.acceptance or '(无)'}"
                f" — 原因: {step.superseded_reason or '(无)'}"
            )
            continue
        ref = ""
        if step.status_ref:
            ref = f" {'依据' if step.status == 'done' else '见'} {step.status_ref}"
        lines.append(
            f"{step.id} [{step.status}{ref}] {step.title}"
            f" — 验收: {step.acceptance or '(无)'}"
            f" — 产生于 v{step.origin_revision}"
        )
    return "\n".join(lines)


# 函数说明：format_audit_history
# 用途：格式化历史，供规划、执行、审计协作使用。
# 参数：
#   rounds：已完成的模型或工具轮次，类型 `Sequence[RoundView]`。
#   each：传给 `clip_preserve` 的输入，类型 `int`；默认 `MANAGER_REPORT_EACH`。
#   total：传给 `_newest_first_budget` 的输入，类型 `int`；默认 `MANAGER_REPORTS_TOTAL`
# 。
# 返回：类型 `str`；返回 `_newest_first_budget(sections, total) or '(还没有审计报告。)'`
# 。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`round_ref` → `clip_preserve` →
# `_newest_first_budget`。
def format_audit_history(
    rounds: Sequence[RoundView],
    *,
    each: int = MANAGER_REPORT_EACH,
    total: int = MANAGER_REPORTS_TOTAL,
) -> str:
    sections = [
        "\n".join(
            (
                f"--- Round {item.index} auditor report（{_KIND_LABELS[item.kind]}）---",
                f"round_id: {round_ref(item.index)}",
                f"对应步骤: {item.step_id or '(整个任务)'}",
                "对应子任务:",
                clip_preserve(item.subtask.strip(), MANAGER_SUBTASK_EACH) or "(无)",
                "auditor 报告原文:",
                clip_preserve(item.auditor_report.strip(), each),
            )
        )
        for item in rounds
        if item.auditor_report.strip()
    ]
    return _newest_first_budget(sections, total) or "(还没有审计报告。)"


# 函数说明：format_harness_feedback
# 用途：格式化`harness_feedback`，供规划、执行、审计协作使用。
# 参数：
#   rounds：已完成的模型或工具轮次，类型 `Sequence[RoundView]`。
#   total：传给 `_newest_first_budget` 的输入，类型 `int`；默认 `MANAGER_FEEDBACK_TOTAL`
# 。
# 返回：类型 `str`；返回 `_newest_first_budget(sections, total) or '(无)'`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_newest_first_budget`。
def format_harness_feedback(
    rounds: Sequence[RoundView],
    *,
    total: int = MANAGER_FEEDBACK_TOTAL,
) -> str:
    sections = [
        f"--- Round {item.index} harness feedback ---\n{item.harness_feedback.strip()}"
        for item in rounds
        if item.harness_feedback.strip()
    ]
    return _newest_first_budget(sections, total) or "(无)"


# 函数说明：format_related_reports
# 用途：格式化`related_reports`，供规划、执行、审计协作使用。
# 参数：
#   rounds：已完成的模型或工具轮次，类型 `Sequence[RoundView]`。
#   refs：传给 `set` 的输入，类型 `Sequence[str]`。
#   total：传给 `_newest_first_budget` 的输入，类型 `int`；默认 `RELATED_REPORTS_TOTAL`
# 。
# 返回：类型 `str`；返回 `_newest_first_budget(sections, total)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`round_ref` → `_newest_first_budget`。
def format_related_reports(
    rounds: Sequence[RoundView],
    refs: Sequence[str],
    *,
    total: int = RELATED_REPORTS_TOTAL,
) -> str:
    wanted = set(refs)
    selected = [item for item in rounds if round_ref(item.index) in wanted]
    sections = [
        "\n".join(
            (
                f"--- Round {item.index} auditor report ---",
                f"round_id: {round_ref(item.index)}",
                "auditor 报告原文:",
                item.auditor_report.strip(),
            )
        )
        for item in selected
        if item.auditor_report.strip()
    ]
    return _newest_first_budget(sections, total)


# 函数说明：build_manager_prompt
# 用途：构建管理者，供规划、执行、审计协作使用。
# 参数：
#   requirements_text：文本输入或配置值，类型 `str`。
#   steps：任务步骤集合，类型 `Sequence[StepView]`。
#   contract：`contract`输入或配置值，类型 `str | None`。
#   state：当前状态快照，类型 `str | None`。
#   rounds：已完成的模型或工具轮次，类型 `Sequence[RoundView]`。
#   round_index：当前执行轮次索引，类型 `int`。
#   round_budget：预算输入或配置值，类型 `int`。
#   once_notes：`once_notes`输入或配置值，类型 `Sequence[str]`；默认 `()`。
# 返回：类型 `str`；返回 `'\n\n'.join(parts)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`format_step_list` → `clip_preserve` →
#  `format_audit_history` → `format_harness_feedback`。
def build_manager_prompt(
    *,
    requirements_text: str,
    steps: Sequence[StepView],
    contract: str | None,
    state: str | None,
    rounds: Sequence[RoundView],
    round_index: int,
    round_budget: int,
    once_notes: Sequence[str] = (),
) -> str:
    remaining = max(0, round_budget - round_index + 1)
    parts = [
        MANAGER_INSTRUCTIONS,
        requirements_text,
        "任务契约和最终状态规则:\n" + TASK_CONTRACT_RULES,
        USER_CLARIFICATION_NOTE,
        "必须遵守的最终状态约束:\n" + FINAL_STATE_SEMANTIC_GUARD,
        "步骤列表（状态由 harness 根据审计维护，你不能修改）:\n" + format_step_list(steps),
        "当前稳定任务契约:\n"
        + (
            clip_preserve(contract.strip(), CONTRACT_MAX)
            if contract and contract.strip()
            else "(还没有任务契约；请在本轮根据权威要求和步骤列表初始化。)"
        ),
        "上一轮当前任务状态:\n"
        + (
            clip_preserve(state.strip(), STATE_MAX)
            if state and state.strip()
            else "(还没有当前任务状态；请根据权威要求初始化。)"
        ),
        "历史 auditor 报告原文（按 round 编号；可信中间状态的权威来源）:\n"
        + format_audit_history(rounds),
        "harness 任务管理反馈（不是审计，只用于协议修正）:\n" + format_harness_feedback(rounds),
        "\n".join(
            (
                "轮次预算:",
                f"- 当前任务管理轮次: {round_index}",
                f"- 配置的轮次上限: {round_budget}",
                f"- 包含本轮在内的剩余轮次: {remaining}",
                "- 如果只剩一轮，不得安排一个明确推迟核心要求的纯前置子任务；无法完成时使用请示用户或阻塞。",
            )
        ),
        "只输出下一步任务管理结果。",
    ]
    notes = [note.strip() for note in once_notes if note and note.strip()]
    if notes:
        parts.append(
            "本轮一次性指令（用户通过界面注入，仅对本轮调度有效；持续有效的要求已在权威要求里）:\n"
            + "\n".join(f"- {note}" for note in notes)
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

EXECUTOR_INSTRUCTIONS = """你是 MuHarness 长任务的 executor，负责完成一个子任务。
- 主目标是 shell、文件、代码、测试、日志、数据处理或服务查询。命令在隔离的 Docker 沙箱里运行，默认禁网。
- 只完成分配给你的子任务，不要顺手处理其他步骤，也不要全局重规划。
- 不要尝试修改任务状态或声称步骤完成；是否完成由独立的 auditor 判定。
- 分配了发布步骤时，直接调用 artifact_publish（Host 工具，不是沙箱命令），
  文件使用 workspace 相对路径。保留真实返回的产物 ID、size_bytes、sha256；
  失败时如实报告，禁止伪造回执或在沙箱搜索同名入口。
- 恢复后若已有发布记录，用 artifact_list 核对，避免盲目重复发布；不能把记录说成原调用已返回。
- 不得伪造证据、模拟测试结果或用手写文件替代真实产物。
- 你不能与真人交互；需要用户输入时停止，并在报告里说明需要任务管理器使用 `下一步: 请示用户`。
- 上下文不足或子任务明显写错时，停止并报告，不要猜测。
- 报告真实运行的命令、文件修改（准确路径）、测试结果、产物路径和剩余问题。不要输出 JSON。"""


# 函数说明：build_executor_prompt
# 用途：构建执行者，供规划、执行、审计协作使用。
# 参数：
#   requirements_text：文本输入或配置值，类型 `str`。
#   contract：`contract`输入或配置值，类型 `str | None`。
#   state：当前状态快照，类型 `str | None`。
#   step：当前任务步骤，类型 `StepView`。
#   subtask：`subtask`输入或配置值，类型 `str`。
#   related_reports：`related_reports`输入或配置值，类型 `str`。
#   workspace_root：文件工具允许访问的工作区根目录，类型 `str`。
# 返回：类型 `str`；返回 `'\n\n'.join(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_or_none` → `clip_preserve`。
def build_executor_prompt(
    *,
    requirements_text: str,
    contract: str | None,
    state: str | None,
    step: StepView,
    subtask: str,
    related_reports: str,
    workspace_root: str,
) -> str:
    return "\n\n".join(
        (
            EXECUTOR_INSTRUCTIONS,
            requirements_text,
            "任务契约规则:\n" + TASK_CONTRACT_RULES,
            USER_CLARIFICATION_NOTE,
            "执行当前子任务必须遵守:\n" + FINAL_STATE_SEMANTIC_GUARD,
            "当前任务状态（由任务管理器维护，事实来自 auditor）:\n"
            + _or_none(clip_preserve((state or "").strip(), STATE_MAX)),
            "稳定任务契约:\n" + _or_none(clip_preserve((contract or "").strip(), CONTRACT_MAX)),
            "本子任务所属步骤（理解上下文用；本轮只需完成分配的子任务）:\n"
            f"{step.id} {step.title}\n步骤验收标准: {step.acceptance or '(无)'}",
            "\n".join(
                (
                    "真实任务状态和交付物:",
                    f"- Workspace 根目录: {workspace_root}",
                    "- 持久文件应写入该 Workspace，不要写入 MuHarness 自己的数据目录。",
                    "- 报告准确路径和可观察的状态，供 auditor 独立核验。你的文字报告只是运行记录，不是任务完成证明。",
                )
            ),
            "分配的子任务合同:\n" + subtask.strip(),
            "按 round id 选择的相关 auditor 报告:\n"
            + (related_reports.strip() or "(任务管理器没有引用相关报告。)"),
            "只完成当前子任务。权威要求里的明确约束和用户修订任何时候都不能违反，即使子任务合同没有复述它们。"
            "把已审计状态和稳定契约视为可信语义边界；不要重复已审计工作，不要使用 suspect/violation、伪造、"
            "不可信或已删除产物。上下文不足时停止并报告，不要猜测或全局重规划。",
        )
    )


# ---------------------------------------------------------------------------
# Auditor
# ---------------------------------------------------------------------------

AUDITOR_INSTRUCTIONS = """你是只读审计者，只检查本轮审计范围。不得修改、创建、删除任务文件。
核验用户要求和步骤验收标准；读取当前交付物，对关键行为取得独立工具证据。
发布验收应通过 evidence_search/evidence_read 读取 artifact_publish 的真实回执，
核对产物 ID、文件大小和 SHA-256；文件哈希不能替代发布成功的证据，不要在
沙箱查找发布命令，也不要自行发布。
若发布后证据归档中断，用只读 artifact_list(task_id=当前任务ID, run_id=原Executor运行ID)
核对已提交的发布记录，避免重复发布；该记录不能证明原调用已经返回。
若验收明确要求原 artifact_publish 返回值，而该返回值缺失，仍报告 unknown/blocked，不能弱化要求。
executor 自述不能代替证据。已有审计证据在对象未变化时可以引用；有变化或证据不足才补查。
不要新增用户没有要求的验收条件，不要为证明不可观察的历史反复运行命令。
Shell 是禁网 Docker 中的 /bin/sh；审计 workspace 只读。不要使用 Bash PIPESTATUS。
测试直接运行并保留退出码，不要用尾随 echo 掩盖失败。相关只读检查可合并成一次工具调用。
最终报告前四个非空行严格为：
状态: complete|incomplete|blocked
完整性: clean|suspect|violation
契约审计: aligned|unknown|needs_revision|invalid
步骤验收: satisfied|not_satisfied|not_applicable
随后只写：
验收项: 每项“要求 — verified/unknown/violated — 证据编号或路径”，不要复制日志全文。
阻断约束: 无，或尚未满足的实际要求。
给任务管理器的状态更新: 简述可信产物和必要下一步。
控制在 800 个汉字左右；禁止前言及重复背景。最终验收覆盖全部用户要求，步骤验收写 not_applicable。
证据不足就明确 unknown，不得为了收尾虚构通过。"""

AUDITOR_CONTRACT_BACKCHECK = """以原始用户要求为权威，契约只是解释。
步骤审计仅检查本步骤验收项和适用的禁止性约束；其他步骤未完成不导致本步失败。
最终验收检查所有用户要求。只有发现具体遗漏、弱化或矛盾时才解释契约问题，不展开通用风险清单。
有 blocking unknown/violated 时不得 complete 或 satisfied；契约未对齐时不得通过。
只有实际篡改、伪造或直接矛盾证据才标记 suspect/violation；报告长度或缺少历史日志本身不证明篡改。
完整性、契约和步骤结论必须与实际证据一致。"""


# 函数说明：build_auditor_prompt
# 用途：构建审计者，供规划、执行、审计协作使用。
# 参数：
#   kind：传给 `AuditKind` 的输入，类型 `AuditKind`。
#   requirements_text：文本输入或配置值，类型 `str`。
#   contract：`contract`输入或配置值，类型 `str | None`。
#   state：当前状态快照，类型 `str | None`。
#   task_id：目标任务标识，类型 `str`。
#   workspace_root：文件工具允许访问的工作区根目录，类型 `str`。
#   audit_revision：传给 `_audit_scope` 的输入，类型 `int`。
#   step：当前任务步骤，类型 `StepView | None`；默认 `None`。
#   other_steps：传给 `_audit_scope` 的输入，类型 `Sequence[StepView]`；默认 `()`。
#   superseded：传给 `_audit_scope` 的输入，类型 `Sequence[SupersededView]`；默认 `()`。
#   subtask：`subtask`输入或配置值，类型 `str`；默认 `''`。
#   executor_output：执行者输出输入或配置值，类型 `str`；默认 `''`。
#   focus：`focus`输入或配置值，类型 `str | None`；默认 `None`。
#   recovery_record：恢复信息记录输入或配置值，类型 `str`；默认 `''`。
#   related_reports：`related_reports`输入或配置值，类型 `str`；默认 `''`。
#   executor_run_id：执行者运行标识，类型 `str | None`；默认 `None`。
# 返回：类型 `str`；返回 `'\n\n'.join(parts)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`AuditKind` → `_or_none` →
# `clip_preserve` → `_audit_scope`。
# 分支与异常：
#   当 `kind is not AuditKind.FINAL_AUDIT and step is None` 时，抛出
# `ValueError(f'{kind.value} audit requires a step')`。
def build_auditor_prompt(
    *,
    kind: AuditKind,
    requirements_text: str,
    contract: str | None,
    state: str | None,
    task_id: str,
    workspace_root: str,
    audit_revision: int,
    step: StepView | None = None,
    other_steps: Sequence[StepView] = (),
    superseded: Sequence[SupersededView] = (),
    subtask: str = "",
    executor_output: str = "",
    focus: str | None = None,
    recovery_record: str = "",
    related_reports: str = "",
    executor_run_id: str | None = None,
) -> str:
    kind = AuditKind(kind)
    if kind is not AuditKind.FINAL_AUDIT and step is None:
        raise ValueError(f"{kind.value} audit requires a step")
    parts = [
        AUDITOR_INSTRUCTIONS,
        requirements_text,
        "必须遵守的最终状态约束:\n" + FINAL_STATE_SEMANTIC_GUARD,
        "当前任务状态（仅作背景）:\n" + _or_none(clip_preserve((state or "").strip(), STATE_MAX)),
        "稳定任务契约（其他角色对权威要求的解释，不能默认它正确）:\n"
        + _or_none(clip_preserve((contract or "").strip(), CONTRACT_MAX)),
        "审计范围:\n" + _audit_scope(kind, step, other_steps, superseded, audit_revision),
        "\n".join(
            (
                "独立证据边界:",
                f"- 真实 Workspace 根目录: {workspace_root}",
                "- executor 的工具原始输出已按任务归档，可用 "
                f'evidence_search(query="{task_id}", task_id="{task_id}", '
                'tool_name="artifact_publish") 查找发布回执。',
                "- 独立检查声明的文件、命令/测试/日志/服务。Executor 文本只是一项声明。",
                "- 历史报告只是运行记录，不是任务交付物或可单独成立的完成证据。若子任务的真实结果本来就是服务状态或面向用户的回答，不要机械要求每个子任务必须创建文件。",
            )
        ),
    ]
    if kind is AuditKind.NORMAL:
        parts.append("刚完成的子任务:\n" + (subtask.strip() or "(无)"))
        parts.append(
            "executor 自然语言输出:\n"
            + (clip_preserve(executor_output.strip(), EXECUTOR_OUTPUT_MAX) or "(executor 没有输出。)")
        )
    elif kind in (AuditKind.AUDIT_ONLY, AuditKind.FINAL_AUDIT):
        parts.append("任务管理器给出的核实重点:\n" + (focus.strip() if focus and focus.strip() else "(无)"))
    else:
        parts.append(recovery_record.strip() or "(没有中断记录。)")
    if executor_run_id:
        parts.append(
            "发布记录核对范围（只读，不重放原调用）：\n"
            f'artifact_list(task_id="{task_id}", run_id="{executor_run_id}")'
        )
    parts.append(
        "相关 auditor 报告（只作背景，不能代替当前只读审计）:\n"
        + (related_reports.strip() or "(无)")
    )
    parts.append(AUDITOR_CONTRACT_BACKCHECK)
    parts.append(
        "只审计上面 `审计范围:` 指定的对象是否真实完成、是否守住权威要求里的约束、是否可信。"
        "按上述简短报告格式输出，约 800 字；只列当前审计范围的验收项、证据和实际缺口。"
    )
    return "\n\n".join(parts)


# 函数说明：_audit_scope
# 用途：审计作用域，供规划、执行、审计协作使用。
# 参数：
#   kind：`kind`输入或配置值，类型 `AuditKind`。
#   step：当前任务步骤，类型 `StepView | None`。
#   other_steps：步骤集合输入或配置值，类型 `Sequence[StepView]`。
#   superseded：`superseded`输入或配置值，类型 `Sequence[SupersededView]`。
#   audit_revision：`audit_revision`输入或配置值，类型 `int`。
# 返回：类型 `str`；返回 `'\n'.join(lines)`。
# 分支与异常：
#   `kind is AuditKind.FINAL_AUDIT` 分支在完成前置处理后返回 `'\n'.join(lines)`。
def _audit_scope(
    kind: AuditKind,
    step: StepView | None,
    other_steps: Sequence[StepView],
    superseded: Sequence[SupersededView],
    audit_revision: int,
) -> str:
    if kind is AuditKind.FINAL_AUDIT:
        lines = [
            "- 最终验收。对照权威要求全文和全部交付物，判断整个任务是否完成。第 1 行 `状态:` 针对整个任务，第 4 行写 `步骤验收: not_applicable`。",
            "- 各步骤的审计结论只是线索，必须重新独立核验最终状态。",
        ]
        if superseded:
            lines.append(
                "- 被取代的步骤（逐条判断取代是否成立: 如果被取代步骤对应的要求在当前权威要求里仍然有效，属于 blocking violated）:"
            )
            lines.extend(
                f"  - {item.step_id} — 原验收: {item.acceptance or '(无)'} — 依据 {item.amendment_id}: "
                f"{item.amendment_text} — 原因: {item.reason or '(无)'}"
                for item in superseded
            )
        lines.append(f"- 本次审计依据要求版本 v{audit_revision}。")
        return "\n".join(lines)
    assert step is not None
    others = [s for s in other_steps if s.id != step.id and s.status not in ("done", "superseded")]
    other_text = "、".join(f"{s.id} {s.title}" for s in others) or "(无)"
    lines = [
        f"- {_KIND_LABELS[kind]}。所属步骤: {step.id} {step.title}",
        f"- 步骤验收标准（决定第 4 行 `步骤验收:`）: {step.acceptance or '(无)'}",
        "- 第 1 行 `状态:` 只针对下面的本轮子任务。"
        if kind is AuditKind.NORMAL
        else "- 本轮没有新的执行，第 1 行 `状态:` 针对所属步骤当前的真实状态。",
        f"- 其他步骤: {other_text}。它们的交付要求只能列入 `范围外约束:`。",
        f"- 本次审计依据要求版本 v{audit_revision}。",
    ]
    return "\n".join(lines)


AUDITOR_FORMAT_REPAIR = """上一份 auditor 报告缺少有效的前四行控制头。这只是格式修正，不是重新审计；不要使用工具或改变环境，只根据已有报告重发同一内容。

前四个非空行必须分别严格选择一个值:
状态: complete | 状态: incomplete | 状态: blocked
完整性: clean | 完整性: suspect | 完整性: violation
契约审计: aligned | 契约审计: unknown | 契约审计: needs_revision | 契约审计: invalid
步骤验收: satisfied | 步骤验收: not_satisfied | 步骤验收: not_applicable

只有报告的验收约束反查明确支持时才能使用 aligned；只有报告逐条确认了步骤验收标准时才能使用 satisfied。无法判断时保守输出:
状态: blocked
完整性: suspect
契约审计: unknown
步骤验收: not_satisfied"""


# 函数说明：build_format_repair_prompt
# 用途：构建`format_repair_prompt`，供规划、执行、审计协作使用。
# 参数：
#   report_text：报告文本输入或配置值，类型 `str`。
# 返回：类型 `str`；返回 `f'{AUDITOR_FORMAT_REPAIR}\n\n上一份 auditor 报告:\n{
# report_text.strip()}\n\n只输出修正后的 auditor 报告…`。
def build_format_repair_prompt(report_text: str) -> str:
    return (
        f"{AUDITOR_FORMAT_REPAIR}\n\n上一份 auditor 报告:\n{report_text.strip()}\n\n"
        "只输出修正后的 auditor 报告，不解释格式修正，不输出 JSON。"
    )


# ---------------------------------------------------------------------------
# harness 合成反馈（写成审计报告格式，Manager 把它当成审计信号处理）
# ---------------------------------------------------------------------------

_SUSPECT_HEADER = "状态: incomplete\n完整性: suspect\n契约审计: unknown\n步骤验收: not_applicable"


# 函数说明：invalid_final_audit_feedback
# 用途：pending: [(步骤 id, 说明)]，例如 ("s2", "最近 round_005: not_satisfied")。
# 参数：
#   pending：待处理项输入或配置值，类型 `Sequence[tuple[str, str]]`。
# 返回：类型 `str`；返回 `f'{_SUSPECT_HEADER}\n审计事实: 任务管理器请求最终验收，但以下
# 步骤尚未通过步骤验收: {listed}。\n缺口: 先为这些步骤安排执行任务或仅审计。\n下…`。
def invalid_final_audit_feedback(pending: Sequence[tuple[str, str]]) -> str:
    """pending: [(步骤 id, 说明)]，例如 ("s2", "最近 round_005: not_satisfied")。"""

    listed = "、".join(f"{step_id}（{note}）" for step_id, note in pending) or "(无)"
    return (
        f"{_SUSPECT_HEADER}\n"
        f"审计事实: 任务管理器请求最终验收，但以下步骤尚未通过步骤验收: {listed}。\n"
        "缺口: 先为这些步骤安排执行任务或仅审计。\n"
        "下一步: 重新任务管理；还有步骤不是 done 或 superseded 时不能输出 `下一步: 最终验收`。"
    )


# 函数说明：invalid_plan_feedback
# 用途：detail 接在“任务管理器输出”后面，例如“没有有效路由”。
# 参数：
#   detail：`detail`输入或配置值，类型 `str`。
# 返回：类型 `str`；返回 `f'{_SUSPECT_HEADER}\n审计事实: 任务管理器输出{detail}，本轮没
# 有执行任何子任务。\n缺口: 输出一个挂在未完成步骤上的子任务，或明确仅审计/最终验收…`。
def invalid_plan_feedback(detail: str) -> str:
    """detail 接在“任务管理器输出”后面，例如“没有有效路由”。"""

    return (
        f"{_SUSPECT_HEADER}\n"
        f"审计事实: 任务管理器输出{detail}，本轮没有执行任何子任务。\n"
        "缺口: 输出一个挂在未完成步骤上的子任务，或明确仅审计/最终验收/请示用户/阻塞。\n"
        "下一步: 使用 `下一步: 执行任务`、`下一步: 仅审计`、`下一步: 最终验收`、`下一步: 请示用户` "
        "或 `下一步: 阻塞` 重新管理。"
    )


# 函数说明：readonly_violation_report
# 用途：在规划、执行、审计协作中处理 `readonly_violation_report`，通过 `'、'.join` 完成
# 首个内部处理步骤。
# 参数：
#   changed_paths：路径集合输入或配置值，类型 `Sequence[str]`。
#   original_report：报告输入或配置值，类型 `str`。
# 返回：类型 `str`；返回 `f'状态: blocked\n完整性: violation\n契约审计: unknown\n步骤验
# 收: not_satisfied\n\nauditor 在只读审计期间改变了任务…`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`clip_preserve`。
def readonly_violation_report(changed_paths: Sequence[str], original_report: str) -> str:
    shown = "、".join(changed_paths[:20]) or "(未列出)"
    return (
        "状态: blocked\n完整性: violation\n契约审计: unknown\n步骤验收: not_satisfied\n\n"
        "auditor 在只读审计期间改变了任务 workspace；本次审计结论全部作废。\n"
        f"变化的路径: {shown}\n"
        "请安排 executor 修复或确认这些路径的状态，再重新审计。\n\n"
        f"以下原始报告仅供诊断:\n{clip_preserve(original_report.strip(), 2_000)}"
    )


# 函数说明：no_report_placeholder
# 用途：返回 `'状态: blocked\n完整性: suspect\n契约审计: unknown\n步骤验收:
# not_satisfied\n审计事实: auditor 没有产生可读取的自然语…`，提供 规划、执行、审计协作
# 的派生值。
# 返回：类型 `str`；返回 `'状态: blocked\n完整性: suspect\n契约审计: unknown\n步骤验收:
# not_satisfied\n审计事实: auditor 没有产生可读取的自然语…`。
def no_report_placeholder() -> str:
    return (
        "状态: blocked\n完整性: suspect\n契约审计: unknown\n步骤验收: not_satisfied\n"
        "审计事实: auditor 没有产生可读取的自然语言审计报告。\n"
        "下一步: 任务管理器应安排仅审计重试，或生成更小的子任务。"
    )


# 函数说明：invalid_header_report
# 用途：格式修复后控制头仍不合法时使用；harness 不根据正文猜测结论。
# 参数：
#   raw_report：报告输入或配置值，类型 `str`。
# 返回：类型 `str`；返回 `f'状态: blocked\n完整性: suspect\n契约审计: unknown\n步骤验收:
#  not_satisfied\n审计事实: auditor 报告缺少有效的前四行…`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`clip_preserve`。
def invalid_header_report(raw_report: str) -> str:
    """格式修复后控制头仍不合法时使用；harness 不根据正文猜测结论。"""

    return (
        "状态: blocked\n完整性: suspect\n契约审计: unknown\n步骤验收: not_satisfied\n"
        "审计事实: auditor 报告缺少有效的前四行控制头，harness 不会根据正文猜测完成状态、完整性、契约审计或步骤验收。\n"
        "缺口: auditor 必须以前四行明确写出状态、完整性、契约审计和步骤验收。\n\n"
        f"原始 auditor 输出摘录:\n{clip_preserve(raw_report.strip(), 1_800)}"
    )


# 函数说明：recovery_record
# 用途：记录恢复信息，供规划、执行、审计协作使用。
# 参数：
#   round_index：当前执行轮次索引，类型 `int`。
#   reason：状态变化、拒绝或降级原因，类型 `str`。
#   subtask：`subtask`输入或配置值，类型 `str`。
#   completed_calls：传给 `'\n'.join` 的输入，类型 `Sequence[str]`。
#   pending_calls：传给 `'\n'.join` 的输入，类型 `Sequence[str]`。
# 返回：类型 `str`；返回 `'\n'.join(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`round_ref` → `clip_preserve`。
def recovery_record(
    *,
    round_index: int,
    reason: str,
    subtask: str,
    completed_calls: Sequence[str],
    pending_calls: Sequence[str],
) -> str:
    completed = "\n".join(completed_calls) or "(没有)"
    pending = "\n".join(pending_calls) or "(没有)"
    return "\n".join(
        (
            "中断说明:",
            f"- {round_ref(round_index)} 的 executor 在运行中被打断（原因: {reason}）。它实际做了多少、是否留下半成品，目前未知。",
            "- 被打断的子任务:",
            subtask.strip() or "(无)",
            "- 中断前已返回结果的工具调用（来自证据库，按时间顺序；可用 evidence_read 查看全文）:",
            clip_preserve(completed, RECOVERY_CALLS_MAX),
            "- 已发出但没有结果的调用（来自检查点；它们可能执行了一部分、全部或完全没有执行）:",
            pending,
            "",
            "请核实当前 workspace 的真实状态: 哪些效果已经发生、是否有半成品或重复内容、重做该子任务是否安全。"
            "如果无法确定重做是否会造成重复或破坏，完整性写 suspect，并在给任务管理器的状态更新里说明需要用户确认什么。",
        )
    )


# 函数说明：boundary_rejection_note
# 用途：在规划、执行、审计协作中处理 `boundary_rejection_note`，通过 `'、'.join` 完成首
# 个内部处理步骤。
# 参数：
#   round_index：当前执行轮次索引，类型 `int`。
#   role：规划、执行、审计等运行角色，类型 `str`。
#   counts：`counts`输入或配置值，类型 `Mapping[str, int]`。
# 返回：类型 `str`；返回 `f'角色边界拦截: {round_ref(round_index)} 的 {role} 尝试调用未
# 授权工具 {listed}，已被拒绝，没有产生副作用。被拒达到 3 次时，该轮审…`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`round_ref`。
def boundary_rejection_note(round_index: int, role: str, counts: Mapping[str, int]) -> str:
    listed = "、".join(f"{name} × {count}" for name, count in sorted(counts.items()))
    return (
        f"角色边界拦截: {round_ref(round_index)} 的 {role} 尝试调用未授权工具 {listed}，已被拒绝，没有产生副作用。"
        "被拒达到 3 次时，该轮审计的完整性被限制为最高 suspect。"
    )


# 函数说明：supersede_rejected_feedback
# 用途：返回 `f'步骤更新被拒绝: 取代: {step_id} | 依据: {basis}。原因: {reason}。该步骤
# 保持原状态。如果确实需要放弃该步骤，请使用 `下一步: 请示用户`。'`，提供 规划、执行、审
# 计协作 的派生值。
# 参数：
#   step_id：目标步骤标识，类型 `str`。
#   basis：`basis`输入或配置值，类型 `str`。
#   reason：状态变化、拒绝或降级原因，类型 `str`。
# 返回：类型 `str`；返回 `f'步骤更新被拒绝: 取代: {step_id} | 依据: {basis}。原因: {
# reason}。该步骤保持原状态。如果确实需要放弃该步骤，请使用 `下一步: 请示用户`。'`。
def supersede_rejected_feedback(step_id: str, basis: str, reason: str) -> str:
    return (
        f"步骤更新被拒绝: 取代: {step_id} | 依据: {basis}。原因: {reason}。该步骤保持原状态。"
        "如果确实需要放弃该步骤，请使用 `下一步: 请示用户`。"
    )


# 函数说明：requirements_updated_feedback
# 用途：在规划、执行、审计协作中处理 `requirements_updated_feedback`，通过 `'；'.join`
# 完成首个内部处理步骤。
# 参数：
#   round_index：当前执行轮次索引，类型 `int`。
#   old_revision：`old_revision`输入或配置值，类型 `int`。
#   new_revision：`new_revision`输入或配置值，类型 `int`。
#   new_amendments：`new_amendments`输入或配置值，类型 `Sequence[tuple[str, str]]`。
#   step_ids：传给 `'、'.join` 的输入，类型 `Sequence[str]`。
# 返回：类型 `str`；返回 `f"要求已更新: {round_ref(round_index)} 的审计依据 v{
# old_revision}，期间权威要求已更新到 v{new_revision}（新增 {…`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`round_ref`。
def requirements_updated_feedback(
    *,
    round_index: int,
    old_revision: int,
    new_revision: int,
    new_amendments: Sequence[tuple[str, str]],
    step_ids: Sequence[str],
) -> str:
    added = "；".join(f"{amendment_id}: {text}" for amendment_id, text in new_amendments)
    steps = "、".join(step_ids) or "(无)"
    return (
        f"要求已更新: {round_ref(round_index)} 的审计依据 v{old_revision}，期间权威要求已更新到 v{new_revision}"
        f"（新增 {added or '(无)'}）。本轮结论只作历史，不用于推进状态；步骤 {steps} 保持未完成。请按新要求重新安排。"
    )


# 函数说明：round_abandoned_feedback
# 用途：reason 例如“权威要求已更新到 v3（新增 A3: …）”或“Task 在本轮期间被修改”。
# 参数：
#   round_index：当前执行轮次索引，类型 `int`。
#   plan_applied：计划输入或配置值，类型 `bool`。
#   reason：状态变化、拒绝或降级原因，类型 `str`。
#   added_step_ids：传给 `'、'.join` 的输入，类型 `Sequence[str]`；默认 `()`。
#   amendment_ids：传给 `'、'.join` 的输入，类型 `Sequence[str]`；默认 `()`。
# 返回：类型 `str`；返回 `'\n'.join(lines)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`round_ref`。
def round_abandoned_feedback(
    *,
    round_index: int,
    plan_applied: bool,
    reason: str,
    added_step_ids: Sequence[str] = (),
    amendment_ids: Sequence[str] = (),
) -> str:
    """reason 例如“权威要求已更新到 v3（新增 A3: …）”或“Task 在本轮期间被修改”。"""

    state = "已应用但 Executor 未启动" if plan_applied else "尚未应用"
    lines = [
        f"本轮作废: {round_ref(round_index)} 的计划{state}，原因: {reason}。本轮没有执行任何子任务。"
    ]
    if plan_applied:
        steps = "、".join(added_step_ids) or "(无)"
        basis = "、".join(amendment_ids) or "对应的修订"
        lines.append(
            f"{round_ref(round_index)} 新增的步骤 {steps} 和契约修改仍在步骤列表和当前契约里；"
            f"如与新要求冲突，用 `取代:` 引用 {basis} 处理，并在契约里改正。"
        )
    lines.append("请按当前权威要求重新规划。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 最终回复
# ---------------------------------------------------------------------------

FINAL_RESPONSE_INSTRUCTIONS = """写给提出这个任务的人的回复。你是唯一直接和他对话的角色，其它角色都是写给下一个角色看的。
- 直接回答任务本身。是问题就给答案，而不是找答案的过程；是改动就说明现在的状态。
- 只能用下面的已验证状态和审计结论。不要用工具，不要检查环境，不要补充任何没被确认过的结论。
- 把权威要求里的用户后续修订视为对原始任务的权威修订。如果修订要求尚未验证或尚未完成的状态，应如实说明，不能编造完成。
- 如实说明：哪些没做到、被阻塞或未验证，以及原因。绝不能把没完成的写成完成。
- 用自然语言写给完全没看过执行过程的人。不要 JSON，不要控制头，不要轮次引用，不要协议小节名。
- 先给结论，再只写读者需要的：改了什么、在哪里、还剩什么。没内容的部分直接省略。如果已验收的执行器交付正文直接回答原任务，必须保留用户要求的全部来源、数字和实质性章节，不能用删减版摘要替代。"""


# 函数说明：format_audit_findings
# 用途：每轮去掉控制头和反查协议，1,200 字符、合计 6,000；最终验收那一轮完整保留且不计入
# 合计。
# 参数：
#   rounds：已完成的模型或工具轮次，类型 `Sequence[RoundView]`。
#   final_round_index：索引输入或配置值，类型 `int | None`；默认 `None`。
# 返回：类型 `str`；返回
# `'\n\n'.join((part for part in (kept, final_section) if part)) or '(没有审计结论。)'`
# 。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`audit_findings_body` →
# `clip_preserve` → `_newest_first_budget`。
# 分支与异常：
#   当 `not item.auditor_report.strip()` 时，跳过当前循环项。
def format_audit_findings(
    rounds: Sequence[RoundView],
    *,
    final_round_index: int | None = None,
) -> str:
    """每轮去掉控制头和反查协议，1,200 字符、合计 6,000；最终验收那一轮完整保留且不计入合计。"""

    final_section: str | None = None
    others: list[str] = []
    for item in rounds:
        if not item.auditor_report.strip():
            continue
        label = f"第 {item.index} 轮（{_KIND_LABELS[item.kind]}）"
        body = audit_findings_body(item.auditor_report)
        if item.index == final_round_index:
            final_section = f"{label}\n{body or '(无正文)'}"
        else:
            others.append(f"{label}\n{clip_preserve(body, FINAL_FINDINGS_EACH) or '(无正文)'}")
    kept = _newest_first_budget(others, FINAL_FINDINGS_TOTAL)
    return "\n\n".join(part for part in (kept, final_section) if part) or "(没有审计结论。)"


# 函数说明：build_final_response_prompt
# 用途：构建响应，供规划、执行、审计协作使用。
# 参数：
#   requirements_text：文本输入或配置值，类型 `str`。
#   outcome：执行或验收结果，类型 `str`。
#   abort_reason：原因输入或配置值，类型 `str | None`。
#   state：当前状态快照，类型 `str | None`。
#   findings：`findings`输入或配置值，类型 `str`。
#   deliverables：`deliverables`输入或配置值，类型 `Sequence[str]`；默认 `()`。
# 返回：类型 `str`；返回 `'\n\n'.join(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_or_none` → `clip_preserve`。
def build_final_response_prompt(
    *,
    requirements_text: str,
    outcome: str,  # complete | blocked | incomplete
    abort_reason: str | None,
    state: str | None,
    findings: str,
    deliverables: Sequence[str] = (),
) -> str:
    delivered = "\n\n".join(item.strip() for item in deliverables if item and item.strip())
    return "\n\n".join(
        (
            FINAL_RESPONSE_INSTRUCTIONS,
            requirements_text,
            f"运行结果: {outcome}（结束原因: {abort_reason or '无'}）",
            "已验证状态:\n" + _or_none((state or "").strip()),
            "审计结论:\n" + findings.strip(),
            "已通过验收的执行器交付正文:\n"
            + (
                clip_preserve(delivered, FINAL_DELIVERABLES_TOTAL)
                if outcome == "complete" and delivered
                else "(没有独立的交付正文。)"
            ),
            "现在写这份回复。只输出回复本身。",
        )
    )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


# 函数说明：round_ref
# 用途：返回 `f'round_{index:03d}'`，提供 规划、执行、审计协作 的派生值。
# 参数：
#   index：当前位置或索引，类型 `int`。
# 返回：类型 `str`；返回 `f'round_{index:03d}'`。
def round_ref(index: int) -> str:
    return f"round_{index:03d}"


# 函数说明：_or_none
# 用途：返回 `text if text.strip() else '(无)'`，提供 规划、执行、审计协作 的派生值。
# 参数：
#   text：待处理的文本，类型 `str`。
# 返回：类型 `str`；返回 `text if text.strip() else '(无)'`。
def _or_none(text: str) -> str:
    return text if text.strip() else "(无)"


# 函数说明：_newest_first_budget
# 用途：新的完整保留，旧的先被截；超出预算的最旧部分整体省略并注明。
# 参数：
#   sections：传给 `reversed` 的输入，类型 `Sequence[str]`。
#   total：`total`输入或配置值，类型 `int`。
# 返回：类型 `str`；返回 `'\n\n'.join(kept)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`clip_preserve` → `kept.reverse` →
# `kept.insert`。
# 分支与异常：
#   `remaining < 400` 分支在完成前置处理后跳过当前循环项。
def _newest_first_budget(sections: Sequence[str], total: int) -> str:
    """新的完整保留，旧的先被截；超出预算的最旧部分整体省略并注明。"""

    kept: list[str] = []
    remaining = total
    dropped = 0
    for section in reversed(sections):
        if remaining < 400:  # 剩下的空间太小，截出来也没法读，整段省略
            dropped += 1
            remaining = 0
            continue
        if len(section) <= remaining:
            kept.append(section)
            remaining -= len(section) + 2
        else:
            kept.append(clip_preserve(section, remaining))
            remaining = 0
    kept.reverse()
    if dropped:
        kept.insert(0, f"...[更早的 {dropped} 段因长度上限省略]...")
    return "\n\n".join(kept)


__all__ = [
    "AUDITOR_CONTRACT_BACKCHECK",
    "AUDITOR_FORMAT_REPAIR",
    "AUDITOR_INSTRUCTIONS",
    "EXECUTOR_INSTRUCTIONS",
    "FINAL_RESPONSE_INSTRUCTIONS",
    "FINAL_STATE_SEMANTIC_GUARD",
    "MANAGER_INSTRUCTIONS",
    "TASK_CONTRACT_RULES",
    "USER_CLARIFICATION_NOTE",
    "AuditKind",
    "RoundView",
    "StepView",
    "SupersededView",
    "boundary_rejection_note",
    "build_auditor_prompt",
    "build_executor_prompt",
    "build_final_response_prompt",
    "build_format_repair_prompt",
    "build_manager_prompt",
    "format_audit_findings",
    "format_audit_history",
    "format_harness_feedback",
    "format_related_reports",
    "format_step_list",
    "invalid_final_audit_feedback",
    "invalid_header_report",
    "invalid_plan_feedback",
    "no_report_placeholder",
    "readonly_violation_report",
    "recovery_record",
    "requirements_updated_feedback",
    "round_abandoned_feedback",
    "round_ref",
    "supersede_rejected_feedback",
]
