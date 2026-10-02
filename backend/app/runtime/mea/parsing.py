
"""解析 Manager 和 Auditor 的自然语言输出（移植 LHH role_prompts.py、auditor_agent.py）。

两个角色都不输出 JSON：Manager 用固定小节 + 一行 ``下一步:`` 路由，Auditor 用前四行
控制头。这里只做文本解析，是否合法（步骤是否存在、取代依据是否成立）由 runner 判断。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

# ---------------------------------------------------------------------------
# Manager 路由
# ---------------------------------------------------------------------------


class Route(StrEnum):

    EXECUTE = "execute"
    AUDIT_ONLY = "audit_only"
    FINAL_AUDIT = "final_audit"
    ASK = "ask"
    BLOCKED = "blocked"
    INVALID = "invalid"


_ROUTE_WORDS: dict[str, Route] = {
    "执行任务": Route.EXECUTE,
    "仅审计": Route.AUDIT_ONLY,
    "最终验收": Route.FINAL_AUDIT,
    "请示用户": Route.ASK,
    "请示": Route.ASK,
    "询问用户": Route.ASK,
    "阻塞": Route.BLOCKED,
}
# 路由后允许跟一段明确分隔的注释，例如 `下一步: 执行任务 — 先做 s2`。
# 不带分隔符的附加文字（`下一步: 执行任务吧`）仍然无效。
_ROUTE_COMMENT_RE = re.compile(r"(?:—|–|--|//|#|[（(])")
_LINE_MARKERS = "*`#>-+ \t　"


# 函数说明：parse_route
# 用途：返回最后一个合法的 ``下一步:`` 路由；没有则 INVALID。
# 参数：
#   text：待处理的文本，类型 `str`。
# 返回：类型 `Route`；返回 `found[0] if found else Route.INVALID`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_last_route_line`。
def parse_route(text: str) -> Route:
    """返回最后一个合法的 ``下一步:`` 路由；没有则 INVALID。"""

    found = _last_route_line(text)
    return found[0] if found else Route.INVALID


# 函数说明：_last_route_line
# 用途：返回 (路由, 该行在 text 中的起始偏移)。
# 参数：
#   text：待处理的文本，类型 `str`。
# 返回：类型 `tuple[Route, int] | None`；返回 `result`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`str(text or '').splitlines` →
# `_route_of_line`。
def _last_route_line(text: str) -> tuple[Route, int] | None:
    """返回 (路由, 该行在 text 中的起始偏移)。"""

    result: tuple[Route, int] | None = None
    offset = 0
    for line in str(text or "").splitlines(keepends=True):
        route = _route_of_line(line)
        if route is not None:
            result = (route, offset)
        offset += len(line)
    return result


# 函数说明：_route_of_line
# 用途：在规划、执行、审计协作中处理 `_route_of_line`，通过 `line.strip().lstrip` 完成首
# 个内部处理步骤。
# 参数：
#   line：`line`输入或配置值，类型 `str`。
# 返回：类型 `Route | None`；按分支返回 `None`；`_ROUTE_WORDS.get(value)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`line.strip().lstrip` →
# `normalized.replace(' ', '').replace('\u3000', '').replace` →
# `normalized.replace(' ', '').replace` → `normalized.replace` → `normalized.startswith`
# 。
# 分支与异常：
#   `normalized.startswith(prefix)` 分支在完成前置处理后结束当前循环。
def _route_of_line(line: str) -> Route | None:
    normalized = line.strip().lstrip(_LINE_MARKERS)
    normalized = normalized.replace(" ", "").replace("　", "").replace("\t", "")
    for prefix in ("下一步:", "下一步："):
        if normalized.startswith(prefix):
            value = normalized[len(prefix) :]
            break
    else:
        return None
    value = _ROUTE_COMMENT_RE.split(value, maxsplit=1)[0]
    value = value.strip("*`。.")
    return _ROUTE_WORDS.get(value)


# ---------------------------------------------------------------------------
# Manager 小节
# ---------------------------------------------------------------------------

_BEFORE_ROUTE_HEADERS = ("当前任务状态", "任务契约", "步骤更新", "依赖判断")
_AFTER_ROUTE_HEADERS = (
    "步骤",
    "任务",
    "验收标准",
    "相关审计报告",
    "相关已审计状态",
    "边界",
    "核实重点",
    "验收重点",
    "问题",
    "选项",
)


# 函数说明：_header_re
# 用途：在规划、执行、审计协作中处理 `_header_re`，通过 `'|'.join` 完成首个内部处理步骤
# 。
# 参数：
#   names：传给 `sorted` 的输入，类型 `tuple[str, ...]`。
# 返回：类型 `re.Pattern[str]`；返回 `re.compile(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`re.escape` → `re.compile`。
def _header_re(names: tuple[str, ...]) -> re.Pattern[str]:
    alternation = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
    return re.compile(
        rf"^[ \t]*(?:\*\*)?[ \t]*(?P<name>{alternation})[ \t]*(?:\*\*)?[ \t]*[:：][ \t]*(?:\*\*)?(?P<rest>.*)$",
        re.MULTILINE,
    )


_BEFORE_ROUTE_RE = _header_re(_BEFORE_ROUTE_HEADERS)
_AFTER_ROUTE_RE = _header_re(_AFTER_ROUTE_HEADERS)
_ROUND_REF_RE = re.compile(r"(?i)\bround_(\d+)\b")


@dataclass(frozen=True, slots=True)
class NewStep:

    title: str
    acceptance: str


@dataclass(frozen=True, slots=True)
class Supersede:

    step_id: str
    basis: str  # 用户修订 id，例如 A2
    reason: str


@dataclass(frozen=True, slots=True)
class StepUpdates:

    added: tuple[NewStep, ...] = ()
    superseded: tuple[Supersede, ...] = ()
    invalid_lines: tuple[str, ...] = ()

    # 函数说明：StepUpdates.empty
    # 用途：返回 `not (self.added or self.superseded or self.invalid_lines)`，提供
    # StepUpdates 的派生值。
    # 返回：类型 `bool`；返回
    # `not (self.added or self.superseded or self.invalid_lines)`。
    @property
    def empty(self) -> bool:
        return not (self.added or self.superseded or self.invalid_lines)


@dataclass(frozen=True, slots=True)
class ManagerOutput:

    route: Route
    plan_text: str  # 从 `当前任务状态:` 到结尾，存进 round.plan_text
    subtask: str  # 从最后一个路由行到结尾，原样交给 executor / auditor
    state: str | None
    contract: str | None
    step_updates: StepUpdates
    step_id: str | None
    focus: str | None  # 仅审计的 `核实重点:` 或最终验收的 `验收重点:`
    related_refs: tuple[str, ...] = ()
    question: str | None = None
    choices: tuple[str, ...] = field(default_factory=tuple)


# 函数说明：parse_manager_output
# 用途：解析管理者输出，供规划、执行、审计协作使用。
# 参数：
#   text：待处理的文本，类型 `str`。
# 返回：类型 `ManagerOutput`；返回 `ManagerOutput(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`str(text or '').replace` →
# `_last_route_line` → `_BEFORE_ROUTE_RE.search` → `metadata.start` → `_sections` →
# `m.start`；另有 6 个调用点。
def parse_manager_output(text: str) -> ManagerOutput:
    raw = str(text or "").replace("\r\n", "\n").strip()
    found = _last_route_line(raw)
    if found is None:
        route, route_at = Route.INVALID, len(raw)
    else:
        route, route_at = found
    head, tail = raw[:route_at], raw[route_at:]
    # Accept route-first output while retaining compatibility with saved legacy plans.
    if found and route_at == 0:
        metadata = _BEFORE_ROUTE_RE.search(tail)
        if metadata:
            head, tail = tail[metadata.start():], tail[:metadata.start()]
    before = _sections(head, _BEFORE_ROUTE_RE)
    after = _sections(tail, _AFTER_ROUTE_RE) if found else {}

    starts = [m.start() for m in _BEFORE_ROUTE_RE.finditer(raw)]
    plan_start = min(starts) if starts and route_at != 0 else 0
    step_value = after.get("步骤")
    focus = after.get("核实重点") if route is Route.AUDIT_ONLY else None
    if route is Route.FINAL_AUDIT:
        focus = after.get("验收重点")
    question = after.get("问题") if route is Route.ASK else None
    return ManagerOutput(
        route=route,
        plan_text=raw[plan_start:],
        subtask=tail.strip(),
        state=before.get("当前任务状态"),
        contract=before.get("任务契约"),
        step_updates=parse_step_updates(before.get("步骤更新")),
        step_id=_first_token(step_value) if step_value else None,
        focus=focus,
        related_refs=_round_refs(
            "\n".join(
                value
                for key in ("相关审计报告", "相关已审计状态")
                if (value := after.get(key))
            )
        ),
        question=question,
        choices=_choices(after.get("选项"), question) if route is Route.ASK else (),
    )


# 函数说明：_sections
# 用途：按已知小节头切分；同名小节出现多次时取最后一次。
# 参数：
#   text：待处理的文本，类型 `str`。
#   header_re：`header_re`输入或配置值，类型 `re.Pattern[str]`。
# 返回：类型 `dict[str, str]`；返回 `sections`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`header_re.finditer` →
# `matches[index + 1].start` → `match.group` → `match.end` → `body.rstrip`。
def _sections(text: str, header_re: re.Pattern[str]) -> dict[str, str]:
    """按已知小节头切分；同名小节出现多次时取最后一次。"""

    matches = list(header_re.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        rest = match.group("rest").strip().strip("*").strip()
        body = text[match.end() : end].strip("\n")
        value = "\n".join(part for part in (rest, body.rstrip()) if part.strip()).strip()
        sections[match.group("name")] = value
    return sections


_NEW_STEP_RE = re.compile(
    r"^新增\s*[:：]\s*(?P<title>.+?)\s*[|｜]\s*验收\s*[:：]\s*(?P<acceptance>.+)$"
)
_SUPERSEDE_RE = re.compile(
    r"^取代\s*[:：]\s*(?P<step>[^|｜\s]+)\s*[|｜]\s*依据\s*[:：]\s*(?P<basis>[^|｜\s]+)"
    r"\s*[|｜]\s*原因\s*[:：]\s*(?P<reason>.+)$"
)
_NONE_RE = re.compile(r"(?i)^(?:无|没有|暂无|none|n/?a)[。.]?$")


# 函数说明：parse_step_updates
# 用途：解析步骤，供规划、执行、审计协作使用。
# 参数：
#   section：`section`输入或配置值，类型 `str | None`。
# 返回：类型 `StepUpdates`；按分支返回 `StepUpdates()`；
# `StepUpdates(tuple(added), tuple(superseded), tuple(invalid))`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`StepUpdates` → `section.splitlines` →
#  `re.sub` → `_NONE_RE.match` → `_NEW_STEP_RE.match` → `NewStep`；另有 3 个调用点。
# 分支与异常：
#   当 `not section` 时，返回 `StepUpdates()`。
#   当 `not line or _NONE_RE.match(line)` 时，跳过当前循环项。
def parse_step_updates(section: str | None) -> StepUpdates:
    if not section:
        return StepUpdates()
    added: list[NewStep] = []
    superseded: list[Supersede] = []
    invalid: list[str] = []
    for raw_line in section.splitlines():
        line = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", raw_line).strip().strip("`")
        if not line or _NONE_RE.match(line):
            continue
        if match := _NEW_STEP_RE.match(line):
            added.append(NewStep(match["title"].strip(), match["acceptance"].strip()))
        elif match := _SUPERSEDE_RE.match(line):
            superseded.append(
                Supersede(
                    step_id=match["step"].strip(),
                    basis=match["basis"].strip().upper(),
                    reason=match["reason"].strip(),
                )
            )
        else:
            invalid.append(raw_line.strip())
    return StepUpdates(tuple(added), tuple(superseded), tuple(invalid))


# 函数说明：_first_token
# 用途：在规划、执行、审计协作中处理 `_first_token`，通过 `re.match` 完成首个内部处理步
# 骤。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str`。
# 返回：类型 `str | None`；返回 `match.group(1) if match else None`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`re.match` → `match.group`。
def _first_token(value: str) -> str | None:
    match = re.match(r"\s*`?([^\s`，,。;；（(]+)", value)
    return match.group(1) if match else None


# 函数说明：_round_refs
# 用途：在规划、执行、审计协作中处理 `_round_refs`，通过 `_ROUND_REF_RE.finditer` 完成首
# 个内部处理步骤。
# 参数：
#   text：待处理的文本，类型 `str`。
# 返回：类型 `tuple[str, ...]`；返回 `tuple(refs)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_ROUND_REF_RE.finditer` →
# `match.group`。
def _round_refs(text: str) -> tuple[str, ...]:
    refs: list[str] = []
    for match in _ROUND_REF_RE.finditer(text):
        ref = f"round_{int(match.group(1)):03d}"
        if int(match.group(1)) > 0 and ref not in refs:
            refs.append(ref)
    return tuple(refs)


# 函数说明：_choices
# 用途：在规划、执行、审计协作中处理 `_choices`，通过 `raw.splitlines` 完成首个内部处理
# 步骤。
# 参数：
#   raw：`raw`输入或配置值，类型 `str | None`。
#   question：`question`输入或配置值，类型 `str | None`。
# 返回：类型 `tuple[str, ...]`；按分支返回 `choices[:6]`；`('是', '否')`；`()`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`raw.splitlines`。
# 分支与异常：
#   当 `choices` 时，返回 `choices[:6]`。
#   当 `question and ('是否' in question or ('是' in question and '否'…` 时，返回
# `('是', '否')`。
def _choices(raw: str | None, question: str | None) -> tuple[str, ...]:
    if raw:
        first_line = raw.splitlines()[0]
        parts = [part.strip() for part in re.split(r"\s*[|、,，/]\s*", first_line)]
        choices = tuple(part for part in parts if part)
        if choices:
            return choices[:6]
    if question and ("是否" in question or ("是" in question and "否" in question)):
        return ("是", "否")
    return ()


# 函数说明：_line_start
# 用途：启动`line`，供规划、执行、审计协作使用。
# 参数：
#   text：待处理的文本，类型 `str`。
#   position：传给 `text.rfind` 的输入，类型 `int`。
# 返回：类型 `int`；返回 `text.rfind('\n', 0, position) + 1`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`text.rfind`。
def _line_start(text: str, position: int) -> int:
    return text.rfind("\n", 0, position) + 1


# ---------------------------------------------------------------------------
# Auditor 控制头
# ---------------------------------------------------------------------------


class AuditStatus(StrEnum):

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    BLOCKED = "blocked"


class Integrity(StrEnum):

    CLEAN = "clean"
    SUSPECT = "suspect"
    VIOLATION = "violation"


class ContractAudit(StrEnum):

    ALIGNED = "aligned"
    UNKNOWN = "unknown"
    NEEDS_REVISION = "needs_revision"
    INVALID = "invalid"


class StepAcceptance(StrEnum):

    SATISFIED = "satisfied"
    NOT_SATISFIED = "not_satisfied"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class ControlHeader:

    status: AuditStatus
    integrity: Integrity
    contract_audit: ContractAudit
    step_acceptance: StepAcceptance

    # 函数说明：ControlHeader.render
    # 用途：生成展示文本ControlHeader，供规划、执行、审计协作使用。
    # 返回：类型 `str`；返回 `format_control_header(self)`。
    # 关键调用（按源码出现顺序，实际执行取决于分支）：`format_control_header`。
    def render(self) -> str:
        return format_control_header(self)


# 函数说明：_control_re
# 用途：返回 `re.compile(…)`，提供 规划、执行、审计协作 的派生值。
# 参数：
#   keys：`keys`输入或配置值，类型 `str`。
#   values：待处理的值集合，类型 `str`。
# 返回：类型 `re.Pattern[str]`；返回 `re.compile(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`re.compile`。
def _control_re(keys: str, values: str) -> re.Pattern[str]:
    return re.compile(
        rf"^\s*(?:\*\*)?\s*(?:{keys})\s*(?:\*\*)?\s*[:：]\s*(?:\*\*)?\s*({values})\s*(?:\*\*)?\s*$",
        re.IGNORECASE,
    )


_STATUS_RE = _control_re(r"状态|status", r"complete|incomplete|blocked|完成|未完成|阻塞")
_INTEGRITY_RE = _control_re(r"完整性|integrity", r"clean|suspect|violation")
_CONTRACT_RE = _control_re(
    r"契约审计|contract(?:[_\s-]*audit)?",
    r"aligned|unknown|needs[_\s-]*revision|invalid|对齐|未知|需修订|需要修订|无效",
)
_STEP_RE = _control_re(
    r"步骤验收|step[_\s-]*acceptance",
    r"satisfied|not[_\s-]*satisfied|not[_\s-]*applicable|已满足|满足|未满足|不适用",
)

_STATUS_VALUES = {"完成": "complete", "未完成": "incomplete", "阻塞": "blocked"}
_CONTRACT_VALUES = {"对齐": "aligned", "未知": "unknown", "需修订": "needs_revision",
                    "需要修订": "needs_revision", "无效": "invalid"}
_STEP_VALUES = {"满足": "satisfied", "已满足": "satisfied", "未满足": "not_satisfied",
                "不适用": "not_applicable"}


# 函数说明：parse_control_header
# 用途：前四个非空行必须依次是状态、完整性、契约审计、步骤验收；任一不合法返回 None。
# 参数：
#   text：待处理的文本，类型 `str`。
# 返回：类型 `ControlHeader | None`；按分支返回 `None`；`ControlHeader(…)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_first_nonempty_lines` →
# `regex.match` → `line.replace` → `_normalize_value` → `m.group` → `ControlHeader`；另
# 有 4 个调用点。
# 分支与异常：
#   当 `len(lines) < 4` 时，返回 `None`。
#   当 `not all(matches)` 时，返回 `None`。
def parse_control_header(text: str) -> ControlHeader | None:
    """前四个非空行必须依次是状态、完整性、契约审计、步骤验收；任一不合法返回 None。"""

    lines = _first_nonempty_lines(text, 4)
    if len(lines) < 4:
        return None
    regexes = (_STATUS_RE, _INTEGRITY_RE, _CONTRACT_RE, _STEP_RE)
    matches = [
        regex.match(line.replace("`", ""))
        for regex, line in zip(regexes, lines, strict=True)
    ]
    if not all(matches):
        return None
    status, integrity, contract, step = (_normalize_value(m.group(1)) for m in matches if m)
    return ControlHeader(
        status=AuditStatus(_STATUS_VALUES.get(status, status)),
        integrity=Integrity(integrity),
        contract_audit=ContractAudit(_CONTRACT_VALUES.get(contract, contract)),
        step_acceptance=StepAcceptance(_STEP_VALUES.get(step, step)),
    )


# 函数说明：format_control_header
# 用途：格式化`control_header`，供规划、执行、审计协作使用。
# 参数：
#   header：`header`输入或配置值，类型 `ControlHeader`。
# 返回：类型 `str`；返回 `'\n'.join(…)`。
def format_control_header(header: ControlHeader) -> str:
    return "\n".join(
        (
            f"状态: {header.status.value}",
            f"完整性: {header.integrity.value}",
            f"契约审计: {header.contract_audit.value}",
            f"步骤验收: {header.step_acceptance.value}",
        )
    )


# 函数说明：strip_control_header
# 用途：去掉前四个非空行（控制头），返回其余正文。
# 参数：
#   text：待处理的文本，类型 `str`。
# 返回：类型 `str`；按分支返回 `'\n'.join(lines[index + 1:]).strip()`；`''`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`str(text or '').splitlines`。
# 分支与异常：
#   当 `not line.strip()` 时，跳过当前循环项。
#   当 `remaining == 0` 时，返回 `'\n'.join(lines[index + 1:]).strip()`。
def strip_control_header(text: str) -> str:
    """去掉前四个非空行（控制头），返回其余正文。"""

    remaining = 4
    lines = str(text or "").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        remaining -= 1
        if remaining == 0:
            return "\n".join(lines[index + 1 :]).strip()
    return ""


# 函数说明：_normalize_value
# 用途：规范化`value`，供规划、执行、审计协作使用。
# 参数：
#   value：待校验、规范化或转换的值，类型 `str`。
# 返回：类型 `str`；返回 `re.sub('[\\s-]+', '_', lowered)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`value.strip().lower` → `re.sub`。
def _normalize_value(value: str) -> str:
    lowered = value.strip().lower()
    return re.sub(r"[\s-]+", "_", lowered)


# 函数说明：_first_nonempty_lines
# 用途：在规划、执行、审计协作中处理 `_first_nonempty_lines`，通过
# `str(text or '').splitlines` 完成首个内部处理步骤。
# 参数：
#   text：待处理的文本，类型 `str`。
#   count：`count`输入或配置值，类型 `int`。
# 返回：类型 `list[str]`；返回 `lines`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`str(text or '').splitlines`。
# 分支与异常：
#   当 `len(lines) >= count` 时，结束当前循环。
def _first_nonempty_lines(text: str, count: int) -> list[str]:
    lines: list[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
            if len(lines) >= count:
                break
    return lines


# ---------------------------------------------------------------------------
# 阻断约束降级
# ---------------------------------------------------------------------------

_BLOCKING_SECTION_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?\s*(?:阻断约束|blocking\s+(?:acceptance\s+)?(?:constraints?|claims?))"
    r"\s*(?:\*\*)?\s*[:：]\s*(?:\*\*)?\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
_NO_BLOCKING_RE = re.compile(
    r"(?ix)^\s*(?:[-*+]\s*|\d+[.)]\s*)?(?:\*\*)?(?:无|没有|暂无|none|nothing|n/?a|not\s+applicable)"
    r"(?:\*\*)?(?:\s*[。.,，;；:：（(].*)?\s*$"
)
# 报告里所有其他小节头都要能结束阻断约束小节，否则后面的小节会被算进去，
# 把本来干净的审计误降级。`范围外约束` 必须在这里：它不参与降级。
_SECTION_BOUNDARY_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?\s*(?:范围外约束|契约结论|原题约束清单|契约覆盖检查|逐项反查"
    r"|可能评分风险|过窄或错误解释|建议契约修订|审计事实|证据|缺口|下一步|可信产物|不可信产物"
    r"|可信/不可信产物|步骤验收逐条|给任务管理器的状态更新|验收约束反查|状态|完整性|契约审计|步骤验收"
    r"|out[-\s]of[-\s]scope\s+constraints|contract\s+conclusion|possible\s+scoring\s+risks"
    r"|over-narrow|recommended\s+contract\s+revision|audit\s+facts|evidence|gaps?|next\s+step"
    r"|state\s+update\s+for\s+manager)\s*(?:\*\*)?\s*[:：]"
)

COMPLETION_GUARD_NOTE = (
    "harness 完成守卫: auditor 列出了非空阻断约束，因此不接受 `complete`、"
    "`satisfied` 和 aligned 契约结论。"
)


# 函数说明：has_blocking_constraints
# 用途：判断`blocking_constraints`是否满足当前实现的条件。
# 参数：
#   report：报告输入或配置值，类型 `str`。
# 返回：类型 `bool`；按分支返回 `not _is_none(rest)`；
# `bool(section) and (not all((_is_none(item) for item in section)))`；`False`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`str(report or '').splitlines` →
# `_BLOCKING_SECTION_RE.match` → `match.group` → `_is_none` →
# `_SECTION_BOUNDARY_RE.match`。
# 分支与异常：
#   当 `not match` 时，跳过当前循环项。
#   当 `rest` 时，返回 `not _is_none(rest)`。
#   当 `not stripped` 时，跳过当前循环项。
#   当 `_SECTION_BOUNDARY_RE.match(stripped)` 时，结束当前循环。
def has_blocking_constraints(report: str) -> bool:
    lines = str(report or "").splitlines()
    for index, line in enumerate(lines):
        match = _BLOCKING_SECTION_RE.match(line)
        if not match:
            continue
        rest = match.group("rest").strip()
        if rest:
            return not _is_none(rest)
        section: list[str] = []
        for following in lines[index + 1 :]:
            stripped = following.strip()
            if not stripped:
                continue
            if _SECTION_BOUNDARY_RE.match(stripped):
                break
            section.append(stripped)
        return bool(section) and not all(_is_none(item) for item in section)
    return False


# 函数说明：_is_none
# 用途：返回 `bool(_NO_BLOCKING_RE.match(text.strip().strip('`')))，提供 规划、执行、审
# 计协作 的派生值。
# 参数：
#   text：待处理的文本，类型 `str`。
# 返回：类型 `bool`；返回 `bool(_NO_BLOCKING_RE.match(text.strip().strip('`')))。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`_NO_BLOCKING_RE.match`。
def _is_none(text: str) -> bool:
    return bool(_NO_BLOCKING_RE.match(text.strip().strip("`")))


@dataclass(frozen=True, slots=True)
class GuardResult:

    header: ControlHeader
    report: str
    downgraded: bool


# 函数说明：apply_blocking_guard
# 用途：``阻断约束:`` 非“无”时：complete→incomplete、satisfied→not_satisfied、aligned→
# unknown。
# 参数：
#   header：传给 `GuardResult` 的输入，类型 `ControlHeader`。
#   report：传给 `has_blocking_constraints` 的输入，类型 `str`。
# 返回：类型 `GuardResult`；按分支返回 `GuardResult(header, report, False)`；
# `GuardResult(guarded, f'{report.rstrip()}\n\n{COMPLETION_GUARD_NOTE}', True)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`has_blocking_constraints` →
# `GuardResult` → `ControlHeader` → `report.rstrip`。
# 分支与异常：
#   当 `not has_blocking_constraints(report)` 时，返回
# `GuardResult(header, report, False)`。
#   当 `guarded == header` 时，返回 `GuardResult(header, report, False)`。
def apply_blocking_guard(header: ControlHeader, report: str) -> GuardResult:
    """``阻断约束:`` 非“无”时：complete→incomplete、satisfied→not_satisfied、aligned→unknown。"""

    if not has_blocking_constraints(report):
        return GuardResult(header, report, False)
    guarded = ControlHeader(
        status=AuditStatus.INCOMPLETE if header.status is AuditStatus.COMPLETE else header.status,
        integrity=header.integrity,
        contract_audit=(
            ContractAudit.UNKNOWN
            if header.contract_audit is ContractAudit.ALIGNED
            else header.contract_audit
        ),
        step_acceptance=(
            StepAcceptance.NOT_SATISFIED
            if header.step_acceptance is StepAcceptance.SATISFIED
            else header.step_acceptance
        ),
    )
    if guarded == header:
        return GuardResult(header, report, False)
    return GuardResult(guarded, f"{report.rstrip()}\n\n{COMPLETION_GUARD_NOTE}", True)


# ---------------------------------------------------------------------------
# 最终回复用：去掉控制头和反查协议，只留审计正文
# ---------------------------------------------------------------------------

_PROTOCOL_RE = re.compile(
    r"^(?:[-*]\s*)?(?:\*\*)?\s*(?:验收约束反查|契约结论|原题约束清单|契约覆盖检查|逐项反查|阻断约束"
    r"|范围外约束|可能评分风险|过窄或错误解释|建议契约修订|给任务管理器的状态更新|步骤验收逐条)\s*[:：]"
)


# 函数说明：audit_findings_body
# 用途：协议小节按提示词要求位于报告末尾，遇到第一个协议小节头就停止。
# 参数：
#   report：传给 `strip_control_header` 的输入，类型 `str`。
# 返回：类型 `str`；返回 `'\n'.join(body)`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：
# `strip_control_header(report).splitlines` → `strip_control_header` →
# `_PROTOCOL_RE.match`。
# 分支与异常：
#   当 `not value` 时，跳过当前循环项。
#   当 `_PROTOCOL_RE.match(value)` 时，结束当前循环。
def audit_findings_body(report: str) -> str:
    """协议小节按提示词要求位于报告末尾，遇到第一个协议小节头就停止。"""

    body: list[str] = []
    for line in strip_control_header(report).splitlines():
        value = line.strip()
        if not value:
            continue
        if _PROTOCOL_RE.match(value):
            break
        body.append(value)
    return "\n".join(body)


# ---------------------------------------------------------------------------
# 截断
# ---------------------------------------------------------------------------


# 函数说明：clip_preserve
# 用途：超长时保留前 65% 和后 35%，中间写明截掉了多少。
# 参数：
#   text：待处理的文本，类型 `str`。
#   max_chars：保留的字符数上限，类型 `int`。
# 返回：类型 `str`；按分支返回 `text`；`text[:head_chars].rstrip() + f'\n\n...[已截断 {
# len(text) - max_chars} 字符；保留开头和结尾]...\n\n' +…`。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`text[:head_chars].rstrip` →
# `text[-tail_chars:].lstrip`。
# 分支与异常：
#   当 `max_chars <= 0 or len(text) <= max_chars` 时，返回 `text`。
def clip_preserve(text: str, max_chars: int) -> str:
    """超长时保留前 65% 和后 35%，中间写明截掉了多少。"""

    text = str(text or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head_chars = max(1, int(max_chars * 0.65))
    tail_chars = max(1, max_chars - head_chars)
    return (
        text[:head_chars].rstrip()
        + f"\n\n...[已截断 {len(text) - max_chars} 字符；保留开头和结尾]...\n\n"
        + text[-tail_chars:].lstrip()
    )


__all__ = [
    "COMPLETION_GUARD_NOTE",
    "AuditStatus",
    "ContractAudit",
    "ControlHeader",
    "GuardResult",
    "Integrity",
    "ManagerOutput",
    "NewStep",
    "Route",
    "StepAcceptance",
    "StepUpdates",
    "Supersede",
    "apply_blocking_guard",
    "audit_findings_body",
    "clip_preserve",
    "format_control_header",
    "has_blocking_constraints",
    "parse_control_header",
    "parse_manager_output",
    "parse_route",
    "parse_step_updates",
    "strip_control_header",
]
