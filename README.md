# MuHarness

MuHarness 是一个本地 AI Agent 工作台，面向文件处理、代码修改、资料检索和复杂多步骤任务。用户通过浏览器提出目标，Agent 调用工具执行，并保存任务进度、执行证据与交付物。

支持两种执行方式：**普通任务**通过模型与工具循环完成；**长任务**由 Manager（管理者）、Executor（执行者）、Auditor（审计者）分工，按阶段执行和验收。通过长期记忆、按需加载的 Skill 与 Skill 自进化，积累可复用的任务经验。

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 模型接入 | 适配 Chat Completions、Responses、Anthropic Messages，统一消息、工具调用和用量 |
| 工具治理 | 文件、Shell、HTTP、搜索工具统一注册与执行，支持参数检查、权限和人工审批 |
| 长任务编排 | 计划确认、角色分工、分步审计、补充要求、暂停继续与停滞检测 |
| 上下文管理 | 输入预算、大输出摘录、结构化滚动摘要，以及必须保留的事项 |
| 上下文透明化 | 查看请求内容、Token 用量、压缩时间线和摘要变化 |
| 长期记忆 | Markdown 记忆、关键词与可选语义检索、融合排序、反思与维护 |
| Skill 自进化 | 从完成任务与执行轨迹中提炼技能候选，人工审核后新增或更新 Skill |
| 恢复与回溯 | 中断检查点、工作区快照、差异预览与带纠正指令的分支会话 |
| 记录与交付 | Trace、Evidence、Artifact 关联执行过程、工具证据与已发布交付物 |
| 扩展与调度 | stdio MCP 接入、按需工具发现与定时自动化 |

**技术栈：** Python / FastAPI / WebSocket JSON-RPC / SQLite FTS5 / Docker / MCP；前端使用 React / TypeScript / Vite。

## 快速开始

建议准备 **Python 3.14、Node.js 20+、npm、Docker Engine** 和模型 API Key。以下使用 PowerShell，从仓库根目录运行；Windows Docker Desktop 需要启动 Linux 容器引擎。

### 1. 准备后端与沙箱

```powershell
New-Item -ItemType Directory -Force .\workspace | Out-Null
docker build -t muharness-sandbox:latest -f .\backend\docker\sandbox\Dockerfile .\backend

cd .\backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

在 `backend/.env` 中选择服务商并填写对应的密钥与模型。例如：

```dotenv
MODEL_DEFAULT_PROVIDER=anthropic
ANTHROPIC_API_KEY=填写你的密钥
ANTHROPIC_MODEL=填写你有权限使用的模型名称
MUHARNESS_SANDBOX_BACKEND=docker
```

也支持 `openai`、`qwen`、`deepseek`，分别设置对应前缀的 `API_KEY` 和 `MODEL`。其他配置见 [环境变量示例](backend/.env.example)。

### 2. 启动后端

在 `backend/` 目录运行：

```powershell
.\.venv\Scripts\python.exe -m app.server --host 127.0.0.1 --port 8000
```

健康检查：<http://127.0.0.1:8000/health>。

### 3. 启动前端

新开终端，从仓库根目录运行：

```powershell
cd .\web
npm ci
npm run dev
```

打开 <http://127.0.0.1:5173>。前端默认连接 `http://127.0.0.1:8000`，可通过 `VITE_AGENT_SERVER_URL` 修改。

## 使用方式

- **普通任务：** 直接提出目标，例如“创建 demo/a.txt，写入 hello，再读取并核对内容”。
- **长任务：** 在 PLAN 模式生成计划，确认后执行；通过长任务面板补充要求、暂停或继续。PLAN 模式本身不写工作区。
- **审批与提问：** 在“待处理”或长任务面板处理操作授权和澄清问题。
- **执行详情：** 进入“账本 → 执行记录 → 执行详情”，查看上下文、执行时间线、证据与交付物。
- **重做此步：** 预览文件差异，恢复快照覆盖的工作区文件，并打开带纠正指令的分支会话；发送后开始执行。

“从阶段 1 重新执行”会创建新任务从头运行，不回退文件；中断恢复则依据检查点继续或先核查现场，三者用途不同。

## 关键设计

### 模型与工具循环

```text
用户目标 → 构建上下文 → 模型请求 → 工具调用
                          ↑              ↓
                          └── 工具结果 ──┘
                          ↓
                     最终回复与记录
```

模型适配层统一协议，AgentLoop 协调执行，ToolExecutor 检查参数、模式、权限与审批。执行受步数、工具轮数、累计 Token 预算和重复调用检查约束；模型写出的普通文本“工具调用”不会直接执行。

文件操作校验工作区路径，Shell 在一次性 Docker 容器内执行，默认禁网。Docker 不可用时拒绝 Shell 执行，不自动回退到宿主机。

### Manager–Executor–Auditor 长任务

Manager 确定当前步骤和要求，Executor 实施，Auditor 独立核验执行证据与交付物。验收通过后推进下一步，未通过则带审计意见继续修正；阻塞或达到运行限制时保留现场并反馈。

三个角色使用独立上下文和工具权限。关键阶段与子 Run 关联持久化，审计补丁按操作 ID 幂等应用，并通过只读限制、快照守卫、轮次预算和停滞检测约束长任务。

### 上下文与记忆

按模型窗口计算输入预算。大工具结果首次进入模型时确定为完整可用文本或带标记的摘录，随后不反复截改旧结果；达到压缩条件后，将较早内容转成结构化滚动摘要，保留当前要求及未完成工具协议需要的信息。

工具原文、执行器保存的 ToolResult 和模型可见摘录分层处理。摘录不是全文，需要细节时通过 Evidence 核查。前端展示请求、压缩过程和摘要变化，便于理解模型实际获得的信息。

长期记忆通过关键词与可选语义通道检索、融合排序，相关候选需要正式读取后才能作为依据；运行后通过反思和维护积累跨任务信息。

### Skill 与自进化

Skill 保存可复用的操作流程，先展示目录，当前阶段需要时才读取正文与资源，并限制活跃数量和上下文占用。

```text
已完成任务与执行轨迹
  → 增量批次扫描
  → 模式挖掘与技能蒸馏
  → 相关性、重叠检查
  → 待审核候选
  → 人工接受或拒绝
  → 新增或更新正式 Skill
```

Skill Learning 记录扫描进度与处理中批次，控制重复处理和失败恢复。`skill_propose` 也可提交已验证流程的候选。候选不会直接生效，审核接受后才创建或更新技能；自进化积累的是操作知识，不涉及模型权重训练。

### 恢复与可追溯性

检查点保存模型与工具阶段，区分已完成调用和结果未知的调用。恢复时不能仅因没有收到返回，就认定有副作用的操作未执行。

Trace 记录模型、工具、审批和状态事件；Evidence 提供工具证据读取；Artifact 保存发布记录。文件发布会保存副本并记录大小与 SHA-256，关联会话、Run 和任务。工具或发布成功不等于目标已经通过验收。

## 项目结构

```text
backend/
├── app/
│   ├── application.py       应用装配与生命周期
│   ├── runtime/             Agent 循环、上下文、长任务、恢复与回溯
│   ├── domain/              会话、任务、记忆、技能学习与交付物
│   ├── tools/               工具定义、发现与执行
│   ├── safety/              审批与沙箱
│   ├── integrations/        MCP 等外部能力接入
│   ├── records/             Trace 与 Evidence
│   ├── models/              模型协议适配
│   ├── model_settings/      模型设置
│   └── server/              HTTP、WebSocket JSON-RPC 与调度
├── tests/                   离线回归与真实模型行为评测
└── docker/sandbox/          Shell 沙箱镜像
web/src/                    页面、组件与状态管理
workspace/                  Agent 默认工作区
```

后端使用 pytest 做离线回归，前端使用 Vitest 与 TypeScript 检查；真实模型评测覆盖工具行为、上下文压缩和 MEA 长任务，结合成功率、Token 用量、调用次数与耗时验证效果。
