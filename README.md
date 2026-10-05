# MuHarness

MuHarness 是一个本地 AI Agent 工作台。通过浏览器提出需求，Agent 调用工具完成文件处理、代码修改等任务，并保存执行记录与交付物。

支持两种执行方式：**普通任务**由模型与工具循环完成；**长任务**由管理者拆解、执行者操作、审计者独立验收，按步骤推进。

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 工具执行 | 文件读写、Shell、HTTP、网页搜索，支持审批与运行预算 |
| 长任务 | 计划确认、分步验收、补充要求、暂停继续与停滞检测 |
| 上下文透明度 | 查看用量、压缩时间线、摘要变化和请求原文，设置“必须记住的事项” |
| 执行回溯 | 从检查点恢复工作区文件，带纠正指令分叉到新会话 |
| 记忆与扩展 | 长期记忆检索、Skills、MCP 工具接入 |
| 记录与交付 | 执行轨迹、工具证据、文件与链接发布、定时任务 |

技术栈：**Python + FastAPI + SQLite**、**React + TypeScript + Vite**，通过 WebSocket JSON-RPC 通信。Shell 在一次性 Docker 容器中执行，默认禁网。

## 快速开始

准备 **Python 3.13+、Node.js 20+、npm、Docker Engine** 和模型 API Key。Windows 使用 Docker Desktop 时，启动 Linux 容器引擎。

以下使用 PowerShell，两个终端均从仓库根目录开始。

### 1. 准备环境与后端

```powershell
New-Item -ItemType Directory -Force .\workspace | Out-Null
docker build -t muharness-sandbox:latest -f .\backend\docker\sandbox\Dockerfile .\backend

cd .\backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

在 `backend/.env` 中设置 `MODEL_DEFAULT_PROVIDER`、对应的 API Key 和模型名称：

| 服务商 | Provider | 密钥 / 模型变量 |
| --- | --- | --- |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` |
| OpenAI | `openai` | `OPENAI_API_KEY` / `OPENAI_MODEL` |
| 通义千问 | `qwen` | `QWEN_API_KEY` / `QWEN_MODEL` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` / `DEEPSEEK_MODEL` |

其他配置见 [环境变量示例](backend/.env.example)。保存后启动：

```powershell
.\.venv\Scripts\python.exe -m app.server --host 127.0.0.1 --port 8000
```

### 2. 启动前端

在第二个终端运行：

```powershell
cd .\web
npm ci
npm run dev
```

打开 <http://127.0.0.1:5173>。前端默认连接 `http://127.0.0.1:8000`，可通过 `VITE_AGENT_SERVER_URL` 修改；后端健康检查为 <http://127.0.0.1:8000/health>。

## 怎么使用

- **普通任务**：直接在普通模式提出需求，例如“创建 demo/a.txt，写入 hello，再读取并报告实际内容”。
- **长任务**：在 PLAN 模式生成计划，确认后执行。通过任务面板继续剩余阶段或从阶段 1 重新执行；PLAN 模式本身不写工作区，发送“继续”不会自动切换模式。
- **审批与提问**：在“待处理”或长任务面板处理。长任务暂停会在当前轮次结束后生效。
- **上下文与回溯**：进入“账本 → 执行记录 → 执行详情”，查看上下文面板和执行时间线，使用“重做此步”。

“重做此步”会先展示文件差异，再恢复快照并打开分支会话；纠正内容预填到输入框，发送后开始执行。“从阶段 1 重新执行”则创建新任务从头运行，不回退文件。

## 项目结构

```text
backend/
├── app/
│   ├── runtime/       Agent 循环、长任务、上下文与回溯
│   ├── domain/        会话、任务、记忆、技能与交付物
│   ├── tools/         工具注册与执行
│   ├── safety/        审批与沙箱
│   ├── models/        模型适配
│   └── server/        后端接口
├── tests/             回归测试与评测
└── docker/            沙箱镜像
web/src/               前端页面、组件与状态管理
workspace/             Agent 工作区
```
