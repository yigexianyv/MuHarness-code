# SWE-bench Lite：mini-SWE-agent 对照 MuHarness 普通模式

这套入口实现 `docs/swebench-lite-eval-plan.md` 的普通模式对照实验。默认清单是
`lite30.json`，逐字复制自用户的 `docs/swebench-lite30.json`，共 30 题。
历史 84 个提交的难度标签作为用户提供的抽样元数据保存，没有重新计算。
`prepare` 会用实际下载的数据逐一验证清单成员、repo、重复 ID 和难度边界。

## 一键执行

只运行 MuHarness 并用官方 harness 判分，在项目根目录运行：

```powershell
.\swebench-one.ps1
.\swebench-one.ps1 -Lite30
.\swebench-one.ps1 -Lite30 -Experiment output\my-lite30
```

这个入口从 `backend/.env` 读取模型和服务配置，也支持 `-Config`、`-Model`、
`-Instance` 等参数。`-Lite30` 再次运行会续跑，单题模式默认使用 dev split。

需要同时运行 mini-SWE-agent 与 MuHarness 对照时，使用下面的入口。

Windows 在项目根目录 PowerShell 运行：

```powershell
.\swebench.ps1
```

前提是 Docker 已启动且已安装 uv。首次缺少评测环境时，脚本会自动创建并安装；
已有环境会复用。模型及地址读取 `backend/tests/eval/swebench/config.example.json`，
密钥读取它指定的环境变量或 `backend/.env`。默认配置使用 DeepSeek；换模型先修改配置。
脚本依次执行：环境自检 → 固定 Lite30 → 单独的 dev gold 检查 → 两组模型运行 →
官方判分 → 报告。只有 dev gold 成功才开始模型调用。
最终报告位于 `output/swe-lite30/report.md`，重复同一命令会续跑已冻结的实验。

```powershell
# 只检查环境，不调用模型，也不拉取题目镜像。
.\swebench.ps1 -CheckOnly
# 自定义模型配置或新实验目录。
.\swebench.ps1 -Config my-config.json -Experiment output\swe-lite30-v2
```

在 Linux / WSL 已准备好下文的三个环境后，同样可用一条命令串起全部阶段：

```bash
cd ~/swe/MuHarness/backend
sudo -E .venv-eval/bin/python -m tests.eval.swebench all \
  --mini-python ../output/venv-mini/bin/python \
  --judge-python ../output/venv-judge/bin/python
```

源码或配置变化时，使用新的 `-Experiment` / `--experiment` 目录；不覆盖已有实验。
不完整的评分仍会输出部分报告，并以非零退出码提醒检查基建错误和缺失结果。

两组共用模型、Chat Completions 服务地址、temperature、最大输出、请求超时；
SDK 重试为 0。默认每题 100 次有效模型调用（摘要和输出修正也计数），
命令上限 120 秒，模型运行上限 1800 秒。

接口临时故障（429、5xx、连接错误、超时）由两组相同的计数层重试，默认最多 2 次，
间隔 5 秒、20 秒（`transient_retries`、`retry_backoff_seconds`）。每次重试都写入
轨迹并计入 `model_calls`，同时记入 `transient_failures` 并从 100 次预算中退还，
避免接口抖动消耗 Agent 预算。只有运行终止在最后一次失败的调用上时，才记为基建错误；
中途失败但已恢复的调用（例如一次失败的上下文摘要）只记入 `recovered_errors`，
不会让已完成的运行被排除出判分。

统一的 `max_output_tokens` 也限制了 MuHarness 遇到截断时放大输出重试（16384）的能力；
这是有意保持两组输出上限一致，报告结论应注明这一点。下载镜像、复制仓库和官方判分不占模型运行时间。
mini 使用安装包自带的官方 SWE-bench 提示和 DefaultAgent；MuHarness 使用生产的
AgentRuntime、默认系统提示、文件工具及上下文摘要，Shell 换成持久容器执行。
没有挂载长期记忆、反思、Skills、MCP、网页或 MEA；Task 和产物发布也未启用。
这是完整 harness 配置之间的对照，不能单独归因于某一个机制。

## 环境

正式跑分建议在 Linux / WSL2 的 Linux 文件系统进行；避免 `/mnt/c`、`/mnt/d`。
Windows 可运行离线测试及 Docker Desktop 集成检查，但文件换行、权限和性能与 Linux
不同，不要把两个平台的结果混进同一个实验。清单默认使用 linux/amd64 镜像。

官方镜像与 mini 使用 root、`BASH_ENV=/root/.bashrc`。为保证容器创建的新文件也能由
宿主文件工具编辑，Linux 控制进程须在专用评测环境中以 root 运行；代码会检查这一点。
三套依赖分开，避免 LiteLLM 或官方判分依赖改动产品环境：

```bash
# 以下假设仓库在 ~/swe/MuHarness，已安装 uv 和可用的 Docker。
cd ~/swe/MuHarness/backend
uv venv --python 3.13 .venv-eval
uv pip install --python .venv-eval/bin/python -r requirements.txt \
  -r tests/eval/swebench/requirements-data.txt

uv venv --python 3.11 ../output/venv-mini
uv pip install --python ../output/venv-mini/bin/python \
  mini-swe-agent==2.4.6 litellm==1.105.1
uv venv --python 3.11 ../output/venv-judge
uv pip install --python ../output/venv-judge/bin/python swebench==5.0.2

# 用 uv pip，不依赖 uv venv 默认不提供的 pip 可执行文件。
uv pip freeze --python ../output/venv-mini/bin/python > ../output/mini.requirements.txt
uv pip freeze --python ../output/venv-judge/bin/python > ../output/judge.requirements.txt
```

实现已验证 mini 2.4.6 的原生 bash 工具调用接口和官方 harness 5.0.2 的 CLI。
其他版本会在 `selfcheck` 检查兼容性，并在实验中冻结版本和源码摘要。
5.x 判分器需要新版数据中的 `image/eval_script/log_parser/eval_type` 等字段；
不要将旧版数据快照与新版判分器混用。

## 从 dev 开始

先按实际模型修改 `config.example.json` 的一份副本，填写模型 ID、服务地址和参数。
配置只写密钥的环境变量名；实际密钥从环境或现有 `backend/.env` 读取。
本实现暂限 OpenAI / DeepSeek / Qwen 的 OpenAI 兼容 Chat Completions 接口，
确保两组协议一致。没有实现 Anthropic Messages 或 Responses 对照。

```bash
cd ~/swe/MuHarness/backend
PY="$PWD/.venv-eval/bin/python"
MINI="$(realpath ../output/venv-mini/bin/python)"
JUDGE="$(realpath ../output/venv-judge/bin/python)"
EXP="$(realpath ../output)/swe-dev-01"

$PY -m tests.eval.swebench selfcheck --mini-python "$MINI" --judge-python "$JUDGE"
$PY -m tests.eval.swebench prepare --out "$EXP" --split dev --selection all \
  --mini-python "$MINI" --judge-python "$JUDGE" \
  --config tests/eval/swebench/config.example.json

# 从该 dev 实验 tasks.json 中任选一个 instance_id 替换 DEV_ID。
# 这个命令把可信 gold 应用到共享工作区，经过相同的补丁导出，再由官方新容器判分。
sudo "$PY" -m tests.eval.swebench selfcheck --experiment "$EXP" \
  --instance-id DEV_ID --mini-python "$MINI" --judge-python "$JUDGE"

# 设置配置指定的密钥；也可使用现有 backend/.env，勿把密钥放进命令参数。
export DEEPSEEK_API_KEY='YOUR_KEY'
sudo --preserve-env=DEEPSEEK_API_KEY "$PY" -m tests.eval.swebench run \
  --experiment "$EXP" --mini-python "$MINI" --filter '^DEV_ID$'
sudo "$PY" -m tests.eval.swebench grade --experiment "$EXP" --judge-python "$JUDGE"
sudo "$PY" -m tests.eval.swebench report --experiment "$EXP"
```

上面的 gold 检查不调用模型，但会拉镜像并执行官方测试；只有 `run` 会调用模型服务。
请先确认 gold round-trip resolved，再做真实模型 dev 单题和 dev 全集。
`selfcheck` 只允许在 dev 实验中读 gold。没有自行实现 F2P/P2P 判分。

## 固定 Lite30 / 全量 300

```bash
EXP="$(realpath ../output)/swe-lite30-01"
$PY -m tests.eval.swebench prepare --out "$EXP" --split test \
  --selection tests/eval/swebench/lite30.json \
  --mini-python "$MINI" --judge-python "$JUDGE"

# 可按 repo 分批跑；filter 不改变已冻结的清单和分母。
sudo --preserve-env=DEEPSEEK_API_KEY "$PY" -m tests.eval.swebench run \
  --experiment "$EXP" --mini-python "$MINI" --filter '^django__' --resume
sudo --preserve-env=DEEPSEEK_API_KEY "$PY" -m tests.eval.swebench run \
  --experiment "$EXP" --mini-python "$MINI" --resume

sudo "$PY" -m tests.eval.swebench export --experiment "$EXP"
sudo "$PY" -m tests.eval.swebench grade --experiment "$EXP" \
  --judge-python "$JUDGE" --max-workers 1 --timeout 1800
sudo "$PY" -m tests.eval.swebench report --experiment "$EXP"
```

全量 Lite300 只需新建实验，`prepare --split test --selection all`。
离线准备可用 `--dataset-file /absolute/path/snapshot.json`（也支持 JSONL），冻结文件 SHA；
在线模式把 Hugging Face revision 先解析成 SHA，再按 SHA 下载。
可用 `--revision SHA` 显式选择数据版本。当地数据标签的 split 由操作者负责，记录中会注明
它是本地快照，不会伪称已取得在线 revision。

`prepare` 不拉镜像。每题首次执行时锁定本机镜像 ID / registry digest，之后两组使用
相同 ID；`--no-pull` 可禁止拉取。官方 5.x 判分输入也替换为该锁定 ID。
容器断网，只有工作区挂入 `/testbed`，控制记录和答案没有挂入容器。
官方镜像可能在数据集 base commit 上追加一个空的初始化提交。启动时会核对
base 的祖先关系及 Git tree ID；只有源码树完全一致才将副本 HEAD 对齐到 base，
源码不同、基准缺失或历史不相关均拒绝启动。原镜像 HEAD 和对齐结果写入 `git_audit`。
Agent 开始前，两组都会清理工作区中超出 base commit 的 git 历史：删除远程、
指向 base 之后的分支和标签（保留 HEAD 祖先上的旧版本标签），清空 reflog 并 gc。
`git rev-list --all --reflog` 与 HEAD 的提交数不一致时拒绝开始；清理前后计数写入
`result.json` 的 `git_audit`，报告另行统计 Agent 查看 git 历史的命令数。
当前串行调度，每题每组独立进程；不共享全局环境、记忆或会话。

## 退出、续跑及产物

正常完成、调用预算用尽或模型超时，都先停止容器里的后台进程，再导出补丁。
以数据集的固定 base commit 为基线，用独立 Git index 导出，包含暂存、新文件和
Agent 自行提交的改动。保留 `raw.patch`，提交补丁只移除原仓库中不存在的 `patch.txt`。
其余临时文件或测试改动不做静默过滤；问题提示已要求修改非测试代码，原始结果供审计。

`--resume` 跳过所有已记录的尝试，包括错误结果；不会重试错题或挑最好的一次。
`--retry-infra`（隐含 `--resume`）只重跑状态为 `infra_error` 的尝试，每题每组最多 2 次；
旧结果改名为 `result.infra-<attempt>.json` 保留，新结果的 `previous_infra_attempts`
指向它们。Agent 的结果（包括超时、空补丁、预算耗尽）永远不会重跑。
一键流程 `all` 默认使用 `--retry-infra`。重试仍失败的基建错误保留预测行，报告标出覆盖缺口。更改配置、依赖或评测/产品源码后须新建实验。
若控制进程被强杀，先确认它和记录的 worker 已退出，删除该实验的 `.controller.lock`
（仅该文件），再运行：

```bash
sudo "$PY" -m tests.eval.swebench recover --experiment "$EXP"
sudo --preserve-env=DEEPSEEK_API_KEY "$PY" -m tests.eval.swebench run \
  --experiment "$EXP" --mini-python "$MINI" --resume
```

`recover` 检查 PID 和容器归属，只导出并封存中断结果，不发起模型调用。
worker 没有完成时记为基建中断；若容器也已丢失，则需要人工检查该 attempt，不自动重做。
当前不支持一个实验内并发运行多个控制进程，也没有自动清理题目镜像或历史尝试目录。
`grade` 通过隔离进程调用官方 CLI；Windows 下仅强制它写出的文本保持 LF，
避免官方 `Path.write_text` 的 CRLF 转换破坏补丁；Linux 下直接使用官方行为。

产物都在实验目录：

- `manifest.json`：清单/配置/数据/源码摘要、Git HEAD、Python/mini/判分器版本与源码摘要。
- `tasks.json`：仅含 ID、repo、base、issue、image；`trusted/dataset.json` 留给可信判分器。
- `runs/{variant}/{instance}/attempt-*`：工作区、逐次模型请求/响应、工具事件、用量、原始/提交补丁。
- `predictions/{variant}.jsonl`：固定 N 行，未运行或空补丁均保留；coverage 文件明确区分状态。
- `judge/`：每次官方判分采用新 run_id 和新容器，保存参数、日志、逐题官方 report。
- `grades/`：绑定预测文件摘要的判分记录；`report.json` / `report.md` 为最终统计。

统计包含配对四格、精确 McNemar、固定种子配对 bootstrap 95% 区间、repo 分组、
难度层分组、退出原因、token / 调用 / 耗时（中位数、P90）。另有判分端计算的过程指标：
补丁应用失败、判分测试超时、未解决题的 FAIL_TO_PASS 部分通过比例、PASS_TO_PASS 回归、
是否改到参考补丁涉及的文件、是否跑过测试或脚本、补丁行数、已恢复接口错误。
参考补丁只在报告阶段读取，不进入 Agent 运行。

官方 5.x 判分器在补丁无法应用或测试超时时不写 `report.json`，但其 `results.json`
将它们计为未解决（分母不变）；这里同样计为未解决并单列来源，不当作未知结果剔除。
缺失判分或官方标记的基建故障不当作模型失败；
空预测按基准提交惯例记未解决并单列，
不伪称经过官方测试。覆盖不完整时只给下界，配对统计只使用完整配对，不据此宣称胜出。
30 题的分层清单不能外推为 Lite300 通过率。
可选 `report --prices INPUT CACHED_INPUT OUTPUT` 输入同币种每百万 token 单价估算费用；
失败调用缺失的用量会单列，估算不是账单。

## 回归验证

```bash
cd ~/swe/MuHarness/backend
uv pip install --python .venv-eval/bin/python -r requirements-dev.txt
MUHARNESS_MINI_PYTHON="$MINI" $PY -m pytest tests/offline/test_swebench.py -q
# 容器测试先准备 muharness-sandbox:latest（backend/docker/sandbox/Dockerfile）。
# JUDGE 选项额外在微型仓库验证官方判分的正确补丁通过、错误补丁不通过。
sudo env MUHARNESS_MINI_PYTHON="$MINI" MUHARNESS_JUDGE_PYTHON="$JUDGE" \
  MUHARNESS_SWEBENCH_DOCKER_TEST=1 \
  "$PY" -m pytest tests/offline/test_swebench.py -q
```

测试使用本地假模型服务验证实际 SDK 和 mini；不访问付费模型。
Docker 测试验证同一文件在宿主和容器可见、命令超时、固定 base 导出、提交后保留修改、
新文件和 patch.txt 处理。它不是 SWE-bench 真实 dev gold 或正式成绩的替代。
