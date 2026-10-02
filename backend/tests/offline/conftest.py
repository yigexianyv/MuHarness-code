"""Environment prerequisites for tests that launch real sandbox containers."""

import shutil
import subprocess

import pytest


# 函数说明：docker_engine
# 用途：在回归测试与测试辅助中处理 `docker_engine`，通过 `shutil.which` 完成首个内部处理
# 步骤。
# 返回：不返回结果值（隐式 None）。
# 关键调用（按源码出现顺序，实际执行取决于分支）：`shutil.which` → `pytest.skip` →
# `subprocess.run`。
# 分支与异常：
#   捕获 `(OSError, subprocess.TimeoutExpired)` 后，执行异常处理调用 `pytest.skip`、
# `type`。
@pytest.fixture(scope="session")
def docker_engine():
    executable = shutil.which("docker")
    if executable is None:
        pytest.skip("real Docker sandbox test: docker executable is unavailable")
    try:
        result = subprocess.run(
            [executable, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        pytest.skip(f"real Docker sandbox test: engine probe failed ({type(error).__name__})")
    if result.returncode != 0:
        pytest.skip("real Docker sandbox test: engine is unavailable; docker info failed")

