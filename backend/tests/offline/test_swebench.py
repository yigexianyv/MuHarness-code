from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tests.eval.swebench import runner
from tests.eval.swebench.common import (
    PACKAGE,
    digest,
    load_experiment,
    read_json,
    validate_config,
    write_json,
)
from tests.eval.swebench.container import TaskContainer, docker
from tests.eval.swebench.dataset import public_task, select_rows
from tests.eval.swebench.grading import official_reports
from tests.eval.swebench.meter import CallLimitExceeded, Meter
from tests.eval.swebench.report import build_report, paired_statistics
from tests.eval.swebench.worker import run_muharness


def task(number=1):
    return {
        "instance_id": f"owner__repo-{number}",
        "repo": "owner/repo",
        "base_commit": "a" * 40,
        "problem_statement": "Fix the bug.",
        "patch": "SECRET_GOLD",
        "test_patch": "SECRET_TESTS",
        "FAIL_TO_PASS": ["SECRET_TEST_NAME"],
        "hints_text": "SECRET_HINT",
    }


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / "experiment"
    runner.prepare(
        root,
        [task(1), task(2)],
        None,
        {},
        {"dataset": "fixture", "split": "dev"},
        {"mini": {"config": {}}, "muharness": {}, "judge": {}},
    )
    return root


def save_attempt(root, variant, instance, patch=b"", status="completed"):
    directory = root / "runs" / variant / instance
    (directory / "attempt-fixture").mkdir(parents=True)
    (directory / "attempt-fixture" / "model.patch").write_bytes(patch)
    result = {
        "instance_id": instance,
        "attempt": "attempt-fixture",
        "fingerprint": read_json(root / "manifest.json")["fingerprint"],
        "patch_sha256": digest(patch),
        "status": status,
        "stop_reason": "run_budget",
        "duration_seconds": 3,
        "usage": {"model_calls": 2, "input_tokens": 20, "output_tokens": 5},
    }
    write_json(directory / "result.json", result)


def test_selection_validates_real_membership_and_keeps_gold_private(experiment):
    public = (experiment / "tasks.json").read_text()
    assert "SECRET" not in public
    assert set(public_task(task())) == {
        "instance_id",
        "repo",
        "base_commit",
        "problem_statement",
        "image",
    }
    assert read_json(experiment / "trusted/dataset.json")[0]["patch"] == "SECRET_GOLD"
    with pytest.raises(ValueError, match="absent"):
        select_rows([task()], {"instances": [{"instance_id": "owner__repo-99"}]})
    with pytest.raises(ValueError, match="duplicate"):
        select_rows([task(), task()], None)
    with pytest.raises(ValueError, match="tier"):
        select_rows(
            [task()],
            {
                "instances": [
                    {
                        "instance_id": "owner__repo-1",
                        "tier": "hard",
                        "hist_resolved": "70/84",
                    }
                ]
            },
        )


def test_shipped_selection_matches_user_manifest():
    selection = read_json(PACKAGE / "lite30.json")
    entries = selection["instances"]
    assert len(entries) == len({row["instance_id"] for row in entries}) == 30
    # Official split membership is checked at prepare, not from this fixture.
    from collections import Counter

    assert Counter(row["tier"] for row in entries) == {
        "easy": 8,
        "medium": 12,
        "hard": 8,
        "unsolved": 2,
    }


def test_frozen_config_and_patch_tampering_fail_closed(experiment):
    assert load_experiment(experiment)["selection"]["count"] == 2
    save_attempt(experiment, "mini", "owner__repo-1", b"patch")
    path = experiment / "runs/mini/owner__repo-1/attempt-fixture/model.patch"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="patch changed"):
        runner.export(experiment, "mini")
    write_json(experiment / "config.json", {})
    with pytest.raises(ValueError, match="Frozen file changed"):
        load_experiment(experiment)


def test_export_preserves_denominator_and_report_unknowns(experiment):
    save_attempt(experiment, "mini", "owner__repo-1")
    rows = [
        json.loads(line)
        for line in runner.export(experiment, "mini").read_text().splitlines()
    ]
    assert len(rows) == 2 and all(row["model_patch"] == "" for row in rows)
    report = build_report(experiment)
    assert report["selected"] == 2
    assert report["variants"]["mini"]["pending"] == 1
    assert report["variants"]["mini"]["known_outcomes"] == 0
    assert report["paired"]["n"] == 0
    assert report["per_instance"][0]["mini"] is None


def test_paired_statistics_and_real_official_report_shape(tmp_path):
    report_dir = tmp_path / "logs/evaluation/unique/model/owner__repo-1"
    write_json(report_dir / "report.json", {"owner__repo-1": {"resolved": True}})
    write_json(
        tmp_path / "infra/report.json",
        {
            "owner__repo-2": {"resolved": False, "infra_failure": True},
        },
    )
    assert official_reports(
        tmp_path, {"owner__repo-1", "owner__repo-2", "missing"}
    ) == {"owner__repo-1": True}
    stats = paired_statistics([(False, True)] * 6 + [(True, True)] * 2)
    assert stats["muharness_only"] == 6 and stats["both"] == 2
    assert stats["mcnemar_exact_p"] == pytest.approx(0.03125)
    assert stats["difference"] == 0.75
    assert stats["bootstrap_ci95"][0] <= 0.75 <= stats["bootstrap_ci95"][1]


def test_meter_counts_failed_attempts_and_redacts_secret(tmp_path):
    meter = Meter(tmp_path, 2, "private-key")
    meter.start({"error": "private-key"})  # failed call still consumes one attempt
    meter.start({"request": "second"})
    meter.finish({}, {"input_tokens": 10, "output_tokens": 3})
    with pytest.raises(CallLimitExceeded):
        meter.start({})
    assert read_json(tmp_path / "usage.json")["usage_missing_calls"] == 1
    assert "private-key" not in (tmp_path / "trajectory.jsonl").read_text()


def test_controller_timeout_exports_patch_and_does_not_replay(experiment, monkeypatch):
    calls = []

    class Container:
        name = "mh-swe-" + "a" * 20
        workspace = experiment / "fixture-workspace"

        def quiesce(self):
            calls.append("quiesce")

        def export_patch(self, base):
            calls.append("export")
            return b"raw", b"partial patch"

        def close(self):
            calls.append("close")

    class Process:
        pid = os.getpid()
        returncode = -1
        killed = False

        def __init__(self, *args, **kwargs):
            calls.append("model-worker")

        def wait(self, timeout):
            if not self.killed:
                raise subprocess.TimeoutExpired("worker", timeout)

        def kill(self):
            self.killed = True

    monkeypatch.setattr(runner, "pin_image", lambda *a, **kw: {"id": "image"})
    monkeypatch.setattr(TaskContainer, "create", lambda *a, **kw: Container())
    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    # Exercise controller timeout handling without the progress poller's real deadline.
    monkeypatch.setattr(
        runner, "wait_with_progress", lambda process, progress, timeout: process.wait(timeout)
    )
    result = runner.run_one(
        experiment,
        public_task(task()),
        "mini",
        load_experiment(experiment),
        sys.executable,
    )
    assert result["stop_reason"] == "task_timeout" and result["status"] == "completed"
    assert calls == ["model-worker", "quiesce", "export", "close"]
    predictions = runner.export(experiment, "mini")
    assert (
        json.loads(predictions.read_text().splitlines()[0])["model_patch"]
        == "partial patch"
    )
    assert (
        runner.completed_result(
            experiment, "mini", "owner__repo-1", result["fingerprint"]
        )
        is not None
    )


def test_recovery_seals_interruption_without_another_model_call(experiment):
    from tests.eval.swebench.recovery import process_alive, recover

    assert process_alive(os.getpid())
    directory = experiment / "runs/mini/owner__repo-1/attempt-orphan"
    write_json(
        directory / "status.json",
        {
            "instance_id": "owner__repo-1",
            "variant": "mini",
            "attempt": "attempt-orphan",
            "fingerprint": read_json(experiment / "manifest.json")["fingerprint"],
            "started_at": "2026-10-15T00:00:00+00:00",
            "patch_sha256": None,
        },
    )
    result = recover(experiment)
    assert len(result) == 1 and result[0]["stop_reason"] == "controller_interrupted"
    assert result[0]["status"] == "infra_error"
    assert recover(experiment) == []


@pytest.fixture
def model_server():
    state = {"requests": [], "commands": ["true"], "error": False}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(request)
            if state["error"]:
                self.send_response(429)
                self.end_headers()
                self.wfile.write(
                    b'{"error":{"message":"rate limited","type":"rate_limit"}}'
                )
                return
            names = {entry["function"]["name"] for entry in request["tools"]}
            tool = (
                "run_shell_command"
                if "run_shell_command" in names
                else next(iter(names))
            )
            index = len(state["requests"]) - 1
            script = state["commands"][min(index, len(state["commands"]) - 1)]
            script += f" # call {index}"
            response = {
                "id": f"test-{index}",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "Inspecting the repository.",
                            "tool_calls": [
                                {
                                    "id": f"call-{index}",
                                    "type": "function",
                                    "function": {
                                        "name": tool,
                                        "arguments": json.dumps({"command": script}),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "total_tokens": 25,
                },
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class FakeContainer:
    def __init__(self, workspace):
        self.workspace = workspace
        self.timeout = 120
        self.commands = []

    def execute(self, script, **kwargs):
        self.commands.append(script)
        return {
            "returncode": 0,
            "output": "ok",
            "exception_info": "",
            "duration_seconds": 0.01,
        }


def test_muharness_uses_real_adapter_and_stops_at_physical_call_budget(
    tmp_path, model_server
):
    config = validate_config(
        {
            "provider": "openai",
            "model": "fixture",
            "base_url": model_server["url"],
            "model_calls": 14,
        }
    )
    request = {
        "config": config,
        "system_prompt": "Fix the issue using the shell.",
        "user_template": "{problem_statement}",
        "task": public_task(task()),
    }
    container = FakeContainer(tmp_path)
    result = asyncio.run(
        run_muharness(request, container, Meter(tmp_path, 14), "fixture-key")
    )
    assert len(model_server["requests"]) == 14
    assert len(container.commands) == 14
    assert result["stop_reason"] == "run_budget"
    assert read_json(tmp_path / "usage.json")["model_calls"] == 14
    assert all(
        row["temperature"] == 0 and row["max_tokens"] == 8192
        for row in model_server["requests"]
    )


@pytest.mark.parametrize("retries", [0, 2])
def test_muharness_retries_are_bounded_visible_and_refunded(
    tmp_path, model_server, retries
):
    model_server["error"] = True
    config = validate_config(
        {
            "provider": "openai",
            "model": "fixture",
            "base_url": model_server["url"],
            "transient_retries": retries,
            "retry_backoff_seconds": 0,
        }
    )
    result = asyncio.run(
        run_muharness(
            {
                "config": config,
                "system_prompt": "Fix it",
                "user_template": "{problem_statement}",
                "task": public_task(task()),
            },
            FakeContainer(tmp_path),
            Meter(tmp_path, 100),
            "fixture-key",
        )
    )
    # The SDK itself never retries; only the worker's counted, logged retries.
    assert len(model_server["requests"]) == retries + 1
    usage = read_json(tmp_path / "usage.json")
    assert usage["model_calls"] == retries + 1
    assert usage["transient_failures"] == retries
    assert result["stop_reason"] == "model_error"
    assert result["model_error_type"] == "RateLimitError"
    assert runner.worker_status(result) == "infra_error"


@pytest.mark.parametrize("mode", ["budget", "error", "submit"])
def test_mini_real_dependency_against_local_model(tmp_path, model_server, mode):
    python = os.environ.get("MUHARNESS_MINI_PYTHON")
    if not python:
        pytest.skip("Set MUHARNESS_MINI_PYTHON for real mini v2 integration")
    model_server["error"] = mode == "error"
    if mode == "submit":
        model_server["commands"] = ["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]
    config = validate_config(
        {
            "provider": "openai",
            "model": "fixture",
            "base_url": model_server["url"],
            "model_calls": 2,
            "retry_backoff_seconds": 0,
        }
    )
    info = runner.probe(python, "mini")
    request = {
        "config": config,
        "task": public_task(task()),
        "mini_config": info["config"],
    }
    write_json(tmp_path / "request.json", request)
    script = """
import json, sys
from pathlib import Path
from tests.eval.swebench.worker import run_mini
from tests.eval.swebench.meter import Meter
request=json.loads(Path(sys.argv[1]).read_text())
class Container:
 def execute(self, *args, **kwargs):
  text = 'ok'
  if 'COMPLETE_TASK' in args[0]: text = 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'
  return {'returncode':0, 'output':text, 'exception_info':'', 'duration_seconds':0}
meter = Meter(Path(sys.argv[1]).parent, 2, 'fixture-key')
result = run_mini(request, Container(), meter, 'fixture-key')
print(json.dumps(result))
"""
    result = subprocess.run(
        [python, "-c", script, str(tmp_path / "request.json")],
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    # error: 1 call + 2 counted transient retries, then the run ends as infra.
    expected = {"budget": 2, "submit": 1, "error": 3}[mode]
    assert len(model_server["requests"]) == expected
    usage = read_json(tmp_path / "usage.json")
    assert usage["input_tokens"] == {"budget": 40, "submit": 20, "error": 0}[mode]
    assert usage["usage_missing_calls"] == (3 if mode == "error" else 0)
    assert usage["transient_failures"] == (2 if mode == "error" else 0)
    worker_result = json.loads(result.stdout.decode().strip().splitlines()[-1])
    assert (runner.worker_status(worker_result) == "infra_error") is (mode == "error")
    assert b"fixture-key" not in (tmp_path / "mini-trajectory.json").read_bytes()


@pytest.mark.skipif(
    os.environ.get("MUHARNESS_SWEBENCH_DOCKER_TEST") != "1",
    reason="Opt-in persistent Docker/bind-mount test",
)
def test_real_container_patch_survives_commits_and_timeout(tmp_path):
    image = "mh-swe-fixture:" + uuid.uuid4().hex[:12]
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("""FROM muharness-sandbox:latest
USER 0:0
RUN mkdir /testbed && cd /testbed && git init \\
 && git config user.email fixture@invalid && git config user.name fixture \\
 && echo old > source.py && git add . && git commit -m base
""")
    container = None
    try:
        docker("build", "--network=none", "-t", image, str(tmp_path), timeout=120)
        base = (
            docker(
                "run",
                "--rm",
                "--network=none",
                "--entrypoint",
                "git",
                "-w",
                "/testbed",
                image,
                "rev-parse",
                "HEAD",
            )
            .stdout.decode()
            .strip()
        )
        container = TaskContainer.create(image, tmp_path / "workspace", base)
        assert container.execute("pwd")["output"].strip() == "/testbed"
        (container.workspace / "source.py").write_bytes(b"new\n")
        assert container.execute("cat source.py")["output"] == "new\n"
        assert container.execute("git add . && git commit -m agent")["returncode"] == 0
        container.execute("echo new > added.py; echo artifact > patch.txt")
        timed = container.execute("sleep 1; echo leaked > late.txt", timeout=0.1)
        assert timed["returncode"] in (124, 137)
        container.quiesce()
        raw, patch = container.export_patch(base)
        assert b"source.py" in patch and b"added.py" in patch
        assert b"patch.txt" in raw and b"patch.txt" not in patch
        assert not (container.workspace / "late.txt").exists()
        assert (
            docker(
                "inspect", "--format", "{{.HostConfig.NetworkMode}}", container.name
            ).stdout.strip()
            == b"none"
        )
        judge_python = os.environ.get("MUHARNESS_JUDGE_PYTHON")
        if judge_python:
            from tests.eval.swebench.grading import invoke_judge

            test_script = (
                "from pathlib import Path\n"
                "def test_fixed():\n"
                "    assert Path('/testbed/source.py').read_text().strip() == 'new'\n"
            )
            import shlex

            eval_script = "\n".join(
                [
                    "cd /testbed",
                    "printf %s "
                    + shlex.quote(test_script)
                    + " > /tmp/test_mh_fixture.py",
                    "cd /tmp",
                    "echo '>>>>> Start Test Output'",
                    "python -m pytest --rootdir=/tmp -q -rA /tmp/test_mh_fixture.py",
                    "echo '>>>>> End Test Output'",
                ]
            )
            dataset = [
                task(1)
                | {
                    "base_commit": base,
                    "image": image,
                    "version": "1",
                    "FAIL_TO_PASS": ["test_mh_fixture.py::test_fixed"],
                    "PASS_TO_PASS": [],
                    "eval_script": eval_script,
                    "eval_type": "pass_and_fail",
                    "log_parser": "parse_log_pytest",
                }
            ]
            dataset.append(dataset[0] | {"instance_id": "owner__repo-2"})
            root = tmp_path / "official-fixture"
            write_json(root / "trusted/dataset.json", dataset)
            write_json(root / "manifest.json", {"source": {"split": "dev"}})
            predictions = root / "predictions.jsonl"
            patches = [patch.decode(), patch.decode().replace("+new\n", "+wrong\n")]
            predictions.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "instance_id": row["instance_id"],
                            "model_name_or_path": "fixture",
                            "model_patch": prediction,
                        }
                    )
                    for row, prediction in zip(dataset, patches, strict=True)
                ),
                encoding="utf-8",
            )
            judged, code = invoke_judge(
                root,
                judge_python,
                predictions,
                label="fixture",
                ids=[row["instance_id"] for row in dataset],
                timeout=60,
            )
            assert code == 0, (judged / "judge.log").read_text(errors="replace")
            assert official_reports(judged, {r["instance_id"] for r in dataset}) == {
                "owner__repo-1": True,
                "owner__repo-2": False,
            }
    finally:
        if container:
            container.close()
        docker("image", "rm", image)


class RateLimitError(Exception):
    pass


def test_recovered_api_error_does_not_discard_a_finished_run(tmp_path):
    from tests.eval.swebench.meter import fatal_error_type

    meter = Meter(tmp_path, 100)
    meter.start({})
    meter.error(RateLimitError())  # e.g. a failed context summary
    meter.start({})
    meter.finish({}, {"input_tokens": 1, "output_tokens": 1})
    assert fatal_error_type(meter.stats) is None
    finished = {"stop_reason": "final_answer", "model_error_type": None}
    assert runner.worker_status(finished) == "completed"
    # Budget exhaustion after an earlier API error is an agent outcome.
    small = Meter(tmp_path / "small", 1)
    (tmp_path / "small").mkdir()
    small.start({})
    small.error(RateLimitError(), will_retry=True)
    small.start({}, retry=1)
    small.finish({}, None)
    with pytest.raises(CallLimitExceeded):
        small.start({})
    assert small.stats["model_calls"] == 2  # retry was refunded, not free
    assert fatal_error_type(small.stats) == "CallLimitExceeded"


def test_apply_failure_and_eval_timeout_are_unresolved_not_unknown(tmp_path):
    from tests.eval.swebench.grading import official_outcomes

    base = tmp_path / "logs/evaluation/run/mh-mini"
    for name, log in (
        ("owner__repo-1", ">>>>> Patch Apply Failed:\nerror"),
        ("owner__repo-2", "Test timed out after 1800 seconds."),
        ("owner__repo-3", "Error response from daemon"),
    ):
        (base / name).mkdir(parents=True)
        (base / name / "run_instance.log").write_text(log)
    write_json(
        base / "owner__repo-4" / "report.json",
        {
            "owner__repo-4": {
                "resolved": False,
                "patch_successfully_applied": True,
                "tests_status": {
                    "FAIL_TO_PASS": {"success": ["a"], "failure": ["b"]},
                    "PASS_TO_PASS": {"success": ["c"], "failure": []},
                },
            }
        },
    )
    ids = {f"owner__repo-{i}" for i in range(1, 5)}
    outcomes = official_outcomes(tmp_path, ids)
    assert outcomes["owner__repo-1"]["source"] == "patch_apply_failed"
    assert outcomes["owner__repo-2"]["source"] == "eval_timeout"
    assert not outcomes["owner__repo-1"]["resolved"]
    assert "owner__repo-3" not in outcomes  # Docker failure stays unknown
    assert outcomes["owner__repo-4"]["fail_to_pass"] == [1, 2]
    assert outcomes["owner__repo-4"]["pass_to_pass"] == [1, 1]


def test_git_history_past_base_is_removed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return (
            subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
            .stdout.decode()
            .strip()
        )

    # Production runs these commands inside the Linux task container. On a
    # Windows host, "bash" may be WSL's, which cannot cd into a Windows temp
    # path, so use Git for Windows' bash or skip.
    bash = "bash"
    if os.name == "nt":
        import shutil
        from pathlib import Path

        git_exe = shutil.which("git")
        candidates = (
            [Path(git_exe).resolve().parents[1] / "bin" / "bash.exe"] if git_exe else []
        ) + [Path(r"C:\Program Files\Git\bin\bash.exe")]
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            pytest.skip("Git for Windows bash not found")
        bash = str(found)

    class Local(TaskContainer):
        def control(self, script, *, timeout=120):
            return subprocess.run(
                [bash, "-c", script],
                cwd=self.workspace,
                check=True,
                capture_output=True,
                timeout=timeout,
            ).stdout

    git("init", "-q", "-b", "main")
    git("config", "user.email", "x@invalid")
    git("config", "user.name", "x")
    (repo / "f.py").write_text("old\n")
    git("add", ".")
    git("commit", "-qm", "old")
    git("tag", "v1")
    (repo / "f.py").write_text("base\n")
    git("commit", "-qam", "base")
    base = git("rev-parse", "HEAD")
    (repo / "f.py").write_text("fix\n")
    git("commit", "-qam", "upstream fix")
    fix = git("rev-parse", "HEAD")
    git("tag", "v2")
    git("update-ref", "refs/remotes/origin/main", fix)
    git("reset", "-q", "--hard", base)
    audit = Local("fixture", repo).sanitize_history(base)
    assert audit["sanitized"]
    assert audit["after"]["reachable_commits"] == audit["after"]["head_commits"]
    assert "refs/tags/v1" in audit["after"]["refs"]
    assert "refs/tags/v2" not in audit["after"]["refs"]
    assert git("rev-parse", "HEAD") == base
    assert subprocess.run(["git", "cat-file", "-e", fix], cwd=repo).returncode != 0


def test_retry_infra_reruns_only_infra_attempts(experiment, monkeypatch):
    calls = []

    def fake_run_one(root, task, variant, manifest, python, **kwargs):
        instance = task["instance_id"]
        directory = root / "runs" / variant / instance
        attempt = f"attempt-{len(calls)}"
        (directory / attempt).mkdir(parents=True)
        status = (
            "infra_error"
            if (variant, instance) == ("muharness", "owner__repo-2")
            else "completed"
        )
        result = {
            "instance_id": instance,
            "attempt": attempt,
            "fingerprint": manifest["fingerprint"],
            "status": status,
            "stop_reason": "x",
            "patch_sha256": None,
            "previous_infra_attempts": kwargs.get("previous_attempts") or [],
        }
        write_json(directory / "result.json", result)
        calls.append((variant, instance))
        return result

    monkeypatch.setattr(runner, "run_one", fake_run_one)
    monkeypatch.setattr(runner, "probe", lambda *a, **k: {})
    monkeypatch.setattr(runner, "python_executable", lambda value: value)
    monkeypatch.setattr("tests.eval.swebench.worker.credential", lambda config: "k")
    manifest = load_experiment(experiment)
    monkeypatch.setattr(
        runner,
        "load_experiment",
        lambda *a, **k: manifest | {"environments": {"mini": {}, "muharness": {}}},
    )
    runner.run(experiment, ["mini", "muharness"], mini_python="p")
    # Second task runs MuHarness first: order alternates by frozen position.
    assert calls == [
        ("mini", "owner__repo-1"),
        ("muharness", "owner__repo-1"),
        ("muharness", "owner__repo-2"),
        ("mini", "owner__repo-2"),
    ]
    runner.run(experiment, ["mini", "muharness"], mini_python="p", resume=True)
    assert len(calls) == 4
    for _ in range(4):
        runner.run(experiment, ["mini", "muharness"], mini_python="p", retry_infra=True)
    assert calls[4:] == [("muharness", "owner__repo-2")] * runner.MAX_INFRA_RETRIES
    directory = experiment / "runs/muharness/owner__repo-2"
    assert len(list(directory.glob("result.infra-*.json"))) == runner.MAX_INFRA_RETRIES
