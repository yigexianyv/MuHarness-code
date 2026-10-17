from __future__ import annotations

import json
import os
import posixpath
import shlex
import subprocess
import uuid
from contextlib import suppress
from pathlib import Path
from time import monotonic

from .common import command, read_json, write_json

SHELL_ENV = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
    "BASH_ENV": "/root/.bashrc",
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": "safe.directory",
    "GIT_CONFIG_VALUE_0": "/testbed",
}


def docker(*args: str, timeout: float = 60, **kwargs) -> subprocess.CompletedProcess:
    return command(["docker", *args], timeout=timeout, **kwargs)


def pin_image(image: str, lock: Path, *, pull: bool = True) -> dict:
    if lock.exists():
        pinned = read_json(lock)
        if pinned["source"] != image:
            raise ValueError("Image source differs from the experiment lock")
        try:
            docker("image", "inspect", pinned["id"])
        except subprocess.CalledProcessError:
            if not pull or not pinned["digest"]:
                raise RuntimeError(
                    "Pinned image missing; restore it or prepare a new experiment"
                ) from None
            docker("pull", "--platform", "linux/amd64", pinned["digest"], timeout=1800)
        current = json.loads(docker("image", "inspect", pinned["id"]).stdout)[0]
        if current["Id"] != pinned["id"]:
            raise RuntimeError("Pinned image ID changed")
        return pinned
    try:
        info = docker("image", "inspect", image)
    except subprocess.CalledProcessError:
        if not pull:
            raise
        docker("pull", "--platform", "linux/amd64", image, timeout=1800)
        info = docker("image", "inspect", image)
    item = json.loads(info.stdout)[0]
    if item["Os"] != "linux" or item["Architecture"] != "amd64":
        raise ValueError("SWE-bench comparison requires a linux/amd64 task image")
    pinned = {
        "source": image,
        "id": item["Id"],
        "digest": next(iter(item.get("RepoDigests") or []), None),
    }
    write_json(lock, pinned)
    return pinned


class TaskContainer:
    """One persistent, offline container plus a bind mount shared with file tools.

    Controller metadata/logs are outside the mount. Only this class owns the
    randomly named container; it never removes unrelated containers or images.
    """

    def __init__(self, name: str, workspace: Path, timeout: float = 120):
        self.name = name
        self.workspace = workspace.resolve()
        self.timeout = timeout
        self.git_audit: dict | None = None
        # Commit the agent starts from and the patch is exported against.
        self.patch_base: str | None = None

    @classmethod
    def create(
        cls, image: str, workspace: Path, base_commit: str, timeout: float = 120
    ):
        workspace = workspace.resolve()
        if os.name == "posix" and os.geteuid() != 0:
            raise RuntimeError(
                "Run the evaluation controller as root in a disposable Linux/WSL "
                "environment; task images use root and share files with host tools"
            )
        workspace.mkdir(parents=True, exist_ok=False)
        name = f"mh-swe-{uuid.uuid4().hex[:20]}"
        seed = name + "-seed"
        container = cls(name, workspace, timeout)
        try:
            # Copy .git too, without running the image's entrypoint.
            docker(
                "create",
                "--name",
                seed,
                "--network=none",
                "--label",
                "mh.swebench=1",
                "--entrypoint",
                "/bin/true",
                image,
            )
            docker("cp", f"{seed}:/testbed/.", str(workspace), timeout=300)
            docker(
                "run",
                "-d",
                "--name",
                name,
                "--network=none",
                "--label",
                "mh.swebench=1",
                "--platform",
                "linux/amd64",
                "--user",
                "0:0",
                "--entrypoint",
                "/bin/sleep",
                "--mount",
                f"type=bind,source={workspace},target=/testbed",
                image,
                "infinity",
            )
            # Root in the image matches mini's official BASH_ENV. Give host file tools
            # ownership of the mount, without modifying file executable bits.
            if os.name == "posix":
                docker(
                    "exec",
                    name,
                    "chown",
                    "-R",
                    f"{os.getuid()}:{os.getgid()}",
                    "/testbed",
                    timeout=300,
                )
            container.control(
                "command -v timeout >/dev/null && command -v git >/dev/null"
            )
            if os.name == "nt":
                # Docker Desktop bind mounts don't preserve POSIX executable bits.
                container.control("git config core.fileMode false")
            if container.control("git status --porcelain --untracked-files=no").strip():
                raise ValueError(
                    "Task image has tracked modifications before the agent starts"
                )
            baseline = container.align_base_commit(base_commit)
            container.patch_base = baseline["patch_base"]
            container.git_audit = baseline | container.sanitize_history(
                container.patch_base
            )
            return container
        except BaseException:
            container.close()
            raise
        finally:
            with suppress(Exception):
                docker("rm", "-f", seed)

    def align_base_commit(self, base_commit: str) -> dict:
        """Pick the commit the agent starts from and the patch is diffed against.

        Official images may add a setup commit on top of the dataset base:
        - same tree as base (empty commit): check out the base itself;
        - different tree (the image build changed tracked files, as in some
          django images): keep the image HEAD. The official harness applies the
          prediction to exactly this state, so a patch against base_commit
          would carry the setup changes and fail to apply. The setup commit must
          descend from base_commit; the files it changed are recorded.
        """
        head = self.control("git rev-parse HEAD").decode().strip()
        base = shlex.quote(base_commit)
        try:
            base_tree = (
                self.control(
                    f"git rev-parse --verify {shlex.quote(base_commit + '^{tree}')}"
                )
                .decode()
                .strip()
            )
        except subprocess.CalledProcessError as exc:
            raise ValueError(
                f"Dataset base_commit {base_commit} is absent from the task image"
            ) from exc
        patch_base, setup_files = base_commit, []
        if head != base_commit:
            try:
                self.control(f"git merge-base --is-ancestor {base} HEAD")
            except subprocess.CalledProcessError as exc:
                raise ValueError(
                    f"Image HEAD {head} does not descend from dataset base_commit "
                    f"{base_commit}"
                ) from exc
            image_tree = self.control("git rev-parse HEAD^{tree}").decode().strip()
            if image_tree == base_tree:
                # No force/reset: preserve the identical working tree and let Git
                # refuse any unexpected local changes rather than discarding them.
                self.control(
                    f"git -c advice.detachedHead=false checkout --detach {base}"
                )
            else:
                patch_base = head
                setup_files = (
                    self.control(f"git diff --name-only {base} HEAD")
                    .decode()
                    .splitlines()
                )
        return {
            "image_head": head,
            "base_commit": base_commit,
            "base_tree": base_tree,
            "normalized_setup_commit": head != base_commit,
            "patch_base": patch_base,
            "setup_commit_files": setup_files,
        }

    def history_counts(self) -> dict:
        def count(args: str) -> int:
            return int(self.control(f"git rev-list --count {args}").decode().strip())

        refs = self.control("git for-each-ref --format='%(refname)'").decode()
        return {
            "head_commits": count("HEAD"),
            "reachable_commits": count("--all --reflog"),
            "refs": sorted(line for line in refs.splitlines() if line),
        }

    def sanitize_history(self, base_commit: str) -> dict:
        """Remove anything that could reveal commits after base_commit.

        `base_commit` here is the patch base (dataset base, or the image's setup
        commit on top of it); only its own ancestry is kept.

        Task images are clones: remotes, other branches, tags and reflogs can still
        point at the upstream fix (`git log --all`). Both variants get exactly the
        same treatment. HEAD (== base_commit) and its ancestry are untouched, so the
        working tree and patch export are unaffected.
        """
        before = self.history_counts()
        if before["reachable_commits"] != before["head_commits"]:
            branch = self.control("git symbolic-ref -q HEAD || true").decode().strip()
            script = " && ".join(
                [
                    'for r in $(git remote); do git remote remove "$r"; done',
                    # Keep the checked-out branch and refs into HEAD's own history
                    # (old release tags matter to setuptools_scm/versioneer);
                    # delete every ref that reaches past base_commit.
                    # List first, then delete: never mutate refs while iterating.
                    "refs=$(git for-each-ref --format='%(refname)')",
                    "for ref in $refs; do "
                    f'[ "$ref" = {shlex.quote(branch)} ] && continue; '
                    'git merge-base --is-ancestor "$ref" HEAD 2>/dev/null '
                    '|| git update-ref -d "$ref"; '
                    "done",
                    "git stash clear",
                    "rm -f .git/FETCH_HEAD .git/ORIG_HEAD .git/MERGE_HEAD",
                    "git reflog expire --expire=now --all",
                    "git gc --prune=now --quiet",
                ]
            )
            self.control(script, timeout=900)
        after = self.history_counts()
        head = self.control("git rev-parse HEAD").decode().strip()
        if head != base_commit or after["reachable_commits"] != after["head_commits"]:
            raise RuntimeError(
                "Could not remove post-base git history from the task workspace"
            )
        return {"before": before, "after": after, "sanitized": before != after}

    def _exec_args(self, cwd: str) -> list[str]:
        args = ["exec", "-w", cwd]
        for key, value in SHELL_ENV.items():
            args += ["-e", f"{key}={value}"]
        # Environment variables/credentials from the controller are never forwarded.
        return [*args, self.name]

    def control(self, script: str, *, timeout: float = 120) -> bytes:
        return docker(
            *self._exec_args("/testbed"), "bash", "-c", script, timeout=timeout
        ).stdout

    def execute(
        self, script: str, *, cwd: str = "/testbed", timeout: float | None = None
    ) -> dict:
        if not isinstance(script, str) or not script.strip():
            raise ValueError("command must be a nonempty string")
        cwd = posixpath.normpath(posixpath.join("/testbed", cwd))
        if cwd != "/testbed" and not cwd.startswith("/testbed/"):
            raise ValueError("working_directory must remain within /testbed")
        limit = self.timeout if timeout is None else min(float(timeout), self.timeout)
        if limit <= 0:
            raise ValueError("timeout must be positive")
        start = monotonic()
        # timeout runs INSIDE the container and kills the command process group;
        # killing just the docker client leaves tests running in the background.
        args = [
            "docker",
            *self._exec_args(cwd),
            "timeout",
            "--signal=TERM",
            "--kill-after=5",
            str(limit),
            "bash",
            "-c",
            script,
        ]
        try:
            result = subprocess.run(args, capture_output=True, timeout=limit + 15)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "Docker exec did not return after in-container timeout"
            ) from exc
        output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
        elapsed = monotonic() - start
        # 124: TERM at the limit. 137: KILL 5s later, but also an OOM kill or a
        # command that kills itself; only call it a timeout if the limit elapsed.
        timed_out = result.returncode == 124 or (
            result.returncode == 137 and elapsed >= limit
        )
        return {
            "output": output,
            "returncode": result.returncode,
            "exception_info": "Command timed out" if timed_out else "",
            "duration_seconds": elapsed,
        }

    def read_only_twin(self) -> TaskContainer:
        """Second container of the same image with the workspace mounted read-only.

        Used for MEA's read-only roles (Manager/Auditor): they can run tests and
        scripts, but any write to /testbed fails instead of altering the patch.
        """
        image = (
            docker("inspect", "-f", "{{.Config.Image}}", self.name)
            .stdout.decode()
            .strip()
        )
        twin = TaskContainer(self.name + "-ro", self.workspace, self.timeout)
        twin.patch_base = self.patch_base
        docker(
            "run",
            "-d",
            "--name",
            twin.name,
            "--network=none",
            "--label",
            "mh.swebench=1",
            "--platform",
            "linux/amd64",
            "--user",
            "0:0",
            "--entrypoint",
            "/bin/sleep",
            "--mount",
            f"type=bind,source={self.workspace},target=/testbed,readonly",
            image,
            "infinity",
        )
        return twin

    def quiesce(self) -> None:
        # Also kills daemonized/background commands and commands orphaned by a
        # worker timeout. The bind mount survives; exporting sees stable files.
        docker("restart", "--time", "1", self.name, timeout=30)

    def export_patch(self, base_commit: str) -> tuple[bytes, bytes]:
        index = f"/tmp/mh-export-{uuid.uuid4().hex}.index"
        prefix = f"GIT_INDEX_FILE={shlex.quote(index)} "
        try:
            self.control(prefix + f"git read-tree {shlex.quote(base_commit)}")
            self.control(prefix + "git add -A -- .")
            diff = (
                prefix
                + "git diff --cached --binary --full-index "
                + f"{shlex.quote(base_commit)} -- ."
            )
            raw = self.control(diff)
            # mini's submission artifact is not a source change. Exclude it only
            # when it wasn't already a tracked project file at the fixed base.
            tracked = self.control(
                f"git ls-tree --name-only {shlex.quote(base_commit)} -- patch.txt"
            )
            patch = (
                raw
                if tracked.strip()
                else self.control(diff + " ':(exclude)patch.txt'")
            )
            return raw, patch
        finally:
            with suppress(Exception):
                self.control(
                    f"rm -f -- {shlex.quote(index)} {shlex.quote(index + '.lock')}"
                )

    def close(self) -> None:
        with suppress(Exception):
            docker("rm", "-f", self.name, timeout=30)
        if not self.name.endswith("-ro"):
            # A killed MEA worker cannot remove its read-only twin itself.
            with suppress(Exception):
                docker("rm", "-f", self.name + "-ro", timeout=30)
