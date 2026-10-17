from __future__ import annotations

import shlex
import subprocess

import pytest

from tests.eval.swebench.container import TaskContainer


@pytest.fixture
def repository(tmp_path):
    def git(*args):
        return (
            subprocess.run(
                ["git", *args], cwd=tmp_path, check=True, capture_output=True
            )
            .stdout.decode()
            .strip()
        )

    git("init", "-q", "-b", "main")
    git("config", "user.email", "fixture@invalid")
    git("config", "user.name", "fixture")
    (tmp_path / "source.py").write_bytes(b"original\n")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")

    class Local(TaskContainer):
        def control(self, script, *, timeout=120):
            return subprocess.run(
                shlex.split(script),
                cwd=self.workspace,
                check=True,
                capture_output=True,
                timeout=timeout,
            ).stdout

    return Local("fixture", tmp_path), git, base


def test_official_empty_setup_commit_is_normalized_without_source_changes(repository):
    container, git, base = repository
    git("commit", "--allow-empty", "-qm", "SWE-bench")
    setup_head = git("rev-parse", "HEAD")
    audit = container.align_base_commit(base)
    assert audit["image_head"] == setup_head
    assert audit["normalized_setup_commit"] is True
    assert git("rev-parse", "HEAD") == base
    assert git("status", "--porcelain") == ""
    assert (container.workspace / "source.py").read_bytes() == b"original\n"


def test_changed_setup_tree_is_used_as_patch_base_without_rewinding(repository):
    container, git, base = repository
    (container.workspace / "source.py").write_bytes(b"different\n")
    git("commit", "-qam", "real source change")
    changed = git("rev-parse", "HEAD")
    audit = container.align_base_commit(base)
    assert audit["image_head"] == changed
    assert audit["base_commit"] == base
    assert audit["patch_base"] == changed
    assert audit["normalized_setup_commit"] is True
    assert audit["setup_commit_files"] == ["source.py"]
    assert git("rev-parse", "HEAD") == changed
    assert (container.workspace / "source.py").read_bytes() == b"different\n"


def test_unrelated_commit_with_identical_tree_is_rejected(repository):
    container, git, base = repository
    tree = git("rev-parse", "HEAD^{tree}")
    orphan = git("commit-tree", tree, "-m", "unrelated root")
    git("checkout", "--detach", orphan)
    with pytest.raises(ValueError, match="does not descend"):
        container.align_base_commit(base)
    assert git("rev-parse", "HEAD") == orphan


def test_matching_base_requires_no_normalization(repository):
    container, git, base = repository
    assert container.align_base_commit(base)["normalized_setup_commit"] is False
    assert git("rev-parse", "HEAD") == base


def test_missing_base_is_rejected(repository):
    container, _, _ = repository
    with pytest.raises(ValueError, match="absent"):
        container.align_base_commit("0" * 40)
