"""
tests/test_diff_tracker_git.py

Integration tests for the GitPython API surface core/diff_tracker.py depends on.

tests/test_diff_tracker.py deliberately stays pure-unit (no git). This module is
its counterpart: it pins the *library contract* that sync_repo() and
initial_index() rely on, so a GitPython upgrade can't silently change behaviour
underneath them.

Everything runs against throwaway repos built in tmp_path. A local bare repo
stands in for "origin" — no network access, no cloning from a remote host.

Covers:
  - git.Repo() on a working clone
  - repo.head.commit.hexsha
  - repo.remotes.origin.pull() and the info.ref / info.note attributes sync_repo prints
  - HEAD advancing after a pull
  - repo.commit(sha) + old.diff(new) across every change_type branch in sync_repo
    (A / M / D / R, with a_path / b_path)
  - InvalidGitRepositoryError / GitCommandError / BadName raised where diff_tracker catches them
  - the exception a *full* hexsha that is no longer in the repo actually raises
    (force-push / rebase upstream), and sync_repo()'s full-reindex fallback for it
"""

import os
import subprocess
from types import SimpleNamespace

import git
import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

# Fixed identity + timestamps: the suite must not depend on the machine's git
# config (CI has no global user.name/user.email) or on wall-clock time.
GIT_ENV = {
    "GIT_CONFIG_GLOBAL":  os.devnull,
    "GIT_CONFIG_SYSTEM":  os.devnull,
    "GIT_AUTHOR_NAME":    "Test Author",
    "GIT_AUTHOR_EMAIL":   "author@example.invalid",
    "GIT_AUTHOR_DATE":    "2020-01-01T00:00:00+00:00",
    "GIT_COMMITTER_NAME": "Test Committer",
    "GIT_COMMITTER_EMAIL": "committer@example.invalid",
    "GIT_COMMITTER_DATE": "2020-01-01T00:00:00+00:00",
}


def _git(cwd, *args):
    """Run a git command in cwd, failing loudly if it errors."""
    subprocess.run(
        ("git",) + args,
        cwd=str(cwd),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _commit_all(work, message):
    _git(work, "add", "-A")
    _git(work, "commit", "-m", message)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def hermetic_git_env(monkeypatch):
    """Isolate git from global/system config and pin identity + dates."""
    for key, value in GIT_ENV.items():
        monkeypatch.setenv(key, value)


@pytest.fixture()
def synced(tmp_path, hermetic_git_env):
    """
    Build a local origin + working clone, then land a second commit upstream
    that exercises all four change types, and pull it into the clone.

    Returns a namespace with the clone's Repo, the SHAs either side of the pull,
    and the FetchInfo list the pull produced.
    """
    upstream = tmp_path / "upstream.git"
    work     = tmp_path / "work"
    clone    = tmp_path / "clone"

    # Upstream bare repo standing in for 'origin'.
    _git(tmp_path, "init", "--bare", "--initial-branch=main", str(upstream))

    # Authoring repo, first commit.
    _git(tmp_path, "init", "--initial-branch=main", str(work))
    _git(work, "config", "user.email", "author@example.invalid")
    _git(work, "config", "user.name", "Test Author")
    _git(work, "config", "commit.gpgsign", "false")

    (work / "modified.py").write_text("def a():\n    return 1\n")
    (work / "deleted.py").write_text("obsolete = True\n")
    # Padded so git's rename detection scores the move as R, not A+D.
    (work / "old_name.py").write_text("# padding line\n" * 40)
    _commit_all(work, "first")
    _git(work, "remote", "add", "origin", str(upstream))
    _git(work, "push", "-u", "origin", "main")

    # The 'indexed repo' that diff_tracker would operate on.
    _git(tmp_path, "clone", str(upstream), str(clone))
    repo = git.Repo(str(clone))
    sha_before = repo.head.commit.hexsha

    # Second commit upstream: one of each change type.
    (work / "modified.py").write_text("def a():\n    return 1\n\ndef b():\n    return 2\n")
    (work / "added.py").write_text("fresh = True\n")
    (work / "deleted.py").unlink()
    (work / "old_name.py").rename(work / "new_name.py")
    _commit_all(work, "second")
    _git(work, "push", "origin", "main")

    pull_info = repo.remotes.origin.pull()

    return SimpleNamespace(
        repo=repo,
        clone=clone,
        upstream=upstream,
        sha_before=sha_before,
        sha_after=repo.head.commit.hexsha,
        pull_info=pull_info,
        tmp_path=tmp_path,
    )


# ── git.Repo() ─────────────────────────────────────────────────────────────────

def test_repo_opens_working_clone(synced):
    assert isinstance(synced.repo, git.Repo)
    assert not synced.repo.bare


# ── repo.head.commit.hexsha ───────────────────────────────────────────────────

def test_head_commit_hexsha_is_full_sha(synced):
    """initial_index() and sync_repo() both persist this value as the sync baseline."""
    sha = synced.sha_after
    assert isinstance(sha, str)
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)


# ── repo.remotes.origin.pull() ────────────────────────────────────────────────

def test_pull_returns_fetch_info(synced):
    assert len(synced.pull_info) >= 1
    assert all(isinstance(i, git.remote.FetchInfo) for i in synced.pull_info)


def test_pull_info_exposes_ref_and_note(synced):
    """sync_repo() prints `info.ref` and `info.note` for every FetchInfo."""
    for info in synced.pull_info:
        assert info.ref is not None
        assert str(info.ref)                 # renders without raising
        assert isinstance(info.note, str)    # `info.note or 'up to date'` stays safe


def test_head_advances_after_pull(synced):
    assert synced.sha_before != synced.sha_after


# ── repo.commit(sha) ──────────────────────────────────────────────────────────

def test_commit_lookup_by_sha_roundtrips(synced):
    assert synced.repo.commit(synced.sha_before).hexsha == synced.sha_before
    assert synced.repo.commit(synced.sha_after).hexsha == synced.sha_after


# ── old_commit.diff(new_commit) ───────────────────────────────────────────────

@pytest.fixture()
def diff_by_path(synced):
    """The diff sync_repo() iterates, keyed by the path it would act on."""
    old = synced.repo.commit(synced.sha_before)
    new = synced.repo.commit(synced.sha_after)
    diff = old.diff(new)
    return {(d.b_path or d.a_path): d for d in diff}


def test_diff_reports_every_change_type(diff_by_path):
    """A / M / D / R are the four branches sync_repo() dispatches on."""
    assert sorted(d.change_type for d in diff_by_path.values()) == ["A", "D", "M", "R"]


def test_added_file_carries_b_path(diff_by_path):
    d = diff_by_path["added.py"]
    assert d.change_type == "A"
    assert d.b_path == "added.py"


def test_modified_file_carries_b_path(diff_by_path):
    d = diff_by_path["modified.py"]
    assert d.change_type == "M"
    assert d.b_path == "modified.py"


def test_deleted_file_carries_a_path(diff_by_path):
    """sync_repo() reads a_path on 'D' to delete the right chunks from Milvus."""
    d = diff_by_path["deleted.py"]
    assert d.change_type == "D"
    assert d.a_path == "deleted.py"


def test_renamed_file_carries_both_paths(diff_by_path):
    """sync_repo() reuses embeddings across a rename, so both sides must be present."""
    d = diff_by_path["new_name.py"]
    assert d.change_type == "R"
    assert d.a_path == "old_name.py"
    assert d.b_path == "new_name.py"


# ── Exceptions diff_tracker catches ───────────────────────────────────────────

def test_invalid_git_repository_error_on_non_repo(tmp_path, hermetic_git_env):
    """Both initial_index() and sync_repo() convert this into a ValueError."""
    plain_dir = tmp_path / "not_a_repo"
    plain_dir.mkdir()
    with pytest.raises(git.InvalidGitRepositoryError):
        git.Repo(str(plain_dir))


def test_bad_name_on_unresolvable_revision(synced):
    """sync_repo() falls back to a full re-index when the old revision won't resolve."""
    with pytest.raises(git.BadName):
        synced.repo.commit("no_such_revision")


def test_git_command_error_on_failed_pull(synced):
    """sync_repo() catches GitCommandError around the pull and re-raises after logging."""
    missing = synced.tmp_path / "vanished.git"
    synced.repo.remotes.origin.set_url(str(missing))
    with pytest.raises(git.GitCommandError):
        synced.repo.remotes.origin.pull()


# ── Unreachable baseline SHA (upstream force-push / rebase) ───────────────────

@pytest.fixture()
def rewritten(tmp_path, hermetic_git_env):
    """
    Simulate the state code-intel lands in after an upstream force-push.

    A clone is taken and its HEAD recorded as the sync baseline. Upstream then
    rewrites that commit away and force-pushes, and the clone is refreshed from
    the new history — so the recorded baseline SHA is a well-formed 40-char
    hexsha that this repository simply does not contain any more.
    """
    upstream = tmp_path / "upstream.git"
    work     = tmp_path / "work"
    clone    = tmp_path / "clone"

    _git(tmp_path, "init", "--bare", "--initial-branch=main", str(upstream))
    _git(tmp_path, "init", "--initial-branch=main", str(work))
    _git(work, "config", "user.email", "author@example.invalid")
    _git(work, "config", "user.name", "Test Author")
    _git(work, "config", "commit.gpgsign", "false")

    (work / "base.py").write_text("base = True\n")
    _commit_all(work, "base")
    (work / "doomed.py").write_text("doomed = True\n")
    _commit_all(work, "doomed")
    _git(work, "remote", "add", "origin", str(upstream))
    _git(work, "push", "-u", "origin", "main")

    _git(tmp_path, "clone", str(upstream), str(clone))
    baseline_sha = git.Repo(str(clone)).head.commit.hexsha

    # Upstream rewrites history: drop the tip commit, land a different one.
    _git(work, "reset", "--hard", "HEAD~1")
    (work / "replacement.py").write_text("replacement = True\n")
    _commit_all(work, "replacement")
    _git(work, "push", "--force", "origin", "main")

    # The indexed checkout follows the rewrite and drops the orphaned object,
    # exactly as a re-clone or a `git gc` would.
    _git(clone, "fetch", "origin")
    _git(clone, "reset", "--hard", "origin/main")
    _git(clone, "reflog", "expire", "--expire=now", "--all")
    _git(clone, "gc", "--prune=now")

    repo = git.Repo(str(clone))
    return SimpleNamespace(
        repo=repo,
        clone=clone,
        upstream=upstream,
        baseline_sha=baseline_sha,
        head_sha=repo.head.commit.hexsha,
        tmp_path=tmp_path,
    )


def test_rewritten_baseline_sha_is_gone_but_well_formed(rewritten):
    """The recorded baseline is still a valid-looking SHA — it just isn't there."""
    assert len(rewritten.baseline_sha) == 40
    assert rewritten.baseline_sha != rewritten.head_sha
    probe = subprocess.run(
        ("git", "cat-file", "-e", f"{rewritten.baseline_sha}^{{commit}}"),
        cwd=str(rewritten.clone),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert probe.returncode != 0


def test_missing_full_hexsha_does_not_raise_bad_name(rewritten):
    """
    The reason `except git.BadName` never fired.

    BadName is for revisions git cannot *parse* into a name. A full hexsha parses
    fine; it fails later, when the object database is asked for the object — and
    that surfaces as a plain ValueError, which BadName does not cover.
    """
    with pytest.raises(Exception) as excinfo:
        rewritten.repo.commit(rewritten.baseline_sha)
    assert not isinstance(excinfo.value, git.BadName)
    assert not isinstance(excinfo.value, git.BadObject)
    assert isinstance(excinfo.value, ValueError)


def test_bad_name_is_not_a_value_error_subclass():
    """So a `except git.BadName` guard cannot incidentally catch the ValueError."""
    assert not issubclass(git.BadName, ValueError)
    assert not issubclass(git.BadObject, ValueError)


def test_lazily_bound_commit_fails_only_when_read(rewritten):
    """
    A Commit can be bound to an absent object without complaint; the failure is
    deferred until an attribute is read (ValueError) or .diff() shells out
    (GitCommandError). _resolve_commit() must force that read itself.
    """
    lazy = git.Commit(rewritten.repo, bytes.fromhex("00" * 19 + "01"))
    assert lazy.hexsha == "0" * 38 + "01"
    with pytest.raises(ValueError):
        assert lazy.tree


# ── sync_repo()'s full-reindex fallback ───────────────────────────────────────

@pytest.fixture()
def dt_on(rewritten, monkeypatch):
    """
    Point diff_tracker at the rewritten clone with the stale baseline recorded,
    and stub initial_index() so the fallback is observable without Milvus.
    """
    import core.diff_tracker as _dt

    monkeypatch.setattr(_dt, "REPOS_DIR", rewritten.tmp_path)
    monkeypatch.setattr(_dt, "SYNC_STATE_PATH", rewritten.tmp_path / "sync_state.json")
    _dt._update_synced_commit("clone", rewritten.baseline_sha)

    reindexed: list[str] = []
    monkeypatch.setattr(_dt, "initial_index", lambda name: reindexed.append(name))
    return SimpleNamespace(module=_dt, reindexed=reindexed)


def test_sync_falls_back_to_full_reindex_when_baseline_is_gone(dt_on):
    """The bug: this used to blow up instead of re-indexing."""
    dt_on.module.sync_repo("clone")
    assert dt_on.reindexed == ["clone"]


def test_resolve_commit_returns_none_for_missing_sha(dt_on, rewritten):
    assert dt_on.module._resolve_commit(rewritten.repo, rewritten.baseline_sha) is None


def test_resolve_commit_returns_none_for_unparseable_revision(dt_on, rewritten):
    """The original BadName case must keep working."""
    assert dt_on.module._resolve_commit(rewritten.repo, "no_such_revision") is None


def test_resolve_commit_returns_commit_for_live_sha(dt_on, rewritten):
    resolved = dt_on.module._resolve_commit(rewritten.repo, rewritten.head_sha)
    assert resolved is not None
    assert resolved.hexsha == rewritten.head_sha


def test_resolve_commit_does_not_swallow_unrelated_errors(dt_on, rewritten):
    """The guard is narrow on purpose — it must not become a bare `except`."""
    class Exploding:
        def commit(self, _rev):
            raise MemoryError("not a resolution failure")

    with pytest.raises(MemoryError):
        dt_on.module._resolve_commit(Exploding(), rewritten.head_sha)
