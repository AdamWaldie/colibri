#!/usr/bin/env python3
"""Is there a newer colibri, and bringing this one up to date (the configurator's
Updates section).

    git clone       fetch the branch this checkout follows; behind/ahead, the new
                    commits' titles, local edits. Updating is `git pull --ff-only`:
                    it never merges and never overwrites an edit, it refuses
                    instead. The engine is then rebuilt from the new sources by
                    the setup's own check (setup_flow.refresh_configured: make
                    redoes what changed, nothing when nothing did).
    release folder  the latest published release against this version; a newer
                    one is downloaded from its page (a running folder is not
                    overwritten in place: Windows locks the files a running
                    server uses).

Standard library only; git itself for a clone.
"""
import json
import os
import shutil
import subprocess
import sys
import urllib.error

HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import setup_catalog  # noqa: E402
import setup_flow  # noqa: E402

RELEASES_PAGE = "https://github.com/JustVugg/colibri/releases/latest"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def checkout_root(here=HERE):
    """The git checkout this file is in, or None (a release folder)."""
    folder = os.path.abspath(here)
    for _ in range(4):
        if os.path.exists(os.path.join(folder, ".git")):
            return folder
        parent = os.path.dirname(folder)
        if parent == folder:
            break
        folder = parent
    return None


def git(root, *args, timeout=120):
    """(return code, stdout, stderr) of a git command in `root`, without a window."""
    done = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace",                  # git writes UTF-8
                          stdin=subprocess.DEVNULL, creationflags=NO_WINDOW,
                          env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
    return done.returncode, done.stdout.strip(), done.stderr.strip()


def git_status(root, run=git, fetch=True):
    """{"branch", "upstream", "behind", "ahead", "dirty", "new": [titles], "error"}."""
    out = {"branch": "", "upstream": "", "behind": 0, "ahead": 0, "dirty": False, "new": [], "error": ""}
    code, branch, err = run(root, "rev-parse", "--abbrev-ref", "HEAD")
    if code:
        out["error"] = err or "not a git checkout"
        return out
    out["branch"] = branch
    code, upstream, _ = run(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if code or not upstream:
        out["error"] = f"the branch {branch} follows no remote branch: nothing to update from"
        return out
    out["upstream"] = upstream
    if fetch:
        remote, _, remote_branch = upstream.partition("/")
        code, _, err = run(root, "fetch", "--quiet", remote, remote_branch)
        if code:
            out["error"] = f"could not reach {remote}: {err.splitlines()[-1] if err else 'git fetch failed'}"
            return out
    code, counts, _ = run(root, "rev-list", "--left-right", "--count", "HEAD...@{u}")
    if not code and counts:
        ahead, behind = (int(x) for x in counts.split()[:2])
        out["ahead"], out["behind"] = ahead, behind
    code, dirty, _ = run(root, "status", "--porcelain", "--untracked-files=no")
    out["dirty"] = bool(dirty) if not code else False
    if out["behind"]:
        code, log, _ = run(root, "log", "--format=%s", "-n", "15", "HEAD..@{u}")
        out["new"] = [line for line in log.splitlines() if line] if not code else []
    return out


def git_words(status):
    """(one line for the window, can update)."""
    if status["error"]:
        return status["error"], False
    if not status["behind"]:
        extra = f" ({status['ahead']} of your own commits not on {status['upstream']})" if status["ahead"] else ""
        return f"Up to date with {status['upstream']}{extra}.", False
    text = (f"{status['behind']} update{'s' if status['behind'] != 1 else ''} on {status['upstream']}: "
            + "; ".join(status["new"][:5]) + ("; ..." if len(status["new"]) > 5 else ""))
    if status["ahead"]:
        return text + (f"\nYour branch also has {status['ahead']} commit(s) of its own: update it with "
                       "git (a merge or rebase) yourself; Update now only fast-forwards."), False
    if status["dirty"]:
        return text + "\nThere are edited files here: commit or undo them first, so nothing is overwritten.", False
    return text, True


def pull_command(root):
    return ["git", "-C", root, "pull", "--ff-only"]


def latest_release(getter=None, api=None):
    """{"tag", "version", "url"} of the latest published release, or None."""
    api = (api or os.environ.get("COLI_RELEASES_API") or setup_flow.RELEASES_API).rstrip("/")
    getter = getter or setup_flow._api_get
    try:
        data = getter(f"{api}/latest")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("tag_name"):
        return None
    return {"tag": data["tag_name"], "version": data["tag_name"].lstrip("v"),
            "url": data.get("html_url") or RELEASES_PAGE}


def release_words(current, release):
    """(one line, newer available)."""
    if not release:
        return f"colibri {current}. Could not reach GitHub to see the latest release.", False
    if setup_catalog.version_tuple(release["version"]) > setup_catalog.version_tuple(current):
        return (f"colibri {release['version']} is available (this is {current}): download it from its page "
                "and unpack it over this folder with colibri stopped."), True
    return f"colibri {current}: the latest release.", False


def check(here=HERE, run=git, getter=None):
    """Everything the Updates section shows, as plain data."""
    root = checkout_root(here)
    current = setup_flow.current_version()
    if root and shutil.which("git"):
        status = git_status(root, run)
        words, can = git_words(status)
        return {"kind": "git", "root": root, "status": status, "words": words, "can_update": can,
                "version": current}
    release = latest_release(getter)
    words, newer = release_words(current, release)
    return {"kind": "release", "root": None, "words": words, "can_update": False, "newer": newer,
            "url": (release or {}).get("url") or RELEASES_PAGE, "version": current}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
