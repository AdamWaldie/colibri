"""Checking for a newer colibri and updating (updater.py), with real git repositories:
an origin, the user's clone, and another clone that pushes what the user will pull.
The release side against a fake GitHub API."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import updater  # noqa: E402

GIT = shutil.which("git")


def sh(cwd, *args):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t", GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, env=env)


@unittest.skipUnless(GIT, "git is not installed")
class GitCheckout(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = self.tmp.name
        self.origin = os.path.join(base, "origin.git")
        sh(base, "init", "--bare", "-b", "main", self.origin)
        self.user = os.path.join(base, "user")
        sh(base, "clone", self.origin, self.user)
        Path(self.user, "c").mkdir()
        Path(self.user, "c", "coli").write_text("v1\n")
        sh(self.user, "add", ".")
        sh(self.user, "commit", "-m", "first")
        sh(self.user, "push", "-u", "origin", "main")
        self.other = os.path.join(base, "other")
        sh(base, "clone", self.origin, self.other)

    def tearDown(self):
        self.tmp.cleanup()

    def push_update(self, title):
        Path(self.other, "c", "coli").write_text(title + "\n")
        sh(self.other, "commit", "-am", title)
        sh(self.other, "push")

    def test_checkout_root_from_the_c_folder(self):
        self.assertEqual(updater.checkout_root(os.path.join(self.user, "c")), self.user)
        self.assertIsNone(updater.checkout_root(self.tmp.name))

    def test_up_to_date(self):
        status = updater.git_status(self.user)
        self.assertEqual((status["upstream"], status["behind"], status["error"]), ("origin/main", 0, ""))
        words, can = updater.git_words(status)
        self.assertIn("Up to date", words)
        self.assertFalse(can)

    def test_an_update_is_seen_and_pulled(self):
        self.push_update("Open app menu")
        self.push_update("coding context")
        status = updater.git_status(self.user)
        self.assertEqual(status["behind"], 2)
        self.assertEqual(status["new"], ["coding context", "Open app menu"])
        words, can = updater.git_words(status)
        self.assertTrue(can)
        self.assertIn("2 updates", words)
        subprocess.run(updater.pull_command(self.user), check=True, capture_output=True)
        self.assertEqual(Path(self.user, "c", "coli").read_text(), "coding context\n")
        self.assertEqual(updater.git_status(self.user)["behind"], 0)

    def test_local_edits_are_never_overwritten(self):
        self.push_update("new")
        Path(self.user, "c", "coli").write_text("my edit\n")
        words, can = updater.git_words(updater.git_status(self.user))
        self.assertFalse(can)
        self.assertIn("edited files", words)

    def test_a_branch_of_its_own_is_left_to_git(self):
        self.push_update("theirs")
        Path(self.user, "mine.txt").write_text("x")
        sh(self.user, "add", "mine.txt")
        sh(self.user, "commit", "-m", "mine")
        status = updater.git_status(self.user)
        self.assertEqual((status["behind"], status["ahead"]), (1, 1))
        words, can = updater.git_words(status)
        self.assertFalse(can)
        self.assertIn("only fast-forwards", words)

    def test_no_upstream_and_no_network(self):
        sh(self.user, "checkout", "-b", "local-only")
        self.assertIn("follows no remote", updater.git_status(self.user)["error"])
        sh(self.user, "checkout", "main")
        sh(self.user, "remote", "set-url", "origin", os.path.join(self.tmp.name, "gone.git"))
        self.assertIn("could not reach origin", updater.git_status(self.user)["error"])


class Releases(unittest.TestCase):
    def test_latest_release(self):
        got = updater.latest_release(getter=lambda url: {"tag_name": "v2.2.0", "html_url": "https://x/r"},
                                     api="https://api.example/releases")
        self.assertEqual(got, {"tag": "v2.2.0", "version": "2.2.0", "url": "https://x/r"})

        def down(url):
            raise OSError("offline")
        self.assertIsNone(updater.latest_release(getter=down, api="https://api.example/releases"))

    def test_release_words(self):
        newer = {"version": "2.2.0", "tag": "v2.2.0", "url": "u"}
        self.assertTrue(updater.release_words("2.1.0", newer)[1])
        self.assertIn("2.2.0 is available", updater.release_words("2.1.0", newer)[0])
        self.assertFalse(updater.release_words("2.2.0", newer)[1])
        self.assertFalse(updater.release_words("2.10.0", newer)[1])          # numeric, not text order
        self.assertIn("Could not reach", updater.release_words("2.1.0", None)[0])

    def test_a_release_folder_is_checked_against_github(self):
        with tempfile.TemporaryDirectory() as folder:
            result = updater.check(here=folder, getter=lambda url: {"tag_name": "v9.0.0", "html_url": "https://r"})
        self.assertEqual((result["kind"], result["newer"], result["url"]), ("release", True, "https://r"))
        self.assertFalse(result["can_update"])


if __name__ == "__main__":
    unittest.main()
