"""Agent Skills in colibri's chat (skills.py): SKILL.md folders found, listed at
/v1/skills, and turned on by a chat request's "skills" field."""
import json
import os
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import skills  # noqa: E402

REVIEW = """---
name: code-review
description: >
  Review a diff for bugs,
  then style.
---
# Code review

Read the diff twice. List bugs first.
"""


def make(root, folder, text):
    path = Path(root, folder)
    path.mkdir(parents=True, exist_ok=True)
    (path / "SKILL.md").write_text(text, encoding="utf-8")


class Folders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.claude = os.path.join(self.tmp.name, "claude")
        self.codex = os.path.join(self.tmp.name, "codex")
        make(self.claude, "review", REVIEW)
        make(self.claude, "plain", "Just do it plainly.\n")
        make(self.codex, "review-again", REVIEW)                 # same name: the first folder wins
        make(self.codex, "pdf", '---\nname: "pdf"\ndescription: "Fill PDF forms"\n---\nUse the form tool.\n')
        make(self.codex, "bad", "---\nname: ../../etc\n---\nx\n")
        Path(self.codex, "not-a-skill").mkdir()
        self.dirs = [self.claude, self.codex, os.path.join(self.tmp.name, "missing")]

    def tearDown(self):
        self.tmp.cleanup()

    def test_discover(self):
        found = skills.discover(self.dirs)
        self.assertEqual(sorted(found), ["code-review", "pdf", "plain"])
        self.assertEqual(found["code-review"]["description"], "Review a diff for bugs, then style.")
        self.assertEqual(found["code-review"]["source"], self.claude)
        self.assertEqual(found["pdf"]["description"], "Fill PDF forms")
        listed = skills.listing(self.dirs)["data"]
        self.assertNotIn("path", listed[0])

    def test_a_bom_from_notepad(self):
        Path(self.claude, "bom").mkdir()
        Path(self.claude, "bom", "SKILL.md").write_bytes(b"\xef\xbb\xbf" + REVIEW.replace("code-review", "bommed").encode())
        found = skills.discover([self.claude])
        self.assertEqual(found["bommed"]["description"], "Review a diff for bugs, then style.")
        self.assertNotIn("description:", skills.instructions(["bommed"], [self.claude]))

    def test_default_folders(self):
        dirs = skills.skill_dirs(env={"COLI_SKILLS_DIRS": os.pathsep.join(["/a", "", "/b"])}, home="/h")
        self.assertEqual(dirs, ["/a", "/b", os.path.join("/h", ".claude", "skills"),
                                os.path.join("/h", ".codex", "skills"), os.path.join("/h", ".agents", "skills")])

    def test_a_request_turns_skills_on(self):
        body = {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}],
                "skills": ["code-review", "plain", "code-review"]}
        skills.expand_body(body, self.dirs)
        self.assertNotIn("skills", body)
        system = body["messages"][0]["content"]
        self.assertTrue(system.startswith("Be brief.\n\nThe user turned on these skills."))
        self.assertIn('<skill name="code-review">\n# Code review\n\nRead the diff twice.', system)
        self.assertNotIn("description: >", system)
        self.assertEqual(system.count('<skill name="code-review">'), 1)
        self.assertIn('<skill name="plain">\nJust do it plainly.\n</skill>', system)

        body = {"messages": [{"role": "user", "content": "hi"}], "skills": ["pdf"]}
        skills.expand_body(body, self.dirs)
        self.assertEqual(body["messages"][0]["role"], "system")
        anthropic = {"system": "S", "messages": "n/a", "skills": ["pdf"]}
        anthropic.pop("messages")
        skills.expand_body(anthropic, self.dirs)
        self.assertIn("Use the form tool.", anthropic["system"])

    def test_bad_requests(self):
        for value in ("code-review", ["../x"], [3]):
            with self.assertRaises(skills.SkillError):
                skills.expand_body({"messages": [], "skills": value}, self.dirs)
        with self.assertRaisesRegex(skills.SkillError, "no skill called 'nope'"):
            skills.expand_body({"messages": [], "skills": ["nope"]}, self.dirs)
        body = {"messages": [], "skills": []}
        self.assertEqual(skills.expand_body(body, self.dirs), {"messages": []})

    def test_size_limits(self):
        make(self.claude, "huge", "x" * (skills.MAX_SKILL_BYTES + 10))
        with self.assertRaisesRegex(skills.SkillError, "larger than 64 KB"):
            skills.expand_body({"messages": [], "skills": ["huge"]}, self.dirs)


class Gateway(unittest.TestCase):
    def setUp(self):
        from test_openai_server import FakeEngine, _post_json, _spawn_test_server, _error_body
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        make(self.tmp.name, "review", REVIEW)
        self.engine = FakeEngine()
        self.base = _spawn_test_server(self, self.engine)
        self.post = lambda path, body: _post_json(self.base, path, body)
        self.error_body = _error_body
        self.dirs = mock.patch.object(skills, "skill_dirs", lambda env=None, home=None: [self.tmp.name])
        self.dirs.start()
        self.addCleanup(self.dirs.stop)

    def test_listed_and_turned_on(self):
        def get(path):
            return urllib.request.urlopen(urllib.request.Request(
                self.base + path, headers={"Authorization": "Bearer secret"}))

        with get("/v1/skills") as response:
            data = json.load(response)["data"]
        self.assertEqual([s["name"] for s in data], ["code-review"])
        with get("/health") as response:
            self.assertTrue(json.load(response)["skills"])
        body = {"model": "test-model", "max_tokens": 4, "skills": ["code-review"],
                "messages": [{"role": "user", "content": "Review this."}]}
        with self.post("/v1/chat/completions", body) as response:
            self.assertEqual(response.status, 200)
        prompt = self.engine.calls[0][0]
        self.assertIn("List bugs first.", prompt if isinstance(prompt, str) else json.dumps(prompt))

    def test_an_unknown_skill_is_a_400(self):
        body = {"model": "test-model", "skills": ["nope"], "messages": [{"role": "user", "content": "x"}]}
        status, error = self.error_body(self, lambda: self.post("/v1/chat/completions", body))
        self.assertEqual(status, 400)
        self.assertIn("no skill called 'nope'", error["message"])
        self.assertEqual(self.engine.calls, [])


if __name__ == "__main__":
    unittest.main()
