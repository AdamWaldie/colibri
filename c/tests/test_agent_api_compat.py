"""What Claude Code and Codex send, served: the gaps an audit of both apps against a
logging gateway found (docs/api.md, "Connect a coding CLI or editor").

- Claude Code 2.1 sends its environment as a `system` message among the others on
  /v1/messages; that was a 400, and the app retried without it.
- It counts tokens with POST /v1/messages/count_tokens, which did not exist.
- Its gateway model discovery reads Anthropic's model list fields.
- A long session needs the server to compact on /v1/messages too, opted into by a
  header the app can always send; Codex, through the Responses bridge, opts in.
"""
import json
import os
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_openai_server import FakeEngine, _post_json, _spawn_test_server  # noqa: E402
import test_context_compaction_e2e as e2e  # noqa: E402


def _get(base, path):
    request = urllib.request.Request(base + path, headers={"Authorization": "Bearer secret"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


class ClaudeCodeRequests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.base = _spawn_test_server(self, self.engine)

    def test_a_system_message_among_the_others_joins_the_system_prompt(self):
        body = {"model": "test-model", "max_tokens": 8,
                "system": [{"type": "text", "text": "You are Claude Code."}],
                "messages": [{"role": "user", "content": [{"type": "text", "text": "list the files"}]},
                             {"role": "system", "content": [{"type": "text", "text": "# Environment\ncwd: /work"}]}]}
        with _post_json(self.base, "/v1/messages", body) as response:
            self.assertEqual(response.status, 200)
        prompt = self.engine.calls[0][0]
        self.assertIn("You are Claude Code.", prompt)
        self.assertIn("cwd: /work", prompt)
        self.assertLess(prompt.index("cwd: /work"), prompt.index("list the files"))   # in the system part

    def test_only_system_messages_is_still_a_400(self):
        body = {"model": "test-model", "max_tokens": 8,
                "messages": [{"role": "system", "content": "context only"}]}
        with self.assertRaises(urllib.error.HTTPError) as caught:
            _post_json(self.base, "/v1/messages", body)
        self.assertEqual(caught.exception.code, 400)
        caught.exception.close()

    def test_count_tokens(self):
        def count(text, **extra):
            body = dict({"model": "test-model", "messages": [{"role": "user", "content": text}]}, **extra)
            with _post_json(self.base, "/v1/messages/count_tokens", body) as response:
                return json.load(response)["input_tokens"]

        short, longer = count("hi"), count("hi " * 400)
        self.assertGreater(short, 0)
        self.assertGreater(longer, short + 200)
        tool = {"name": "Bash", "description": "Run a command " * 20,
                "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}}
        self.assertGreater(count("hi", tools=[tool]), short)          # the tools are part of the prompt
        self.assertEqual(self.engine.calls, [])                        # counting never generates

    def test_model_list_in_both_shapes(self):
        listing = _get(self.base, "/v1/models")
        entry = listing["data"][0]
        self.assertEqual((entry["id"], entry["object"]), ("test-model", "model"))     # OpenAI
        self.assertEqual((entry["type"], entry["display_name"]), ("model", "test-model"))   # Anthropic
        self.assertRegex(entry["created_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertEqual((listing["has_more"], listing["first_id"], listing["last_id"]),
                         (False, "test-model", "test-model"))


@unittest.skipUnless(os.name == "posix", "the mock engine is a shebang script the gateway execs directly")
class MessagesCompaction(unittest.TestCase):
    """The compaction end-to-end harness, through /v1/messages."""

    @classmethod
    def setUpClass(cls):
        e2e.ContextCompactionE2E.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        e2e.ContextCompactionE2E.tearDownClass.__func__(cls)

    def prompts(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def post(self, header=None):
        openai = e2e.conversation(6)
        body = {"model": self.model, "max_tokens": 16, "system": openai[0]["content"],
                "messages": openai[1:]}
        headers = {"Content-Type": "application/json"}
        if header:
            headers["x-colibri-context-compaction"] = header
        request = urllib.request.Request(self.base + "/messages", json.dumps(body).encode(), headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    def test_1_without_the_header_the_400_stays(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post()
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("context", caught.exception.read().decode().lower())

    def test_2_the_header_compacts_and_answers(self):
        start = len(self.prompts())
        reply = self.post("auto")
        self.assertEqual(reply["content"][-1]["text"], "Noted, turn answered.")
        seen = self.prompts()[start:]
        self.assertTrue(any("Summarize our conversation so far" in p["prompt"] for p in seen))
        last = seen[-1]["prompt"]
        self.assertIn(e2e.SUMMARY, last)                     # the turn runs on the summary
        self.assertIn("What was the last topic?", last)


class CodexBridge(unittest.TestCase):
    def test_the_responses_bridge_opts_into_compaction(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ui"))
        import responses_shim
        chat, _custom = responses_shim.to_chat({"model": "m", "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}]})
        self.assertEqual(chat["context_compaction"], "auto")


if __name__ == "__main__":
    unittest.main()
