"""Coding apps pointed at colibri (agent_connect.py): the addresses from setup.json,
the model list from a live server, each app's configuration, and what opening one
writes. Nothing here starts a terminal or touches the user's own settings."""
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python 3.10
    tomllib = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent_connect  # noqa: E402

CFG = {"host": "0.0.0.0", "port": 8123}


def conn(**kw):
    return agent_connect.connection(CFG, model=kw.pop("model", "qwen3.6-colibri"),
                                    ctx=kw.pop("ctx", 65536), **kw)


class Connection(unittest.TestCase):
    def test_wildcard_bind_is_reached_on_loopback(self):
        c = conn()
        self.assertEqual(c["base"], "http://127.0.0.1:8123")
        self.assertEqual(c["v1"], "http://127.0.0.1:8123/v1")
        self.assertEqual(c["bridge_v1"], "http://127.0.0.1:8790/v1")
        self.assertTrue(c["network"])

    def test_defaults_without_a_setup(self):
        c = agent_connect.connection(None)
        self.assertEqual(c["base"], "http://127.0.0.1:8000")
        self.assertEqual(c["model"], "<model-id>")
        self.assertEqual(c["ctx"], agent_connect.DEFAULT_CTX)


class ServedModels(unittest.TestCase):
    def opener(self, payload=None, error=None):
        def open_(request, timeout):
            self.request = request
            if error:
                raise error
            return io.BytesIO(json.dumps(payload).encode())
        return open_

    def test_ids_and_context(self):
        data = {"data": [{"id": "glm-5.2-colibri", "context_length": 32768},
                         {"id": "x", "max_model_len": 4096}, {"id": "y"}, {"no": "id"}]}
        got = agent_connect.served_models("http://h:1", key="k", opener=self.opener(data))
        self.assertEqual(got, [("glm-5.2-colibri", 32768), ("x", 4096), ("y", 65536)])
        self.assertEqual(self.request.full_url, "http://h:1/v1/models")
        self.assertEqual(self.request.get_header("Authorization"), "Bearer k")

    def test_no_server(self):
        self.assertEqual(agent_connect.served_models("http://h:1", opener=self.opener(error=OSError())), [])


class AppConfigurations(unittest.TestCase):
    @unittest.skipIf(tomllib is None, "tomllib is Python 3.11+")
    def test_codex_default_and_profile_are_valid_toml(self):
        default = tomllib.loads(agent_connect.codex_toml(conn(), default=True))
        self.assertEqual(default["model_provider"], "colibri")
        self.assertEqual(default["model"], "qwen3.6-colibri")
        provider = default["model_providers"]["colibri"]
        self.assertEqual(provider["wire_api"], "responses")
        self.assertEqual(provider["base_url"], "http://127.0.0.1:8790/v1")   # the bridge, not the server
        profile = tomllib.loads(agent_connect.codex_toml(conn(), default=False))
        self.assertNotIn("model_provider", profile)                          # the user's default untouched
        self.assertEqual(profile["profiles"]["colibri"]["model_context_window"], 65536)

    def test_opencode_provider(self):
        cfg = agent_connect.opencode_config(conn())
        provider = cfg["provider"]["colibri"]
        self.assertEqual(provider["options"]["baseURL"], "http://127.0.0.1:8123/v1")
        self.assertEqual(cfg["model"], "colibri/qwen3.6-colibri")
        self.assertEqual(provider["models"]["qwen3.6-colibri"]["limit"], {"context": 65536, "output": 32768})

    def test_claude_code_uses_the_anthropic_endpoint_for_every_tier(self):
        env = agent_connect.claude_env(conn())
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8123")   # no /v1: Claude adds it
        for name in ("ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
                     "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL"):
            self.assertEqual(env[name], "qwen3.6-colibri")

    def test_hermes_yaml(self):
        text = agent_connect.hermes_yaml(conn())
        self.assertIn('base_url: "http://127.0.0.1:8123/v1"', text)
        self.assertIn("context_length: 65536", text)

    def test_every_app_has_instructions(self):
        for entry in agent_connect.TOOLS:
            with self.subTest(app=entry["id"]):
                self.assertIn("qwen3.6-colibri", agent_connect.instructions(entry["id"], conn()))

    def test_instructions_say_when_the_server_is_down(self):
        self.assertIn("not answering", agent_connect.instructions("generic", agent_connect.connection(CFG)))

    def test_env_lines(self):
        env = {"A": "1", "B": "x y"}
        self.assertEqual(agent_connect.env_lines(env), 'set "A=1"\nset "B=x y"')
        self.assertEqual(agent_connect.env_lines(env, "powershell"), "$env:A = '1'\n$env:B = 'x y'")
        self.assertEqual(agent_connect.env_lines(env, persist=True), 'setx A "1"\nsetx B "x y"')


class Launching(unittest.TestCase):
    def test_codex_and_hermes_get_a_home_of_colibris_own(self):
        with tempfile.TemporaryDirectory() as home:
            env, _args, files = agent_connect.launch_plan("codex", conn(), home=home)
            self.assertEqual(env["CODEX_HOME"], os.path.join(home, "codex"))
            self.assertEqual(list(files), [os.path.join(home, "codex", "config.toml")])
            env, _args, files = agent_connect.launch_plan("hermes", conn(), home=home)
            self.assertEqual(env["HERMES_HOME"], os.path.join(home, "hermes"))
            for path in files:
                self.assertTrue(path.startswith(home))     # never the user's ~/.codex or ~/.hermes

    def test_opencode_through_the_environment(self):
        env, args, files = agent_connect.launch_plan("opencode", conn())
        self.assertEqual(json.loads(env["OPENCODE_CONFIG_CONTENT"])["model"], "colibri/qwen3.6-colibri")
        self.assertEqual(args, ["-m", "colibri/qwen3.6-colibri"])
        self.assertEqual(files, {})

    def test_refused_without_a_model_or_with_an_odd_key(self):
        with self.assertRaises(ValueError):
            agent_connect.launch_plan("claude", agent_connect.connection(CFG))
        with self.assertRaises(ValueError):
            agent_connect.launch_plan("claude", conn(key='a" & calc'))
        with self.assertRaises(ValueError):
            agent_connect.launch_plan("t3code", conn())

    def test_launcher_script(self):
        text = agent_connect.launcher_script("opencode", r"C:\npm\opencode.cmd", ["-m", "colibri/q"],
                                             r"D:\work\proj", "100% local")
        self.assertIn('cd /d "D:\\work\\proj"', text)
        self.assertIn('call "C:\\npm\\opencode.cmd" "-m" "colibri/q"', text)
        self.assertIn("100%% local", text)
        self.assertTrue(text.endswith("\r\n"))

    def test_bridge_command_runs_coli_ui_in_front_of_the_server(self):
        cmd = agent_connect.bridge_command(conn(), python="py", coli="coli")
        self.assertEqual(cmd, ["py", "coli", "ui", "--no-browser", "--port", "8790",
                               "--backend", "http://127.0.0.1:8123"])

    def test_find_exe(self):
        entry = agent_connect.tool("claude")
        self.assertEqual(agent_connect.find_exe(entry, which=lambda n: r"C:\x\claude.cmd"), r"C:\x\claude.cmd")
        self.assertIsNone(agent_connect.find_exe(entry, which=lambda n: None))
        self.assertIsNone(agent_connect.find_exe(agent_connect.tool("generic")))


if __name__ == "__main__":
    unittest.main()
