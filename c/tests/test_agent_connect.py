"""Coding apps pointed at colibri (agent_connect.py): the addresses from setup.json,
the model list from a live server, each app's configuration, and what opening one
writes. Nothing here starts a terminal or touches the user's own settings."""
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
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
        for name in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
                     "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL"):
            self.assertEqual(env[name], "qwen3.6-colibri")
        self.assertNotIn("ANTHROPIC_SMALL_FAST_MODEL", env)                    # deprecated

    def test_claude_code_knows_the_real_context_and_compacts(self):
        env = agent_connect.claude_env(conn(ctx=65536))
        self.assertEqual(env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "65536")
        self.assertEqual(env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"], "16000")
        self.assertEqual(env["ANTHROPIC_CUSTOM_HEADERS"], "x-colibri-context-compaction: auto")
        self.assertEqual(env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"], "1")
        self.assertNotIn("CLAUDE_CODE_AUTO_COMPACT_WINDOW", env)    # below Claude Code's 100k minimum
        big = agent_connect.claude_env(conn(ctx=131072))
        self.assertEqual(big["CLAUDE_CODE_AUTO_COMPACT_WINDOW"], "131072")
        self.assertEqual(agent_connect.claude_output_tokens(8192), 2048)

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
    def test_codex_keeps_the_users_home_and_hermes_gets_its_own(self):
        with tempfile.TemporaryDirectory() as home:
            env, args, files = agent_connect.launch_plan("codex", conn(), home=home)
            self.assertNotIn("CODEX_HOME", env)                    # their skills, MCP, plugins stay
            self.assertEqual(files, {})
            pairs = dict(a.split("=", 1) for a in args[1::2])
            self.assertEqual(args[0::2], ["-c"] * len(pairs))
            self.assertEqual(pairs["model_provider"], "colibri")
            self.assertEqual(pairs["model_providers.colibri.base_url"], "http://127.0.0.1:8790/v1")
            self.assertEqual(pairs["model_providers.colibri.wire_api"], "responses")
            self.assertEqual(pairs["model_context_window"], "65536")
            self.assertFalse(any('"' in a for a in args))           # nothing for cmd.exe to mangle
            env, _args, files = agent_connect.launch_plan("hermes", conn(), home=home)
            self.assertEqual(env["HERMES_HOME"], os.path.join(home, "hermes"))
            for path in files:
                self.assertTrue(path.startswith(home))     # never the user's ~/.codex or ~/.hermes

    def test_aider_through_its_openai_settings(self):
        env, args, files = agent_connect.launch_plan("aider", conn())
        self.assertEqual(env, {"OPENAI_API_BASE": "http://127.0.0.1:8123/v1", "OPENAI_API_KEY": "local"})
        self.assertEqual(args, ["--model", "openai/qwen3.6-colibri", "--no-show-model-warnings"])
        self.assertEqual(files, {})

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


try:
    import yaml  # only to check the YAML colibri writes parses as intended
except ImportError:
    yaml = None

CONTINUE_DEFAULT = "name: Local Assistant\nversion: 1.0.0\nschema: v1\nmodels: []\ncontext:\n  - provider: code\n"
CONTINUE_USER = ("name: Mine\nversion: 1.0.0\nschema: v1\nmodels:\n  # my cloud model\n"
                 "  - name: GPT\n    provider: openai\n    model: gpt-5\n    apiKey: sk-x\nrules:\n  - be brief\n")


class VSCodeExtensions(unittest.TestCase):
    def models(self, text):
        return yaml.safe_load(text)["models"] if yaml else None

    def test_continue_entry_into_the_default_file(self):
        out = agent_connect.continue_merge(CONTINUE_DEFAULT, conn())
        self.assertIn("context:\n  - provider: code\n", out)
        if yaml:
            data = yaml.safe_load(out)
            (model,) = data["models"]
            self.assertEqual((model["provider"], model["model"], model["apiBase"]),
                             ("openai", "qwen3.6-colibri", "http://127.0.0.1:8123/v1"))
            self.assertEqual(model["defaultCompletionOptions"]["contextLength"], 65536)
            self.assertIn("tool_use", model["capabilities"])
            self.assertEqual(data["context"], [{"provider": "code"}])

    def test_continue_keeps_the_users_models_and_rewrites_only_its_own(self):
        once = agent_connect.continue_merge(CONTINUE_USER, conn())
        twice = agent_connect.continue_merge(once, conn(model="glm-5", ctx=131072))
        self.assertEqual(twice.count(agent_connect.CONTINUE_BEGIN), 1)
        self.assertIn("# my cloud model", twice)
        self.assertTrue(twice.endswith("rules:\n  - be brief\n"))
        if yaml:
            names = [m["name"] for m in self.models(twice)]
            self.assertEqual(names, ["colibri glm-5", "GPT"])
            self.assertEqual(yaml.safe_load(twice)["rules"], ["be brief"])

    def test_continue_list_at_the_keys_indent_and_a_new_file(self):
        flush = "name: x\nversion: 1\nschema: v1\nmodels:\n- name: A\n  provider: ollama\n  model: a\n"
        out = agent_connect.continue_merge(flush, conn())
        new = agent_connect.continue_merge(None, conn())
        if yaml:
            self.assertEqual([m["name"] for m in self.models(out)], ["colibri qwen3.6-colibri", "A"])
            self.assertEqual(yaml.safe_load(new)["schema"], "v1")
            self.assertEqual(len(self.models(new)), 1)
        with self.assertRaises(ValueError):
            agent_connect.continue_merge("models: [{name: a}]\n", conn())

    def test_write_continue_keeps_a_backup_and_refuses_old_json(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "config.yaml").write_text(CONTINUE_USER)
            path, backup, words = agent_connect.write_settings("continue", conn(), continue_home=folder)
            self.assertEqual(Path(backup).read_text(), CONTINUE_USER)
            self.assertIn(agent_connect.CONTINUE_BEGIN, Path(path).read_text())
            self.assertIn("Continue", words)
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "config.json").write_text("{}")
            with self.assertRaises(ValueError):
                agent_connect.write_settings("continue", conn(), continue_home=folder)
            self.assertFalse(Path(folder, "config.yaml").exists())

    def test_roo_settings_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path, backup, _ = agent_connect.write_settings("roo", conn(), home=folder)
            data = json.loads(Path(path).read_text())
        profiles = data["providerProfiles"]
        profile = profiles["apiConfigs"][profiles["currentApiConfigName"]]
        self.assertIsNone(backup)
        self.assertEqual((profile["apiProvider"], profile["openAiBaseUrl"], profile["openAiModelId"]),
                         ("openai", "http://127.0.0.1:8123/v1", "qwen3.6-colibri"))
        self.assertEqual(profile["openAiCustomModelInfo"]["contextWindow"], 65536)
        self.assertIn("supportsPromptCache", profile["openAiCustomModelInfo"])   # required by Roo's schema
        self.assertTrue(profile["id"])

    def test_cline_and_unknown_model_are_not_written(self):
        with self.assertRaises(ValueError):
            agent_connect.write_settings("cline", conn())
        with self.assertRaises(ValueError):
            agent_connect.write_settings("roo", agent_connect.connection(CFG))
        self.assertIn("OpenAI Compatible", agent_connect.instructions("cline", conn()))

    def test_extension_found_in_any_editor_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            os.makedirs(os.path.join(folder, "rooveterinaryinc.roo-cline-3.30.0"))
            found = agent_connect.vscode_extension(("rooveterinaryinc.roo-cline-",), roots=[folder + "x", folder])
            self.assertTrue(found.endswith("roo-cline-3.30.0"))
            self.assertIsNone(agent_connect.vscode_extension(("continue.continue-",), roots=[folder]))
        entry = agent_connect.tool("continue")
        self.assertEqual(agent_connect.find_exe(entry, extension=lambda p: "/ext/" + p[0]),
                         "/ext/continue.continue-")



class NetworkAccess(unittest.TestCase):
    NET = {"host": "0.0.0.0", "port": 8123, "env": {"COLI_API_KEY": "Abc_def-0123456789xyz"}}

    def test_the_setup_key_and_the_address_for_other_computers(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            c = agent_connect.connection(self.NET, model="q", lan="192.168.1.20")
        self.assertEqual((c["key"], c["network"], c["base"]), ("Abc_def-0123456789xyz", True, "http://127.0.0.1:8123"))
        text = agent_connect.instructions("generic", c)
        self.assertIn("API key: Abc_def-0123456789xyz   (required", text)
        self.assertIn("From another computer: http://192.168.1.20:8123/v1", text)
        fixed = agent_connect.connection({"host": "10.0.0.7", "port": 8000}, key="k" * 20)
        self.assertEqual((fixed["lan"], fixed["base"]), ("10.0.0.7", "http://10.0.0.7:8000"))

    def test_this_pc_only_keeps_the_old_wording(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            c = agent_connect.connection({"host": "127.0.0.1", "port": 8123}, model="q")
        self.assertEqual((c["key"], c["network"], c["lan"]), ("local", False, None))
        self.assertIn("enforced only when", agent_connect.instructions("generic", c))

    def test_lan_address(self):
        self.assertEqual(agent_connect.lan_address(probe=lambda: "192.168.1.20"), "192.168.1.20")
        self.assertIsNone(agent_connect.lan_address(probe=lambda: "127.0.1.1"))

        def offline():
            raise OSError("no route")
        self.assertIsNone(agent_connect.lan_address(probe=offline))


if __name__ == "__main__":
    unittest.main()
