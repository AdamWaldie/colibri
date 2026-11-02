#!/usr/bin/env python3
"""Coding apps pointed at the colibri server: what each one needs, and one opened in a
new terminal window already connected (the configurator's "Connect apps" tab).

colibri serves the OpenAI chat API (/v1/chat/completions) and the Anthropic Messages
API (/v1/messages) on the port `coli setup` configured, and lists the model at
/v1/models. That is enough for Claude Code (Anthropic API), opencode, Hermes Agent,
OpenClaw, pi, aider, crush, Continue, Cline and other OpenAI-compatible clients.
Codex speaks only the Responses API, which the chat UI's gateway (`coli ui`,
ui/responses_shim.py) translates: Codex connects to that bridge, not to the server.

Launching never rewrites the user's own settings: Claude Code and opencode get
colibri through environment variables, Codex and Hermes through a home folder of
their own under the setup folder (CODEX_HOME, HERMES_HOME). The "permanent" text is
for the user to paste where they choose. The same rules as ui/agents.py, for a
configurator that runs natively on Windows rather than from WSL.

Standard library only.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import setup_flow  # noqa: E402

BRIDGE_PORT = 8790             # `coli ui`'s default port
DEFAULT_CTX = 65536
MODEL_RE = re.compile(r"^[A-Za-z0-9._:/-]{1,128}$")
KEY_RE = re.compile(r"^[A-Za-z0-9._-]{1,200}$")


# ---------------------------------------------------------------- where the server is


def connection(cfg, model=None, ctx=None, key="local", bridge_port=BRIDGE_PORT):
    """The addresses and identity an app needs, from the saved run configuration.
    A wildcard bind (0.0.0.0) is not an address to connect to: apps on this machine
    use 127.0.0.1."""
    cfg = cfg or {}
    host = cfg.get("host") or setup_flow.DEFAULT_HOST
    shown = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    port = int(cfg.get("port") or setup_flow.DEFAULT_PORT)
    base = f"http://{shown}:{port}"
    return {"base": base, "v1": f"{base}/v1", "bridge": f"http://127.0.0.1:{bridge_port}",
            "bridge_v1": f"http://127.0.0.1:{bridge_port}/v1", "bridge_port": bridge_port,
            "port": port, "model": model or "<model-id>", "ctx": int(ctx or DEFAULT_CTX),
            "key": key or "local", "network": host in ("0.0.0.0", "::", "")}


def served_models(base, key="", timeout=3.0, opener=None):
    """[(id, context tokens)] the running server lists, [] when it does not answer."""
    request = urllib.request.Request(base.rstrip("/") + "/v1/models")
    if key:
        request.add_header("Authorization", f"Bearer {key}")
    try:
        with (opener or urllib.request.urlopen)(request, timeout=timeout) as response:
            data = json.load(response).get("data") or []
    except (OSError, ValueError, AttributeError):
        return []
    out = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            ctx = item.get("context_length") or item.get("max_model_len") or DEFAULT_CTX
            out.append((item["id"], int(ctx) if isinstance(ctx, (int, float)) and ctx > 0 else DEFAULT_CTX))
    return out


def answers(url, timeout=1.5):
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True        # something answers there
    except OSError:
        return False


def bridge_command(conn, python=None, coli=None):
    """`coli ui` in front of the server: Codex's Responses API over the chat API."""
    return [python or sys.executable, coli or os.path.join(HERE, "coli"), "ui", "--no-browser",
            "--port", str(conn["bridge_port"]), "--backend", conn["base"]]


# ---------------------------------------------------------------- the apps


def _out_tokens(ctx):
    return max(1024, min(32768, ctx // 2))


def codex_toml(conn, default=True):
    """Codex's configuration for colibri. default=True makes it the default provider
    (a CODEX_HOME of colibri's own, or T3 Code, which runs Codex underneath);
    default=False is a profile to add to the user's own ~/.codex/config.toml."""
    provider = ("[model_providers.colibri]\n"
                'name = "colibri"\n'
                f'base_url = "{conn["bridge_v1"]}"\n'
                'env_key = "COLIBRI_API_KEY"\n'
                'wire_api = "responses"\n')
    if default:
        return (f'model = "{conn["model"]}"\n'
                'model_provider = "colibri"\n'
                f'model_context_window = {conn["ctx"]}\n\n' + provider)
    return (provider + "\n[profiles.colibri]\n"
            'model_provider = "colibri"\n'
            f'model = "{conn["model"]}"\n'
            f'model_context_window = {conn["ctx"]}\n')


def opencode_config(conn):
    return {"$schema": "https://opencode.ai/config.json",
            "provider": {"colibri": {"npm": "@ai-sdk/openai-compatible", "name": "colibri",
                                     "options": {"baseURL": conn["v1"], "apiKey": conn["key"]},
                                     "models": {conn["model"]: {"name": f"{conn['model']} (colibri)",
                                                                "limit": {"context": conn["ctx"],
                                                                          "output": _out_tokens(conn["ctx"])}}}}},
            "model": f"colibri/{conn['model']}"}


def hermes_yaml(conn):
    return ("model:\n"
            f"  default: {json.dumps(conn['model'])}\n"
            "  provider: custom\n"
            f"  base_url: {json.dumps(conn['v1'])}\n"
            f"  api_key: {json.dumps(conn['key'])}\n"
            f"  context_length: {conn['ctx']}\n")


def claude_env(conn):
    model = conn["model"]
    return {"ANTHROPIC_BASE_URL": conn["base"], "ANTHROPIC_AUTH_TOKEN": conn["key"],
            "ANTHROPIC_MODEL": model, "ANTHROPIC_SMALL_FAST_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model, "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_TELEMETRY": "1"}


def env_lines(env, shell="cmd", persist=False):
    """Variables as lines for cmd (`set`), PowerShell (`$env:`), or persisted for new
    windows (`setx`, which takes effect in terminals opened afterwards)."""
    lines = []
    for key, value in env.items():
        if persist:
            lines.append(f'setx {key} "{value}"')
        elif shell == "powershell":
            lines.append(f"$env:{key} = '{value}'")
        elif shell == "sh":
            lines.append(f"export {key}='{value}'")
        else:
            lines.append(f'set "{key}={value}"')
    return "\n".join(lines)


TOOLS = (
    {"id": "claude", "name": "Claude Code", "exe": ("claude",), "api": "Anthropic (/v1/messages)",
     "launch": True},
    {"id": "codex", "name": "Codex CLI", "exe": ("codex",), "api": "Responses, through the bridge",
     "launch": True, "bridge": True},
    {"id": "opencode", "name": "opencode", "exe": ("opencode",), "api": "OpenAI-compatible", "launch": True},
    {"id": "hermes", "name": "Hermes Agent", "exe": ("hermes",), "api": "OpenAI-compatible", "launch": True},
    {"id": "t3code", "name": "T3 Code", "exe": (), "api": "runs Codex or Claude Code", "launch": False,
     "bridge": True},
    {"id": "generic", "name": "Any OpenAI-compatible app", "exe": (), "api": "OpenAI-compatible",
     "launch": False},
)


def tool(tool_id):
    for entry in TOOLS:
        if entry["id"] == tool_id:
            return entry
    raise KeyError(tool_id)


def find_exe(entry, which=shutil.which):
    for name in entry["exe"]:
        found = which(name)
        if found:
            return found
    return None


def instructions(tool_id, conn):
    """What the user reads (and copies) for one app: a plain-text recipe."""
    v1, model, key, ctx = conn["v1"], conn["model"], conn["key"], conn["ctx"]
    head = (f"Server: {conn['base']}   model: {model}   context: {ctx} tokens   key: any value "
            f"(\"{key}\"; enforced only when the server runs with COLI_API_KEY)\n")
    if conn["model"] == "<model-id>":
        head += "The server is not answering: start it, then Refresh, to fill in the model id.\n"
    if tool_id == "claude":
        env = claude_env(conn)
        return head + (
            "\nClaude Code talks to colibri's Anthropic endpoint. 'Open in terminal' starts it in the "
            "chosen folder with these variables (your own settings are not changed).\n\n"
            "This terminal only (cmd):\n" + env_lines(env) + "\nclaude\n\n"
            "This terminal only (PowerShell):\n" + env_lines(env, "powershell") + "\nclaude\n\n"
            "Permanently for one project: put them in <project>\\.claude\\settings.local.json:\n"
            + json.dumps({"env": env}, indent=2) + "\n\n"
            "Putting the same \"env\" block in %USERPROFILE%\\.claude\\settings.json makes EVERY "
            "Claude Code session use colibri instead of Anthropic's models.\n\n"
            "Expect a long first answer: Claude Code sends a large system prompt and tool list "
            "(10-20k tokens) that the model reads before it writes. A model that runs from RAM "
            "(Qwen3.6, Qwen3-Coder) is the practical choice; give the server a context of 65536 or more.")
    if tool_id == "codex":
        return head + (
            f"\nCodex speaks only the Responses API. colibri's chat UI gateway translates it: "
            f"'Start bridge' runs `coli ui` on {conn['bridge']} in front of the server, and Codex "
            f"connects to {conn['bridge_v1']}. 'Open in terminal' starts the bridge if needed and "
            "runs Codex with a CODEX_HOME of colibri's own (your ~/.codex is not touched).\n\n"
            "Permanently: add this to %USERPROFILE%\\.codex\\config.toml\n\n"
            + codex_toml(conn, default=False) +
            f"\nthen, with the bridge running:\n{env_lines({'COLIBRI_API_KEY': key}, persist=True)}\n"
            "codex --profile colibri")
    if tool_id == "opencode":
        conf = opencode_config(conn)
        return head + (
            "\n'Open in terminal' passes this provider to opencode in OPENCODE_CONFIG_CONTENT "
            "(your opencode.json is not changed).\n\n"
            "Permanently: merge it into %USERPROFILE%\\.config\\opencode\\opencode.json (or "
            "opencode.json in a project):\n\n" + json.dumps(conf, indent=2) +
            f"\n\nthen: opencode -m colibri/{model}")
    if tool_id == "hermes":
        return head + (
            "\n'Open in terminal' runs Hermes with HERMES_HOME set to a folder of colibri's own "
            "(your ~/.hermes is not changed). Hermes needs a context of at least 64,000 tokens: "
            "set Context to 65536 or more on the Run tab and Install / Apply.\n\n"
            "Permanently: %USERPROFILE%\\.hermes\\config.yaml\n\n" + hermes_yaml(conn))
    if tool_id == "t3code":
        return head + (
            "\nT3 Code is a desktop app that runs the Codex CLI (and Claude Code) for you, so it "
            "uses whatever those are configured with:\n\n"
            "Through Codex: start the bridge ('Start bridge' on the Codex entry, or keep "
            "`coli ui` running), set the key once in a terminal:\n"
            + env_lines({"COLIBRI_API_KEY": key}, persist=True) +
            "\nand make colibri Codex's default at the top of %USERPROFILE%\\.codex\\config.toml "
            "(this changes plain `codex` too; keep a copy of the file):\n\n"
            + codex_toml(conn, default=True) +
            "\nThrough Claude Code: put the Claude Code entry's \"env\" block in "
            "%USERPROFILE%\\.claude\\settings.json.\n\n"
            "Restart T3 Code afterwards. Not verified against every T3 Code release: if it "
            "starts Codex with its own provider options, use the Codex CLI entry instead.")
    if tool_id == "generic":
        return head + (
            "\nFor any app with an 'OpenAI-compatible' (or 'custom OpenAI') provider: Continue, "
            "Cline, Roo Code, aider, crush, pi, OpenClaw, Zed, LibreChat, Open WebUI, ...\n\n"
            f"  Base URL   {v1}\n"
            f"  API key    {key}   (any value; the app may insist on one)\n"
            f"  Model      {model}\n"
            f"  Context    {ctx}\n\n"
            "Apps that take environment variables:\n"
            + env_lines({"OPENAI_BASE_URL": v1, "OPENAI_API_BASE": v1, "OPENAI_API_KEY": key}) +
            f"\n\naider:  aider --model openai/{model}\n\n"
            "Anthropic-only apps: base URL " + conn["base"] + " (the /v1/messages endpoint).\n\n"
            "Apps whose only local option is 'Ollama' (its /api/chat protocol) cannot use colibri "
            "directly: colibri speaks the OpenAI and Anthropic APIs, not Ollama's. Cloud apps "
            "(a website, or an editor whose requests go through its vendor's servers) cannot "
            "reach 127.0.0.1 at all.\n\n"
            "Other computers on the network: set Address to 0.0.0.0 on the Run tab, Install / "
            "Apply, use this computer's IP address in the URL, and start the server with "
            "COLI_API_KEY set so a key is required.")
    raise KeyError(tool_id)


# ---------------------------------------------------------------- opening one


def agents_home():
    return os.path.join(setup_flow.setup_home(), "agents")


def launch_plan(tool_id, conn, home=None):
    """(environment additions, arguments after the program, files to write) for
    opening `tool_id` connected to colibri. Files go under colibri's own folder."""
    if not MODEL_RE.match(conn["model"]):
        raise ValueError("the server's model id is not known yet: start the server, then Refresh")
    if not KEY_RE.match(conn["key"]):
        raise ValueError("the API key may hold letters, digits, '.', '-' and '_' only")
    home = home or agents_home()
    if tool_id == "claude":
        return claude_env(conn), [], {}
    if tool_id == "opencode":
        return ({"OPENCODE_CONFIG_CONTENT": json.dumps(opencode_config(conn), separators=(",", ":"))},
                ["-m", f"colibri/{conn['model']}"], {})
    if tool_id == "codex":
        folder = os.path.join(home, "codex")
        return ({"CODEX_HOME": folder, "COLIBRI_API_KEY": conn["key"]}, [],
                {os.path.join(folder, "config.toml"): codex_toml(conn, default=True)})
    if tool_id == "hermes":
        folder = os.path.join(home, "hermes")
        return ({"HERMES_HOME": folder}, [], {os.path.join(folder, "config.yaml"): hermes_yaml(conn)})
    raise ValueError(f"{tool(tool_id)['name']} is not opened from here: follow its instructions")


def launcher_script(name, exe, args, folder, banner):
    """A .cmd that runs the app in `folder` and keeps the window open."""
    quoted = " ".join(f'"{a}"' for a in args)
    lines = ["@echo off", f"title colibri - {name}", f'cd /d "{folder}"',
             "echo " + banner.replace("%", "%%"), f'call "{exe}" {quoted}'.rstrip()]
    return "\r\n".join(lines) + "\r\n"


def open_in_terminal(tool_id, conn, folder, exe, base_env=None):
    """Open the app in a new console window, connected. Returns the script path."""
    if sys.platform != "win32":
        raise OSError("opening a terminal is done on Windows; elsewhere use the commands shown")
    if not os.path.isdir(folder):
        raise ValueError(f"{folder} is not a folder")
    env_add, args, files = launch_plan(tool_id, conn)
    for path, text in files.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
    entry = tool(tool_id)
    os.makedirs(agents_home(), exist_ok=True)
    script = os.path.join(agents_home(), f"{tool_id}.cmd")
    banner = f"colibri: {entry['name']} with {conn['model']} on {conn['base']}"
    with open(script, "w", encoding="utf-8", newline="") as handle:
        handle.write(launcher_script(entry["name"], exe, args, folder, banner))
    env = dict(os.environ if base_env is None else base_env)
    env.update(env_add)
    subprocess.Popen(["cmd.exe", "/k", script], cwd=folder, env=env,
                     creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
    return script


def start_bridge(conn, python=None):
    """Start `coli ui` in the background (no window) unless something answers there."""
    if answers(conn["bridge"] + "/"):
        return False
    flags = 0
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    log = os.path.join(setup_flow.setup_home(), "logs", "bridge.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "ab") as out:
        subprocess.Popen(bridge_command(conn, python=python), cwd=HERE, stdin=subprocess.DEVNULL,
                         stdout=out, stderr=subprocess.STDOUT, creationflags=flags,
                         start_new_session=sys.platform != "win32")
    return True
