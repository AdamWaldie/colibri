#!/usr/bin/env python3
"""colibri configurator: a window for the one-step setup.

`coli setup` already does the work (detect the hardware, pick a model, build or
fetch the engine, download with resume, write setup.json). This is a graphical
front end to it, made for Windows: double-click CONFIGURE.bat at the top of the
folder, or run `py -3 c\\configurator.py`.

    status strip   is the server running, which model, Open chat / Start / Stop
    1 Model        the catalog against this machine (what fits and why), or a
                   model folder that is already on disk
    2 Location     where models are downloaded (the drives and their free space),
                   extra shard folders, a Hugging Face token for gated models
    3 Options      backend (auto / CUDA / Vulkan / CPU), address, port, context
    4 Install      what will happen, in words, then `coli setup` with a progress bar
    Connect apps   Claude Code, Codex, opencode, Hermes, T3 Code and any
                   OpenAI-compatible app pointed at the running model
    This PC        the hardware report and where colibri keeps its files
    Details        the saved run configuration (setup.json) and the logs

Every action runs the real launcher (`coli setup ...`, `coli start`, `coli stop`,
`coli status`, `coli logs`); the command is shown and its output kept under
"Output", so whatever the window does can also be typed in a terminal. The window
keeps no settings of its own: setup.json, written by `coli setup`, stays the only
configuration. The token is passed to the child process only, never saved.

Under pythonw.exe there is no console: everything that starts a console program
(the hardware probes, the launcher) runs in a child without a window, so nothing
flashes on screen.
"""
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import webbrowser

HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import agent_connect  # noqa: E402
import model_fetch  # noqa: E402
import model_library  # noqa: E402
import setup_catalog  # noqa: E402
import setup_download  # noqa: E402
import setup_flow  # noqa: E402
import setup_hw  # noqa: E402
import updater  # noqa: E402
import win_shortcuts  # noqa: E402

GB = setup_catalog.GB
BACKENDS = ("auto", "cpu", "vulkan", "cuda")
BACKEND_WORDS = {"auto": "automatic (CUDA if it builds for the card, else Vulkan, else the CPU)",
                 "cuda": "NVIDIA CUDA", "vulkan": "Vulkan (any GPU with a Vulkan driver)", "cpu": "the CPU only"}
COLI = os.path.join(HERE, "coli")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0

FIT_WORDS = {"good": "Runs well", "tight": "Runs, tight on RAM", "needs-ram": "Needs more RAM",
             "needs-disk": "Needs more disk", "unsupported": "Not for this PC"}
PHASE_WORDS = {"starting": "Checking this PC", "build": "Building the engine",
               "download": "Downloading the model", "ready": "Set up", "started": "Started",
               "error": "Stopped with an error", "interrupted": "Interrupted"}
INSTALL_HINTS = {"claude": "npm install -g @anthropic-ai/claude-code",
                 "codex": "npm install -g @openai/codex",
                 "opencode": "npm install -g opencode-ai",
                 "hermes": "see Hermes Agent's install guide",
                 "aider": "python -m pip install aider-install, then aider-install"}


# ---------------------------------------------------------------- pure logic (tested)


def python_for_children(executable=None):
    """A console-less pythonw.exe runs the window; the launcher it starts should be
    python.exe next to it, so its own children (the engine, make) behave as in a
    terminal. Elsewhere, the interpreter running this file."""
    exe = executable or sys.executable or "python3"
    folder, name = os.path.split(exe)
    if name.lower() == "pythonw.exe":
        candidate = os.path.join(folder, "python.exe")
        if os.path.isfile(candidate):
            return candidate
    return exe


def setup_command(options, python="python", coli=COLI):
    """The `coli setup` command for the choices in the window.

    options keys (all optional):
        model        catalog id            -> --model ID
        model_dir    a model folder on disk (wins over model) -> --model-dir DIR
        models_root  download folder       -> --dir DIR
        backend      auto/cpu/vulkan/cuda  -> --backend (auto is the default, left out)
        host, port, ctx                    -> --host / --port / --ctx (defaults left out)
        start        "background" | "no"   -> --background | --no-start
        no_browser, no_verify, reconfigure -> the flags of the same name
    --yes is always passed: the window has already asked every question, and the
    child has no terminal to answer in."""
    cmd = [python, coli, "setup", "--yes"]
    model_dir = (options.get("model_dir") or "").strip()
    if model_dir:
        cmd += ["--model-dir", model_dir]
    elif options.get("model"):
        cmd += ["--model", options["model"]]
    root = (options.get("models_root") or "").strip()
    if root and not model_dir:
        cmd += ["--dir", root]
    backend = options.get("backend") or "auto"
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}")
    if backend != "auto":
        cmd += ["--backend", backend]
    host = (options.get("host") or "").strip()
    if host and host != setup_flow.DEFAULT_HOST:
        cmd += ["--host", host]
    port = str(options.get("port") or "").strip()
    if port:
        if not port.isdigit() or not 1 <= int(port) <= 65535:
            raise ValueError("the port must be a number between 1 and 65535")
        if int(port) != setup_flow.DEFAULT_PORT:
            cmd += ["--port", str(int(port))]
    ctx = str(options.get("ctx") or "").strip()
    if ctx:
        if not ctx.isdigit():
            raise ValueError("the context is a number of tokens, or 0 for automatic")
        if int(ctx):
            cmd += ["--ctx", str(int(ctx))]
    start = options.get("start") or "background"
    if start == "no":
        cmd.append("--no-start")
    elif start == "background":
        cmd.append("--background")
    else:
        raise ValueError(f"unknown start mode {start!r}")
    if options.get("no_browser"):
        cmd.append("--no-browser")
    if options.get("no_verify"):
        cmd.append("--no-verify")
    if options.get("reconfigure"):
        cmd.append("--reconfigure")
    return cmd


def child_env(extra=None, base=None):
    """The environment for a launcher run: unbuffered UTF-8 output (the window shows
    it live) without colour codes, plus the window's extra variables (the token,
    the extra shard folders)."""
    env = dict(os.environ if base is None else base)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["COLI_COLOR"] = "0"
    env["NO_COLOR"] = "1"
    for key, value in (extra or {}).items():
        if str(value).strip():
            env[key] = str(value).strip()
    return env


def extra_env(token="", split_name="", split_dirs=""):
    """The variables a run adds: HF_TOKEN for gated repositories, and the family's
    extra-shard-folder variable for a model split across drives."""
    extra = {}
    if token.strip():
        extra["HF_TOKEN"] = token.strip()
    if split_name and split_dirs.strip():
        extra[split_name] = split_dirs.strip()
    return extra


def display_command(cmd, extra=None, windows=None):
    """The command as a person would type it (`coli ...`), the token hidden."""
    windows = sys.platform == "win32" if windows is None else windows
    shown = list(cmd)
    if len(shown) >= 2 and os.path.basename(shown[1]) in ("coli", "setup_hw.py"):
        shown = (["coli"] if os.path.basename(shown[1]) == "coli" else ["python", shown[1]]) + shown[2:]
    if windows:
        text = subprocess.list2cmdline(shown)
    else:
        import shlex
        text = " ".join(shlex.quote(part) for part in shown)
    prefix = ""
    for key, value in (extra or {}).items():
        value = "<hidden>" if key == "HF_TOKEN" else value
        prefix += f'set "{key}={value}" && ' if windows else f"{key}={value} "
    return prefix + text


def detect_command(target, python="python"):
    """The hardware report as JSON, from a child process: on Windows the probes start
    PowerShell and nvidia-smi, which would each flash a console window if started
    from the window's own console-less process."""
    return [python, os.path.join(HERE, "setup_hw.py"), target, "--json"]


def kill_tree_command(pid):
    """Cancel on Windows: the launcher and everything it started (make, curl)."""
    return ["taskkill", "/PID", str(int(pid)), "/T", "/F"]


def catalog_rows(ram_total, free_bytes, downloaded=None, os_name=None, machine=None):
    """The catalog assessed for this machine, as plain rows for the model table."""
    rows = setup_catalog.recommend(ram_total, free_bytes,
                                   os_name=os_name or setup_flow.host_os(),
                                   machine=machine or setup_flow.host_machine(),
                                   downloaded=downloaded)
    return [setup_catalog.as_dict(row) for row in rows]


def downloaded_bytes(models_root):
    """Bytes already on disk per catalog id under models_root (a partial download
    needs less free space to finish)."""
    found = {}
    for entry in setup_catalog.catalog():
        folder = os.path.join(models_root, entry.id)
        manifest = setup_download.read_manifest(folder)
        if manifest:
            found[entry.id] = setup_download.bytes_present(
                folder, manifest.get("files", []), setup_flow.extra_shard_dirs(entry.family))
    return found


def space_note(entry, free):
    """One line on whether `entry` fits in `free` bytes ('' when either is unknown)."""
    if entry is None or free is None:
        return ""
    need = entry.disk_gb * GB * 1.02 + 2 * GB
    if free >= need:
        return f"{entry.name} needs {setup_hw.fmt_gb(need)}: fits"
    return f"{entry.name} needs {setup_hw.fmt_gb(need)}: NOT ENOUGH SPACE, choose another drive"


def fit_words(status):
    return FIT_WORDS.get(status, status)


def progress_from_state(state, now=None):
    """install-state.json as (fraction or None for 'busy', one line of text)."""
    if not state:
        return None, "Working..."
    phase = state.get("phase") or ""
    words = PHASE_WORDS.get(phase, phase.capitalize() or "Working")
    if phase == "download" and state.get("total"):
        done, total = state.get("done") or 0, state["total"]
        text = f"{words}: {setup_hw.fmt_gb(done)} of {setup_hw.fmt_gb(total)}"
        if state.get("rate"):
            text += f"  ({setup_download.human_rate(state['rate'])})"
        eta = state.get("eta_seconds")
        if eta:
            text += "  about " + (f"{eta / 3600:.1f} h" if eta >= 3600 else
                                  f"{max(1, round(eta / 60))} min") + " left"
        return min(1.0, done / total), text
    if phase == "error" and state.get("message"):
        return None, f"{words}: {state['message'].splitlines()[0]}"
    return None, words + "..."


def plan_lines(options, row=None, free=None):
    """What Install will do, in words, and the warnings to read first:
    ([lines], [warnings]). row is the model table's row for the chosen model."""
    lines, warnings = [], []
    model_dir = (options.get("model_dir") or "").strip()
    if model_dir:
        lines.append(f"Use the model already in {model_dir} (nothing to download).")
    elif row:
        folder = os.path.join(options.get("models_root") or setup_flow.default_models_root(), row["id"])
        lines.append(f"Download {row['name']}: {row['download_gb']:.0f} GB into {folder}.")
        lines.append("   The download resumes where it stopped if it is interrupted or cancelled.")
        if not row["fits"]:
            warnings.append(f"{row['name']}: {row['reason']}.")
        if row.get("license_note"):
            warnings.append(f"{row['name']}: {row['license_note']}.")
        entry = None
        try:
            entry = setup_catalog.by_id(row["id"])
        except KeyError:
            pass
        note = space_note(entry, free)
        if "NOT ENOUGH" in note:
            warnings.append(note + " (step 2).")
    else:
        warnings.append("Choose a model first (step 1).")
    lines.append(f"Get the engine for {BACKEND_WORDS.get(options.get('backend') or 'auto', '?')}: "
                 "the prebuilt one, or built on this PC when a compiler is installed.")
    host = (options.get("host") or setup_flow.DEFAULT_HOST).strip()
    port = str(options.get("port") or setup_flow.DEFAULT_PORT).strip()
    ctx = str(options.get("ctx") or "0").strip()
    lines.append(f"Serve it at {setup_flow.urls(host, port)['browser']} with "
                 + ("an automatic context." if ctx in ("", "0") else f"a {ctx}-token context."))
    if host in ("0.0.0.0", "::"):
        warnings.append("Address 0.0.0.0 makes the server reachable from other computers on the "
                        "network; start it with COLI_API_KEY set to require a key.")
    if options.get("start", "background") == "background":
        lines.append("Then start the server in the background"
                     + ("." if options.get("no_browser") else " and open the chat in the browser."))
    else:
        lines.append("Then stop: start it later with Start at the top.")
    return lines, warnings


def server_words(server, cfg):
    """The status strip: (colour, text) for the server state."""
    if not cfg:
        return "grey", "Not set up yet: choose a model, then Install (steps 1 to 4)"
    name = (cfg.get("model") or {}).get("name") or "the model"
    if cfg.get("status") != "ready":
        return "orange", f"{name}: setup not finished. Install continues where it stopped"
    state = (server or {}).get("state")
    if state == "ready":
        speed = server.get("tokens_per_second")
        return "green", f"Running: {name}" + (f"  ({speed} tok/s last answer)" if speed else "")
    if state == "loading":
        return "orange", f"Loading {name}... (large models take minutes)"
    return "grey", f"Stopped: {name} is set up. Press Start"


CODING_STEPS = (131072, 65536)      # what Claude Code and Codex want, then what they need
CODING_MIN = 65536
CODING_SHARE = 0.40                 # of the memory there is, for the conversation state


def coding_context(model_dir, available=None, resolve=None, geometry=None):
    """(context, words) for coding apps on this model and this PC: 131072 when its
    conversation state fits in CODING_SHARE of the memory there is (Claude Code can
    then compact by itself), else 65536 (the least Claude Code and Codex work with),
    else (None, why and what to do instead). The state is the model's own, from the
    planner colibri sizes its server with (resource_plan)."""
    import family_registry
    import resource_plan
    resolve = resolve or family_registry.resolve_model
    geometry = geometry or family_registry.planner_geometry
    if available is None:
        available = resource_plan.memory_available()
    resolved = resolve(model_dir)
    family = resolved.descriptor
    limit = family.limits.max_context
    if limit < CODING_MIN:
        return None, (f"{family.display_name} holds at most {limit:,} tokens; Claude Code and Codex "
                      f"need {CODING_MIN:,}. Choose a model with a longer context.")
    need_at = {}
    for step in CODING_STEPS:
        if step > limit:
            continue
        g = geometry(resolved, step)
        need = g.context_state_bytes + (g.workspace_bytes
                                        if family.id in resource_plan.CHAT_CONTEXT_WORKSPACE else 0)
        need_at[step] = need
        if need <= CODING_SHARE * available:
            return step, (f"{step:,} tokens: {setup_hw.fmt_gb(need)} of conversation memory, of "
                          f"{setup_hw.fmt_gb(available)} available")
    need = need_at.get(CODING_MIN)
    return None, (f"{CODING_MIN:,} tokens would take {setup_hw.fmt_gb(need)} of conversation memory, "
                  f"more than this PC can spare ({setup_hw.fmt_gb(available)} available). A smaller "
                  "model (Qwen3-Coder-30B, Qwen3.6-35B) suits coding apps here.")


def coding_advice(ctx):
    """One line on whether the server's context suits coding apps."""
    if not ctx:
        return "Start the server to see its context."
    if ctx < CODING_MIN:
        return (f"Context {ctx:,} tokens: too small for Claude Code and Codex (they need "
                f"{CODING_MIN:,}). Press Set up for coding apps.")
    if ctx < max(CODING_STEPS):
        return (f"Context {ctx:,} tokens: enough for Claude Code and Codex. "
                f"{max(CODING_STEPS):,} would also let Claude Code compact by itself.")
    return f"Context {ctx:,} tokens: ready for coding apps."


def app_menu_items(found, platform):
    """[(tool id, label, enabled)] for the top bar's Open app menu: every app the
    window can open, the ones not installed (or off Windows) shown but greyed."""
    items = []
    for entry in agent_connect.TOOLS:
        if not entry["launch"]:
            continue
        if platform != "win32":
            items.append((entry["id"], f"{entry['name']} (opens from Windows)", False))
        elif found.get(entry["id"]):
            items.append((entry["id"], entry["name"], True))
        else:
            items.append((entry["id"], f"{entry['name']}: not installed", False))
    return items


def waiting_words(name, seconds):
    minutes, rest = divmod(int(seconds), 60)
    return f"Loading the model for {name}: {minutes}:{rest:02d} (large models take minutes)"


def open_reason(entry, exe, platform, served):
    """'' when the app can be opened from the window, else why not, in words."""
    if not entry["launch"]:
        if entry.get("write"):
            return ("Write settings adds colibri to this app's configuration (a copy of the old "
                    "file is kept), or follow the instructions on the right.")
        return "Follow the instructions on the right for this app."
    if platform != "win32":
        return "Opening a terminal is for Windows; use the commands on the right."
    if not exe:
        hint = INSTALL_HINTS.get(entry["id"])
        return f"{entry['name']} is not installed (or not on PATH)" + (f": {hint}" if hint else ".")
    if not served:
        return "Start the server first (Start, at the top), then Refresh."
    return ""


def config_summary(cfg):
    """The saved run configuration as (label, value) lines."""
    if not cfg:
        return [("status", "not set up yet: choose a model, then Install")]
    model = cfg.get("model") or {}
    backend = cfg.get("backend") or cfg.get("requested_backend") or "?"
    asked = cfg.get("requested_backend")
    if asked and cfg.get("backend") and asked != cfg["backend"]:
        backend += f"  (asked for: {asked})"
    lines = [
        ("status", cfg.get("status") or "?"),
        ("model", (model.get("name") or "?") + (f"  ({model['id']})" if model.get("id") else "")),
        ("repository", model.get("repo") or "(a folder on disk)"),
        ("model folder", cfg.get("model_dir") or "?"),
        ("backend", backend),
        ("engine", cfg.get("engine") or "(not built yet)"),
        ("address", (f"{cfg.get('host') or setup_flow.DEFAULT_HOST}:"
                     f"{cfg.get('port') or setup_flow.DEFAULT_PORT}")),
        ("context", str(cfg.get("ctx") or "automatic")),
    ]
    for key, value in (cfg.get("env") or {}).items():
        lines.append((f"env {key}", str(value)))
    if cfg.get("failed_builds"):
        lines.append(("failed builds", ", ".join(cfg["failed_builds"])))
    if cfg.get("gpu_pending"):
        lines.append(("GPU pending", ", ".join(cfg["gpu_pending"])
                      + "  (install the GPU packages, then Install again)"))
    if cfg.get("updated"):
        lines.append(("saved", cfg["updated"]))
    return lines


LIBRARY_WORDS = {"yes": "Ready", "incomplete": "Download not finished", "import": "After Import",
                 "no": "No"}


def fetch_summary(plan, models_root):
    """What Look up found, in words: (text, can download)."""
    chosen = plan["chosen"]
    size = setup_hw.fmt_gb(chosen["size"])
    if not plan["supported"]:
        return (f"{plan['name']}: {plan['why']}. colibri can't run it, so it isn't offered "
                "(colibri's engines serve large mixture-of-experts families such as Qwen3 MoE, GLM, "
                "DeepSeek V4)."), False
    free = setup_hw.disk_free(models_root)
    text = (f"{plan['name']} ({chosen['label']}, {size} download): {plan['why']}.\n"
            f"Saved in {model_fetch.download_dir(plan['name'], models_root)}, "
            f"imported as {model_fetch.import_dir(plan['name'], chosen, models_root)}.")
    if free is not None and free < chosen["size"] + (1 << 30):
        return text + f"\nNOT ENOUGH SPACE there: {setup_hw.fmt_gb(free)} free.", False
    return text, True


def library_words(runs):
    return LIBRARY_WORDS.get(runs, runs)


def prefs_path():
    return os.path.join(setup_flow.setup_home(), "configurator.json")


def load_prefs(path=None):
    """The window's own small memory: the extra folders "My models" scans."""
    try:
        with open(path or prefs_path(), encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_prefs(prefs, path=None):
    path = path or prefs_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(prefs, handle, indent=2)
    except OSError:
        pass


def split_name_for(family_id):
    return setup_flow.split_dirs_name(family_id) if family_id else ""


def drive_report():
    """(root, free bytes, total bytes) for every place a model could go: the fixed
    drives on Windows, the home folder elsewhere."""
    roots = setup_hw.windows_fixed_drives() or [os.path.expanduser("~")]
    out = []
    for root in roots:
        try:
            usage = shutil.disk_usage(root)
            out.append((root, usage.free, usage.total))
        except OSError:
            continue
    return out


# ---------------------------------------------------------------- the window


def _no_console_streams():
    """pythonw.exe gives no stdout/stderr: a stray print would raise. Errors go to a
    log in the setup folder instead."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        try:
            path = setup_flow.log_path("configurator")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            sys.stderr = open(path, "a", encoding="utf-8")
        except OSError:
            sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _dpi_aware():
    """Sharp text on a scaled Windows display, instead of a blurry bitmap stretch."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def main():  # pragma: no cover - the window itself is exercised by hand and by screenshots
    _no_console_streams()
    _dpi_aware()
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("colibri")
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    width, height = min(1100, sw - 60), min(780, sh - 80)
    root.geometry(f"{width}x{height}+{max(0, (sw - width) // 2)}+{max(0, (sh - height) // 3)}")
    root.minsize(min(860, width), min(600, height))
    icon = os.path.join(HERE, "..", "desktop", "src-tauri", "icons", "icon.ico")
    if sys.platform == "win32" and os.path.isfile(icon):
        try:
            root.iconbitmap(icon)
        except tk.TclError:
            pass
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    base_font = ("Segoe UI", 10) if sys.platform == "win32" else ("TkDefaultFont", 10)
    style.configure("Head.TLabel", font=(base_font[0], 12, "bold"))
    style.configure("Status.TLabel", font=(base_font[0], 11, "bold"))
    style.configure("Hint.TLabel", foreground="#5f6368")
    style.configure("Warn.TLabel", foreground="#b00020")
    style.configure("Big.TButton", font=(base_font[0], 11, "bold"), padding=(18, 8))

    def report_error(exc, value, tb):
        text = "".join(traceback.format_exception(exc, value, tb))
        print(text, file=sys.stderr, flush=True)
        messagebox.showerror("colibri", f"Something went wrong:\n\n{value}\n\nDetails: "
                                        f"{setup_flow.log_path('configurator')}")

    root.report_callback_exception = report_error

    saved = setup_flow.load_config() or {}
    saved_model = (saved.get("model") or {}).get("id") or ""
    saved_local = bool(saved.get("model_dir")) and not saved_model
    state = {"hw": None, "rows": [], "proc": None, "proc_kind": "", "cfg": saved, "server": None,
             "ready": False, "drives": [], "apps_models": [], "apps_found": {}, "closing": False}
    events = queue.Queue()
    py = python_for_children()

    v_model = tk.StringVar(value=saved_model)
    v_use_local = tk.BooleanVar(value=saved_local)
    v_model_dir = tk.StringVar(value=saved.get("model_dir", "") if saved_local else "")
    v_root = tk.StringVar(value=saved.get("models_root")
                          or (os.path.dirname(saved["model_dir"]) if saved_model and saved.get("model_dir") else "")
                          or setup_flow.default_models_root())
    v_backend = tk.StringVar(value=saved.get("requested_backend") or "auto")
    v_host = tk.StringVar(value=saved.get("host") or setup_flow.DEFAULT_HOST)
    v_port = tk.StringVar(value=str(saved.get("port") or setup_flow.DEFAULT_PORT))
    v_ctx = tk.StringVar(value=str(saved.get("ctx") or 0))
    v_start = tk.StringVar(value="background")
    v_no_browser = tk.BooleanVar(value=False)
    v_no_verify = tk.BooleanVar(value=False)
    v_show_all = tk.BooleanVar(value=False)
    v_token = tk.StringVar(value="")
    family = saved.get("family")
    v_split = tk.StringVar(value=(saved.get("env") or {}).get(split_name_for(family), "") if family else "")
    v_status = tk.StringVar(value="")
    v_cmd = tk.StringVar(value="")
    v_server = tk.StringVar(value="")

    # ---- status strip
    strip = ttk.Frame(root, padding=(12, 10, 12, 6))
    strip.pack(fill="x")
    dot = tk.Canvas(strip, width=16, height=16, highlightthickness=0, bg=root.cget("bg"))
    dot_id = dot.create_oval(2, 2, 14, 14, fill="grey", outline="")
    dot.pack(side="left")
    ttk.Label(strip, textvariable=v_server, style="Status.TLabel").pack(side="left", padx=8)
    b_stop = ttk.Button(strip, text="Stop", command=lambda: run_coli(["stop"], "stop"))
    b_stop.pack(side="right")
    b_start = ttk.Button(strip, text="Start", command=lambda: run_coli(
        ["start", "--background"] + (["--no-browser"] if v_no_browser.get() else []), "start"))
    b_start.pack(side="right", padx=4)
    b_chat = ttk.Button(strip, text="Open chat", command=lambda: open_dashboard())
    b_chat.pack(side="right")
    # Claude Code, Codex, opencode... on the local model, in one click: the server is
    # started and waited for when needed, then the app opens in the project folder.
    b_apps = ttk.Menubutton(strip, text="Open app")
    apps_menu = tk.Menu(b_apps, tearoff=False)
    b_apps["menu"] = apps_menu
    b_apps.pack(side="right", padx=(0, 8))
    ttk.Separator(root).pack(fill="x")

    # ---- bottom: progress, status, output (packed before the tabs so it keeps its room)
    bottom = ttk.Frame(root, padding=(12, 6, 12, 8))
    bottom.pack(fill="x", side="bottom")
    bar_row = ttk.Frame(bottom)
    bar_row.pack(fill="x")
    progress = ttk.Progressbar(bar_row, mode="determinate", maximum=1000, length=240)
    progress.pack(side="left")
    # the buttons first: a long status line is cut, never the buttons
    b_toggle = ttk.Button(bar_row, text="Show output", command=lambda: toggle_output())
    b_toggle.pack(side="right")
    b_cancel = ttk.Button(bar_row, text="Cancel", command=lambda: cancel(), state="disabled")
    b_cancel.pack(side="right", padx=4)
    ttk.Label(bar_row, textvariable=v_status).pack(side="left", padx=10, fill="x", expand=True)
    out_frame = ttk.Frame(bottom)
    log = tk.Text(out_frame, height=12, wrap="none", font=("Consolas", 9), background="#111111",
                  foreground="#dddddd", insertbackground="#dddddd")
    log_y = ttk.Scrollbar(out_frame, orient="vertical", command=log.yview)
    log.configure(yscrollcommand=log_y.set)
    log_y.pack(side="right", fill="y")
    log.pack(side="left", fill="both", expand=True)
    out_tools = ttk.Frame(bottom)
    ttk.Button(out_tools, text="Clear", command=lambda: log.delete("1.0", "end")).pack(side="right")
    ttk.Button(out_tools, text="Copy all", command=lambda: (
        root.clipboard_clear(), root.clipboard_append(log.get("1.0", "end-1c")))).pack(side="right", padx=4)

    # ---- tabs
    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=12, pady=(8, 0))

    def tab(title):
        frame = ttk.Frame(nb, padding=12)
        nb.add(frame, text=title)
        return frame

    def form_row(parent, r, label, var, browse=None, show="", hint="", width=None):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", pady=4)
        entry = ttk.Entry(parent, textvariable=var, show=show, width=width or 0)
        entry.grid(row=r, column=1, sticky="w" if width else "ew", padx=6)
        if browse:
            ttk.Button(parent, text="Browse...", command=browse).grid(row=r, column=2)
        if hint:
            ttk.Label(parent, text=hint, style="Hint.TLabel", wraplength=760,
                      justify="left").grid(row=r + 1, column=1, columnspan=2, sticky="w", padx=6)
        return entry

    def next_button(parent, label, target):
        ttk.Button(parent, text=label, command=lambda: nb.select(target)).pack(side="right")

    # 1 Model
    t_model = tab(" 1  Model ")
    top = ttk.Frame(t_model)
    top.pack(fill="x")
    ttk.Label(top, text="Choose a model", style="Head.TLabel").pack(side="left")
    ttk.Checkbutton(top, text="also show models that do not fit this PC", variable=v_show_all,
                    command=lambda: fill_models()).pack(side="right")
    ttk.Label(t_model, style="Hint.TLabel", wraplength=1000, justify="left",
              text="Checked against this PC's memory and disk. Small models run from RAM and answer "
                   "quickly; large ones keep part of the model on the SSD and are much slower.").pack(anchor="w")
    ttk.Radiobutton(t_model, text="Download one from the list", variable=v_use_local, value=False,
                    command=lambda: refresh_mode()).pack(anchor="w", pady=(8, 0))
    models = ttk.Treeview(t_model, columns=("size", "ram", "fit", "runs"),
                          show="tree headings", height=8, selectmode="browse")
    models.heading("#0", text="Model")
    models.column("#0", width=280)
    for col, text, width_ in (("size", "Download", 90), ("ram", "RAM needed / best", 130),
                              ("fit", "On this PC", 140), ("runs", "Runs", 220)):
        models.heading(col, text=text)
        models.column(col, width=width_, anchor="w")
    models.tag_configure("rec", font=(base_font[0], 10, "bold"))
    models.tag_configure("nofit", foreground="#9aa0a6")
    models.insert("", "end", iid="__wait", text="Checking this PC...", values=("", "", "", ""))
    models.pack(fill="both", expand=True, pady=4)
    model_info = ttk.Label(t_model, text="", wraplength=1000, justify="left")
    model_info.pack(anchor="w", fill="x")
    local = ttk.Frame(t_model)
    local.pack(fill="x", pady=(8, 0))
    ttk.Radiobutton(local, text="Or use a model folder already on this PC:", variable=v_use_local,
                    value=True, command=lambda: refresh_mode()).pack(side="left")
    local_entry = ttk.Entry(local, textvariable=v_model_dir)
    local_entry.pack(side="left", fill="x", expand=True, padx=6)
    local_btn = ttk.Button(local, text="Browse...",
                           command=lambda: pick_dir(v_model_dir, "The model folder (with config.json)"))
    local_btn.pack(side="left")
    nav = ttk.Frame(t_model)
    nav.pack(fill="x", pady=(10, 0))

    # 2 Location
    t_loc = tab(" 2  Location ")
    ttk.Label(t_loc, text="Where models are saved", style="Head.TLabel").pack(anchor="w")
    ttk.Label(t_loc, style="Hint.TLabel", wraplength=1000, justify="left",
              text="Pick the fastest internal SSD with room (double-click a drive). Large models are "
                   "read from it while they answer: never a network or USB drive.").pack(anchor="w")
    drives = ttk.Treeview(t_loc, columns=("free", "total"), height=4)
    drives.heading("#0", text="Drive")
    drives.heading("free", text="Free")
    drives.heading("total", text="Size")
    drives.pack(fill="x", pady=6)
    loc = ttk.Frame(t_loc)
    loc.pack(fill="x")
    loc.columnconfigure(1, weight=1)
    form_row(loc, 0, "Models folder", v_root, browse=lambda: pick_dir(v_root, "Where to save models"),
             hint="Each model gets its own subfolder here, for example D:\\colibri-models\\qwen36-35b.")
    free_label = ttk.Label(loc, text="")
    free_label.grid(row=2, column=1, sticky="w", padx=6)
    adv = ttk.LabelFrame(t_loc, text="Only if you need them", padding=8)
    adv.pack(fill="x", pady=(12, 0))
    adv.columnconfigure(1, weight=1)
    form_row(adv, 0, "Hugging Face token", v_token, show="*",
             hint="For gated models only. Used for this install and never saved; "
                  "`hf auth login` stores one permanently instead.")
    split_entry = form_row(adv, 2, "Extra shard folders", v_split,
                           hint="For a very large model split across drives, when its engine supports it. "
                                "Separate folders with ';'.")
    nav = ttk.Frame(t_loc)
    nav.pack(fill="x", side="bottom")

    # 3 Options
    t_run = tab(" 3  Options ")
    ttk.Label(t_run, text="How it runs", style="Head.TLabel").pack(anchor="w")
    ttk.Label(t_run, style="Hint.TLabel", text="The defaults suit most PCs.").pack(anchor="w", pady=(0, 8))
    opts = ttk.Frame(t_run)
    opts.pack(fill="x")
    opts.columnconfigure(1, weight=1)
    ttk.Label(opts, text="Use").grid(row=0, column=0, sticky="nw", pady=4)
    bk = ttk.Frame(opts)
    bk.grid(row=0, column=1, sticky="w", padx=6)
    for value, text in (("auto", "Automatic (recommended)"), ("cuda", "NVIDIA GPU (CUDA)"),
                        ("vulkan", "Any GPU (Vulkan)"), ("cpu", "CPU only")):
        ttk.Radiobutton(bk, text=text, value=value, variable=v_backend,
                        command=lambda: update_plan()).pack(side="left", padx=(0, 14))
    ttk.Label(opts, style="Hint.TLabel", wraplength=760, justify="left",
              text="Automatic tries CUDA, then Vulkan, then the CPU, and falls back if a build fails."
              ).grid(row=1, column=1, sticky="w", padx=6)
    form_row(opts, 2, "Context (tokens)", v_ctx, width=10,
             hint="How much text the model keeps in mind. 0 = automatic. Coding apps such as Hermes "
                  "need 65536 or more.")
    form_row(opts, 4, "Port", v_port, width=8)
    form_row(opts, 5, "Address", v_host, width=18,
             hint="127.0.0.1 = this PC only. 0.0.0.0 = other computers on your network too.")
    ttk.Label(opts, text="After installing").grid(row=7, column=0, sticky="w", pady=(12, 4))
    st = ttk.Frame(opts)
    st.grid(row=7, column=1, sticky="w", padx=6, pady=(12, 4))
    ttk.Radiobutton(st, text="Start the server", value="background", variable=v_start,
                    command=lambda: update_plan()).pack(side="left", padx=(0, 14))
    ttk.Radiobutton(st, text="Don't start it yet", value="no", variable=v_start,
                    command=lambda: update_plan()).pack(side="left")
    ttk.Checkbutton(opts, text="Don't open the chat in the browser", variable=v_no_browser,
                    command=lambda: update_plan()).grid(row=8, column=1, sticky="w", padx=6)
    ttk.Checkbutton(opts, text="Skip checking downloaded files (faster, less safe)",
                    variable=v_no_verify, command=lambda: update_plan()).grid(row=9, column=1, sticky="w", padx=6)
    nav = ttk.Frame(t_run)
    nav.pack(fill="x", side="bottom")

    # 4 Install
    t_inst = tab(" 4  Install ")
    inst_head = ttk.Label(t_inst, text="Ready to install", style="Head.TLabel")
    inst_head.pack(anchor="w")
    plan_box = ttk.Label(t_inst, text="", justify="left", wraplength=1000)
    plan_box.pack(anchor="w", pady=(8, 4))
    warn_box = ttk.Label(t_inst, text="", justify="left", wraplength=1000, style="Warn.TLabel")
    warn_box.pack(anchor="w")
    b_apply = ttk.Button(t_inst, text="Install", style="Big.TButton", command=lambda: apply())
    b_apply.pack(anchor="w", pady=14)
    ttk.Label(t_inst, style="Hint.TLabel", wraplength=1000, justify="left",
              text="Safe to cancel or close the window at any time: Install again continues where it "
                   "stopped. Already installed? Install just checks and starts it.").pack(anchor="w")
    cmd_row = ttk.Frame(t_inst)
    cmd_row.pack(fill="x", side="bottom")
    ttk.Label(cmd_row, text="Same as typing:", style="Hint.TLabel").pack(side="left")
    ttk.Entry(cmd_row, textvariable=v_cmd, state="readonly", font=("Consolas", 9)
              ).pack(side="left", fill="x", expand=True, padx=6)
    ttk.Button(cmd_row, text="Copy", command=lambda: (root.clipboard_clear(), root.clipboard_append(v_cmd.get()))
               ).pack(side="left")

    # My models
    t_lib = tab(" My models ")
    ttk.Label(t_lib, text="Models already on this PC", style="Head.TLabel").pack(anchor="w")
    ttk.Label(t_lib, style="Hint.TLabel", wraplength=1000, justify="left",
              text="colibri's own models on every drive, and GGUF models from Ollama, LM Studio and the "
                   "Hugging Face cache. A GGUF runs in colibri after Import (the file stays where it is) "
                   "when its architecture is one colibri's engines serve; the others say why not."
              ).pack(anchor="w")
    get_box = ttk.LabelFrame(t_lib, text="Get a new model", padding=8)
    get_box.pack(fill="x", pady=(8, 0))
    v_get = tk.StringVar(value="")
    v_variant = tk.StringVar(value="")
    get_row = ttk.Frame(get_box)
    get_row.pack(fill="x")
    get_entry = ttk.Entry(get_row, textvariable=v_get)
    get_entry.pack(side="left", fill="x", expand=True)
    b_lookup = ttk.Button(get_row, text="Look up", command=lambda: look_up())
    b_lookup.pack(side="left", padx=6)
    get_variant = ttk.Combobox(get_row, textvariable=v_variant, width=22, state="disabled")
    get_variant.pack(side="left")
    b_fetch = ttk.Button(get_row, text="Download and use", state="disabled", command=lambda: fetch_model())
    b_fetch.pack(side="left", padx=6)
    get_info = ttk.Label(get_box, style="Hint.TLabel", wraplength=980, justify="left",
                         text="An Ollama name (qwen3-coder:30b), a Hugging Face repository "
                              "(unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF, or hf.co/owner/repo:Q4_K_M), "
                              "or a link to a .gguf file. Checked before anything is downloaded.")
    get_info.pack(anchor="w", pady=(6, 0))
    get_entry.bind("<Return>", lambda _e: look_up())
    lib_bar = ttk.Frame(t_lib)
    lib_bar.pack(fill="x", pady=(8, 0))
    ttk.Button(lib_bar, text="Scan again", command=lambda: scan_models()).pack(side="left")
    ttk.Button(lib_bar, text="Add a folder to scan...", command=lambda: add_scan_folder()).pack(side="left", padx=6)
    lib_status = ttk.Label(lib_bar, text="", style="Hint.TLabel")
    lib_status.pack(side="left", padx=8)
    lib = ttk.Treeview(t_lib, columns=("source", "size", "runs", "where"), show="tree headings",
                       height=9, selectmode="browse")
    lib.heading("#0", text="Model")
    lib.column("#0", width=260)
    for col, text, width_ in (("source", "From", 100), ("size", "Size", 80), ("runs", "In colibri", 150),
                              ("where", "Where", 380)):
        lib.heading(col, text=text)
        lib.column(col, width=width_, anchor="w")
    lib.tag_configure("no", foreground="#9aa0a6")
    lib.tag_configure("current", font=(base_font[0], 10, "bold"))
    lib.pack(fill="both", expand=True, pady=6)
    lib_info = ttk.Label(t_lib, text="", wraplength=1000, justify="left")
    lib_info.pack(anchor="w")
    lib_btns = ttk.Frame(t_lib)
    lib_btns.pack(fill="x", pady=(6, 0))
    b_use = ttk.Button(lib_btns, text="Use this model", command=lambda: use_model())
    b_use.pack(side="left")
    b_import = ttk.Button(lib_btns, text="Import and use", command=lambda: import_model())
    b_import.pack(side="left", padx=6)
    b_move = ttk.Button(lib_btns, text="Move to another drive...", command=lambda: move_selected())
    b_move.pack(side="left")
    b_open = ttk.Button(lib_btns, text="Open folder", command=lambda: open_selected())
    b_open.pack(side="left", padx=6)
    lib_state = {"rows": [], "prefs": load_prefs(), "plan": None}

    # Connect apps
    t_apps = tab(" Connect apps ")
    v_app_model = tk.StringVar(value="")
    v_app_key = tk.StringVar(value="local")
    v_app_folder = tk.StringVar(value=os.path.expanduser("~"))
    ttk.Label(t_apps, text="Use this model from a coding app", style="Head.TLabel").pack(anchor="w")
    ttk.Label(t_apps, style="Hint.TLabel", wraplength=1000, justify="left",
              text="Like a local Ollama: while the server runs, these apps can use it instead of a cloud "
                   "model. Pick an app; Open in terminal starts it in your project already connected, "
                   "without changing its own settings.").pack(anchor="w")
    coding_row = ttk.Frame(t_apps)
    coding_row.pack(fill="x", pady=(6, 0))
    coding_info = ttk.Label(coding_row, text="", wraplength=760, justify="left")
    coding_info.pack(side="left", fill="x", expand=True)
    b_coding = ttk.Button(coding_row, text="Set up for coding apps", command=lambda: set_up_coding())
    b_coding.pack(side="right")
    head = ttk.Frame(t_apps)
    head.pack(fill="x", pady=(8, 0))
    ttk.Label(head, text="Model").pack(side="left")
    app_model = ttk.Combobox(head, textvariable=v_app_model, width=30)
    app_model.pack(side="left", padx=(4, 12))
    ttk.Label(head, text="API key").pack(side="left")
    ttk.Entry(head, textvariable=v_app_key, width=12).pack(side="left", padx=(4, 12))
    ttk.Button(head, text="Refresh", command=lambda: refresh_apps()).pack(side="left")
    app_server = ttk.Label(head, text="", style="Hint.TLabel")
    app_server.pack(side="left", padx=10)
    body = ttk.PanedWindow(t_apps, orient="horizontal")
    body.pack(fill="both", expand=True, pady=6)
    app_list = ttk.Treeview(body, columns=("state",), show="tree headings", height=7, selectmode="browse")
    app_list.heading("#0", text="App")
    app_list.heading("state", text="On this PC")
    app_list.column("#0", width=200)
    app_list.column("state", width=100)
    for entry in agent_connect.TOOLS:
        app_list.insert("", "end", iid=entry["id"], text=entry["name"], values=("",))
    body.add(app_list, weight=1)
    right = ttk.Frame(body)
    body.add(right, weight=3)
    app_text = tk.Text(right, wrap="word", font=("Consolas", 9), height=10, relief="flat")
    app_ysb = ttk.Scrollbar(right, orient="vertical", command=app_text.yview)
    app_text.configure(yscrollcommand=app_ysb.set)
    app_ysb.pack(side="right", fill="y")
    app_text.pack(side="left", fill="both", expand=True)
    foot = ttk.Frame(t_apps)
    foot.pack(fill="x")
    ttk.Label(foot, text="Project folder").pack(side="left")
    ttk.Entry(foot, textvariable=v_app_folder).pack(side="left", fill="x", expand=True, padx=6)
    ttk.Button(foot, text="Browse...", command=lambda: pick_dir(v_app_folder, "The project to open the app in")
               ).pack(side="left")
    b_app_open = ttk.Button(foot, text="Open in terminal", command=lambda: open_app())
    b_app_open.pack(side="left", padx=(12, 4))
    b_bridge = ttk.Button(foot, text="Start Codex bridge", command=lambda: start_bridge())
    b_bridge.pack(side="left")
    b_app_write = ttk.Button(foot, text="Write settings", command=lambda: write_app_settings())
    b_app_write.pack(side="left", padx=(4, 0))
    ttk.Button(foot, text="Copy instructions", command=lambda: (
        root.clipboard_clear(), root.clipboard_append(app_text.get("1.0", "end-1c")))).pack(side="left", padx=4)
    app_reason = ttk.Label(t_apps, text="", style="Hint.TLabel")
    app_reason.pack(anchor="w", pady=(4, 0))

    # This PC
    t_pc = tab(" This PC ")
    ttk.Label(t_pc, text="What colibri sees on this PC", style="Head.TLabel").pack(anchor="w")
    hw_text = tk.Text(t_pc, height=9, wrap="word", font=("Consolas", 10), relief="flat")
    hw_text.pack(fill="x", pady=(6, 10))
    hw_text.insert("1.0", "Checking the hardware (the GPU check takes a few seconds)...")
    hw_text.configure(state="disabled")
    paths = ttk.Frame(t_pc)
    paths.pack(fill="x")
    paths.columnconfigure(1, weight=1)
    for r, (label, path) in enumerate((("Settings and logs", setup_flow.setup_home()),
                                       ("colibri folder", os.path.dirname(HERE)))):
        ttk.Label(paths, text=label).grid(row=r, column=0, sticky="w", pady=3)
        ttk.Label(paths, text=path).grid(row=r, column=1, sticky="w", padx=6)
        ttk.Button(paths, text="Open", command=lambda p=path: open_path(p)).grid(row=r, column=2)
    ttk.Button(t_pc, text="Check again", command=lambda: detect()).pack(anchor="w", pady=10)
    sc_box = ttk.LabelFrame(t_pc, text="Shortcut", padding=8)
    sc_box.pack(fill="x", pady=(6, 0))
    sc_info = ttk.Label(sc_box, text="", wraplength=900, justify="left")
    sc_info.pack(anchor="w")
    sc_row = ttk.Frame(sc_box)
    sc_row.pack(anchor="w", pady=(6, 0))
    b_sc_add = ttk.Button(sc_row, text="Add colibri to the Start menu and desktop",
                          command=lambda: shortcuts("create"))
    b_sc_add.pack(side="left")
    b_sc_remove = ttk.Button(sc_row, text="Remove the shortcuts", command=lambda: shortcuts("remove"))
    b_sc_remove.pack(side="left", padx=6)
    up_box = ttk.LabelFrame(t_pc, text="Updates", padding=8)
    up_box.pack(fill="x", pady=(6, 0))
    up_info = ttk.Label(up_box, text="", wraplength=900, justify="left")
    up_info.pack(anchor="w")
    up_row = ttk.Frame(up_box)
    up_row.pack(anchor="w", pady=(6, 0))
    ttk.Button(up_row, text="Check for updates", command=lambda: check_updates()).pack(side="left")
    b_update = ttk.Button(up_row, text="Update now", state="disabled", command=lambda: update_now())
    b_update.pack(side="left", padx=6)
    b_download = ttk.Button(up_row, text="Open the download page", state="disabled",
                            command=lambda: webbrowser.open(lib_state.get("updates", {}).get("url")
                                                            or updater.RELEASES_PAGE))

    # Details
    t_cur = tab(" Details ")
    cur = ttk.Treeview(t_cur, columns=("value",), show="tree", height=14)
    cur.column("#0", width=170)
    cur.column("value", width=780)
    cur.pack(fill="both", expand=True)
    cb = ttk.Frame(t_cur)
    cb.pack(fill="x", pady=6)
    ttk.Button(cb, text="Reload", command=lambda: load_current()).pack(side="left")
    ttk.Button(cb, text="Open setup.json", command=lambda: open_path(setup_flow.config_path())
               ).pack(side="left", padx=6)
    ttk.Button(cb, text="Server log", command=lambda: run_coli(["logs", "-n", "200"], "show")).pack(side="left")
    ttk.Button(cb, text="Install log", command=lambda: run_coli(["logs", "--install", "-n", "200"], "show")
               ).pack(side="left", padx=6)
    ttk.Button(cb, text="Full status", command=lambda: run_coli(["status"], "show")).pack(side="left")

    # navigation between the numbered steps
    for frame, label, target in ((t_model, "Next: location  >", t_loc), (t_loc, "Next: options  >", t_run),
                                 (t_run, "Next: install  >", t_inst)):
        holder = [c for c in frame.winfo_children() if isinstance(c, ttk.Frame)][-1]
        next_button(holder, label, target)

    # ---- behaviour
    def pick_dir(var, title):
        start = var.get() if os.path.isdir(var.get() or "") else os.path.expanduser("~")
        chosen = filedialog.askdirectory(title=title, initialdir=start, mustexist=False)
        if chosen:
            var.set(os.path.normpath(chosen))

    def open_path(path):
        target = path if os.path.exists(path) else os.path.dirname(path)
        if not os.path.exists(target):
            messagebox.showinfo("colibri", f"{path} does not exist yet: install first.")
            return
        if sys.platform == "win32":
            os.startfile(target)  # a folder or file the user asked to open
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", target])

    def open_dashboard():
        cfg = state["cfg"] or {}
        host = cfg.get("host") or v_host.get() or setup_flow.DEFAULT_HOST
        port = cfg.get("port") or v_port.get() or setup_flow.DEFAULT_PORT
        webbrowser.open(setup_flow.urls(host, port)["browser"])

    def show_output(on=True):
        if on and not out_frame.winfo_ismapped():
            out_tools.pack(fill="x", side="bottom", pady=(4, 0))
            out_frame.pack(fill="both", expand=True, pady=(6, 0))
            b_toggle.configure(text="Hide output")
        elif not on and out_frame.winfo_ismapped():
            out_frame.pack_forget()
            out_tools.pack_forget()
            b_toggle.configure(text="Show output")

    def toggle_output():
        show_output(not out_frame.winfo_ismapped())

    def log_write(text):
        # A carriage return is a progress line redrawn in place.
        for i, part in enumerate(text.split("\r")):
            if i:
                log.delete("end-1c linestart", "end-1c")
            log.insert("end", part)
        if int(log.index("end-1c").split(".")[0]) > 5000:
            log.delete("1.0", "1000.0")
        log.see("end")

    def options():
        return {"model": v_model.get(), "model_dir": v_model_dir.get() if v_use_local.get() else "",
                "models_root": v_root.get(), "backend": v_backend.get(), "host": v_host.get(),
                "port": v_port.get(), "ctx": v_ctx.get(), "start": v_start.get(),
                "no_browser": v_no_browser.get(), "no_verify": v_no_verify.get()}

    def current_family():
        if v_use_local.get():
            cfg = state["cfg"]
            return cfg.get("family") if cfg.get("model_dir") == v_model_dir.get() else ""
        try:
            return setup_catalog.by_id(v_model.get()).family if v_model.get() else ""
        except KeyError:
            return ""

    def run_extra():
        return extra_env(v_token.get(), split_name_for(current_family()), v_split.get())

    def chosen_row():
        if v_use_local.get():
            return None
        return next((r for r in state["rows"] if r["id"] == v_model.get()), None)

    def update_plan(*_):
        try:
            v_cmd.set(display_command(setup_command(options(), python=py), run_extra()))
            valid = True
        except ValueError as error:
            v_cmd.set(f"(not valid yet: {error})")
            valid = False
        split_entry.configure(state="normal" if split_name_for(current_family()) else "disabled")
        path = v_root.get()
        free = setup_hw.disk_free(path) if path else None
        try:
            entry = None if v_use_local.get() or not v_model.get() else setup_catalog.by_id(v_model.get())
        except KeyError:
            entry = None
        note = space_note(entry, free)
        text = f"{setup_hw.fmt_gb(free)} free there" if free is not None else ""
        free_label.configure(text=f"{text}; {note}" if note else text,
                             style="Warn.TLabel" if "NOT ENOUGH" in note else "TLabel")
        lines, warnings = plan_lines(options(), chosen_row(), free)
        if not valid:
            warnings.insert(0, v_cmd.get())
        plan_box.configure(text="\n".join(f"  {i}.  {line}" if not line.startswith("   ") else line
                                          for i, line in zip(_numbers(lines), lines)))
        warn_box.configure(text="\n".join(f"  !  {w}" for w in warnings))
        busy = state["proc"] is not None
        if busy and state["proc_kind"] == "setup":
            inst_head.configure(text="Installing...  (progress at the bottom)")
        elif state["ready"]:
            inst_head.configure(text="Installed. Install again to apply changed choices; "
                                     "Open chat or Connect apps to use it")
        else:
            inst_head.configure(text="Ready to install")
        b_apply.configure(state="disabled" if busy or not valid or
                          (not v_use_local.get() and not v_model.get()) else "normal")

    def _numbers(lines):
        n = 0
        for line in lines:
            if not line.startswith("   "):
                n += 1
            yield n

    for var in (v_root, v_host, v_port, v_ctx, v_model_dir, v_token, v_split, v_model):
        var.trace_add("write", update_plan)

    def refresh_mode():
        mode = "normal" if v_use_local.get() else "disabled"
        local_entry.configure(state=mode)
        local_btn.configure(state=mode)
        update_plan()

    def fill_models():
        models.delete(*models.get_children())
        hidden = 0
        for r in state["rows"]:
            if not r["fits"] and not v_show_all.get() and r["id"] != v_model.get():
                hidden += 1
                continue
            tags = ("rec",) if r["recommended"] else (() if r["fits"] else ("nofit",))
            models.insert("", "end", iid=r["id"], tags=tags,
                          text=r["name"] + ("   (recommended)" if r["recommended"] else ""),
                          values=(f"{r['download_gb']:.0f} GB", f"{r['ram_min_gb']:g} / {r['ram_good_gb']:g} GB",
                                  fit_words(r["status"]),
                                  "makes images" if r["kind"] == "image" else r["runs"]))
        if hidden:
            models.insert("", "end", iid="__more", tags=("nofit",),
                          text=f"+ {hidden} more (too big for this PC)",
                          values=("", "", "", ""))
        if not v_model.get() and not v_use_local.get():
            v_model.set(next((r["id"] for r in state["rows"] if r["recommended"]), ""))
        if v_model.get() and models.exists(v_model.get()):
            models.selection_set(v_model.get())
            models.see(v_model.get())

    def on_select(_event=None):
        sel = [s for s in models.selection() if not s.startswith("__")]
        if not sel:
            return
        iid = sel[0]
        r = next((x for x in state["rows"] if x["id"] == iid), None)
        if r:
            text = f"{r['name']}: {r['summary']}.\n{fit_words(r['status'])}: {r['reason']}."
            if r["license_note"]:
                text += f"\nLicense: {r['license_note']}"
            model_info.configure(text=text)
        if _event is not None:
            v_model.set(iid)
            if v_use_local.get():
                v_use_local.set(False)
                refresh_mode()
        update_plan()

    models.bind("<<TreeviewSelect>>", on_select)
    models.bind("<Double-1>", lambda _e: nb.select(t_loc))

    def on_drive(_event):
        sel = drives.selection()
        if sel:
            v_root.set(os.path.join(drives.item(sel[0], "text"), "colibri-models"))

    drives.bind("<Double-1>", on_drive)

    def load_current():
        state["cfg"] = cfg = setup_flow.load_config() or {}
        cur.delete(*cur.get_children())
        for label, value in config_summary(cfg):
            cur.insert("", "end", text=label, values=(value,))
        if cfg:
            cur.insert("", "end", text="runnable", values=("yes" if state["ready"] else "no: Install finishes it",))
        server = state["server"]
        if server:
            cur.insert("", "end", text="server", values=(server.get("state", "?")
                       + (f"  (pid {server['pid']})" if server.get("pid") else ""),))
            cur.insert("", "end", text="chat", values=(server["urls"]["browser"],))
            cur.insert("", "end", text="OpenAI base URL", values=(server["urls"]["openai_base_url"],))
            cur.insert("", "end", text="Anthropic base URL", values=(server["urls"]["anthropic_base_url"],))

    def show_server():
        cfg = state["cfg"]
        colour, text = server_words(state["server"], cfg)
        dot.itemconfigure(dot_id, fill={"green": "#1e8e3e", "orange": "#f29900", "grey": "#9aa0a6"}[colour])
        v_server.set(text)
        running = (state["server"] or {}).get("state") in ("ready", "loading")
        busy = state["proc"] is not None
        b_start.configure(state="normal" if state["ready"] and not running and not busy else "disabled")
        b_stop.configure(state="normal" if running and not busy else "disabled")
        b_chat.configure(state="normal" if (state["server"] or {}).get("state") == "ready" else "disabled")

    def watch_server():
        """Every few seconds: the saved configuration, whether it is runnable, the server."""
        while not state["closing"]:
            try:
                cfg = setup_flow.load_config() or {}
                ready = setup_flow.config_ready(cfg) if cfg else False
                server = setup_flow.server_status(cfg) if cfg.get("port") else None
                events.put(("server", (cfg, ready, server)))
            except Exception:  # a status probe never stops the window
                traceback.print_exc(file=sys.stderr)
            for _ in range(8):
                if state["closing"]:
                    return
                time.sleep(0.5)

    def detect():
        target = v_root.get() or setup_flow.default_models_root()
        hw_text.configure(state="normal")
        hw_text.delete("1.0", "end")
        hw_text.insert("1.0", "Checking the hardware...")
        hw_text.configure(state="disabled")

        def work():
            try:
                done = subprocess.run(detect_command(target, py), capture_output=True, text=True,
                                      encoding="utf-8", errors="replace", timeout=180,
                                      stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
                hw = json.loads(done.stdout)
                rows = catalog_rows(hw["memory"].get("total"), setup_hw.disk_free(target),
                                    downloaded_bytes(target), machine=hw["os"].get("machine"))
                events.put(("hw", (hw, rows, drive_report())))
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                events.put(("error", f"The hardware check failed ({error}); the model list "
                                     "assumes nothing about this PC."))
                rows = catalog_rows(None, setup_hw.disk_free(target), {})
                events.put(("hw", (None, rows, drive_report())))

        threading.Thread(target=work, daemon=True).start()

    def busy(kind):
        state["proc_kind"] = kind
        b_cancel.configure(state="normal" if kind or state.get("app_wait") else "disabled")
        show_server()
        update_plan()

    def launch(cmd, extra, kind, then=None):
        if state["proc"] or state.get("moving"):
            messagebox.showinfo("colibri", "Something is already running: wait for it, or Cancel.")
            return
        log_write(f"\n> {display_command(cmd, extra)}\n")
        try:
            proc = subprocess.Popen(cmd, cwd=HERE, env=child_env(extra), stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
        except OSError as error:
            log_write(f"could not start it: {error}\n")
            show_output()
            return
        state["proc"] = proc
        state["then"] = then
        busy(kind)
        progress.configure(mode="indeterminate")
        progress.start(12)
        v_status.set({"setup": "Installing...", "start": "Starting the server...",
                      "stop": "Stopping the server...", "import": "Importing the GGUF...",
                      "use": "Switching to the model...",
                      "fetch": "Getting the model..."}.get(kind, "Working..."))
        if kind == "show":
            show_output()

        def pump():
            while True:
                chunk = proc.stdout.read1(4096)
                if not chunk:
                    break
                events.put(("out", chunk.decode("utf-8", "replace")))
            events.put(("done", (kind, proc.wait())))

        threading.Thread(target=pump, daemon=True).start()

    def run_coli(args, kind):
        launch([py, COLI] + args, {}, kind)

    def apply():
        if v_use_local.get():
            if not v_model_dir.get().strip():
                messagebox.showwarning("colibri", "Choose the model folder first (step 1).")
                return
            if not os.path.isfile(os.path.join(v_model_dir.get(), "config.json")) and not \
                    messagebox.askyesno("colibri", f"{v_model_dir.get()} has no config.json. Use it anyway?"):
                return
        try:
            cmd = setup_command(options(), python=py)
        except ValueError as error:
            messagebox.showerror("colibri", str(error))
            return
        _, warnings = plan_lines(options(), chosen_row(), setup_hw.disk_free(v_root.get()) if v_root.get() else None)
        if warnings and not messagebox.askyesno("colibri", "Before you install:\n\n" + "\n\n".join(warnings)
                                                + "\n\nInstall anyway?"):
            return
        launch(cmd, run_extra(), "setup")

    def cancel():
        if state.get("app_wait"):
            state["app_wait_cancel"] = True
            v_status.set("Stopped waiting (the server keeps starting in the background).")
            return
        if state.get("moving"):
            state["move_cancel"] = True
            v_status.set("Cancelling the move...")
            return
        proc = state["proc"]
        if proc and proc.poll() is None:
            if sys.platform == "win32":
                subprocess.run(kill_tree_command(proc.pid), capture_output=True, creationflags=NO_WINDOW)
            else:
                proc.terminate()
            state["cancelled"] = True
            log_write("\n(cancelled: Install again continues where it stopped)\n")

    # Connect apps
    def app_conn():
        found = dict(state["apps_models"])
        model = v_app_model.get().strip() or None
        return agent_connect.connection(state["cfg"], model=model, ctx=found.get(model),
                                        key=v_app_key.get().strip() or "local")

    def show_app(*_):
        sel = app_list.selection()
        tool_id = sel[0] if sel else "generic"
        entry = agent_connect.tool(tool_id)
        app_text.configure(state="normal")
        app_text.delete("1.0", "end")
        app_text.insert("1.0", agent_connect.instructions(tool_id, app_conn()))
        app_text.configure(state="disabled")
        reason = open_reason(entry, state["apps_found"].get(tool_id), sys.platform, bool(state["apps_models"]))
        b_app_open.configure(state="disabled" if reason else "normal")
        app_reason.configure(text=reason)
        b_bridge.configure(state="normal" if entry.get("bridge") else "disabled")
        b_app_write.configure(state="normal" if entry.get("write") else "disabled")

    app_list.bind("<<TreeviewSelect>>", show_app)
    v_app_model.trace_add("write", show_app)
    v_app_key.trace_add("write", show_app)

    def refresh_apps():
        conn = app_conn()
        app_server.configure(text=f"asking {conn['base']} ...")

        def work():
            served = agent_connect.served_models(conn["base"], v_app_key.get().strip())
            found = {e["id"]: agent_connect.find_exe(e) for e in agent_connect.TOOLS}
            bridge = agent_connect.answers(conn["bridge"] + "/")
            events.put(("apps", (conn["base"], served, found, bridge)))

        threading.Thread(target=work, daemon=True).start()

    def open_app():
        sel = app_list.selection()
        if not sel:
            return
        tool_id = sel[0]
        conn = app_conn()
        try:
            if agent_connect.tool(tool_id).get("bridge") and agent_connect.start_bridge(conn, python=py):
                log_write(f"\nstarted the Codex bridge (coli ui) on {conn['bridge']}\n")
            script = agent_connect.open_in_terminal(tool_id, conn, v_app_folder.get(),
                                                    state["apps_found"].get(tool_id))
        except (ValueError, OSError) as error:
            messagebox.showerror("colibri", str(error))
            return
        v_status.set(f"Opened {agent_connect.tool(tool_id)['name']} in a new window.")
        log_write(f"\nopened {agent_connect.tool(tool_id)['name']} ({script})\n")

    def write_app_settings():
        sel = app_list.selection()
        if not sel or not agent_connect.tool(sel[0]).get("write"):
            return
        entry = agent_connect.tool(sel[0])
        if not messagebox.askyesno("colibri", f"Write colibri into {entry['name']}'s settings?\n\n"
                                              "A copy of an existing file is kept beside it."):
            return
        try:
            path, backup, words = agent_connect.write_settings(sel[0], app_conn())
        except (ValueError, OSError) as error:
            messagebox.showerror("colibri", str(error))
            return
        root.clipboard_clear()
        root.clipboard_append(path)
        log_write(f"\nwrote {path}" + (f" (the old one: {backup})" if backup else "") + "\n")
        v_status.set(f"Wrote {entry['name']}'s settings.")
        messagebox.showinfo("colibri", words + "\n\n(The file's path is on the clipboard.)")

    def start_bridge():
        conn = app_conn()
        try:
            started = agent_connect.start_bridge(conn, python=py)
        except OSError as error:
            messagebox.showerror("colibri", str(error))
            return
        v_status.set(f"Codex bridge {'started' if started else 'already running'} on {conn['bridge']}.")
        root.after(3000, refresh_apps)

    # Set up for coding apps: the context Claude Code and Codex need
    def set_up_coding():
        cfg = state["cfg"] or {}
        if not cfg.get("model_dir") or not state["ready"]:
            messagebox.showinfo("colibri", "Set up a model first: steps 1 to 4, or My models.")
            nb.select(t_model)
            return
        coding_info.configure(text="Working out the context this PC can give the model...")

        def work():
            try:
                events.put(("coding", coding_context(cfg["model_dir"])))
            except Exception as error:  # an unreadable model: say so, keep the window
                traceback.print_exc(file=sys.stderr)
                events.put(("coding", (None, f"Could not size the context: {error}")))

        threading.Thread(target=work, daemon=True).start()

    def apply_coding(ctx, words):
        current = int((state["cfg"] or {}).get("ctx") or 0)
        if ctx is None:
            coding_info.configure(text=words)
            messagebox.showwarning("colibri", words)
            return
        if current == ctx:
            coding_info.configure(text=f"Already set: {words}.")
            return
        running = (state["server"] or {}).get("state") in ("ready", "loading")
        if not messagebox.askyesno(
                "colibri", f"Set the context to {words}?\n\nClaude Code and Codex then have room for "
                           "their instructions and tools, and the setup keeps it for every start."
                           + ("\n\nThe server restarts to use it." if running else "")):
            coding_info.configure(text=coding_advice(current or None))
            return
        v_ctx.set(str(ctx))
        restart = (lambda: launch([py, COLI, "stop"], {}, "stop",
                                  then=lambda: run_coli(["start", "--background", "--no-browser"], "start"))) \
            if running else None
        launch([py, COLI, "setup", "--yes", "--ctx", str(ctx), "--no-start"], {}, "setup", then=restart)

    # Shortcuts: colibri in the Start menu and on the desktop
    def shortcuts(action):
        if sys.platform != "win32":
            sc_info.configure(text="The shortcut is made on Windows.")
            return
        sc_info.configure(text="Working...")

        def work():
            try:
                if action == "create":
                    win_shortcuts.create()
                elif action == "remove":
                    win_shortcuts.remove()
                events.put(("shortcuts", (win_shortcuts.status(), None)))
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                events.put(("shortcuts", (None, str(error))))

        threading.Thread(target=work, daemon=True).start()

    def show_shortcuts(found, error):
        if error:
            sc_info.configure(text=f"Could not do it: {error}")
            return
        both = found and all(found.values())
        if both:
            sc_info.configure(text="colibri is in the Start menu and on the desktop: it opens this window "
                                   "(no command line needed). Moved the colibri folder? Add them again.")
        elif found and any(found.values()):
            sc_info.configure(text="One of the two shortcuts is missing: add them again.")
        else:
            sc_info.configure(text="Open this window from the Start menu or the desktop, with the "
                                   "colibri icon, instead of CONFIGURE.bat.")
        b_sc_remove.configure(state="normal" if found and any(found.values()) else "disabled")

    def offer_shortcuts():
        """Asked once, the first time the window opens on Windows."""
        if sys.platform != "win32" or lib_state["prefs"].get("shortcut_offered"):
            return
        lib_state["prefs"]["shortcut_offered"] = True
        save_prefs(lib_state["prefs"])
        if messagebox.askyesno("colibri", "Add colibri to the Start menu and the desktop?\n\n"
                                          "It opens this window, with the colibri icon. "
                                          "(This PC tab: add or remove them later.)"):
            shortcuts("create")

    # Updates: a newer colibri, and bringing this one up to date
    def check_updates(quiet=False):
        up_info.configure(text="Checking for updates...")
        b_update.configure(state="disabled")

        def work():
            try:
                events.put(("updates", (updater.check(), quiet)))
            except Exception as error:  # a check never stops the window
                events.put(("updates", ({"kind": "error", "words": f"Could not check: {error}",
                                         "can_update": False}, quiet)))

        threading.Thread(target=work, daemon=True).start()

    def show_updates(result, quiet):
        lib_state["updates"] = result
        lib_state["prefs"]["last_update_check"] = time.time()
        save_prefs(lib_state["prefs"])
        up_info.configure(text=f"colibri {result.get('version', '?')}. {result['words']}")
        b_update.configure(state="normal" if result.get("can_update") and state["proc"] is None else "disabled")
        b_update.pack_forget() if result.get("kind") == "release" else b_update.pack(side="left", padx=6)
        b_download.configure(state="normal" if result.get("kind") == "release" else "disabled")
        if result.get("can_update") or result.get("newer"):
            v_status.set("A colibri update is available: This PC tab, Updates.")

    def update_now():
        result = lib_state.get("updates") or {}
        if not result.get("can_update"):
            return
        running = (state["server"] or {}).get("state") in ("ready", "loading")
        if not messagebox.askyesno(
                "colibri", "Update colibri now?\n\n" + result["words"] + "\n\nThe engine is rebuilt from the "
                           "new code (only what changed)" + ("; the server stops and starts again." if running
                                                              else ".")):
            return

        def done():
            v_status.set("Updated. Close and open this window again to use its new version too.")
            check_updates(quiet=True)

        def rebuild():
            then = (lambda: run_coli(["start", "--background", "--no-browser"], "start")) if running else done
            if state["ready"]:
                launch([py, COLI, "setup", "--yes", "--no-start"], {}, "setup", then=then)
            else:
                then()

        pull = lambda: launch(updater.pull_command(result["root"]), {}, "update", then=rebuild)  # noqa: E731
        if running:
            launch([py, COLI, "stop"], {}, "stop", then=pull)
        else:
            pull()

    # Open app: one click from the top bar
    def fill_apps_menu():
        apps_menu.delete(0, "end")
        for tool_id, label, enabled in app_menu_items(state["apps_found"], sys.platform):
            apps_menu.add_command(label=label, state="normal" if enabled else "disabled",
                                  command=lambda t=tool_id: one_click(t))
        apps_menu.add_separator()
        folder = lib_state["prefs"].get("project_folder")
        apps_menu.add_command(label=f"Project folder: {folder or 'asked the first time'}...",
                              command=lambda: project_folder(ask=True))

    def project_folder(ask=False):
        folder = lib_state["prefs"].get("project_folder")
        if ask or not folder or not os.path.isdir(folder):
            chosen = filedialog.askdirectory(title="The project folder the app opens in",
                                             initialdir=folder if folder and os.path.isdir(folder)
                                             else os.path.expanduser("~"))
            if not chosen:
                return None
            folder = os.path.normpath(chosen)
            lib_state["prefs"]["project_folder"] = folder
            save_prefs(lib_state["prefs"])
            fill_apps_menu()
        v_app_folder.set(folder)
        return folder

    def one_click(tool_id):
        name = agent_connect.tool(tool_id)["name"]
        if state.get("app_wait"):
            messagebox.showinfo("colibri", "Already waiting for the model to load.")
            return
        if not state["cfg"] or not state["ready"]:
            messagebox.showinfo("colibri", f"Set up a model before opening {name}: steps 1 to 4, "
                                           "or My models.")
            nb.select(t_model)
            return
        if not project_folder():
            return
        if state["apps_models"] and (state["server"] or {}).get("state") == "ready":
            finish_open(tool_id)
            return
        if (state["server"] or {}).get("state") not in ("ready", "loading"):
            if state["proc"] is not None:
                messagebox.showinfo("colibri", "Something is already running: wait for it, or Cancel.")
                return
            run_coli(["start", "--background", "--no-browser"], "start")
        wait_for_model(tool_id)

    def wait_for_model(tool_id):
        state["app_wait"], state["app_wait_cancel"] = tool_id, False
        b_cancel.configure(state="normal")
        progress.configure(mode="indeterminate")
        progress.start(12)
        cfg, conn, key, started = dict(state["cfg"]), app_conn(), v_app_key.get().strip(), time.time()

        def work():
            while not state.get("app_wait_cancel") and not state["closing"]:
                ready = False
                try:
                    ready = setup_flow.server_status(cfg).get("state") == "ready"
                except Exception:  # still starting: no answer yet
                    pass
                served = agent_connect.served_models(conn["base"], key) if ready else []
                if served:
                    events.put(("app_ready", (tool_id, served)))
                    return
                if time.time() - started > 3600:
                    break
                events.put(("app_wait", (tool_id, time.time() - started)))
                time.sleep(2)
            events.put(("app_ready", (tool_id, None)))

        threading.Thread(target=work, daemon=True).start()

    def finish_open(tool_id):
        entry = agent_connect.tool(tool_id)
        exe = state["apps_found"].get(tool_id) or agent_connect.find_exe(entry)
        if not exe:
            messagebox.showinfo("colibri", open_reason(entry, None, sys.platform, True))
            return
        if not v_app_model.get() and state["apps_models"]:
            v_app_model.set(state["apps_models"][0][0])
        conn = app_conn()
        try:
            if entry.get("bridge") and agent_connect.start_bridge(conn, python=py):
                log_write(f"\nstarted the Codex bridge (coli ui) on {conn['bridge']}\n")
            script = agent_connect.open_in_terminal(tool_id, conn, v_app_folder.get(), exe)
        except (ValueError, OSError) as error:
            messagebox.showerror("colibri", str(error))
            return
        v_status.set(f"Opened {entry['name']} on {conn['model']} in {v_app_folder.get()}.")
        log_write(f"\nopened {entry['name']} ({script})\n")

    # My models
    def scan_models():
        lib_status.configure(text="Looking on every drive...")
        folders = list(lib_state["prefs"].get("scan_folders") or [])
        cfg = dict(state["cfg"] or {})

        def work():
            try:
                rows = model_library.scan(folders, cfg=cfg)
            except Exception as error:  # a scan never stops the window
                traceback.print_exc(file=sys.stderr)
                events.put(("error", f"The scan stopped: {error}"))
                rows = []
            events.put(("library", rows))

        threading.Thread(target=work, daemon=True).start()

    def add_scan_folder():
        chosen = filedialog.askdirectory(title="A folder with models (GGUF files or colibri model folders)")
        if not chosen:
            return
        folders = lib_state["prefs"].setdefault("scan_folders", [])
        if os.path.normpath(chosen) not in folders:
            folders.append(os.path.normpath(chosen))
            save_prefs(lib_state["prefs"])
        scan_models()

    def current_model_dir():
        return os.path.normcase(os.path.abspath((state["cfg"] or {}).get("model_dir") or "")) \
            if (state["cfg"] or {}).get("model_dir") else ""

    def fill_library():
        keep = lib.selection()
        lib.delete(*lib.get_children())
        current = current_model_dir()
        counts = {"yes": 0, "import": 0}
        for index, r in enumerate(lib_state["rows"]):
            is_current = current and os.path.normcase(os.path.abspath(r["path"])) == current
            tags = ("current",) if is_current else (("no",) if r["runs"] == "no" else ())
            counts[r["runs"]] = counts.get(r["runs"], 0) + 1
            words = library_words(r["runs"])
            if r["runs"] == "import" and os.path.isfile(os.path.join(model_library.import_target(
                    r, v_root.get() or setup_flow.default_models_root()), "config.json")):
                words = "Imported: Use it"
            lib.insert("", "end", iid=str(index), tags=tags,
                       text=r["name"] + ("   (in use)" if is_current else ""),
                       values=(r["source"], setup_hw.fmt_gb(r["size"]) if r["size"] else "",
                               words,
                               os.path.dirname(r["path"]) if r["kind"] == "gguf" else r["path"]))
        if not lib_state["rows"]:
            lib.insert("", "end", iid="__none", tags=("no",), values=("", "", "", ""),
                       text="No models found yet: install one (steps 1 to 4), or Add a folder to scan")
        lib_status.configure(text=f"{len(lib_state['rows'])} found: {counts['yes']} ready, "
                                  f"{counts['import']} to import")
        if keep and lib.exists(keep[0]):
            lib.selection_set(keep[0])
        on_lib_select()

    def lib_row():
        sel = [s for s in lib.selection() if not s.startswith("__")]
        return lib_state["rows"][int(sel[0])] if sel else None

    def on_lib_select(_event=None):
        r = lib_row()
        busy_now = state["proc"] is not None or state.get("moving")
        if not r:
            lib_info.configure(text="Pick a model to use, import or move it.")
            for button in (b_use, b_import, b_move, b_open):
                button.configure(state="disabled")
            return
        text = f"{r['path']}\n{r['why']}"
        if r["source"] == "Ollama" and r["runs"] == "import":
            text += ("\nImport reads Ollama's own file, without copying it: keep the model in Ollama "
                     "(after `ollama rm` it has to be imported again from another copy).")
        lib_info.configure(text=text)
        b_use.configure(state="normal" if r["runs"] in ("yes", "incomplete") and not busy_now else "disabled")
        b_import.configure(state="normal" if r["runs"] == "import" and not busy_now else "disabled")
        b_move.configure(state="normal" if r["kind"] == "folder" and not busy_now else "disabled")
        b_open.configure(state="normal")

    lib.bind("<<TreeviewSelect>>", on_lib_select)
    lib.bind("<Double-1>", lambda _e: (use_model() if (lib_row() or {}).get("runs") in ("yes", "incomplete")
                                       else import_model() if (lib_row() or {}).get("runs") == "import" else None))

    def server_running():
        return (state["server"] or {}).get("state") in ("ready", "loading")

    def switch_to(row):
        """`coli setup` for a folder already on disk; the running server is stopped first."""
        cmd = model_library.use_command(row, python=py, coli=COLI, start=v_start.get() != "no")
        if v_no_browser.get():
            cmd.append("--no-browser")
        if server_running():
            launch([py, COLI, "stop"], {}, "stop", then=lambda: launch(cmd, {}, "use"))
        else:
            launch(cmd, {}, "use")

    def use_model():
        r = lib_row()
        if not r:
            return
        is_current = current_model_dir() == os.path.normcase(os.path.abspath(r["path"]))
        if is_current and state["ready"]:
            if not server_running():
                run_coli(["start", "--background"] + (["--no-browser"] if v_no_browser.get() else []), "start")
            else:
                v_status.set(f"{r['name']} is already in use.")
            return
        question = f"Use {r['name']} from now on?"
        if server_running():
            question += "\n\nThe running server is stopped first."
        if r["runs"] == "incomplete":
            question += "\n\nIts download is not finished: it continues first."
        if messagebox.askyesno("colibri", question):
            switch_to(r)

    def import_model():
        r = lib_row()
        if not r or r["runs"] != "import":
            return
        root_dir = v_root.get() or setup_flow.default_models_root()
        out = model_library.import_target(r, root_dir)
        target = {"source": "colibri", "id": os.path.basename(out), "path": out, "repo": "", "name": r["name"]}
        if os.path.isfile(os.path.join(out, "config.json")):
            if messagebox.askyesno("colibri", f"{r['name']} was imported before ({out}). Use it now?"):
                switch_to(target)
            return
        if not messagebox.askyesno(
                "colibri", f"Import {r['name']} into colibri and use it?\n\n"
                           f"A small folder is made in {out}; the weights are read from where they are now "
                           f"({setup_hw.fmt_gb(r['size'])}, nothing is copied)."):
            return
        launch(model_library.import_command(r, out, python=py, coli=COLI), {}, "import",
               then=lambda: switch_to(target))

    def move_selected():
        r = lib_row()
        if not r or r["kind"] != "folder":
            return
        dest = filedialog.askdirectory(title=f"Move {r['name']} into which folder? (for example E:\\colibri-models)")
        if not dest:
            return
        dest = os.path.normpath(dest)
        in_use = server_running() and current_model_dir() == os.path.normcase(os.path.abspath(r["path"]))
        problem = model_library.move_problem(r["path"], dest, size=r["size"], in_use=in_use)
        if problem:
            messagebox.showwarning("colibri", f"{r['name']} cannot move there: {problem}.")
            return
        quick = model_library.same_drive(r["path"], dest)
        if not messagebox.askyesno(
                "colibri", f"Move {r['name']} ({setup_hw.fmt_gb(r['size'])}) to {dest}?\n\n"
                           + ("Same drive: this is instant." if quick else
                              "Another drive: it is copied and checked, and only then removed from "
                              f"{os.path.dirname(r['path'])}. Cancel at any time leaves it where it is.")):
            return
        state["moving"], state["move_cancel"] = True, False
        busy("move")
        progress.configure(mode="indeterminate")
        progress.start(12)
        v_status.set(f"Moving {r['name']}...")
        on_lib_select()

        def work():
            try:
                new = model_library.move_model(
                    r["path"], dest, progress=lambda d, t: events.put(("move", (d, t))),
                    cancelled=lambda: state.get("move_cancel", False))
                events.put(("moved", (r["path"], new, None)))
            except (model_library.MoveError, OSError) as error:
                events.put(("moved", (r["path"], None, str(error))))

        threading.Thread(target=work, daemon=True).start()

    def open_selected():
        r = lib_row()
        if r:
            open_path(r["path"] if r["kind"] == "folder" else os.path.dirname(r["path"]))

    # Get a model
    def models_root():
        return v_root.get() or (state["cfg"] or {}).get("models_root") or setup_flow.default_models_root()

    def look_up(variant=None):
        text = v_get.get().strip()
        if not text:
            return
        b_lookup.configure(state="disabled")
        b_fetch.configure(state="disabled")
        get_info.configure(text=f"Looking up {text}... (reads only the first megabytes)")
        token = v_token.get().strip()

        def work():
            try:
                events.put(("resolved", (model_fetch.resolve(text, token, variant=variant), None)))
            except model_fetch.FetchError as error:
                events.put(("resolved", (None, str(error))))
            except Exception as error:  # a lookup never stops the window
                traceback.print_exc(file=sys.stderr)
                events.put(("resolved", (None, f"the lookup failed: {error}")))

        threading.Thread(target=work, daemon=True).start()

    def variant_label(v):
        return f"{v['label']}  ({setup_hw.fmt_gb(v['size'])})"

    def on_variant(_event=None):
        plan = lib_state["plan"]
        if not plan:
            return
        picked = next((v for v in plan["variants"] if variant_label(v) == v_variant.get()), None)
        if picked and picked is not plan["chosen"]:
            look_up(picked["label"])

    get_variant.bind("<<ComboboxSelected>>", on_variant)

    def show_plan(plan, error):
        lib_state["plan"] = plan
        b_lookup.configure(state="normal")
        if error:
            get_info.configure(text=f"Not found: {error}")
            get_variant.configure(state="disabled", values=[])
            b_fetch.configure(state="disabled")
            return
        get_variant.configure(values=[variant_label(v) for v in plan["variants"]],
                              state="readonly" if len(plan["variants"]) > 1 else "disabled")
        v_variant.set(variant_label(plan["chosen"]))
        text, ok = fetch_summary(plan, models_root())
        get_info.configure(text=text)
        b_fetch.configure(state="normal" if ok and state["proc"] is None and not state.get("moving") else "disabled")

    def fetch_model():
        plan = lib_state["plan"]
        if not plan or not plan["supported"]:
            return
        root_dir = models_root()
        target = model_fetch.import_dir(plan["name"], plan["chosen"], root_dir)
        row = {"source": "colibri", "id": os.path.basename(target), "path": target, "repo": "", "name": plan["name"]}
        if os.path.isfile(os.path.join(target, "config.json")):
            if messagebox.askyesno("colibri", f"{plan['name']} is already here ({target}). Use it now?"):
                switch_to(row)
            return
        if not messagebox.askyesno(
                "colibri", f"Download {plan['name']} ({plan['chosen']['label']}, "
                           f"{setup_hw.fmt_gb(plan['chosen']['size'])}) and use it?\n\n"
                           "It can be cancelled and continued later; every byte is checked."):
            return
        cmd = model_fetch.fetch_command(v_get.get().strip(), plan["chosen"]["label"], root_dir, python=py)
        launch(cmd, extra_env(v_token.get()), "fetch", then=lambda: switch_to(row))

    # ---- the event loop's side
    def poll():
        try:
            while True:
                kind, payload = events.get_nowait()
                if kind == "out":
                    log_write(payload)
                elif kind == "done":
                    what, code = payload
                    state["proc"] = None
                    progress.stop()
                    progress.configure(mode="determinate", value=1000 if code == 0 else 0)
                    was_cancelled = state.pop("cancelled", False)
                    then = state.pop("then", None)
                    if was_cancelled:
                        v_status.set("Cancelled. Install again continues where it stopped.")
                    elif code == 0:
                        v_status.set({"setup": "Installed.", "start": "Started: the model is loading.",
                                      "stop": "Stopped.", "import": "Imported.",
                                      "use": "Switched: the model is loading.",
                                      "fetch": "Downloaded and imported."}.get(what, "Done."))
                    else:
                        v_status.set("It stopped with an error: the output below says why.")
                        show_output()
                    busy("")
                    load_current()
                    refresh_apps()
                    if what in ("import", "use", "setup", "stop", "fetch"):
                        scan_models()
                    if then and code == 0 and not was_cancelled:
                        then()
                elif kind == "app_wait":
                    tool_id, seconds = payload
                    v_status.set(waiting_words(agent_connect.tool(tool_id)["name"], seconds))
                elif kind == "app_ready":
                    tool_id, served = payload
                    state["app_wait"] = None
                    progress.stop()
                    progress.configure(mode="determinate", value=1000 if served else 0)
                    busy(state["proc_kind"] if state["proc"] is not None else "")
                    if served:
                        state["apps_models"] = served
                        finish_open(tool_id)
                        refresh_apps()
                    elif not state.get("app_wait_cancel"):
                        v_status.set("The model did not load within an hour: see Details, Server log.")
                elif kind == "updates":
                    show_updates(*payload)
                elif kind == "coding":
                    apply_coding(*payload)
                elif kind == "shortcuts":
                    show_shortcuts(*payload)
                elif kind == "resolved":
                    show_plan(*payload)
                elif kind == "library":
                    lib_state["rows"] = payload
                    fill_library()
                elif kind == "move":
                    done_bytes, total = payload
                    progress.configure(mode="determinate", value=int(1000 * done_bytes / max(1, total)))
                    v_status.set(f"Moving the model: {setup_hw.fmt_gb(done_bytes)} of {setup_hw.fmt_gb(total)}")
                elif kind == "moved":
                    old, new, error = payload
                    state["moving"] = False
                    progress.stop()
                    progress.configure(mode="determinate", value=1000 if new else 0)
                    busy("")
                    if new:
                        cfg = setup_flow.load_config() or {}
                        moved = model_library.follow_move(cfg, old, new)
                        if moved is not cfg:
                            setup_flow.save_config(moved)
                        v_status.set(f"Moved to {new}" + ("; the setup now uses it there." if moved is not cfg else "."))
                    else:
                        v_status.set(f"Not moved: {error}")
                        messagebox.showwarning("colibri", f"The model was not moved:\n\n{error}")
                    load_current()
                    scan_models()
                elif kind == "server":
                    cfg, ready, server = payload
                    changed = (cfg != state["cfg"] or ready != state["ready"]
                               or (server or {}).get("state") != (state["server"] or {}).get("state"))
                    state["cfg"], state["ready"], state["server"] = cfg, ready, server
                    show_server()
                    update_plan()
                    if changed:
                        load_current()
                        refresh_apps()
                elif kind == "hw":
                    hw, rows, drive_rows = payload
                    state["hw"], state["rows"] = hw, rows
                    if hw:
                        hw_text.configure(state="normal")
                        hw_text.delete("1.0", "end")
                        hw_text.insert("1.0", setup_hw.format_report(hw))
                        hw_text.configure(state="disabled")
                    drives.delete(*drives.get_children())
                    for drive, free, total in drive_rows:
                        drives.insert("", "end", text=drive,
                                      values=(setup_hw.fmt_gb(free), setup_hw.fmt_gb(total)))
                    fill_models()
                    on_select()
                elif kind == "apps":
                    base, served, found, bridge = payload
                    state["apps_models"], state["apps_found"] = served, found
                    app_model.configure(values=[m for m, _ in served])
                    if served and v_app_model.get() not in [m for m, _ in served]:
                        v_app_model.set(served[0][0])
                    app_server.configure(text=(f"server answering at {base}" if served else
                                               f"no server at {base}")
                                         + ("; Codex bridge running" if bridge else ""))
                    for entry in agent_connect.TOOLS:
                        text = (("installed" if found.get(entry["id"]) else "not found")
                                if entry["exe"] or entry.get("ext") else "")
                        app_list.set(entry["id"], "state", text)
                    show_app()
                    fill_apps_menu()
                    served_ctx = dict(served).get(v_app_model.get()) if served else None
                    coding_info.configure(text=coding_advice(served_ctx))
                elif kind == "error":
                    v_status.set(payload)
                    log_write(payload + "\n")
        except queue.Empty:
            pass
        if state["proc"] is not None and state["proc_kind"] in ("setup", "fetch"):
            fraction, text = progress_from_state(setup_flow.read_state())
            if fraction is None:
                if str(progress.cget("mode")) != "indeterminate":
                    progress.configure(mode="indeterminate")
                    progress.start(12)
            else:
                if str(progress.cget("mode")) != "determinate":
                    progress.stop()
                    progress.configure(mode="determinate")
                progress.configure(value=int(fraction * 1000))
            v_status.set(text)
        root.after(250, poll)

    def on_close():
        if state.get("moving"):
            if not messagebox.askyesno("colibri", "A model is being moved. Stop the move and close?\n"
                                                  "(The model stays in its old place.)"):
                return
            state["move_cancel"] = True
        proc = state["proc"]
        if proc and proc.poll() is None:
            if not messagebox.askyesno("colibri", "Something is still running. Stop it and close?\n"
                                                  "(Install continues where it stopped next time.)"):
                return
            cancel()
        state["closing"] = True
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    refresh_mode()
    detect()
    app_list.selection_set("claude")
    scan_models()
    if time.time() - float(lib_state["prefs"].get("last_update_check") or 0) > 86400:
        root.after(3000, lambda: check_updates(quiet=True))      # at most once a day, quietly
    else:
        up_info.configure(text=f"colibri {setup_flow.current_version()}. Check for updates to see what is new.")
    if sys.platform == "win32":
        shortcuts("status")
        root.after(1500, offer_shortcuts)
    else:
        show_shortcuts(None, None)
    threading.Thread(target=watch_server, daemon=True).start()
    show_server()
    load_current()
    poll()
    nb.select(t_apps if saved.get("status") == "ready" else t_model)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
