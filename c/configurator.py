#!/usr/bin/env python3
"""colibri configurator: a window for the one-step setup.

`coli setup` already does the work (detect the hardware, pick a model, build or
fetch the engine, download with resume, write setup.json). This is a graphical
front end to it, made for Windows: double-click CONFIGURE.bat at the top of the
folder, or run `py -3 c\\configurator.py`.

    Machine        what the setup sees: CPU, RAM, GPU, the free space on each drive
    Model          the catalog against this machine (what fits and why), or a
                   model folder that is already on disk
    Locations      where models are downloaded, extra shard folders for a model
                   split across drives, the Hugging Face token for gated models
    Run            backend (auto / CPU / Vulkan / CUDA), address, port, context
    Current setup  the saved run configuration (setup.json) and the server's state

Every action runs the real launcher (`coli setup ...`, `coli start`, `coli stop`,
`coli status`, `coli logs`) and shows its output, and the command is shown
before it runs, so whatever the window does can also be typed in a terminal.
The window keeps no settings of its own: setup.json, written by `coli setup`,
stays the only configuration. The token is passed to the child process only,
never saved.
"""
import os
import queue
import shutil
import subprocess
import sys
import threading
import webbrowser

HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import setup_catalog  # noqa: E402
import setup_download  # noqa: E402
import setup_flow  # noqa: E402
import setup_hw  # noqa: E402

GB = setup_catalog.GB
BACKENDS = ("auto", "cpu", "vulkan", "cuda")
COLI = os.path.join(HERE, "coli")


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
    """The environment for a launcher run: unbuffered UTF-8 output (the log pane
    shows it live) without colour codes, plus the window's extra variables (the
    token, the extra shard folders)."""
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
    if len(shown) >= 2 and os.path.basename(shown[1]) == "coli":
        shown = ["coli"] + shown[2:]
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


def config_summary(cfg):
    """The saved run configuration as (label, value) lines."""
    if not cfg:
        return [("status", "not set up yet: choose a model, then Install / Apply")]
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
                      + "  (install the GPU packages, then Install / Apply again)"))
    if cfg.get("updated"):
        lines.append(("saved", cfg["updated"]))
    return lines


def split_name_for(family_id):
    return setup_flow.split_dirs_name(family_id) if family_id else ""


# ---------------------------------------------------------------- the window


def main():  # pragma: no cover - the window itself is exercised by hand
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("colibri configurator")
    root.geometry("1060x740")
    root.minsize(880, 620)
    icon = os.path.join(HERE, "..", "desktop", "src-tauri", "icons", "icon.ico")
    if sys.platform == "win32" and os.path.isfile(icon):
        try:
            root.iconbitmap(icon)
        except tk.TclError:
            pass
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    style.configure("Head.TLabel", font=("Segoe UI", 11, "bold"))
    style.configure("Hint.TLabel", foreground="#666666")

    saved = setup_flow.load_config() or {}
    saved_model = (saved.get("model") or {}).get("id") or ""
    saved_local = bool(saved.get("model_dir")) and not saved_model
    state = {"hw": None, "rows": [], "proc": None, "cfg": saved}
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
    split_saved = ""
    family = saved.get("family")
    if family and split_name_for(family):
        split_saved = (saved.get("env") or {}).get(split_name_for(family), "")
    v_split = tk.StringVar(value=split_saved)
    v_status = tk.StringVar(value="Ready.")
    v_cmd = tk.StringVar(value="")

    # ---- layout: tabs on top; command line, buttons and the output below
    paned = ttk.PanedWindow(root, orient="vertical")
    paned.pack(fill="both", expand=True, padx=8, pady=(8, 0))
    nb = ttk.Notebook(paned)
    paned.add(nb, weight=3)
    bottom = ttk.Frame(paned)
    paned.add(bottom, weight=2)

    def tab(title):
        frame = ttk.Frame(nb, padding=10)
        nb.add(frame, text=title)
        return frame

    def form_row(parent, r, label, var, browse=None, show="", hint=""):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", pady=4)
        entry = ttk.Entry(parent, textvariable=var, show=show)
        entry.grid(row=r, column=1, sticky="ew", padx=6)
        if browse:
            ttk.Button(parent, text="Browse...", command=browse).grid(row=r, column=2)
        if hint:
            ttk.Label(parent, text=hint, style="Hint.TLabel", wraplength=780,
                      justify="left").grid(row=r + 1, column=1, sticky="w", padx=6)
        return entry

    # Machine
    t_machine = tab("Machine")
    ttk.Label(t_machine, text="What the setup sees on this computer", style="Head.TLabel").pack(anchor="w")
    hw_text = tk.Text(t_machine, height=9, wrap="word", font=("Consolas", 10), relief="flat")
    hw_text.pack(fill="x", pady=(6, 10))
    hw_text.insert("1.0", "Detecting the hardware (the GPU probe takes a few seconds)...")
    hw_text.configure(state="disabled")
    ttk.Label(t_machine, text="Drives", style="Head.TLabel").pack(anchor="w")
    drives = ttk.Treeview(t_machine, columns=("free", "total"), height=5)
    drives.heading("#0", text="Drive")
    drives.heading("free", text="Free")
    drives.heading("total", text="Size")
    drives.pack(fill="x", pady=4)
    ttk.Label(t_machine, style="Hint.TLabel", wraplength=900, justify="left",
              text="Large models stream their experts from the disk while answering: put them on "
                   "the fastest local NVMe drive, never a network or USB drive. Double-click a "
                   "drive to download models there.").pack(anchor="w")
    ttk.Button(t_machine, text="Detect again", command=lambda: detect()).pack(anchor="w", pady=6)

    # Model
    t_model = tab("Model")
    top = ttk.Frame(t_model)
    top.pack(fill="x")
    ttk.Radiobutton(top, text="Download a model from the catalog", variable=v_use_local,
                    value=False, command=lambda: refresh_mode()).pack(side="left")
    ttk.Checkbutton(top, text="also show models that do not fit this machine", variable=v_show_all,
                    command=lambda: fill_models()).pack(side="right")
    models = ttk.Treeview(t_model, columns=("size", "ram", "status", "runs", "id"),
                          show="tree headings", height=10, selectmode="browse")
    models.heading("#0", text="Model")
    models.column("#0", width=250)
    for col, text, width in (("size", "Download", 85), ("ram", "RAM min / best", 110),
                             ("status", "Fits", 85), ("runs", "Runs", 200), ("id", "id", 170)):
        models.heading(col, text=text)
        models.column(col, width=width, anchor="w")
    models.tag_configure("rec", font=("Segoe UI", 9, "bold"))
    models.tag_configure("nofit", foreground="#999999")
    models.pack(fill="both", expand=True, pady=6)
    model_info = ttk.Label(t_model, text="", wraplength=980, justify="left")
    model_info.pack(anchor="w", fill="x")
    local = ttk.Frame(t_model)
    local.pack(fill="x", pady=(8, 0))
    ttk.Radiobutton(local, text="Use a model folder already on disk:", variable=v_use_local,
                    value=True, command=lambda: refresh_mode()).pack(side="left")
    local_entry = ttk.Entry(local, textvariable=v_model_dir)
    local_entry.pack(side="left", fill="x", expand=True, padx=6)
    local_btn = ttk.Button(local, text="Browse...",
                           command=lambda: pick_dir(v_model_dir, "The model folder (with config.json)"))
    local_btn.pack(side="left")

    # Locations
    t_loc = tab("Locations")
    t_loc.columnconfigure(1, weight=1)
    form_row(t_loc, 0, "Models folder", v_root, browse=lambda: pick_dir(v_root, "Where to download models"),
             hint="Each catalog model goes in its own subfolder here (for example "
                  "D:\\colibri-models\\qwen36-35b). A stopped download resumes in the same folder.")
    free_label = ttk.Label(t_loc, text="")
    free_label.grid(row=2, column=1, sticky="w", padx=6)
    split_entry = form_row(t_loc, 3, "Extra shard folders", v_split,
                           hint="Only for a model split across drives, when its engine reads extra "
                                "folders (the variable is shown in the command). Separate folders "
                                "with ';'. The setup keeps them in setup.json.")
    form_row(t_loc, 5, "Hugging Face token", v_token, show="*",
             hint="Only for gated repositories. Passed to this run as HF_TOKEN and never written "
                  "to disk; `hf auth login` stores one permanently instead.")
    ttk.Separator(t_loc).grid(row=7, column=0, columnspan=3, sticky="ew", pady=10)
    ttk.Label(t_loc, text="Setup folder").grid(row=8, column=0, sticky="w")
    ttk.Label(t_loc, text=setup_flow.setup_home()).grid(row=8, column=1, sticky="w", padx=6)
    ttk.Button(t_loc, text="Open", command=lambda: open_path(setup_flow.setup_home())).grid(row=8, column=2)
    ttk.Label(t_loc, style="Hint.TLabel", wraplength=780, justify="left",
              text="setup.json, install-state.json, logs and the prebuilt runtime live here "
                   "(set COLI_SETUP_HOME before starting this window to move it)."
              ).grid(row=9, column=1, sticky="w", padx=6)
    ttk.Label(t_loc, text="colibri folder").grid(row=10, column=0, sticky="w", pady=(8, 0))
    ttk.Label(t_loc, text=os.path.dirname(HERE)).grid(row=10, column=1, sticky="w", padx=6, pady=(8, 0))
    ttk.Button(t_loc, text="Open", command=lambda: open_path(os.path.dirname(HERE))
               ).grid(row=10, column=2, pady=(8, 0))

    # Run
    t_run = tab("Run")
    t_run.columnconfigure(1, weight=1)
    ttk.Label(t_run, text="Backend").grid(row=0, column=0, sticky="w", pady=4)
    bk = ttk.Frame(t_run)
    bk.grid(row=0, column=1, sticky="w", padx=6)
    for value, text in (("auto", "Automatic"), ("cuda", "NVIDIA CUDA"),
                        ("vulkan", "Vulkan (any GPU)"), ("cpu", "CPU only")):
        ttk.Radiobutton(bk, text=text, value=value, variable=v_backend,
                        command=lambda: update_cmd()).pack(side="left", padx=(0, 12))
    ttk.Label(t_run, style="Hint.TLabel", wraplength=780, justify="left",
              text="Automatic picks CUDA when the toolkit builds for the card, then Vulkan, then "
                   "the CPU; a build that fails falls back to the next one.").grid(
        row=1, column=1, sticky="w", padx=6)
    form_row(t_run, 2, "Address", v_host,
             hint="127.0.0.1 keeps the server on this computer; 0.0.0.0 opens it to the network.")
    form_row(t_run, 4, "Port", v_port)
    form_row(t_run, 5, "Context (tokens)", v_ctx, hint="0 = automatic, sized from the memory there is.")
    ttk.Label(t_run, text="After the setup").grid(row=7, column=0, sticky="w", pady=(10, 4))
    st = ttk.Frame(t_run)
    st.grid(row=7, column=1, sticky="w", padx=6, pady=(10, 4))
    ttk.Radiobutton(st, text="Start the server in the background", value="background", variable=v_start,
                    command=lambda: update_cmd()).pack(side="left", padx=(0, 12))
    ttk.Radiobutton(st, text="Do not start", value="no", variable=v_start,
                    command=lambda: update_cmd()).pack(side="left")
    ttk.Checkbutton(t_run, text="Do not open the browser", variable=v_no_browser,
                    command=lambda: update_cmd()).grid(row=8, column=1, sticky="w", padx=6)
    ttk.Checkbutton(t_run, text="Skip the checksum of downloaded files (faster, less safe)",
                    variable=v_no_verify, command=lambda: update_cmd()).grid(row=9, column=1, sticky="w", padx=6)

    # Current setup
    t_cur = tab("Current setup")
    cur = ttk.Treeview(t_cur, columns=("value",), show="tree", height=14)
    cur.column("#0", width=170)
    cur.column("value", width=780)
    cur.pack(fill="both", expand=True)
    cb = ttk.Frame(t_cur)
    cb.pack(fill="x", pady=6)
    ttk.Button(cb, text="Reload", command=lambda: load_current()).pack(side="left")
    ttk.Button(cb, text="Open setup.json", command=lambda: open_path(setup_flow.config_path())
               ).pack(side="left", padx=6)
    ttk.Button(cb, text="Server log", command=lambda: run_coli(["logs", "-n", "200"])).pack(side="left")
    ttk.Button(cb, text="Install log", command=lambda: run_coli(["logs", "--install", "-n", "200"])
               ).pack(side="left", padx=6)
    ttk.Button(cb, text="Choose again (reconfigure)", command=lambda: (nb.select(t_model))).pack(side="right")

    # Bottom: command, actions, output
    cmdbar = ttk.Frame(bottom)
    cmdbar.pack(fill="x", pady=(6, 2))
    ttk.Label(cmdbar, text="Command:").pack(side="left")
    ttk.Entry(cmdbar, textvariable=v_cmd, state="readonly", font=("Consolas", 9)
              ).pack(side="left", fill="x", expand=True, padx=6)
    ttk.Button(cmdbar, text="Copy", command=lambda: (root.clipboard_clear(), root.clipboard_append(v_cmd.get()))
               ).pack(side="left")
    actions = ttk.Frame(bottom)
    actions.pack(fill="x", pady=4)
    b_apply = ttk.Button(actions, text="Install / Apply", command=lambda: apply())
    b_apply.pack(side="left")
    b_start = ttk.Button(actions, text="Start", command=lambda: run_coli(
        ["start", "--background"] + (["--no-browser"] if v_no_browser.get() else [])))
    b_start.pack(side="left", padx=4)
    b_stop = ttk.Button(actions, text="Stop", command=lambda: run_coli(["stop"]))
    b_stop.pack(side="left")
    b_status = ttk.Button(actions, text="Status", command=lambda: run_coli(["status"]))
    b_status.pack(side="left", padx=4)
    ttk.Button(actions, text="Open dashboard", command=lambda: open_dashboard()).pack(side="left")
    ttk.Button(actions, text="Clear output", command=lambda: log.delete("1.0", "end")).pack(side="right", padx=4)
    b_cancel = ttk.Button(actions, text="Cancel", command=lambda: cancel(), state="disabled")
    b_cancel.pack(side="right")
    out = ttk.Frame(bottom)
    out.pack(fill="both", expand=True)
    log = tk.Text(out, height=10, wrap="none", font=("Consolas", 9), background="#111111",
                  foreground="#dddddd", insertbackground="#dddddd")
    ysb = ttk.Scrollbar(out, orient="vertical", command=log.yview)
    log.configure(yscrollcommand=ysb.set)
    ysb.pack(side="right", fill="y")
    log.pack(side="left", fill="both", expand=True)
    ttk.Label(root, textvariable=v_status, anchor="w", relief="sunken").pack(fill="x", side="bottom")

    # ---- behaviour
    def pick_dir(var, title):
        start = var.get() if os.path.isdir(var.get() or "") else os.path.expanduser("~")
        chosen = filedialog.askdirectory(title=title, initialdir=start, mustexist=False)
        if chosen:
            var.set(os.path.normpath(chosen))

    def open_path(path):
        target = path if os.path.exists(path) else os.path.dirname(path)
        if not os.path.exists(target):
            messagebox.showinfo("colibri", f"{path} does not exist yet: run the setup first.")
            return
        if sys.platform == "win32":
            os.startfile(target)  # a folder or file the user asked to open
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", target])

    def open_dashboard():
        cfg = setup_flow.load_config() or {}
        host = cfg.get("host") or v_host.get() or setup_flow.DEFAULT_HOST
        port = cfg.get("port") or v_port.get() or setup_flow.DEFAULT_PORT
        webbrowser.open(setup_flow.urls(host, port)["browser"])

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

    def update_cmd(*_):
        try:
            v_cmd.set(display_command(setup_command(options(), python=py), run_extra()))
        except ValueError as error:
            v_cmd.set(f"(not valid yet: {error})")
        split_entry.configure(state="normal" if split_name_for(current_family()) else "disabled")
        path = v_root.get()
        free = setup_hw.disk_free(path) if path else None
        try:
            entry = None if v_use_local.get() or not v_model.get() else setup_catalog.by_id(v_model.get())
        except KeyError:
            entry = None
        text = f"{setup_hw.fmt_gb(free)} free there" if free is not None else ""
        note = space_note(entry, free)
        free_label.configure(text=f"{text}; {note}" if note else text,
                             foreground="#b00020" if "NOT ENOUGH" in note else "")

    for var in (v_root, v_host, v_port, v_ctx, v_model_dir, v_token, v_split):
        var.trace_add("write", update_cmd)

    def refresh_mode():
        mode = "normal" if v_use_local.get() else "disabled"
        local_entry.configure(state=mode)
        local_btn.configure(state=mode)
        update_cmd()

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
                                  r["status"], "makes images" if r["kind"] == "image" else r["runs"], r["id"]))
        if not v_model.get() and not v_use_local.get():
            rec = next((r["id"] for r in state["rows"] if r["recommended"]), "")
            v_model.set(rec)
        if v_model.get() and models.exists(v_model.get()):
            models.selection_set(v_model.get())
            models.see(v_model.get())
        if hidden:
            v_status.set(f"{hidden} more models need more RAM or disk than this machine has "
                         "(tick 'also show' to see why).")

    def on_select(_event=None):
        sel = models.selection()
        if not sel:
            return
        iid = sel[0]
        r = next((x for x in state["rows"] if x["id"] == iid), None)
        if r:
            text = f"{r['name']}  ({r['repo']})\n{r['summary']}.  {r['reason']}."
            if r["license_note"]:
                text += f"\nLicense: {r['license_note']}"
            if r["doc"]:
                text += f"\nMore: {r['doc']} in the colibri folder"
            model_info.configure(text=text)
        if _event is not None:
            v_model.set(iid)
            if v_use_local.get():
                v_use_local.set(False)
                refresh_mode()
        update_cmd()

    models.bind("<<TreeviewSelect>>", on_select)

    def on_drive(_event):
        sel = drives.selection()
        if sel:
            v_root.set(os.path.join(drives.item(sel[0], "text"), "colibri-models"))
            nb.select(t_loc)

    drives.bind("<Double-1>", on_drive)

    def load_current():
        state["cfg"] = cfg = setup_flow.load_config() or {}
        cur.delete(*cur.get_children())
        for label, value in config_summary(cfg):
            cur.insert("", "end", text=label, values=(value,))
        if not cfg:
            return
        cur.insert("", "end", text="runnable",
                   values=("yes" if setup_flow.config_ready(cfg) else "no: Install / Apply finishes it",))
        try:
            server = setup_flow.server_status(cfg) if cfg.get("port") else None
        except Exception:  # a status probe never breaks the window
            server = None
        if server:
            cur.insert("", "end", text="server", values=(server.get("state", "?")
                       + (f"  (pid {server['pid']})" if server.get("pid") else ""),))
            cur.insert("", "end", text="dashboard", values=(server["urls"]["browser"],))
            cur.insert("", "end", text="OpenAI base URL", values=(server["urls"]["openai_base_url"],))

    def detect():
        v_status.set("Detecting the hardware...")
        target = v_root.get() or setup_flow.default_models_root()

        def work():
            try:
                hw = setup_hw.detect(target)
                rows = catalog_rows(hw["memory"].get("total"), setup_hw.disk_free(target),
                                    downloaded_bytes(target), machine=hw["os"].get("machine"))
                events.put(("hw", (hw, rows, drive_report())))
            except Exception as error:  # shown, not raised: the window stays usable
                events.put(("error", f"hardware detection failed: {error}"))

        threading.Thread(target=work, daemon=True).start()

    def busy(on):
        for button in (b_apply, b_start, b_stop, b_status):
            button.configure(state="disabled" if on else "normal")
        b_cancel.configure(state="normal" if on else "disabled")

    def launch(cmd, extra):
        if state["proc"]:
            messagebox.showinfo("colibri", "A command is already running: wait for it, or Cancel.")
            return
        log_write(f"\n> {display_command(cmd, extra)}\n")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        try:
            proc = subprocess.Popen(cmd, cwd=HERE, env=child_env(extra), stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=flags)
        except OSError as error:
            log_write(f"could not start it: {error}\n")
            return
        state["proc"] = proc
        busy(True)
        v_status.set("Running...")

        def pump():
            while True:
                chunk = proc.stdout.read1(4096)
                if not chunk:
                    break
                events.put(("out", chunk.decode("utf-8", "replace")))
            events.put(("done", proc.wait()))

        threading.Thread(target=pump, daemon=True).start()

    def run_coli(args):
        launch([py, COLI] + args, {})

    def apply():
        if v_use_local.get():
            if not v_model_dir.get().strip():
                messagebox.showwarning("colibri", "Choose the model folder first.")
                return
            if not os.path.isfile(os.path.join(v_model_dir.get(), "config.json")) and not \
                    messagebox.askyesno("colibri", f"{v_model_dir.get()} has no config.json. Use it anyway?"):
                return
        elif not v_model.get():
            messagebox.showwarning("colibri", "Choose a model on the Model tab first.")
            nb.select(t_model)
            return
        try:
            cmd = setup_command(options(), python=py)
        except ValueError as error:
            messagebox.showerror("colibri", str(error))
            return
        r = None if v_use_local.get() else next((x for x in state["rows"] if x["id"] == v_model.get()), None)
        if r and not r["fits"] and not messagebox.askyesno(
                "colibri", f"{r['name']}: {r['reason']}.\n\nInstall it anyway?"):
            return
        if r and r["license_note"] and not messagebox.askyesno(
                "colibri", f"{r['name']}: {r['license_note']}.\n\nContinue?"):
            return
        if r:
            log_write(f"\n{r['name']}: a {r['download_gb']:.0f} GB download. It is safe to Cancel or "
                      "close this window: Install / Apply continues where it stopped.\n")
        launch(cmd, run_extra())

    def cancel():
        proc = state["proc"]
        if proc and proc.poll() is None:
            proc.terminate()
            log_write("\n(cancelled: Install / Apply continues from where it stopped)\n")

    def poll():
        try:
            while True:
                kind, payload = events.get_nowait()
                if kind == "out":
                    log_write(payload)
                elif kind == "done":
                    state["proc"] = None
                    busy(False)
                    v_status.set("Finished." if payload == 0 else f"Stopped with code {payload}: see the output.")
                    load_current()
                elif kind == "hw":
                    hw, rows, drive_rows = payload
                    state["hw"], state["rows"] = hw, rows
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
                    if v_status.get().startswith("Detecting"):
                        v_status.set("Hardware detected.")
                elif kind == "error":
                    v_status.set(payload)
                    log_write(payload + "\n")
        except queue.Empty:
            pass
        root.after(100, poll)

    def on_close():
        proc = state["proc"]
        if proc and proc.poll() is None:
            if not messagebox.askyesno("colibri", "A command is still running. Stop it and close?\n"
                                                  "(A download continues from where it stopped next time.)"):
                return
            proc.terminate()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    refresh_mode()
    load_current()
    detect()
    poll()
    nb.select(t_cur if saved else t_model)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
