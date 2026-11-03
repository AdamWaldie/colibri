#!/usr/bin/env python3
"""The models already on this PC, wherever they are, and moving one to another drive
(the configurator's "My models" tab).

Where it looks:

    colibri       the folders `coli setup` downloads into: the configured one, the
                  default ~/colibri-models, and <drive>:\\colibri-models on every
                  fixed drive; each model is a subfolder with config.json (or the
                  gguf.json of an imported GGUF)
    Ollama        its own store (OLLAMA_MODELS, else ~/.ollama/models): the
                  manifests name each model ("qwen3-coder:30b") and point at a blob
                  that is a GGUF file
    LM Studio     ~/.lmstudio/models and the older ~/.cache/lm-studio/models
    Hugging Face  the hub cache (HF_HUB_CACHE, HF_HOME/hub, ~/.cache/huggingface/hub)
    folder        any folder the user adds

A GGUF runs in colibri once `coli import` has turned it into a model folder (in
place: the weights stay where they are), and only for the architectures the
importer knows (tools/gguf_import.py ARCHES): the large mixture-of-experts
families colibri's engines serve. Every other GGUF is listed with the reason it
is not offered.

Moving a model folder: a rename on the same drive; across drives a copy (sizes
checked file by file) into <dest>.partial, renamed into place, and only then the
original deleted. The caller updates setup.json with follow_move() when the moved
folder is the configured one.

Standard library only; nothing here starts a process.
"""
import json
import os
import re
import shutil
import struct
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import setup_download  # noqa: E402
import setup_flow  # noqa: E402
import setup_hw  # noqa: E402

GB = 1_000_000_000
SOURCES = ("colibri", "Ollama", "LM Studio", "Hugging Face", "folder")
OLLAMA_MODEL_LAYER = "application/vnd.ollama.image.model"
SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)
NOT_A_MODEL_RE = re.compile(r"^(mmproj|mtp)[-_.]", re.I)   # vision projectors, MTP heads


# ---------------------------------------------------------------- what colibri runs


def supported_arches():
    """The GGUF architectures `coli import` turns into a colibri model: the importer's
    own table, read from its source so nothing has to be imported to know it."""
    try:
        with open(os.path.join(HERE, "tools", "gguf_import.py"), encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return set()
    table = re.search(r"^ARCHES\s*=\s*\{(.*?)\}\s*$", text, re.M | re.S)
    return set(re.findall(r'"([A-Za-z0-9_.-]+)"\s*:', table.group(1))) if table else set()


# ---------------------------------------------------------------- GGUF header


GGUF_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
               10: "<Q", 11: "<q", 12: "<d"}
GGUF_STRING, GGUF_ARRAY = 8, 9


def gguf_info(path, max_keys=4096):
    """{"arch", "name"} from a GGUF's metadata, reading only until both are seen
    (they come first in the files llama.cpp and Ollama write). None for a file
    that is not a readable GGUF."""
    found = {}
    try:
        with open(path, "rb") as handle:
            head = handle.read(24)
            if len(head) < 24 or head[:4] != b"GGUF":
                return None
            version, _tensors, n_kv = struct.unpack("<IQQ", head[4:24])
            if version < 2:
                return None

            def read(n):
                data = handle.read(n)
                if len(data) != n:
                    raise ValueError("truncated")
                return data

            def string():
                (n,) = struct.unpack("<Q", read(8))
                if n > 1 << 24:
                    raise ValueError("implausible string length")
                return read(n).decode("utf-8", "replace")

            def skip(kind):
                if kind in GGUF_SCALAR:
                    read(struct.calcsize(GGUF_SCALAR[kind]))
                elif kind == GGUF_STRING:
                    string()
                elif kind == GGUF_ARRAY:
                    inner, count = struct.unpack("<IQ", read(12))
                    if inner in GGUF_SCALAR:
                        handle.seek(struct.calcsize(GGUF_SCALAR[inner]) * count, os.SEEK_CUR)
                    else:
                        for _ in range(count):
                            skip(inner)
                else:
                    raise ValueError(f"unknown metadata type {kind}")

            for _ in range(min(n_kv, max_keys)):
                key = string()
                (kind,) = struct.unpack("<I", read(4))
                if key in ("general.architecture", "general.name") and kind == GGUF_STRING:
                    found[key] = string()
                    if len(found) == 2:
                        break
                else:
                    skip(kind)
    except (OSError, ValueError, struct.error):
        pass
    if not found.get("general.architecture"):
        return None
    return {"arch": found["general.architecture"], "name": found.get("general.name") or ""}


# ---------------------------------------------------------------- where to look


def _unique(paths):
    seen, out = set(), []
    for path in paths:
        full = os.path.abspath(os.path.expanduser(path))
        key = os.path.normcase(full)
        if key not in seen:
            seen.add(key)
            out.append(full)
    return out


def colibri_roots(cfg=None, drives=None):
    """colibri's own model folders: configured, default, and <drive>\\colibri-models."""
    cfg = cfg if cfg is not None else (setup_flow.load_config() or {})
    out = []
    if cfg.get("models_root"):
        out.append(cfg["models_root"])
    if cfg.get("model_dir"):
        out.append(os.path.dirname(cfg["model_dir"]))
    out.append(setup_flow.default_models_root())
    for drive in (setup_hw.windows_fixed_drives() if drives is None else drives):
        out.append(os.path.join(drive, "colibri-models"))
    return _unique(out)


def ollama_root(env=None, home=None):
    env = os.environ if env is None else env
    return env.get("OLLAMA_MODELS") or os.path.join(home or os.path.expanduser("~"), ".ollama", "models")


def other_roots(env=None, home=None):
    """[(source, folder)] for the GGUF stores of other tools."""
    env = os.environ if env is None else env
    home = home or os.path.expanduser("~")
    if env.get("HF_HUB_CACHE"):
        hub = env["HF_HUB_CACHE"]
    elif env.get("HF_HOME"):
        hub = os.path.join(env["HF_HOME"], "hub")
    else:
        hub = os.path.join(home, ".cache", "huggingface", "hub")
    return [("LM Studio", os.path.join(home, ".lmstudio", "models")),
            ("LM Studio", os.path.join(home, ".cache", "lm-studio", "models")),
            ("Hugging Face", hub)]


# ---------------------------------------------------------------- scanning


def folder_size(path):
    total = 0
    for base, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(base, name))
            except OSError:
                pass
    return total


def colibri_models(root):
    """Model folders directly under one colibri root."""
    rows = []
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return rows
    for name in names:
        path = os.path.join(root, name)
        if name.startswith(".") or name.endswith(".partial") or not os.path.isdir(path):
            continue
        has_cfg = os.path.isfile(os.path.join(path, "config.json"))
        has_gguf = os.path.isfile(os.path.join(path, "gguf.json"))
        manifest = setup_download.read_manifest(path)
        if not (has_cfg or has_gguf or manifest):
            continue
        complete = setup_download.is_complete(path, []) if manifest else True
        title = name
        try:
            with open(os.path.join(path, "config.json"), encoding="utf-8") as handle:
                gguf = (json.load(handle) or {}).get("colibri_gguf")
            if isinstance(gguf, dict) and gguf.get("name"):
                title = gguf["name"]
        except (OSError, ValueError, AttributeError):
            pass
        ok = complete and has_cfg
        rows.append({"source": "colibri", "name": title, "id": name, "path": path, "kind": "folder",
                     "size": folder_size(path), "runs": "yes" if ok else "incomplete",
                     "why": "ready" if ok else
                     "download not finished: Use continues it" if manifest else "no config.json",
                     "repo": (manifest or {}).get("repo") or "", "linked_gguf": has_gguf})
    return rows


def ollama_models(root, arches):
    """Ollama's models: each manifest's model layer is a GGUF blob."""
    rows = []
    manifests = os.path.join(root, "manifests")
    for base, _dirs, files in os.walk(manifests):
        for tag in files:
            path = os.path.join(base, tag)
            try:
                with open(path, encoding="utf-8") as handle:
                    manifest = json.load(handle)
            except (OSError, ValueError):
                continue
            layer = next((item for item in (manifest.get("layers") or [])
                          if isinstance(item, dict) and item.get("mediaType") == OLLAMA_MODEL_LAYER), None)
            if not layer or not str(layer.get("digest", "")).startswith("sha256:"):
                continue
            blob = os.path.join(root, "blobs", layer["digest"].replace(":", "-"))
            if not os.path.isfile(blob):
                continue
            parts = os.path.relpath(path, manifests).replace(os.sep, "/").split("/")
            # registry/namespace/model/tag; the official library leaves its namespace out of the name
            model = parts[-2] if len(parts) >= 2 else parts[-1]
            namespace = parts[-3] if len(parts) >= 3 else ""
            name = (f"{namespace}/{model}" if namespace and namespace != "library" else model) + f":{parts[-1]}"
            rows.append(gguf_row("Ollama", name, blob, arches, size=layer.get("size")))
    return rows


def _split_parts(first):
    match = SPLIT_RE.search(first)
    total = int(match.group(2))
    return [SPLIT_RE.sub(f"-{i:05d}-of-{total:05d}.gguf", first) for i in range(1, total + 1)]


def gguf_files(root, source, arches, depth=6):
    """Every model GGUF under a folder (the first part of a split one; no vision
    projectors or MTP heads), at most `depth` folders down."""
    rows = []
    root = os.path.abspath(root)
    start = root.rstrip(os.sep).count(os.sep)
    for base, dirs, files in os.walk(root):
        if base.rstrip(os.sep).count(os.sep) - start >= depth:
            dirs[:] = []
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in sorted(files):
            if not name.lower().endswith(".gguf") or NOT_A_MODEL_RE.match(name):
                continue
            split = SPLIT_RE.search(name)
            if split and int(split.group(1)) != 1:
                continue
            path = os.path.join(base, name)
            size = sum(os.path.getsize(p) for p in _split_parts(path) if os.path.isfile(p)) if split else None
            title = SPLIT_RE.sub("", name) if split else name[:-5]
            if source == "Hugging Face":
                repo = re.search(r"models--(.+?)--(.+?)[\\/]", path)
                if repo:
                    title = f"{repo.group(1)}/{repo.group(2)}: {title}"
            rows.append(gguf_row(source, title, path, arches, size=size))
    return rows


def gguf_row(source, name, path, arches, size=None):
    info = gguf_info(path)
    arch = (info or {}).get("arch") or ""
    if size is None:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
    if not info:
        runs, why = "no", "not a GGUF file colibri can read"
    elif arch in arches:
        runs, why = "import", f"{arch}: runs after Import (the file stays where it is)"
    else:
        runs, why = "no", f"{arch}: an architecture colibri's engines do not run"
    return {"source": source, "name": name, "id": name, "path": path, "kind": "gguf",
            "size": int(size or 0), "runs": runs, "why": why, "arch": arch}


def scan(extra_folders=(), cfg=None, env=None, home=None, drives=None):
    """Everything found: what runs now first, then what runs after Import, then the rest."""
    arches = supported_arches()
    rows = []
    for root in colibri_roots(cfg, drives):
        rows += colibri_models(root)
    rows += ollama_models(ollama_root(env, home), arches)
    for source, root in other_roots(env, home):
        if os.path.isdir(root):
            rows += gguf_files(root, source, arches)
    for folder in extra_folders:
        if os.path.isdir(folder):
            rows += colibri_models(folder)
            rows += gguf_files(folder, "folder", arches)
    seen, out = set(), []
    for row in rows:
        key = os.path.normcase(os.path.realpath(row["path"]))
        if key not in seen:
            seen.add(key)
            out.append(row)
    order = {"yes": 0, "incomplete": 1, "import": 2, "no": 3}
    out.sort(key=lambda r: (order.get(r["runs"], 9), SOURCES.index(r["source"]), r["name"].lower()))
    return out


# ---------------------------------------------------------------- import and use


def import_target(row, models_root):
    """Where `coli import` writes the colibri folder for a GGUF: the models folder,
    named after the model ("qwen3-coder:30b" -> qwen3-coder-30b-gguf)."""
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", row["name"]).strip("-.") or "imported"
    return os.path.join(models_root, base[:80] + "-gguf")


def import_command(row, out, python="python", coli=None):
    return [python, coli or os.path.join(HERE, "coli"), "import", row["path"], "--out", out]


def use_command(row, python="python", coli=None, start=True):
    """`coli setup` for a model folder already on disk. A catalog download keeps its
    catalog identity (--model ID --dir ROOT), so its expert profile still applies."""
    cmd = [python, coli or os.path.join(HERE, "coli"), "setup", "--yes"]
    entry = None
    try:
        import setup_catalog
        entry = setup_catalog.by_id(row.get("id") or "")
    except (KeyError, ImportError):
        pass
    if entry and row["source"] == "colibri" and row.get("repo") == entry.repo:
        cmd += ["--model", entry.id, "--dir", os.path.dirname(row["path"])]
    else:
        cmd += ["--model-dir", row["path"]]
    cmd.append("--background" if start else "--no-start")
    return cmd


# ---------------------------------------------------------------- moving


class MoveError(RuntimeError):
    pass


def same_drive(a, b):
    """Whether a rename can move a to b (the same volume)."""
    a, b = os.path.abspath(a), os.path.abspath(b)
    if sys.platform == "win32":
        return os.path.splitdrive(a)[0].lower() == os.path.splitdrive(b)[0].lower()
    try:
        return os.stat(a).st_dev == os.stat(setup_hw.existing_ancestor(b)).st_dev
    except OSError:
        return False


def move_problem(src, dest_root, size=None, free=None, in_use=False):
    """'' when src can move into dest_root, else why not, in words."""
    src = os.path.abspath(src)
    dest = os.path.join(os.path.abspath(dest_root), os.path.basename(src))
    if in_use:
        return "the server is using this model: Stop it first"
    if not os.path.isdir(src):
        return f"{src} is not a folder"
    if os.path.normcase(dest) == os.path.normcase(src):
        return "it is already there"
    if os.path.normcase(os.path.abspath(dest_root)).startswith(os.path.normcase(src) + os.sep) or \
            os.path.normcase(os.path.abspath(dest_root)) == os.path.normcase(src):
        return "the destination is inside the model's own folder"
    if os.path.exists(dest):
        return f"{dest} already exists"
    if not same_drive(src, dest_root):
        size = folder_size(src) if size is None else size
        free = setup_hw.disk_free(dest_root) if free is None else free
        if free is not None and free < size + GB:
            return (f"not enough space: it needs {setup_hw.fmt_gb(size)}, "
                    f"{setup_hw.fmt_gb(free)} free there")
    return ""


def move_model(src, dest_root, progress=None, cancelled=None):
    """Move a model folder into dest_root; returns the new path. Raises MoveError.
    progress(done_bytes, total_bytes) is called while copying across drives;
    cancelled() returning True stops the copy and removes what it wrote, leaving
    the original untouched."""
    problem = move_problem(src, dest_root)
    if problem:
        raise MoveError(problem)
    src = os.path.abspath(src)
    dest = os.path.join(os.path.abspath(dest_root), os.path.basename(src))
    os.makedirs(dest_root, exist_ok=True)
    if same_drive(src, dest_root):
        try:
            os.replace(src, dest)
        except OSError as error:
            raise MoveError(f"could not move it ({error}); is a program using its files?") from error
        return dest
    partial = dest + ".partial"
    shutil.rmtree(partial, ignore_errors=True)
    total = folder_size(src)
    done = 0

    def check():
        if cancelled and cancelled():
            raise MoveError("cancelled: the model is still in its old place")

    try:
        for base, _dirs, files in os.walk(src):
            target = os.path.join(partial, os.path.relpath(base, src))
            os.makedirs(target, exist_ok=True)
            for name in files:
                check()
                source = os.path.join(base, name)
                copy = os.path.join(target, name)
                with open(source, "rb") as fin, open(copy, "wb") as fout:
                    while True:
                        chunk = fin.read(16 << 20)
                        if not chunk:
                            break
                        fout.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, total)
                        check()
                shutil.copystat(source, copy)
                if os.path.getsize(copy) != os.path.getsize(source):
                    raise MoveError(f"{copy}: the copy's size does not match the original")
        os.replace(partial, dest)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    shutil.rmtree(src)
    return dest


def follow_move(cfg, old, new):
    """The run configuration after a model folder moved from old to new (unchanged
    when it is about another model)."""
    norm = lambda p: os.path.normcase(os.path.abspath(p))  # noqa: E731
    if not cfg or not cfg.get("model_dir") or norm(cfg["model_dir"]) != norm(old):
        return cfg
    cfg = dict(cfg, model_dir=new)
    if cfg.get("models_root") and norm(cfg["models_root"]) == norm(os.path.dirname(old)):
        cfg["models_root"] = os.path.dirname(new)
    args = list(cfg.get("args") or [])
    for i in range(len(args) - 1):
        if args[i] == "--model" and norm(args[i + 1]) == norm(old):
            args[i + 1] = new
    cfg["args"] = args
    return cfg
