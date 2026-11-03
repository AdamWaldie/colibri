#!/usr/bin/env python3
"""Get a model from the internet into colibri: Ollama's library, a Hugging Face
GGUF repository, or a direct link to a .gguf file (the configurator's
"Get a model" box, and `python c/model_fetch.py ...` from a terminal).

    resolve(text)  what the text names, without downloading it: the variants
                   (quantizations) there are, their sizes, and the architecture,
                   read from the first megabytes of the file. colibri runs the
                   architectures tools/gguf_import.py knows; any other is refused
                   before a byte of the model is downloaded.
    main()         download (resumable, every byte checked against the SHA-256
                   Ollama and Hugging Face publish), then `coli import` into
                   <models folder>/<name>-gguf. Progress goes to install-state.json,
                   as `coli setup`'s download does, so the window shows it.

What the text may be:

    qwen3-coder:30b, qwen3-coder          Ollama's library (tag "latest" if none)
    user/model:tag, ollama.com/library/x  Ollama, a user's namespace or a link
    unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF, hf.co/org/repo, a huggingface.co link
    hf.co/org/repo:Q4_K_M                 a Hugging Face repo and its variant
    https://.../model.gguf                any direct link to a GGUF file

The downloaded GGUF stays in <models folder>/gguf-downloads/<name>/; the imported
folder reads it there (link mode), so both together are the model.

Standard library only.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import model_library  # noqa: E402
import setup_download  # noqa: E402
import setup_flow  # noqa: E402

OLLAMA_REGISTRY = "https://registry.ollama.ai"
MANIFEST_ACCEPT = "application/vnd.docker.distribution.manifest.v2+json"
PEEK_BYTES = 8 << 20
QUANT_RE = re.compile(r"(?i)(?:^|[-_.])((?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4(?:_MOE)?))(?=[-_.]|$)")
PREFERRED = ("Q4_K_M", "UD-Q4_K_XL", "Q4_K_S", "Q4_0", "Q5_K_M", "Q8_0")
USER_AGENT = "colibri-model-fetch"


class FetchError(RuntimeError):
    pass


# ---------------------------------------------------------------- what the text names


def parse_query(text):
    """{"source": "ollama", "namespace", "model", "tag"} | {"source": "hf", "repo",
    "revision", "variant"[, "file"]} | {"source": "url", "url"}. Raises FetchError
    for text that names nothing it can fetch."""
    text = (text or "").strip()
    if not text:
        raise FetchError("type a model name, a Hugging Face repository or a link")
    lower = text.lower()
    if lower.startswith(("http://", "https://")):
        url = urllib.parse.urlparse(text)
        host = url.netloc.lower()
        parts = [p for p in url.path.split("/") if p]
        if host.endswith("huggingface.co") or host == "hf.co":
            if parts and parts[0] in ("datasets", "spaces"):
                raise FetchError("that is not a model repository")
            if len(parts) >= 5 and parts[2] in ("resolve", "blob") and parts[-1].lower().endswith(".gguf"):
                return {"source": "hf", "repo": f"{parts[0]}/{parts[1]}", "revision": parts[3],
                        "file": "/".join(parts[4:]), "variant": ""}
            if len(parts) >= 2:
                return {"source": "hf", "repo": f"{parts[0]}/{parts[1]}", "revision": "main", "variant": ""}
            raise FetchError("that Hugging Face link names no repository")
        if host.endswith("ollama.com"):
            path = "/".join(parts)
            return parse_query(path[len("library/"):] if path.startswith("library/") else path)
        if url.path.lower().endswith(".gguf"):
            return {"source": "url", "url": text}
        raise FetchError("a link has to point at a .gguf file, a Hugging Face repository or ollama.com")
    if lower.startswith(("hf.co/", "huggingface.co/")):
        repo, _, variant = text.split("/", 1)[1].partition(":")
        if repo.count("/") != 1:
            raise FetchError("a Hugging Face repository is owner/name")
        return {"source": "hf", "repo": repo, "revision": "main", "variant": variant}
    name, _, tag = text.partition(":")
    if not re.fullmatch(r"[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)?", name) or \
            (tag and not re.fullmatch(r"[A-Za-z0-9._-]+", tag)):
        raise FetchError(f"{text!r} is not a model name")
    if "/" in name and not tag:
        return {"source": "hf", "repo": name, "revision": "main", "variant": ""}
    namespace, _, model = name.rpartition("/")
    return {"source": "ollama", "namespace": namespace or "library", "model": model, "tag": tag or "latest"}


def display_name(query):
    if query["source"] == "ollama":
        ns = "" if query["namespace"] == "library" else query["namespace"] + "/"
        return f"{ns}{query['model']}:{query['tag']}"
    if query["source"] == "hf":
        return query["repo"]
    return os.path.basename(urllib.parse.urlparse(query["url"]).path)


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.")[:80] or "model"


# ---------------------------------------------------------------- HTTP


def _request(url, headers=None):
    request = urllib.request.Request(url)
    request.add_header("User-Agent", USER_AGENT)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    return request


def _json(url, headers=None, opener=None, timeout=30):
    try:
        with (opener or urllib.request.urlopen)(_request(url, headers), timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            raise FetchError("access denied: a gated or private repository needs a Hugging Face token") from error
        if error.code == 404:
            raise FetchError("not found: check the name") from error
        raise FetchError(f"the server answered {error.code}") from error
    except (OSError, ValueError) as error:
        raise FetchError(f"could not reach it ({error})") from error


def peek_arch(url, headers=None, opener=None):
    """The GGUF architecture from the first megabytes of a remote file, or None."""
    headers = dict(headers or {}, Range=f"bytes=0-{PEEK_BYTES - 1}")
    try:
        with (opener or urllib.request.urlopen)(_request(url, headers), timeout=60) as response:
            data = response.read(PEEK_BYTES)
    except (OSError, ValueError):
        return None
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as handle:
        handle.write(data)
        path = handle.name
    try:
        return model_library.gguf_info(path)
    finally:
        os.unlink(path)


def _auth(token):
    return {"Authorization": f"Bearer {token}"} if token else {}


# ---------------------------------------------------------------- the sources


def ollama_variants(query, opener=None):
    base = f"{OLLAMA_REGISTRY}/v2/{query['namespace']}/{query['model']}"
    manifest = _json(f"{base}/manifests/{query['tag']}", {"Accept": MANIFEST_ACCEPT}, opener)
    layer = next((item for item in manifest.get("layers") or []
                  if item.get("mediaType") == model_library.OLLAMA_MODEL_LAYER), None)
    if not layer:
        raise FetchError(f"{display_name(query)} has no model file (an adapter, or a cloud-only model?)")
    digest = layer["digest"]
    return [{"label": query["tag"], "size": int(layer["size"]),
             "files": [{"path": f"{safe_name(query['model'])}-{safe_name(query['tag'])}.gguf",
                        "size": int(layer["size"]), "sha256": digest.split(":", 1)[1],
                        "url": f"{base}/blobs/{digest}"}]}]


def quant_of(path):
    name = os.path.basename(model_library.SPLIT_RE.sub(".gguf", path))[:-5]
    match = None
    for match in QUANT_RE.finditer(name):
        pass
    return match.group(1).upper() if match else name


def hf_variants(query, token="", opener=None):
    """The GGUFs of a repository grouped by quantization (a split model is one
    variant); vision projectors and MTP heads left out."""
    revision = query.get("revision", "main")
    try:
        files = setup_download.list_repo_files(query["repo"], revision, include=("*.gguf",),
                                               token=token or None, opener=opener)
    except setup_download.DownloadError as error:
        raise FetchError(str(error)) from error
    wanted = model_library.SPLIT_RE.sub("", query["file"]) if query.get("file") else None
    groups = {}
    for spec in files:
        if model_library.NOT_A_MODEL_RE.match(os.path.basename(spec["path"])):
            continue
        if wanted and model_library.SPLIT_RE.sub("", spec["path"]) != wanted and spec["path"] != query["file"]:
            continue
        label = quant_of(spec["path"])
        group = groups.setdefault(label, {"label": label, "size": 0, "files": []})
        group["files"].append(dict(spec, url=setup_download.file_url(query["repo"], revision, spec["path"])))
        group["size"] += int(spec["size"])
    for group in groups.values():
        group["files"].sort(key=lambda s: s["path"])
    if not groups:
        raise FetchError(f"{query['repo']} has no GGUF files colibri could import")
    return sorted(groups.values(), key=lambda g: g["size"])


def url_variants(query, opener=None):
    try:
        with (opener or urllib.request.urlopen)(_request(query["url"], {"Range": "bytes=0-0"}), timeout=30) as response:
            total = response.headers.get("Content-Range", "").rpartition("/")[2]
            size = int(total) if total.isdigit() else int(response.headers.get("Content-Length") or 0)
    except (OSError, ValueError) as error:
        raise FetchError(f"could not reach the link ({error})") from error
    name = os.path.basename(urllib.parse.urlparse(query["url"]).path)
    return [{"label": quant_of(name), "size": size, "files": [{"path": name, "size": size, "url": query["url"]}]}]


def preferred(variants):
    by_label = {v["label"].upper(): v for v in variants}
    for label in PREFERRED:
        if label in by_label:
            return by_label[label]
    q4 = [v for v in variants if v["label"].upper().startswith(("Q4", "UD-Q4", "IQ4"))]
    return (q4 or variants)[0]


def check(variant, token="", opener=None):
    """{"arch", "supported", "why"} for one variant, from its first file's header."""
    first = variant["files"][0]
    info = peek_arch(first["url"], _auth(token) if "huggingface.co" in first["url"] else None, opener)
    if not info:
        return {"arch": "", "supported": False, "why": "its first megabytes are not a GGUF header colibri can read"}
    if info["arch"] not in model_library.supported_arches():
        return {"arch": info["arch"], "supported": False,
                "why": f"{info['arch']} is an architecture colibri's engines do not run"}
    return {"arch": info["arch"], "supported": True,
            "why": f"{info['arch']}: colibri runs it" + (f" ({info['name']})" if info.get("name") else "")}


def resolve(text, token="", opener=None, variant=None):
    """What the text names, without downloading the model:
    {"query", "name", "variants", "chosen", "arch", "supported", "why"}."""
    query = parse_query(text)
    if query["source"] == "ollama":
        variants = ollama_variants(query, opener)
    elif query["source"] == "hf":
        variants = hf_variants(query, token, opener)
    else:
        variants = url_variants(query, opener)
    wanted = variant or query.get("variant")
    if wanted:
        chosen = next((v for v in variants if v["label"].upper() == wanted.upper()), None)
        if not chosen:
            raise FetchError(f"there is no {wanted} variant; there are " + ", ".join(v["label"] for v in variants))
    else:
        chosen = preferred(variants)
    return dict(check(chosen, token, opener), query=query, name=display_name(query),
                variants=variants, chosen=chosen)


# ---------------------------------------------------------------- where it goes


def download_dir(name, models_root):
    return os.path.join(models_root, "gguf-downloads", safe_name(name))


def import_dir(name, variant, models_root):
    base = safe_name(name.replace("/", "-").replace(":", "-"))
    label = safe_name(variant["label"])
    return os.path.join(models_root, (base if label.lower() in base.lower() else f"{base}-{label}") + "-gguf")


def fetch_command(text, variant, models_root, python="python"):
    cmd = [python, os.path.join(HERE, "model_fetch.py"), text, "--root", models_root]
    return cmd + (["--variant", variant] if variant else [])


# ---------------------------------------------------------------- download and import


def download(plan, models_root, token="", out=print, opener=None):
    """Every file of the chosen variant into gguf-downloads/<name>/; returns the
    first file's path. Resumable; checked against the published SHA-256."""
    folder = download_dir(plan["name"], models_root)
    files = plan["chosen"]["files"]
    total = sum(int(f["size"]) for f in files)
    have = 0
    for spec in files:
        for suffix in ("", ".part"):
            path = os.path.join(folder, spec["path"] + suffix)
            if os.path.isfile(path):
                have += os.path.getsize(path)
                break
    free = model_library.setup_hw.disk_free(models_root)
    if free is not None and free < total - have + (1 << 30):
        raise FetchError(f"not enough space: {model_library.setup_hw.fmt_gb(total - have)} more needed, "
                         f"{model_library.setup_hw.fmt_gb(free)} free in {models_root}")
    started, done_before = time.time(), 0
    for spec in files:
        dest = os.path.join(folder, spec["path"])
        out(f"downloading {spec['path']} ({model_library.setup_hw.fmt_gb(spec['size'])})")

        def progress(have_bytes, _size, _base=done_before):
            done = _base + have_bytes
            rate = done / max(1e-3, time.time() - started)
            setup_flow.write_state_throttled("download", model=plan["name"], done=done, total=total, rate=rate,
                                             eta_seconds=(total - done) / rate if rate else None, model_dir=folder)

        setup_download.download_file(spec["url"], dest, spec, progress=progress, opener=opener,
                                     token=token if "huggingface.co" in spec["url"] else None)
        done_before += int(spec["size"])
    setup_flow.write_state("download", model=plan["name"], done=total, total=total, model_dir=folder)
    return os.path.join(folder, files[0]["path"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download a model from Ollama, Hugging Face or a link, "
                                             "then import it into colibri.")
    ap.add_argument("what", help="qwen3-coder:30b, owner/repo, hf.co/owner/repo:Q4_K_M, or a .gguf link")
    ap.add_argument("--root", default=None, help="the models folder (default: the one `coli setup` uses)")
    ap.add_argument("--variant", default=None, help="the quantization (default: Q4_K_M or the nearest)")
    ap.add_argument("--info", action="store_true", help="only say what it is, as JSON; download nothing")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):          # a cp437/cp1252 terminal and a model name with ™
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    token = setup_download.hf_token() or ""
    cfg = setup_flow.load_config() or {}
    root = os.path.abspath(os.path.expanduser(a.root or cfg.get("models_root") or setup_flow.default_models_root()))
    try:
        plan = resolve(a.what, token, variant=a.variant)
        target = import_dir(plan["name"], plan["chosen"], root)
        if a.info:
            print(json.dumps({"name": plan["name"], "arch": plan["arch"], "supported": plan["supported"],
                              "why": plan["why"], "chosen": plan["chosen"]["label"], "import_dir": target,
                              "variants": [{"label": v["label"], "size": v["size"]} for v in plan["variants"]]},
                             indent=2))
            return 0
        if not plan["supported"]:
            raise FetchError(f"{plan['name']}: {plan['why']}; nothing was downloaded")
        print(f"{plan['name']} ({plan['chosen']['label']}, {model_library.setup_hw.fmt_gb(plan['chosen']['size'])}): "
              f"{plan['why']}", flush=True)
        setup_flow.write_state("starting", model=plan["name"])
        first = download(plan, root, token, out=lambda line: print(line, flush=True))
        print(f"importing into {target}", flush=True)
        setup_flow.write_state("build", model=plan["name"])
        code = subprocess.call([sys.executable, os.path.join(HERE, "coli"), "import", first, "--out", target])
        if code:
            raise FetchError(f"the import stopped (code {code}); the download is kept: run this again to retry")
        setup_flow.write_state("ready", model=plan["name"], model_dir=target)
        print(f"READY {target}", flush=True)
        return 0
    except (FetchError, setup_download.DownloadError, OSError) as error:
        # OSError: a full disk, or Windows holding the finished file (antivirus) at the rename
        setup_flow.write_state("error", message=str(error))
        print(f"\nstopped: {error}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        setup_flow.write_state("interrupted")
        print("\ninterrupted: run the same command again to continue", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
