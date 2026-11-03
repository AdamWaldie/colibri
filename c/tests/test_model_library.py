"""Finding the models already on this PC (colibri's folders, Ollama, LM Studio, the
Hugging Face cache, added folders) and moving one to another drive (model_library.py).

Real files in a temporary folder: GGUF headers written byte for byte as llama.cpp
writes them, an Ollama store laid out as Ollama lays it out, model folders with
colibri's download manifest."""
import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import model_library  # noqa: E402
import setup_download  # noqa: E402


def gguf_bytes(arch, name="", extra_keys=0, tokens=0):
    """A GGUF v3 header: a few keys and a token array before the architecture (as
    some writers order them), the architecture and name, then more; no tensors."""
    def string(text):
        data = text.encode()
        return struct.pack("<Q", len(data)) + data

    kvs = []
    for i in range(extra_keys):
        kvs.append(string(f"x.key{i}") + struct.pack("<I", 4) + struct.pack("<I", i))
    kvs.append(string("tokenizer.ggml.tokens") + struct.pack("<I", 9) + struct.pack("<IQ", 8, tokens)
               + b"".join(string(f"t{i}") for i in range(tokens)))
    kvs.append(string("general.architecture") + struct.pack("<I", 8) + string(arch))
    if name:
        kvs.append(string("general.name") + struct.pack("<I", 8) + string(name))
    kvs.append(string("x.scores") + struct.pack("<I", 9) + struct.pack("<IQ", 6, 3) + b"\0" * 12)
    return b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + b"".join(kvs)


def write(path, data=b""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)
    return path


def manifest(folder, files, complete=True, repo="r/q"):
    data = {"repo": repo, "complete": complete,
            "files": [{"path": name, "size": size} for name, size in files]}
    write(os.path.join(folder, setup_download.MANIFEST), json.dumps(data).encode())


class Header(unittest.TestCase):
    def test_architecture_and_name_after_other_keys_and_arrays(self):
        with tempfile.TemporaryDirectory() as d:
            path = write(os.path.join(d, "m.gguf"), gguf_bytes("qwen3moe", "Qwen3 Coder 30B", extra_keys=3, tokens=50))
            self.assertEqual(model_library.gguf_info(path), {"arch": "qwen3moe", "name": "Qwen3 Coder 30B"})

    def test_not_a_gguf_or_truncated(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(model_library.gguf_info(write(os.path.join(d, "a.gguf"), b"PK\x03\x04" + b"\0" * 40)))
            whole = gguf_bytes("llama", tokens=5)
            cut = whole[:whole.index(b"general.architecture") - 4]
            self.assertIsNone(model_library.gguf_info(write(os.path.join(d, "b.gguf"), cut)))
            self.assertIsNone(model_library.gguf_info(os.path.join(d, "missing.gguf")))

    def test_supported_arches_come_from_the_importer(self):
        arches = model_library.supported_arches()
        for arch in ("qwen3moe", "qwen35moe", "glm-dsa", "deepseek4"):
            self.assertIn(arch, arches)
        self.assertNotIn("llama", arches)


class Scanning(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = self.tmp.name
        self.root = os.path.join(self.home, "colibri-models")

    def tearDown(self):
        self.tmp.cleanup()

    def ollama(self, name, tag, arch, namespace="library"):
        store = os.path.join(self.home, ".ollama", "models")
        digest = "sha256:" + hashlib.sha256(f"{namespace}/{name}:{tag}".encode()).hexdigest()
        write(os.path.join(store, "blobs", digest.replace(":", "-")), gguf_bytes(arch, tokens=3))
        data = {"layers": [{"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:" + "0" * 64},
                           {"mediaType": "application/vnd.ollama.image.model", "digest": digest, "size": 123}]}
        write(os.path.join(store, "manifests", "registry.ollama.ai", namespace, name, tag), json.dumps(data).encode())

    def scan(self, extra=()):
        return model_library.scan(extra, cfg={"models_root": self.root}, env={}, home=self.home, drives=[])

    def test_everything_found_and_ordered(self):
        # colibri: a finished catalog download, an unfinished one, a stray folder
        done = os.path.join(self.root, "qwen36-35b")
        write(os.path.join(done, "config.json"), b"{}")
        write(os.path.join(done, "model.safetensors"), b"x" * 10)
        manifest(done, [("model.safetensors", 10)])
        half = os.path.join(self.root, "glm-5.2")
        write(os.path.join(half, "config.json"), b"{}")
        manifest(half, [("model.safetensors", 999)], complete=False)
        write(os.path.join(self.root, "notes", "readme.txt"), b"not a model")
        # Ollama: one colibri runs, one it does not, one from a user namespace
        self.ollama("qwen3-coder", "30b", "qwen3moe")
        self.ollama("llama3.2", "latest", "llama")
        self.ollama("mymodel", "q4", "qwen35moe", namespace="someone")
        # LM Studio, the Hugging Face cache, an added folder (a split model and side files)
        studio = os.path.join(self.home, ".lmstudio", "models", "pub", "repo")
        write(os.path.join(studio, "Model-Q4_K_M.gguf"), gguf_bytes("glm-dsa"))
        write(os.path.join(studio, "mmproj-F16.gguf"), gguf_bytes("clip"))
        hf = os.path.join(self.home, ".cache", "huggingface", "hub", "models--unsloth--Qwen3-GGUF", "snapshots", "abc")
        write(os.path.join(hf, "Qwen3-Q8_0.gguf"), gguf_bytes("qwen3"))
        extra = os.path.join(self.home, "Downloads")
        write(os.path.join(extra, "Big-00001-of-00002.gguf"), gguf_bytes("deepseek4"))
        write(os.path.join(extra, "Big-00002-of-00002.gguf"), b"y" * 100)
        write(os.path.join(extra, "mtp-Big-Q8_0.gguf"), gguf_bytes("qwen4exp"))

        rows = self.scan([extra])
        by = {r["name"]: r for r in rows}
        self.assertEqual(by["qwen36-35b"]["runs"], "yes")
        self.assertEqual(by["glm-5.2"]["runs"], "incomplete")
        self.assertNotIn("notes", by)
        self.assertEqual(by["qwen3-coder:30b"]["source"], "Ollama")
        self.assertEqual(by["qwen3-coder:30b"]["runs"], "import")
        self.assertEqual(by["qwen3-coder:30b"]["size"], 123)                  # the manifest's size
        self.assertEqual(by["llama3.2:latest"]["runs"], "no")
        self.assertIn("llama", by["llama3.2:latest"]["why"])
        self.assertIn("someone/mymodel:q4", by)
        self.assertEqual(by["Model-Q4_K_M"]["source"], "LM Studio")
        self.assertEqual(by["unsloth/Qwen3-GGUF: Qwen3-Q8_0"]["source"], "Hugging Face")
        big = by["Big"]
        self.assertEqual(big["runs"], "import")
        self.assertEqual(big["size"], len(gguf_bytes("deepseek4")) + 100)   # both parts counted
        self.assertFalse(any("mmproj" in n or "mtp-" in n or "00002" in n for n in by))
        order = [r["runs"] for r in rows]
        self.assertEqual(order, sorted(order, key={"yes": 0, "incomplete": 1, "import": 2, "no": 3}.get))

    def test_nothing_installed_is_an_empty_list(self):
        self.assertEqual(self.scan(), [])

    def test_ollama_models_dir_from_the_environment(self):
        store = os.path.join(self.home, "elsewhere")
        self.assertEqual(model_library.ollama_root({"OLLAMA_MODELS": store}, self.home), store)
        self.assertEqual(model_library.ollama_root({}, self.home), os.path.join(self.home, ".ollama", "models"))

    def test_hugging_face_cache_from_the_environment(self):
        roots = {p for _s, p in model_library.other_roots({"HF_HOME": "/hf"}, self.home)}
        self.assertIn(os.path.join("/hf", "hub"), roots)
        roots = {p for _s, p in model_library.other_roots({"HF_HUB_CACHE": "/c"}, self.home)}
        self.assertIn("/c", roots)

    def test_colibri_roots_include_every_drive(self):
        roots = model_library.colibri_roots({"models_root": "/m"}, drives=["D:\\", "E:\\"])
        self.assertEqual(roots[0], os.path.abspath("/m"))
        self.assertIn(os.path.abspath(os.path.join("D:\\", "colibri-models")), roots)


class ImportAndUse(unittest.TestCase):
    def test_import_target_and_command(self):
        row = {"name": "qwen3-coder:30b", "path": "/o/blobs/sha256-ab", "source": "Ollama"}
        out = model_library.import_target(row, "/m")
        self.assertEqual(out, os.path.join("/m", "qwen3-coder-30b-gguf"))
        self.assertEqual(model_library.import_command(row, out, python="py", coli="coli"),
                         ["py", "coli", "import", "/o/blobs/sha256-ab", "--out", out])

    def test_use_keeps_catalog_identity(self):
        import setup_catalog
        entry = setup_catalog.by_id("qwen36-35b")
        row = {"source": "colibri", "id": "qwen36-35b", "path": "/d/m/qwen36-35b", "repo": entry.repo}
        self.assertEqual(model_library.use_command(row, python="py", coli="coli")[3:],
                         ["--yes", "--model", "qwen36-35b", "--dir", "/d/m", "--background"])
        other = {"source": "colibri", "id": "mine", "path": "/d/m/mine", "repo": ""}
        self.assertEqual(model_library.use_command(other, python="py", coli="coli", start=False)[3:],
                         ["--yes", "--model-dir", "/d/m/mine", "--no-start"])

    def test_use_commands_parse_with_coli_setup(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_configurator import coli_setup_parser
        parser = coli_setup_parser()
        for row in ({"source": "colibri", "id": "qwen36-35b", "path": "/d/m/qwen36-35b",
                     "repo": __import__("setup_catalog").by_id("qwen36-35b").repo},
                    {"source": "colibri", "id": "x", "path": "/d/x", "repo": ""}):
            parser.parse_args(model_library.use_command(row, python="py", coli="coli")[2:])


class Moving(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "a", "qwen36-35b")
        write(os.path.join(self.src, "config.json"), b"{}")
        write(os.path.join(self.src, "sub", "w.bin"), b"w" * (3 << 20))
        self.dest_root = os.path.join(self.tmp.name, "b", "colibri-models")

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_drive_is_a_rename(self):
        new = model_library.move_model(self.src, self.dest_root)
        self.assertEqual(new, os.path.join(self.dest_root, "qwen36-35b"))
        self.assertFalse(os.path.exists(self.src))
        self.assertEqual(os.path.getsize(os.path.join(new, "sub", "w.bin")), 3 << 20)

    def test_across_drives_copies_then_removes(self):
        seen = []
        with mock.patch.object(model_library, "same_drive", return_value=False), \
                mock.patch.object(model_library.setup_hw, "disk_free", return_value=100 * model_library.GB):
            new = model_library.move_model(self.src, self.dest_root, progress=lambda d, t: seen.append((d, t)))
        self.assertFalse(os.path.exists(self.src))
        self.assertFalse(os.path.exists(new + ".partial"))
        self.assertEqual(os.path.getsize(os.path.join(new, "sub", "w.bin")), 3 << 20)
        self.assertEqual(seen[-1][0], seen[-1][1])

    def test_cancel_leaves_the_original(self):
        with mock.patch.object(model_library, "same_drive", return_value=False), \
                mock.patch.object(model_library.setup_hw, "disk_free", return_value=100 * model_library.GB), \
                self.assertRaises(model_library.MoveError):
            model_library.move_model(self.src, self.dest_root, cancelled=lambda: True)
        self.assertTrue(os.path.isfile(os.path.join(self.src, "sub", "w.bin")))
        self.assertFalse(os.path.exists(os.path.join(self.dest_root, "qwen36-35b.partial")))
        self.assertFalse(os.path.exists(os.path.join(self.dest_root, "qwen36-35b")))

    def test_refusals(self):
        self.assertIn("Stop it first", model_library.move_problem(self.src, self.dest_root, in_use=True))
        self.assertIn("already there", model_library.move_problem(self.src, os.path.dirname(self.src)))
        self.assertIn("inside", model_library.move_problem(self.src, os.path.join(self.src, "x")))
        with mock.patch.object(model_library, "same_drive", return_value=False):
            self.assertIn("not enough space", model_library.move_problem(
                self.src, os.path.join(self.tmp.name, "c"), size=50 * model_library.GB,
                free=10 * model_library.GB))
        os.makedirs(os.path.join(self.dest_root, "qwen36-35b"))
        self.assertIn("already exists", model_library.move_problem(self.src, self.dest_root))

    def test_setup_json_follows_the_configured_model(self):
        old = os.path.join(self.tmp.name, "D", "colibri-models", "q")
        new = os.path.join(self.tmp.name, "E", "colibri-models", "q")
        cfg = {"model_dir": old, "models_root": os.path.dirname(old),
               "args": ["web", "--model", old, "--port", "8000"]}
        moved = model_library.follow_move(cfg, old, new)
        self.assertEqual(moved["model_dir"], new)
        self.assertEqual(moved["models_root"], os.path.dirname(new))
        self.assertEqual(moved["args"], ["web", "--model", new, "--port", "8000"])
        self.assertIs(model_library.follow_move(cfg, os.path.join(self.tmp.name, "other"), new), cfg)


if __name__ == "__main__":
    unittest.main()
