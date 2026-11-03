"""Getting a model from Ollama's library, Hugging Face or a link (model_fetch.py),
against fake servers: what the text names, the variants, the architecture read
from the first megabytes, where it goes, and the download checked by SHA-256."""
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import configurator  # noqa: E402
import model_fetch  # noqa: E402
from test_model_library import gguf_bytes  # noqa: E402


class Response(io.BytesIO):
    def __init__(self, data, headers=None):
        super().__init__(data)
        self.headers = headers or {}
        self.status = 206 if "Content-Range" in (headers or {}) else 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def getheader(self, name, default=None):
        return self.headers.get(name, default)

    def info(self):
        return self.headers


def server(routes):
    """An opener answering URL prefixes from `routes` ({prefix: bytes | dict | int})."""
    seen = []

    def opener(request, timeout=None):
        url = request.full_url if hasattr(request, "full_url") else request
        seen.append((url, dict(request.header_items()) if hasattr(request, "header_items") else {}))
        for prefix, answer in routes.items():
            if url.startswith(prefix):
                if isinstance(answer, int):
                    raise urllib.error.HTTPError(url, answer, "x", {}, None)
                if isinstance(answer, dict):
                    return Response(json.dumps(answer).encode())
                return Response(answer, {"Content-Range": f"bytes 0-{len(answer) - 1}/{len(answer)}",
                                         "Content-Length": str(len(answer))})
        raise urllib.error.HTTPError(url, 404, "not found", {}, None)

    opener.seen = seen
    return opener


class ParseQuery(unittest.TestCase):
    def test_ollama_names(self):
        self.assertEqual(model_fetch.parse_query("qwen3-coder:30b"),
                         {"source": "ollama", "namespace": "library", "model": "qwen3-coder", "tag": "30b"})
        self.assertEqual(model_fetch.parse_query("qwen3-coder")["tag"], "latest")
        self.assertEqual(model_fetch.parse_query("someone/mine:q4")["namespace"], "someone")
        self.assertEqual(model_fetch.parse_query("https://ollama.com/library/qwen3:0.6b")["model"], "qwen3")

    def test_hugging_face(self):
        q = model_fetch.parse_query("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF")
        self.assertEqual((q["source"], q["repo"], q["variant"]), ("hf", "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", ""))
        self.assertEqual(model_fetch.parse_query("hf.co/org/repo:Q4_K_M")["variant"], "Q4_K_M")
        link = model_fetch.parse_query("https://huggingface.co/org/repo/resolve/main/sub/m-Q8_0.gguf?download=true")
        self.assertEqual((link["repo"], link["file"]), ("org/repo", "sub/m-Q8_0.gguf"))
        self.assertEqual(model_fetch.parse_query("https://huggingface.co/org/repo")["repo"], "org/repo")

    def test_direct_link_and_refusals(self):
        self.assertEqual(model_fetch.parse_query("https://example.com/x/model.gguf")["source"], "url")
        for bad in ("", "  ", "a b", "https://example.com/page.html", "hf.co/onlyowner",
                    "https://huggingface.co/datasets/x/y", "x:y z"):
            with self.subTest(bad=bad), self.assertRaises(model_fetch.FetchError):
                model_fetch.parse_query(bad)


class Variants(unittest.TestCase):
    def test_quant_labels(self):
        cases = {"Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf": "Q4_K_M",
                 "Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf": "UD-Q4_K_XL",
                 "BF16/Qwen3-BF16-00001-of-00002.gguf": "BF16",
                 "model.IQ4_XS.gguf": "IQ4_XS", "plain.gguf": "plain"}
        for path, label in cases.items():
            self.assertEqual(model_fetch.quant_of(path), label, path)

    def test_preferred(self):
        variants = [{"label": lab, "size": i} for i, lab in enumerate(["Q2_K", "Q4_K_S", "Q4_K_M", "Q8_0"])]
        self.assertEqual(model_fetch.preferred(variants)["label"], "Q4_K_M")
        self.assertEqual(model_fetch.preferred([{"label": "Q2_K"}, {"label": "IQ4_NL"}])["label"], "IQ4_NL")
        self.assertEqual(model_fetch.preferred([{"label": "BF16"}])["label"], "BF16")

    def test_hugging_face_grouping(self):
        listing = [{"path": "m-Q4_K_M.gguf", "size": 10, "sha256": "a"},
                   {"path": "BF16/m-BF16-00001-of-00002.gguf", "size": 30, "sha256": "b"},
                   {"path": "BF16/m-BF16-00002-of-00002.gguf", "size": 30, "sha256": "c"},
                   {"path": "mmproj-F16.gguf", "size": 5, "sha256": "d"}]
        with mock.patch.object(model_fetch.setup_download, "list_repo_files", return_value=listing):
            variants = model_fetch.hf_variants({"repo": "o/r", "revision": "main"})
        self.assertEqual([(v["label"], v["size"], len(v["files"])) for v in variants],
                         [("Q4_K_M", 10, 1), ("BF16", 60, 2)])
        self.assertTrue(variants[0]["files"][0]["url"].endswith("/o/r/resolve/main/m-Q4_K_M.gguf"))

    def test_ollama_manifest(self):
        digest = "sha256:" + "ab" * 32
        opener = server({"https://registry.ollama.ai/v2/library/qwen3/manifests/0.6b": {"layers": [
            {"mediaType": "application/vnd.ollama.image.license", "digest": "sha256:" + "00" * 32, "size": 1},
            {"mediaType": "application/vnd.ollama.image.model", "digest": digest, "size": 522640096}]}})
        variants = model_fetch.ollama_variants(model_fetch.parse_query("qwen3:0.6b"), opener)
        spec = variants[0]["files"][0]
        self.assertEqual((spec["size"], spec["sha256"]), (522640096, "ab" * 32))
        self.assertEqual(spec["url"], f"https://registry.ollama.ai/v2/library/qwen3/blobs/{digest}")
        headers = {k.lower(): v for k, v in opener.seen[0][1].items()}
        self.assertEqual(headers.get("accept"), model_fetch.MANIFEST_ACCEPT)

    def test_missing_model_and_gated_repo(self):
        with self.assertRaisesRegex(model_fetch.FetchError, "not found"):
            model_fetch.ollama_variants(model_fetch.parse_query("nope:1"), server({}))
        with self.assertRaisesRegex(model_fetch.FetchError, "token"):
            model_fetch.ollama_variants(model_fetch.parse_query("x:1"), server({"https://registry": 401}))


class ArchitectureCheck(unittest.TestCase):
    def test_supported_and_refused_from_the_header(self):
        ok = server({"https://h/ok.gguf": gguf_bytes("qwen3moe", "Qwen3 Coder") + b"\0" * 64})
        result = model_fetch.check({"files": [{"url": "https://h/ok.gguf"}]}, opener=ok)
        self.assertTrue(result["supported"])
        self.assertIn("Qwen3 Coder", result["why"])
        headers = {k.lower(): v for k, v in ok.seen[0][1].items()}
        self.assertTrue(headers.get("range", "").startswith("bytes=0-"))     # a partial read only
        no = server({"https://h/no.gguf": gguf_bytes("llama")})
        self.assertFalse(model_fetch.check({"files": [{"url": "https://h/no.gguf"}]}, opener=no)["supported"])
        junk = server({"https://h/j.gguf": b"<html>"})
        self.assertIn("not a GGUF", model_fetch.check({"files": [{"url": "https://h/j.gguf"}]}, opener=junk)["why"])

    def test_resolve_picks_q4_and_checks_it(self):
        listing = [{"path": f"m-{q}.gguf", "size": s, "sha256": q} for q, s in (("Q2_K", 5), ("Q4_K_M", 9))]
        opener = server({"https://huggingface.co/o/r/resolve/main/m-Q4_K_M.gguf": gguf_bytes("qwen3moe")})
        with mock.patch.object(model_fetch.setup_download, "list_repo_files", return_value=listing):
            plan = model_fetch.resolve("o/r", opener=opener)
            self.assertEqual(plan["chosen"]["label"], "Q4_K_M")
            self.assertTrue(plan["supported"])
            with self.assertRaisesRegex(model_fetch.FetchError, "no Q9 variant"):
                model_fetch.resolve("o/r", opener=opener, variant="Q9")


class Places(unittest.TestCase):
    def test_folders(self):
        plan = {"name": "qwen3-coder:30b", "chosen": {"label": "30b"}}
        self.assertEqual(model_fetch.download_dir(plan["name"], "/m"),
                         os.path.join("/m", "gguf-downloads", "qwen3-coder-30b"))
        self.assertEqual(model_fetch.import_dir(plan["name"], plan["chosen"], "/m"),
                         os.path.join("/m", "qwen3-coder-30b-gguf"))
        self.assertEqual(model_fetch.import_dir("o/Model-GGUF", {"label": "Q4_K_M"}, "/m"),
                         os.path.join("/m", "o-Model-GGUF-Q4_K_M-gguf"))

    def test_fetch_command(self):
        self.assertEqual(model_fetch.fetch_command("o/r", "Q4_K_M", "/m", python="py")[2:],
                         ["o/r", "--root", "/m", "--variant", "Q4_K_M"])


class Download(unittest.TestCase):
    def test_download_checks_the_sha256(self):
        data = b"G" * 4096
        good = {"name": "t:1", "chosen": {"label": "1", "size": len(data), "files": [
            {"path": "t.gguf", "size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "url": "https://h/t"}]}}
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, {"COLI_SETUP_HOME": root}):
            first = model_fetch.download(good, root, out=lambda *_: None, opener=server({"https://h/t": data}))
            with open(first, "rb") as handle:
                self.assertEqual(handle.read(), data)
            bad = json.loads(json.dumps(good))
            bad["name"] = "t:2"
            bad["chosen"]["files"][0]["sha256"] = "0" * 64
            with self.assertRaises(model_fetch.setup_download.DownloadError):
                model_fetch.download(bad, root, out=lambda *_: None, opener=server({"https://h/t": data}))

    def test_not_enough_space(self):
        plan = {"name": "big:1", "chosen": {"label": "1", "size": 10 ** 12, "files": [
            {"path": "b.gguf", "size": 10 ** 12, "url": "https://h/b"}]}}
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.object(model_fetch.model_library.setup_hw, "disk_free", return_value=10 ** 9), \
                self.assertRaisesRegex(model_fetch.FetchError, "not enough space"):
            model_fetch.download(plan, root, out=lambda *_: None, opener=server({}))


class Summary(unittest.TestCase):
    def test_fetch_summary(self):
        plan = {"name": "qwen3:0.6b", "supported": True, "why": "qwen3: colibri runs it",
                "chosen": {"label": "0.6b", "size": 522640096}}
        with mock.patch.object(configurator.setup_hw, "disk_free", return_value=10 ** 12):
            text, ok = configurator.fetch_summary(plan, "/m")
        self.assertTrue(ok)
        self.assertIn("qwen3-0.6b-gguf", text)
        with mock.patch.object(configurator.setup_hw, "disk_free", return_value=10 ** 6):
            self.assertEqual(configurator.fetch_summary(plan, "/m")[1], False)
        text, ok = configurator.fetch_summary(dict(plan, supported=False, why="llama is not run"), "/m")
        self.assertFalse(ok)
        self.assertIn("can't run it", text)


if __name__ == "__main__":
    unittest.main()
