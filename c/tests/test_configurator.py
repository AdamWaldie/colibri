"""The configurator window's logic, without a window.

The window itself only collects choices; what it runs is decided by the pure
functions tested here: the `coli setup` command for the choices, the environment
the child gets, the command shown to the user (token hidden), and the summary of
setup.json. The command is checked against the real argument parser of `coli
setup`, so a flag renamed there fails here instead of in the window.
"""
import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import configurator  # noqa: E402
import setup_catalog  # noqa: E402

GB = 10**9


def coli_setup_parser():
    """The `setup` subparser exactly as `coli` builds it (copied from its source,
    not re-declared here)."""
    source = (Path(configurator.HERE) / "coli").read_text(encoding="utf-8")
    start = source.index('    pset=sub.add_parser("setup"')
    end = source.index('    pstart=sub.add_parser("start"')
    ap = argparse.ArgumentParser(prog="coli")
    sub = ap.add_subparsers(dest="cmd")
    exec(compile("\n".join(line[4:] for line in source[start:end].splitlines()),
                 "coli-setup-parser", "exec"), {"sub": sub, "os": os})
    return ap


class SetupCommand(unittest.TestCase):
    def parse(self, cmd):
        self.assertEqual(cmd[2], "setup")
        return coli_setup_parser().parse_args(cmd[2:])

    def test_defaults_are_left_to_coli(self):
        cmd = configurator.setup_command({"model": "qwen36-35b"}, python="py", coli="coli")
        self.assertEqual(cmd, ["py", "coli", "setup", "--yes", "--model", "qwen36-35b", "--background"])

    def test_every_choice_maps_to_a_flag_coli_accepts(self):
        cmd = configurator.setup_command({
            "model": "glm-5.2", "models_root": r"D:\colibri-models", "backend": "cuda",
            "host": "0.0.0.0", "port": "8080", "ctx": "32768", "start": "no",
            "no_browser": True, "no_verify": True, "reconfigure": True}, python="py", coli="coli")
        a = self.parse(cmd)
        self.assertTrue(a.yes)
        self.assertEqual(a.pick, "glm-5.2")
        self.assertEqual(a.dir, r"D:\colibri-models")
        self.assertEqual(a.backend, "cuda")
        self.assertEqual((a.host, a.port, a.ctx), ("0.0.0.0", 8080, 32768))
        self.assertTrue(a.no_start and a.no_browser and a.no_verify and a.reconfigure)
        self.assertFalse(a.background)

    def test_a_folder_on_disk_wins_over_the_catalog_and_the_download_folder(self):
        cmd = configurator.setup_command({"model": "qwen36-35b", "model_dir": r"E:\models\mine",
                                          "models_root": r"D:\colibri-models"}, python="py", coli="coli")
        a = self.parse(cmd)
        self.assertEqual(a.model_dir, r"E:\models\mine")
        self.assertIsNone(a.pick)
        self.assertIsNone(a.dir)

    def test_default_host_port_and_automatic_context_add_nothing(self):
        cmd = configurator.setup_command({"model": "x", "host": "127.0.0.1", "port": "8000", "ctx": "0"},
                                         python="py", coli="coli")
        for flag in ("--host", "--port", "--ctx"):
            self.assertNotIn(flag, cmd)

    def test_bad_values_are_refused_before_anything_runs(self):
        for options in ({"port": "eighty"}, {"port": "70000"}, {"port": "0"}, {"ctx": "-5"},
                        {"ctx": "lots"}, {"backend": "metal"}, {"start": "foreground"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                configurator.setup_command(dict(options, model="x"), python="py", coli="coli")

    def test_every_catalog_model_makes_a_valid_command(self):
        for entry in setup_catalog.catalog():
            with self.subTest(model=entry.id):
                a = self.parse(configurator.setup_command({"model": entry.id}, python="py", coli="coli"))
                self.assertEqual(a.pick, entry.id)


class Environment(unittest.TestCase):
    def test_child_output_is_unbuffered_utf8_without_colour(self):
        env = configurator.child_env(base={"PATH": "x"})
        self.assertEqual(env["PYTHONUNBUFFERED"], "1")
        self.assertEqual(env["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(env["COLI_COLOR"], "0")
        self.assertEqual(env["PATH"], "x")

    def test_token_and_split_folders_only_when_given(self):
        self.assertEqual(configurator.extra_env("", "SPLIT", ""), {})
        self.assertEqual(configurator.extra_env("  hf_abc ", "", "D:\\a"), {"HF_TOKEN": "hf_abc"})
        self.assertEqual(configurator.extra_env("", "SPLIT", r"D:\a;E:\b"), {"SPLIT": r"D:\a;E:\b"})

    def test_the_window_never_changes_its_own_environment(self):
        before = dict(os.environ)
        configurator.child_env({"HF_TOKEN": "secret"})
        self.assertEqual(dict(os.environ), before)


class DisplayedCommand(unittest.TestCase):
    def test_shown_as_coli_with_the_token_hidden_on_windows(self):
        cmd = [r"C:\Python312\python.exe", os.path.join("x", "coli"), "setup", "--dir", r"D:\my models"]
        text = configurator.display_command(cmd, {"HF_TOKEN": "hf_secret", "SPLIT": r"E:\b"}, windows=True)
        self.assertNotIn("hf_secret", text)
        self.assertIn('set "HF_TOKEN=<hidden>" && ', text)
        self.assertTrue(text.endswith('coli setup --dir "D:\\my models"'))

    def test_shell_quoting_elsewhere(self):
        text = configurator.display_command(["python3", "/x/coli", "setup", "--dir", "/a b"], windows=False)
        self.assertEqual(text, "coli setup --dir '/a b'")


class PythonForChildren(unittest.TestCase):
    def test_pythonw_is_swapped_for_python_next_to_it(self):
        with tempfile.TemporaryDirectory() as folder:
            pyw = os.path.join(folder, "pythonw.exe")
            py = os.path.join(folder, "python.exe")
            Path(pyw).touch()
            self.assertEqual(configurator.python_for_children(pyw), pyw)  # no python.exe there
            Path(py).touch()
            self.assertEqual(configurator.python_for_children(pyw), py)

    def test_python_is_kept(self):
        self.assertEqual(configurator.python_for_children("/usr/bin/python3"), "/usr/bin/python3")


class CatalogAndSpace(unittest.TestCase):
    def test_rows_are_the_catalog_assessed_for_the_machine(self):
        rows = configurator.catalog_rows(32 * GB, 2000 * GB, os_name="win32", machine="amd64")
        self.assertEqual({r["id"] for r in rows}, {e.id for e in setup_catalog.catalog()})
        self.assertEqual(sum(r["recommended"] for r in rows), 1)

    def test_space_note(self):
        entry = setup_catalog.by_id("qwen36-35b")
        self.assertIn("fits", configurator.space_note(entry, 100 * GB))
        self.assertIn("NOT ENOUGH", configurator.space_note(entry, 10 * GB))
        self.assertEqual(configurator.space_note(entry, None), "")
        self.assertEqual(configurator.space_note(None, 10 * GB), "")

    def test_partial_downloads_are_counted(self):
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(configurator.setup_download, "read_manifest",
                                   side_effect=lambda f: {"files": []} if f.endswith("qwen36-35b") else None), \
                    mock.patch.object(configurator.setup_download, "bytes_present", return_value=5 * GB):
                self.assertEqual(configurator.downloaded_bytes(root), {"qwen36-35b": 5 * GB})


class ConfigSummary(unittest.TestCase):
    def test_nothing_saved(self):
        self.assertEqual(configurator.config_summary(None)[0][0], "status")

    def test_saved_configuration(self):
        cfg = {"status": "ready", "model": {"id": "qwen36-35b", "name": "Qwen3.6-35B-A3B", "repo": "r/q"},
               "model_dir": r"D:\colibri-models\qwen36-35b", "backend": "vulkan", "requested_backend": "cuda",
               "engine": r"C:\colibri\c\qwen36.exe", "host": "127.0.0.1", "port": 8000, "ctx": 0,
               "env": {"COLI_VULKAN": "1"}, "failed_builds": ["cuda"], "updated": "2026-11-02T10:00:00"}
        lines = dict(configurator.config_summary(cfg))
        self.assertEqual(lines["model"], "Qwen3.6-35B-A3B  (qwen36-35b)")
        self.assertEqual(lines["backend"], "vulkan  (asked for: cuda)")
        self.assertEqual(lines["context"], "automatic")
        self.assertEqual(lines["env COLI_VULKAN"], "1")
        self.assertEqual(lines["failed builds"], "cuda")

    def test_local_folder(self):
        lines = dict(configurator.config_summary({"model": {"name": "GLM"}, "model_dir": "/m"}))
        self.assertEqual(lines["repository"], "(a folder on disk)")


if __name__ == "__main__":
    unittest.main()
