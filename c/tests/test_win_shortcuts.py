"""The colibri shortcut (win_shortcuts.py) and the window reaching release archives.

Windows is faked: PowerShell is a function that records the script it is given
and answers the folders a real one would; the script itself is checked line by
line (WScript.Shell, single-quoted paths, pythonw, the icon). The release side:
pack_python's walk from `coli` must reach the configurator, the shortcut module
and the icon, or a release archive has no window to open."""
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import win_shortcuts  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


class Target(unittest.TestCase):
    def test_pythonw_beside_python(self):
        with tempfile.TemporaryDirectory() as d:
            python = os.path.join(d, "python.exe")
            pathlib.Path(python).touch()
            self.assertEqual(win_shortcuts.windowless_python(python), python)        # no pythonw there
            pathlib.Path(d, "pythonw.exe").touch()
            self.assertEqual(win_shortcuts.windowless_python(python), os.path.join(d, "pythonw.exe"))

    def test_the_shortcut_opens_the_configurator_in_the_colibri_folder(self):
        program, arguments, workdir = win_shortcuts.target("C:/Py/pythonw.exe", here="C:/colibri/c")
        self.assertEqual(program, "C:/Py/pythonw.exe")
        self.assertEqual(arguments, '"' + os.path.join("C:/colibri/c", "configurator.py") + '"')
        self.assertEqual(workdir, "C:/colibri/c")


class Script(unittest.TestCase):
    def test_quoting_keeps_paths_literal(self):
        self.assertEqual(win_shortcuts.ps_quote("C:\\Users\\O'Brien\\$env:x"), "'C:\\Users\\O''Brien\\$env:x'")

    def test_create_script(self):
        script = win_shortcuts.create_script(["C:\\D\\colibri.lnk", "C:\\S\\colibri.lnk"], "C:\\Py\\pythonw.exe",
                                             '"C:\\c\\configurator.py"', "C:\\c", icon="C:\\c\\colibri.ico")
        self.assertIn("New-Object -ComObject WScript.Shell", script)
        self.assertEqual(script.count("$s.Save()"), 2)
        self.assertIn("$s.TargetPath = 'C:\\Py\\pythonw.exe'", script)
        self.assertIn("$s.Arguments = '\"C:\\c\\configurator.py\"'", script)
        self.assertIn("$s.IconLocation = 'C:\\c\\colibri.ico,0'", script)
        self.assertIn("$ErrorActionPreference = 'Stop'", script)
        self.assertNotIn("IconLocation", win_shortcuts.create_script(["x.lnk"], "p", "a", "w"))


class FakeWindows(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.where = {"desktop": os.path.join(self.tmp.name, "Desktop"),
                      "programs": os.path.join(self.tmp.name, "Programs")}
        for folder in self.where.values():
            os.makedirs(folder)
        self.scripts = []

    def tearDown(self):
        self.tmp.cleanup()

    def run_ps(self, script):
        self.scripts.append(script)
        if "GetFolderPath" in script:
            return json.dumps(self.where) + "\n"
        return ""

    def test_folders_come_from_windows(self):
        self.assertEqual(win_shortcuts.folders(run=self.run_ps), self.where)

    def test_create_status_remove(self):
        self.assertEqual(win_shortcuts.status(run=self.run_ps), {"desktop": False, "start_menu": False})
        made = win_shortcuts.create(run=self.run_ps, executable="C:/Py/python.exe")
        self.assertEqual(made, [os.path.join(self.where["desktop"], "colibri.lnk"),
                                os.path.join(self.where["programs"], "colibri.lnk")])
        self.assertIn("CreateShortcut", self.scripts[-1])
        self.assertIn("colibri.ico,0", self.scripts[-1])          # the icon ships next to the module
        for path in made:                                            # what WScript.Shell would have written
            pathlib.Path(path).touch()
        self.assertEqual(win_shortcuts.status(run=self.run_ps), {"desktop": True, "start_menu": True})
        self.assertEqual(sorted(win_shortcuts.remove(run=self.run_ps)), sorted(made))
        self.assertEqual(win_shortcuts.remove(run=self.run_ps), [])

    def test_off_windows_it_says_so(self):
        if sys.platform == "win32":
            self.skipTest("this is Windows")
        with self.assertRaises(OSError):
            win_shortcuts.folders()


class ReleaseArchive(unittest.TestCase):
    def test_the_window_its_modules_and_the_icon_are_packed(self):
        spec = importlib.util.spec_from_file_location("pack_python_for_shortcuts",
                                                      HERE.parent / "tools" / "pack_python.py")
        pack = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pack)
        names = {pathlib.Path(p).name for p in pack.needed(HERE.parent)}
        for name in ("configurator.py", "win_shortcuts.py", "colibri.ico", "agent_connect.py",
                     "model_library.py", "model_fetch.py", "documents.py"):
            self.assertIn(name, names)


if __name__ == "__main__":
    unittest.main()
