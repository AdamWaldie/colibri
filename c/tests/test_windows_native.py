"""The configurator's Windows-only behaviour, on real Windows (skipped elsewhere):
a real .lnk through PowerShell, a real .cmd launcher run by cmd.exe with awkward
paths, the process-tree kill, and the window itself opened and closed under
pythonw.exe (no stdout, no stderr) without an error."""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

WINDOWS = sys.platform == "win32"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@unittest.skipUnless(WINDOWS, "native Windows behaviour")
class Shortcut(unittest.TestCase):
    def test_a_real_shortcut_opens_the_configurator(self):
        import win_shortcuts
        with tempfile.TemporaryDirectory() as folder:
            where = {"desktop": os.path.join(folder, "Desk top"), "programs": os.path.join(folder, "Start (menu)")}
            os.makedirs(where["desktop"])
            os.makedirs(where["programs"])
            made = win_shortcuts.create(where)
            self.assertEqual(win_shortcuts.status(where), {"desktop": True, "start_menu": True})
            read = win_shortcuts.powershell(
                "$s=(New-Object -ComObject WScript.Shell).CreateShortcut(" + win_shortcuts.ps_quote(made[0]) + ");"
                "[pscustomobject]@{t=$s.TargetPath;a=$s.Arguments;w=$s.WorkingDirectory} | ConvertTo-Json -Compress")
            link = json.loads(read.strip().splitlines()[-1])
            self.assertTrue(link["t"].lower().endswith(".exe"), link)
            self.assertIn("configurator.py", link["a"])
            self.assertEqual(os.path.normcase(link["w"]), os.path.normcase(win_shortcuts.HERE))
            self.assertEqual(len(win_shortcuts.remove(where)), 2)

    def test_the_real_folders(self):
        import win_shortcuts
        found = win_shortcuts.folders()
        self.assertTrue(os.path.isabs(found["desktop"]) and os.path.isabs(found["programs"]), found)


@unittest.skipUnless(WINDOWS, "native Windows behaviour")
class Launcher(unittest.TestCase):
    def test_the_cmd_launcher_passes_arguments_and_folder_through(self):
        import agent_connect
        with tempfile.TemporaryDirectory() as root:
            project = os.path.join(root, "My Project (100%) & co")
            os.makedirs(project)
            out = os.path.join(root, "args.txt")
            fake = os.path.join(root, "fake app.cmd")         # like claude.cmd / codex.cmd from npm
            Path(fake).write_text(f'@echo off\r\ncd > "{out}"\r\necho [%~1] >> "{out}"\r\necho [%~2] >> "{out}"\r\n'
                                  f'echo [%~3] >> "{out}"\r\necho [%COLIBRI_API_KEY%] >> "{out}"\r\n', encoding="utf-8")
            args = ["-c", "model_providers.colibri.base_url=http://127.0.0.1:8790/v1", "-m"]
            script = os.path.join(root, "launch.cmd")
            with open(script, "w", encoding="utf-8", newline="") as handle:
                handle.write(agent_connect.launcher_script("Fake", fake, args, project, "colibri: 50% & <ready>"))
            env = dict(os.environ, COLIBRI_API_KEY="k_1-2")
            subprocess.run(["cmd.exe", "/c", script], check=True, env=env, capture_output=True, timeout=60)
            lines = Path(out).read_text(encoding="utf-8", errors="replace").splitlines()
        self.assertEqual(os.path.normcase(lines[0].strip()), os.path.normcase(project))
        self.assertEqual([line.strip() for line in lines[1:]],
                         ["[-c]", "[model_providers.colibri.base_url=http://127.0.0.1:8790/v1]", "[-m]", "[k_1-2]"])


@unittest.skipUnless(WINDOWS, "native Windows behaviour")
class Processes(unittest.TestCase):
    def test_cancel_kills_the_whole_tree(self):
        import configurator
        child = ("import subprocess,sys,time;"
                 "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']);"
                 "print(p.pid,flush=True);time.sleep(120)")
        parent = subprocess.Popen([sys.executable, "-c", child], stdout=subprocess.PIPE, text=True,
                                  creationflags=NO_WINDOW)
        grandchild = int(parent.stdout.readline())
        subprocess.run(configurator.kill_tree_command(parent.pid), capture_output=True, timeout=60,
                       creationflags=NO_WINDOW)
        parent.wait(timeout=30)
        alive = subprocess.run(["tasklist", "/FI", f"PID eq {grandchild}", "/NH"], capture_output=True,
                               text=True, timeout=60).stdout
        self.assertNotIn(str(grandchild), alive)


@unittest.skipUnless(WINDOWS, "native Windows behaviour")
class Window(unittest.TestCase):
    def test_the_window_opens_every_tab_and_closes_under_pythonw(self):
        pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
        if not os.path.isfile(pythonw):
            self.skipTest("no pythonw.exe beside this python")
        with tempfile.TemporaryDirectory() as home:
            result = os.path.join(home, "result.json")
            script = os.path.join(home, "smoke.py")
            Path(script).write_text(textwrap.dedent(f"""
                import json, os, sys, traceback
                os.environ["COLI_SETUP_HOME"] = {home!r}
                sys.path.insert(0, {str(HERE.parent)!r})
                out = {{"errors": [], "tabs": [], "stdout_none": sys.stdout is None}}
                try:
                    import tkinter as tk
                    from tkinter import messagebox, ttk
                    for name in ("showerror", "showwarning", "showinfo"):
                        setattr(messagebox, name, lambda *a, _n=name, **k: out["errors"].append([_n, *map(str, a)]))
                    for name in ("askyesno", "askokcancel", "askyesnocancel"):
                        setattr(messagebox, name, lambda *a, **k: False)
                    real = tk.Tk.mainloop
                    def walk(widget):
                        yield widget
                        for child in widget.winfo_children():
                            yield from walk(child)
                    def mainloop(self, n=0):
                        def visit(step=0):
                            books = [w for w in walk(self) if isinstance(w, ttk.Notebook)]
                            tabs = books[0].tabs() if books else []
                            if step < len(tabs):
                                books[0].select(tabs[step])
                                out["tabs"].append(books[0].tab(tabs[step], "text").strip())
                                self.after(700, visit, step + 1)
                            else:
                                self.after(1500, self.destroy)
                        self.after(3000, visit)
                        real(self, n)
                    tk.Tk.mainloop = mainloop
                    import configurator
                    configurator.main()
                except BaseException as error:
                    if not (isinstance(error, SystemExit) and not error.code):
                        out["errors"].append(["crash", traceback.format_exc()])
                with open({result!r}, "w", encoding="utf-8") as handle:
                    json.dump(out, handle)
            """), encoding="utf-8")
            started = time.time()
            subprocess.run([pythonw, script], timeout=180)
            data = json.loads(Path(result).read_text(encoding="utf-8"))
            log = Path(home, "logs", "configurator.log")
            logged = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        self.assertTrue(data["stdout_none"], "not run console-less")
        self.assertEqual(data["errors"], [], logged)
        self.assertNotIn("Traceback", logged)
        self.assertGreaterEqual(len(data["tabs"]), 8, data["tabs"])
        self.assertLess(time.time() - started, 180)


if __name__ == "__main__":
    unittest.main()
