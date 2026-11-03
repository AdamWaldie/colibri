#!/usr/bin/env python3
"""A "colibri" shortcut on the Windows desktop and in the Start menu, opening the
configurator (the window CONFIGURE.bat opens) with the colibri icon, and nothing
else: no terminal, no console window.

The shortcut runs pythonw.exe (the Python that runs this file, its console-less
twin) on configurator.py, in the colibri folder. Windows' own shortcut object
(WScript.Shell through PowerShell) writes the .lnk; this module only builds the
script and reads back where the desktop and the Start menu are, which Windows
decides (a desktop redirected to OneDrive included). Per user: nothing needs
administrator rights.

    folders()   {"desktop": ..., "programs": ...} for this user
    status()    {"desktop": bool, "start_menu": bool}
    create()    both shortcuts, written or rewritten (a moved colibri folder fixed)
    remove()    both, when they exist
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
NAME = "colibri"
LNK = NAME + ".lnk"
ICON = os.path.join(HERE, "colibri.ico")
DESCRIPTION = "colibri: set up, run and connect the local model"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def windowless_python(executable=None):
    """pythonw.exe beside the running python.exe: the shortcut opens a window, never a console."""
    exe = executable or sys.executable or "python"
    folder, name = os.path.split(exe)
    if name.lower() == "python.exe":
        candidate = os.path.join(folder, "pythonw.exe")
        if os.path.isfile(candidate):
            return candidate
    return exe


def target(executable=None, here=HERE):
    """(program, arguments, working folder) the shortcut runs."""
    return windowless_python(executable), f'"{os.path.join(here, "configurator.py")}"', here


def ps_quote(text):
    """A PowerShell single-quoted string: nothing inside is expanded."""
    return "'" + str(text).replace("'", "''") + "'"


FOLDERS_PS = ("[pscustomobject]@{desktop=[Environment]::GetFolderPath('Desktop');"
              "programs=[Environment]::GetFolderPath('Programs')} | ConvertTo-Json -Compress")


def create_script(paths, program, arguments, workdir, icon=None, description=DESCRIPTION):
    """PowerShell that writes one .lnk per path through WScript.Shell."""
    lines = ["$ErrorActionPreference = 'Stop'", "$shell = New-Object -ComObject WScript.Shell"]
    for path in paths:
        lines += [f"$s = $shell.CreateShortcut({ps_quote(path)})",
                  f"$s.TargetPath = {ps_quote(program)}",
                  f"$s.Arguments = {ps_quote(arguments)}",
                  f"$s.WorkingDirectory = {ps_quote(workdir)}",
                  f"$s.Description = {ps_quote(description)}",
                  "$s.WindowStyle = 1"]
        if icon:
            lines.append(f"$s.IconLocation = {ps_quote(icon + ',0')}")
        lines.append("$s.Save()")
    return "\n".join(lines)


def powershell(script, timeout=60):
    """Run a PowerShell script without a window; its standard output."""
    # PowerShell writes a pipe in the OEM code page; a desktop under "C:\Users\José" or
    # "Área de Trabalho" would come back garbled. Ask it for UTF-8 and read UTF-8.
    script = "[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false\n" + script
    done = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                           "-Command", script], capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace",
                          creationflags=NO_WINDOW, stdin=subprocess.DEVNULL)
    if done.returncode:
        raise OSError((done.stderr or done.stdout or "PowerShell failed").strip().splitlines()[-1])
    return done.stdout


def folders(run=powershell):
    if sys.platform != "win32" and run is powershell:
        raise OSError("shortcuts are made on Windows")
    found = json.loads(run(FOLDERS_PS).strip().splitlines()[-1])
    return {"desktop": found["desktop"], "programs": found["programs"]}


def paths(where):
    return {"desktop": os.path.join(where["desktop"], LNK), "start_menu": os.path.join(where["programs"], LNK)}


def status(where=None, run=powershell):
    where = where or folders(run)
    return {key: os.path.isfile(path) for key, path in paths(where).items()}


def create(where=None, run=powershell, executable=None):
    """Both shortcuts; returns their paths."""
    where = where or folders(run)
    program, arguments, workdir = target(executable)
    places = list(paths(where).values())
    run(create_script(places, program, arguments, workdir, ICON if os.path.isfile(ICON) else None))
    return places


def remove(where=None, run=powershell):
    """Both shortcuts, the ones that exist; returns what was removed."""
    where = where or folders(run)
    gone = []
    for path in paths(where).values():
        if os.path.isfile(path):
            os.remove(path)
            gone.append(path)
    return gone


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "create"
    if action == "remove":
        print("\n".join(remove()) or "no shortcut to remove")
    else:
        print("\n".join(create()))
