@echo off
rem colibri configurator: a window to choose the model, where models are
rem downloaded, the backend (CUDA, Vulkan or CPU), the address and port, and to
rem install, start and stop colibri. Double-click this file.
rem
rem It drives the same `coli setup` that START-HERE.bat runs and shows each
rem command before it runs, so everything it does can also be typed in a terminal.
setlocal
title colibri configurator
cd /d "%~dp0"

if not exist "%~dp0c\configurator.py" (
    echo This file belongs at the top of the colibri folder, next to the c folder.
    goto :hold_fail
)

rem The same Python search as START-HERE.bat: the py launcher first, then
rem python on the PATH (with a version check, the Store alias prints nothing),
rem then a per-user install.
set "COLI_PY_EXE="
set "COLI_PY_ARG="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1 && (set "COLI_PY_EXE=py" & set "COLI_PY_ARG=-3")
if not defined COLI_PY_EXE python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1 && set "COLI_PY_EXE=python"
if not defined COLI_PY_EXE call :find_user_python
if not defined COLI_PY_EXE goto :no_python

rem The window needs Tk, which the python.org installer includes ("tcl/tk and IDLE").
"%COLI_PY_EXE%" %COLI_PY_ARG% -c "import tkinter" >nul 2>&1
if errorlevel 1 goto :no_tk

rem pythonw.exe runs the window without keeping this console open: pyw for the
rem py launcher, pythonw next to python.exe otherwise.
set "COLI_PYW="
if /i "%COLI_PY_EXE%"=="py" (
    where pyw >nul 2>&1 && set "COLI_PYW=pyw"
) else if /i "%COLI_PY_EXE%"=="python" (
    where pythonw >nul 2>&1 && set "COLI_PYW=pythonw"
) else (
    if exist "%COLI_PY_EXE:python.exe=pythonw.exe%" set "COLI_PYW=%COLI_PY_EXE:python.exe=pythonw.exe%"
)
if defined COLI_PYW (
    start "" "%COLI_PYW%" %COLI_PY_ARG% "%~dp0c\configurator.py" %*
    exit /b 0
)
"%COLI_PY_EXE%" %COLI_PY_ARG% "%~dp0c\configurator.py" %*
if errorlevel 1 goto :hold_fail
exit /b 0

:find_user_python
for %%V in (314 313 312 311 310) do (
    if not defined COLI_PY_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" set "COLI_PY_EXE=%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
)
exit /b 0

:no_tk
echo This Python has no Tk (the tkinter module), which the configurator window needs.
echo Run the python.org installer again, choose Modify and tick "tcl/tk and IDLE".
echo START-HERE.bat does the same setup in this console without a window.
goto :hold_fail

:no_python
echo colibri needs Python 3.10 or newer, and it was not found.
echo Double-click START-HERE.bat: it offers to install Python with winget.
echo Or install it from https://www.python.org/downloads/ and tick
echo "Add python.exe to PATH" in the installer, then double-click CONFIGURE.bat again.

:hold_fail
echo.
pause
exit /b 1
