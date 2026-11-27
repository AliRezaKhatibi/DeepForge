@echo off
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
    echo Run scripts\setup-windows.cmd first.
    exit /b 1
)
".venv\Scripts\python.exe" -m deepforge.gui
exit /b %errorlevel%
