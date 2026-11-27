@echo off
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" (
    where python >nul 2>&1
    if errorlevel 1 (
        echo Python 3.10 or later must be installed and available on PATH.
        exit /b 1
    )
    python -c "import sys; assert sys.version_info >= (3,10), 'Python 3.10 or later required'"
    if errorlevel 1 exit /b 1
    python -m venv .venv
    if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -m pip install ".[images]"
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m deepforge --version
if errorlevel 1 exit /b 1
echo Setup complete. Run scripts\launch-windows.cmd to open Studio.
echo PyTorch is optional and must be installed separately for Model Inspector.
