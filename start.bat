@echo off
setlocal
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
    echo Missing .venv. Follow the installation steps in README.md.
    exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
exit /b %errorlevel%
