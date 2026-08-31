@echo off
chcp 65001 > NUL
title "NEW LOW-LATENCY ASR STT SERVER AND DASHBOARD (PORT 8088)"

echo =============================================================================
echo 🟢 LAUNCHING NEW LOW-LATENCY ASR STT SERVER WITH GUI...
echo =============================================================================
echo.

cd /d "%~dp0"

if exist "..\venv\Scripts\python.exe" (
    ..\venv\Scripts\python.exe new_server.py
) else if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe new_server.py
) else (
    python new_server.py
)

if errorlevel 1 (
    echo.
    echo Retrying with Python3...
    python3 new_server.py
)
pause
