@echo off
chcp 65001 > NUL
title INFINIX ASR SERVER - REAL-TIME LIVE DASHBOARD (PORT 8088)

echo =============================================================================
echo 🟢 LAUNCHING INFINIX LAPTOP ASR LIVE SERVER DASHBOARD...
echo =============================================================================
echo.
python live_server_console.py
if errorlevel 1 (
    echo.
    echo Retrying with Python3...
    python3 live_server_console.py
)
pause
