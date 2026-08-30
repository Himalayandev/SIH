@echo off
chcp 65001 > NUL
title INFINIX ASR SERVER - LIVE STREAMING LOG MONITOR

echo =============================================================================
echo 🟢 LIVE STREAMING LOG MONITOR (Press Ctrl+C to Exit)
echo =============================================================================
echo.
powershell -Command "Get-Content -Path 'server_live.log' -Wait -Tail 20"
pause
