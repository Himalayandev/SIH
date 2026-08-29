@echo off
chcp 65001 > NUL
title Infinix Laptop ASR Server (Port 8088)

echo =============================================================================
echo 🟢 STARTING INFINIX LAPTOP ASR SERVER (TCP PORT 8088)
echo =============================================================================
echo.
echo [1/2] Detecting Local Network IP Addresses for ESP32 Firmware Configuration:
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
    echo    - ESP32 Target IP: %%a
)
echo.
echo [2/2] Launching Python Faster-Whisper ASR Engine...
echo =============================================================================
python server_faster_whisper.py
if errorlevel 1 (
    echo.
    echo ⚠️ Server exited with error. Retrying with python3...
    python3 server_faster_whisper.py
)
pause
