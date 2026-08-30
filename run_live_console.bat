@echo off
title SIH ASR Live Server & Security Authorization Console (Port 8088)
color 0B
echo =========================================================================
echo 🚀 STARTING SIH LOW-LATENCY ASR SERVER & SECURITY CONSOLE...
echo =========================================================================
cd /d "%~dp0"
venv\Scripts\python.exe infinix_laptop_server\live_server_console.py
pause
