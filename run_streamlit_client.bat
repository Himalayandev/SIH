@echo off
chcp 65001 > NUL
title "KWS STREAMLIT EDGE CLIENT DASHBOARD"

echo =============================================================================
echo 🟢 LAUNCHING KWS STREAMLIT EDGE CLIENT DASHBOARD...
echo =============================================================================
echo.

cd /d "%~dp0"

if exist "..\venv\Scripts\python.exe" (
    ..\venv\Scripts\python.exe -m streamlit run streamlit_kws_client.py
) else if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe -m streamlit run streamlit_kws_client.py
) else (
    streamlit run streamlit_kws_client.py
)

if errorlevel 1 (
    echo.
    echo Retrying with Python3...
    python3 -m streamlit run streamlit_kws_client.py
)
pause
