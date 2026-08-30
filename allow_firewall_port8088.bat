@echo off
echo ============================================================
echo 🛡️ Adding Windows Firewall Rule for Infinix ASR Server (Port 8088)
echo ============================================================

:: Check for administrative permissions
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [INFO] Running with Administrator privileges...
    netsh advfirewall firewall add rule name="Infinix ASR Server Port 8088" dir=in action=allow protocol=TCP localport=8088
    echo.
    echo ✅ Firewall Rule Added Successfully! Inbound TCP Traffic on Port 8088 is now ALLOWED.
) else (
    echo [INFO] Requesting Administrator Privileges...
    powershell -Command "Start-Process '%~0' -Verb RunAs"
    exit /b
)

echo.
echo ============================================================
echo 📡 Current Server IP Address on Wi-Fi:
echo.
ipconfig | findstr /i "IPv4 Address"
echo ============================================================
pause
