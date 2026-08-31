@echo off
echo ============================================================
echo 🛡️ Unlocking Windows Firewall for Port 8088 & Ping (ALL PROFILES)
echo ============================================================

:: Check for administrative permissions
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [INFO] Running with Administrator privileges...
    
    :: Delete previous rules if exist
    netsh advfirewall firewall delete rule name="Infinix ASR Server Port 8088" >nul 2>&1
    netsh advfirewall firewall delete rule name="Allow ICMPv4 Inbound Ping" >nul 2>&1

    :: Add TCP Port 8088 inbound rule for ALL profiles (Public, Private, Domain)
    netsh advfirewall firewall add rule name="Infinix ASR Server Port 8088" dir=in action=allow protocol=TCP localport=8088 profile=any
    
    :: Add ICMPv4 (Ping) inbound rule for ALL profiles
    netsh advfirewall firewall add rule name="Allow ICMPv4 Inbound Ping" dir=in action=allow protocol=ICMPv4:8,any profile=any

    echo.
    echo ✅ SUCCESS! Port 8088 & Inbound Ping are NOW UNLOCKED on ALL Networks!
) else (
    echo [INFO] Requesting Administrator Privileges...
    powershell -Command "Start-Process '%~0' -Verb RunAs"
    exit /b
)

echo.
echo ============================================================
echo 📡 Current Server IP Addresses (Use Hotspot IP if Wi-Fi blocks):
echo.
ipconfig | findstr /i "IPv4 Address"
echo ============================================================
pause
