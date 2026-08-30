@echo off
echo ============================================================
echo 🚀 Syncing Git Changes for Himalayandev and Namanbhatt
echo ============================================================

set PATH=%~dp0..\git_portable\cmd;%~dp0..\git_portable\mingw64\bin;%PATH%

cd /d "%~dp0"

echo.
echo 📥 Pulling latest from Himalayandev (origin main)...
git pull origin main --rebase

echo.
echo 📤 Pushing final changes to Himalayandev (origin main)...
git push origin main

echo.
echo 📥 Pulling latest from Namanbhatt (namanbhatt main)...
git pull namanbhatt main --rebase

echo.
echo 📤 Pushing final changes to Namanbhatt (namanbhatt main)...
git push namanbhatt main

echo.
echo ✅ Sync complete!
pause
