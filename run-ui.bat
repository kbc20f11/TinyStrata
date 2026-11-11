@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0ui\start.ps1" (
  echo TinyStrata UI helper is missing. Update TinyStrata with the UI installer.
  pause
  exit /b 1
)
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%~dp0ui\start.ps1" %*
set "TINYSTRATA_UI_EXIT=%ERRORLEVEL%"
if not "%TINYSTRATA_UI_EXIT%"=="0" pause
exit /b %TINYSTRATA_UI_EXIT%
