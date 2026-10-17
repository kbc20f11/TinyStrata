@echo off
setlocal
cd /d "%~dp0"
for %%I in ("%~dp0.") do set "TINYSTRATA_CHECKOUT_NAME=%%~nxI"
set "TINYSTRATA_PYTHON=%~dp0..\%TINYSTRATA_CHECKOUT_NAME%-work\runtime\.venv\Scripts\python.exe"
if exist "%TINYSTRATA_PYTHON%" (
  "%TINYSTRATA_PYTHON%" -B run.py %*
) else if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -B run.py %*
) else (
  python -B run.py %*
)
if errorlevel 1 pause
