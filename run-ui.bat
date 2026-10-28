@echo off
setlocal
cd /d "%~dp0"
for %%I in ("%~dp0.") do set "TINYSTRATA_CHECKOUT_NAME=%%~nxI"
set "TINYSTRATA_PYTHON=%~dp0..\%TINYSTRATA_CHECKOUT_NAME%-work\runtime\.venv\Scripts\python.exe"
set "TINYSTRATA_UI_SOURCE=%~dp0"
for /f "usebackq delims=" %%P in (`powershell.exe -NoLogo -NoProfile -Command "$ErrorActionPreference = 'Stop'; $sourceRoot = (Get-Item -LiteralPath $env:TINYSTRATA_UI_SOURCE).FullName; $workRoot = Join-Path (Split-Path -Parent $sourceRoot) ((Split-Path -Leaf $sourceRoot) + '-work'); $locationPath = Join-Path $workRoot 'runtime\ui\location.json'; if (Test-Path -LiteralPath $locationPath) { try { $location = ConvertFrom-Json -InputObject (Get-Content -LiteralPath $locationPath -Raw); if ([IO.Path]::GetFullPath($location.source_dir) -eq $sourceRoot) { $workRoot = [IO.Path]::GetFullPath($location.work_dir) } } catch { } }; Join-Path $workRoot 'runtime\.venv\Scripts\python.exe'"`) do set "TINYSTRATA_PYTHON=%%P"
if exist "%TINYSTRATA_PYTHON%" (
  "%TINYSTRATA_PYTHON%" -B ui\launch.py %*
) else if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -B ui\launch.py %*
) else (
  python -B ui\launch.py %*
)
if errorlevel 1 pause
