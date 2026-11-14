#Requires -Version 5.1
<#
.SYNOPSIS
Install TinyStrata with its optional browser UI.
.DESCRIPTION
Run with: irm 'https://raw.githubusercontent.com/kbc20f11/TinyStrata/refs/heads/main/install-ui.ps1' | iex
Uses the same API installer, with the UI enabled. API-only installs use install.ps1.
#>
[CmdletBinding()]
param(
    [string]$SourceDir,
    [string]$WorkDir,
    [ValidateSet('auto', 'cuda', 'hip', 'sycl')][string]$Backend = 'auto',
    # Explicit Intel uses separate ui-sycl history/settings and an experimental shortcut.
    [string]$IntelRuntimeDirectory,
    [ValidateSet('b390-qwen36')][string]$IntelProfile = 'b390-qwen36',
    [string]$HipDirectory,
    # An omitted value is empty under irm | iex and selects GPU auto-detection.
    [ValidatePattern('^(gfx[0-9a-f]+(;gfx[0-9a-f]+)*)?$')][string]$HipArchitectures,
    [ValidateSet('qwen36', 'qwen3.6', 'lfm2', 'lfm2.5', 'gemma4', 'gemma-4', 'clover', 'tiel', 'tiel-coder')][string]$ModelFamily = 'qwen36',
    [ValidateRange(0, 65535)][int]$Port = 0,
    [ValidateRange(1, 65535)][int]$UiPort = 3000,
    [switch]$NoStart,
    [switch]$SkipModelDownload,
    [switch]$CheckOnly,
    [switch]$Rebuild,
    [switch]$Update
)

$tinyStrataUiForward = @{}
foreach ($parameterName in $PSBoundParameters.Keys) { $tinyStrataUiForward[$parameterName] = $PSBoundParameters[$parameterName] }
$tinyStrataUiForward.WithUI = $true
$tinyStrataUiForward.SkipModelDownload = $true
$tinyStrataUiOldErrorAction = $ErrorActionPreference
$tinyStrataUiOldSecurityProtocol = [Net.ServicePointManager]::SecurityProtocol
try {
    $ErrorActionPreference = 'Stop'
    $tinyStrataUiInstaller = $null
    if (-not $SourceDir -and $PSScriptRoot) {
        # A downloaded ScriptBlock has no installation directory of its own.
        # Retain the caller's local installation, without treating a folder
        # containing only downloaded bootstrap scripts as a source tree.
        $tinyStrataUiLocalSource = $true
        foreach ($tinyStrataUiSourceFile in @('run.py', 'requirements.txt', 'CMakeLists.txt', 'serve\server.py')) {
            if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot $tinyStrataUiSourceFile) -PathType Leaf)) { $tinyStrataUiLocalSource = $false; break }
        }
        if ($tinyStrataUiLocalSource) { $tinyStrataUiForward.SourceDir = $PSScriptRoot }
    }
    # Archive installs and remote IEX always run the pinned latest installer.
    # A local checkout stays local; CheckOnly stays offline even with -Update.
    $tinyStrataUiGitCheckout = $PSScriptRoot -and (Test-Path -LiteralPath (Join-Path $PSScriptRoot '.git'))
    if (($CheckOnly -or ($tinyStrataUiGitCheckout -and -not $Update)) -and $PSScriptRoot -and (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'install.ps1') -PathType Leaf)) {
        $tinyStrataUiCandidate = Join-Path $PSScriptRoot 'install.ps1'
        if ((Get-Command -Name $tinyStrataUiCandidate).Parameters.ContainsKey('WithUI')) { $tinyStrataUiInstaller = $tinyStrataUiCandidate }
    }
    if ($tinyStrataUiInstaller) {
        & $tinyStrataUiInstaller @tinyStrataUiForward
    } else {
        # Resolve main once so this bootstrap does not execute a changing raw URL.
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        $tinyStrataUiCommit = Invoke-RestMethod -Uri 'https://api.github.com/repos/kbc20f11/TinyStrata/commits/main' `
            -Headers @{ 'User-Agent' = 'TinyStrata-UI-Installer'; Accept = 'application/vnd.github+json'; 'Cache-Control' = 'no-cache' } -TimeoutSec 30
        $tinyStrataUiRevision = [string]$tinyStrataUiCommit.sha
        if ($tinyStrataUiRevision -notmatch '^[a-f0-9]{40}$') { throw 'GitHub returned an invalid TinyStrata revision.' }
        $tinyStrataUiForward.SourceRevision = $tinyStrataUiRevision
        $tinyStrataUiResponse = Invoke-WebRequest -Uri ("https://raw.githubusercontent.com/kbc20f11/TinyStrata/$tinyStrataUiRevision/install.ps1") -UseBasicParsing -TimeoutSec 60
        & ([ScriptBlock]::Create([string]$tinyStrataUiResponse.Content)) @tinyStrataUiForward
    }
} finally {
    $ErrorActionPreference = $tinyStrataUiOldErrorAction
    [Net.ServicePointManager]::SecurityProtocol = $tinyStrataUiOldSecurityProtocol
}
