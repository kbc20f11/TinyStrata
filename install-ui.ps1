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
    [ValidateSet('auto', 'cuda', 'hip')][string]$Backend = 'auto',
    [string]$HipDirectory,
    [ValidatePattern('^gfx[0-9a-f]+(;gfx[0-9a-f]+)*$')][string]$HipArchitectures,
    [ValidateSet('qwen36', 'qwen3.6', 'lfm2', 'lfm2.5', 'gemma4', 'gemma-4', 'clover')][string]$ModelFamily = 'qwen36',
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
$tinyStrataUiOldErrorAction = $ErrorActionPreference
$tinyStrataUiOldSecurityProtocol = [Net.ServicePointManager]::SecurityProtocol
try {
    $ErrorActionPreference = 'Stop'
    $tinyStrataUiInstaller = $null
    if ($PSScriptRoot -and (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'install.ps1') -PathType Leaf)) {
        $tinyStrataUiCandidate = Join-Path $PSScriptRoot 'install.ps1'
        if ((Get-Command -Name $tinyStrataUiCandidate).Parameters.ContainsKey('WithUI')) { $tinyStrataUiInstaller = $tinyStrataUiCandidate }
    }
    if ($tinyStrataUiInstaller) {
        & $tinyStrataUiInstaller @tinyStrataUiForward
    } else {
        # Resolve main once so this bootstrap does not execute a changing raw URL.
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        $tinyStrataUiCommit = Invoke-RestMethod -Uri 'https://api.github.com/repos/kbc20f11/TinyStrata/commits/main' `
            -Headers @{ 'User-Agent' = 'TinyStrata-UI-Installer'; Accept = 'application/vnd.github+json' } -TimeoutSec 30
        $tinyStrataUiRevision = [string]$tinyStrataUiCommit.sha
        if ($tinyStrataUiRevision -notmatch '^[a-f0-9]{40}$') { throw 'GitHub returned an invalid TinyStrata revision.' }
        $tinyStrataUiResponse = Invoke-WebRequest -Uri ("https://raw.githubusercontent.com/kbc20f11/TinyStrata/$tinyStrataUiRevision/install.ps1") -UseBasicParsing -TimeoutSec 60
        & ([ScriptBlock]::Create([string]$tinyStrataUiResponse.Content)) @tinyStrataUiForward
    }
} finally {
    $ErrorActionPreference = $tinyStrataUiOldErrorAction
    [Net.ServicePointManager]::SecurityProtocol = $tinyStrataUiOldSecurityProtocol
}
