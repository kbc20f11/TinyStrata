#Requires -Version 5.1
<#
.SYNOPSIS
Install and start TinyStrata on Windows.
.DESCRIPTION
Run with: irm 'https://raw.githubusercontent.com/kbc20f11/TinyStrata/refs/heads/main/install.ps1' | iex
Uses local sources when available, otherwise downloads them from GitHub.
Other downloads use official Python, NVIDIA, AMD and Microsoft endpoints.
The selected GPU driver must already be installed. AMD builds use the private
pinned ROCm 10.2.0a20260930 SDK (HIP 7.17) by default; -HipDirectory selects a
custom SDK. No persistent PATH or execution policy changes are made. Runtime
data stays in the sibling TinyStrata-work folder.
#>
[CmdletBinding()]
param(
    [string]$SourceDir,
    [string]$WorkDir,
    [ValidateSet('auto', 'cuda', 'hip')][string]$Backend = 'auto',
    [string]$HipDirectory,
    # An omitted value is empty under irm | iex and selects GPU auto-detection.
    [ValidatePattern('^(gfx[0-9a-f]+(;gfx[0-9a-f]+)*)?$')][string]$HipArchitectures,
    [ValidateSet('qwen36', 'qwen3.6', 'lfm2', 'lfm2.5', 'gemma4', 'gemma-4', 'clover', 'tiel', 'tiel-coder')][string]$ModelFamily = 'qwen36',
    [ValidateRange(0, 65535)][int]$Port = 0,
    [switch]$WithUI,
    [ValidateRange(1, 65535)][int]$UiPort = 3000,
    [switch]$NoStart,
    [switch]$SkipModelDownload,
    [switch]$CheckOnly,
    [switch]$Rebuild,
    [switch]$Update,
    # Passed by the UI bootstrap after it resolves main once.
    [ValidatePattern('^([a-f0-9]{40})?$')][string]$SourceRevision
)

function Get-VerifiedDownload {
    param([string]$Uri, [string]$Destination, [string]$Sha256)
    if ($Sha256 -notmatch '^[a-fA-F0-9]{64}$' -or ([Uri]$Uri).Scheme -ne 'https') { throw 'Invalid verified download parameters.' }
    if ((Test-Path -LiteralPath $Destination -PathType Leaf) -and (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash -eq $Sha256) { return $Destination }
    New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force | Out-Null
    $partial = $Destination + '.part'
    Write-Host "Downloading $([IO.Path]::GetFileName($Destination))..."
    Get-TinyStrataOfficialFile -Uri $Uri -Destination $partial
    if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $Sha256) { throw "Download checksum mismatch: $Destination" }
    Move-Item -LiteralPath $partial -Destination $Destination -Force
    return $Destination
}

# Local source-build fallback, used only when no native engine is available.
# Reference: https://github.com/microsoft/vswhere/wiki/Find-VC
# Reference: https://docs.nvidia.com/cuda/archive/13.2.2/cuda-installation-guide-microsoft-windows/index.html

function Get-TinyStrataOfficialFile {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$Uri, [Parameter(Mandatory = $true)][string]$Destination)
    $parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    if (([Uri]$Uri).Scheme -ne 'https') { throw 'Official downloads require HTTPS.' }
    $client = New-Object Net.WebClient
    try {
        $download = $client.DownloadFileTaskAsync([Uri]$Uri, $Destination)
        $deadline = [DateTime]::UtcNow.AddMinutes(45)
        while (-not $download.Wait(15000)) {
            Write-Host "Still downloading $([IO.Path]::GetFileName($Destination))..."
            if ([DateTime]::UtcNow -gt $deadline) { $client.CancelAsync(); throw 'Download timed out; run the installer again to retry.' }
        }
        $null = $download.GetAwaiter().GetResult()
    } finally { $client.Dispose() }
}

function Test-TinyStrataSource {
    param([string]$Directory)
    foreach ($name in @('run.py', 'requirements.txt', 'CMakeLists.txt', 'serve\server.py')) {
        if (-not (Test-Path -LiteralPath (Join-Path $Directory $name) -PathType Leaf)) { return $false }
    }
    return $true
}

function Assert-TinyStrataSourceDestination {
    param([string]$Directory)
    if (Test-Path -LiteralPath $Directory) {
        $item = Get-Item -LiteralPath $Directory -Force
        if (-not $item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
            @(Get-ChildItem -LiteralPath $Directory -Force).Count -gt 0) {
            throw "Source destination is not an empty directory: $Directory. Choose another -SourceDir; existing files were left unchanged."
        }
    }
}

function Get-TinyStrataSourceFingerprint {
    param([string]$Directory, [switch]$NativeOnly)
    $root = [IO.Path]::GetFullPath($Directory).TrimEnd('\')
    $files = New-Object 'System.Collections.Generic.List[string]'
    $pending = New-Object 'System.Collections.Generic.Stack[string]'
    if ($NativeOnly) {
        $files.Add((Join-Path $root 'CMakeLists.txt'))
        foreach ($name in @('src', 'include', 'cmake', 'third_party')) { $pending.Push((Join-Path $root $name)) }
    } else { $pending.Push($root) }
    while ($pending.Count) {
        $directoryPath = $pending.Pop()
        $directoryItem = Get-Item -LiteralPath $directoryPath -Force -ErrorAction Stop
        if ($directoryItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Cannot fingerprint linked source paths.' }
        foreach ($item in Get-ChildItem -LiteralPath $directoryPath -Force -ErrorAction Stop) {
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Cannot fingerprint linked source paths.' }
            if ($item.PSIsContainer) { $pending.Push($item.FullName) }
            elseif ($item.FullName -ne (Join-Path $root '.tinystrata-source.json')) { $files.Add($item.FullName) }
        }
    }
    $text = New-Object Text.StringBuilder
    # Culture/host collation differs between Windows PowerShell and pwsh.
    $files.Sort([StringComparer]::Ordinal)
    foreach ($path in $files) {
        $relative = $path.Substring($root.Length + 1).Replace('\', '/')
        $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256 -ErrorAction Stop).Hash.ToLowerInvariant()
        $null = $text.Append($relative).Append([char]0).Append($hash).Append("`n")
    }
    if ($NativeOnly) {
        # Build flags are native inputs too; ordinary installer/UI edits are not.
        $tokens = $null; $errors = $null
        $ast = [System.Management.Automation.Language.Parser]::ParseFile((Join-Path $root 'install.ps1'), [ref]$tokens, [ref]$errors)
        if ($errors.Count) { throw 'Cannot fingerprint invalid native build policy.' }
        foreach ($name in @('Invoke-TinyStrataPortableBuild', 'Get-TinyStrataBuildArchitectures')) {
            $definition = $ast.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name }, $true)
            if (-not $definition) { throw "Native build policy is missing: $name" }
            $null = $text.Append($name).Append([char]0).Append($definition.Extent.Text.Replace("`r`n", "`n")).Append("`n")
        }
    }
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($text.ToString())))).Replace('-', '').ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function Test-TinyStrataSourceRevision {
    param([string]$Directory, [string]$Revision)
    if ($Revision -notmatch '^[a-f0-9]{40}$') { return $false }
    try {
        $stamp = Get-Content -LiteralPath (Join-Path $Directory '.tinystrata-source.json') -Raw -ErrorAction Stop | ConvertFrom-Json
        return ($stamp.schema -eq 1 -and $stamp.revision -eq $Revision -and
                $stamp.tree_fingerprint -eq (Get-TinyStrataSourceFingerprint -Directory $Directory))
    } catch { return $false }
}

function Write-TinyStrataNativeBuildStamp {
    param([string]$Directory, [string]$SourceRoot, [ValidateSet('cuda', 'hip')][string]$Backend)
    $files = [ordered]@{}
    foreach ($name in @('tinystrata.exe', 'tinystrata-lfm2.exe', 'tinystrata-gemma4.exe', 'tinystrata-vision.exe') + $(if ($Backend -eq 'cuda') { @('strata_kq.dll') } else { @() })) {
        $files[$name] = (Get-FileHash -LiteralPath (Join-Path $Directory $name) -Algorithm SHA256 -ErrorAction Stop).Hash.ToLowerInvariant()
    }
    $stamp = [ordered]@{ schema = 1; backend = $Backend; source_fingerprint = (Get-TinyStrataSourceFingerprint -Directory $SourceRoot -NativeOnly); files = $files }
    $path = Join-Path $Directory '.tinystrata-native-build.json'
    [IO.File]::WriteAllText($path, ($stamp | ConvertTo-Json -Depth 4), (New-Object Text.UTF8Encoding($false)))
}

function Test-TinyStrataNativeBuildStamp {
    param([string]$Directory, [string]$Fingerprint, [ValidateSet('cuda', 'hip')][string]$Backend)
    try {
        $stamp = Get-Content -LiteralPath (Join-Path $Directory '.tinystrata-native-build.json') -Raw -ErrorAction Stop | ConvertFrom-Json
        if ($stamp.schema -ne 1 -or $stamp.backend -ne $Backend -or $stamp.source_fingerprint -ne $Fingerprint) { return $false }
        foreach ($name in @('tinystrata.exe', 'tinystrata-lfm2.exe', 'tinystrata-gemma4.exe', 'tinystrata-vision.exe') + $(if ($Backend -eq 'cuda') { @('strata_kq.dll') } else { @() })) {
            if ($stamp.files.$name -ne (Get-FileHash -LiteralPath (Join-Path $Directory $name) -Algorithm SHA256 -ErrorAction Stop).Hash.ToLowerInvariant()) { return $false }
        }
        return $true
    } catch { return $false }
}

function Initialize-TinyStrataSource {
    param([string]$SourceRoot, [string]$CacheRoot, [switch]$Refresh, [string]$Revision)
    $SourceRoot = [IO.Path]::GetFullPath($SourceRoot).TrimEnd('\')
    $existing = Test-TinyStrataSource -Directory $SourceRoot
    $backup = $null
    if ($existing -and -not $Refresh) { return }
    if ($existing) {
        $sourceItem = Get-Item -LiteralPath $SourceRoot -Force
        if (($sourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
            (Test-Path -LiteralPath (Join-Path $SourceRoot '.git'))) {
            throw 'Update requires an archive installation, not a Git checkout or linked source directory.'
        }
    } else { Assert-TinyStrataSourceDestination -Directory $SourceRoot }
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    if (-not $Revision) {
        $commit = Invoke-RestMethod -Uri 'https://api.github.com/repos/kbc20f11/TinyStrata/commits/main' `
            -Headers @{ 'User-Agent' = 'TinyStrata-Installer'; Accept = 'application/vnd.github+json'; 'Cache-Control' = 'no-cache' } -TimeoutSec 30
        $Revision = [string]$commit.sha
    }
    if ($revision -notmatch '^[a-f0-9]{40}$') { throw 'GitHub returned an invalid TinyStrata revision.' }
    if ($existing -and (Test-TinyStrataSourceRevision -Directory $SourceRoot -Revision $Revision)) {
        return [pscustomobject]@{ SourceRoot = $SourceRoot; Backup = $null; Changed = $false; Revision = $Revision }
    }
    $archive = Join-Path $CacheRoot ("TinyStrata-$revision.zip")
    Write-Host "Downloading TinyStrata sources ($($revision.Substring(0, 7)))..."
    Get-TinyStrataOfficialFile -Uri ("https://codeload.github.com/kbc20f11/TinyStrata/zip/$revision") -Destination $archive
    $sourceParent = Split-Path -Parent $SourceRoot
    New-Item -ItemType Directory -Path $sourceParent -Force | Out-Null
    $staging = Join-Path $sourceParent ('.TinyStrata-source-' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $staging | Out-Null
    $unpacked = Join-Path $staging ("TinyStrata-$revision")
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($archive)
    try {
        foreach ($entry in $zip.Entries) {
            if ($entry.FullName -match ':') { throw 'GitHub source archive contains an unexpected path.' }
            $target = [IO.Path]::GetFullPath((Join-Path $staging $entry.FullName.Replace('/', '\')))
            if (-not $entry.Name -and $target.TrimEnd('\') -eq $unpacked) { continue }
            if (-not $target.StartsWith($unpacked + '\', [StringComparison]::OrdinalIgnoreCase)) {
                throw 'GitHub source archive contains an unexpected path.'
            }
            if (-not $entry.Name) { continue }
            New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $false)
        }
    } finally { $zip.Dispose() }
    if (-not (Test-TinyStrataSource -Directory $unpacked)) { throw 'Downloaded TinyStrata sources are incomplete.' }
    $stamp = [ordered]@{ schema = 1; revision = $Revision; tree_fingerprint = (Get-TinyStrataSourceFingerprint -Directory $unpacked) }
    [IO.File]::WriteAllText((Join-Path $unpacked '.tinystrata-source.json'), ($stamp | ConvertTo-Json), (New-Object Text.UTF8Encoding($false)))
    if ($existing) {
        # Replace archive sources reversibly; runtime settings and models stay
        # in the separate work directory. Never update a Git checkout in place.
        $sourceItem = Get-Item -LiteralPath $SourceRoot -Force
        if (($sourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
            (Test-Path -LiteralPath (Join-Path $SourceRoot '.git'))) {
            throw 'The source directory changed while downloading; it was left unchanged.'
        }
        # Directory.Move must stay on the source volume even with a custom WorkDir.
        $backupParent = [IO.Path]::GetFullPath((Join-Path $sourceParent '.TinyStrata-source-backups'))
        $backup = [IO.Path]::GetFullPath((Join-Path $backupParent ((Split-Path -Leaf $SourceRoot) + '-' + [Guid]::NewGuid().ToString('N'))))
        if (-not $backup.StartsWith($backupParent.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Invalid source backup destination.'
        }
        New-Item -ItemType Directory -Path $backupParent -Force | Out-Null
        [IO.Directory]::Move($SourceRoot, $backup)
        try { [IO.Directory]::Move($unpacked, $SourceRoot) }
        catch {
            if (-not (Test-Path -LiteralPath $SourceRoot)) { [IO.Directory]::Move($backup, $SourceRoot) }
            throw
        }
        Write-Host "Previous sources: $backup"
    } else {
        Assert-TinyStrataSourceDestination -Directory $SourceRoot
        if (Test-Path -LiteralPath $SourceRoot) { [IO.Directory]::Delete($SourceRoot, $false) }
        [IO.Directory]::Move($unpacked, $SourceRoot)
    }
    [IO.Directory]::Delete($staging, $false)
    Write-Host "Source: $SourceRoot"
    return [pscustomobject]@{ SourceRoot = $SourceRoot; Backup = $backup; Changed = $true; Revision = $Revision }
}

function Restore-TinyStrataSource {
    param($Change)
    if (-not $Change -or -not $Change.Backup) { return }
    if (-not (Test-Path -LiteralPath $Change.Backup -PathType Container)) { throw 'The previous source backup is missing.' }
    $failed = $Change.Backup + '.failed-update'
    if (Test-Path -LiteralPath $Change.SourceRoot) { [IO.Directory]::Move($Change.SourceRoot, $failed) }
    [IO.Directory]::Move($Change.Backup, $Change.SourceRoot)
    Write-Warning "Update preparation failed; previous sources restored. Unpublished sources: $failed"
}

function Ensure-TinyStrataBuildTools {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$CacheDirectory)
    try { return Find-TinyStrataBuildTools } catch { Write-Host 'Installing the Microsoft C++ Build Tools required for the first native build.' }
    $bootstrap = Join-Path $CacheDirectory 'vs_buildtools.exe'
    Get-TinyStrataOfficialFile -Uri 'https://aka.ms/vs/17/release/vs_BuildTools.exe' -Destination $bootstrap
    $signature = Get-AuthenticodeSignature -LiteralPath $bootstrap
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch '(?:^|,\s*)O=Microsoft Corporation(?:,|$)') {
        throw 'The downloaded Visual Studio installer does not have a valid Microsoft signature.'
    }
    Write-Host 'Windows may ask for administrator approval. Visual Studio Build Tools can take several minutes and several GB of disk space.'
    $parameters = @{
        FilePath = $bootstrap
        ArgumentList = '--wait --passive --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --add Microsoft.VisualStudio.Component.VC.CMake.Project'
        PassThru = $true
        WindowStyle = 'Hidden'
    }
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { $parameters.Verb = 'RunAs' }
    $process = Start-Process @parameters
    while (-not $process.WaitForExit(30000)) { Write-Host 'Microsoft C++ Build Tools installation is still running...' }
    if ($process.ExitCode -eq 3010) { throw 'Microsoft Build Tools installation needs a restart. Restart Windows and run the TinyStrata installer again.' }
    if ($process.ExitCode -ne 0) { throw "Microsoft Build Tools installer failed with exit code $($process.ExitCode)." }
    return Find-TinyStrataBuildTools
}

function Get-TinyStrataCudaCompilerVersion {
    param([string]$Text)
    if ($Text -match '\bV(\d+\.\d+\.\d+)\b') { return [version]$Matches[1] }
    return $null
}

function Test-TinyStrataCudaCompilerVersion {
    param([version]$Version)
    # CUDA 13.2.0/13.2.1 (nvcc .51/.78) miscompile IQ*_S byte extraction
    # on Blackwell. Check nvcc's patch version, not just "release 13.2".
    # https://github.com/ggml-org/llama.cpp/issues/28581
    return $null -ne $Version -and $Version -ge [version]'13.2.86'
}

function Test-TinyStrataCudaToolkit {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$Directory)
    $compiler = Join-Path $Directory 'bin\nvcc.exe'
    if (-not (Test-Path -LiteralPath $compiler -PathType Leaf)) { return $false }
    $text = (& $compiler --version 2>$null) -join [Environment]::NewLine
    if ($LASTEXITCODE -ne 0) { return $false }
    $compilerVersion = Get-TinyStrataCudaCompilerVersion -Text $text
    if (-not (Test-TinyStrataCudaCompilerVersion -Version $compilerVersion)) { return $false }
    foreach ($relative in @('include\cuda.h', 'include\cublas_v2.h', 'lib\x64\cudart.lib', 'lib\x64\cublas.lib')) {
        if (-not (Test-Path -LiteralPath (Join-Path $Directory $relative) -PathType Leaf)) { return $false }
    }
    return $true
}

function Ensure-TinyStrataCudaToolkit {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$DestinationDirectory,
        [Parameter(Mandatory = $true)][string]$CacheDirectory,
        [string[]]$SearchDirectories = @()
    )
    $candidates = @($SearchDirectories) + @($DestinationDirectory)
    Get-ChildItem Env: | Where-Object { $_.Name -eq 'CUDA_PATH' -or $_.Name -like 'CUDA_PATH_V*' } | ForEach-Object { $candidates += $_.Value }
    $installedRoot = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path -LiteralPath $installedRoot -PathType Container) {
        Get-ChildItem -LiteralPath $installedRoot -Directory | Sort-Object Name -Descending | ForEach-Object { $candidates += $_.FullName }
    }
    $command = Get-Command nvcc.exe -ErrorAction SilentlyContinue
    if ($command) { $candidates += Split-Path -Parent (Split-Path -Parent $command.Source) }
    foreach ($candidate in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
        if (Test-TinyStrataCudaToolkit -Directory $candidate) { return [IO.Path]::GetFullPath($candidate) }
    }

    Write-Host 'Preparing CUDA 13.2.2 or newer from NVIDIA; earlier CUDA 13.2 compilers can corrupt IQ model output on Blackwell. No driver or system CUDA installation is changed.'
    $baseUri = 'https://developer.download.nvidia.com/compute/cuda/redist/'
    $manifestPath = Join-Path $CacheDirectory 'redistrib_13.2.2.json'
    Get-TinyStrataOfficialFile -Uri ($baseUri + 'redistrib_13.2.2.json') -Destination $manifestPath
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($manifest.release_label -ne '13.2.2') { throw 'NVIDIA CUDA manifest has an unexpected release.' }
    $destinationRoot = [IO.Path]::GetFullPath($DestinationDirectory)
    New-Item -ItemType Directory -Path $destinationRoot -Force | Out-Null
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    foreach ($name in @('cuda_nvcc', 'cuda_cudart', 'cuda_cccl', 'cuda_crt', 'libnvvm', 'libcublas')) {
        $package = $manifest.$name.'windows-x86_64'
        if (-not $package -or $package.relative_path -notmatch '^[A-Za-z0-9_./-]+\.zip$' -or $package.relative_path -match '(?:^|/)\.\.(?:/|$)' -or $package.sha256 -notmatch '^[a-fA-F0-9]{64}$') { throw "NVIDIA CUDA manifest has an invalid package: $name" }
        $archive = Join-Path $CacheDirectory ($name + '-13.2.2.zip')
        $verified = (Test-Path -LiteralPath $archive -PathType Leaf) -and ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -eq $package.sha256)
        if (-not $verified) {
            $partial = $archive + '.part'
            Write-Host "Downloading NVIDIA $name..."
            Get-TinyStrataOfficialFile -Uri ($baseUri + $package.relative_path) -Destination $partial
            if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $package.sha256) { throw "NVIDIA CUDA checksum mismatch: $name" }
            Move-Item -LiteralPath $partial -Destination $archive -Force
        }
        Write-Host "Extracting verified NVIDIA $name..."
        $zip = [IO.Compression.ZipFile]::OpenRead($archive)
        try {
            foreach ($entry in $zip.Entries) {
                if (-not $entry.Name) { continue }
                $parts = $entry.FullName.Replace('\', '/').Split('/')
                if ($parts.Length -lt 2) { continue }
                $relative = ($parts[1..($parts.Length - 1)] -join [IO.Path]::DirectorySeparatorChar)
                $target = [IO.Path]::GetFullPath((Join-Path $destinationRoot $relative))
                if (-not $target.StartsWith($destinationRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'NVIDIA archive contains an unsafe path.' }
                New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
                [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $true)
            }
        }
        finally { $zip.Dispose() }
    }
    if (-not (Test-TinyStrataCudaToolkit -Directory $destinationRoot)) { throw 'The downloaded NVIDIA CUDA toolkit is incomplete.' }
    return $destinationRoot
}

# HIP toolchain/device discovery follows Strata's tools/hip/build_windows.bat
# (MIT), upstream cbfd5b926b794b356195d3cb600191b0c3567540. No driver or SDK
# is replaced by these helpers. A private pinned SDK can be prepared below.
function Get-TinyStrataBackend {
    param([ValidateSet('auto', 'cuda', 'hip')][string]$Requested = 'auto', $Configuration)
    if ($Requested -ne 'auto') { return $Requested }
    if ($Configuration -and $Configuration.backend -in @('cuda', 'hip')) { return [string]$Configuration.backend }
    $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($smi) {
        $devices = @(& $smi.Source -L 2>$null)
        if ($LASTEXITCODE -eq 0 -and ($devices -join "`n") -match 'GPU \d+:') { return 'cuda' }
    }
    $amd = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.PNPDeviceID -match '(?i)VEN_1002' })
    if ($amd.Count -gt 0) { return 'hip' }
    throw 'No usable NVIDIA GPU or AMD Radeon adapter was detected. Install its GPU driver, or select -Backend cuda/hip explicitly.'
}

function Get-TinyStrataHipCompiler {
    param([string]$Directory)
    foreach ($relative in @('lib\llvm\bin\clang++.exe', 'bin\clang++.exe')) {
        $candidate = Join-Path $Directory $relative
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    return $null
}

function Find-TinyStrataHipToolkit {
    param([string]$Directory)
    if ($Directory) {
        if ((Test-Path -LiteralPath $Directory -PathType Container) -and
            (Get-TinyStrataHipCompiler -Directory $Directory) -and
            (Test-Path -LiteralPath (Join-Path $Directory 'include\hip\hip_runtime.h') -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $Directory).Path
        }
        throw "The selected AMD SDK is incomplete or does not exist: $Directory"
    }
    $roots = @()
    foreach ($name in @('HIP_PATH', 'ROCM_PATH')) { if ([Environment]::GetEnvironmentVariable($name)) { $roots += [Environment]::GetEnvironmentVariable($name) } }
    $sdk = Get-Command rocm-sdk.exe -ErrorAction SilentlyContinue
    if ($sdk) {
        $reported = @(& $sdk.Source path --root 2>$null)
        if ($LASTEXITCODE -eq 0) { $roots += @($reported | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Container) }) }
    }
    $installed = Join-Path $env:ProgramFiles 'AMD\ROCm'
    if (Test-Path -LiteralPath $installed -PathType Container) {
        $roots += $installed
        $roots += @(Get-ChildItem -LiteralPath $installed -Directory | Sort-Object Name -Descending | ForEach-Object FullName)
    }
    foreach ($root in @($roots | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $root -PathType Container)) { continue }
        if ((Get-TinyStrataHipCompiler -Directory $root) -and (Test-Path -LiteralPath (Join-Path $root 'include\hip\hip_runtime.h') -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $root).Path
        }
    }
    throw 'AMD builds require an existing ROCm/HIP SDK with Clang, HIP headers and hipBLAS. Install the AMD SDK, then rerun with -Backend hip -HipDirectory <SDK root>. This installer does not replace GPU drivers.'
}

function Get-TinyStrataHipArchitectures {
    param([string]$Directory, [string]$Requested)
    if ($Requested) { return $Requested }
    if ($env:STRATA_HIP_ARCHS -match '^gfx[0-9a-f]+(;gfx[0-9a-f]+)*$') { return $env:STRATA_HIP_ARCHS }
    foreach ($relative in @('lib\llvm\bin\amdgpu-arch.exe', 'bin\amdgpu-arch.exe')) {
        $query = Join-Path $Directory $relative
        if (-not (Test-Path -LiteralPath $query -PathType Leaf)) { continue }
        # Ignore an unsupported integrated Radeon when a supported discrete
        # card is also present. Keep this list aligned with hip_backend.cmake.
        $supported = @('gfx1010', 'gfx1011', 'gfx1012', 'gfx1030', 'gfx1031', 'gfx1034', 'gfx1100', 'gfx1101', 'gfx1102', 'gfx1103', 'gfx1150', 'gfx1151', 'gfx1152', 'gfx1200', 'gfx1201')
        $archs = @(& $query 2>$null | ForEach-Object { $_.Trim() } | Where-Object { $_ -in $supported } | Select-Object -Unique)
        if ($LASTEXITCODE -eq 0 -and $archs.Count -gt 0) { return $archs -join ';' }
    }
    throw 'Could not determine the AMD GPU architecture. Pass -HipArchitectures gfx1201 (or the gfx target reported by your AMD SDK).'
}

function Test-TinyStrataRocmPackages {
    param([string]$Python, [string[]]$Names)
    $probe = "import importlib.metadata as m,sys; print(all(m.version(n)=='10.2.0a20260930' for n in sys.argv[1:]))"
    $report = @(& $Python -I -c $probe @Names 2>$null)
    return $LASTEXITCODE -eq 0 -and ($report -join '').Trim() -eq 'True'
}

function Get-TinyStrataPrivateHipRoot {
    param([string]$SdkExecutable)
    $reported = @(& $SdkExecutable path --root 2>$null)
    if ($LASTEXITCODE -ne 0 -or $reported.Count -ne 1) { throw 'The private AMD SDK could not report its root directory.' }
    return Find-TinyStrataHipToolkit -Directory ([string]$reported[0])
}

function Get-TinyStrataPrivateHipDirectory {
    param([string]$DestinationDirectory)
    $sdk = Join-Path $DestinationDirectory 'Scripts\rocm-sdk.exe'
    $python = Join-Path $DestinationDirectory 'Scripts\python.exe'
    if ((Test-Path -LiteralPath $sdk -PathType Leaf) -and
        -not (Test-TinyStrataRocmPackages -Python $python -Names @('rocm', 'rocm-sdk-core', 'rocm-sdk-devel', 'rocm-sdk-libraries'))) {
        return $DestinationDirectory + '-10.2.0a20260930'
    }
    return $DestinationDirectory
}

function Ensure-TinyStrataHipToolkit {
    param([string]$DestinationDirectory, [string]$Python, [string]$Directory, [string]$Architectures)
    # An explicitly supplied path is a deliberate selection; never replace it
    # with a downloaded SDK when it is misspelled or incomplete.
    if ($Directory) { return Find-TinyStrataHipToolkit -Directory $Directory }
    $basePackages = @('rocm', 'rocm-sdk-core', 'rocm-sdk-devel', 'rocm-sdk-libraries')
    # An old runtime may still be referenced by another saved model profile.
    # Keep it immutable and prepare a new pin in a different environment.
    $DestinationDirectory = Get-TinyStrataPrivateHipDirectory -DestinationDirectory $DestinationDirectory
    $venvPython = Join-Path $DestinationDirectory 'Scripts\python.exe'
    $sdk = Join-Path $DestinationDirectory 'Scripts\rocm-sdk.exe'
    # Automatic installs always use the tested private pin. A global HIP_PATH
    # or system SDK can be older and must not silently replace this choice.
    # Explicit -HipDirectory above remains a deliberate custom SDK selection.
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        & $Python -I -m venv $DestinationDirectory | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the private AMD SDK environment.' }
    }
    $baseReady = (Test-Path -LiteralPath $sdk -PathType Leaf) -and (Test-TinyStrataRocmPackages -Python $venvPython -Names $basePackages)
    # AMD ships the small "rocm" metapackage as an sdist at this pin; the
    # compiler/runtime/device payloads must still be prebuilt wheels.
    $pipOptions = @('-I', '-m', 'pip', '--isolated', '--disable-pip-version-check', '--no-input', 'install', '--only-binary=:all:', '--no-binary=rocm', '--index-url', 'https://nightly.repo.amd.com/rocm/whl-next/')
    if (-not $baseReady) {
        Write-Host 'Preparing the private AMD ROCm compiler and libraries (several GB; no driver or system PATH changes)...'
        & $venvPython @pipOptions 'rocm[libraries,devel]==10.2.0a20260930' | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'The AMD ROCm SDK download/install failed. Rerun to reuse completed downloads.' }
        & $sdk init | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'The AMD ROCm SDK initialization failed.' }
    }
    $root = Get-TinyStrataPrivateHipRoot -SdkExecutable $sdk
    $targets = Get-TinyStrataHipArchitectures -Directory $root -Requested $Architectures
    $devicePackages = @($targets -split ';' | ForEach-Object { 'rocm-sdk-device-' + $_ })
    if (-not (Test-TinyStrataRocmPackages -Python $venvPython -Names $devicePackages)) {
        $extras = 'libraries,devel,' + (($targets -split ';' | ForEach-Object { 'device-' + $_ }) -join ',')
        Write-Host "Preparing AMD device libraries for $targets..."
        & $venvPython @pipOptions "rocm[$extras]==10.2.0a20260930" | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'The AMD GPU device-library download/install failed.' }
        & $sdk init | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'The AMD ROCm device-library initialization failed.' }
        $root = Get-TinyStrataPrivateHipRoot -SdkExecutable $sdk
    }
    return $root
}

function Get-TinyStrataHipRuntimeDirectories {
    param([string]$Directory)
    return @(@('bin', 'lib', 'lib\llvm\bin') | ForEach-Object { Join-Path $Directory $_ } | Where-Object { Test-Path -LiteralPath $_ -PathType Container })
}

function Get-TinyStrataBuildArchitectures {
    [CmdletBinding()]
    param()
    $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($smi) {
        $capabilities = @(& $smi.Source --query-gpu=compute_cap --format=csv,noheader 2>$null)
        if ($LASTEXITCODE -eq 0) {
            $architectures = @($capabilities | ForEach-Object { if ($_.Trim() -match '^(8\.6|8\.9|12\.0)$') { $Matches[1].Replace('.', '') } } | Select-Object -Unique)
            if ($architectures.Count -gt 0) { return $architectures -join ';' }
        }
    }
    return '86;89;120'
}

function Find-TinyStrataBuildTools {
    [CmdletBinding()]
    param()
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
        throw 'Source builds require Visual Studio 2022 or 2026 C++ Build Tools. The portable TinyStrata package does not need them.'
    }
    # Read UTF-8 directly: the PowerShell console code page can corrupt the
    # localized descriptions in vswhere's JSON before ConvertFrom-Json sees it.
    $queryInfo = New-Object Diagnostics.ProcessStartInfo
    $queryInfo.FileName = $vswhere
    $queryInfo.Arguments = '-products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -format json -utf8'
    $queryInfo.UseShellExecute = $false
    $queryInfo.CreateNoWindow = $true
    $queryInfo.RedirectStandardOutput = $true
    $queryInfo.RedirectStandardError = $true
    $queryInfo.StandardOutputEncoding = New-Object Text.UTF8Encoding($false)
    $queryInfo.StandardErrorEncoding = New-Object Text.UTF8Encoding($false)
    $query = New-Object Diagnostics.Process
    $query.StartInfo = $queryInfo
    try {
        if (-not $query.Start()) { throw 'Could not start the Visual Studio installation query.' }
        $stdout = $query.StandardOutput.ReadToEndAsync()
        $stderr = $query.StandardError.ReadToEndAsync()
        $query.WaitForExit()
        $instances = $stdout.GetAwaiter().GetResult()
        $queryError = $stderr.GetAwaiter().GetResult()
        if ($query.ExitCode -ne 0) { throw "Could not query Visual Studio installations: $queryError" }
    } finally { $query.Dispose() }
    $parsedInstances = ConvertFrom-Json -InputObject $instances
    foreach ($instance in @($parsedInstances | Sort-Object { [version]$_.installationVersion } -Descending)) {
        if ([version]$instance.installationVersion -lt [version]'17.0' -or [version]$instance.installationVersion -ge [version]'19.0') { continue }
        $vsRoot = $instance.installationPath
        $devCmd = Join-Path $vsRoot 'Common7\Tools\VsDevCmd.bat'
        $cmake = Join-Path $vsRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
        $ninja = Join-Path $vsRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
        if (-not (Test-Path -LiteralPath $cmake)) {
            $command = Get-Command cmake.exe -ErrorAction SilentlyContinue
            if ($command) { $cmake = $command.Source }
        }
        if (-not (Test-Path -LiteralPath $ninja)) {
            $command = Get-Command ninja.exe -ErrorAction SilentlyContinue
            if ($command) { $ninja = $command.Source }
        }
        if ((Test-Path -LiteralPath $devCmd -PathType Leaf) -and (Test-Path -LiteralPath $cmake -PathType Leaf) -and (Test-Path -LiteralPath $ninja -PathType Leaf)) {
            return [pscustomobject]@{ VisualStudio = $vsRoot; Version = $instance.installationVersion; DevCmd = $devCmd; CMake = $cmake; Ninja = $ninja }
        }
    }
    throw 'Source builds need Visual Studio 2022/2026 C++ tools, a Windows SDK, CMake and Ninja. Install the portable TinyStrata release to avoid development tools.'
}

function Test-TinyStrataVisionBackend {
    param([string]$ExecutablePath, [string[]]$SearchDirectories = @(), [string]$Backend = 'cuda')
    if (-not (Test-Path -LiteralPath $ExecutablePath -PathType Leaf)) { return $false }
    if ($Backend -eq 'hip') { return @(Get-TinyStrataHipImports -ExecutablePath $ExecutablePath).Count -gt 0 }
    $previousPath = $env:PATH
    try {
        $env:PATH = (@($SearchDirectories) + @((Split-Path -Parent $ExecutablePath), $previousPath)) -join ';'
        $report = & $ExecutablePath --capabilities 2>$null
        if ($LASTEXITCODE -ne 0) { return $false }
        $capabilities = ($report -join [Environment]::NewLine) | ConvertFrom-Json
        return $capabilities.backend -eq $Backend -or ($Backend -eq 'cuda' -and $capabilities.cuda -eq $true) -or ($Backend -eq 'hip' -and $capabilities.hip -eq $true)
    } catch {
        return $false
    } finally {
        $env:PATH = $previousPath
    }
}

function Test-TinyStrataEngineBackend {
    param([string]$ExecutablePath, [string[]]$SearchDirectories = @(), [string]$Backend = 'cuda')
    if (Test-TinyStrataVisionBackend -ExecutablePath $ExecutablePath -SearchDirectories $SearchDirectories -Backend $Backend) { return $true }
    # Keep older CUDA builds reusable even though they predate --capabilities.
    if (-not (Test-Path -LiteralPath $ExecutablePath -PathType Leaf)) { return $false }
    $imports = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($ExecutablePath))
    return ($Backend -eq 'hip' -and $imports.Contains('amdhip64')) -or ($Backend -eq 'cuda' -and $imports.Contains('cudart64_'))
}

function Invoke-TinyStrataPortableBuild {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$SourceDirectory,
        [Parameter(Mandatory = $true)][string]$BuildDirectory,
        [string]$CudaDirectory,
        [ValidateSet('cuda', 'hip')][string]$Backend = 'cuda',
        [string]$HipDirectory,
        [ValidatePattern('^gfx[0-9a-f]+(;gfx[0-9a-f]+)*$')][string]$HipArchitectures,
        [ValidatePattern('^(86|89|120)(;(86|89|120))*$')][string]$Architectures = (Get-TinyStrataBuildArchitectures),
        [ValidateRange(1, 32)][int]$Parallel = 4,
        [switch]$Clean
    )
    $tools = Find-TinyStrataBuildTools
    $sourceRoot = (Resolve-Path -LiteralPath $SourceDirectory).Path
    $buildRoot = [System.IO.Path]::GetFullPath($BuildDirectory)
    if ($Backend -eq 'hip') {
        $hipRoot = Find-TinyStrataHipToolkit -Directory $HipDirectory
        $hipArchs = Get-TinyStrataHipArchitectures -Directory $hipRoot -Requested $HipArchitectures
        $hipCompiler = Get-TinyStrataHipCompiler -Directory $hipRoot
    } else {
        $cudaRoot = (Resolve-Path -LiteralPath $CudaDirectory).Path
        $nvcc = Join-Path $cudaRoot 'bin\nvcc.exe'
        if (-not (Test-Path -LiteralPath $nvcc -PathType Leaf)) { throw "CUDA compiler not found: $nvcc" }
        $versionText = (& $nvcc --version) -join [Environment]::NewLine
        if ($LASTEXITCODE -ne 0) { throw 'Cannot determine the CUDA toolkit version.' }
        $compilerVersion = Get-TinyStrataCudaCompilerVersion -Text $versionText
        if (-not $compilerVersion) { throw 'Cannot determine the CUDA compiler patch version from nvcc --version.' }
        if (-not (Test-TinyStrataCudaCompilerVersion -Version $compilerVersion)) {
            throw 'This portable build helper requires CUDA 13.2.2 (nvcc 13.2.86) or newer. CUDA 13.2.0/13.2.1 miscompile IQ model kernels on Blackwell.'
        }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot 'CMakeLists.txt') -PathType Leaf)) { throw 'Source directory is not a TinyStrata checkout.' }

    # Read Microsoft's developer-shell environment; do not mutate the user's
    # persistent PATH, install tools, alter existing build caches, or elevate.
    $before = @{}
    Get-ChildItem Env: | ForEach-Object { $before[$_.Name] = $_.Value }
    try {
        # ProcessStartInfo avoids Windows PowerShell 5.1 re-escaping CMD's
        # nested quotes around a Visual Studio path containing spaces.
        $setupInfo = New-Object Diagnostics.ProcessStartInfo
        $setupInfo.FileName = $env:ComSpec
        $setupInfo.Arguments = '/d /s /c ""{0}" -no_logo -arch=amd64 -host_arch=amd64 && set"' -f $tools.DevCmd
        $setupInfo.UseShellExecute = $false
        $setupInfo.CreateNoWindow = $true
        $setupInfo.RedirectStandardOutput = $true
        $setupInfo.RedirectStandardError = $true
        $setupProcess = [Diagnostics.Process]::Start($setupInfo)
        $setupOutput = $setupProcess.StandardOutput.ReadToEndAsync()
        $setupError = $setupProcess.StandardError.ReadToEndAsync()
        $setupProcess.WaitForExit()
        if ($setupProcess.ExitCode -ne 0) { throw "Visual Studio C++ environment setup failed: $($setupError.Result)" }
        $developerEnvironment = $setupOutput.Result -split '\r?\n'
        foreach ($line in $developerEnvironment) {
            if ($line -match '^([^=]+)=(.*)$') {
                [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
            }
        }
        $env:VSLANG = '1033'
        $configure = @('-S', $sourceRoot, '-B', $buildRoot, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', "-DTINYSTRATA_BACKEND=$($Backend.ToUpperInvariant())", '-DTINYSTRATA_LFM2=ON', '-DTINYSTRATA_GEMMA4=ON', '-DSTRATA_PORTABLE=ON', "-DCMAKE_MAKE_PROGRAM=$($tools.Ninja)")
        if ($Backend -eq 'hip') {
            $env:HIP_PLATFORM = 'amd'
            $env:HIP_PATH = $hipRoot
            $env:ROCM_PATH = $hipRoot
            $env:PATH = ((Get-TinyStrataHipRuntimeDirectories -Directory $hipRoot) + @($env:PATH)) -join ';'
            $clangC = Join-Path (Split-Path -Parent $hipCompiler) 'clang.exe'
            $bitcode = Join-Path $hipRoot 'lib\llvm\amdgcn\bitcode'
            if (-not (Test-Path -LiteralPath $bitcode -PathType Container)) { $bitcode = Join-Path $hipRoot 'amdgcn\bitcode' }
            # An SDK under "C:\Program Files" has a space and backslashes. PowerShell 5.1 strips the quotes from a native
            # argument, which split -DCMAKE_HIP_FLAGS in two, and a backslash path written into CMakeHIPCompiler.cmake is an
            # invalid escape. Pass the flags through an initial-cache file and give CMake forward-slash paths.
            $hipRootCMake = $hipRoot.Replace('\','/')
            $hipFlags = '--rocm-path=\"{0}\" --rocm-device-lib-path=\"{1}\"' -f $hipRootCMake, $bitcode.Replace('\','/')
            $hipCache = Join-Path $buildRoot 'tinystrata-hip-flags.cmake'
            New-Item -ItemType Directory -Path $buildRoot -Force | Out-Null
            [IO.File]::WriteAllText($hipCache, ('set(CMAKE_HIP_FLAGS "{0}" CACHE STRING "" FORCE)' -f $hipFlags), (New-Object Text.UTF8Encoding($false)))
            $configure += @('-C', $hipCache, "-DCMAKE_HIP_ARCHITECTURES=$hipArchs", "-DCMAKE_C_COMPILER=$clangC", "-DCMAKE_CXX_COMPILER=$hipCompiler", "-DCMAKE_HIP_COMPILER=$hipCompiler", "-DCMAKE_HIP_COMPILER_ROCM_ROOT=$hipRootCMake", "-DCMAKE_PREFIX_PATH=$hipRootCMake")
            Write-Host "Building TinyStrata for AMD HIP architecture(s) $hipArchs. This can take several minutes."
        } else {
            $env:CUDA_PATH = $cudaRoot
            $env:PATH = (Join-Path $cudaRoot 'bin') + ';' + (Join-Path $cudaRoot 'bin\x64') + ';' + $env:PATH
            $configure += @("-DCMAKE_CUDA_ARCHITECTURES=$Architectures", "-DCMAKE_CUDA_COMPILER=$nvcc", "-DCUDAToolkit_ROOT=$cudaRoot")
            Write-Host "Building TinyStrata for CUDA architecture(s) $Architectures. This can take several minutes."
        }
        & $tools.CMake @configure | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'TinyStrata CMake configuration failed.' }
        $buildArguments = @('--build', $buildRoot, '--config', 'Release', '--parallel', [string]$Parallel)
        if ($Clean) { $buildArguments += '--clean-first' }
        & $tools.CMake @buildArguments | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'TinyStrata native compilation failed.' }
        $exe = Join-Path $buildRoot 'tinystrata.exe'
        $kq = Join-Path $buildRoot 'strata_kq.dll'
        $visionExe = Join-Path $buildRoot 'tinystrata-vision.exe'
        $lfmExe = Join-Path $buildRoot 'tinystrata-lfm2.exe'
        $gemmaExe = Join-Path $buildRoot 'tinystrata-gemma4.exe'
        if (-not (Test-Path -LiteralPath $exe -PathType Leaf) -or ($Backend -eq 'cuda' -and -not (Test-Path -LiteralPath $kq -PathType Leaf)) -or
            -not (Test-Path -LiteralPath $visionExe -PathType Leaf) -or -not (Test-Path -LiteralPath $lfmExe -PathType Leaf) -or
            -not (Test-Path -LiteralPath $gemmaExe -PathType Leaf)) {
            throw 'Expected Qwen, LFM, Gemma or vision build outputs are missing.'
        }
        if ($Backend -eq 'hip') { Install-TinyStrataHipRuntime -SearchDirectories @(Get-TinyStrataHipRuntimeDirectories -Directory $hipRoot) -EngineDirectory $buildRoot -SdkRoot $hipRoot }
        if (-not (Test-TinyStrataVisionBackend -ExecutablePath $visionExe -Backend $Backend)) { throw "The image encoder was not built with $Backend support." }
        foreach ($engine in @($exe, $lfmExe, $gemmaExe)) {
            if (-not (Test-TinyStrataEngineBackend -ExecutablePath $engine -Backend $Backend)) { throw "The engine was not built with $Backend support: $engine" }
        }
        if ($Backend -eq 'hip') { $kq = $null } # HIP MMQ is linked statically, not strata_kq.dll.
        return [pscustomobject]@{ Exe = $exe; LfmExe = $lfmExe; GemmaExe = $gemmaExe; VisionExe = $visionExe; QuantizedKernels = $kq; CudaDirectory = $cudaRoot; HipDirectory = $hipRoot; Backend = $Backend; Architectures = $(if ($Backend -eq 'hip') { $hipArchs } else { $Architectures }) }
    }
    finally {
        foreach ($entry in @(Get-ChildItem Env:)) {
            if (-not $before.ContainsKey($entry.Name)) { [Environment]::SetEnvironmentVariable($entry.Name, $null, 'Process') }
        }
        foreach ($key in $before.Keys) { [Environment]::SetEnvironmentVariable($key, $before[$key], 'Process') }
    }
}

# Functions merged into install.ps1. Shared helper contract:
# Get-VerifiedDownload -Uri <https URL> -Destination <file> -Sha256 <hex>
# returns the verified local file path and writes progress to the host only.

function Expand-TinyStrataRuntimeArchive {
    param(
        [Parameter(Mandatory=$true)][string]$Archive,
        [Parameter(Mandatory=$true)][string]$Destination,
        [string]$IncludePattern = '.*',
        [switch]$Flatten
    )
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Directory]::CreateDirectory($Destination) | Out-Null
    $root = [IO.Path]::GetFullPath($Destination).TrimEnd('\') + '\'
    $zip = [IO.Compression.ZipFile]::OpenRead($Archive)
    try {
        foreach ($entry in $zip.Entries) {
            if (-not $entry.Name -or $entry.FullName -notmatch $IncludePattern) { continue }
            $relative = $entry.FullName.Replace('/', '\')
            if ($Flatten) { $relative = $entry.Name }
            $target = [IO.Path]::GetFullPath((Join-Path $Destination $relative))
            if (-not $target.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Unsafe path in archive: $($entry.FullName)"
            }
            [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target)) | Out-Null
            $inputStream = $entry.Open()
            try {
                $outputStream = [IO.File]::Create($target)
                try { $inputStream.CopyTo($outputStream) } finally { $outputStream.Dispose() }
            } finally { $inputStream.Dispose() }
        }
    } finally { $zip.Dispose() }
}

function Test-TinyStrataPython {
    param([string]$Executable, [switch]$RequirePip)
    if (-not $Executable -or -not (Test-Path -LiteralPath $Executable -PathType Leaf)) { return $false }
    $check = "import struct,sys,venv,ensurepip;assert sys.version_info >= (3,10) and struct.calcsize('P')==8"
    if ($RequirePip) { $check = "import struct,sys,pip;assert sys.version_info >= (3,10) and struct.calcsize('P')==8" }
    try {
        & $Executable -I -B -c $check 2>$null | Out-Null
        return $LASTEXITCODE -eq 0
    } catch { return $false }
}

function Find-TinyStrataPython {
    param([string]$RuntimeRoot)
    $candidates = New-Object System.Collections.Generic.List[string]
    $privatePython = Join-Path $RuntimeRoot 'python-3.13.16\python.exe'
    $candidates.Add($privatePython)
    foreach ($name in @('python.exe', 'python3.exe')) {
        $command = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($command -and $command.Source -notmatch '\\WindowsApps\\') { $candidates.Add($command.Source) }
    }
    $launcher = Get-Command py.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($launcher) {
        try {
            $launchedPython = & $launcher.Source -3 -I -B -c 'import sys;print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0 -and $launchedPython) { $candidates.Add(($launchedPython | Select-Object -Last 1)) }
        } catch { }
    }
    foreach ($registryRoot in @('HKCU:\Software\Python\PythonCore', 'HKLM:\Software\Python\PythonCore')) {
        foreach ($key in @(Get-ChildItem -LiteralPath $registryRoot -ErrorAction SilentlyContinue)) {
            $installKey = Get-Item -LiteralPath ($key.PSPath + '\InstallPath') -ErrorAction SilentlyContinue
            if ($installKey) {
                $registeredPython = $installKey.GetValue('ExecutablePath')
                if (-not $registeredPython) { $registeredPython = Join-Path $installKey.GetValue('') 'python.exe' }
                if ($registeredPython) { $candidates.Add($registeredPython) }
            }
        }
    }
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (Test-TinyStrataPython -Executable $candidate) { return $candidate }
    }
    return $null
}

function Backup-TinyStrataPython {
    param([string]$VenvRoot, [hashtable]$Transaction)
    if ($null -eq $Transaction -or $Transaction.ContainsKey('PythonBackup') -or -not (Test-Path -LiteralPath $VenvRoot -PathType Container)) { return }
    $backup = $VenvRoot + '.rollback-' + [Guid]::NewGuid().ToString('N')
    # Never traverse a linked venv while making a rollback snapshot.
    if (((Get-Item -LiteralPath $VenvRoot -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) -or
        @(Get-ChildItem -LiteralPath $VenvRoot -Recurse -Force | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) {
        throw 'The Python environment contains linked paths; dependency updates were not started.'
    }
    try { Copy-Item -LiteralPath $VenvRoot -Destination $backup -Recurse -Force -ErrorAction Stop }
    catch {
        if (([IO.Path]::GetFullPath($backup)).StartsWith([IO.Path]::GetFullPath($VenvRoot) + '.rollback-', [StringComparison]::OrdinalIgnoreCase) -and (Test-Path -LiteralPath $backup)) {
            Remove-Item -LiteralPath $backup -Recurse -Force -ErrorAction SilentlyContinue
        }
        throw "Could not snapshot the existing Python environment. It was not changed: $backup. $_"
    }
    $Transaction.PythonBackup = $backup
    $Transaction.PythonVenv = [IO.Path]::GetFullPath($VenvRoot)
}

function Complete-TinyStrataPythonTransaction {
    param([hashtable]$Transaction, [switch]$Rollback)
    if ($null -eq $Transaction -or -not $Transaction.ContainsKey('PythonBackup')) { return }
    $venv = $Transaction.PythonVenv
    $backup = [IO.Path]::GetFullPath($Transaction.PythonBackup)
    if (-not $backup.StartsWith($venv + '.rollback-', [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid Python rollback snapshot.' }
    if ($Rollback) {
        $failed = $venv + '.failed-' + [Guid]::NewGuid().ToString('N')
        if (Test-Path -LiteralPath $venv) { [IO.Directory]::Move($venv, $failed) }
        [IO.Directory]::Move($backup, $venv)
        if (Test-Path -LiteralPath $failed) {
            if (-not ([IO.Path]::GetFullPath($failed)).StartsWith($venv + '.failed-', [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid failed Python environment directory.' }
            Remove-Item -LiteralPath $failed -Recurse -Force
        }
    } else { Remove-Item -LiteralPath $backup -Recurse -Force }
    $Transaction.Remove('PythonBackup')
}

function Initialize-TinyStrataPython {
    param(
        [Parameter(Mandatory=$true)][string]$WorkRoot,
        [Parameter(Mandatory=$true)][string]$SourceRoot,
        [Parameter(Mandatory=$true)][string]$CacheRoot,
        [hashtable]$Transaction
    )
    $runtimeRoot = Join-Path $WorkRoot 'runtime'
    $venvRoot = Join-Path $runtimeRoot '.venv'
    $venvPython = Join-Path $venvRoot 'Scripts\python.exe'
    [IO.Directory]::CreateDirectory($runtimeRoot) | Out-Null
    [IO.Directory]::CreateDirectory($CacheRoot) | Out-Null
    if (-not (Test-TinyStrataPython -Executable $venvPython -RequirePip)) {
        $basePython = Find-TinyStrataPython -RuntimeRoot $runtimeRoot
        if (-not $basePython) {
            Write-Host 'Downloading a private Python runtime...'
            # Full regular runtime from Python.org's Windows release manifest.
            # Unlike the embeddable ZIP, this includes venv and ensurepip.
            $archive = Get-VerifiedDownload `
                -Uri 'https://www.python.org/ftp/python/3.13.16/python-3.13.16-amd64.zip' `
                -Destination (Join-Path $CacheRoot 'python-3.13.16-amd64.zip') `
                -Sha256 'bbf675bb5e763c1efbb09a3a461b259d81598a63c30c4b0d7ea11b9f063df159'
            $pythonRoot = Join-Path $runtimeRoot 'python-3.13.16'
            Expand-TinyStrataRuntimeArchive -Archive $archive -Destination $pythonRoot
            $basePython = Join-Path $pythonRoot 'python.exe'
            if (-not (Test-TinyStrataPython -Executable $basePython)) { throw 'The private Python runtime did not start.' }
        }
        Write-Host 'Preparing the private Python environment...'
        Backup-TinyStrataPython -VenvRoot $venvRoot -Transaction $Transaction
        & $basePython -I -B -m venv $venvRoot | Out-Host
        if ($LASTEXITCODE -ne 0 -or -not (Test-TinyStrataPython -Executable $venvPython -RequirePip)) {
            throw 'Could not create the private Python environment.'
        }
    }
    # A healthy environment is reused without pip install or modifications.
    $requirements = Join-Path $SourceRoot 'requirements.txt'
    $requirementCheck = @'
import importlib.metadata, re, sys
for raw in open(sys.argv[1], encoding='utf-8-sig'):
    spec = raw.split('#', 1)[0].strip()
    if not spec:
        continue
    match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s]+)', spec)
    if not match or importlib.metadata.version(match[1]) != match[2]:
        raise SystemExit(1)
'@
    $requirementsHealthy = $false
    try {
        $requirementCheck | & $venvPython -I -B - $requirements 2>$null | Out-Null
        $requirementsHealthy = $LASTEXITCODE -eq 0
        if ($requirementsHealthy) {
            & $venvPython -I -B -m pip --isolated --disable-pip-version-check check 2>$null | Out-Null
            $requirementsHealthy = $LASTEXITCODE -eq 0
        }
    } catch { $requirementsHealthy = $false }
    if ($requirementsHealthy) {
        Write-Host 'Reusing the existing Python environment.'
        return $venvPython
    }
    Write-Host 'Installing API dependencies...'
    Backup-TinyStrataPython -VenvRoot $venvRoot -Transaction $Transaction
    & $venvPython -I -B -m pip --isolated --disable-pip-version-check --no-input install `
        --index-url https://pypi.org/simple --only-binary=:all: `
        -r $requirements | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'API dependency installation failed.' }
    & $venvPython -I -B -m pip --isolated --disable-pip-version-check check | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'API dependency verification failed.' }
    return $venvPython
}

function Get-TinyStrataHipRuntimeFiles {
    param([string[]]$SearchDirectories)
    # Pick one SDK directory, never fill holes with another SDK or System32.
    $source = $null
    $runtimes = @()
    foreach ($directory in $SearchDirectories) {
        $runtimes = @(Get-ChildItem -LiteralPath $directory -Filter 'amdhip64*.dll' -File -ErrorAction SilentlyContinue)
        if ($runtimes.Count) { $source = $directory; break }
    }
    if (-not $source) { throw 'The selected AMD SDK has no HIP runtime DLL.' }
    $comgr = @(Get-ChildItem -LiteralPath $source -Filter 'amd_comgr*.dll' -File)
    if (-not $comgr.Count) { throw "The selected AMD SDK is missing its matching COMGR DLL: $source" }
    $files = @($runtimes) + @($comgr)
    # Import tables and dynamic-load names both contain the DLL basename. Check
    # ASCII and UTF-16 references without loading code or initializing a GPU.
    # Older SDKs predate kpack; require it only when referenced, but include it
    # when present. Likewise do not substitute a differently named COMGR DLL.
    $references = @()
    foreach ($runtime in $runtimes) {
        $bytes = [IO.File]::ReadAllBytes($runtime.FullName)
        foreach ($encoding in @([Text.Encoding]::ASCII, [Text.Encoding]::Unicode)) {
            $references += @([regex]::Matches($encoding.GetString($bytes), '(?i)(?:rocm_kpack|amd_comgr[0-9_]*)\.dll') | ForEach-Object Value)
        }
    }
    foreach ($name in @($references | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath (Join-Path $source $name) -PathType Leaf)) {
            throw "The selected AMD SDK is missing HIP runtime dependency $name in $source. No HIP DLLs were copied."
        }
    }
    $kpack = Join-Path $source 'rocm_kpack.dll'
    if (Test-Path -LiteralPath $kpack -PathType Leaf) { $files += Get-Item -LiteralPath $kpack }
    return @($files | Sort-Object Name -Unique)
}

function Get-TinyStrataHipImports {
    param([string]$ExecutablePath)
    # Read PE import descriptors only: never load an executable or its DLLs.
    $bytes = [IO.File]::ReadAllBytes($ExecutablePath)
    if ($bytes.Length -lt 64 -or [BitConverter]::ToUInt16($bytes, 0) -ne 0x5a4d) { return @() }
    try {
        $pe = [BitConverter]::ToInt32($bytes, 60)
        if ($pe -lt 0 -or $pe + 24 -gt $bytes.Length -or [BitConverter]::ToUInt32($bytes, $pe) -ne 0x4550) { throw 'invalid PE signature' }
        $sections = [BitConverter]::ToUInt16($bytes, $pe + 6)
        $optionalSize = [BitConverter]::ToUInt16($bytes, $pe + 20)
        $optional = $pe + 24
        $magic = [BitConverter]::ToUInt16($bytes, $optional)
        $directory = if ($magic -eq 0x20b) { 112 } elseif ($magic -eq 0x10b) { 96 } else { throw 'invalid optional header' }
        $sectionTable = $optional + $optionalSize
        if ($optionalSize -lt $directory + 16 -or $sections -gt 96 -or $sectionTable + 40 * $sections -gt $bytes.Length) { throw 'invalid PE headers' }
        $importRva = [BitConverter]::ToUInt32($bytes, $optional + $directory + 8)
        if (-not $importRva) { return @() }
        $offsetOf = {
            param([uint32]$Rva)
            for ($index = 0; $index -lt $sections; $index++) {
                $section = $sectionTable + 40 * $index
                $address = [BitConverter]::ToUInt32($bytes, $section + 12)
                $size = [BitConverter]::ToUInt32($bytes, $section + 16)
                $raw = [BitConverter]::ToUInt32($bytes, $section + 20)
                if ($Rva -ge $address -and $Rva -lt [long]$address + $size) {
                    $result = [long]$raw + $Rva - $address
                    if ($result -ge $bytes.Length) { throw 'invalid section data' }
                    return [int]$result
                }
            }
            throw 'unmapped import RVA'
        }
        $descriptor = & $offsetOf $importRva
        for ($count = 0; $count -lt 4096; $count++) {
            if ($descriptor + 20 -gt $bytes.Length) { throw 'truncated import table' }
            $nameRva = [BitConverter]::ToUInt32($bytes, $descriptor + 12)
            if (-not $nameRva) { return }
            $nameOffset = & $offsetOf $nameRva
            $end = [Array]::IndexOf($bytes, [byte]0, $nameOffset)
            if ($end -lt $nameOffset -or $end - $nameOffset -gt 256) { throw 'invalid import name' }
            $name = [Text.Encoding]::ASCII.GetString($bytes, $nameOffset, $end - $nameOffset)
            if ($name -match '^amdhip64(?:_[0-9]+)?\.dll$') { $name }
            $descriptor += 20
        }
        throw 'unterminated import table'
    } catch { throw "Cannot inspect HIP imports in $ExecutablePath : $_" }
}

function Install-TinyStrataHipRuntime {
    param([string[]]$SearchDirectories, [Parameter(Mandatory=$true)][string]$EngineDirectory, [string]$SdkRoot)
    # Windows searches the executable directory before System32, then PATH.
    # AMD's Windows llama.cpp guide requires this matching three-DLL set:
    # https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/llamacpp.html#copy-the-hip-runtime-libraries
    $files = @(Get-TinyStrataHipRuntimeFiles -SearchDirectories $SearchDirectories)
    $destination = [IO.Path]::GetFullPath($EngineDirectory).TrimEnd('\')
    $windows = [IO.Path]::GetFullPath([Environment]::GetEnvironmentVariable('SystemRoot')).TrimEnd('\')
    if ($destination -eq $windows -or $destination.StartsWith($windows + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'HIP runtime DLLs must be installed beside TinyStrata, outside the Windows directory.'
    }
    $runtimeImports = @()
    foreach ($executable in @(Get-ChildItem -LiteralPath $destination -Filter 'tinystrata*.exe' -File -ErrorAction SilentlyContinue)) {
        foreach ($name in @(Get-TinyStrataHipImports -ExecutablePath $executable.FullName)) {
            $runtimeImports += $name
            if ($files.Name -notcontains $name) {
                throw "The selected AMD SDK does not provide $name required by $($executable.Name). Rebuild TinyStrata with this SDK; an old app-local or system DLL will not be substituted."
            }
        }
    }
    $runtimeImports = @($runtimeImports | Select-Object -Unique)
    if ($runtimeImports.Count -gt 1) { throw 'The native engines require different HIP runtimes. Rebuild them with one SDK.' }
    $runtimeName = if ($runtimeImports.Count) { $runtimeImports[0] } else { @($files | Where-Object { $_.Name -like 'amdhip64*.dll' })[0].Name }
    $dependencies = @()
    foreach ($dependency in @('hipblas.dll', 'rocblas.dll', 'rocsolver.dll', 'libhipblaslt.dll', 'libtensilelite-host.dll')) {
        foreach ($directory in $SearchDirectories) {
            $path = Join-Path $directory $dependency
            if (Test-Path -LiteralPath $path -PathType Leaf) {
                $dependencies += [ordered]@{ path = [IO.Path]::GetFullPath($path); sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }
                break
            }
        }
    }
    if (@($dependencies | Where-Object { [IO.Path]::GetFileName($_.path) -eq 'hipblas.dll' }).Count -ne 1 -or
        @($dependencies | Where-Object { [IO.Path]::GetFileName($_.path) -eq 'rocblas.dll' }).Count -ne 1) {
        throw 'The selected AMD SDK is missing matching HIP/rocBLAS math libraries. No runtime DLLs were copied.'
    }
    $manifest = [ordered]@{
        schema = 1; runtime_dll = $runtimeName
        lib_dirs = @($SearchDirectories | ForEach-Object { [IO.Path]::GetFullPath($_) } | Select-Object -Unique)
        files = @($files | ForEach-Object { [ordered]@{ name = $_.Name; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() } })
        dependencies = $dependencies
    }
    if ($SdkRoot) {
        if (-not (Test-Path -LiteralPath $SdkRoot -PathType Container)) { throw 'The selected HIP SDK root is missing.' }
        $manifest['sdk_root'] = [IO.Path]::GetFullPath($SdkRoot)
    }
    $manifestBytes = (New-Object Text.UTF8Encoding($false)).GetBytes(($manifest | ConvertTo-Json -Depth 5))
    $payloads = @($files | ForEach-Object { [pscustomobject]@{ Name = $_.Name; Source = $_.FullName; Bytes = $null } })
    $payloads += [pscustomobject]@{ Name = '.tinystrata-hip-runtime.json'; Source = $null; Bytes = $manifestBytes }

    # Native CLI guard uses this fixed-format proof after checking the actual
    # loaded app-local runtime/version and all three hashes independently.
    # Publish it in the same rollback transaction as the JSON and core DLLs.
    $proofBytes = [byte[]]@()
    $runtimeSource = @($files | Where-Object { $_.Name -eq $runtimeName })[0]
    $comgrFiles = @($files | Where-Object { $_.Name -match '^amd_comgr(?:_[0-9]+)?\.dll$' })
    $runtimeBytes = [IO.File]::ReadAllBytes($runtimeSource.FullName)
    $references = @()
    foreach ($encoding in @([Text.Encoding]::ASCII, [Text.Encoding]::Unicode)) {
        $references += @([regex]::Matches($encoding.GetString($runtimeBytes), '(?i)amd_comgr(?:_[0-9]+)?\.dll') | ForEach-Object Value)
    }
    $selectedComgr = @($comgrFiles | Where-Object { $references -contains $_.Name })
    if ($selectedComgr.Count -eq 0 -and $comgrFiles.Count -eq 1) { $selectedComgr = $comgrFiles }
    $kpackFiles = @($files | Where-Object { $_.Name -eq 'rocm_kpack.dll' })
    if ($selectedComgr.Count -eq 1 -and $kpackFiles.Count -eq 1) {
        $proofFiles = @($runtimeSource, $selectedComgr[0], $kpackFiles[0]) | Sort-Object Name
        $proofText = "TINYSTRATA_HIP_VMM_V1`n"
        foreach ($file in $proofFiles) {
            $proofText += $file.Name.ToLowerInvariant() + ' ' + (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant() + "`n"
        }
        $proofBytes = [Text.Encoding]::ASCII.GetBytes($proofText)
    }
    # An older SDK writes an empty proof, invalidating any prior small-grain
    # grant without blocking its verified/coarse runtime compatibility path.
    $payloads += [pscustomobject]@{ Name = '.tinystrata-hip-vmm.txt'; Source = $null; Bytes = $proofBytes }

    $changes = @()
    foreach ($file in $payloads) {
        $target = Join-Path $destination $file.Name
        if (Test-Path -LiteralPath $target -PathType Leaf) {
            if ($null -ne $file.Bytes) { $same = [Convert]::ToBase64String([IO.File]::ReadAllBytes($target)) -eq [Convert]::ToBase64String($file.Bytes) }
            else { $same = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -eq (Get-FileHash -LiteralPath $file.Source -Algorithm SHA256).Hash }
            if ($same) { continue }
        }
        if ((Test-Path -LiteralPath $target) -and
            ((Get-Item -LiteralPath $target).PSIsContainer -or ((Get-Item -LiteralPath $target).Attributes -band [IO.FileAttributes]::ReadOnly))) {
            throw "Cannot install the selected AMD runtime: $target is not a writable DLL file. Choose a writable TinyStrata engine directory."
        }
        $changes += [pscustomobject]@{ Source = $file.Source; Bytes = $file.Bytes; Target = $target; Name = $file.Name; Existed = (Test-Path -LiteralPath $target -PathType Leaf) }
    }
    if (-not $changes.Count) { return }
    $stage = Join-Path $destination ('.tinystrata-hip-' + [Guid]::NewGuid().ToString('N'))
    $installed = @()
    $keepBackup = $false
    try {
        [IO.Directory]::CreateDirectory($stage) | Out-Null
        # Read/validate the complete source set before replacing any target.
        foreach ($change in $changes) {
            $staged = Join-Path $stage $change.Name
            if ($null -ne $change.Bytes) { [IO.File]::WriteAllBytes($staged, $change.Bytes) }
            else { Copy-Item -LiteralPath $change.Source -Destination $staged -ErrorAction Stop }
        }
        # SDK updates or a failed copy must not publish a mixed generation.
        # Compare staged bytes to the manifest captured before any replacement.
        foreach ($entry in $manifest.files) {
            $staged = Join-Path $stage $entry.name
            $checked = if (Test-Path -LiteralPath $staged -PathType Leaf) { $staged } else { Join-Path $destination $entry.name }
            if ((Get-FileHash -LiteralPath $checked -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) {
                throw "The selected HIP SDK changed while preparing $($entry.name). No runtime DLLs were published."
            }
        }
        foreach ($entry in $dependencies) {
            if ((Get-FileHash -LiteralPath $entry.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) {
                throw "The selected HIP math SDK changed while preparing $($entry.path). No runtime DLLs were published."
            }
        }
        foreach ($change in $changes) {
            $staged = Join-Path $stage $change.Name
            if ($change.Existed) { [IO.File]::Replace($staged, $change.Target, (Join-Path $stage ($change.Name + '.old'))) }
            else { [IO.File]::Move($staged, $change.Target) }
            $installed += $change
        }
    } catch {
        $failure = $_
        foreach ($change in @($installed | Sort-Object Name -Descending)) {
            try {
                if ($change.Existed) { [IO.File]::Replace((Join-Path $stage ($change.Name + '.old')), $change.Target, (Join-Path $stage ($change.Name + '.failed'))) }
                else { [IO.File]::Delete($change.Target) }
            } catch { $keepBackup = $true }
        }
        $extra = if ($keepBackup) { " Recovery files remain in $stage." } else { ' Existing HIP DLLs were left unchanged.' }
        throw "Cannot install the selected AMD runtime beside TinyStrata. The engine directory must be writable and its DLLs must not be in use: $destination. $failure$extra"
    } finally {
        # This generated staging directory is confined to the named engine directory.
        if (-not $keepBackup -and (Test-Path -LiteralPath $stage)) {
            if (-not ([IO.Path]::GetFullPath($stage)).StartsWith($destination + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid HIP staging directory.' }
            Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction Stop
        }
    }
}

function Initialize-TinyStrataRuntime {
    param(
        [Parameter(Mandatory=$true)][string]$RuntimeRoot,
        [Parameter(Mandatory=$true)][string]$CacheRoot,
        [string[]]$SearchDirectories = @(),
        [string[]]$HipRuntimeDirectories = @(),
        [string]$HipSdkRoot,
        [string]$EngineDirectory,
        [ValidateSet('cuda', 'hip')][string]$Backend = 'cuda'
    )
    [IO.Directory]::CreateDirectory($RuntimeRoot) | Out-Null
    [IO.Directory]::CreateDirectory($CacheRoot) | Out-Null
    $cudaDirectories = New-Object System.Collections.Generic.List[string]
    $search = @($SearchDirectories | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Container) } | Select-Object -Unique)
    # Only the deliberately selected SDK can provide HIP binaries. Loader
    # fallback directories may already contain a previous app-local runtime.
    $runtimeSearch = $search
    if ($Backend -eq 'hip') {
        $runtimeSearch = @($HipRuntimeDirectories | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Container) } | Select-Object -Unique)
        if (-not $runtimeSearch.Count) { throw 'The selected AMD SDK runtime directories are required.' }
    }
    $haveCuda = $true
    foreach ($dll in $(if ($Backend -eq 'hip') { @('amdhip64*.dll', 'hipblas.dll', 'rocblas.dll') } else { @('cudart64_13.dll', 'cublas64_13.dll', 'cublasLt64_13.dll') })) {
        $found = $null
        foreach ($directory in $runtimeSearch) {
            $candidate = @(Get-ChildItem -LiteralPath $directory -Filter $dll -File -ErrorAction SilentlyContinue)
            if ($candidate.Count -gt 0) { $found = $directory; break }
        }
        if (-not $found) { $haveCuda = $false; break }
        $cudaDirectories.Add($found)
    }
    if ($Backend -eq 'hip' -and -not $haveCuda) { throw 'The selected AMD SDK is missing HIP/hipBLAS/rocBLAS runtime DLLs in its bin/lib folders.' }
    if ($Backend -eq 'hip') {
        if (-not $EngineDirectory) { throw 'A TinyStrata engine directory is required to select the AMD runtime on Windows.' }
        Install-TinyStrataHipRuntime -SearchDirectories $runtimeSearch -EngineDirectory $EngineDirectory -SdkRoot $HipSdkRoot
    }
    # Keep already installed, sufficiently recent CRT libraries. Older system
    # copies must not override newer MSVC exports required by the engine.
    $systemDirectory = [Environment]::SystemDirectory
    $haveVc = $true
    foreach ($dll in @('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'vcomp140.dll')) {
        $candidate = Join-Path $systemDirectory $dll
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { $haveVc = $false; break }
        $info = [Diagnostics.FileVersionInfo]::GetVersionInfo($candidate)
        if ($info.FileMajorPart -lt 14 -or ($info.FileMajorPart -eq 14 -and $info.FileMinorPart -lt 51)) {
            $haveVc = $false; break
        }
    }
    # Pinned official CUDA 13.2.2 redistributables. Only runtime DLLs are
    # extracted; this does not install the full toolkit or NVIDIA driver.
    $packages = @(
        @{
            Name = 'cuda_cudart-13.2.2.zip'
            Uri = 'https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-13.2.86-archive.zip'
            Sha256 = '024e0c1055343d9d2a7d35c90fb7695db14cc2f05bb962198eb18b92c73b32be'
            Pattern = '/bin/(x64/)?[^/]+\.dll$'
        },
        @{
            Name = 'libcublas-13.2.2.zip'
            Uri = 'https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-13.4.1.3-archive.zip'
            Sha256 = 'bc6f590490134f4dbe29672b72e536c44362cc5f738a0f7e48c8c3c141e9b24a'
            Pattern = '/bin/(x64/)?[^/]+\.dll$'
        },
        @{
            Name = 'vc-runtime-14.52.36815.vsix'
            Uri = 'https://download.visualstudio.microsoft.com/download/pr/f78411dd-a322-4033-97d4-6a65512ee337/3a1c232793a80f6d794b21847e1dc90df1ca3db005a5f70548b510249bd2ca58/Microsoft.VC.14.52.CRT.Redist.X64.base.vsix'
            Sha256 = '3a1c232793a80f6d794b21847e1dc90df1ca3db005a5f70548b510249bd2ca58'
            Pattern = '/x64/Microsoft\.VC[0-9]+\.(CRT|OpenMP)/[^/]+\.dll$'
        }
    )
    foreach ($package in $packages) {
        if ($package.Name -like 'vc-runtime*' -and $haveVc) { continue }
        if ($package.Name -notlike 'vc-runtime*' -and $haveCuda) { continue }
        Write-Host ('Preparing runtime: ' + $package.Name)
        $archive = Get-VerifiedDownload -Uri $package.Uri `
            -Destination (Join-Path $CacheRoot $package.Name) -Sha256 $package.Sha256
        Expand-TinyStrataRuntimeArchive -Archive $archive -Destination $RuntimeRoot `
            -IncludePattern $package.Pattern -Flatten
    }
    $required = @()
    if (-not $haveCuda) { $required += @('cudart64_13.dll', 'cublas64_13.dll', 'cublasLt64_13.dll') }
    if (-not $haveVc) { $required += @('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'vcomp140.dll') }
    foreach ($requiredDll in $required) {
        if (-not (Test-Path -LiteralPath (Join-Path $RuntimeRoot $requiredDll) -PathType Leaf)) {
            throw "Runtime package is missing $requiredDll."
        }
    }
    if (-not $haveVc -and $EngineDirectory) {
        [IO.Directory]::CreateDirectory($EngineDirectory) | Out-Null
        # App-local CRTs override older System32 versions. PATH alone does not.
        $vcNames = @('msvcp140*.dll', 'vcruntime140*.dll', 'vcomp140.dll', 'concrt140.dll', 'vccorlib140.dll')
        foreach ($pattern in $vcNames) {
            foreach ($file in @(Get-ChildItem -LiteralPath $RuntimeRoot -Filter $pattern -File)) {
                $target = Join-Path $EngineDirectory $file.Name
                if ([IO.Path]::GetFullPath($target) -eq [IO.Path]::GetFullPath($file.FullName)) { continue }
                Copy-Item -LiteralPath $file.FullName -Destination $target -Force
            }
        }
    }
    $directories = @()
    if ($haveCuda) { $directories += @($cudaDirectories | Select-Object -Unique) }
    if ($Backend -eq 'hip') { $directories += $search }
    if (-not $haveCuda -or -not $haveVc) { $directories += $RuntimeRoot }
    return @($directories | Select-Object -Unique)
}

function Get-TinyStrataConfig {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path -PathType Leaf) { return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json }
    return $null
}

function Get-TinyStrataSavedProcess {
    param([string]$StatePath, [string]$ConfigPath, [string]$SourcePath, [int]$ApiPort)
    if (-not (Test-Path -LiteralPath $StatePath -PathType Leaf)) { return $null }
    try {
        $state = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
        if ($state.config -ne $ConfigPath -or $state.source -ne $SourcePath -or $state.port -ne $ApiPort) { return $null }
        $process = Get-Process -Id $state.pid -ErrorAction Stop
        # PowerShell 7.5+ decodes ISO JSON dates as DateTime; 5.1 keeps strings.
        $started = if ($state.started -is [DateTime]) { $state.started.ToUniversalTime().ToString('o') } else { [string]$state.started }
        if ($process.StartTime.ToUniversalTime().ToString('o') -ne $started) { return $null }
        $details = Get-CimInstance Win32_Process -Filter ("ProcessId={0}" -f $state.pid)
        if (-not $details.CommandLine -or $details.CommandLine.IndexOf($ConfigPath, [StringComparison]::OrdinalIgnoreCase) -lt 0 -or
            $details.CommandLine -notmatch 'serve\.server') { return $null }
        return $process
    } catch { return $null }
}

function Test-TinyStrataPortOwner {
    param([int]$ApiPort, [int]$RootProcessId)
    $listeners = @(Get-NetTCPConnection -LocalPort $ApiPort -State Listen -ErrorAction SilentlyContinue)
    if ($listeners.Count -eq 0) { return $false }
    foreach ($listener in $listeners) {
        $owner = [int]$listener.OwningProcess
        $matched = $false
        for ($depth = 0; $depth -lt 8 -and $owner -gt 0; $depth++) {
            if ($owner -eq $RootProcessId) { $matched = $true; break }
            $details = Get-CimInstance Win32_Process -Filter ("ProcessId={0}" -f $owner)
            if (-not $details) { break }
            $owner = [int]$details.ParentProcessId
        }
        if (-not $matched) { return $false }
    }
    return $true
}

function Invoke-TinyStrataApi {
    param([string]$BaseUrl, [string]$Path, [string]$Method = 'GET', [hashtable]$Headers = @{})
    $request = @{ Uri = $BaseUrl + $Path; Method = $Method; Headers = $Headers; TimeoutSec = 5; UseBasicParsing = $true; ErrorAction = 'Stop' }
    if ($Method -eq 'POST') { $request.ContentType = 'application/json'; $request.Body = '{}' }
    return Invoke-RestMethod @request
}

function Test-TinyStrataInferenceReady {
    param($Health, $Configuration)
    $requireImages = -not ($Configuration -and ($Configuration.vision_mode -eq 'no' -or $Configuration.arch -eq 'lfm2moe'))
    return [bool]($Health.loaded -and (-not $requireImages -or $Health.images))
}

function Get-TinyStrataModelFamily {
    param([string]$Name = 'qwen36')
    switch ($Name) {
        { $_ -in @('qwen36', 'qwen3.6') } {
            return [pscustomobject]@{ Name = 'qwen36'; ConfigFile = 'config.json'; StateFile = 'server-state.json'; NativeFile = 'tinystrata.exe'; Port = 18080 }
        }
        { $_ -in @('lfm2', 'lfm2.5') } {
            return [pscustomobject]@{ Name = 'lfm2'; ConfigFile = 'config-lfm2.json'; StateFile = 'server-state-lfm2.json'; NativeFile = 'tinystrata-lfm2.exe'; Port = 18081 }
        }
        { $_ -in @('gemma4', 'gemma-4') } {
            return [pscustomobject]@{ Name = 'gemma4'; ConfigFile = 'config-gemma4.json'; StateFile = 'server-state-gemma4.json'; NativeFile = 'tinystrata-gemma4.exe'; Port = 18082 }
        }
        'clover' {
            return [pscustomobject]@{ Name = 'clover'; ConfigFile = 'config-clover.json'; StateFile = 'server-state-clover.json'; NativeFile = 'tinystrata.exe'; Port = 18083 }
        }
        { $_ -in @('tiel', 'tiel-coder') } {
            return [pscustomobject]@{ Name = 'tiel'; ConfigFile = 'config-tiel-coder.json'; StateFile = 'server-state-tiel-coder.json'; NativeFile = 'tinystrata.exe'; Port = 18084 }
        }
        default { throw "Unsupported TinyStrata model family: $Name" }
    }
}

function Initialize-TinyStrataUiSource {
    param([string]$SourceRoot, [string]$WorkRoot)
    $uiRoot = Join-Path $SourceRoot 'ui'
    $runner = Join-Path $SourceRoot 'run-ui.bat'
    $required = @('ui\launch.py', 'ui\server.py', 'ui\requirements.txt', 'ui\web\index.html', 'run-ui.bat')
    $complete = $true
    foreach ($path in $required) {
        if (-not (Test-Path -LiteralPath (Join-Path $SourceRoot $path) -PathType Leaf)) { $complete = $false; break }
    }
    if ($complete) { return }
    if ((Test-Path -LiteralPath (Join-Path $SourceRoot '.git')) -or
        ((Get-Item -LiteralPath $SourceRoot -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw 'The optional UI is missing from this Git checkout or linked source directory. Update the checkout first; the API was left unchanged.'
    }
    if (Test-Path -LiteralPath $uiRoot) {
        $existingUiPaths = @(Get-Item -LiteralPath $uiRoot -Force) + @(Get-ChildItem -LiteralPath $uiRoot -Recurse -Force)
        if (@($existingUiPaths | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count -gt 0) {
            throw 'The existing UI contains a linked path; it was left unchanged.'
        }
    }
    if ((Test-Path -LiteralPath $runner) -and ((Get-Item -LiteralPath $runner -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw 'The UI launcher is a linked path; it was left unchanged.'
    }
    $cacheRoot = Join-Path $WorkRoot 'installer'
    $commit = Invoke-RestMethod -Uri 'https://api.github.com/repos/kbc20f11/TinyStrata/commits/main' `
        -Headers @{ 'User-Agent' = 'TinyStrata-UI-Installer'; Accept = 'application/vnd.github+json' } -TimeoutSec 30
    $revision = [string]$commit.sha
    if ($revision -notmatch '^[a-f0-9]{40}$') { throw 'GitHub returned an invalid TinyStrata revision.' }
    $archive = Join-Path $cacheRoot ("TinyStrata-$revision.zip")
    Write-Host "Preparing the optional UI ($($revision.Substring(0, 7)))..."
    Get-TinyStrataOfficialFile -Uri ("https://codeload.github.com/kbc20f11/TinyStrata/zip/$revision") -Destination $archive
    $staging = Join-Path $cacheRoot ('ui-source-' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $staging -Force | Out-Null
    $unpacked = Join-Path $staging ("TinyStrata-$revision")
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($archive)
    try {
        foreach ($entry in $zip.Entries) {
            if ($entry.FullName -match ':') { throw 'GitHub source archive contains an unexpected path.' }
            $target = [IO.Path]::GetFullPath((Join-Path $staging $entry.FullName.Replace('/', '\')))
            if (-not $entry.Name -and $target.TrimEnd('\') -eq $unpacked) { continue }
            if (-not $target.StartsWith($unpacked + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'GitHub source archive contains an unexpected path.' }
            $relative = $target.Substring($unpacked.Length + 1)
            if (-not $entry.Name -or ($relative -ne 'run-ui.bat' -and -not $relative.StartsWith('ui\', [StringComparison]::OrdinalIgnoreCase))) { continue }
            New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $false)
        }
    } finally { $zip.Dispose() }
    foreach ($path in $required) {
        if (-not (Test-Path -LiteralPath (Join-Path $unpacked $path) -PathType Leaf)) { throw 'Downloaded TinyStrata UI sources are incomplete.' }
    }
    # Add only the optional UI. API source, binaries, settings and model files
    # stay in place, so an already running API does not need to restart.
    $backup = Join-Path $staging 'previous-ui'
    New-Item -ItemType Directory -Path $backup | Out-Null
    foreach ($existing in @($uiRoot, $runner)) {
        if (Test-Path -LiteralPath $existing) { Copy-Item -LiteralPath $existing -Destination $backup -Recurse -Force }
    }
    Copy-Item -LiteralPath (Join-Path $unpacked 'ui') -Destination $SourceRoot -Recurse -Force
    Copy-Item -LiteralPath (Join-Path $unpacked 'run-ui.bat') -Destination $runner -Force
}

function Initialize-TinyStrataUiDependencies {
    param([string]$SourceRoot, [string]$WorkRoot, [string]$Python)
    $requirements = Join-Path $SourceRoot 'ui\requirements.txt'
    $venvRoot = Join-Path $WorkRoot 'runtime\.venv'
    if (-not (Test-Path -LiteralPath $requirements -PathType Leaf)) { throw 'Optional UI requirements are missing. Update the UI sources first.' }
    # Never direct pip at a global interpreter, even if an existing runtime path
    # was replaced or linked. pip also gets its own --require-virtualenv guard.
    $privateCheck = 'from pathlib import Path;import sys;assert sys.prefix != sys.base_prefix and Path(sys.prefix).resolve()==Path(sys.argv[1]).resolve()'
    $privateReady = $false
    try {
        & $Python -I -B -c $privateCheck $venvRoot 2>$null | Out-Null
        $privateReady = $LASTEXITCODE -eq 0
    } catch { $privateReady = $false }
    if (-not $privateReady) { throw 'Optional UI dependencies require this installation''s private Python virtual environment; no packages were installed.' }
    $requirementCheck = @'
import importlib.metadata, re, sys
for raw in open(sys.argv[1], encoding='utf-8-sig'):
    spec = raw.split('#', 1)[0].strip()
    if not spec:
        continue
    match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s]+)', spec)
    if not match or importlib.metadata.version(match[1]) != match[2]:
        raise SystemExit(1)
'@
    # DDGS exposes a lazy proxy from its package root. Import the implementation
    # as well: metadata/pip check alone cannot detect missing package files or a
    # dependency which cannot actually load in this private interpreter.
    $importCheck = 'from ddgs.ddgs import DDGS; assert callable(DDGS)'
    $healthy = $false
    $repairUiPackages = $false
    try {
        $requirementCheck | & $Python -I -B - $requirements 2>$null | Out-Null
        $healthy = $LASTEXITCODE -eq 0
        if ($healthy) {
            & $Python -I -B -m pip --isolated --require-virtualenv --disable-pip-version-check check 2>$null | Out-Null
            $healthy = $LASTEXITCODE -eq 0
        }
        if ($healthy) {
            $repairUiPackages = $true # Also retain repair intent if native stderr throws in Windows PowerShell.
            & $Python -I -B -c $importCheck 2>$null | Out-Null
            $healthy = $LASTEXITCODE -eq 0
            $repairUiPackages = -not $healthy
        }
    } catch { $healthy = $false }
    if ($healthy) { Write-Host 'Reusing the optional UI dependencies.'; return }
    Write-Host 'Installing optional UI search dependencies in the private environment...'
    $installArguments = @('-I', '-B', '-m', 'pip', '--isolated', '--require-virtualenv', '--disable-pip-version-check', '--no-input', 'install',
                          '--index-url', 'https://pypi.org/simple', '--only-binary=:all:')
    if ($repairUiPackages) { $installArguments += '--force-reinstall' }
    $installArguments += @('-r', $requirements)
    & $Python @installArguments | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'Optional UI dependency installation failed. The API was left unchanged.' }
    & $Python -I -B -m pip --isolated --require-virtualenv --disable-pip-version-check check | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'Optional UI dependency verification failed. The API was left unchanged.' }
    $requirementCheck | & $Python -I -B - $requirements | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'Optional UI dependency version verification failed. The API was left unchanged.' }
    & $Python -I -B -c $importCheck | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "DDGS could not be loaded by the UI's private Python: $Python. The UI was not started; inspect the import error above." }
}

function Initialize-TinyStrataDesktopShortcut {
    param([string]$SourceRoot, [string]$WorkRoot,
          [string]$DesktopDirectory = [Environment]::GetFolderPath('DesktopDirectory'))
    # Resolve the user's actual shell Desktop, including OneDrive/redirection;
    # never assume that it is a directory named Desktop under USERPROFILE.
    if (-not $DesktopDirectory) { throw 'The current user desktop directory could not be found.' }
    $source = [IO.Path]::GetFullPath($SourceRoot).TrimEnd('\')
    $work = [IO.Path]::GetFullPath($WorkRoot).TrimEnd('\')
    $stateRoot = Join-Path $work 'runtime\ui'
    $helperPath = Join-Path $stateRoot 'desktop-launch.ps1'
    $shortcutPath = Join-Path $DesktopDirectory 'TinyStrata.lnk'
    foreach ($path in @($helperPath, $shortcutPath)) {
        if (Test-Path -LiteralPath $path) {
            $item = Get-Item -LiteralPath $path -Force
            if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                throw "The shortcut destination is linked or is a directory; it was left unchanged: $path"
            }
        }
    }
    [IO.Directory]::CreateDirectory($stateRoot) | Out-Null
    [IO.Directory]::CreateDirectory($DesktopDirectory) | Out-Null
    # Keep this launcher outside the source/update directory. No model or port
    # is pinned here: ui/launch.py reads the saved UI selections on each click.
    $helperText = ('$tinyStrataSource = ''{0}''' -f $source.Replace("'", "''")) + "`r`n" +
                  ('$tinyStrataWork = ''{0}''' -f $work.Replace("'", "''")) + "`r`n"
    $helperText += @'
$ErrorActionPreference = 'Stop'
$tinyStrataState = Join-Path $tinyStrataWork 'runtime\ui'
$tinyStrataStdout = Join-Path $tinyStrataState 'desktop-launch.stdout.log'
$tinyStrataStderr = Join-Path $tinyStrataState 'desktop-launch.stderr.log'
function ConvertTo-TinyStrataWindowsArgument {
    param([string]$Value)
    # Quote for CreateProcess/CommandLineToArgvW, not for another shell.
    return '"' + [regex]::Replace([regex]::Replace($Value, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
}
try {
    $tinyStrataPython = Join-Path $tinyStrataWork 'runtime\.venv\Scripts\python.exe'
    $tinyStrataLauncher = Join-Path $tinyStrataSource 'ui\launch.py'
    if (-not (Test-Path -LiteralPath $tinyStrataPython -PathType Leaf) -or
        -not (Test-Path -LiteralPath $tinyStrataLauncher -PathType Leaf)) {
        throw 'TinyStrata files are missing. Run the UI installer again to repair the installation.'
    }
    $tinyStrataArguments = @('-B', $tinyStrataLauncher, '--source-dir', $tinyStrataSource, '--work-dir', $tinyStrataWork)
    $tinyStrataCommandLine = ($tinyStrataArguments | ForEach-Object { ConvertTo-TinyStrataWindowsArgument $_ }) -join ' '
    $tinyStrataProcess = Start-Process -FilePath $tinyStrataPython -ArgumentList $tinyStrataCommandLine `
        -WorkingDirectory $tinyStrataSource -WindowStyle Hidden -RedirectStandardOutput $tinyStrataStdout `
        -RedirectStandardError $tinyStrataStderr -Wait -PassThru
    if ($tinyStrataProcess.ExitCode -ne 0) { throw "TinyStrata launcher exited with code $($tinyStrataProcess.ExitCode)." }
} catch {
    ($_ | Out-String) | Add-Content -LiteralPath $tinyStrataStderr -Encoding UTF8
    try {
        Add-Type -AssemblyName System.Windows.Forms
        [void][System.Windows.Forms.MessageBox]::Show("TinyStrata could not start.`n`nSee: $tinyStrataStderr", 'TinyStrata', 'OK', 'Error')
    } catch { }
    exit 1
}
exit 0
'@
    # Windows PowerShell 5.1 needs the BOM to read Unicode installation paths.
    $temporary = $helperPath + '.' + [Guid]::NewGuid().ToString('N') + '.tmp'
    [IO.File]::WriteAllText($temporary, $helperText, (New-Object Text.UTF8Encoding($true)))
    Move-Item -LiteralPath $temporary -Destination $helperPath -Force
    $shell = New-Object -ComObject WScript.Shell
    # Save a complete link before replacing our named shortcut. An interrupted
    # COM save must not truncate an existing working link or touch other files.
    $shortcutTemporary = Join-Path $DesktopDirectory ('.TinyStrata-' + [Guid]::NewGuid().ToString('N') + '.lnk')
    $shortcut = $shell.CreateShortcut($shortcutTemporary)
    $shortcut.TargetPath = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $shortcut.Arguments = '-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}"' -f $helperPath
    $shortcut.WorkingDirectory = $source
    $shortcut.Description = 'Start TinyStrata and open the chat UI'
    $shortcut.WindowStyle = 7
    $icon = Join-Path $source 'ui\web\terminal-logo.ico'
    $shortcut.IconLocation = if (Test-Path -LiteralPath $icon -PathType Leaf) { $icon + ',0' }
                             else { (Join-Path $env:WINDIR 'System32\shell32.dll') + ',1' }
    try {
        $shortcut.Save()
        if (-not (Test-Path -LiteralPath $shortcutTemporary -PathType Leaf)) {
            throw "Windows did not save the desktop shortcut in $DesktopDirectory."
        }
        Move-Item -LiteralPath $shortcutTemporary -Destination $shortcutPath -Force -ErrorAction Stop
    } finally {
        if (Test-Path -LiteralPath $shortcutTemporary -PathType Leaf) {
            Remove-Item -LiteralPath $shortcutTemporary -Force -ErrorAction SilentlyContinue
        }
    }
    Write-Host "Desktop shortcut: $shortcutPath"
}

function Initialize-TinyStrataOptionalUI {
    param([string]$SourceRoot, [string]$WorkRoot, [int]$ApiPort,
          [int]$UiPort = 3000, [string]$Family = 'qwen36', [switch]$ConfigureOnly)
    $launcher = Join-Path $SourceRoot 'ui\launch.py'
    $python = Join-Path $WorkRoot 'runtime\.venv\Scripts\python.exe'
    Initialize-TinyStrataUiSource -SourceRoot $SourceRoot -WorkRoot $WorkRoot
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "The UI needs the installed Python runtime: $python" }
    Initialize-TinyStrataUiDependencies -SourceRoot $SourceRoot -WorkRoot $WorkRoot -Python $python
    # The entry point belongs to installation, not to the first successful UI
    # start. Keep it available for NoStart and for recovery after a start error.
    try { Initialize-TinyStrataDesktopShortcut -SourceRoot $SourceRoot -WorkRoot $WorkRoot }
    catch {
        $shortcutFailure = "The desktop shortcut could not be created: $($_.Exception.Message) Start TinyStrata with $(Join-Path $SourceRoot 'run-ui.bat')."
        try {
            $shortcutLog = Join-Path $WorkRoot 'runtime\ui\shortcut-install.log'
            [IO.Directory]::CreateDirectory((Split-Path -Parent $shortcutLog)) | Out-Null
            [IO.File]::AppendAllText($shortcutLog, ([DateTime]::UtcNow.ToString('o') + ' ' + $shortcutFailure + [Environment]::NewLine), (New-Object Text.UTF8Encoding($false)))
            $shortcutFailure += " Details: $shortcutLog"
        } catch { }
        Write-Warning $shortcutFailure
    }
    $arguments = @('-B', $launcher, '--source-dir', $SourceRoot, '--work-dir', $WorkRoot,
                   '--api-port', [string]$ApiPort, '--port', [string]$UiPort, '--model-family', $Family)
    if ($ConfigureOnly) { $arguments += '--configure-only' }
    else { $arguments += '--no-api-start' } # The installer already prepared the API, including intentional unloaded mode.
    try {
        & $python @arguments | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE -ne 0) { throw "UI launcher exited with code $LASTEXITCODE." }
    } catch {
        $logs = Join-Path $WorkRoot 'runtime\ui'
        throw "The optional UI could not be prepared or started: $($_.Exception.Message) The API was not stopped. See $logs\launch.log and $logs\ui.stderr.log; retry with the TinyStrata desktop shortcut or $(Join-Path $SourceRoot 'run-ui.bat')."
    }
    if ($ConfigureOnly) { Write-Host "UI prepared. Start it with $(Join-Path $SourceRoot 'run-ui.bat')." }
}

function Enter-TinyStrataInstallationGuard {
    param([string]$WorkRoot)
    [IO.Directory]::CreateDirectory($WorkRoot) | Out-Null
    $path = Join-Path ([IO.Path]::GetFullPath($WorkRoot)) '.tinystrata-installing.json'
    if ((Test-Path -LiteralPath $path) -and ((Get-Item -LiteralPath $path -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "The installer guard is linked and was left unchanged: $path"
    }
    $stream = $null
    try {
        # Readers may inspect the guard; a second writer cannot acquire it.
        # OpenOrCreate also lets a new installer repair an unlocked stale guard.
        $stream = [IO.File]::Open($path, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::Read)
        $owner = [Diagnostics.Process]::GetCurrentProcess()
        $epoch = [DateTime]::SpecifyKind([DateTime]'1970-01-01', [DateTimeKind]::Utc)
        $token = [Guid]::NewGuid().ToString('D')
        $record = [ordered]@{ schema = 1; pid = $owner.Id; started_unix = ($owner.StartTime.ToUniversalTime() - $epoch).TotalSeconds; token = $token }
        $bytes = (New-Object Text.UTF8Encoding($false)).GetBytes(($record | ConvertTo-Json))
        $stream.SetLength(0)
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush($true)
        return [pscustomobject]@{ Path = $path; Stream = $stream; Token = $token }
    } catch {
        if ($stream) { $stream.Dispose() }
        throw "Cannot acquire the TinyStrata installer guard. Another installer may be preparing this installation, or the work directory is not writable: $path. $_"
    }
}

function Exit-TinyStrataInstallationGuard {
    param($Guard)
    if (-not $Guard) { return }
    $Guard.Stream.Dispose()
    for ($attempt = 0; $attempt -lt 5; $attempt++) {
        try {
            if (-not (Test-Path -LiteralPath $Guard.Path -PathType Leaf)) { return }
            $record = [IO.File]::ReadAllText($Guard.Path) | ConvertFrom-Json
            # A competing installer may acquire the unlocked file first. Never
            # remove its guard; FileShare.Read also prevents deletion while locked.
            if ($record.token -eq $Guard.Token) { [IO.File]::Delete($Guard.Path) }
            return
        } catch {
            if ($attempt -eq 4) { throw "The installer finished but could not remove its guard. Run the installer again to repair it: $($Guard.Path). $_" }
            # A concurrent read of the marker can briefly deny deletion on Windows.
            [Threading.Thread]::Sleep(20)
        }
    }
}

function ConvertTo-TinyStrataUpdateArgument {
    param([AllowEmptyString()][string]$Value)
    return '"' + [regex]::Replace([regex]::Replace($Value, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
}

function Invoke-TinyStrataUpdateServiceControl {
    param([ValidateSet('stop', 'restore')][string]$Operation,
          [string]$SourceRoot, [string]$WorkRoot, $Snapshot)
    $source = [IO.Path]::GetFullPath($SourceRoot).TrimEnd('\')
    $work = [IO.Path]::GetFullPath($WorkRoot).TrimEnd('\')
    $python = Join-Path $work 'runtime\.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        $states = @()
        $runtime = Join-Path $work 'runtime'
        if (Test-Path -LiteralPath $runtime -PathType Container) {
            $states += @(Get-ChildItem -LiteralPath $runtime -Filter 'server-state*.json' -File)
            $uiState = Join-Path $runtime 'ui\ui-process.json'
            if (Test-Path -LiteralPath $uiState -PathType Leaf) { $states += Get-Item -LiteralPath $uiState }
        }
        foreach ($file in $states) {
            $record = [IO.File]::ReadAllText($file.FullName) | ConvertFrom-Json
            if ($record.pid -and (Get-Process -Id $record.pid -ErrorAction SilentlyContinue)) {
                throw 'Cannot safely inspect a running saved service without this installation private Python. No process was stopped.'
            }
        }
        if ($Operation -eq 'restore' -and ($Snapshot.Apis.Count -or $Snapshot.Uis.Count)) {
            throw 'The installation private Python is missing; previous services could not be restored.'
        }
        return [pscustomobject]@{ SourceRoot = $source; WorkRoot = $work; Api = $null; Apis = @(); Ui = $null; Uis = @(); Stopped = $false }
    }
    # The inspector is an immutable part of this already loaded installer.
    # It has no model imports and exchanges JSON through UTF-8 stdin/stdout.
    # Process handles, creation times, exact argv/cwd, private interpreter and
    # every listening PID are verified before any service is terminated.
    $code = @'
"""Embedded by the installer; no model imports and no process-name-only kills."""
import ctypes,datetime,hashlib,http.client,json,os,subprocess,sys,time
from pathlib import Path
from urllib.parse import quote
import psutil

class PendingServer(RuntimeError):pass

PROFILES={'qwen3.6-35b-a3b':('qwen36','config.json','server-state.json'),
 'gemma-4-26b-a4b-it':('gemma4','config-gemma4.json','server-state-gemma4.json'),
 'lfm2.5-24b-a2b':('lfm2','config-lfm2.json','server-state-lfm2.json'),
 'clover-35b-a3b-preview':('clover','config-clover.json','server-state-clover.json'),
 'tiel-coder-35b-a3b':('tiel','config-tiel-coder.json','server-state-tiel-coder.json')}
def norm(p):return os.path.normcase(str(Path(p).resolve()))
def same(a,b):
    try:return os.path.samefile(a,b)
    except OSError:return False
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def stamp(value):return datetime.datetime.fromisoformat(str(value).replace('Z','+00:00')).timestamp()
def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.update-'+str(os.getpid())+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(p)
def process(pid):
    try:return psutil.Process(int(pid))
    except (psutil.NoSuchProcess,ValueError,TypeError):return None
def listeners(port):
    return {x.pid for x in psutil.net_connections(kind='tcp') if x.status==psutil.CONN_LISTEN and x.laddr.port==port}
def request(port,path,config=None,method='GET'):
    headers={};cfg=read(config) if config else {}
    key=os.environ.get('STRATA_API_KEY') or cfg.get('api_key')
    if isinstance(key,list):key=key[0] if key else None
    if key:headers['Authorization']='Bearer '+key
    body=b'{}' if method=='POST' else None
    if body:headers['Content-Type']='application/json'
    c=http.client.HTTPConnection('127.0.0.1',port,timeout=300 if method=='POST' else 5)
    try:
        c.request(method,path,body,headers);r=c.getresponse();data=json.loads(r.read(2**20))
        if r.status!=200:raise RuntimeError(f'HTTP {r.status} from local port {port} {path}')
        return data
    finally:c.close()
def meta(p):
    with p.oneshot():return {'pid':p.pid,'parent':p.ppid(),'started':p.create_time(),'command':p.cmdline(),
                            'cwd':p.cwd(),'executable':p.exe()}
def options(command,kind,source):
    if len(command)<3 or command[1]!='-B':raise RuntimeError('The saved interpreter command is not a managed launch')
    if kind=='api':
        if command[2:4]!=['-m','serve.server']:raise RuntimeError('The API command is not the managed server')
        rest=command[4:];allowed={'--engine','--config','--port','--host'}
    else:
        if norm(command[2])!=norm(source/'ui/server.py'):raise RuntimeError('The UI command belongs to another source directory')
        rest=command[3:];allowed={'--port','--api-port','--data-dir'}
    if len(rest)%2:raise RuntimeError('The saved server command has unexpected arguments')
    result={}
    for i in range(0,len(rest),2):
        if rest[i] not in allowed or rest[i] in result:raise RuntimeError('The saved server command has unexpected options')
        result[rest[i]]=rest[i+1]
    return result
def verify_process(m,command,source,python):
    if not same_invocation(m['command'],command,python) or norm(m['cwd'])!=norm(source):raise RuntimeError('Server command or working directory changed; no process was stopped')
    if not any(same(m['executable'],p) for p in interpreter_images(python)):
        raise RuntimeError('The server interpreter image belongs to another installation')
def interpreter_images(python):
    # A Microsoft Store App Execution Alias is not the physical image reported
    # for its process. This inspector is itself launched by the verified private
    # venv, so its current OS image is an independent trusted identity too.
    return (python,getattr(sys,'_base_executable',sys.executable),psutil.Process(os.getpid()).exe())
def same_invocation(actual,expected,python):
    images=interpreter_images(python)
    return (isinstance(actual,list) and isinstance(expected,list) and len(actual)==len(expected) and
            len(actual)>1 and actual[1:]==expected[1:] and
            any(same(actual[0],p) for p in images) and any(same(expected[0],p) for p in images))
def connection_value(value,*,key=False):
    if value is None or value=='' or value==[]:return ()
    if key:
        values=value if isinstance(value,list) else str(value).split(',')
        return tuple(sorted(set(str(v).strip() for v in values)))
    return tuple(value) if isinstance(value,list) else (value,)
def host_environment(value):
    return {v.strip().lower().rstrip('.') for v in value.split(',') if v.strip()}
def verify_connection_settings(original,active,owner):
    # Compare privately before any unload/stop. Never put connection values,
    # process environment contents or secret fingerprints into the snapshot.
    before,after=read(original),read(active)
    for field in ('api_key','allowed_hosts','cors_origins','trusted_origins'):
        if connection_value(before.get(field),key=field=='api_key')!=connection_value(after.get(field),key=field=='api_key'):
            raise RuntimeError('The running API uses different per-profile connection settings; services were left unchanged. Align those settings explicitly before updating.')
    try:
        environment=owner.environ()
        inherited={name:environment.get(name,'') for name in ('STRATA_API_KEY','STRATA_ALLOWED_HOSTS')}
        del environment
    except (psutil.Error,OSError):
        raise RuntimeError('The running API connection environment could not be verified; services were left unchanged') from None
    old_key=connection_value(inherited['STRATA_API_KEY'],key=True)
    new_key=connection_value(os.environ.get('STRATA_API_KEY',''),key=True)
    configured_key=connection_value(before.get('api_key'),key=True)
    if (old_key or configured_key)!=(new_key or configured_key):
        raise RuntimeError('The running API uses different inherited connection settings; services were left unchanged. Use the same connection environment before updating.')
    configured_hosts={str(v).strip().lower().rstrip('.') for v in connection_value(before.get('allowed_hosts'))}
    if (configured_hosts|host_environment(inherited['STRATA_ALLOWED_HOSTS']))!=(configured_hosts|host_environment(os.environ.get('STRATA_ALLOWED_HOSTS',''))):
        raise RuntimeError('The running API uses different inherited connection settings; services were left unchanged. Use the same connection environment before updating.')
def collect(source,work):
    runtime=work/'runtime';python=runtime/'.venv/Scripts/python.exe';records=[]
    paths=[*runtime.glob('server-state*.json')]
    uistate=runtime/'ui/ui-process.json'
    if uistate.exists():paths.append(uistate)
    for path in paths:
        state=read(path)
        if not isinstance(state,dict) or type(state.get('pid')) is not int:continue
        root=process(state['pid'])
        if root is None:continue
        m=meta(root)
        if 'started' not in state or abs(m['started']-stamp(state['started']))>.00001:
            raise RuntimeError(f'Saved process identity changed for {path.name}; it was left unchanged')
        kind='ui' if path==uistate else 'api';command=m['command'];args=options(command,kind,source)
        verify_process(m,command,source,python)
        port=int(args.get('--port',0))
        if not 0<port<65536:raise RuntimeError('Invalid saved server port')
        if kind=='api':
            config=Path(state.get('config',''))
            if norm(state.get('source',''))!=norm(source) or int(state.get('port',0))!=port:
                raise RuntimeError('Saved API source or port does not match its running command')
            if norm(args.get('--config',''))!=norm(config) or norm(config.parent)!=norm(runtime/'.tinystrata') or not config.is_file():
                raise RuntimeError('The API configuration is outside this installation')
            if args.get('--engine')!='tinystrata':raise RuntimeError('The API uses a different engine')
        else:
            if not same_invocation(state.get('command'),command,python):raise RuntimeError('Saved UI command does not match its running process')
            config=None
            if '--data-dir' in args and norm(args['--data-dir'])!=norm(runtime/'ui'):
                raise RuntimeError('The UI history directory belongs to another installation')
        owners=listeners(port)
        if not owners:raise PendingServer(f'The recorded {kind} is still starting on port {port}; retry after it is ready')
        verified=[m]
        if kind=='ui' and state.get('launcher_pid')==m['parent']:
            wrapper=process(m['parent'])
            if wrapper is not None:
                wm=meta(wrapper)
                if wm['started']>m['started']:raise RuntimeError('The saved UI wrapper PID has been reused')
                verify_process(wm,command,source,python)
                verified.append(wm)
        for owner in owners:
            if owner==m['pid']:continue
            child=process(owner)
            if child is None:raise RuntimeError('The listener changed during inspection')
            cm=meta(child)
            if cm['parent']!=m['pid'] or cm['started']<m['started']:raise RuntimeError('A server port belongs to another process')
            verify_process(cm,command,source,python);verified.append(cm)
        health=request(port,'/local-ui-health' if kind=='ui' else '/health')
        if kind=='ui':
            if health.get('app')!='tinystrata-ui' or health.get('pid') not in owners or int(health.get('api_port',0))!=int(args.get('--api-port',0)):
                raise RuntimeError('The UI listener identity does not match this installation')
            for key,expected in (('source_dir',source),('data_dir',runtime/'ui')):
                if key in health and norm(health[key])!=norm(expected):raise RuntimeError('The UI serves another installation')
            item={'Kind':'ui','Port':port,'ApiPort':int(args['--api-port']),'StatePath':str(path),'Processes':verified,'Command':command}
        else:
            if health.get('service') not in ('strata','tinystrata') or health.get('status')!='ok' or health.get('model') not in PROFILES:
                raise RuntimeError('The API is switching or does not identify a supported active model')
            model=health['model'];family,filename,statename=PROFILES[model]
            active=runtime/'.tinystrata'/filename
            if family=='qwen36' and active.exists() and read(active).get('arch')=='gpt-oss':active=active.with_name('config-qwen36.json')
            if not active.is_file():raise RuntimeError('The active model profile is missing; the API was left running')
            owner=process(next(iter(owners)))
            if owner is None:raise RuntimeError('The API listener changed during connection-setting inspection')
            verify_connection_settings(config,active,owner)
            item={'Kind':'api','Family':family,'Port':port,'Host':args.get('--host') or read(config).get('host') or '127.0.0.1','Config':str(active),'Model':model,'Loaded':bool(health.get('loaded')),
                  'OriginalConfig':str(config),'StatePath':str(runtime/statename),'Processes':verified,'Command':command}
        if any(x['Kind']==kind and x['Port']==port for x in records):continue
        records.append(item)
    return records
def idle(item):
    if item['Kind']!='api':return
    status=request(item['Port'],'/status',item['OriginalConfig'])
    control=request(item['Port'],'/v1/model/switch',item['OriginalConfig'])
    library=request(item['Port'],'/v1/models',item['OriginalConfig'])
    if status.get('busy') or status.get('queued') or control.get('switching') or control.get('in_flight'):
        raise RuntimeError(f"TinyStrata on port {item['Port']} is busy; no service was stopped")
    if control.get('downloading') or any(x.get('management',{}).get('delete_reason')=='downloading' for x in library.get('data',[])):
        raise RuntimeError('A model download is running; no service was stopped')
    if 'downloading' not in control:
        # First upgrade: the already loaded legacy API has no installer gate
        # or aggregate download flag. Query every supported model's job, rather
        # than relying on the active model's "active" delete-lock reason.
        for row in library.get('data',[]):
            if row.get('id') in PROFILES:
                job=request(item['Port'],'/v1/model/download?model='+quote(row['id'],safe=''),item['OriginalConfig'])
                if job.get('state') in ('downloading','verifying','loading') or job.get('running'):
                    raise RuntimeError('A model download is running; no service was stopped')
def hold(m):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32];kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.CloseHandle.argtypes=[ctypes.c_void_p];kernel.TerminateProcess.argtypes=[ctypes.c_void_p,ctypes.c_uint];kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint]
    handle=kernel.OpenProcess(0x1000|0x0001|0x100000,False,m['pid'])
    if not handle:raise RuntimeError('Cannot acquire a verified process handle; services were left unchanged')
    p=process(m['pid'])
    if p is None or meta(p)!=m:
        kernel.CloseHandle(handle);raise RuntimeError('Process identity changed during inspection')
    return kernel,handle
def active_children(p,known):
    console=Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/conhost.exe'
    result=[]
    for child in p.children(recursive=True):
        if child.pid in known:continue
        # CREATE_NEW_PROCESS_GROUP may add Windows' own console host. It exits
        # with its verified parent; it is never a process-name kill target.
        if child.ppid() in known and same(child.exe(),console):continue
        result.append(child)
    return result
def stop(source,work):
    items=collect(source,work);apis=[x for x in items if x['Kind']=='api'];uis=[x for x in items if x['Kind']=='ui']
    selected=next((a for u in uis for a in apis if a['Port']==u['ApiPort']),apis[0] if apis else None)
    snapshot={'SourceRoot':str(source),'WorkRoot':str(work),'Api':selected,'Apis':apis,'Ui':uis[0] if uis else None,'Uis':uis,'Stopped':False}
    handles=[];stopping=False
    try:
        for item in items:
            idle(item)
            for m in item['Processes']:handles.append((item,m,*hold(m)))
        known={m['pid'] for item in items for m in item['Processes']}
        for item in uis:
            for m in item['Processes']:
                p=process(m['pid'])
                if p is not None and active_children(p,known):
                    raise RuntimeError('The UI has active tool/search child processes; no service was stopped')
        for item in apis:
            idle(item)
            result=request(item['Port'],'/unload',item['OriginalConfig'],'POST')
            if result.get('status') not in ('unloaded','not loaded'):raise RuntimeError('The model did not unload safely; services were left running')
        # Unload has joined native workers. Never kill an unverified model or
        # arbitrary descendant merely because it shares a process name.
        for item in apis:
            idle(item)
            for m in item['Processes']:
                p=process(m['pid'])
                extra=[(c.pid,c.name()) for c in active_children(p,known)] if p is not None else []
                if extra:
                    raise RuntimeError(f'The API still has active child processes {extra}; no API was terminated')
        stopping=True
        for item,m,kernel,handle in reversed(handles):
            if kernel.WaitForSingleObject(handle,0)==0:continue
            p=process(m['pid'])
            try:observed=meta(p) if p is not None else None
            except psutil.NoSuchProcess:
                # A venv redirector exits naturally as soon as its verified
                # listening child has ended. Its held original handle proves
                # this is completion, rather than a reused PID.
                if kernel.WaitForSingleObject(handle,100)==0:continue
                raise
            if observed!=m:
                if kernel.WaitForSingleObject(handle,0)==0:continue
                raise RuntimeError('Server identity changed before stopping')
            if not kernel.TerminateProcess(handle,0) or kernel.WaitForSingleObject(handle,5000)!=0:
                raise RuntimeError('A verified server did not stop')
        snapshot['Stopped']=True
        return snapshot
    except BaseException as error:
        error.snapshot=snapshot
        raise
    finally:
        for _,_,kernel,handle in handles:kernel.CloseHandle(handle)
def revision(source):
    digest=hashlib.sha256()
    for name in ('server.py','launch.py','stdio.py','chat_history.py','web_search.py','tool_settings.py','sandbox_runtime.py'):
        digest.update(name.encode()+b'\0');digest.update((source/'ui'/name).read_bytes())
    return digest.hexdigest()
def resume(source,work,snapshot):
    if norm(snapshot.get('SourceRoot',''))!=norm(source) or norm(snapshot.get('WorkRoot',''))!=norm(work):
        raise RuntimeError('The saved service snapshot belongs to a different installation')
    python=work/'runtime/.venv/Scripts/python.exe';results=[]
    for item in [*snapshot.get('Apis',[]),*snapshot.get('Uis',[])]:
        kind=item['Kind'];port=item['Port'];owners=listeners(port)
        if owners:
            # A normal successful installer may already have restarted it. A
            # changed/foreign current listener is never killed or overwritten.
            current=next((x for x in collect(source,work) if x['Kind']==kind and x['Port']==port),None)
            if current is None:raise RuntimeError(f'Port {port} is now owned by another service; it was left unchanged')
            if kind=='api' and current['Model']==item['Model'] and item['Loaded'] and not current['Loaded']:
                idle(current)
                result=request(port,'/load',current['OriginalConfig'],'POST')
                if result.get('status') not in ('loaded','already loaded'):raise RuntimeError('The previously loaded model could not be restored')
            results.append({'kind':kind,'port':port,'reused':True,'model':current.get('Model')});continue
        if kind=='api':
            config=Path(item['Config'])
            if not config.is_file():raise RuntimeError('The saved active API profile is missing; no replacement model was selected')
            command=[str(python),'-B','-m','serve.server','--engine','tinystrata','--config',str(config),'--port',str(port),'--host',item.get('Host','127.0.0.1')]
        else:command=[str(python),'-B',str(source/'ui/server.py'),'--port',str(port),'--api-port',str(item['ApiPort']),'--data-dir',str(work/'runtime/ui')]
        logs=work/'runtime/ui' if kind=='ui' else work/'runtime';logs.mkdir(parents=True,exist_ok=True)
        with (logs/(kind+'.stdout.log')).open('ab') as out,(logs/(kind+'.stderr.log')).open('ab') as err:
            p=subprocess.Popen(command,cwd=source,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'},creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
        created=psutil.Process(p.pid).create_time()
        state={'pid':p.pid,'started':datetime.datetime.fromtimestamp(created,datetime.timezone.utc).isoformat().replace('+00:00','Z')}
        state.update({'config':item['Config'],'source':str(source),'port':port} if kind=='api' else {'command':command})
        atomic(item['StatePath'],state)
        deadline=time.monotonic()+(180 if kind=='api' else 20)
        while time.monotonic()<deadline:
            if p.poll() is not None:raise RuntimeError(f'The restored {kind} exited with code {p.returncode}; inspect {logs}')
            try:
                current=next((x for x in collect(source,work) if x['Kind']==kind and x['Port']==port),None)
                if current:
                    if kind=='api' and current['Model']!=item['Model']:raise RuntimeError('The restored API selected a different model')
                    if kind=='ui' and request(port,'/local-ui-health').get('backend_revision')!=revision(source):raise RuntimeError('The restored UI loaded a stale backend')
                    break
            except (ConnectionError,OSError,http.client.HTTPException,PendingServer):pass
            time.sleep(.2)
        else:raise RuntimeError(f'The restored {kind} did not become ready; inspect {logs}')
        # Restore a previously loaded model only; never initiate a download.
        if kind=='api' and item['Loaded'] and not current.get('Loaded'):
            result=request(port,'/load',item['Config'],'POST')
            if result.get('status') not in ('loaded','already loaded'):raise RuntimeError('The previously loaded model could not be restored')
        results.append({'kind':kind,'port':port,'reused':False,'model':item.get('Model')})
    return results
def main():
    req=json.load(sys.stdin);source=Path(req['SourceRoot']).resolve();work=Path(req['WorkRoot']).resolve()
    if source==work or source in work.parents:raise RuntimeError('The work directory must remain outside the source')
    if norm(sys.prefix)!=norm(work/'runtime/.venv'):raise RuntimeError('Service inspection requires this installation private Python')
    marker=work/'.tinystrata-installing.json'
    if req['Operation']=='stop':
        guard=read(marker);owner=process(guard.get('pid'));ancestors={p.pid for p in psutil.Process().parents()}
        if (guard.get('schema')!=1 or owner is None or owner.pid not in ancestors or
            abs(owner.create_time()-float(guard.get('started_unix',0)))>.01):
            raise RuntimeError('Acquire this installer installation guard before stopping services')
    elif marker.exists():raise RuntimeError('Release the installation guard before restoring services')
    try:
        result=stop(source,work) if req['Operation']=='stop' else resume(source,work,req['Snapshot'])
        print(json.dumps({'ok':True,'result':result},ensure_ascii=True))
    except Exception as error:
        print(json.dumps({'ok':False,'error':str(error),'snapshot':getattr(error,'snapshot',None)},ensure_ascii=True))
        return 1
    return 0
if __name__=='__main__':raise SystemExit(main())
'@
    $request = @{ Operation = $Operation; SourceRoot = $source; WorkRoot = $work; Snapshot = $Snapshot } | ConvertTo-Json -Depth 30 -Compress
    $start = New-Object Diagnostics.ProcessStartInfo
    $start.FileName = $python
    $start.Arguments = (@('-I', '-X', 'utf8', '-B', '-c', $code) | ForEach-Object { ConvertTo-TinyStrataUpdateArgument ([string]$_) }) -join ' '
    $start.WorkingDirectory = $source
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardInput = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $start.StandardOutputEncoding = New-Object Text.UTF8Encoding($false)
    $start.StandardErrorEncoding = New-Object Text.UTF8Encoding($false)
    $start.EnvironmentVariables['PYTHONUTF8'] = '1'
    $start.EnvironmentVariables['PYTHONIOENCODING'] = 'utf-8'
    $child = New-Object Diagnostics.Process
    $child.StartInfo = $start
    try {
        if (-not $child.Start()) { throw 'The service inspector did not start.' }
        $stdout = $child.StandardOutput.ReadToEndAsync()
        $stderr = $child.StandardError.ReadToEndAsync()
        $bytes = (New-Object Text.UTF8Encoding($false)).GetBytes($request)
        $child.StandardInput.BaseStream.Write($bytes, 0, $bytes.Length)
        $child.StandardInput.Close()
        $child.WaitForExit()
        $text = $stdout.GetAwaiter().GetResult()
        $errorText = $stderr.GetAwaiter().GetResult()
        if (-not $text.Trim()) { throw "The service inspector exited with code $($child.ExitCode) before returning JSON. $errorText" }
        try { $result = ConvertFrom-Json -InputObject $text }
        catch { throw "The service inspector could not return its result. $errorText" }
        if ($child.ExitCode -ne 0 -or -not $result.ok) {
            $failure = New-Object InvalidOperationException ([string]$result.error)
            if ($result.snapshot) { $failure.Data['TinyStrataServices'] = $result.snapshot }
            throw $failure
        }
        return $result.result
    } finally { $child.Dispose() }
}

function Stop-TinyStrataOwnedServicesForUpdate {
    param([string]$SourceRoot, [string]$WorkRoot)
    # Caller acquires the installation guard before this call. New APIs reject
    # new work under that guard; legacy APIs receive two idle checks + unload.
    return Invoke-TinyStrataUpdateServiceControl -Operation stop -SourceRoot $SourceRoot -WorkRoot $WorkRoot
}

function Restore-TinyStrataOwnedServicesAfterUpdate {
    param($Snapshot, [string]$SourceRoot, [string]$WorkRoot, [switch]$NoBrowser)
    if (-not $Snapshot) { return }
    # Call only after restoring/committing sources and releasing the guard.
    # Existing newly started owned listeners are reused. Settings, histories,
    # model files and API keys are never copied back from an old snapshot.
    $restored = @(Invoke-TinyStrataUpdateServiceControl -Operation restore -SourceRoot $SourceRoot -WorkRoot $WorkRoot -Snapshot $Snapshot)
    foreach ($service in $restored) {
        if ($service.kind) { Write-Host ("Restored {0} on port {1}." -f $service.kind, $service.port) }
    }
}

function Assert-TinyStrataInstallationIdle {
    param([string]$SourceRoot, [string]$WorkRoot)
    $source = [IO.Path]::GetFullPath($SourceRoot).TrimEnd('\')
    $work = [IO.Path]::GetFullPath($WorkRoot).TrimEnd('\')
    foreach ($process in @(Get-CimInstance Win32_Process -ErrorAction Stop)) {
        $command = [string]$process.CommandLine
        $path = [string]$process.ExecutablePath
        $native = $process.Name -like 'tinystrata*.exe' -and
                  ($path.StartsWith($work + '\', [StringComparison]::OrdinalIgnoreCase) -or
                   $path.StartsWith($source + '\', [StringComparison]::OrdinalIgnoreCase) -or
                   $command.IndexOf($work, [StringComparison]::OrdinalIgnoreCase) -ge 0)
        $python = $process.Name -like 'python*.exe' -and
                  ($command -match 'serve\.server|ui[\\/]launch\.py|ui[\\/]server\.py|\bui\.server\b') -and
                  ($command.IndexOf($source, [StringComparison]::OrdinalIgnoreCase) -ge 0 -or
                   $command.IndexOf($work, [StringComparison]::OrdinalIgnoreCase) -ge 0)
        if ($native -or $python) { throw "Stop TinyStrata's API, UI and model processes for this installation before preparing an update or runtime (PID $($process.ProcessId))." }
    }
}

function New-TinyStrataNativeGeneration {
    param([string]$WorkRoot, [string]$Backend, [string]$FromDirectory)
    $root = [IO.Path]::GetFullPath((Join-Path $WorkRoot 'native'))
    $generation = Join-Path $root ($Backend + '-' + [Guid]::NewGuid().ToString('N'))
    [IO.Directory]::CreateDirectory($generation) | Out-Null
    if ($FromDirectory) {
        # Copy only application binaries and their app-local CRT, never build
        # caches or old SDK runtime DLLs/manifest. The selected SDK supplies those.
        foreach ($file in @(Get-ChildItem -LiteralPath $FromDirectory -File)) {
            if ($file.Name -match '^(tinystrata(?:-[a-z0-9]+)?\.exe|strata[^\\/]*\.dll|(?:msvcp140[^\\/]*|vcruntime140[^\\/]*|vcomp140|concrt140|vccorlib140)\.dll)$') {
                Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $generation $file.Name) -ErrorAction Stop
            }
        }
    }
    return $generation
}

function Remove-TinyStrataNativeGeneration {
    param([string]$WorkRoot, [string]$Directory)
    if (-not $Directory -or -not (Test-Path -LiteralPath $Directory)) { return }
    $root = [IO.Path]::GetFullPath((Join-Path $WorkRoot 'native')).TrimEnd('\') + '\'
    $target = [IO.Path]::GetFullPath($Directory)
    if (-not $target.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid unpublished native generation directory.' }
    Remove-Item -LiteralPath $target -Recurse -Force
}

function Invoke-TinyStrataInstaller {
    param([string]$SourceDirectory, [string]$WorkingDirectory, [int]$ApiPort,
          [bool]$DoNotStart, [bool]$DoNotDownload, [bool]$InspectOnly, [bool]$BuildAgain, [bool]$RefreshSource,
          [string]$Family = 'qwen36', [string]$Backend = 'auto', [string]$HipDirectory, [string]$HipArchitectures,
          [bool]$IncludeUI = $false, [int]$UiPort = 3000, [string]$Revision, [bool]$FamilyExplicit = $false,
          [bool]$UiPortExplicit = $false)
    if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitProcess) { throw 'Run this installer in 64-bit Windows PowerShell or PowerShell.' }
    $sourceRoot = [IO.Path]::GetFullPath($SourceDirectory).TrimEnd('\')
    $sourceReady = Test-TinyStrataSource -Directory $sourceRoot
    $archiveInstall = $sourceReady -and -not (Test-Path -LiteralPath (Join-Path $sourceRoot '.git')) -and
                      -not ((Get-Item -LiteralPath $sourceRoot -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)
    if ($Revision -and $archiveInstall) { $RefreshSource = $true }
    if ($RefreshSource -and $Revision -and (Test-TinyStrataSourceRevision -Directory $sourceRoot -Revision $Revision)) { $RefreshSource = $false }
    if ($IncludeUI) { $DoNotDownload = $true }
    if (-not $sourceReady) { Assert-TinyStrataSourceDestination -Directory $sourceRoot }
    if (-not $WorkingDirectory) { $WorkingDirectory = $sourceRoot + '-work' }
    $workRoot = [IO.Path]::GetFullPath($WorkingDirectory).TrimEnd('\')
    if ($workRoot -eq $sourceRoot -or $workRoot.StartsWith($sourceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'WorkDir must be outside the source directory.' }
    $runtimeRoot = Join-Path $workRoot 'runtime'
    $portAutomatic = $ApiPort -eq 0
    $selectedFamily = Get-TinyStrataModelFamily -Name $Family
    $configPath = Join-Path (Join-Path $runtimeRoot '.tinystrata') $selectedFamily.ConfigFile
    $statePath = Join-Path $runtimeRoot $selectedFamily.StateFile
    $cfg = Get-TinyStrataConfig -Path $configPath
    # Preserve a retired model in the legacy primary file during an update.
    if ($selectedFamily.Name -eq 'qwen36' -and $cfg -and $cfg.arch -eq 'gpt-oss') {
        $configPath = Join-Path (Join-Path $runtimeRoot '.tinystrata') 'config-qwen36.json'
        $cfg = Get-TinyStrataConfig -Path $configPath
    }
    $selectedBackend = Get-TinyStrataBackend -Requested $Backend -Configuration $cfg
    if ($ApiPort -eq 0) { $ApiPort = if ($cfg -and $cfg.port) { [int]$cfg.port } else { $selectedFamily.Port } }
    if ($ApiPort -lt 1 -or $ApiPort -gt 65535) { throw 'Port must be between 1 and 65535.' }
    if ($IncludeUI -and ($UiPort -lt 1 -or $UiPort -gt 65535 -or $UiPort -eq $ApiPort)) {
        throw 'UiPort must be between 1 and 65535 and different from the API port.'
    }
    $baseUrl = "http://127.0.0.1:$ApiPort"
    if ($InspectOnly) {
        Write-Host "Source:  $sourceRoot"
        Write-Host "Work:    $workRoot"
        Write-Host "Config:  $configPath"
        Write-Host "API:     $baseUrl/v1"
        if ($IncludeUI) { Write-Host "UI:      http://127.0.0.1:$UiPort" }
        if (-not $sourceReady -or $RefreshSource) { Write-Host 'First download TinyStrata sources from GitHub (no git installation required).' }
        Write-Host "Backend: $selectedBackend"
        Write-Host 'Plan: reuse installed components; prepare Python/dependencies; reuse or build the selected GPU engine; prepare its runtime; save configuration.'
        if ($selectedBackend -eq 'hip') { Write-Host 'AMD: use the private pinned ROCm 10.2.0a20260930 SDK (HIP 7.17) by default; -HipDirectory selects a custom SDK; Qwen/LFM select native FP8 when available, otherwise INT8; Gemma uses INT8.' }
        if (-not $DoNotStart) { Write-Host $(if ($DoNotDownload) { 'Then start the API; download a model later from Models.' } else { 'Then start the API and install the fixed model through its API if needed.' }) }
        if ($IncludeUI) { Write-Host $(if ($DoNotStart) { 'Prepare the optional UI launcher without starting services.' } else { 'Then start the optional UI and open it in the browser.' }) }
        return
    }
    $updateGuard = $null
    $ownedServices = $null
    $updateFailure = $null
    try {
    if ($RefreshSource -or $BuildAgain -or $HipDirectory -or ($Revision -and $IncludeUI -and $archiveInstall)) {
        $updateGuard = Enter-TinyStrataInstallationGuard -WorkRoot $workRoot
        $ownedServices = Stop-TinyStrataOwnedServicesForUpdate -SourceRoot $sourceRoot -WorkRoot $workRoot
        if ($ownedServices.Api) {
            if (-not $FamilyExplicit) {
                $selectedFamily = Get-TinyStrataModelFamily -Name $ownedServices.Api.Family
                $configPath = [string]$ownedServices.Api.Config
                $statePath = Join-Path $runtimeRoot $selectedFamily.StateFile
                $cfg = Get-TinyStrataConfig -Path $configPath
                $selectedBackend = Get-TinyStrataBackend -Requested $Backend -Configuration $cfg
            }
            if ($portAutomatic) { $ApiPort = [int]$ownedServices.Api.Port }
        }
        if ($ownedServices.Ui -and -not $UiPortExplicit) { $UiPort = [int]$ownedServices.Ui.Port }
        if ($IncludeUI -and $ApiPort -eq $UiPort) { throw 'The UI and API must use different ports.' }
        $baseUrl = "http://127.0.0.1:$ApiPort"
    }
    # Inspect occupied ports before changing configuration or installing anything.
    if (Test-Path -LiteralPath $statePath -PathType Leaf) {
        try {
            $savedState = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
            $previousServer = Get-TinyStrataSavedProcess -StatePath $statePath -ConfigPath $configPath -SourcePath $sourceRoot -ApiPort $savedState.port
        } catch { $previousServer = $null }
        if ($previousServer -and $savedState.port -ne $ApiPort) { throw "This installation already runs on port $($savedState.port). Stop that instance before changing its port." }
    }
    $server = Get-TinyStrataSavedProcess -StatePath $statePath -ConfigPath $configPath -SourcePath $sourceRoot -ApiPort $ApiPort
    if ($server -and $DoNotStart) {
        if ($IncludeUI) { Initialize-TinyStrataOptionalUI -SourceRoot $sourceRoot -WorkRoot $workRoot -ApiPort $ApiPort -UiPort $UiPort -Family $selectedFamily.Name -ConfigureOnly }
        Write-Host "Installation already prepared; existing API left unchanged (PID $($server.Id))."
        return
    }
    if (-not $DoNotStart) {
        $listeners = @(Get-NetTCPConnection -LocalPort $ApiPort -State Listen -ErrorAction SilentlyContinue)
        if ($listeners.Count -gt 0 -and (-not $server -or -not (Test-TinyStrataPortOwner -ApiPort $ApiPort -RootProcessId $server.Id))) {
            throw "Port $ApiPort is already in use by another process. It was left unchanged. Use -Port to choose another port."
        }
    }
    if (-not $server) {
        $sourceChange = $null
        $generation = $null
        $candidateConfig = $null
        $installationGuard = $updateGuard
        $pythonTransaction = @{}
        $preparationCommitted = $false
        try {
        if (-not $installationGuard) { $installationGuard = Enter-TinyStrataInstallationGuard -WorkRoot $workRoot }
        Assert-TinyStrataInstallationIdle -SourceRoot $sourceRoot -WorkRoot $workRoot
        if ($selectedBackend -eq 'cuda') {
            $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
            if (-not $smi) { throw 'A working NVIDIA GPU driver is required for -Backend cuda.' }
            & $smi.Source -L | ForEach-Object { Write-Host $_ }
            if ($LASTEXITCODE -ne 0) { throw 'The NVIDIA driver is not working. GPU drivers are not replaced by this installer.' }
        }
        $cacheRoot = Join-Path $workRoot 'downloads'
        if (-not $sourceReady -or $RefreshSource) { $sourceChange = Initialize-TinyStrataSource -SourceRoot $sourceRoot -CacheRoot $cacheRoot -Refresh:$RefreshSource -Revision $Revision }
        $nativeFingerprint = Get-TinyStrataSourceFingerprint -Directory $sourceRoot -NativeOnly
        $python = Initialize-TinyStrataPython -WorkRoot $workRoot -SourceRoot $sourceRoot -CacheRoot $cacheRoot -Transaction $pythonTransaction
        if ($selectedBackend -eq 'hip') {
            $hipRoot = Ensure-TinyStrataHipToolkit -DestinationDirectory (Join-Path $workRoot 'rocm-venv') -Python $python -Directory $HipDirectory -Architectures $HipArchitectures
            $hipArchs = Get-TinyStrataHipArchitectures -Directory $hipRoot -Requested $HipArchitectures
            Write-Host "AMD HIP SDK: $hipRoot ($hipArchs); automatic KV selection."
        }
        $search = @()
        if ($cfg -and $cfg.lib_dirs -and $cfg.backend -eq $selectedBackend) { $search += @($cfg.lib_dirs) }
        $cudaRoots = @()
        if (Test-Path -LiteralPath $workRoot -PathType Container) {
            $cudaRoots += @(Get-ChildItem -LiteralPath $workRoot -Directory -Filter 'cuda-*' | Sort-Object Name -Descending | ForEach-Object { $_.FullName })
        }
        if ($env:CUDA_PATH) { $cudaRoots += $env:CUDA_PATH }
        foreach ($cudaRoot in $cudaRoots) { $search += Join-Path $cudaRoot 'bin'; $search += Join-Path $cudaRoot 'bin\x64' }
        $hipRuntimeDirs = @()
        if ($selectedBackend -eq 'hip') {
            $hipRuntimeDirs = @(Get-TinyStrataHipRuntimeDirectories -Directory $hipRoot)
            $search = $hipRuntimeDirs
        }
        $search += $runtimeRoot
        $exe = $null
        if (-not $BuildAgain) {
            $candidates = @()
            if ($cfg -and $cfg.exe) {
                $savedExe = [string]$cfg.exe
                if (-not [IO.Path]::IsPathRooted($savedExe)) { $savedExe = Join-Path $sourceRoot $savedExe }
                $candidates += Join-Path (Split-Path -Parent $savedExe) 'tinystrata.exe'
            }
            foreach ($folder in $(if ($selectedBackend -eq 'hip') { @('build-hip', 'build-hip\Release') } else { @('build', 'build\Release', 'build-cuda-132', 'build-cuda') })) { $candidates += Join-Path $workRoot ($folder + '\tinystrata.exe') }
            foreach ($folder in @('bin', 'build', 'build\Release')) { $candidates += Join-Path $sourceRoot ($folder + '\tinystrata.exe') }
            foreach ($candidate in $candidates) {
                if ((Test-Path -LiteralPath $candidate -PathType Leaf) -and
                    (Test-TinyStrataNativeBuildStamp -Directory (Split-Path -Parent $candidate) -Fingerprint $nativeFingerprint -Backend $selectedBackend) -and
                    ($selectedBackend -eq 'hip' -or (Test-Path -LiteralPath (Join-Path (Split-Path -Parent $candidate) 'strata_kq.dll') -PathType Leaf)) -and
                    (Test-Path -LiteralPath (Join-Path (Split-Path -Parent $candidate) 'tinystrata-lfm2.exe') -PathType Leaf) -and
                    (Test-Path -LiteralPath (Join-Path (Split-Path -Parent $candidate) 'tinystrata-gemma4.exe') -PathType Leaf) -and
                    (Test-TinyStrataEngineBackend -ExecutablePath $candidate -SearchDirectories $search -Backend $selectedBackend) -and
                    (Test-TinyStrataVisionBackend -ExecutablePath (Join-Path (Split-Path -Parent $candidate) 'tinystrata-vision.exe') -SearchDirectories $search -Backend $selectedBackend)) { $exe = $candidate; break }
            }
        }
        if (-not $exe) {
            Write-Host 'No local TinyStrata engine found; preparing the first source build.'
            $null = Ensure-TinyStrataBuildTools -CacheDirectory $cacheRoot
            $generation = New-TinyStrataNativeGeneration -WorkRoot $workRoot -Backend $selectedBackend
            if ($selectedBackend -eq 'hip') {
                $build = Invoke-TinyStrataPortableBuild -SourceDirectory $sourceRoot -BuildDirectory $generation -Backend hip -HipDirectory $hipRoot -HipArchitectures $hipArchs
            } else {
                $cuda = Ensure-TinyStrataCudaToolkit -DestinationDirectory (Join-Path $workRoot 'cuda-13.2') -CacheDirectory $cacheRoot -SearchDirectories $cudaRoots
                $build = Invoke-TinyStrataPortableBuild -SourceDirectory $sourceRoot -BuildDirectory $generation -CudaDirectory $cuda
                $search += Join-Path $cuda 'bin'; $search += Join-Path $cuda 'bin\x64'
            }
            $exe = $build.Exe
        } elseif ($selectedBackend -eq 'hip') {
            # Prepare a new immutable runtime generation. Other saved profiles
            # and failed updates keep their previous core/math combination.
            $generation = New-TinyStrataNativeGeneration -WorkRoot $workRoot -Backend hip -FromDirectory (Split-Path -Parent $exe)
            $exe = Join-Path $generation 'tinystrata.exe'
        }
        $search += Split-Path -Parent $exe
        $search += $runtimeRoot
        $dllDirs = @(Initialize-TinyStrataRuntime -RuntimeRoot $runtimeRoot -CacheRoot $cacheRoot -SearchDirectories $search -HipRuntimeDirectories $hipRuntimeDirs -HipSdkRoot $hipRoot -EngineDirectory (Split-Path -Parent $exe) -Backend $selectedBackend)
        if ($generation) { Write-TinyStrataNativeBuildStamp -Directory $generation -SourceRoot $sourceRoot -Backend $selectedBackend }
        if ($cfg -and $cfg.lib_dirs -and $cfg.backend -eq $selectedBackend) {
            # Keep HIP math libraries from the selected SDK ahead of a saved
            # SDK's directories; its matching core runtime is now app-local.
            if ($selectedBackend -eq 'hip') { $dllDirs = @($dllDirs + @($cfg.lib_dirs) | Select-Object -Unique) }
            else { $dllDirs = @(@($cfg.lib_dirs) + $dllDirs | Select-Object -Unique) }
        }
        $selectedExe = Join-Path (Split-Path -Parent $exe) $selectedFamily.NativeFile
        [IO.Directory]::CreateDirectory((Split-Path -Parent $configPath)) | Out-Null
        $candidateConfig = $configPath + '.prepare-' + [Guid]::NewGuid().ToString('N')
        if (Test-Path -LiteralPath $configPath -PathType Leaf) { Copy-Item -LiteralPath $configPath -Destination $candidateConfig -ErrorAction Stop }
        $arguments = @('-B', (Join-Path $sourceRoot 'run.py'), '--model-family', $selectedFamily.Name,
                       '--config', $candidateConfig, '--exe', $selectedExe, '--backend', $selectedBackend, '--configure-only')
        $arguments += @('--vision-exe', (Join-Path (Split-Path -Parent $exe) 'tinystrata-vision.exe'))
        foreach ($directory in $dllDirs) { $arguments += @('--lib-dir', $directory) }
        $arguments += @('--port', [string]$ApiPort)
        Write-Host 'Preparing API settings (hardware and VRAM settings remain automatic by default)...'
        $previousTransaction = [Environment]::GetEnvironmentVariable('TINYSTRATA_INSTALLER_TRANSACTION', 'Process')
        try {
            [Environment]::SetEnvironmentVariable('TINYSTRATA_INSTALLER_TRANSACTION', $installationGuard.Token, 'Process')
            & $python @arguments
        } finally { [Environment]::SetEnvironmentVariable('TINYSTRATA_INSTALLER_TRANSACTION', $previousTransaction, 'Process') }
        if ($LASTEXITCODE -ne 0) { throw 'TinyStrata configuration failed.' }
        $prepared = Get-TinyStrataConfig -Path $candidateConfig
        if (-not $prepared -or $prepared.backend -ne $selectedBackend -or $prepared.exe -ne $selectedExe) { throw 'Prepared configuration does not match the selected native runtime.' }
        if (Test-Path -LiteralPath $configPath -PathType Leaf) { [IO.File]::Replace($candidateConfig, $configPath, ($configPath + '.bak')) }
        else { [IO.File]::Move($candidateConfig, $configPath) }
        $preparationCommitted = $true
        $cfg = $prepared
        } catch {
            $preparationError = $_
            try { Remove-TinyStrataNativeGeneration -WorkRoot $workRoot -Directory $generation } catch { Write-Warning "Unpublished native files remain: $generation" }
            try { Complete-TinyStrataPythonTransaction -Transaction $pythonTransaction -Rollback } catch { Write-Warning "Python rollback needs repair from $($pythonTransaction.PythonBackup): $_" }
            Restore-TinyStrataSource -Change $sourceChange
            throw $preparationError
        } finally {
            try {
            if ($preparationCommitted) {
                try { Complete-TinyStrataPythonTransaction -Transaction $pythonTransaction } catch { Write-Warning "The old Python snapshot could not be removed: $($pythonTransaction.PythonBackup)" }
            }
            if ($candidateConfig) {
                foreach ($temporary in @($candidateConfig, ($candidateConfig + '.bak'), ($candidateConfig + '.tmp'))) { if (Test-Path -LiteralPath $temporary -PathType Leaf) { [IO.File]::Delete($temporary) } }
            }
            } finally {
                Exit-TinyStrataInstallationGuard -Guard $installationGuard
                $updateGuard = $null
            }
        }
        if ($DoNotStart) {
            if ($IncludeUI) { Initialize-TinyStrataOptionalUI -SourceRoot $sourceRoot -WorkRoot $workRoot -ApiPort $ApiPort -UiPort $UiPort -Family $selectedFamily.Name -ConfigureOnly }
            Write-Host "Installed. Config: $configPath"
            Write-Host 'Run install.ps1 again to start the API.'
            return
        }
        $stdout = Join-Path $runtimeRoot 'api.stdout.log'
        $stderr = Join-Path $runtimeRoot 'api.stderr.log'
        # Windows command-line quoting is confined to validated path arguments.
        $commandLine = '-B -m serve.server --engine tinystrata --config "{0}" --port {1}' -f $configPath, $ApiPort
        $server = Start-Process -FilePath $python -ArgumentList $commandLine -WorkingDirectory $sourceRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
        $server.Refresh()
        $state = [ordered]@{ pid = $server.Id; started = $server.StartTime.ToUniversalTime().ToString('o'); config = $configPath; source = $sourceRoot; port = $ApiPort }
        [IO.File]::WriteAllText($statePath, ($state | ConvertTo-Json), (New-Object Text.UTF8Encoding($false)))
        Write-Host "Starting the API in the background (PID $($server.Id))..."
    } else {
        Write-Host "Reusing the existing TinyStrata API (PID $($server.Id))."
    }
    $headers = @{}
    $key = if ($env:STRATA_API_KEY) { $env:STRATA_API_KEY } elseif ($cfg -and $cfg.api_key) { @($cfg.api_key)[0] } else { $null }
    if ($key) { $headers.Authorization = 'Bearer ' + $key }
    $deadline = [DateTime]::UtcNow.AddMinutes(3)
    $health = $null
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($server.HasExited) { throw "The API exited. See $(Join-Path $runtimeRoot 'api.stderr.log') and the engine log." }
        try {
            $health = Invoke-TinyStrataApi -BaseUrl $baseUrl -Path '/health'
            if ($health.status -eq 'ok' -and (Test-TinyStrataPortOwner -ApiPort $ApiPort -RootProcessId $server.Id)) { break }
            $health = $null
        } catch { $health = $null }
        Start-Sleep -Milliseconds 500
    }
    if (-not $health) { throw "API startup timed out. See logs in $runtimeRoot." }
    if (-not (Test-TinyStrataInferenceReady -Health $health -Configuration $cfg) -and -not $DoNotDownload) {
        Write-Host ("Installing the pinned {0} model artifacts through the API..." -f $selectedFamily.Name)
        $null = Invoke-TinyStrataApi -BaseUrl $baseUrl -Path '/v1/model/download' -Method POST -Headers $headers
        $deadline = [DateTime]::UtcNow.AddHours(4)
        $lastUpdate = [DateTime]::MinValue
        while ([DateTime]::UtcNow -lt $deadline) {
            if ($server.HasExited) { throw 'The API stopped during model installation.' }
            $job = Invoke-TinyStrataApi -BaseUrl $baseUrl -Path '/v1/model/download' -Headers $headers
            if ($job.state -eq 'ready') { break }
            if ($job.state -in @('error', 'cancelled')) { throw "Model installation $($job.state): $($job.error). Run the installer again to retry/resume." }
            if (([DateTime]::UtcNow - $lastUpdate).TotalSeconds -ge 15) {
                Write-Host ("Model: {0} ({1:N2} / {2:N2} GB)" -f $job.state, ($job.bytes_downloaded / 1e9), ($job.total_bytes / 1e9))
                $lastUpdate = [DateTime]::UtcNow
            }
            Start-Sleep -Seconds 2
        }
        if ($job.state -ne 'ready') { throw 'Model installation timed out; the API remains available for retry.' }
        $health = Invoke-TinyStrataApi -BaseUrl $baseUrl -Path '/health'
        if (-not (Test-TinyStrataInferenceReady -Health $health -Configuration $cfg)) {
            throw 'The download finished but inference is not ready for the selected vision mode; inspect the API logs.'
        }
    }
    Write-Host "API:    $baseUrl/v1"
    Write-Host "Config: $configPath"
    Write-Host "Logs:   $runtimeRoot"
    if ($health.loaded) { Write-Host 'TinyStrata is ready for inference.' } else { Write-Host 'API ready; model installation was skipped.' }
    if ($IncludeUI) { Initialize-TinyStrataOptionalUI -SourceRoot $sourceRoot -WorkRoot $workRoot -ApiPort $ApiPort -UiPort $UiPort -Family $selectedFamily.Name }
    } catch {
        $updateFailure = $_
        if (-not $ownedServices -and $_.Exception.Data.Contains('TinyStrataServices')) { $ownedServices = $_.Exception.Data['TinyStrataServices'] }
        throw
    } finally {
        if ($updateGuard) { Exit-TinyStrataInstallationGuard -Guard $updateGuard }
        if ($ownedServices) {
            try { Restore-TinyStrataOwnedServicesAfterUpdate -Snapshot $ownedServices -SourceRoot $sourceRoot -WorkRoot $workRoot -NoBrowser }
            catch {
                if ($updateFailure) { Write-Warning ("The update failed and its previous services could not be restored: " + $_.Exception.Message) }
                else { throw }
            }
        }
    }
}

$tinyStrataOldErrorAction = $ErrorActionPreference
$tinyStrataOldSecurityProtocol = [Net.ServicePointManager]::SecurityProtocol
try {
    $ErrorActionPreference = 'Stop'
    if (-not $SourceDir) {
        if ($PSScriptRoot -and (Test-TinyStrataSource -Directory $PSScriptRoot)) { $SourceDir = $PSScriptRoot }
        elseif (Test-TinyStrataSource -Directory (Get-Location).Path) { $SourceDir = (Get-Location).Path }
        else { $SourceDir = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'TinyStrata' }
    }
    if ($WithUI) {
        $tinyStrataSettingsWork = if ($WorkDir) { $WorkDir } else { $SourceDir + '-work' }
        $tinyStrataSettingsPath = Join-Path $tinyStrataSettingsWork 'runtime\ui\settings.json'
        if (Test-Path -LiteralPath $tinyStrataSettingsPath -PathType Leaf) {
            try {
                $tinyStrataSavedSettings = Get-Content -LiteralPath $tinyStrataSettingsPath -Raw | ConvertFrom-Json
                if (-not $PSBoundParameters.ContainsKey('ModelFamily')) {
                    $tinyStrataSavedFamily = [string]$tinyStrataSavedSettings.model_family
                    $ModelFamily = (Get-TinyStrataModelFamily -Name $tinyStrataSavedFamily).Name
                }
                if (-not $PSBoundParameters.ContainsKey('UiPort') -and $tinyStrataSavedSettings.port -ge 1 -and $tinyStrataSavedSettings.port -le 65535) {
                    $UiPort = [int]$tinyStrataSavedSettings.port
                }
            } catch { } # Invalid/retired selections use the normal supported default.
        }
    }
    Invoke-TinyStrataInstaller -SourceDirectory $SourceDir -WorkingDirectory $WorkDir -ApiPort $Port -DoNotStart $NoStart.IsPresent -DoNotDownload ($WithUI.IsPresent -or $SkipModelDownload.IsPresent) -InspectOnly $CheckOnly.IsPresent -BuildAgain $Rebuild.IsPresent -RefreshSource $Update.IsPresent -Family $ModelFamily -Backend $Backend -HipDirectory $HipDirectory -HipArchitectures $HipArchitectures -IncludeUI $WithUI.IsPresent -UiPort $UiPort -Revision $SourceRevision -FamilyExplicit $PSBoundParameters.ContainsKey('ModelFamily') -UiPortExplicit $PSBoundParameters.ContainsKey('UiPort')
} finally {
    $ErrorActionPreference = $tinyStrataOldErrorAction
    [Net.ServicePointManager]::SecurityProtocol = $tinyStrataOldSecurityProtocol
}
