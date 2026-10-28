#Requires -Version 5.1
<#
.SYNOPSIS
Install and start TinyStrata on Windows.
.DESCRIPTION
Run with: irm 'https://raw.githubusercontent.com/kbc20f11/TinyStrata/refs/heads/main/install.ps1' | iex
Uses local sources when available, otherwise downloads them from GitHub.
Other downloads use official Python, NVIDIA, AMD and Microsoft endpoints.
The selected GPU driver must already be installed. AMD builds use an existing
ROCm/HIP SDK or prepare a private pinned AMD SDK in the work directory. No persistent PATH or execution
policy changes are made. Runtime data stays in the sibling TinyStrata-work folder.
#>
[CmdletBinding()]
param(
    [string]$SourceDir,
    [string]$WorkDir,
    [ValidateSet('auto', 'cuda', 'hip')][string]$Backend = 'auto',
    [string]$HipDirectory,
    [ValidatePattern('^gfx[0-9a-f]+(;gfx[0-9a-f]+)*$')][string]$HipArchitectures,
    [ValidateSet('qwen36', 'qwen3.6', 'lfm2', 'lfm2.5', 'gemma4', 'gemma-4')][string]$ModelFamily = 'qwen36',
    [ValidateRange(0, 65535)][int]$Port = 0,
    [switch]$WithUI,
    [ValidateRange(1, 65535)][int]$UiPort = 3000,
    [switch]$NoStart,
    [switch]$SkipModelDownload,
    [switch]$CheckOnly,
    [switch]$Rebuild,
    [switch]$Update
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
# Reference: https://docs.nvidia.com/cuda/archive/13.2.0/cuda-installation-guide-microsoft-windows/index.html

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

function Initialize-TinyStrataSource {
    param([string]$SourceRoot, [string]$CacheRoot, [switch]$Refresh)
    $SourceRoot = [IO.Path]::GetFullPath($SourceRoot).TrimEnd('\')
    $existing = Test-TinyStrataSource -Directory $SourceRoot
    if ($existing -and -not $Refresh) { return }
    if ($existing) {
        $sourceItem = Get-Item -LiteralPath $SourceRoot -Force
        if (($sourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
            (Test-Path -LiteralPath (Join-Path $SourceRoot '.git'))) {
            throw 'Update requires an archive installation, not a Git checkout or linked source directory.'
        }
    } else { Assert-TinyStrataSourceDestination -Directory $SourceRoot }
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    $commit = Invoke-RestMethod -Uri 'https://api.github.com/repos/kbc20f11/TinyStrata/commits/main' `
        -Headers @{ 'User-Agent' = 'TinyStrata-Installer'; Accept = 'application/vnd.github+json' } -TimeoutSec 30
    $revision = [string]$commit.sha
    if ($revision -notmatch '^[a-f0-9]{40}$') { throw 'GitHub returned an invalid TinyStrata revision.' }
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
    if ($existing) {
        # Replace archive sources reversibly; runtime settings and models stay
        # in the separate work directory. Never update a Git checkout in place.
        $sourceItem = Get-Item -LiteralPath $SourceRoot -Force
        if (($sourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
            (Test-Path -LiteralPath (Join-Path $SourceRoot '.git'))) {
            throw 'The source directory changed while downloading; it was left unchanged.'
        }
        $backupParent = [IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $CacheRoot) 'source-backups'))
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

function Test-TinyStrataCudaToolkit {
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$Directory)
    $compiler = Join-Path $Directory 'bin\nvcc.exe'
    if (-not (Test-Path -LiteralPath $compiler -PathType Leaf)) { return $false }
    $text = (& $compiler --version 2>$null) -join [Environment]::NewLine
    if ($LASTEXITCODE -ne 0 -or $text -notmatch 'release\s+(\d+\.\d+)' -or [version]$Matches[1] -lt [version]'13.2') { return $false }
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

    Write-Host 'Downloading the CUDA compiler and libraries from NVIDIA (no driver or system CUDA installation is changed).'
    $baseUri = 'https://developer.download.nvidia.com/compute/cuda/redist/'
    $manifestPath = Join-Path $CacheDirectory 'redistrib_13.2.0.json'
    Get-TinyStrataOfficialFile -Uri ($baseUri + 'redistrib_13.2.0.json') -Destination $manifestPath
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $destinationRoot = [IO.Path]::GetFullPath($DestinationDirectory)
    New-Item -ItemType Directory -Path $destinationRoot -Force | Out-Null
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    foreach ($name in @('cuda_nvcc', 'cuda_cudart', 'cuda_cccl', 'cuda_crt', 'libnvvm', 'libcublas')) {
        $package = $manifest.$name.'windows-x86_64'
        if (-not $package -or $package.relative_path -notmatch '^[A-Za-z0-9_./-]+\.zip$' -or $package.relative_path -match '(?:^|/)\.\.(?:/|$)' -or $package.sha256 -notmatch '^[a-fA-F0-9]{64}$') { throw "NVIDIA CUDA manifest has an invalid package: $name" }
        $archive = Join-Path $CacheDirectory ($name + '-13.2.zip')
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

function Ensure-TinyStrataHipToolkit {
    param([string]$DestinationDirectory, [string]$Python, [string]$Directory, [string]$Architectures)
    # An explicitly supplied path is a deliberate selection; never replace it
    # with a downloaded SDK when it is misspelled or incomplete.
    if ($Directory) { return Find-TinyStrataHipToolkit -Directory $Directory }
    $venvPython = Join-Path $DestinationDirectory 'Scripts\python.exe'
    $sdk = Join-Path $DestinationDirectory 'Scripts\rocm-sdk.exe'
    if (-not (Test-Path -LiteralPath $sdk -PathType Leaf)) {
        try { return Find-TinyStrataHipToolkit } catch { }
    }
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        & $Python -I -m venv $DestinationDirectory | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the private AMD SDK environment.' }
    }
    $basePackages = @('rocm', 'rocm-sdk-devel', 'rocm-sdk-libraries')
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
        if ($LASTEXITCODE -ne 0 -or $versionText -notmatch 'release\s+(\d+\.\d+)') { throw 'Cannot determine the CUDA toolkit version.' }
        if ([version]$Matches[1] -lt [version]'13.2') { throw 'This portable build helper requires CUDA 13.2 or newer.' }
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

function Initialize-TinyStrataPython {
    param(
        [Parameter(Mandatory=$true)][string]$WorkRoot,
        [Parameter(Mandatory=$true)][string]$SourceRoot,
        [Parameter(Mandatory=$true)][string]$CacheRoot
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
    & $venvPython -I -B -m pip --isolated --disable-pip-version-check --no-input install `
        --index-url https://pypi.org/simple --only-binary=:all: `
        -r $requirements | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'API dependency installation failed.' }
    & $venvPython -I -B -m pip --isolated --disable-pip-version-check check | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'API dependency verification failed.' }
    return $venvPython
}

function Initialize-TinyStrataRuntime {
    param(
        [Parameter(Mandatory=$true)][string]$RuntimeRoot,
        [Parameter(Mandatory=$true)][string]$CacheRoot,
        [string[]]$SearchDirectories = @(),
        [string]$EngineDirectory,
        [ValidateSet('cuda', 'hip')][string]$Backend = 'cuda'
    )
    [IO.Directory]::CreateDirectory($RuntimeRoot) | Out-Null
    [IO.Directory]::CreateDirectory($CacheRoot) | Out-Null
    $cudaDirectories = New-Object System.Collections.Generic.List[string]
    $search = @($SearchDirectories | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Container) } | Select-Object -Unique)
    $haveCuda = $true
    foreach ($dll in $(if ($Backend -eq 'hip') { @('amdhip64*.dll', 'hipblas.dll', 'rocblas.dll') } else { @('cudart64_13.dll', 'cublas64_13.dll', 'cublasLt64_13.dll') })) {
        $found = $null
        foreach ($directory in $search) {
            $candidate = @(Get-ChildItem -LiteralPath $directory -Filter $dll -File -ErrorAction SilentlyContinue)
            if ($candidate.Count -gt 0) { $found = $directory; break }
        }
        if (-not $found) { $haveCuda = $false; break }
        $cudaDirectories.Add($found)
    }
    if ($Backend -eq 'hip' -and -not $haveCuda) { throw 'The selected AMD SDK is missing HIP/hipBLAS/rocBLAS runtime DLLs in its bin/lib folders.' }
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
    # Pinned official CUDA 13.2 redistributables. Only runtime DLLs are
    # extracted; this does not install the full toolkit or NVIDIA driver.
    $packages = @(
        @{
            Name = 'cuda_cudart-13.2.zip'
            Uri = 'https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-13.2.51-archive.zip'
            Sha256 = '906b430b53e6396f7cbaa6874fbe7f9307a184e830b119a4ad3f58dc6e0165c6'
            Pattern = '/bin/(x64/)?[^/]+\.dll$'
        },
        @{
            Name = 'libcublas-13.2.zip'
            Uri = 'https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-13.3.0.5-archive.zip'
            Sha256 = '438a88f07af3721eda15ed1f4d2b957ea810e29061b2a6292837f35520dbd212'
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
        default { throw "Unsupported TinyStrata model family: $Name" }
    }
}

function Initialize-TinyStrataUiSource {
    param([string]$SourceRoot, [string]$WorkRoot)
    $uiRoot = Join-Path $SourceRoot 'ui'
    $runner = Join-Path $SourceRoot 'run-ui.bat'
    $required = @('ui\launch.py', 'ui\server.py', 'ui\web\index.html', 'run-ui.bat')
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

function Initialize-TinyStrataOptionalUI {
    param([string]$SourceRoot, [string]$WorkRoot, [int]$ApiPort,
          [int]$UiPort = 3000, [string]$Family = 'qwen36', [switch]$ConfigureOnly)
    $launcher = Join-Path $SourceRoot 'ui\launch.py'
    $python = Join-Path $WorkRoot 'runtime\.venv\Scripts\python.exe'
    Initialize-TinyStrataUiSource -SourceRoot $SourceRoot -WorkRoot $WorkRoot
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "The UI needs the installed Python runtime: $python" }
    $arguments = @('-B', $launcher, '--source-dir', $SourceRoot, '--work-dir', $WorkRoot,
                   '--api-port', [string]$ApiPort, '--port', [string]$UiPort, '--model-family', $Family)
    if ($ConfigureOnly) { $arguments += '--configure-only' }
    else { $arguments += '--no-api-start' } # The installer already prepared the API, including intentional unloaded mode.
    & $python @arguments | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) {
        throw 'The optional UI could not be prepared or started. The API was not stopped; see the UI logs in the work directory.'
    }
    if ($ConfigureOnly) { Write-Host "UI prepared. Start it with $(Join-Path $SourceRoot 'run-ui.bat')." }
}

function Invoke-TinyStrataInstaller {
    param([string]$SourceDirectory, [string]$WorkingDirectory, [int]$ApiPort,
          [bool]$DoNotStart, [bool]$DoNotDownload, [bool]$InspectOnly, [bool]$BuildAgain, [bool]$RefreshSource,
          [string]$Family = 'qwen36', [string]$Backend = 'auto', [string]$HipDirectory, [string]$HipArchitectures,
          [bool]$IncludeUI = $false, [int]$UiPort = 3000)
    if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitProcess) { throw 'Run this installer in 64-bit Windows PowerShell or PowerShell.' }
    $sourceRoot = [IO.Path]::GetFullPath($SourceDirectory).TrimEnd('\')
    $sourceReady = Test-TinyStrataSource -Directory $sourceRoot
    if (-not $sourceReady) { Assert-TinyStrataSourceDestination -Directory $sourceRoot }
    if (-not $WorkingDirectory) { $WorkingDirectory = $sourceRoot + '-work' }
    $workRoot = [IO.Path]::GetFullPath($WorkingDirectory).TrimEnd('\')
    if ($workRoot -eq $sourceRoot -or $workRoot.StartsWith($sourceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'WorkDir must be outside the source directory.' }
    $runtimeRoot = Join-Path $workRoot 'runtime'
    $selectedFamily = Get-TinyStrataModelFamily -Name $Family
    $configPath = Join-Path (Join-Path $runtimeRoot '.tinystrata') $selectedFamily.ConfigFile
    $statePath = Join-Path $runtimeRoot $selectedFamily.StateFile
    $cfg = Get-TinyStrataConfig -Path $configPath
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
        if ($selectedBackend -eq 'hip') { Write-Host 'AMD: reuse an existing SDK or prepare a private ROCm/HIP SDK; KV is INT8.' }
        if (-not $DoNotStart) { Write-Host 'Then start the API and install the fixed model through its API if needed.' }
        if ($IncludeUI) { Write-Host $(if ($DoNotStart) { 'Prepare the optional UI launcher without starting services.' } else { 'Then start the optional UI and open it in the browser.' }) }
        return
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
    if ($server -and ($BuildAgain -or $RefreshSource)) { throw 'Stop the API for this installation before requesting an update or rebuild.' }
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
        if ($selectedBackend -eq 'cuda') {
            $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
            if (-not $smi) { throw 'A working NVIDIA GPU driver is required for -Backend cuda.' }
            & $smi.Source -L | ForEach-Object { Write-Host $_ }
            if ($LASTEXITCODE -ne 0) { throw 'The NVIDIA driver is not working. GPU drivers are not replaced by this installer.' }
        }
        $cacheRoot = Join-Path $workRoot 'downloads'
        if (-not $sourceReady -or $RefreshSource) { Initialize-TinyStrataSource -SourceRoot $sourceRoot -CacheRoot $cacheRoot -Refresh:$RefreshSource }
        $python = Initialize-TinyStrataPython -WorkRoot $workRoot -SourceRoot $sourceRoot -CacheRoot $cacheRoot
        if ($selectedBackend -eq 'hip') {
            $hipRoot = Ensure-TinyStrataHipToolkit -DestinationDirectory (Join-Path $workRoot 'rocm-venv') -Python $python -Directory $HipDirectory -Architectures $HipArchitectures
            $hipArchs = Get-TinyStrataHipArchitectures -Directory $hipRoot -Requested $HipArchitectures
            Write-Host "AMD HIP SDK: $hipRoot ($hipArchs); INT8 KV."
        }
        $search = @()
        if ($cfg -and $cfg.lib_dirs -and $cfg.backend -eq $selectedBackend) { $search += @($cfg.lib_dirs) }
        $cudaRoots = @()
        if (Test-Path -LiteralPath $workRoot -PathType Container) {
            $cudaRoots += @(Get-ChildItem -LiteralPath $workRoot -Directory -Filter 'cuda-*' | Sort-Object Name -Descending | ForEach-Object { $_.FullName })
        }
        if ($env:CUDA_PATH) { $cudaRoots += $env:CUDA_PATH }
        foreach ($cudaRoot in $cudaRoots) { $search += Join-Path $cudaRoot 'bin'; $search += Join-Path $cudaRoot 'bin\x64' }
        if ($selectedBackend -eq 'hip') { $search = @(Get-TinyStrataHipRuntimeDirectories -Directory $hipRoot) }
        $search += $runtimeRoot
        $exe = $null
        if (-not $BuildAgain -and -not $RefreshSource) {
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
            if ($selectedBackend -eq 'hip') {
                $build = Invoke-TinyStrataPortableBuild -SourceDirectory $sourceRoot -BuildDirectory (Join-Path $workRoot 'build-hip') -Backend hip -HipDirectory $hipRoot -HipArchitectures $hipArchs -Clean:($BuildAgain -or $RefreshSource)
            } else {
                $cuda = Ensure-TinyStrataCudaToolkit -DestinationDirectory (Join-Path $workRoot 'cuda-13.2') -CacheDirectory $cacheRoot -SearchDirectories $cudaRoots
                $build = Invoke-TinyStrataPortableBuild -SourceDirectory $sourceRoot -BuildDirectory (Join-Path $workRoot 'build') -CudaDirectory $cuda -Clean:($BuildAgain -or $RefreshSource)
                $search += Join-Path $cuda 'bin'; $search += Join-Path $cuda 'bin\x64'
            }
            $exe = $build.Exe
        }
        $search += Split-Path -Parent $exe
        $search += $runtimeRoot
        $dllDirs = @(Initialize-TinyStrataRuntime -RuntimeRoot $runtimeRoot -CacheRoot $cacheRoot -SearchDirectories $search -EngineDirectory (Split-Path -Parent $exe) -Backend $selectedBackend)
        if ($cfg -and $cfg.lib_dirs -and $cfg.backend -eq $selectedBackend) { $dllDirs = @(@($cfg.lib_dirs) + $dllDirs | Select-Object -Unique) }
        $selectedExe = Join-Path (Split-Path -Parent $exe) $selectedFamily.NativeFile
        $arguments = @('-B', (Join-Path $sourceRoot 'run.py'), '--model-family', $selectedFamily.Name,
                       '--config', $configPath, '--exe', $selectedExe, '--backend', $selectedBackend, '--configure-only')
        $arguments += @('--vision-exe', (Join-Path (Split-Path -Parent $exe) 'tinystrata-vision.exe'))
        foreach ($directory in $dllDirs) { $arguments += @('--lib-dir', $directory) }
        $arguments += @('--port', [string]$ApiPort)
        Write-Host 'Preparing API settings (hardware and VRAM settings remain automatic by default)...'
        & $python @arguments
        if ($LASTEXITCODE -ne 0) { throw 'TinyStrata configuration failed.' }
        $cfg = Get-TinyStrataConfig -Path $configPath
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
    Invoke-TinyStrataInstaller -SourceDirectory $SourceDir -WorkingDirectory $WorkDir -ApiPort $Port -DoNotStart $NoStart.IsPresent -DoNotDownload $SkipModelDownload.IsPresent -InspectOnly $CheckOnly.IsPresent -BuildAgain $Rebuild.IsPresent -RefreshSource $Update.IsPresent -Family $ModelFamily -Backend $Backend -HipDirectory $HipDirectory -HipArchitectures $HipArchitectures -IncludeUI $WithUI.IsPresent -UiPort $UiPort
} finally {
    $ErrorActionPreference = $tinyStrataOldErrorAction
    [Net.ServicePointManager]::SecurityProtocol = $tinyStrataOldSecurityProtocol
}
