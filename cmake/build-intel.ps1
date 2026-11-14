[CmdletBinding()]
param(
    [string]$BuildDirectory = (Join-Path $PSScriptRoot '..\build-intel'),
    [string]$OneApiRoot = 'C:\Program Files (x86)\Intel\oneAPI',
    [string]$LevelZeroRoot = '',
    [string]$OpenVinoDirectory = '',
    [ValidateSet('', 'intel_gpu_ptl_h')]
    [string]$AotTarget = '',
    [switch]$MklPrefill,
    [switch]$Int8Xmx,
    [switch]$Inspect
)
$ErrorActionPreference = 'Stop'
$sourceDirectory = Split-Path $PSScriptRoot -Parent
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Install Visual Studio C++ Build Tools first.' }
$vsDirectory = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsDirectory) { throw 'Visual Studio C++ tools were not found.' }
$vsEnvironment = Join-Path $vsDirectory 'Common7\Tools\VsDevCmd.bat'
$intelEnvironment = Join-Path $OneApiRoot 'setvars.bat'
if (-not (Test-Path -LiteralPath $intelEnvironment)) { throw "Intel environment not found: $intelEnvironment" }
# Import into this process only; do not rewrite the user's global PATH.
$environmentCommand = 'call "{0}" -arch=x64 -host_arch=x64 >nul && call "{1}" --force >nul && set' -f $vsEnvironment, $intelEnvironment
$environmentLines = & $env:ComSpec /d /s /c $environmentCommand
if ($LASTEXITCODE -ne 0) { throw 'Failed to initialize the MSVC/oneAPI build environment.' }
foreach ($line in $environmentLines) {
    if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
}
$configureArguments = @('-S', $sourceDirectory, '-B', $BuildDirectory, '-G', 'Ninja',
    '-DTINYSTRATA_BACKEND=SYCL', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_COMPILER=icx')
# Always pass the value, including empty, so reusing a build directory without
# -AotTarget restores generic SPIR-V instead of keeping an old cache choice.
$configureArguments += "-DTINYSTRATA_INTEL_AOT_TARGET=$AotTarget"
if ($AotTarget) {
    $oclocCommand = Get-Command ocloc.exe -ErrorAction SilentlyContinue
    if (-not $oclocCommand) {
        $oclocRoot = Join-Path $OneApiRoot 'ocloc'
        $oclocCandidates = @((Join-Path $oclocRoot 'latest\bin\ocloc.exe'))
        if (Test-Path -LiteralPath $oclocRoot -PathType Container) {
            $oclocCandidates += @(Get-ChildItem -LiteralPath $oclocRoot -Directory | Sort-Object Name -Descending |
                ForEach-Object { Join-Path $_.FullName 'bin\ocloc.exe' })
        }
        $oclocExecutable = $oclocCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
        if (-not $oclocExecutable) { throw 'Intel AOT requires OCLOC with PTL-H support; install/initialize the oneAPI OCLOC component.' }
        $env:Path = (Split-Path $oclocExecutable -Parent) + ';' + $env:Path
        $oclocCommand = Get-Command ocloc.exe -ErrorAction Stop
    }
    $configureArguments += "-DTINYSTRATA_INTEL_OCLOC_EXECUTABLE=$($oclocCommand.Source)"
}
if ($LevelZeroRoot) { $configureArguments += "-DCMAKE_PREFIX_PATH=$LevelZeroRoot" }
if ($MklPrefill) { $configureArguments += '-DTINYSTRATA_INTEL_MKL=ON' }
else { $configureArguments += '-DTINYSTRATA_INTEL_MKL=OFF' }
if ($Int8Xmx) { $configureArguments += '-DTINYSTRATA_INTEL_INT8_XMX=ON' }
else { $configureArguments += '-DTINYSTRATA_INTEL_INT8_XMX=OFF' }
if ($OpenVinoDirectory) {
    $configureArguments += '-DTINYSTRATA_INTEL_NPU=ON'
    $configureArguments += "-DOpenVINO_DIR=$OpenVinoDirectory"
    $openVinoLibraries = Join-Path (Split-Path $OpenVinoDirectory -Parent) 'libs'
    $env:Path = $openVinoLibraries + ';' + $env:Path
}
else { $configureArguments += '-DTINYSTRATA_INTEL_NPU=OFF' }
& cmake @configureArguments
if ($LASTEXITCODE -ne 0) { throw 'Intel CMake configuration failed.' }
& cmake --build $BuildDirectory --parallel 4
if ($LASTEXITCODE -ne 0) { throw 'Intel build failed.' }
& ctest --test-dir $BuildDirectory --output-on-failure
if ($LASTEXITCODE -ne 0) { throw 'Intel policy tests failed.' }
if ($Inspect) {
    & (Join-Path $BuildDirectory 'tinystrata-intel-probe.exe') --inspect
    if ($LASTEXITCODE -ne 0) { throw 'Intel device inspection failed.' }
}
