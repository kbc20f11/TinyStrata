#Requires -Version 5.1
<# Pure fixture update transactions; no native executable, GPU, download or real installation is started. #>
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'installer-hip-runtime.ps1')
foreach ($name in @('Invoke-TinyStrataInstaller', 'Get-TinyStrataConfig', 'Get-TinyStrataModelFamily', 'Restore-TinyStrataSource', 'Assert-TinyStrataInstallationIdle', 'New-TinyStrataNativeGeneration', 'Remove-TinyStrataNativeGeneration', 'Get-TinyStrataPrivateHipDirectory', 'Enter-TinyStrataInstallationGuard', 'Exit-TinyStrataInstallationGuard', 'Backup-TinyStrataPython', 'Complete-TinyStrataPythonTransaction')) {
    $definition = $runtimeAst.Find({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name}, $true)
    if (-not $definition) { throw "Missing function $name" }
    Invoke-Expression $definition.Extent.Text
}
$transactionRoot = Join-Path ([IO.Path]::GetTempPath()) ('tinystrata-update-test-' + [Guid]::NewGuid().ToString('N'))
$script:fixtureProcesses = @()
$script:fixtureFailure = ''
$script:fixtureEvents = @()
$script:fixtureNativeChanged = $false
$originalTransactionToken = [Environment]::GetEnvironmentVariable('TINYSTRATA_INSTALLER_TRANSACTION', 'Process')
function Test-TinyStrataSource { param($Directory) return (Test-Path -LiteralPath $Directory -PathType Container) }
function Get-TinyStrataBackend {param($Requested,$Configuration) return 'hip'}
function Get-TinyStrataSavedProcess {param($StatePath,$ConfigPath,$SourcePath,$ApiPort) return $null}
function Get-CimInstance {param($ClassName,$ErrorAction) return $script:fixtureProcesses}
function Initialize-TinyStrataSource {
    param($SourceRoot,$CacheRoot,[switch]$Refresh,$Revision)
    $backup = $SourceRoot + '.previous'
    [IO.Directory]::Move($SourceRoot, $backup)
    Write-RuntimeFixture $SourceRoot 'revision.txt' 'new-source'
    $script:fixtureNativeChanged = $true
    return [pscustomobject]@{ SourceRoot=$SourceRoot; Backup=$backup }
}
function Get-TinyStrataSourceFingerprint { param($Directory,[switch]$NativeOnly) return 'fixture-native-fingerprint' }
function Test-TinyStrataNativeBuildStamp {
    param($Directory,$Fingerprint,$Backend)
    return -not $script:fixtureNativeChanged
}
function Write-TinyStrataNativeBuildStamp { param($Directory,$SourceRoot,$Backend) }
function Stop-TinyStrataOwnedServicesForUpdate { param($SourceRoot,$WorkRoot) return $null }
function Restore-TinyStrataOwnedServicesAfterUpdate { param($Snapshot,$SourceRoot,$WorkRoot,[switch]$NoBrowser) }
function Initialize-TinyStrataPython {
    param($WorkRoot,$SourceRoot,$CacheRoot,$Transaction)
    if ($script:fixtureScenario -like 'venv-*') {
        $venv=Join-Path $WorkRoot 'runtime\.venv'
        Backup-TinyStrataPython -VenvRoot $venv -Transaction $Transaction
        Write-RuntimeFixture $venv 'requirements-sentinel.txt' 'new-pip-dependencies'
    }
    return 'Invoke-FixturePython'
}
function Ensure-TinyStrataHipToolkit {param($DestinationDirectory,$Python,$Directory,$Architectures) return $script:fixtureSdk}
function Get-TinyStrataHipArchitectures {param($Directory,$Requested) return 'gfx1200'}
function Get-TinyStrataHipRuntimeDirectories {param($Directory) return @($Directory)}
function Test-TinyStrataEngineBackend {param($ExecutablePath,$SearchDirectories,$Backend) return $true}
function Test-TinyStrataVisionBackend {param($ExecutablePath,$SearchDirectories,$Backend) return $true}
function Ensure-TinyStrataBuildTools {param($CacheDirectory)}
function Invoke-TinyStrataPortableBuild {
    param($SourceDirectory,$BuildDirectory,$Backend,$HipDirectory,$HipArchitectures,[switch]$Clean)
    $script:fixtureEvents += ('build:' + $BuildDirectory)
    if ($Clean) { throw 'A staged update must not clean a previous build' }
    if ($script:fixtureFailure -eq 'build') { throw 'fixture build failure' }
    foreach ($name in @('tinystrata.exe','tinystrata-lfm2.exe','tinystrata-gemma4.exe','tinystrata-vision.exe')) { Write-RuntimeFixture $BuildDirectory $name 'new-native' }
    return [pscustomobject]@{Exe=(Join-Path $BuildDirectory 'tinystrata.exe')}
}
function Invoke-FixturePython {
    $configIndex = [Array]::IndexOf($args, '--config')
    $exeIndex = [Array]::IndexOf($args, '--exe')
    $path = [string]$args[$configIndex + 1]
    $guard = Get-Content -LiteralPath (Join-Path $script:fixtureWork '.tinystrata-installing.json') -Raw | ConvertFrom-Json
    Assert-Runtime ($guard.token -eq [Environment]::GetEnvironmentVariable('TINYSTRATA_INSTALLER_TRANSACTION', 'Process')) 'configure-only child receives only its owner transaction token'
    $script:fixtureEvents += 'configure'
    if ($script:fixtureFailure -eq 'configure') { $global:LASTEXITCODE=1; return }
    $cfg = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    $cfg.exe = [string]$args[$exeIndex + 1]
    $cfg.lib_dirs = @($script:fixtureSdk)
    [IO.File]::WriteAllText($path, ($cfg | ConvertTo-Json -Depth 5), (New-Object Text.UTF8Encoding($false)))
    $global:LASTEXITCODE=0
}
function Start-Process { throw 'A fixture must never start a real process' }
try {
    foreach ($scenario in @('configure', 'build', 'updated-configure', 'publish', 'success', 'venv-build', 'venv-success')) {
        $root = Join-Path $transactionRoot $scenario
        $source = Join-Path $root 'source'
        $work = Join-Path $root 'work'
        $script:fixtureWork=$work
        $script:fixtureScenario=$scenario
        $oldEngine = Join-Path $work 'build-hip'
        $oldSdk = Join-Path $root 'sdk-A'
        $script:fixtureSdk = Join-Path $root 'sdk-B'
        $script:fixtureEvents=@()
        $script:fixtureNativeChanged=$false
        $script:fixtureFailure = if ($scenario -eq 'updated-configure') { 'configure' } elseif ($scenario -eq 'venv-build') { 'build' } elseif ($scenario -in @('success','publish','venv-success')) { '' } else { $scenario }
        Write-RuntimeFixture $source 'revision.txt' 'old-source'
        New-RuntimeSdk $oldSdk
        New-RuntimeSdk $script:fixtureSdk
        Write-RuntimeFixture $script:fixtureSdk 'amdhip64_7.dll' 'MZ sdk-B rocm_kpack.dll amd_comgr.dll'
        foreach ($name in @('tinystrata.exe','tinystrata-lfm2.exe','tinystrata-gemma4.exe','tinystrata-vision.exe')) { Write-RuntimeFixture $oldEngine $name 'old-native' }
        Install-TinyStrataHipRuntime @($oldSdk) $oldEngine
        $oldRuntimeHash = (Get-FileHash -LiteralPath (Join-Path $oldEngine 'amdhip64_7.dll')).Hash
        $oldManifest = [IO.File]::ReadAllText((Join-Path $oldEngine '.tinystrata-hip-runtime.json'))
        $configDir = Join-Path $work 'runtime\.tinystrata'
        $configPath = Join-Path $configDir 'config.json'
        [IO.Directory]::CreateDirectory($configDir) | Out-Null
        $oldCfg = [ordered]@{backend='hip';port=18080;exe=(Join-Path $oldEngine 'tinystrata.exe');lib_dirs=@($oldSdk)}
        [IO.File]::WriteAllText($configPath, ($oldCfg | ConvertTo-Json), (New-Object Text.UTF8Encoding($false)))
        $oldConfigText = [IO.File]::ReadAllText($configPath)
        Write-RuntimeFixture (Join-Path $work 'runtime\.venv') 'requirements-sentinel.txt' 'old-pip-dependencies'
        [Environment]::SetEnvironmentVariable('TINYSTRATA_INSTALLER_TRANSACTION', 'preexisting-parent-token', 'Process')
        $configLock = if ($scenario -eq 'publish') { [IO.File]::Open($configPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read) } else { $null }
        $failed = $false
        try {
            Invoke-TinyStrataInstaller -SourceDirectory $source -WorkingDirectory $work -ApiPort 18080 -DoNotStart $true -DoNotDownload $true -InspectOnly $false -BuildAgain $false -RefreshSource ($scenario -in @('build','updated-configure','venv-build')) -Backend hip -HipDirectory $script:fixtureSdk
        } catch {
            $failed=$true
            if ($scenario -ne 'publish' -and $_.Exception.Message -notlike '*fixture build failure*' -and $_.Exception.Message -notlike '*configuration failed*') { throw }
        } finally { if ($configLock) { $configLock.Dispose() } }
        Assert-Runtime ($failed -eq ($scenario -notin @('success','venv-success'))) "$scenario transaction result"
        Assert-Runtime ([Environment]::GetEnvironmentVariable('TINYSTRATA_INSTALLER_TRANSACTION', 'Process') -eq 'preexisting-parent-token') "$scenario restores the parent token"
        Assert-Runtime (-not (Test-Path -LiteralPath (Join-Path $work '.tinystrata-installing.json'))) "$scenario releases and removes the installation guard"
        $expectedDependencies=if($scenario -eq 'venv-success'){'new-pip-dependencies'}else{'old-pip-dependencies'}
        Assert-Runtime ([IO.File]::ReadAllText((Join-Path $work 'runtime\.venv\requirements-sentinel.txt')) -eq $expectedDependencies) "$scenario keeps Python dependencies coherent with the published config"
        Assert-Runtime (@(Get-ChildItem -LiteralPath (Join-Path $work 'runtime') -Directory -Filter '.venv.rollback-*').Count -eq 0) "$scenario removes finished Python snapshots"
        Assert-Runtime ((Get-FileHash -LiteralPath (Join-Path $oldEngine 'amdhip64_7.dll')).Hash -eq $oldRuntimeHash -and [IO.File]::ReadAllText((Join-Path $oldEngine '.tinystrata-hip-runtime.json')) -eq $oldManifest) "$scenario preserves the previous core/manifest"
        Assert-Runtime ([IO.File]::ReadAllText((Join-Path $oldEngine 'tinystrata.exe')) -eq 'old-native') "$scenario never cleans or overwrites previous native binaries"
        Assert-Runtime (@(Get-ChildItem -LiteralPath $configDir -Filter '*.prepare-*' -File).Count -eq 0) "$scenario cleans unpublished config files"
        if ($failed) {
            Assert-Runtime ([IO.File]::ReadAllText($configPath) -eq $oldConfigText) "$scenario retains the old config"
            Assert-Runtime ([IO.File]::ReadAllText((Join-Path $source 'revision.txt')) -eq 'old-source') "$scenario restores previous sources"
            Assert-Runtime (@(Get-ChildItem -LiteralPath (Join-Path $work 'native') -Directory -ErrorAction SilentlyContinue).Count -eq 0) "$scenario removes the unpublished generation"
        } else {
            $published = Get-TinyStrataConfig $configPath
            Assert-Runtime ($published.exe -ne $oldCfg.exe -and $published.lib_dirs[0] -eq $script:fixtureSdk) 'successful SDK switch publishes the new generation and SDK together'
            $publishedManifest = Get-Content -LiteralPath (Join-Path (Split-Path -Parent $published.exe) '.tinystrata-hip-runtime.json') -Raw | ConvertFrom-Json
            Assert-Runtime ($publishedManifest.lib_dirs[0] -eq [IO.Path]::GetFullPath($script:fixtureSdk)) 'successful generation pins SDK B math and core'
            Assert-Runtime (-not (Test-Path -LiteralPath (Join-Path (Split-Path -Parent $published.exe) 'unrelated.dll'))) 'generation never copies unrelated SDK binaries'
        }
    }
    # The check covers other model families, resident vision and the UI before mutation.
    foreach ($process in @(
        [pscustomobject]@{Name='python.exe';ProcessId=41;ExecutablePath='';CommandLine="python -m serve.server --config $transactionRoot\work\runtime\.tinystrata\config-clover.json"},
        [pscustomobject]@{Name='tinystrata-vision.exe';ProcessId=42;ExecutablePath="$transactionRoot\work\native\hip-old\tinystrata-vision.exe";CommandLine=''},
        [pscustomobject]@{Name='python.exe';ProcessId=43;ExecutablePath='';CommandLine="python $transactionRoot\source\ui\launch.py --work-dir $transactionRoot\work"}
    )) {
        $script:fixtureProcesses=@($process)
        Assert-RuntimeThrows { Assert-TinyStrataInstallationIdle -SourceRoot "$transactionRoot\source" -WorkRoot "$transactionRoot\work" } 'Stop TinyStrata'
    }
    $script:fixtureProcesses=@([pscustomobject]@{Name='python.exe';ProcessId=44;ExecutablePath='';CommandLine='python C:\another-install\ui\launch.py'})
    Assert-TinyStrataInstallationIdle -SourceRoot "$transactionRoot\source" -WorkRoot "$transactionRoot\work"
    Assert-Runtime $true 'another installation is left alone'
    $script:packageMatch=$true
    function Test-TinyStrataRocmPackages {param($Python,$Names) return $script:packageMatch}
    $sdkEnvironment=Join-Path $transactionRoot 'private-rocm-venv'
    Write-RuntimeFixture (Join-Path $sdkEnvironment 'Scripts') 'rocm-sdk.exe' 'old-sdk-tool-never-executed'
    Assert-Runtime ((Get-TinyStrataPrivateHipDirectory $sdkEnvironment) -eq $sdkEnvironment) 'an existing matching pin reuses the legacy private SDK'
    $script:packageMatch=$false
    Assert-Runtime ((Get-TinyStrataPrivateHipDirectory $sdkEnvironment) -eq ($sdkEnvironment+'-10.2.0a20260930')) 'a pin change selects a separate versioned SDK directory'
    Assert-Runtime ([IO.File]::ReadAllText((Join-Path $sdkEnvironment 'Scripts\rocm-sdk.exe')) -eq 'old-sdk-tool-never-executed') 'pin selection never mutates the previous SDK'
    function Get-TinyStrataSavedProcess {param($StatePath,$ConfigPath,$SourcePath,$ApiPort) return [pscustomobject]@{Id=51}}
    function Stop-TinyStrataOwnedServicesForUpdate {
        param($SourceRoot,$WorkRoot)
        $failure = New-Object InvalidOperationException 'fixture verified service quiesce requested'
        $failure.Data['TinyStrataServices'] = [pscustomobject]@{ Stopped = $true }
        throw $failure
    }
    function Restore-TinyStrataOwnedServicesAfterUpdate { param($Snapshot,$SourceRoot,$WorkRoot,[switch]$NoBrowser) throw 'fixture restore also failed' }
    Assert-RuntimeThrows {
        Invoke-TinyStrataInstaller -SourceDirectory $source -WorkingDirectory $work -ApiPort 18080 -DoNotStart $true -DoNotDownload $true -InspectOnly $false -BuildAgain $false -RefreshSource $false -Backend hip -HipDirectory $script:fixtureSdk
    } 'fixture verified service quiesce requested'
    Assert-Runtime (-not (Test-Path -LiteralPath (Join-Path $work '.tinystrata-installing.json'))) 'a secondary restore failure preserves the original error and releases the guard'
    $guardWork=Join-Path $transactionRoot 'guard-ownership'
    $guard=Enter-TinyStrataInstallationGuard $guardWork
    try {
        $record=Get-Content -LiteralPath $guard.Path -Raw | ConvertFrom-Json
        Assert-Runtime ($record.schema -eq 1 -and $record.pid -eq $PID -and $record.started_unix -gt 0 -and $record.token -eq $guard.Token) 'guard durably records the installer process identity'
        Assert-RuntimeThrows { Enter-TinyStrataInstallationGuard $guardWork } 'Another installer may be preparing'
    } finally { Exit-TinyStrataInstallationGuard $guard }
    Write-RuntimeFixture $guardWork '.tinystrata-installing.json' '{malformed stale guard'
    $repaired=Enter-TinyStrataInstallationGuard $guardWork
    Assert-Runtime ((Get-Content -LiteralPath $repaired.Path -Raw | ConvertFrom-Json).token -eq $repaired.Token) 'an unlocked stale or malformed guard is repaired by a new installer'
    Exit-TinyStrataInstallationGuard $repaired
    Assert-Runtime (-not (Test-Path -LiteralPath $repaired.Path)) 'repaired guard is removed at completion'
    Write-Host "Installer update transaction fixtures: $runtimeChecks total checks passed."
} finally {
    [Environment]::SetEnvironmentVariable('TINYSTRATA_INSTALLER_TRANSACTION', $originalTransactionToken, 'Process')
    $target=[IO.Path]::GetFullPath($transactionRoot)
    $temporaryRoot=[IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\')+'\'
    if (-not $target.StartsWith($temporaryRoot,[StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid fixture cleanup path' }
    if(Test-Path -LiteralPath $target){Remove-Item -LiteralPath $target -Recurse -Force}
}
