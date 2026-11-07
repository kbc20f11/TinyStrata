#Requires -Version 5.1
<#
Run with powershell.exe or pwsh.exe -NoProfile -File tests/installer-hip-runtime.ps1.
Pure file fixtures: no DLL loading, GPU initialization, network, or installer entrypoint.
#>
$ErrorActionPreference = 'Stop'
$runtimeChecks = 0
$runtimeDownloads = New-Object 'System.Collections.Generic.List[string]'
$runtimeFixture = Join-Path ([IO.Path]::GetTempPath()) ('tinystrata-runtime-test-' + [Guid]::NewGuid().ToString('N'))

function Assert-Runtime {
    param([bool]$Passed, [string]$Name)
    if (-not $Passed) { throw "HIP runtime regression: $Name" }
    $script:runtimeChecks++
}
function Assert-RuntimeThrows {
    param([scriptblock]$Action, [string]$Message)
    $failure = ''
    try { $null = & $Action } catch { $failure = $_.Exception.Message }
    Assert-Runtime ($failure -like ('*' + $Message + '*')) "expected error containing $Message; got $failure"
}
function Write-RuntimeFixture {
    param([string]$Directory, [string]$Name, [string]$Text)
    [IO.Directory]::CreateDirectory($Directory) | Out-Null
    [IO.File]::WriteAllBytes((Join-Path $Directory $Name), [Text.Encoding]::ASCII.GetBytes($Text))
}
function New-RuntimeSdk {
    param([string]$Directory, [switch]$Legacy)
    if ($Legacy) {
        [IO.Directory]::CreateDirectory($Directory) | Out-Null
        [IO.File]::WriteAllBytes((Join-Path $Directory 'amdhip64.dll'), [Text.Encoding]::Unicode.GetBytes('MZ legacy amd_comgr_3.dll'))
        Write-RuntimeFixture $Directory 'amd_comgr_3.dll' 'old-sdk-comgr'
    } else {
        Write-RuntimeFixture $Directory 'amdhip64_7.dll' 'MZ selected-sdk rocm_kpack.dll amd_comgr.dll'
        Write-RuntimeFixture $Directory 'rocm_kpack.dll' 'selected-sdk-kpack'
        Write-RuntimeFixture $Directory 'amd_comgr.dll' 'selected-sdk-comgr'
    }
    Write-RuntimeFixture $Directory 'hipblas.dll' 'selected-sdk-hipblas'
    Write-RuntimeFixture $Directory 'rocblas.dll' 'selected-sdk-rocblas'
    Write-RuntimeFixture $Directory 'unrelated.dll' 'must-not-be-copied'
}
function Write-RuntimePeFixture {
    param([string]$Path, [string]$Import)
    # Minimal PE32+ headers/import descriptors; this fixture is never executed.
    $bytes = New-Object byte[] 1024
    [BitConverter]::GetBytes([uint16]0x5a4d).CopyTo($bytes, 0)
    [BitConverter]::GetBytes([int]128).CopyTo($bytes, 60)
    [BitConverter]::GetBytes([uint32]0x4550).CopyTo($bytes, 128)
    [BitConverter]::GetBytes([uint16]1).CopyTo($bytes, 134)
    [BitConverter]::GetBytes([uint16]240).CopyTo($bytes, 148)
    [BitConverter]::GetBytes([uint16]0x20b).CopyTo($bytes, 152)
    [BitConverter]::GetBytes([uint32]0x1000).CopyTo($bytes, 272)
    [BitConverter]::GetBytes([uint32]0x1000).CopyTo($bytes, 404)
    [BitConverter]::GetBytes([uint32]512).CopyTo($bytes, 408)
    [BitConverter]::GetBytes([uint32]512).CopyTo($bytes, 412)
    [BitConverter]::GetBytes([uint32]0x1040).CopyTo($bytes, 524)
    [Text.Encoding]::ASCII.GetBytes($Import).CopyTo($bytes, 576)
    [IO.File]::WriteAllBytes($Path, $bytes)
}

# Extract only function definitions; never dot-source the install/start body.
$runtimeTokens = $null
$runtimeErrors = $null
$runtimeAst = [System.Management.Automation.Language.Parser]::ParseFile((Join-Path (Split-Path -Parent $PSScriptRoot) 'install.ps1'), [ref]$runtimeTokens, [ref]$runtimeErrors)
if ($runtimeErrors.Count) { throw "Installer syntax errors: $runtimeErrors" }
foreach ($name in @('Get-TinyStrataHipRuntimeFiles', 'Get-TinyStrataHipImports', 'Install-TinyStrataHipRuntime', 'Initialize-TinyStrataRuntime')) {
    $definition = $runtimeAst.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name }, $true)
    if (-not $definition) { throw "Missing installer function $name" }
    Invoke-Expression $definition.Extent.Text
}
# Runtime setup may need the existing CRT fallback, depending on this host.
# Stub its archive extraction, so these fixtures work with either CRT version.
function Get-VerifiedDownload {
    param($Uri, $Destination, $Sha256)
    $script:runtimeDownloads.Add([IO.Path]::GetFileName($Destination))
    return $Destination
}
function Expand-TinyStrataRuntimeArchive {
    param($Archive, $Destination, $IncludePattern, [switch]$Flatten)
    foreach ($name in @('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'vcomp140.dll', 'cudart64_13.dll', 'cublas64_13.dll', 'cublasLt64_13.dll')) {
        Write-RuntimeFixture $Destination $name ('archive-fixture-' + $name)
    }
}

try {
    $sdk = Join-Path $runtimeFixture 'selected SDK with spaces'
    $other = Join-Path $runtimeFixture 'other-sdk'
    $engine = Join-Path $runtimeFixture 'engine'
    New-RuntimeSdk $sdk
    New-RuntimeSdk $other
    Write-RuntimeFixture $other 'amdhip64_7.dll' 'MZ wrong-sdk rocm_kpack.dll amd_comgr.dll'
    foreach ($name in @('amdhip64_7.dll', 'amd_comgr.dll', 'rocm_kpack.dll')) { Write-RuntimeFixture $engine $name 'old-app-local-copy' }
    $null = Initialize-TinyStrataRuntime -Backend hip -SearchDirectories @($sdk, $other) -HipRuntimeDirectories @($sdk) -HipSdkRoot $sdk -EngineDirectory $engine -RuntimeRoot (Join-Path $runtimeFixture 'runtime') -CacheRoot (Join-Path $runtimeFixture 'cache')
    foreach ($name in @('amdhip64_7.dll', 'amd_comgr.dll', 'rocm_kpack.dll')) {
        Assert-Runtime ((Get-FileHash -LiteralPath (Join-Path $sdk $name)).Hash -eq (Get-FileHash -LiteralPath (Join-Path $engine $name)).Hash) "matching selected SDK $name is app-local, ahead of System32/PATH"
    }
    $manifestPath = Join-Path $engine '.tinystrata-hip-runtime.json'
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    Assert-Runtime ($manifest.schema -eq 1 -and $manifest.runtime_dll -eq 'amdhip64_7.dll') 'manifest names the selected runtime'
    Assert-Runtime ($manifest.sdk_root -eq [IO.Path]::GetFullPath($sdk)) 'manifest records the chosen SDK root for runtime discovery'
    Assert-Runtime (@($manifest.lib_dirs).Count -eq 1 -and $manifest.lib_dirs[0] -eq [IO.Path]::GetFullPath($sdk)) 'manifest pins selected SDK directories only'
    Assert-Runtime (@($manifest.files).Count -eq 3 -and @($manifest.files | Where-Object { $_.sha256 -cnotmatch '^[a-f0-9]{64}$' }).Count -eq 0) 'manifest records the entire matching core set with normalized hashes'
    Assert-Runtime (@($manifest.dependencies).Count -eq 2 -and @($manifest.dependencies | Where-Object { -not $_.path.StartsWith([IO.Path]::GetFullPath($sdk)) }).Count -eq 0) 'manifest records selected SDK math identities without copying them'
    Assert-Runtime (-not (Test-Path -LiteralPath (Join-Path $engine 'unrelated.dll'))) 'only allowlisted HIP runtime files copied'
    Assert-Runtime (-not (Test-Path -LiteralPath (Join-Path $engine 'hipblas.dll'))) 'math libraries and their data remain in the SDK'

    $proofPath = Join-Path $engine '.tinystrata-hip-vmm.txt'
    $proofLines = [IO.File]::ReadAllLines($proofPath)
    Assert-Runtime ($proofLines.Count -eq 4 -and $proofLines[0] -ceq 'TINYSTRATA_HIP_VMM_V1') 'native proof has the exact fixed-format header and three records'
    $proofExpected = @($manifest.files | Sort-Object name | ForEach-Object { $_.name.ToLowerInvariant() + ' ' + $_.sha256 })
    Assert-Runtime (($proofLines[1..3] -join ';') -ceq ($proofExpected -join ';')) 'native proof hashes match the JSON-verified SDK core set'

    Assert-Runtime (@($runtimeDownloads | Where-Object { $_ -notlike 'vc-runtime*' }).Count -eq 0) 'HIP setup never downloads CUDA'
    $noMath = Join-Path $runtimeFixture 'selected-sdk-missing-math'
    New-RuntimeSdk $noMath
    Remove-Item -LiteralPath (Join-Path $noMath 'hipblas.dll')
    Assert-RuntimeThrows { Install-TinyStrataHipRuntime @($noMath) $engine } 'HIP/rocBLAS math'
    Assert-Runtime ((Get-FileHash -LiteralPath (Join-Path $engine 'amdhip64_7.dll')).Hash -eq (Get-FileHash -LiteralPath (Join-Path $sdk 'amdhip64_7.dll')).Hash) 'missing SDK math fails before any app-local replacement'
    $noRuntime = Join-Path $runtimeFixture 'selected-sdk-missing-runtime'
    New-RuntimeSdk $noRuntime
    Remove-Item -LiteralPath (Join-Path $noRuntime 'amdhip64_7.dll')
    Assert-RuntimeThrows {
        Initialize-TinyStrataRuntime -Backend hip -SearchDirectories @($noRuntime, $engine, $sdk) -HipRuntimeDirectories @($noRuntime) -EngineDirectory $engine -RuntimeRoot (Join-Path $runtimeFixture 'missing-runtime') -CacheRoot (Join-Path $runtimeFixture 'cache')
    } 'selected AMD SDK is missing HIP'
    Assert-Runtime ((Get-FileHash -LiteralPath (Join-Path $engine 'amdhip64_7.dll')).Hash -eq (Get-FileHash -LiteralPath (Join-Path $sdk 'amdhip64_7.dll')).Hash) 'missing selected SDK runtime cannot reuse an old engine or fallback SDK'
    Assert-RuntimeThrows {
        Initialize-TinyStrataRuntime -Backend hip -SearchDirectories @($sdk) -EngineDirectory $engine -RuntimeRoot (Join-Path $runtimeFixture 'no-sdk-choice') -CacheRoot (Join-Path $runtimeFixture 'cache')
    } 'selected AMD SDK runtime directories are required'

    # Idempotent setup can reuse matching app-local DLLs, including read-only files.
    $readonly = Join-Path $engine 'amdhip64_7.dll'
    (Get-Item -LiteralPath $readonly).IsReadOnly = $true
    Install-TinyStrataHipRuntime -SearchDirectories @($sdk) -EngineDirectory $engine
    Assert-Runtime ((Get-Item -LiteralPath $readonly).IsReadOnly) 'unchanged read-only DLL is reused'
    (Get-Item -LiteralPath $readonly).IsReadOnly = $false

    $incomplete = Join-Path $runtimeFixture 'missing-kpack'
    New-RuntimeSdk $incomplete
    Remove-Item -LiteralPath (Join-Path $incomplete 'rocm_kpack.dll')
    Assert-RuntimeThrows { Install-TinyStrataHipRuntime @($incomplete, $other) $engine } 'rocm_kpack.dll'
    Assert-Runtime ((Get-FileHash -LiteralPath $readonly).Hash -eq (Get-FileHash -LiteralPath (Join-Path $sdk 'amdhip64_7.dll')).Hash) 'missing dependency cannot partially replace runtime or borrow from another SDK'
    $emptyTarget = Join-Path $runtimeFixture 'not-created'
    Assert-RuntimeThrows { Install-TinyStrataHipRuntime @($incomplete) $emptyTarget } 'rocm_kpack.dll'
    Assert-Runtime (-not (Test-Path -LiteralPath $emptyTarget)) 'source-set validation precedes destination writes'
    Remove-Item -LiteralPath (Join-Path $incomplete 'amd_comgr.dll')
    Assert-RuntimeThrows { Install-TinyStrataHipRuntime @($incomplete, $other) $engine } 'COMGR'

    $legacy = Join-Path $runtimeFixture 'legacy-sdk'
    $legacyTarget = Join-Path $runtimeFixture 'legacy-engine'
    New-RuntimeSdk $legacy -Legacy
    Install-TinyStrataHipRuntime @($legacy) $legacyTarget
    Assert-Runtime ((Test-Path -LiteralPath (Join-Path $legacyTarget 'amdhip64.dll')) -and (Test-Path -LiteralPath (Join-Path $legacyTarget 'amd_comgr_3.dll'))) 'legacy non-kpack SDK retains its own runtime/COMGR names'
    Assert-Runtime (-not (Test-Path -LiteralPath (Join-Path $legacyTarget 'rocm_kpack.dll'))) 'kpack is optional for a runtime that does not reference it'
    Assert-Runtime ((Get-Item -LiteralPath (Join-Path $legacyTarget '.tinystrata-hip-vmm.txt')).Length -eq 0) 'older SDK gets no native small-grain proof grant'
    Write-RuntimePeFixture (Join-Path $legacyTarget 'tinystrata.exe') 'amdhip64_7.dll'
    Write-RuntimeFixture $legacyTarget 'amdhip64_7.dll' 'stale-7-must-not-make-old-sdk-compatible'
    Assert-RuntimeThrows { Install-TinyStrataHipRuntime @($legacy) $legacyTarget } 'does not provide amdhip64_7.dll'
    Assert-Runtime ([IO.File]::ReadAllText((Join-Path $legacyTarget 'amdhip64_7.dll')) -eq 'stale-7-must-not-make-old-sdk-compatible') 'an incompatible runtime basename fails before reusing stale app-local DLLs'
    Write-RuntimePeFixture (Join-Path $legacyTarget 'tinystrata.exe') 'amdhip64.dll'
    Install-TinyStrataHipRuntime @($legacy) $legacyTarget
    Assert-Runtime (@(Get-TinyStrataHipImports (Join-Path $legacyTarget 'tinystrata.exe'))[0] -eq 'amdhip64.dll') 'matching legacy native import remains compatible'
    Remove-Item -LiteralPath (Join-Path $legacy 'amd_comgr_3.dll')
    Write-RuntimeFixture $legacy 'amd_comgr.dll' 'wrong-comgr-name'
    Assert-RuntimeThrows { Install-TinyStrataHipRuntime @($legacy) $legacyTarget } 'amd_comgr_3.dll'

    # A read-only target fails before any other target is replaced.
    foreach ($name in @('amdhip64_7.dll', 'amd_comgr.dll', 'rocm_kpack.dll')) { Write-RuntimeFixture $engine $name 'preserved-old-set' }
    (Get-Item -LiteralPath $readonly).IsReadOnly = $true
    Assert-RuntimeThrows { Install-TinyStrataHipRuntime @($sdk) $engine } 'writable'
    Assert-Runtime ([IO.File]::ReadAllText((Join-Path $engine 'amd_comgr.dll')) -eq 'preserved-old-set') 'read-only target does not produce a mixed set'
    (Get-Item -LiteralPath $readonly).IsReadOnly = $false
    # If replacement fails after the first file, the earlier replacement rolls back.
    $lock = [IO.File]::Open($readonly, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
    $oldManifest = [IO.File]::ReadAllText($manifestPath)
    try { Assert-RuntimeThrows { Install-TinyStrataHipRuntime @($sdk) $engine } 'must be writable' } finally { $lock.Dispose() }
    foreach ($name in @('amdhip64_7.dll', 'amd_comgr.dll', 'rocm_kpack.dll')) {
        Assert-Runtime ([IO.File]::ReadAllText((Join-Path $engine $name)) -eq 'preserved-old-set') "failed replacement rolls back $name"
    }
    Assert-Runtime ([IO.File]::ReadAllText($manifestPath) -eq $oldManifest) 'failed core replacement retains the previous manifest too'
    Write-RuntimeFixture $sdk 'amdhip64_7.dll' 'MZ new-sdk-revision rocm_kpack.dll amd_comgr.dll'
    $lock = [IO.File]::Open($manifestPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
    try { Assert-RuntimeThrows { Install-TinyStrataHipRuntime @($sdk) $engine } 'must be writable' } finally { $lock.Dispose() }
    foreach ($name in @('amdhip64_7.dll', 'amd_comgr.dll', 'rocm_kpack.dll')) {
        Assert-Runtime ([IO.File]::ReadAllText((Join-Path $engine $name)) -eq 'preserved-old-set') "manifest commit failure rolls back $name"
    }
    Assert-Runtime ([IO.File]::ReadAllText($manifestPath) -eq $oldManifest) 'manifest commit failure preserves its old bytes'

    $oldProof = [IO.File]::ReadAllBytes($proofPath)
    $lock = [IO.File]::Open($proofPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
    try { Assert-RuntimeThrows { Install-TinyStrataHipRuntime @($sdk) $engine } 'must be writable' } finally { $lock.Dispose() }
    foreach ($name in @('amdhip64_7.dll', 'amd_comgr.dll', 'rocm_kpack.dll')) {
        Assert-Runtime ([IO.File]::ReadAllText((Join-Path $engine $name)) -eq 'preserved-old-set') "proof commit failure rolls back $name"
    }
    Assert-Runtime ([IO.File]::ReadAllText($manifestPath) -eq $oldManifest) 'proof commit failure rolls back the JSON manifest'
    Assert-Runtime ([Convert]::ToBase64String([IO.File]::ReadAllBytes($proofPath)) -eq [Convert]::ToBase64String($oldProof)) 'proof commit failure preserves the old proof'

    Assert-Runtime (@(Get-ChildItem -LiteralPath $engine -Directory -Filter '.tinystrata-hip-*').Count -eq 0) 'successful rollback removes staging files'
    Assert-RuntimeThrows { Install-TinyStrataHipRuntime @($sdk) ([Environment]::SystemDirectory) } 'outside the Windows directory'

    $cuda = Join-Path $runtimeFixture 'cuda-sdk'
    $cudaEngine = Join-Path $runtimeFixture 'cuda-engine'
    foreach ($name in @('cudart64_13.dll', 'cublas64_13.dll', 'cublasLt64_13.dll')) { Write-RuntimeFixture $cuda $name 'cuda-runtime-fixture' }
    $result = @(Initialize-TinyStrataRuntime -Backend cuda -SearchDirectories @($cuda, $incomplete) -EngineDirectory $cudaEngine -RuntimeRoot (Join-Path $runtimeFixture 'cuda-runtime') -CacheRoot (Join-Path $runtimeFixture 'cuda-cache'))
    Assert-Runtime ($result -contains $cuda) 'CUDA retains its existing library search directory'
    Assert-Runtime (-not (Test-Path -LiteralPath (Join-Path $cudaEngine 'amdhip64_7.dll'))) 'CUDA never inspects or installs an available incomplete HIP set'

    # Exercise the installer's actual saved-directory merge without invoking
    # its install/start body. Both SDKs contain math DLLs with the same names.
    $merge = $runtimeAst.Find({ param($node)
        $node -is [System.Management.Automation.Language.IfStatementAst] -and
        $node.Extent.Text.StartsWith('if ($cfg -and $cfg.lib_dirs -and $cfg.backend -eq $selectedBackend)') -and
        $node.Extent.Text.Contains('$dllDirs =')
    }, $true)
    if (-not $merge) { throw 'Missing installer runtime directory merge' }
    foreach ($backend in @('hip', 'cuda')) {
        $merged = @(& {
            param($mergeCode, $backend, $fresh, $saved)
            $selectedBackend = $backend
            $cfg = [pscustomobject]@{ backend = $backend; lib_dirs = $saved }
            $dllDirs = $fresh
            Invoke-Expression $mergeCode
            $dllDirs
        } $merge.Extent.Text $backend @($sdk, $engine) @($other, $sdk))
        $expected = if ($backend -eq 'hip') { @($sdk, $engine, $other) } else { @($other, $sdk, $engine) }
        Assert-Runtime (($merged -join ';') -eq ($expected -join ';')) "$backend saved-directory order and deduplication"
        $firstMath = @($merged | Where-Object { Test-Path -LiteralPath (Join-Path $_ 'hipblas.dll') })[0]
        Assert-Runtime ($firstMath -eq $expected[0]) "$backend resolves math from the intended first SDK"
    }
    $unrelatedBackend = @(& {
        param($mergeCode, $fresh, $saved)
        $selectedBackend = 'hip'
        $cfg = [pscustomobject]@{ backend = 'cuda'; lib_dirs = $saved }
        $dllDirs = $fresh
        Invoke-Expression $mergeCode
        $dllDirs
    } $merge.Extent.Text @($sdk, $engine) @($other))
    Assert-Runtime (($unrelatedBackend -join ';') -eq (@($sdk, $engine) -join ';')) 'different-backend saved directories remain excluded'
    # Run the real SDK-selection function with file fixtures and commands that
    # are mocked before any SDK executable/venv/pip/driver could be invoked.
    & {
        param($ast, $fixtureRoot)
        foreach ($name in @('Ensure-TinyStrataHipToolkit', 'Get-TinyStrataPrivateHipDirectory')) {
            $definition = $ast.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name }, $true)
            Invoke-Expression $definition.Extent.Text
        }
        $script:sdkSelectionCalls = @()
        $script:privateBootstrapArgs = @()
        function Find-TinyStrataHipToolkit {
            param($Directory)
            $script:sdkSelectionCalls += $Directory
            if ($Directory) { return $Directory }
            return $env:HIP_PATH  # Simulated available but obsolete system SDK.
        }
        function Test-TinyStrataRocmPackages { param($Python, $Names) return $true }
        function Get-TinyStrataPrivateHipRoot { param($SdkExecutable) return 'verified-private-sdk-root' }
        function Get-TinyStrataHipArchitectures { param($Directory, $Requested) return 'gfx1200' }
        function Invoke-FixturePrivateBootstrap {
            $script:privateBootstrapArgs = @($args)
            throw 'fixture-private-bootstrap-no-executable-run'
        }
        $previousHipPath = $env:HIP_PATH
        try {
            $env:HIP_PATH = Join-Path $fixtureRoot 'obsolete-system-sdk-7.2'
            $fresh = Join-Path $fixtureRoot 'fresh-private-pin'
            Assert-RuntimeThrows { Ensure-TinyStrataHipToolkit -DestinationDirectory $fresh -Python Invoke-FixturePrivateBootstrap -Architectures gfx1200 } 'fixture-private-bootstrap'
            Assert-Runtime ($script:sdkSelectionCalls.Count -eq 0) 'fresh automatic install does not discover the old global/system SDK'
            Assert-Runtime (($script:privateBootstrapArgs -join ';') -eq (@('-I', '-m', 'venv', $fresh) -join ';')) 'fresh automatic install starts the private pinned SDK bootstrap'
            $cached = Join-Path $fixtureRoot 'cached-private-pin'
            Write-RuntimeFixture (Join-Path $cached 'Scripts') 'rocm-sdk.exe' 'never-executed-sdk-fixture'
            Write-RuntimeFixture (Join-Path $cached 'Scripts') 'python.exe' 'never-executed-python-fixture'
            $selected = Ensure-TinyStrataHipToolkit -DestinationDirectory $cached -Python Invoke-FixturePrivateBootstrap -Architectures gfx1200
            Assert-Runtime ($selected -eq 'verified-private-sdk-root' -and $script:sdkSelectionCalls.Count -eq 0) 'cached matching private pin is reused before old HIP_PATH without pip or GPU probes'
            $custom = Join-Path $fixtureRoot 'explicit-custom-sdk'
            $selected = Ensure-TinyStrataHipToolkit -DestinationDirectory $fresh -Python Invoke-FixturePrivateBootstrap -Directory $custom -Architectures gfx1200
            Assert-Runtime ($selected -eq $custom -and $script:sdkSelectionCalls.Count -eq 1 -and $script:sdkSelectionCalls[0] -eq $custom) 'explicit HipDirectory remains authoritative'
            Assert-Runtime ($env:HIP_PATH -eq (Join-Path $fixtureRoot 'obsolete-system-sdk-7.2')) 'SDK selection never changes global HIP_PATH'
        } finally { $env:HIP_PATH = $previousHipPath }
    } $runtimeAst $runtimeFixture
    Write-Host "Installer HIP runtime fixtures: $runtimeChecks checks passed."
} finally {
    $runtimeAbsolute = [IO.Path]::GetFullPath($runtimeFixture)
    $runtimeTemp = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (-not $runtimeAbsolute.StartsWith($runtimeTemp, [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid fixture cleanup directory.' }
    if (Test-Path -LiteralPath $runtimeAbsolute) { Remove-Item -LiteralPath $runtimeAbsolute -Recurse -Force }
}
