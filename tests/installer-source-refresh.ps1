#Requires -Version 5.1
<# Actual private archive/source/stamp transactions; no network, compiler, GPU or installer execution. #>
$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile((Join-Path $repository 'install.ps1'), [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw 'Installer parse failed.' }
foreach ($name in @('Test-TinyStrataSource', 'Assert-TinyStrataSourceDestination', 'Get-TinyStrataSourceFingerprint',
                    'Test-TinyStrataSourceRevision', 'Write-TinyStrataNativeBuildStamp', 'Test-TinyStrataNativeBuildStamp',
                    'Initialize-TinyStrataSource', 'Restore-TinyStrataSource')) {
    $definition = $ast.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name }, $true)
    if (-not $definition) { throw "Missing source helper: $name" }
    Invoke-Expression $definition.Extent.Text
}
$checks = 0
function Assert-Source([bool]$Passed, [string]$Message) {
    if (-not $Passed) { throw "Source fixture failed: $Message" }
    $script:checks++
}
function Assert-SourceThrows([scriptblock]$Action, [string]$Pattern) {
    $caught = $false
    try { $null = & $Action } catch { $caught = $_.Exception.Message -match $Pattern }
    Assert-Source $caught "Expected failure: $Pattern"
}
function Write-SourceFixture([string]$Root, [string]$Name, [string]$Value) {
    $path = Join-Path $Root $Name
    [IO.Directory]::CreateDirectory((Split-Path -Parent $path)) | Out-Null
    [IO.File]::WriteAllText($path, $Value, (New-Object Text.UTF8Encoding($false)))
}
$fixtureRoot = Join-Path ([IO.Path]::GetTempPath()) ('tinystrata-source-refresh-' + [Guid]::NewGuid().ToString('N'))
$script:fixtureRequests = 0
$script:fixtureTransfers = 0
$script:fixtureRevision = 'a' * 40
function Invoke-RestMethod { param($Uri, $Headers, $TimeoutSec) $script:fixtureRequests++; return [pscustomobject]@{ sha = $script:fixtureRevision } }
function Get-TinyStrataOfficialFile {
    param($Uri, $Destination)
    $script:fixtureTransfers++
    [IO.Directory]::CreateDirectory((Split-Path -Parent $Destination)) | Out-Null
    [IO.File]::Copy($script:fixtureArchive, $Destination, $true)
}
try {
    $payloadParent = Join-Path $fixtureRoot 'payload'
    $payload = Join-Path $payloadParent ('TinyStrata-' + $script:fixtureRevision)
    foreach ($name in @('run.py', 'requirements.txt', 'CMakeLists.txt', 'serve\server.py',
                        'src\engine.cpp', 'include\engine.hpp', 'cmake\engine.cmake', 'third_party\fixed.h', 'ui\web\app.js')) {
        Write-SourceFixture $payload $name ('current fixture ' + $name)
    }
    Copy-Item -LiteralPath (Join-Path $repository 'install.ps1') -Destination (Join-Path $payload 'install.ps1')
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $script:fixtureArchive = Join-Path $fixtureRoot 'official-fixture.zip'
    [IO.Compression.ZipFile]::CreateFromDirectory($payloadParent, $script:fixtureArchive)
    $source = Join-Path $fixtureRoot ('source-' + [char]0x65E5 + [char]0x672C)
    foreach ($name in @('run.py', 'requirements.txt', 'CMakeLists.txt', 'serve\server.py', 'src\gptoss.cpp')) {
        Write-SourceFixture $source $name ('old fixture ' + $name)
    }
    $work = Join-Path $fixtureRoot 'work'
    foreach ($name in @('runtime\.tinystrata\config-gemma4.json', 'runtime\.tinystrata\model-files\model.gguf',
                        'runtime\ui\chat-history.sqlite3', 'runtime\ui\search-settings.json')) {
        Write-SourceFixture $work $name ('private retained fixture ' + $name)
    }
    $before = @{}
    foreach ($file in Get-ChildItem -LiteralPath $work -Recurse -File) { $before[$file.FullName] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash }
    $change = Initialize-TinyStrataSource -SourceRoot $source -CacheRoot (Join-Path $work 'downloads') -Refresh -Revision $script:fixtureRevision
    Assert-Source ($change.Changed -and $change.Backup -and $script:fixtureRequests -eq 0 -and $script:fixtureTransfers -eq 1) 'pinned installer/source uses one revision without a second main lookup'
    Assert-Source (-not (Test-Path -LiteralPath (Join-Path $source 'src\gptoss.cpp'))) 'full source replacement removes obsolete model files'
    Assert-Source (Test-Path -LiteralPath (Join-Path $change.Backup 'src\gptoss.cpp')) 'previous source is retained reversibly'
    Assert-Source (Test-TinyStrataSourceRevision -Directory $source -Revision $script:fixtureRevision) 'new source tree and revision stamp agree'
    $same = Initialize-TinyStrataSource -SourceRoot $source -CacheRoot (Join-Path $work 'downloads') -Refresh -Revision $script:fixtureRevision
    Assert-Source (-not $same.Changed -and -not $same.Backup -and $script:fixtureTransfers -eq 1) 'same intact revision does not transfer or replace sources'
    Write-SourceFixture $source 'src\obsolete-again.cpp' 'overlay leftovers'
    Assert-Source (-not (Test-TinyStrataSourceRevision -Directory $source -Revision $script:fixtureRevision)) 'extra obsolete file invalidates stamped source tree'
    $repair = Initialize-TinyStrataSource -SourceRoot $source -CacheRoot (Join-Path $work 'downloads') -Refresh -Revision $script:fixtureRevision
    Assert-Source ($repair.Changed -and -not (Test-Path -LiteralPath (Join-Path $source 'src\obsolete-again.cpp'))) 'same-command repair removes partial-overlay leftovers'
    Write-SourceFixture $source 'serve\server.py' 'damaged source'
    Assert-Source (-not (Test-TinyStrataSourceRevision -Directory $source -Revision $script:fixtureRevision)) 'changed source invalidates stamped tree'
    $null = Initialize-TinyStrataSource -SourceRoot $source -CacheRoot (Join-Path $work 'downloads') -Refresh -Revision $script:fixtureRevision
    Assert-Source (Test-TinyStrataSourceRevision -Directory $source -Revision $script:fixtureRevision) 'changed same-revision source is repaired'
    foreach ($path in $before.Keys) { Assert-Source ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -eq $before[$path]) 'work profile/model/history/key bytes stay intact' }

    $native = Join-Path $fixtureRoot 'native'
    foreach ($name in @('tinystrata.exe', 'tinystrata-lfm2.exe', 'tinystrata-gemma4.exe', 'tinystrata-vision.exe', 'strata_kq.dll')) { Write-SourceFixture $native $name ('not executable fixture ' + $name) }
    $fingerprint = Get-TinyStrataSourceFingerprint -Directory $source -NativeOnly
    $originalCulture = [Threading.Thread]::CurrentThread.CurrentCulture
    try {
        foreach ($culture in @('en-US', 'ja-JP', 'tr-TR')) {
            [Threading.Thread]::CurrentThread.CurrentCulture = New-Object Globalization.CultureInfo $culture
            Assert-Source ((Get-TinyStrataSourceFingerprint -Directory $source -NativeOnly) -eq $fingerprint) 'native input ordering is independent of host culture'
        }
    } finally { [Threading.Thread]::CurrentThread.CurrentCulture = $originalCulture }
    Assert-Source (-not (Test-TinyStrataNativeBuildStamp -Directory $native -Fingerprint $fingerprint -Backend cuda)) 'unstamped legacy binaries are not automatically trusted'
    Write-TinyStrataNativeBuildStamp -Directory $native -SourceRoot $source -Backend cuda
    Assert-Source (Test-TinyStrataNativeBuildStamp -Directory $native -Fingerprint $fingerprint -Backend cuda) 'matching source and native byte proof can reuse one generation'
    Write-SourceFixture $source 'ui\web\app.js' 'UI-only update'
    Assert-Source ((Get-TinyStrataSourceFingerprint -Directory $source -NativeOnly) -eq $fingerprint) 'UI-only source changes do not force a native rebuild'
    Write-SourceFixture $source 'src\engine.cpp' 'changed native implementation'
    $changed = Get-TinyStrataSourceFingerprint -Directory $source -NativeOnly
    Assert-Source ($changed -ne $fingerprint -and -not (Test-TinyStrataNativeBuildStamp -Directory $native -Fingerprint $changed -Backend cuda)) 'native source changes invalidate native cache'
    Assert-Source (-not (Test-TinyStrataNativeBuildStamp -Directory $native -Fingerprint $fingerprint -Backend hip)) 'another backend cannot reuse the CUDA stamp'
    Write-SourceFixture $native 'tinystrata-gemma4.exe' 'changed native binary'
    Assert-Source (-not (Test-TinyStrataNativeBuildStamp -Directory $native -Fingerprint $fingerprint -Backend cuda)) 'changed binary invalidates native byte proof'

    $goodArchive = $script:fixtureArchive
    $script:fixtureArchive = Join-Path $fixtureRoot 'unsafe-source.zip'
    $badZip = [IO.Compression.ZipFile]::Open($script:fixtureArchive, [IO.Compression.ZipArchiveMode]::Create)
    try { $null = $badZip.CreateEntry(('TinyStrata-' + ('b' * 40) + '/src/file:unexpected')) } finally { $badZip.Dispose() }
    $sourceBefore = Get-TinyStrataSourceFingerprint -Directory $source
    Assert-SourceThrows { Initialize-TinyStrataSource -SourceRoot $source -CacheRoot (Join-Path $work 'downloads') -Refresh -Revision ('b' * 40) } 'unexpected path'
    Assert-Source ((Get-TinyStrataSourceFingerprint -Directory $source) -eq $sourceBefore) 'unsafe archive never publishes over previous sources'
    $script:fixtureArchive = $goodArchive
    Write-SourceFixture $source '.git' 'gitdir: fixture checkout'
    $transfersBefore = $script:fixtureTransfers
    Assert-SourceThrows { Initialize-TinyStrataSource -SourceRoot $source -CacheRoot (Join-Path $work 'downloads') -Refresh -Revision $script:fixtureRevision } 'not a Git checkout'
    Assert-Source ($script:fixtureTransfers -eq $transfersBefore -and [IO.File]::ReadAllText((Join-Path $source 'src\engine.cpp')) -eq 'changed native implementation') 'Git edits are not downloaded over or replaced'
    Assert-SourceThrows { Initialize-TinyStrataSource -SourceRoot (Join-Path $fixtureRoot 'invalid') -CacheRoot (Join-Path $work 'downloads') -Revision 'invalid-revision' } 'invalid TinyStrata revision'
    Assert-Source ($script:fixtureTransfers -eq $transfersBefore) 'invalid pinned revision is refused before transfer'
    [pscustomobject]@{ Result = 'passed'; Checks = $checks; NetworkRequests = 0; NativeProcesses = 0; ModelTransfers = 0; PrivateArchiveCopies = $script:fixtureTransfers } | ConvertTo-Json -Compress
} finally {
    $target = [IO.Path]::GetFullPath($fixtureRoot)
    $temporary = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (-not $target.StartsWith($temporary, [StringComparison]::OrdinalIgnoreCase) -or
        -not ([IO.Path]::GetFileName($target)).StartsWith('tinystrata-source-refresh-')) { throw 'Invalid private fixture cleanup target.' }
    if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
}
