#Requires -Version 5.1
<# Private byte/file/config fixtures only. Never loads a DLL, probes hardware,
   downloads dependencies, starts an API/UI/model, or changes ordinary data. #>
param([string]$PythonExecutable = 'python.exe')
$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
. (Join-Path $repository 'cmake\install-intel-runtime.ps1')
. (Join-Path $repository 'cmake\intel-runtime-package.ps1')
$checks = 0
function Assert-Intel([bool]$Value, [string]$Message) {
    if (-not $Value) { throw "Intel installer fixture: $Message" }
    $script:checks++
}
function Assert-IntelRejected([scriptblock]$Action, [string]$Message) {
    $rejected = $false
    try { & $Action | Out-Null } catch { $rejected = $true }
    Assert-Intel $rejected $Message
}
function Write-IntelFixture([string]$Root, [string]$Relative, [string]$Text) {
    $path = Join-Path $Root $Relative
    [IO.Directory]::CreateDirectory((Split-Path -Parent $path)) | Out-Null
    [IO.File]::WriteAllText($path, $Text, (New-Object Text.UTF8Encoding($false)))
}
function Write-IntelManifest($Stage, $Manifest) {
    Write-IntelFixture $Stage '.tinystrata-intel-runtime.json' ($Manifest | ConvertTo-Json -Depth 12)
}
function New-IntelFixtureStage($Directory) {
    $entries = @(
        @('tinystrata-intel.exe','tinystrata'), @('sycl9.dll','compiler'), @('mkl_sycl_blas.6.dll','mkl'),
        @('licenses/TinyStrata/LICENSE','tinystrata'), @('licenses/TinyStrata/NOTICE','tinystrata'),
        @('licenses/TinyStrata/third_party/ggml/LICENSE','tinystrata'),
        @('licenses/TinyStrata/third_party/licenses/gufo.txt','tinystrata'),
        @('licenses/compiler/credist.txt','compiler'), @('licenses/mkl/license.rtf','mkl')
    )
    $files = @()
    foreach ($entry in $entries) {
        Write-IntelFixture $Directory $entry[0] ('fixture ' + $entry[0])
        $path = Join-Path $Directory $entry[0]
        $files += [pscustomobject]@{ RelativePath=$entry[0]; Component=$entry[1]; RedistributionEvidence='fixture notice';
            Source='Z:\not-trusted-sdk\never-read.dll'; Bytes=(Get-Item -LiteralPath $path).Length;
            Sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }
    }
    $manifest = [pscustomobject]@{ Schema=1; Backend='sycl'; State='staged'; Experimental=$true; ModelFamilies=@();
        ModelsIncluded=$false; DriverIncluded=$false; GlobalPathChanged=$false; MklEnabled=$true;
        Destination='Z:\not-trusted-stage'; DynamicRuntimeSeeds=@('sycl9.dll'); Files=$files }
    Write-IntelManifest $Directory $manifest
    return $manifest
}
$fixtureRoot = [IO.Path]::GetFullPath((Join-Path ([IO.Path]::GetTempPath()) ('tinystrata-intel-install-' + [Guid]::NewGuid().ToString('N'))))
$originalToken = [Environment]::GetEnvironmentVariable('TINYSTRATA_INSTALLER_TRANSACTION','Process')
try {
    $stage = Join-Path $fixtureRoot 'stage [literal]'
    $manifest = New-IntelFixtureStage $stage
    $destination = Join-Path $fixtureRoot 'native\sycl-fixture'
    $result = Install-TinyStrataIntelRuntime -StageDirectory $stage -Destination $destination
    Assert-Intel ($result.FileCount -eq 9 -and (Test-Path -LiteralPath $result.Exe -PathType Leaf)) 'verified stage imports without following SDK Source provenance'
    $installed = Get-Content -LiteralPath $result.Manifest -Raw -Encoding UTF8 | ConvertFrom-Json
    Assert-Intel ($installed.ImportedFromManifestSha256 -match '^[a-f0-9]{64}$' -and $installed.Destination -eq $destination) 'import publishes provenance after all copied hashes pass'
    Assert-Intel (@($installed.ModelFamilies).Count -eq 0 -and $installed.Experimental) 'import preserves the formal adoption gate'
    Assert-IntelRejected { Install-TinyStrataIntelRuntime $stage $destination } 'published generation cannot be overwritten'
    Assert-IntelRejected { Install-TinyStrataIntelRuntime $stage (Join-Path $stage 'child') } 'destination cannot overlap source'
    $empty = Join-Path $fixtureRoot 'empty'
    [IO.Directory]::CreateDirectory($empty) | Out-Null
    $null = Install-TinyStrataIntelRuntime $stage $empty
    Assert-Intel (Test-Path -LiteralPath (Join-Path $empty '.tinystrata-intel-runtime.json')) 'explicit empty destination is accepted'
    foreach ($bad in @('../escape.dll','/rooted.dll','C:\absolute.dll','file.dll:stream','nested/../../escape.dll','CON.dll','trailing.','a//b.dll')) {
        $manifest.Files[0].RelativePath = $bad
        Write-IntelManifest $stage $manifest
        Assert-IntelRejected { Get-TinyStrataIntelStagePlan $stage } ('reject unsafe relative path: ' + $bad)
    }
    $manifest.Files[0].RelativePath = 'tinystrata-intel.exe'
    $originalFiles = @($manifest.Files)
    $duplicate=$manifest.Files[0] | Select-Object *
    $duplicate.RelativePath='TINYSTRATA-INTEL.EXE'
    $manifest.Files += $duplicate
    Write-IntelManifest $stage $manifest
    Assert-IntelRejected { Get-TinyStrataIntelStagePlan $stage } 'case-insensitive duplicate inventory rejected'
    $manifest.Files = $originalFiles
    foreach($badPayload in @('intelocl64.dll','ze_loader.dll','igc64.dll','ocloc.dll','nested/allowed.dll')){
        $manifest.Files[0].RelativePath=$badPayload
        Write-IntelManifest $stage $manifest
        Assert-IntelRejected { Get-TinyStrataIntelStagePlan $stage } ('reject listed driver/compiler or nested payload: '+$badPayload)
    }
    $manifest.Files[0].RelativePath='tinystrata-intel.exe'
    $hash = $manifest.Files[0].Sha256
    $manifest.Files[0].Sha256 = '0' * 64
    Write-IntelManifest $stage $manifest
    Assert-IntelRejected { Install-TinyStrataIntelRuntime $stage (Join-Path $fixtureRoot 'bad-hash') } 'hash mismatch rejected before destination creation'
    Assert-Intel (-not (Test-Path -LiteralPath (Join-Path $fixtureRoot 'bad-hash'))) 'invalid stage leaves destination absent'
    $manifest.Files[0].Sha256 = $hash
    $manifest.Files = @($originalFiles | Where-Object { $_.RelativePath -ne 'licenses/mkl/license.rtf' })
    Write-IntelManifest $stage $manifest
    Assert-IntelRejected { Get-TinyStrataIntelStagePlan $stage } 'payload without component license rejected'
    $manifest.Files = $originalFiles
    foreach ($flag in @('ModelsIncluded','DriverIncluded','GlobalPathChanged')) {
        $manifest.$flag = $true
        Write-IntelManifest $stage $manifest
        Assert-IntelRejected { Get-TinyStrataIntelStagePlan $stage } ('reject invalid stage scope: ' + $flag)
        $manifest.$flag = $false
    }
    $manifest.DriverIncluded='false'
    Write-IntelManifest $stage $manifest
    Assert-IntelRejected { Get-TinyStrataIntelStagePlan $stage } 'string scope flags cannot masquerade as JSON false'
    $manifest.DriverIncluded=$false
    $manifest.ModelFamilies=@('qwen35moe')
    Write-IntelManifest $stage $manifest
    Assert-IntelRejected { Get-TinyStrataIntelStagePlan $stage } 'manifest cannot promote an unadopted family'
    $manifest.ModelFamilies=@()
    Write-IntelManifest $stage $manifest
    Write-IntelFixture $stage 'unexpected.gguf' 'not allowed'
    Assert-IntelRejected { Get-TinyStrataIntelStagePlan $stage } 'unlisted file is never silently imported'
    [IO.File]::Delete((Join-Path $stage 'unexpected.gguf'))
    $failedImport = Join-Path $fixtureRoot 'copy-failure'
    $script:hashFailureRoot = $failedImport
    $copyRejected = & {
        param($Stage,$Target)
        function Get-FileHash {
            param($LiteralPath,$Algorithm)
            if ($LiteralPath.StartsWith($script:hashFailureRoot + '\',[StringComparison]::OrdinalIgnoreCase)) {
                Write-IntelFixture $script:hashFailureRoot 'keep-racing.txt' 'caller owned file'
                return [pscustomobject]@{ Hash=('0' * 64) }
            }
            return Microsoft.PowerShell.Utility\Get-FileHash -LiteralPath $LiteralPath -Algorithm $Algorithm
        }
        try { Install-TinyStrataIntelRuntime $Stage $Target | Out-Null; return $false }
        catch { return $true }
    } $stage $failedImport
    Assert-Intel ($copyRejected -and (Test-Path -LiteralPath (Join-Path $failedImport 'keep-racing.txt')) -and
        -not (Test-Path -LiteralPath (Join-Path $failedImport 'tinystrata-intel.exe')) -and
        -not (Test-Path -LiteralPath (Join-Path $failedImport '.tinystrata-intel-runtime.json'))) 'copy/hash failure deletes only its own files and publishes no manifest'
    # A CPU-only directory junction demonstrates that hash-valid aliases are
    # refused; it is removed as a link, never recursively through its target.
    $linkedStage = Join-Path $fixtureRoot 'linked-stage'
    $linkTarget = Join-Path $fixtureRoot 'link-target'
    [IO.Directory]::CreateDirectory($linkTarget) | Out-Null
    $null = New-Item -ItemType Junction -Path $linkedStage -Target $linkTarget
    try { Assert-IntelRejected { Get-TinyStrataIntelStagePlan $linkedStage } 'linked stage root rejected' }
    finally { [IO.Directory]::Delete($linkedStage) }

    $tokens=$null; $parseErrors=$null
    $installerAst = [Management.Automation.Language.Parser]::ParseFile((Join-Path $repository 'install.ps1'), [ref]$tokens, [ref]$parseErrors)
    Assert-Intel ($parseErrors.Count -eq 0) 'installer parses under PowerShell 5.1'
    foreach ($name in @('Invoke-TinyStrataInstaller','Get-TinyStrataBackend','Get-TinyStrataConfig','Get-TinyStrataModelFamily',
        'Enter-TinyStrataInstallationGuard','Exit-TinyStrataInstallationGuard','Remove-TinyStrataNativeGeneration','Initialize-TinyStrataDesktopShortcut','Assert-TinyStrataIntelCapabilities','Assert-TinyStrataIntelManagedDeviceSelection','Select-TinyStrataBackend','Test-TinyStrataInteractiveConsole','Get-VerifiedDownload')) {
        $definition = $installerAst.Find({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name }, $true)
        Invoke-Expression $definition.Extent.Text
    }
    Assert-Intel ((Get-TinyStrataModelFamily qwen36 -Backend sycl).ConfigFile -eq 'config-sycl.json') 'explicit Intel has a dedicated config'
    Assert-IntelRejected { Get-TinyStrataModelFamily gemma4 -Backend sycl } 'unsupported family rejected before preparation'
    Assert-IntelRejected {Assert-TinyStrataIntelManagedDeviceSelection @{gpu=1;args=@('--model','fixture.gguf')}} 'custom nonzero GPU cannot be certified with the device0 probe'
    Assert-IntelRejected {Assert-TinyStrataIntelManagedDeviceSelection @{gpu=0;args=@('--device','1')}} 'native explicit nonzero device cannot be certified with the device0 probe'
    $device0=@{gpu=0;args=@('--device','0','--model','fixture.gguf')}
    $device0Before=$device0|ConvertTo-Json -Compress
    Assert-TinyStrataIntelManagedDeviceSelection $device0
    Assert-Intel (($device0|ConvertTo-Json -Compress) -ceq $device0Before) 'validated device0 configuration is accepted without mutation'
    foreach($case in @(
        @{Ids=@('PCI\VEN_8086&DEV_B082');Saved=$null;Smi=$false;Want='sycl'},
        @{Ids=@('PCI\VEN_8086&DEV_B082');Saved='cuda';Smi=$false;Want='sycl'},
        @{Ids=@('PCI\VEN_8086&DEV_B082','PCI\VEN_1002&DEV_1234');Saved=$null;Smi=$false;Want='hip'},
        @{Ids=@('PCI\VEN_8086&DEV_B082','PCI\VEN_10DE&DEV_1234');Saved=$null;Smi=$true;Want='cuda'},
        @{Ids=@('PCI\VEN_8086&DEV_B082','PCI\VEN_10DE&DEV_1234','PCI\VEN_1002&DEV_1234');Saved='hip';Smi=$true;Want='hip'})) {
        $selected=& {
            param($Case)
            function Get-CimInstance {param($ClassName,$ErrorAction) return @($Case.Ids | ForEach-Object{[pscustomobject]@{PNPDeviceID=$_}})}
            function Get-Command {param($Name,$ErrorAction) if($Case.Smi){return [pscustomobject]@{Source='Invoke-FakeNvidia'}};return $null}
            function Invoke-FakeNvidia {$global:LASTEXITCODE=0;return 'GPU 0: fixture CUDA adapter'}
            $cfg=if($Case.Saved){@{backend=$Case.Saved}}else{$null}
            Get-TinyStrataBackend -Configuration $cfg
        } $case
        Assert-Intel ($selected -eq $case.Want) ('candidate priority preserves CUDA/HIP and enables Intel-only: '+$case.Want)
    }
    $menu=& {
        function Test-TinyStrataInteractiveConsole {return $true}
        function Read-Host {return '3'}
        Select-TinyStrataBackend -Interactive $true
    }
    Assert-Intel ($menu -eq 'sycl') 'explicit optional menu can choose Intel without probing/starting it'
    $unattended=& {
        function Read-Host {throw 'Default installation must never prompt'}
        Select-TinyStrataBackend -Requested sycl
    }
    Assert-Intel ($unattended -eq 'sycl') 'noninteractive explicit backend does not prompt'
    Assert-IntelRejected { & {function Test-TinyStrataInteractiveConsole {return $false}; Select-TinyStrataBackend -Interactive $true} } 'menu request in noninteractive host fails before waiting for input'

    $pendingCatalog=Get-Content -LiteralPath (Join-Path $repository 'cmake\intel-runtime-catalog.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $pendingCatalog.availability='pending_release'; $pendingCatalog.artifacts=@()
    $pendingPath=Join-Path $fixtureRoot 'pending-catalog.json'
    Write-IntelFixture $fixtureRoot 'pending-catalog.json' ($pendingCatalog | ConvertTo-Json -Depth 8)
    Assert-IntelRejected {Get-TinyStrataIntelRuntimeArtifact -CatalogPath $pendingPath} 'pending release catalog cannot enable a placeholder download'
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $script:fixtureRuntimeZip=Join-Path $fixtureRoot 'fixture-runtime.zip'
    [IO.Compression.ZipFile]::CreateFromDirectory($stage,$script:fixtureRuntimeZip)
    $publicCatalog=Get-Content -LiteralPath (Join-Path $repository 'cmake\intel-runtime-catalog.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $publicCatalog.availability='published'
    $artifact=[pscustomobject]@{id='fixture-b390';profile='b390-qwen36';model='qwen3.6-35b-a3b';aot_target='intel_gpu_ptl_h';experimental=$true;
        url='https://github.com/kbc20f11/TinyStrata/releases/download/fixture/runtime.zip';bytes=(Get-Item -LiteralPath $script:fixtureRuntimeZip).Length;
        sha256=(Get-FileHash -LiteralPath $script:fixtureRuntimeZip -Algorithm SHA256).Hash.ToLowerInvariant();
        manifest_sha256=(Get-FileHash -LiteralPath (Join-Path $stage '.tinystrata-intel-runtime.json') -Algorithm SHA256).Hash.ToLowerInvariant();
        executable_sha256=(Get-FileHash -LiteralPath (Join-Path $stage 'tinystrata-intel.exe') -Algorithm SHA256).Hash.ToLowerInvariant()}
    $publicCatalog.artifacts=@($artifact)
    $catalogFile=Join-Path $fixtureRoot 'published-catalog.json'
    Write-IntelFixture $fixtureRoot 'published-catalog.json' ($publicCatalog | ConvertTo-Json -Depth 8)
    $script:fixtureDownloadCount=0
    function Get-TinyStrataOfficialFile {
        param($Uri,$Destination)
        Assert-Intel ($Uri -eq 'https://github.com/kbc20f11/TinyStrata/releases/download/fixture/runtime.zip') 'download URL comes only from the pinned catalog'
        $script:fixtureDownloadCount++
        [IO.File]::Copy($script:fixtureRuntimeZip,$Destination,$false)
    }
    $receipt=Receive-TinyStrataIntelRuntime -CatalogPath $catalogFile -CacheRoot (Join-Path $fixtureRoot 'download-cache')
    Assert-Intel ($receipt.Artifact.sha256 -eq $artifact.sha256 -and (Get-TinyStrataIntelStagePlan $receipt.StageDirectory).Files.Count -eq 9) 'pinned receipt executes real ZIP/hash/license checks with only HTTP boundary mocked'
    $secondReceipt=Receive-TinyStrataIntelRuntime -CatalogPath $catalogFile -CacheRoot (Join-Path $fixtureRoot 'download-cache')
    Assert-Intel ($script:fixtureDownloadCount -eq 1 -and $receipt.StageDirectory -ne $secondReceipt.StageDirectory) 'cached verified archive is reused and stages remain separate immutable preparations'
    $goodSha=$artifact.sha256;$artifact.sha256='0'*64
    Write-IntelFixture $fixtureRoot 'published-catalog.json' ($publicCatalog | ConvertTo-Json -Depth 8)
    Assert-IntelRejected {Receive-TinyStrataIntelRuntime -CatalogPath $catalogFile -CacheRoot (Join-Path $fixtureRoot 'bad-download-cache')} 'download hash mismatch is rejected before extraction/adoption'
    $artifact.sha256=$goodSha
    foreach($url in @('http://github.com/kbc20f11/TinyStrata/releases/download/tag/a.zip','https://example.com/a.zip','https://github.com/kbc20f11/TinyStrata/releases/latest/download/a.zip')){
        $artifact.url=$url
        Write-IntelFixture $fixtureRoot 'published-catalog.json' ($publicCatalog | ConvertTo-Json -Depth 8)
        Assert-IntelRejected {Get-TinyStrataIntelRuntimeArtifact -CatalogPath $catalogFile} 'unpinned/nonofficial runtime URL is rejected'
    }
    $artifact.url='https://github.com/kbc20f11/TinyStrata/releases/download/fixture/runtime.zip'
    Write-IntelFixture $fixtureRoot 'published-catalog.json' ($publicCatalog | ConvertTo-Json -Depth 8)
    $unsafeIndex=0
    foreach($names in @(@('../escape.dll'),@('C:/escape.dll'),@('licenses/../escape.dll'),@('file.dll:stream'),@('CON.dll'),@('a.dll','A.dll'))){
        $unsafeIndex++
        $path=Join-Path $fixtureRoot ('unsafe-'+$unsafeIndex+'.zip')
        $archive=[IO.Compression.ZipFile]::Open($path,[IO.Compression.ZipArchiveMode]::Create)
        try{foreach($name in $names){$entry=$archive.CreateEntry($name);$stream=$entry.Open();try{$stream.WriteByte(1)}finally{$stream.Dispose()}}}finally{$archive.Dispose()}
        $unsafeDestination=Join-Path $fixtureRoot ('unsafe-output-'+$unsafeIndex)
        Assert-IntelRejected {Expand-TinyStrataIntelRuntimeArchive -Archive $path -Destination $unsafeDestination} 'unsafe archive path/case collision is rejected before any extraction'
        Assert-Intel (-not(Test-Path -LiteralPath $unsafeDestination)) 'rejected archive leaves no destination or escaped files'
    }
    $shortcutWork=Join-Path $fixtureRoot 'shortcut-work'
    $shortcutSource=Join-Path $fixtureRoot 'shortcut-source'
    $desktop=Join-Path $fixtureRoot 'private-desktop'
    [IO.Directory]::CreateDirectory($shortcutSource) | Out-Null
    Write-IntelFixture $shortcutWork 'runtime\ui\history.json' 'ordinary history'
    Initialize-TinyStrataDesktopShortcut -SourceRoot $shortcutSource -WorkRoot $shortcutWork -DesktopDirectory $desktop
    $normalLinkHash=(Get-FileHash -LiteralPath (Join-Path $desktop 'TinyStrata.lnk') -Algorithm SHA256).Hash
    Initialize-TinyStrataDesktopShortcut -SourceRoot $shortcutSource -WorkRoot $shortcutWork -DesktopDirectory $desktop -Backend sycl -IntelProfile b390-qwen36
    $intelHelper=Join-Path $shortcutWork 'runtime\ui-sycl\desktop-launch.ps1'
    $intelLink=Join-Path $desktop 'TinyStrata Intel (Experimental).lnk'
    Assert-Intel ((Test-Path -LiteralPath $intelLink) -and (Test-Path -LiteralPath $intelHelper)) 'Intel shortcut and launcher have distinct names and state paths'
    Assert-Intel ((Get-FileHash -LiteralPath (Join-Path $desktop 'TinyStrata.lnk') -Algorithm SHA256).Hash -eq $normalLinkHash -and
        [IO.File]::ReadAllText((Join-Path $shortcutWork 'runtime\ui\history.json')) -eq 'ordinary history') 'Intel shortcut installation preserves the ordinary shortcut and history'
    $helperText=[IO.File]::ReadAllText($intelHelper)
    Assert-Intel ($helperText.Contains("`$tinyStrataBackend = 'sycl'") -and $helperText.Contains("'--backend', 'sycl'") -and
        -not $helperText.Contains("'--intel-profile'")) 'persistent Intel helper selects its backend and follows the saved/current Intel model profile'
    [void][Management.Automation.Language.Parser]::ParseFile($intelHelper,[ref]$tokens,[ref]$parseErrors)
    Assert-Intel ($parseErrors.Count -eq 0) 'generated Intel launcher parses without running it'

    # Exercise real entrypoint setting selection while replacing only the
    # installer function: explicit Intel reads ui-sycl, including custom ports.
    $installerText=[IO.File]::ReadAllText((Join-Path $repository 'install.ps1'))
    $header=$installerText.Substring(0,$installerAst.ParamBlock.Extent.EndOffset)
    $tail=$installerText.Substring($installerText.IndexOf('$tinyStrataOldErrorAction ='))
    $familyDefinition=$installerAst.Find({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Get-TinyStrataModelFamily' },$true).Extent.Text
    $capture=@'
function Invoke-TinyStrataInstaller {
    param($SourceDirectory,$WorkingDirectory,$ApiPort,$DoNotStart,$DoNotDownload,$InspectOnly,$BuildAgain,$RefreshSource,$Family,$Backend,$HipDirectory,$HipArchitectures,$IncludeUI,$UiPort,$Revision,$FamilyExplicit,$UiPortExplicit,$IntelRuntimeDirectory,$IntelProfile)
    return [pscustomobject]@{ Family=$Family; Backend=$Backend; UiPort=$UiPort; IntelRuntimeDirectory=$IntelRuntimeDirectory; IntelProfile=$IntelProfile }
}
'@
    $entry=[scriptblock]::Create($header + "`n" + $familyDefinition + "`n" + $capture + "`n" + $tail)
    $freshEntry=& $entry -SourceDir $shortcutSource -WorkDir $shortcutWork -Backend sycl -WithUI -IntelRuntimeDirectory $stage
    Assert-Intel ($freshEntry.UiPort -eq 3005 -and $freshEntry.Family -eq 'qwen36' -and $freshEntry.IntelRuntimeDirectory -eq $stage) 'Intel entrypoint forwards stage/profile and isolated fresh UI default'
    Write-IntelFixture $shortcutWork 'runtime\ui-sycl\settings.json' '{"model_family":"qwen36","port":3017}'
    $repeatEntry=& $entry -SourceDir $shortcutSource -WorkDir $shortcutWork -Backend sycl -WithUI
    Assert-Intel ($repeatEntry.UiPort -eq 3017 -and $repeatEntry.IntelProfile -eq 'b390-qwen36') 'Intel entrypoint preserves its custom saved UI port'
    function Test-TinyStrataSource { param($Directory) return $true }
    function Get-TinyStrataSavedProcess { param($StatePath,$ConfigPath,$SourcePath,$ApiPort) return $null }
    function Get-NetTCPConnection { param($LocalPort,$State,$ErrorAction) return @() }
    function Assert-TinyStrataInstallationIdle { param($SourceRoot,$WorkRoot) }
    function Initialize-TinyStrataPython { param($WorkRoot,$SourceRoot,$CacheRoot,$Transaction) return 'Invoke-IntelFixturePython' }
    function Complete-TinyStrataPythonTransaction { param($Transaction,[switch]$Rollback) }
    function Restore-TinyStrataSource { param($Change) }
    function Initialize-TinyStrataRuntime { throw 'Intel must not enter CUDA/HIP runtime discovery' }
    function Get-TinyStrataSourceFingerprint { throw 'Intel stage must not masquerade as a CUDA/HIP source build' }
    function Start-Process { throw 'CPU fixtures must never start a process' }
    # Run the actual shipping probe/validation script and Python capability
    # helper. Only the native --capabilities subprocess is replaced; hashing,
    # expected-SHA binding and profile/build/legacy policy execute for real.
    $adapter=Join-Path $fixtureRoot 'capability-adapter.py'
    $adapterCode=@'
import json, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
repository, report_path, code_path = sys.argv[1:4]
sys.argv = [sys.argv[0], *sys.argv[4:]]
sys.path.insert(0, repository)
from serve import intel_capabilities as caps
fixture = json.loads(Path(report_path).read_text(encoding="utf-8-sig"))
def fake_native(command, **options):
    if command == [sys.argv[2], "--capabilities"]: report = fixture["capabilities"]
    elif command == [sys.argv[2], "--device-info", "--device", "0"]: report = fixture["device"]
    else: raise AssertionError(command)
    assert options["stdin"] == subprocess.DEVNULL
    assert options["timeout"] <= 60
    assert options["env"]["PATH"].split(";")[0] == str(Path(sys.argv[2]).parent)
    assert not any("oneapi" in part.lower() for part in options["env"]["PATH"].split(";"))
    return SimpleNamespace(returncode=0, stdout=json.dumps(report), stderr="")
caps.subprocess.run = fake_native
exec(compile(Path(code_path).read_text(encoding="utf-8-sig"), "shipping-installer-capability-gate", "exec"), {"__name__": "__main__"})
'@
    [IO.File]::WriteAllText($adapter,$adapterCode,(New-Object Text.UTF8Encoding($false)))
    function Invoke-IntelFixturePython {
        if ($args.Count -ge 8 -and $args[0] -eq '-I' -and $args[1] -eq '-B' -and $args[2] -eq '-') {
            $script:probeCallCount++
            $probeCode=($input | Out-String)
            $codePath=Join-Path $script:transactionWork 'capability-script.py'
            Write-IntelFixture $script:transactionWork 'capability-script.py' $probeCode
            $report=[ordered]@{backend='sycl';experimental=$true;model_families=@();kv_types=@('fp32','int8');images=$false;mtp=$true;
                experimental_mtp_max_depth=7;serve=$true;capabilities_schema=1;experimental_models=@('qwen3.6-35b-a3b','tiel-coder-35b-a3b');
                experimental_profiles=@{'b390-qwen36'=1;'b390-tiel-coder'=1};build=@{mkl=$true;int8_xmx=$true;aot_target='intel_gpu_ptl_h'};
                weight_types=@{scalar=@(0,1,8,10,11,14,18,21,22,23,30);dp4a=@(8,10,11,14,18,21,22,23);int8_xmx=@(14,18,21,22)}}
            switch ($script:capabilityMode) {
                'mkl-on-xmx-off' { $report.build.int8_xmx=$false; $report.experimental_profiles=@{}; $report.weight_types.int8_xmx=@() }
                'profile-missing' { $report.experimental_profiles=@{} }
                'legacy-new' { foreach($key in @('capabilities_schema','experimental_models','experimental_profiles','build','weight_types')){$report.Remove($key)} }
                'legacy-reuse' { foreach($key in @('capabilities_schema','experimental_models','experimental_profiles','build','weight_types')){$report.Remove($key)} }
                'exe-changed' { [IO.File]::AppendAllText([string]$args[4],'changed after manifest copy') }
                'tiel-no-type18' {foreach($path in @('scalar','dp4a','int8_xmx')){$report.weight_types[$path]=@($report.weight_types[$path]|Where-Object{$_ -ne 18})}}
                'legacy-tiel' {foreach($key in @('capabilities_schema','experimental_models','experimental_profiles','build','weight_types')){$report.Remove($key)}}
                'package-target-mismatch' {$report.build.aot_target=''}
            }
            $device=[ordered]@{schema=1;backend='sycl';device_index=0;vendor_id=32902;device_id=45186;integrated=$true;architecture='panther_lake_xe3';adapter_luid='00000000:00001234';adapter_node_mask=1}
            switch($script:capabilityMode){
                'discrete' {$device.integrated=$false}
                'unknown-device' {$device.device_id=12345}
                'ptl-u' {$device.device_id=45187}
            }
            $reportPath=Join-Path $script:transactionWork 'capability-report.json'
            Write-IntelFixture $script:transactionWork 'capability-report.json' (@{capabilities=$report;device=$device} | ConvertTo-Json -Depth 8)
            $probeArguments=@($args[3..($args.Count-1)])
            & $PythonExecutable -I -B $adapter $repository $reportPath $codePath @probeArguments
            $global:LASTEXITCODE=$LASTEXITCODE
            return
        }
        $script:configureCallCount++
        $script:configureArguments = @($args)
        $path = [string]$args[[Array]::IndexOf($args,'--config')+1]
        $exe = [string]$args[[Array]::IndexOf($args,'--exe')+1]
        $guard = Get-Content -LiteralPath (Join-Path $script:transactionWork '.tinystrata-installing.json') -Raw | ConvertFrom-Json
        Assert-Intel ($guard.token -eq [Environment]::GetEnvironmentVariable('TINYSTRATA_INSTALLER_TRANSACTION','Process')) 'candidate configure receives the owner guard token'
        Assert-Intel ($args -contains '--intel-profile' -and $args -notcontains '--vision-exe') 'Intel configure selects the profile and omits the vision engine'
        if ($script:failConfigure) { $global:LASTEXITCODE=1; return }
        $cfg = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
        $cfg.exe=$exe; $cfg.lib_dirs=@((Split-Path -Parent $exe))
        Write-IntelFixture (Split-Path -Parent $path) (Split-Path -Leaf $path) ($cfg | ConvertTo-Json -Depth 8)
        $global:LASTEXITCODE=0
    }
    foreach ($scenario in @('configure-fail','success','mkl-on-xmx-off','profile-missing','legacy-new','exe-changed','discrete','unknown-device','ptl-u','tiel-success','tiel-no-type18','legacy-tiel','package-success','package-bad-sha','package-target-mismatch')) {
        $fail=$scenario -notin @('success','tiel-success','package-success')
        $fromPackage=$scenario -like 'package-*'
        $family=if($scenario -in @('tiel-success','tiel-no-type18','legacy-tiel')){'tiel'}else{'qwen36'}
        $familySpec=Get-TinyStrataModelFamily -Name $family -Backend sycl
        $script:failConfigure=$scenario -eq 'configure-fail'
        $script:capabilityMode=$scenario
        $script:configureCallCount=0;$script:probeCallCount=0
        $work = Join-Path $fixtureRoot ('transaction-' + $scenario)
        $script:transactionWork=$work
        # Source and work must be siblings, including the copied helper only.
        $source = Join-Path $fixtureRoot ('source-' + $scenario)
        [IO.Directory]::CreateDirectory((Join-Path $source 'cmake')) | Out-Null
        Copy-Item -LiteralPath (Join-Path $repository 'cmake\install-intel-runtime.ps1') -Destination (Join-Path $source 'cmake\install-intel-runtime.ps1')
        Copy-Item -LiteralPath (Join-Path $repository 'cmake\intel-runtime-package.ps1') -Destination (Join-Path $source 'cmake\intel-runtime-package.ps1')
        Copy-Item -LiteralPath (Join-Path $repository 'cmake\intel-runtime-catalog.json') -Destination (Join-Path $source 'cmake\intel-runtime-catalog.json')
        if($fromPackage){
            $fixtureCatalog=$publicCatalog|ConvertTo-Json -Depth 8|ConvertFrom-Json
            if($scenario -eq 'package-bad-sha'){$fixtureCatalog.artifacts[0].sha256='0'*64}
            Write-IntelFixture $source 'cmake\intel-runtime-catalog.json' ($fixtureCatalog|ConvertTo-Json -Depth 8)
        }
        $oldDirectory = Join-Path $work 'native\sycl-00000000000000000000000000000000'
        Write-IntelFixture $oldDirectory 'keep.txt' 'previous generation'
        $configRoot = Join-Path $work 'runtime\.tinystrata'
        $cfg = [ordered]@{backend='sycl'; intel_profile=$familySpec.IntelProfile; model_name=$familySpec.Model; arch='qwen35moe'; vision_mode='no';
            exe=(Join-Path $oldDirectory 'tinystrata-intel.exe'); lib_dirs=@($oldDirectory); api_key='fixture-secret-do-not-print'; args=@('--model','fixture.gguf')}
        Write-IntelFixture $configRoot $familySpec.ConfigFile ($cfg | ConvertTo-Json -Depth 6)
        if($family -eq 'tiel'){Write-IntelFixture $configRoot 'config-sycl.json' '{"backend":"sycl","model_name":"qwen3.6-35b-a3b","sentinel":"Qwen remains unchanged"}'}
        Write-IntelFixture $configRoot 'config.json' '{"backend":"cuda","sentinel":"unchanged"}'
        Write-IntelFixture $work 'runtime\ui\settings.json' '{"model_family":"gemma4","port":3000}'
        Write-IntelFixture $work 'runtime\ui\history.json' 'old history'
        $configPath = Join-Path $configRoot $familySpec.ConfigFile
        $qwenHash=if($family -eq 'tiel'){(Get-FileHash -LiteralPath (Join-Path $configRoot 'config-sycl.json') -Algorithm SHA256).Hash}else{$null}
        $before = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash
        $normalHash = (Get-FileHash -LiteralPath (Join-Path $configRoot 'config.json') -Algorithm SHA256).Hash
        $uiHash = (Get-FileHash -LiteralPath (Join-Path $work 'runtime\ui\settings.json') -Algorithm SHA256).Hash
        $caught=$false
        $runtimeArgs=if($fromPackage){@{BuildAgain=$true}}else{@{IntelRuntimeDirectory=$stage}}
        try { Invoke-TinyStrataInstaller -SourceDirectory $source -WorkingDirectory $work -Backend sycl -Family $family -DoNotStart $true -DoNotDownload $true @runtimeArgs }
        catch { if (-not $fail) { throw }; $caught=$true }
        Assert-Intel ($caught -eq $fail) 'configure failure rolls back, successful configure commits'
        Assert-Intel ($script:probeCallCount -eq $(if($scenario -eq 'package-bad-sha'){0}else{1})) 'only verified copied executable reaches native metadata/device inspection'
        if ($scenario -notin @('configure-fail','success','tiel-success','package-success')) {
            Assert-Intel ($script:configureCallCount -eq 0) ('incompatible capability rejected before config preparation: '+$scenario)
        }
        $nativeDirs = @(Get-ChildItem -LiteralPath (Join-Path $work 'native') -Directory)
        Assert-Intel ($nativeDirs.Count -eq $(if($fail){1}else{2})) 'only unpublished generation is removed on failure'
        Assert-Intel ((Get-FileHash -LiteralPath (Join-Path $configRoot 'config.json') -Algorithm SHA256).Hash -eq $normalHash) 'CUDA/HIP primary config stays byte-identical'
        if($family -eq 'tiel'){Assert-Intel ((Get-FileHash -LiteralPath (Join-Path $configRoot 'config-sycl.json') -Algorithm SHA256).Hash -eq $qwenHash) 'Intel Tiel installation preserves the Intel Qwen profile byte-for-byte'}
        Assert-Intel ((Get-FileHash -LiteralPath (Join-Path $work 'runtime\ui\settings.json') -Algorithm SHA256).Hash -eq $uiHash -and
            [IO.File]::ReadAllText((Join-Path $work 'runtime\ui\history.json')) -eq 'old history') 'ordinary UI history/settings stay unchanged'
        Assert-Intel (-not (Test-Path -LiteralPath (Join-Path $work '.tinystrata-installing.json')) -and
            @(Get-ChildItem -LiteralPath $configRoot -Filter '*.prepare-*').Count -eq 0) 'guard and candidate files are cleaned up'
        if ($fail) { Assert-Intel ((Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash -eq $before) 'failed candidate preserves old Intel config exactly' }
        else {
            $committed=Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
            Assert-Intel ($committed.api_key -ceq $cfg.api_key -and $committed.lib_dirs.Count -eq 1 -and $committed.exe -ne $cfg.exe) 'successful candidate preserves explicit credentials and selects only its new generation'
            $firstExe=$committed.exe
            $script:capabilityMode=if($family -eq 'qwen36'){'legacy-reuse'}else{'tiel-success'}
            Invoke-TinyStrataInstaller -SourceDirectory $source -WorkingDirectory $work -Backend sycl -Family $family -DoNotStart $true -DoNotDownload $true
            $reused=Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
            Assert-Intel ($reused.exe -eq $firstExe -and @(Get-ChildItem -LiteralPath (Join-Path $work 'native') -Directory).Count -eq 2) 'repeat configure verifies and reuses the published immutable generation'
            Assert-Intel ($script:probeCallCount -eq 2) 'existing generation revalidates its model-specific capability contract'
            if($family -eq 'tiel'){
                $beforeMismatch=(Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash
                $script:capabilityMode='legacy-tiel'
                Assert-IntelRejected {Invoke-TinyStrataInstaller -SourceDirectory $source -WorkingDirectory $work -Backend sycl -Family tiel -DoNotStart $true -DoNotDownload $true} 'Tiel never inherits Qwen legacy metadata permission on reuse'
                Assert-IntelRejected {Invoke-TinyStrataInstaller -SourceDirectory $source -WorkingDirectory $work -Backend sycl -Family tiel -IntelProfile b390-qwen36 -DoNotStart $true -DoNotDownload $true} 'explicit mismatched model/profile is rejected before preparation'
                Assert-Intel ((Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash -eq $beforeMismatch) 'rejected Tiel legacy/profile operations keep the saved config unchanged'
            }
        }
    }
    Assert-Intel ([Environment]::GetEnvironmentVariable('TINYSTRATA_INSTALLER_TRANSACTION','Process') -ceq $originalToken) 'installer transaction environment is restored'
    # Exercise launcher logic in an isolated interpreter with mocked process and
    # HTTP boundaries. It never listens, opens a browser or executes a model.
    $embedded=@($installerAst.FindAll({ param($node) $node -is [Management.Automation.Language.StringConstantExpressionAst] -and
        $node.Value.StartsWith('"""Embedded by the installer;') },$true))
    if ($embedded.Count -ne 1) { throw 'Shipping updater inspector is missing.' }
    $inspectorPath=Join-Path $fixtureRoot 'shipping-inspector.py'
    [IO.File]::WriteAllText($inspectorPath,$embedded[0].Value,(New-Object Text.UTF8Encoding($false)))
    $uiChecks = @'
import argparse, datetime, json, sys, tempfile, unittest
from pathlib import Path
from unittest import mock
sys.path.insert(0, sys.argv[1])
from ui import launch
n={'__name__':'intel_installer_fixture'}
exec(compile(Path(sys.argv[2]).read_text(encoding='utf-8'),'shipping-inspector','exec'),n)

class IntelLauncherFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='tinystrata-intel-ui-')
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'source'
        self.source.mkdir()
        self.work = Path(self.temp.name) / 'custom-work'
        self.normal = self.work / 'runtime' / 'ui'
        self.normal.mkdir(parents=True)
        self.normal.joinpath('settings.json').write_text('{"model_family":"gemma4","port":3000}', encoding='utf-8')
        self.normal.joinpath('history.json').write_text('kept history', encoding='utf-8')
        self.pointer = self.source.with_name('source-work') / 'runtime' / 'ui' / 'location.json'
        self.pointer.parent.mkdir(parents=True)
        self.pointer.write_text('{"existing":"normal"}', encoding='utf-8')
        self.before = {p:p.read_bytes() for p in (*self.normal.iterdir(), self.pointer)}
        self.args = argparse.Namespace(api_port=None, port=None, model_family=None, backend='sycl', intel_profile='b390-qwen36')
        self.launcher = launch.Launcher(self.args, self.source, self.work)

    def test_separate_settings_and_location(self):
        self.assertEqual((self.launcher.api_port,self.launcher.port),(18085,3005))
        self.assertEqual(self.launcher.family,'qwen36')
        self.launcher.save_settings()
        self.assertEqual(self.launcher.state_dir,self.work/'runtime/ui-sycl')
        self.assertEqual(launch.read_json(self.launcher.state_dir/'settings.json')['backend'],'sycl')
        self.assertTrue(self.source.with_name('source-work').joinpath('runtime/ui-sycl/location.json').is_file())
        for path, content in self.before.items(): self.assertEqual(path.read_bytes(),content)

    def test_explicit_api_identity_only(self):
        health = {'status':'ok','service':'strata','model':'qwen3.6-35b-a3b','backend':'sycl','intel_profile':'b390-qwen36','experimental':True,'loaded':False}
        with mock.patch.object(launch,'port_occupied',return_value=True), mock.patch.object(launch,'request_json',return_value=health), mock.patch.object(self.launcher,'spawn') as spawn:
            self.launcher.ensure_api(no_start=True)
            spawn.assert_not_called()
        for field, value in [('backend','cuda'),('backend','hip'),('intel_profile','generic'),('experimental',False),('model','gemma-4-26b-a4b-it')]:
            with self.subTest(field=field,value=value), mock.patch.object(launch,'port_occupied',return_value=True), mock.patch.object(launch,'request_json',return_value={**health,field:value}), mock.patch.object(self.launcher,'spawn') as spawn, self.assertRaises(RuntimeError):
                self.launcher.ensure_api(no_start=True)
                spawn.assert_not_called()
        self.assertFalse(self.launcher.selected_api_identity({k:v for k,v in health.items() if k!='backend'}))

    def test_preparation_and_state_use_selected_config(self):
        config=launch.profile_path(self.launcher.runtime,'qwen36','sycl')
        self.assertEqual(config.name,'config-sycl.json')
        process=mock.Mock(pid=123)
        process.wait.return_value=0
        with mock.patch.object(self.launcher,'spawn',return_value=process) as spawn:
            self.launcher.prepare_api(config)
            args=spawn.call_args.args[1]
            self.assertEqual(args[args.index('--config')+1],str(config))
            self.assertEqual(args[args.index('--backend')+1],'sycl')
            self.assertEqual(args[args.index('--intel-profile')+1],'b390-qwen36')
        with mock.patch.object(launch,'process_started',return_value='fixture'):
            self.launcher.save_api_process(process,config)
        self.assertTrue((self.launcher.runtime/'server-state-sycl.json').is_file())
        self.assertFalse((self.launcher.runtime/'server-state.json').exists())

    def test_unsupported_family_is_rejected(self):
        for family in ('gemma4','clover','tiel','lfm2','gpt-oss'):
            self.args.model_family=family
            with self.subTest(family=family), self.assertRaises(ValueError):
                launch.Launcher(self.args,self.source,self.work)

    def test_tiel_profile_config_and_process_state_are_separate(self):
        self.args.model_family='tiel'
        self.args.intel_profile='b390-tiel-coder'
        launcher=launch.Launcher(self.args,self.source,self.work)
        self.assertEqual(launcher.family,'tiel')
        self.assertEqual(launcher.intel_profile,'b390-tiel-coder')
        config=launch.profile_path(launcher.runtime,'tiel','sycl')
        self.assertEqual(config.name,'config-tiel-coder-sycl.json')
        self.assertEqual(launch.profile_path(launcher.runtime,'tiel').name,'config-tiel-coder.json')
        with mock.patch.object(launch,'process_started',return_value='fixture'):
            launcher.save_api_process(mock.Mock(pid=123),config)
        self.assertTrue((launcher.runtime/'server-state-tiel-coder-sycl.json').is_file())
        self.assertFalse((launcher.runtime/'server-state-sycl.json').exists())

    def test_omitted_profile_resolves_to_explicit_tiel_family(self):
        self.args.model_family='tiel-coder'
        self.args.intel_profile=None
        launcher=launch.Launcher(self.args,self.source,self.work)
        self.assertEqual(launcher.intel_profile,'b390-tiel-coder')

    def test_default_intel_shortcut_follows_the_validated_active_model_pair(self):
        self.args.intel_profile=None
        launcher=launch.Launcher(self.args,self.source,self.work)
        health={'status':'ok','service':'strata','model':'tiel-coder-35b-a3b','backend':'sycl','intel_profile':'b390-tiel-coder','experimental':True,'loaded':True}
        with mock.patch.object(launch,'port_occupied',return_value=True),mock.patch.object(launch,'request_json',return_value=health),mock.patch.object(launcher,'spawn') as spawn:
            launcher.ensure_api(no_start=True)
            spawn.assert_not_called()
        self.assertEqual((launcher.family,launcher.intel_profile),('tiel','b390-tiel-coder'))
        launcher.save_settings()
        self.assertEqual(launch.read_json(launcher.state_dir/'settings.json')['intel_profile'],'b390-tiel-coder')
        self.assertFalse(launcher.selected_api_identity({**health,'intel_profile':'b390-qwen36'}))

    def test_default_launcher_still_uses_normal_selection(self):
        normal=launch.Launcher(argparse.Namespace(api_port=None,port=None,model_family=None),self.source,self.work)
        self.assertEqual(normal.state_dir,self.normal)
        self.assertEqual(normal.family,'gemma4')
        self.assertEqual((normal.api_port,normal.port),(18080,3000))
        self.assertTrue(normal.selected_api_identity({'status':'ok','service':'strata','model':'gemma-4-26b-a4b-it'}))

    def test_cuda_hip_update_refuses_intel_api_before_profile_mapping_or_unload(self):
        runtime=self.work/'runtime'
        config=runtime/'.tinystrata/config-sycl.json'
        config.parent.mkdir()
        config.write_text(json.dumps({'backend':'sycl'}),encoding='utf-8')
        command=[sys.executable,'-B','-m','serve.server','--engine','tinystrata','--config',str(config),'--port','18085']
        state={'pid':555,'started':datetime.datetime.fromtimestamp(100,datetime.timezone.utc).isoformat(),'config':str(config),'source':str(self.source),'port':18085}
        (runtime/'server-state-sycl.json').write_text(json.dumps(state),encoding='utf-8')
        health={'status':'ok','service':'strata','backend':'sycl','model':'qwen3.6-35b-a3b'}
        request=mock.Mock(return_value=health)
        hold=mock.Mock(side_effect=AssertionError('no termination handle'))
        with mock.patch.dict(n,process=lambda pid:object(),meta=lambda p:{'pid':555,'started':100,'command':command},verify_process=lambda *a:None,listeners=lambda port:{555},request=request,hold=hold,verify_connection_settings=mock.Mock(side_effect=AssertionError('no CUDA profile mapping'))), self.assertRaisesRegex(RuntimeError,'separate experimental Intel'):
            n['stop'](self.source,self.work)
        self.assertEqual([c.args[1] for c in request.call_args_list],['/health'])
        hold.assert_not_called()

    def test_cuda_hip_update_refuses_isolated_intel_ui_before_any_service_mutation(self):
        state={'pid':555,'started':datetime.datetime.fromtimestamp(100,datetime.timezone.utc).isoformat()}
        (self.launcher.state_dir/'ui-process.json').write_text(json.dumps(state),encoding='utf-8')
        request=mock.Mock(side_effect=AssertionError('no HTTP needed'))
        hold=mock.Mock(side_effect=AssertionError('no termination handle'))
        with mock.patch.dict(n,process=lambda pid:object(),meta=lambda p:{'pid':555,'started':100},request=request,hold=hold), self.assertRaisesRegex(RuntimeError,'separate experimental Intel'):
            n['stop'](self.source,self.work)
        request.assert_not_called()
        hold.assert_not_called()

sys.argv=[sys.argv[0]]
unittest.main()
'@
    $uiScript = Join-Path $fixtureRoot 'ui-launcher-fixture.py'
    [IO.File]::WriteAllText($uiScript,$uiChecks,(New-Object Text.UTF8Encoding($false)))
    & $PythonExecutable -B $uiScript $repository $inspectorPath
    if ($LASTEXITCODE -ne 0) { throw 'Focused Intel UI launcher fixtures failed.' }
    Write-Host "Intel installer runtime fixtures passed ($checks checks)."
} finally {
    [Environment]::SetEnvironmentVariable('TINYSTRATA_INSTALLER_TRANSACTION',$originalToken,'Process')
    $resolved = [IO.Path]::GetFullPath($fixtureRoot)
    $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if ($resolved.StartsWith($tempRoot,[StringComparison]::OrdinalIgnoreCase) -and (Split-Path -Leaf $resolved) -like 'tinystrata-intel-install-*') {
        if (Test-Path -LiteralPath $resolved) { Remove-Item -LiteralPath $resolved -Recurse -Force }
    } else { throw 'Refusing fixture cleanup outside the verified temporary root.' }
}
