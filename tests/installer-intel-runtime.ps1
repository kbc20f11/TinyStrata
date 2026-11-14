#Requires -Version 5.1
<# Private byte/file/config fixtures only. Never loads a DLL, probes hardware,
   downloads dependencies, starts an API/UI/model, or changes ordinary data. #>
param([string]$PythonExecutable = 'python.exe')
$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
. (Join-Path $repository 'cmake\install-intel-runtime.ps1')
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
        'Enter-TinyStrataInstallationGuard','Exit-TinyStrataInstallationGuard','Remove-TinyStrataNativeGeneration','Initialize-TinyStrataDesktopShortcut')) {
        $definition = $installerAst.Find({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name }, $true)
        Invoke-Expression $definition.Extent.Text
    }
    Assert-Intel ((Get-TinyStrataModelFamily qwen36 -Backend sycl).ConfigFile -eq 'config-sycl.json') 'explicit Intel has a dedicated config'
    Assert-IntelRejected { Get-TinyStrataModelFamily gemma4 -Backend sycl } 'unsupported family rejected before preparation'
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
    Assert-Intel ($helperText.Contains("`$tinyStrataBackend = 'sycl'") -and $helperText.Contains("`$tinyStrataIntelProfile = 'b390-qwen36'") -and
        $helperText.Contains("'--backend', 'sycl', '--intel-profile'")) 'persistent Intel helper passes the explicit backend and profile'
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
    function Invoke-IntelFixturePython {
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
    foreach ($fail in @($true,$false)) {
        $script:failConfigure=$fail
        $work = Join-Path $fixtureRoot ('transaction-' + $fail)
        $script:transactionWork=$work
        # Source and work must be siblings, including the copied helper only.
        $source = Join-Path $fixtureRoot ('source-' + $fail)
        [IO.Directory]::CreateDirectory((Join-Path $source 'cmake')) | Out-Null
        Copy-Item -LiteralPath (Join-Path $repository 'cmake\install-intel-runtime.ps1') -Destination (Join-Path $source 'cmake\install-intel-runtime.ps1')
        $oldDirectory = Join-Path $work 'native\sycl-00000000000000000000000000000000'
        Write-IntelFixture $oldDirectory 'keep.txt' 'previous generation'
        $configRoot = Join-Path $work 'runtime\.tinystrata'
        $cfg = [ordered]@{backend='sycl'; intel_profile='b390-qwen36'; model_name='qwen3.6-35b-a3b'; arch='qwen35moe'; vision_mode='no';
            exe=(Join-Path $oldDirectory 'tinystrata-intel.exe'); lib_dirs=@($oldDirectory); api_key='fixture-secret-do-not-print'; args=@('--model','fixture.gguf')}
        Write-IntelFixture $configRoot 'config-sycl.json' ($cfg | ConvertTo-Json -Depth 6)
        Write-IntelFixture $configRoot 'config.json' '{"backend":"cuda","sentinel":"unchanged"}'
        Write-IntelFixture $work 'runtime\ui\settings.json' '{"model_family":"gemma4","port":3000}'
        Write-IntelFixture $work 'runtime\ui\history.json' 'old history'
        $configPath = Join-Path $configRoot 'config-sycl.json'
        $before = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash
        $normalHash = (Get-FileHash -LiteralPath (Join-Path $configRoot 'config.json') -Algorithm SHA256).Hash
        $uiHash = (Get-FileHash -LiteralPath (Join-Path $work 'runtime\ui\settings.json') -Algorithm SHA256).Hash
        $caught=$false
        try { Invoke-TinyStrataInstaller -SourceDirectory $source -WorkingDirectory $work -Backend sycl -IntelRuntimeDirectory $stage -DoNotStart $true -DoNotDownload $true }
        catch { if (-not $fail) { throw }; $caught=$true }
        Assert-Intel ($caught -eq $fail) 'configure failure rolls back, successful configure commits'
        $nativeDirs = @(Get-ChildItem -LiteralPath (Join-Path $work 'native') -Directory)
        Assert-Intel ($nativeDirs.Count -eq $(if($fail){1}else{2})) 'only unpublished generation is removed on failure'
        Assert-Intel ((Get-FileHash -LiteralPath (Join-Path $configRoot 'config.json') -Algorithm SHA256).Hash -eq $normalHash) 'CUDA/HIP primary config stays byte-identical'
        Assert-Intel ((Get-FileHash -LiteralPath (Join-Path $work 'runtime\ui\settings.json') -Algorithm SHA256).Hash -eq $uiHash -and
            [IO.File]::ReadAllText((Join-Path $work 'runtime\ui\history.json')) -eq 'old history') 'ordinary UI history/settings stay unchanged'
        Assert-Intel (-not (Test-Path -LiteralPath (Join-Path $work '.tinystrata-installing.json')) -and
            @(Get-ChildItem -LiteralPath $configRoot -Filter '*.prepare-*').Count -eq 0) 'guard and candidate files are cleaned up'
        if ($fail) { Assert-Intel ((Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash -eq $before) 'failed candidate preserves old Intel config exactly' }
        else {
            $committed=Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
            Assert-Intel ($committed.api_key -ceq $cfg.api_key -and $committed.lib_dirs.Count -eq 1 -and $committed.exe -ne $cfg.exe) 'successful candidate preserves explicit credentials and selects only its new generation'
            $firstExe=$committed.exe
            Invoke-TinyStrataInstaller -SourceDirectory $source -WorkingDirectory $work -Backend sycl -DoNotStart $true -DoNotDownload $true
            $reused=Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
            Assert-Intel ($reused.exe -eq $firstExe -and @(Get-ChildItem -LiteralPath (Join-Path $work 'native') -Directory).Count -eq 2) 'repeat configure verifies and reuses the published immutable generation'
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
