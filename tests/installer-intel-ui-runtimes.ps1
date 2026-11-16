#Requires -Version 5.1
# CPU-only: tiny real inventory/import/config transactions; network/native and
# Python configure are fixtures. Never starts an app, model, browser or SDK.
$ErrorActionPreference='Stop'
$repository=Split-Path -Parent $PSScriptRoot
. (Join-Path $repository 'cmake\install-intel-runtime.ps1')
. (Join-Path $repository 'cmake\intel-runtime-package.ps1')
. (Join-Path $repository 'cmake\install-intel-ui-runtimes.ps1')
function Import-FixtureFunctions($Path,$Names) {
    $tokens=$null;$errors=$null;$ast=[Management.Automation.Language.Parser]::ParseFile($Path,[ref]$tokens,[ref]$errors)
    if($errors.Count){throw ($errors|Out-String)}
    foreach($name in $Names){$node=$ast.Find({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name},$true);if(-not $node){throw $name};. ([scriptblock]::Create($node.Extent.Text))}
    # Dot-sourced into this function; explicitly retain only the named definitions.
    foreach($name in $Names){Set-Item -Path ('Function:script:'+ $name) -Value (Get-Item ('Function:'+ $name)).ScriptBlock}
}
Import-FixtureFunctions (Join-Path $repository 'install.ps1') @('Get-TinyStrataConfig','Get-TinyStrataModelFamily','Assert-TinyStrataIntelManagedDeviceSelection','Test-TinyStrataIntelRuntimeCurrent','Remove-TinyStrataNativeGeneration','Get-TinyStrataSourceFingerprint','Invoke-TinyStrataInstaller','Enter-TinyStrataInstallationGuard','Exit-TinyStrataInstallationGuard')
Import-FixtureFunctions (Join-Path $PSScriptRoot 'installer-intel-runtime.ps1') @('Write-IntelFixture','Write-IntelManifest','New-IntelFixtureStage')
$checks=0
function Check([bool]$Ok,[string]$Message){if(-not $Ok){throw $Message};$script:checks++}
function Reject([scriptblock]$Action){$failed=$false;try{& $Action}catch{$failed=$true};Check $failed 'Expected a fail-closed preparation/publish refusal'}
$fixture=[IO.Path]::GetFullPath((Join-Path ([IO.Path]::GetTempPath()) ('intel-ui-runtime-cpu-'+[Guid]::NewGuid().ToString('N'))))
$originalToken=[Environment]::GetEnvironmentVariable('TINYSTRATA_INSTALLER_TRANSACTION','Process')
$script:failModel='';$script:configureFail='';$script:receiveFail='';$script:downloads=@();$script:probes=@();$script:imageProbes=@();$script:imageCapFail=$false
function Receive-TinyStrataIntelRuntime {
    param($CatalogPath,$CacheRoot,$Profile,$Model,$AotTarget)
    $script:downloads+=@($Model)
    if($Model -eq $script:receiveFail){throw 'CPU download failure'}
    $artifact=Get-TinyStrataIntelRuntimeArtifact -CatalogPath $CatalogPath -Profile $Profile -Model $Model -AotTarget $AotTarget
    return [pscustomobject]@{Artifact=$artifact;StageDirectory=$script:stages[$artifact.manifest_sha256]}
}
function Assert-TinyStrataIntelCapabilities {
    param($Python,$SourceRoot,$Executable,$DllDirectories,$ExpectedSha256,$Profile,$Model,$AllowLegacy,$CatalogPath,$ExpectedAotTarget,$RequireImages)
    $script:probes+=@($Model)
    Check (-not $AllowLegacy -and (Get-FileHash -LiteralPath $Executable -Algorithm SHA256).Hash -eq $ExpectedSha256) 'Only hash-bound new metadata reaches the model probe fixture'
    if($Model -eq $script:failModel){throw 'CPU native-contract rejection'}
    if($RequireImages){$script:imageProbes+=@($Model);if($script:imageCapFail){throw 'CPU image-contract rejection'}}
}
function Invoke-IntelUiFixturePython {
    $config=[string]$args[[Array]::IndexOf($args,'--config')+1]
    $family=Get-TinyStrataModelFamily $args[[Array]::IndexOf($args,'--model-family')+1] -Backend sycl
    Check ($args -contains '--configure-only' -and $args -notcontains '--model') 'Preparation never generates or downloads a model/assistant/projector'
    $guardPath=Join-Path (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $config))) '.tinystrata-installing.json'
    $expectedGuard=if(Test-Path -LiteralPath $guardPath){(Get-Content -LiteralPath $guardPath -Raw|ConvertFrom-Json).token}else{'fixture-guard'}
    Check ([Environment]::GetEnvironmentVariable('TINYSTRATA_INSTALLER_TRANSACTION','Process') -ceq $expectedGuard) 'Preparation receives only its existing guard token'
    if($family.Model -eq $script:configureFail){$global:LASTEXITCODE=1;return}
    $cfg=if(Test-Path -LiteralPath $config){Get-TinyStrataConfig $config}else{[pscustomobject]@{args=@('--max-context','auto','--kv-type','auto','--chunk','auto');model_bootstrap=$true}}
    $exe=[string]$args[[Array]::IndexOf($args,'--exe')+1]
    $mode=[string]$args[[Array]::IndexOf($args,'--vision')+1]
    Check ($mode -in @('gpu','no') -and ($family.Arch -ne 'lfm2moe' -or $mode -eq 'no')) 'Every Intel prepare has a bounded explicit mode and LFM remains text only'
    if($mode -eq 'gpu'){
        $imageExe=[string]$args[[Array]::IndexOf($args,'--vision-exe')+1]
        Check ($imageExe -eq (Join-Path (Split-Path -Parent $exe) 'tinystrata-vision.exe') -and (Test-Path -LiteralPath $imageExe -PathType Leaf)) 'Images use only the selected verified app-local encoder'
        $cfg|Add-Member -NotePropertyName vision_exe -NotePropertyValue $imageExe -Force
    }else{Check ($args -notcontains '--vision-exe') 'Text-only Intel preparation does not attach an image encoder'}
    foreach($entry in @{backend='sycl';model_name=$family.Model;arch=$family.Arch;intel_profile=[string]$args[[Array]::IndexOf($args,'--intel-profile')+1];vision_mode=$mode;exe=$exe;lib_dirs=@((Split-Path -Parent $exe))}.GetEnumerator()){$cfg|Add-Member -NotePropertyName $entry.Key -NotePropertyValue $entry.Value -Force}
    [IO.File]::WriteAllText($config,($cfg|ConvertTo-Json -Depth 12))
    $global:LASTEXITCODE=0
}
function Prepare-Case($Name,$Catalog) {
    $work=Join-Path $fixture $Name;$root=Join-Path $work 'runtime\.tinystrata'
    $selected=Install-TinyStrataIntelRuntime -StageDirectory $script:qwenStage -Destination (Join-Path $work 'native\sycl-00000000000000000000000000000000')
    Write-IntelFixture $root 'config-sycl.json' '{"backend":"sycl","api_key":"private-fixture-retained","sentinel":"old"}'
    Write-IntelFixture $root 'config-sycl.json.bak' '{"older":"retained backup"}'
    Write-IntelFixture $root 'config-gemma4-sycl.json' '{"backend":"sycl","model_name":"gemma-4-26b-a4b-it","arch":"gemma4","intel_profile":"b390-gemma4","api_key":"private-extra-key","sampling":{"temperature":0.3},"args":["--max-context","65536"]}'
    Write-IntelFixture $work 'runtime\ui-sycl\settings.json' '{"retained":true}'
    Write-IntelFixture $work 'runtime\ui-sycl\history.sqlite3' 'private history sentinel'
    $path=Join-Path $root 'config-sycl.json';$candidate=$path+'.prepare-fixture';[IO.File]::WriteAllText($candidate,'{"prepared":"selected Qwen"}')
    $records=New-Object 'System.Collections.Generic.List[object]';$records.Add([pscustomobject]@{Path=$path;Candidate=$candidate})
    $cache=@{};$cache[($script:qwenArtifact.manifest_sha256+':'+$script:qwenArtifact.executable_sha256)]=[pscustomobject]@{Exe=$selected.Exe;DllDirectories=$selected.DllDirectories}
    $catalogPath=Join-Path $work 'catalog.json';[IO.File]::WriteAllText($catalogPath,($Catalog|ConvertTo-Json -Depth 12))
    return [pscustomobject]@{Work=$work;Path=$path;Catalog=$catalogPath;Configurations=$records;Generations=(New-Object 'System.Collections.Generic.List[string]');Cache=$cache}
}
function Prepare-Extras($Case,[bool]$Offline=$false){Prepare-TinyStrataIntelUiRuntimes -SourceRoot $repository -WorkRoot $Case.Work -Python Invoke-IntelUiFixturePython -CatalogPath $Case.Catalog -CacheRoot (Join-Path $Case.Work 'downloads') -SelectedModel 'qwen3.6-35b-a3b' -GuardToken fixture-guard -AotTarget intel_gpu_ptl_h -Offline $Offline -RuntimeCache $Case.Cache -Configurations $Case.Configurations -Generations $Case.Generations}
try {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $script:stages=@{};$script:zips=@{};$artifacts=@()
    foreach($name in @('qwen36','gemma4','lfm2')){
        $stage=Join-Path $fixture ('stage-'+$name);$manifest=New-IntelFixtureStage $stage
        Write-IntelFixture $stage 'tinystrata-intel.exe' ('CPU non-executable fixture '+$name)
        Write-IntelFixture $stage 'tinystrata-vision.exe' ('CPU non-executable image fixture '+$name)
        $image=Get-Item -LiteralPath (Join-Path $stage 'tinystrata-vision.exe')
        $manifest.Files+=@([pscustomobject]@{RelativePath='tinystrata-vision.exe';Component='tinystrata';RedistributionEvidence='fixture notice';Source='Z:\never-read';Bytes=$image.Length;Sha256=(Get-FileHash -LiteralPath $image.FullName -Algorithm SHA256).Hash.ToLowerInvariant()})
        $manifest.Files[0].Bytes=(Get-Item -LiteralPath (Join-Path $stage 'tinystrata-intel.exe')).Length
        $manifest.Files[0].Sha256=(Get-FileHash -LiteralPath (Join-Path $stage 'tinystrata-intel.exe') -Algorithm SHA256).Hash.ToLowerInvariant();Write-IntelManifest $stage $manifest
        $hash=(Get-FileHash -LiteralPath (Join-Path $stage '.tinystrata-intel-runtime.json') -Algorithm SHA256).Hash.ToLowerInvariant();$script:stages[$hash]=$stage
        $zip=Join-Path $fixture ($name+'.zip');[IO.Compression.ZipFile]::CreateFromDirectory($stage,$zip)
        $zipSha=(Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant();$script:zips[$zipSha]=$zip
        $families=if($name -eq 'qwen36'){@('qwen36','tiel','clover')}else{@($name)}
        foreach($familyName in $families){$family=Get-TinyStrataModelFamily $familyName -Backend sycl;$artifacts+=[pscustomobject]@{id=('fixture-'+$familyName);model=$family.Model;profile=$family.IntelProfile;aot_target='intel_gpu_ptl_h';experimental=$true;url=('https://github.com/kbc20f11/TinyStrata/releases/download/fixture/'+$name+'.zip');bytes=(Get-Item -LiteralPath $zip).Length;sha256=$zipSha;manifest_sha256=$hash;executable_sha256=$manifest.Files[0].Sha256}}
        if($name -eq 'qwen36'){$script:qwenStage=$stage;$script:qwenArtifact=$artifacts[0]}
    }
    $catalog=[pscustomobject]@{schema=1;backend='sycl';experimental=$true;availability='published';artifacts=$artifacts;supported_devices=@([pscustomobject]@{vendor_id=32902;device_ids=@(45186);integrated=$true;architecture='panther_lake_xe3';profile='b390-qwen36';aot_target='intel_gpu_ptl_h'})}
    foreach($scenario in @('success','offline','pending','absent','custom-runtime','download-fail','configure-fail','cap-fail')){
        if($scenario -eq 'configure-fail'){[Environment]::SetEnvironmentVariable('TINYSTRATA_INSTALLER_TRANSACTION','retained-parent-value','Process')}
        elseif($null -eq $originalToken){Remove-Item -LiteralPath Env:\TINYSTRATA_INSTALLER_TRANSACTION -ErrorAction SilentlyContinue}
        else{[Environment]::SetEnvironmentVariable('TINYSTRATA_INSTALLER_TRANSACTION',$originalToken,'Process')}
        $caseToken=[Environment]::GetEnvironmentVariable('TINYSTRATA_INSTALLER_TRANSACTION','Process')
        $current=$catalog|ConvertTo-Json -Depth 12|ConvertFrom-Json
        if($scenario -eq 'pending'){$current.availability='pending_release'}
        if($scenario -eq 'absent'){$current.artifacts=@($current.artifacts|Where-Object {$_.model -in @('qwen3.6-35b-a3b','tiel-coder-35b-a3b')})}
        $case=Prepare-Case $scenario $current;$script:downloads=@();$script:probes=@()
        if($scenario -eq 'custom-runtime'){
            $customPath=Join-Path $case.Work 'runtime\.tinystrata\config-gemma4-sycl.json'
            $custom=Get-TinyStrataConfig $customPath;$custom|Add-Member -NotePropertyName exe -NotePropertyValue (Join-Path $case.Work 'custom-engine.exe')
            [IO.File]::WriteAllText($customPath,($custom|ConvertTo-Json -Depth 12))
        }
        $script:receiveFail=if($scenario -eq 'download-fail'){'gemma-4-26b-a4b-it'}else{''}
        $script:configureFail=if($scenario -eq 'configure-fail'){'lfm2.5-24b-a2b'}else{''}
        $script:failModel=if($scenario -eq 'cap-fail'){'clover-35b-a3b-preview'}else{''}
        $protected=@{};foreach($file in @(Get-ChildItem -LiteralPath (Join-Path $case.Work 'runtime') -Recurse -File|Where-Object {$_.Name -notlike '*.prepare-*'})){$protected[$file.FullName]=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash}
        if($scenario -like '*-fail'){Reject {Prepare-Extras $case}}
        else{Prepare-Extras $case ($scenario -eq 'offline')}
        foreach($path in $protected.Keys){Check ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -eq $protected[$path]) 'Every live config/key/UI/history byte is unchanged before publication'}
        Check ([Environment]::GetEnvironmentVariable('TINYSTRATA_INSTALLER_TRANSACTION','Process') -ceq $caseToken) 'Exact inherited transaction value/absence is restored'
        if($scenario -eq 'success'){
            Check ($case.Configurations.Count -eq 5 -and $case.Generations.Count -eq 2 -and $script:downloads.Count -eq 2 -and $script:probes.Count -eq 4) 'Fresh Qwen UI provisions all five identities with only two additional dedicated runtime imports'
            Publish-TinyStrataIntelConfigurations $case.Configurations
            $gemma=Get-TinyStrataConfig (Join-Path $case.Work 'runtime\.tinystrata\config-gemma4-sycl.json')
            Check ($gemma.api_key -ceq 'private-extra-key' -and $gemma.sampling.temperature -eq 0.3 -and $gemma.args[1] -eq '65536') 'Additional existing keys, sampling and explicit context are preserved'
            Check (@(Get-ChildItem -LiteralPath $case.Work -Recurse -Filter '*.gguf').Count -eq 0) 'No model payload is provisioned'
            $repeat=New-Object 'System.Collections.Generic.List[object]';$repeatGenerations=New-Object 'System.Collections.Generic.List[string]';$script:downloads=@()
            Prepare-TinyStrataIntelUiRuntimes -SourceRoot $repository -WorkRoot $case.Work -Python Invoke-IntelUiFixturePython -CatalogPath $case.Catalog -CacheRoot (Join-Path $case.Work 'downloads') -SelectedModel 'qwen3.6-35b-a3b' -GuardToken fixture-guard -AotTarget intel_gpu_ptl_h -Offline $false -RuntimeCache @{} -Configurations $repeat -Generations $repeatGenerations
            Check ($repeat.Count -eq 4 -and $repeatGenerations.Count -eq 0 -and $script:downloads.Count -eq 0) 'Current published generations are revalidated/reused without another runtime download'
        } elseif($scenario -in @('offline','pending')) {Check ($case.Configurations.Count -eq 1 -and $script:downloads.Count -eq 0 -and $script:probes.Count -eq 0) 'Offline or unpublished catalogs do not fetch another runtime'}
        elseif($scenario -eq 'absent'){Check ($case.Configurations.Count -eq 2 -and $script:downloads.Count -eq 0) 'Absent new-model descriptors remain unavailable instead of borrowing Qwen'}
        elseif($scenario -eq 'custom-runtime'){Check ($case.Configurations.Count -eq 4 -and $script:downloads.Count -eq 1 -and $script:probes -notcontains 'gemma-4-26b-a4b-it') 'Inactive custom runtimes are neither replaced nor certified by WithUI provisioning'}
        else {foreach($generation in $case.Generations){Remove-TinyStrataNativeGeneration $case.Work $generation};Check (@(Get-ChildItem -LiteralPath (Join-Path $case.Work 'native') -Directory).Count -eq 1) 'Caller rollback removes only new unpublished runtime generations'}
    }
    # Real installer branch, real Receive/extraction/import/helper/publish/guard;
    # only network, native, Python bootstrap/configure and Desktop boundaries fake.
    function Test-TinyStrataSource {param($Directory)return $true}
    function Get-TinyStrataBackend {param($Requested,$Configuration)return 'sycl'}
    function Get-TinyStrataIntelRuntimeSelection {param($Model,$RequestedProfile)return [pscustomobject]@{Profile='b390-qwen36';AotTarget='intel_gpu_ptl_h'}}
    function Get-TinyStrataSavedProcess {param($StatePath,$ConfigPath,$SourcePath,$ApiPort)return $null}
    function Assert-TinyStrataInstallationIdle {param($SourceRoot,$WorkRoot)}
    function Initialize-TinyStrataPython {param($WorkRoot,$SourceRoot,$CacheRoot,$Transaction)return 'Invoke-IntelUiFixturePython'}
    function Complete-TinyStrataPythonTransaction {param($Transaction,[switch]$Rollback)}
    function Restore-TinyStrataSource {param($Change)}
    function Initialize-TinyStrataOptionalUI {param($SourceRoot,$WorkRoot,$ApiPort,$UiPort,$Family,[switch]$ConfigureOnly,$Backend,$IntelProfile,$Vision)$script:uiCalls++;$script:uiVision=$Vision}
    function Get-VerifiedDownload {param($Uri,$Destination,$Sha256)
        Check ($script:zips.ContainsKey($Sha256)) 'Only a pinned fixture ZIP can be requested'
        $script:networkBoundaries++
        [IO.Directory]::CreateDirectory((Split-Path -Parent $Destination))|Out-Null
        [IO.File]::Copy($script:zips[$Sha256],$Destination,$false)
    }
    foreach($scenario in @('wired-success','wired-extra-fail','wired-offline','wired-vision-gpu','wired-vision-saved','wired-vision-no','wired-vision-cap-fail')){
        $case=Prepare-Case $scenario $catalog;$script:configureFail=if($scenario -eq 'wired-extra-fail'){'lfm2.5-24b-a2b'}else{''}
        $script:failModel='';$script:receiveFail='';$script:networkBoundaries=0;$script:uiCalls=0;$script:uiVision='';$script:imageProbes=@();$script:imageCapFail=$scenario -eq 'wired-vision-cap-fail'
        if($scenario -in @('wired-vision-saved','wired-vision-no')){
            foreach($path in @($case.Path,(Join-Path $case.Work 'runtime\.tinystrata\config-gemma4-sycl.json'))){
                $saved=Get-TinyStrataConfig $path;$saved|Add-Member -NotePropertyName vision_mode -NotePropertyValue gpu -Force
                [IO.File]::WriteAllText($path,($saved|ConvertTo-Json -Depth 12))
            }
        }
        $source=Join-Path $fixture ($scenario+'-source');[IO.Directory]::CreateDirectory((Join-Path $source 'cmake'))|Out-Null
        foreach($helper in @('install-intel-runtime.ps1','intel-runtime-package.ps1','install-intel-ui-runtimes.ps1')){[IO.File]::Copy((Join-Path $repository ('cmake\'+$helper)),(Join-Path $source ('cmake\'+$helper)))}
        [IO.File]::Copy($case.Catalog,(Join-Path $source 'cmake\intel-runtime-catalog.json'))
        $old=[IO.File]::ReadAllText($case.Path)
        $arguments=@{SourceDirectory=$source;WorkingDirectory=$case.Work;Backend='sycl';Family='qwen36';FamilyExplicit=$true;IncludeUI=$true;DoNotStart=$true;DoNotDownload=$true}
        if($scenario -eq 'wired-offline'){$arguments.IntelRuntimeDirectory=$script:qwenStage}
        if($scenario -in @('wired-vision-gpu','wired-vision-cap-fail')){$arguments.Vision='gpu'}
        if($scenario -eq 'wired-vision-no'){$arguments.Vision='no'}
        if($scenario -in @('wired-extra-fail','wired-vision-cap-fail')){Reject {Invoke-TinyStrataInstaller @arguments}}
        else{Invoke-TinyStrataInstaller @arguments}
        if($scenario -in @('wired-extra-fail','wired-vision-cap-fail')){
            Check ([IO.File]::ReadAllText($case.Path) -ceq $old -and $script:uiCalls -eq 0) 'Full installer failure publishes no selected config/UI'
            Check (@(Get-ChildItem -LiteralPath (Join-Path $case.Work 'native') -Directory).Count -eq 1) 'Full installer rollback retains only its prior immutable generation'
        }elseif($scenario -like 'wired-vision-*'){
            foreach($name in @('qwen36','tiel','gemma4','lfm2','clover')){
                $family=Get-TinyStrataModelFamily $name -Backend sycl;$prepared=Get-TinyStrataConfig (Join-Path $case.Work ('runtime\.tinystrata\'+$family.ConfigFile))
                $expected=if($name -eq 'lfm2' -or $scenario -eq 'wired-vision-no'){'no'}elseif($scenario -eq 'wired-vision-gpu' -or $name -in @('qwen36','gemma4')){'gpu'}else{'no'}
                Check ($prepared.vision_mode -eq $expected) ('Explicit/saved model-specific vision selection: '+$name+' '+$scenario)
            }
            Check ($script:imageProbes.Count -eq $(if($scenario -eq 'wired-vision-gpu'){4}elseif($scenario -eq 'wired-vision-saved'){2}else{0})) 'Image capability proof is required exactly for the selected image-capable model identities'
            $gemma=Get-TinyStrataConfig (Join-Path $case.Work 'runtime\.tinystrata\config-gemma4-sycl.json')
            Check ($gemma.api_key -ceq 'private-extra-key' -and $gemma.sampling.temperature -eq 0.3 -and $gemma.args[1] -eq '65536') 'Vision adoption preserves model KV, sampling and private key'
            $expectedUi=if($scenario -eq 'wired-vision-gpu'){'gpu'}elseif($scenario -eq 'wired-vision-no'){'no'}else{$null}
            Check ($script:uiCalls -eq 1 -and $script:uiVision -eq $expectedUi) 'Only an explicit image choice reaches the UI launcher'
        }elseif($scenario -eq 'wired-success'){
            Check ($script:networkBoundaries -eq 3 -and $script:uiCalls -eq 1) 'Fresh WithUI fetches three runtime payloads and prepares UI only after all models are configured'
            Check ((Get-TinyStrataConfig (Join-Path $case.Work 'runtime\.tinystrata\config-clover-sycl.json')).model_bootstrap -eq $true) 'Fresh UI has a dedicated Clover bootstrap profile before any model download'
            Check (@(Get-ChildItem -LiteralPath (Join-Path $case.Work 'native') -Directory).Count -eq 4) 'Shared Qwen/Tiel/Clover package is imported once'
        }else{Check ($script:networkBoundaries -eq 0 -and $script:uiCalls -eq 1) 'Explicit offline stage never triggers another runtime download through WithUI'}
        Check (-not(Test-Path -LiteralPath (Join-Path $case.Work '.tinystrata-installing.json'))) 'Full installer releases its real transaction guard'
        Check (@(Get-ChildItem -LiteralPath (Join-Path $case.Work 'runtime\.tinystrata') -Filter '*.prepare-*').Count -eq 1) 'Full installer cleans its own candidates and leaves the pre-existing fixture candidate untouched'
    }
    $script:imageCapFail=$false
    # Real running-selected branch with a mixed catalog: selected Qwen stays
    # current while one dedicated peer is stale/missing. No process or network
    # boundary is allowed to hide the refusal, including ordinary WithUI reuse.
    function Get-TinyStrataSavedProcess {param($StatePath,$ConfigPath,$SourcePath,$ApiPort)return [pscustomobject]@{Id=123;HasExited=$false}}
    function Get-NetTCPConnection {param($LocalPort,$State,$ErrorAction)return @()}
    function Test-TinyStrataPortOwner {param($ApiPort,$RootProcessId)return $true}
    function Test-TinyStrataInferenceReady {param($Health,$Configuration)return $true}
    function Start-Process {throw 'Running-current CPU fixtures must never start another process'}
    function Invoke-TinyStrataApi {param($BaseUrl,$Path,$Method,$Headers)
        $script:apiCalls++
        return [pscustomobject]@{status='ok';backend='sycl';experimental=$true;intel_profile='b390-qwen36';model='qwen3.6-35b-a3b';loaded=$false}
    }
    foreach($scenario in @('all-current','ordinary-current','missing-gemma','stale-gemma','stale-lfm','ordinary-stale',
        'corrupt-gemma','custom-gemma','generic-gemma','unpublished-gemma','API-only-stale')){
        $case=Prepare-Case ('running-peer-'+$scenario) $catalog
        $script:configureFail='';$script:failModel='';$script:receiveFail=''
        Prepare-Extras $case;Publish-TinyStrataIntelConfigurations $case.Configurations
        $selected=$case.Cache[($script:qwenArtifact.manifest_sha256+':'+$script:qwenArtifact.executable_sha256)]
        $cfg=[ordered]@{backend='sycl';model_name='qwen3.6-35b-a3b';arch='qwen35moe';intel_profile='b390-qwen36';
            vision_mode='no';exe=$selected.Exe;lib_dirs=$selected.DllDirectories;api_key='retained-no-print';args=@('--model','fixture.gguf','--max-context','auto')}
        [IO.File]::WriteAllText($case.Path,($cfg|ConvertTo-Json -Depth 10))
        $gemmaPath=Join-Path $case.Work 'runtime\.tinystrata\config-gemma4-sycl.json'
        $current=Get-Content -LiteralPath $case.Catalog -Raw|ConvertFrom-Json
        if($scenario -in @('stale-gemma','ordinary-stale','API-only-stale')){
            @($current.artifacts|Where-Object {$_.model -eq 'gemma-4-26b-a4b-it'})[0].manifest_sha256='a'*64
        }elseif($scenario -eq 'stale-lfm'){
            @($current.artifacts|Where-Object {$_.model -eq 'lfm2.5-24b-a2b'})[0].manifest_sha256='b'*64
        }elseif($scenario -eq 'missing-gemma'){[IO.File]::Delete($gemmaPath)}
        elseif($scenario -eq 'corrupt-gemma'){
            $peer=Get-TinyStrataConfig $gemmaPath;[IO.File]::WriteAllText((Join-Path (Split-Path -Parent $peer.exe) 'sycl9.dll'),'corrupt peer payload')
        }elseif($scenario -eq 'custom-gemma'){
            $peer=Get-TinyStrataConfig $gemmaPath;$peer.exe=Join-Path $case.Work 'custom\tinystrata-intel.exe';$peer.lib_dirs=@((Split-Path -Parent $peer.exe))
            [IO.File]::WriteAllText($gemmaPath,($peer|ConvertTo-Json -Depth 12))
            @($current.artifacts|Where-Object {$_.model -eq 'gemma-4-26b-a4b-it'})[0].manifest_sha256='c'*64
        }elseif($scenario -eq 'generic-gemma'){
            $peer=Get-TinyStrataConfig $gemmaPath;$peer.intel_profile='generic';[IO.File]::WriteAllText($gemmaPath,($peer|ConvertTo-Json -Depth 12))
            $generic=@($current.artifacts|Where-Object {$_.model -eq 'gemma-4-26b-a4b-it'})[0]|Select-Object *
            $generic.id='fixture-generic-gemma';$generic.profile='generic';$generic.aot_target='';$current.artifacts+=@($generic)
        }elseif($scenario -eq 'unpublished-gemma'){
            $current.artifacts=@($current.artifacts|Where-Object {$_.model -ne 'gemma-4-26b-a4b-it'});[IO.File]::Delete($gemmaPath)
        }
        [IO.File]::WriteAllText($case.Catalog,($current|ConvertTo-Json -Depth 12))
        $source=Join-Path $fixture ('running-peer-source-'+$scenario);[IO.Directory]::CreateDirectory((Join-Path $source 'cmake'))|Out-Null
        foreach($helper in @('install-intel-runtime.ps1','intel-runtime-package.ps1','install-intel-ui-runtimes.ps1')){
            [IO.File]::Copy((Join-Path $repository ('cmake\'+$helper)),(Join-Path $source ('cmake\'+$helper)))
        }
        [IO.File]::Copy($case.Catalog,(Join-Path $source 'cmake\intel-runtime-catalog.json'))
        $protected=@{};foreach($file in @(Get-ChildItem -LiteralPath (Join-Path $case.Work 'runtime') -Recurse -File)){$protected[$file.FullName]=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash}
        $generationCount=@(Get-ChildItem -LiteralPath (Join-Path $case.Work 'native') -Directory).Count
        $script:networkBoundaries=0;$script:downloads=@();$script:probes=@();$script:uiCalls=0;$script:apiCalls=0
        $withUI=$scenario -ne 'API-only-stale';$ordinary=$scenario -like 'ordinary-*'
        $arguments=@{SourceDirectory=$source;WorkingDirectory=$case.Work;Backend='sycl';Family='qwen36';FamilyExplicit=$true;
            IncludeUI=$withUI;DoNotStart=(-not $ordinary);DoNotDownload=$true}
        $mustReject=$scenario -in @('missing-gemma','stale-gemma','stale-lfm','ordinary-stale','corrupt-gemma')
        $failed=$false
        try{Invoke-TinyStrataInstaller @arguments}catch{
            if(-not $mustReject){throw};$failed=$true
            Check ($_.Exception.Message -match 'published Intel UI runtime.*Stop this installation.*API and UI.*rerun') 'Stale/missing peer is refused with the explicit stop/rerun instruction'
        }
        Check ($failed -eq $mustReject) ('Running-selected mixed-catalog peer result: '+$scenario)
        Check ($script:networkBoundaries -eq 0 -and $script:downloads.Count -eq 0 -and $script:probes.Count -eq 0) 'Running peers are read-only checked, never downloaded/imported/probed hot'
        Check ($script:uiCalls -eq [int]($withUI -and -not $mustReject)) 'Peer refusal occurs before any UI preparation/settings boundary'
        Check ($script:apiCalls -eq [int]($ordinary -and -not $mustReject)) 'Ordinary current reuse reaches only the existing-API health boundary'
        Check (@(Get-ChildItem -LiteralPath (Join-Path $case.Work 'native') -Directory).Count -eq $generationCount) 'Running peer checks create no generation'
        foreach($path in $protected.Keys){Check ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -eq $protected[$path]) 'Running peer checks preserve every config/key/UI/history/candidate byte'}
        Check (-not(Test-Path -LiteralPath (Join-Path $case.Work '.tinystrata-installing.json'))) 'Running peer checks do not acquire or leave an update guard'
    }
    # Real file replacement failure after the first publish must restore both
    # original configs and their prior .bak, including paths with literal [].
    $root=Join-Path $fixture 'publish [literal]';$records=New-Object 'System.Collections.Generic.List[object]';$before=@{}
    foreach($name in @('first','second')){Write-IntelFixture $root ($name+'.json') ('old '+$name);Write-IntelFixture $root ($name+'.json.bak') ('older '+$name);Write-IntelFixture $root ($name+'.json.prepare-new') ('new '+$name);$records.Add([pscustomobject]@{Path=(Join-Path $root ($name+'.json'));Candidate=(Join-Path $root ($name+'.json.prepare-new'))})}
    foreach($file in @(Get-ChildItem -LiteralPath $root -File|Where-Object {$_.Name -notlike '*.prepare-*'})){$before[$file.FullName]=[IO.File]::ReadAllText($file.FullName)}
    $lock=[IO.File]::Open($records[1].Candidate,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read)
    try{Reject {Publish-TinyStrataIntelConfigurations $records}}finally{$lock.Dispose()}
    foreach($path in $before.Keys){Check ([IO.File]::ReadAllText($path) -ceq $before[$path]) 'A later publish failure restores every original configuration and old backup'}
    Check (@(Get-ChildItem -LiteralPath $root -Filter '*.prepare-rollback-*').Count -eq 0) 'Successful rollback removes only its temporary snapshots'
    $native=Join-Path $fixture 'fingerprint';foreach($folder in @('src','include','cmake','third_party')){[IO.Directory]::CreateDirectory((Join-Path $native $folder))|Out-Null}
    [IO.File]::Copy((Join-Path $repository 'install.ps1'),(Join-Path $native 'install.ps1'))
    Write-IntelFixture $native 'CMakeLists.txt' 'CPU source fixture';Write-IntelFixture $native 'src/unit.cpp' 'shared native input'
    Write-IntelFixture $native 'cmake/install-intel-ui-runtimes.ps1' 'Intel-only helper before'
    $before=@{};foreach($backend in @('cuda','hip')){$before[$backend]=Get-TinyStrataSourceFingerprint $native -NativeOnly -Backend $backend}
    Write-IntelFixture $native 'cmake/install-intel-ui-runtimes.ps1' 'Intel-only helper after'
    foreach($backend in @('cuda','hip')){Check ((Get-TinyStrataSourceFingerprint $native -NativeOnly -Backend $backend) -ceq $before[$backend]) 'Intel-only UI helper edits do not invalidate CUDA/HIP native sources'}
    Write-IntelFixture $native 'cmake/unknown-input.cmake' 'conservative unknown source'
    foreach($backend in @('cuda','hip')){Check ((Get-TinyStrataSourceFingerprint $native -NativeOnly -Backend $backend) -cne $before[$backend]) 'Unknown native inputs retain conservative CUDA/HIP invalidation'}
    Write-Output "Intel UI runtime preparation fixtures: $checks checks PASS; no network/native/model/API/browser."
} finally {
    if($null -eq $originalToken){Remove-Item -LiteralPath Env:\TINYSTRATA_INSTALLER_TRANSACTION -ErrorAction SilentlyContinue}else{[Environment]::SetEnvironmentVariable('TINYSTRATA_INSTALLER_TRANSACTION',$originalToken,'Process')}
    $resolved=[IO.Path]::GetFullPath($fixture);$temp=[IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\')+'\'
    if(-not $resolved.StartsWith($temp,[StringComparison]::OrdinalIgnoreCase) -or (Split-Path -Leaf $resolved) -notlike 'intel-ui-runtime-cpu-*'){throw 'Unsafe fixture cleanup target'}
    if(Test-Path -LiteralPath $resolved){Remove-Item -LiteralPath $resolved -Recurse -Force}
}
