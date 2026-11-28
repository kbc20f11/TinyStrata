#Requires -Version 5.1
# CPU-only admission fixtures. No process, listener, native runtime or network.
$ErrorActionPreference='Stop'
$repository=Split-Path -Parent $PSScriptRoot
$tokens=$null;$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $repository 'install.ps1'),[ref]$tokens,[ref]$errors)
if($errors.Count){throw 'Installer parse failure'}
foreach($name in @('Invoke-TinyStrataInstaller','Get-TinyStrataModelFamily','Get-TinyStrataConfig','Test-TinyStrataInferenceReady','Enter-TinyStrataInstallationGuard','Exit-TinyStrataInstallationGuard')){
    $node=$ast.Find({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name},$true)
    Invoke-Expression $node.Extent.Text
}
$root=Join-Path ([IO.Path]::GetTempPath()) ('tinystrata-shared-discovery-'+[Guid]::NewGuid().ToString('N'))
function Test-TinyStrataSource {param($Directory)return $true}
function Get-TinyStrataBackend {param($Requested,$Configuration)return $Requested}
function Get-TinyStrataIntelRuntimeSelection {param($Model,$RequestedProfile)return [pscustomobject]@{Profile='b390-qwen36';AotTarget='intel_gpu_ptl_h'}}
function Assert-TinyStrataIntelManagedDeviceSelection {param($Configuration)}
function Get-TinyStrataRunningApiProfiles {param($SourceRoot,$RuntimeRoot,$Families,$Port,$Backend)return @()}
function Get-Process {param($Id,$ErrorAction)if($Id -eq 43210){return [pscustomobject]@{Id=43210}}}
function Stop-TinyStrataOwnedServicesForUpdate {param($SourceRoot,$WorkRoot)
    if(-not(Test-Path -LiteralPath (Join-Path $WorkRoot '.tinystrata-installing.json'))){throw 'Missing installation guard'}
    # Presence never authorizes termination. The shipping strict inspector is
    # separately tested; stop here before even a fixture settings mutation.
    $script:inspected++
    throw 'Reached strict owned inspector before preparing shared files'
}
try{
    foreach($case in @('intel-ui-only','cuda-api-before-intel-prepare','intel-api-before-cuda-prepare')){
        $source=Join-Path $root ($case+'-source');$work=Join-Path $root ($case+'-work')
        [IO.Directory]::CreateDirectory((Join-Path $source '.git'))|Out-Null
        $relative=if($case -eq 'intel-ui-only'){'runtime\ui-sycl\ui-process.json'}elseif($case -eq 'cuda-api-before-intel-prepare'){'runtime\server-state.json'}else{'runtime\server-state-sycl.json'}
        $path=Join-Path $work $relative
        [IO.Directory]::CreateDirectory((Split-Path -Parent $path))|Out-Null
        [IO.File]::WriteAllText($path,'{"pid":43210}')
        $before=[IO.File]::ReadAllText($path);$script:inspected=0;$failed=$false
        try{Invoke-TinyStrataInstaller -SourceDirectory $source -WorkingDirectory $work -Backend $(if($case -eq 'intel-api-before-cuda-prepare'){'cuda'}else{'sycl'}) -DoNotStart $true}
        catch{if($_.Exception.Message -notmatch 'Reached strict owned inspector'){throw};$failed=$true}
        if(-not $failed -or $script:inspected -ne 1 -or [IO.File]::ReadAllText($path) -cne $before -or
            (Test-Path -LiteralPath (Join-Path $work '.tinystrata-installing.json'))){throw ('Shared-root discovery failed: '+$case)}
    }
    # An explicit mode matching the current app is a no-op, including WithUI.
    # Use the actual helper/catalog definitions for peer identity/mode checks;
    # no stage is loaded and no process/HTTP/UI boundary is real.
    function Get-TinyStrataSavedProcess {param($StatePath,$ConfigPath,$SourcePath,$ApiPort)return $script:currentServer}
    function Get-TinyStrataRunningApiProfiles {param($SourceRoot,$RuntimeRoot,$Families,$Port,$Backend)
        $family=Get-TinyStrataModelFamily qwen36 -Backend sycl
        return [pscustomobject]@{Family=$family;ConfigPath=$script:currentConfig;Configuration=(Get-TinyStrataConfig $script:currentConfig);
            StatePath=(Join-Path $RuntimeRoot $family.StateFile);Port=19085;Server=$script:currentServer}
    }
    function Test-TinyStrataRunningApiCurrent {param($Server,$Configuration,$SourceRoot,$Backend,$WorkRoot,$IncludeUI,$Vision)
        if(-not $IncludeUI){throw 'Requested mode must reach the normal freshness check'}
        $script:currentChecks++
        return $Vision -ceq $Configuration.vision_mode
    }
    function Test-TinyStrataIntelRuntimeCurrent {param($Configuration,$WorkRoot,$Artifact)return $true}
    function Get-NetTCPConnection {param($LocalPort,$State,$ErrorAction)return [pscustomobject]@{OwningProcess=74}}
    function Test-TinyStrataPortOwner {param($ApiPort,$RootProcessId)return $true}
    function Invoke-TinyStrataApi {param($BaseUrl,$Path,$Method,$Headers)
        if($Path -ne '/health'){throw 'No model download or mutation is allowed on no-op reuse'}
        return [pscustomobject]@{status='ok';backend='sycl';intel_profile='b390-qwen36';experimental=$false;model='qwen3.6-35b-a3b';loaded=$false;images=$false}
    }
    function Initialize-TinyStrataOptionalUI {param($SourceRoot,$WorkRoot,$ApiPort,$UiPort,$Family,[switch]$ConfigureOnly,$Backend,$IntelProfile,$Vision)
        if($ApiPort -ne 19085 -or $UiPort -ne 3905 -or $Backend -ne 'sycl' -or $Vision -cne $script:mode){throw 'No-op UI selections changed'}
        $script:uiReuses++
    }
    function Start-Process {throw 'A no-op must not start a replacement API'}
    foreach($mode in @('no','gpu')){
        $script:mode=$mode;$work=Join-Path $root ('current-explicit-'+$mode)
        $configs=Join-Path $work 'runtime\.tinystrata';[IO.Directory]::CreateDirectory($configs)|Out-Null
        foreach($name in @('qwen36','tiel','gemma4','lfm2','clover')){
            $family=Get-TinyStrataModelFamily $name -Backend sycl
            $cfg=[ordered]@{backend='sycl';model_name=$family.Model;arch=$family.Arch;intel_profile=$family.IntelProfile;port=19085;
                vision_mode=$(if($name -eq 'lfm2'){'no'}else{$mode});api_key='retained-no-op-key';
                exe=(Join-Path $work ('native\sycl-'+('0'*32)+'\tinystrata-intel.exe'))}
            [IO.File]::WriteAllText((Join-Path $configs $family.ConfigFile),($cfg|ConvertTo-Json -Depth 5))
        }
        $settings=Join-Path $work 'runtime\ui-sycl\settings.json';[IO.Directory]::CreateDirectory((Split-Path -Parent $settings))|Out-Null
        [IO.File]::WriteAllText($settings,'{"model_family":"qwen36","port":3905}')
        $script:currentConfig=Join-Path $configs 'config-sycl.json';$before=[IO.File]::ReadAllText($script:currentConfig)
        $script:currentServer=[pscustomobject]@{Id=74;HasExited=$false};$script:currentChecks=0;$script:uiReuses=0;$script:inspected=0
        Invoke-TinyStrataInstaller -SourceDirectory $repository -WorkingDirectory $work -Backend sycl -IncludeUI $true -Vision $mode -DoNotDownload $true
        if($script:currentChecks -ne 1 -or $script:uiReuses -ne 1 -or $script:inspected -ne 0 -or
            [IO.File]::ReadAllText($script:currentConfig) -cne $before -or (Test-Path -LiteralPath (Join-Path $work '.tinystrata-installing.json'))){throw ('Matching explicit Vision was not a no-op: '+$mode)}
    }
    $failed=$false;$script:inspected=0;$script:currentChecks=0
    try{Invoke-TinyStrataInstaller -SourceDirectory $repository -WorkingDirectory $work -Backend sycl -IncludeUI $true -Vision no -DoNotDownload $true}
    catch{if($_.Exception.Message -notmatch 'Reached strict owned inspector'){throw};$failed=$true}
    if(-not $failed -or $script:inspected -ne 1 -or $script:currentChecks -ne 1 -or
        [IO.File]::ReadAllText($script:currentConfig) -cne $before -or (Test-Path -LiteralPath (Join-Path $work '.tinystrata-installing.json'))){throw 'A changed Vision selection did not request a guarded transaction'}
    Write-Output 'Shared-root/UI-only discovery 3 + unchanged Vision 2 + changed Vision 1: PASS; no process or shared-file mutation.'
}finally{
    $resolved=[IO.Path]::GetFullPath($root);$temp=[IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\')+'\'
    if(-not $resolved.StartsWith($temp,[StringComparison]::OrdinalIgnoreCase) -or (Split-Path -Leaf $resolved) -notlike 'tinystrata-shared-discovery-*'){throw 'Unsafe fixture cleanup target'}
    if(Test-Path -LiteralPath $resolved){Remove-Item -LiteralPath $resolved -Recurse -Force}
}
