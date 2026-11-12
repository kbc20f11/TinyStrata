#Requires -Version 5.1
<# Actual entrypoint/default/ready-flow functions with private files and mocked services only. #>
$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$installer = Join-Path $repository 'install.ps1'
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($installer, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw 'Installer parse failed.' }
$text = [IO.File]::ReadAllText($installer)
$header = $text.Substring(0, $ast.ParamBlock.Extent.EndOffset)
$tail = $text.Substring($text.IndexOf('$tinyStrataOldErrorAction ='))
$familyDefinition = $ast.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Get-TinyStrataModelFamily' }, $true).Extent.Text
$checks = 0
function Assert-Flow([bool]$Passed, [string]$Message) {
    if (-not $Passed) { throw "UI install flow fixture failed: $Message" }
    $script:checks++
}
$fixtureRoot = Join-Path ([IO.Path]::GetTempPath()) ('tinystrata-ui-flow-' + [Guid]::NewGuid().ToString('N'))
$capture = @'
function Invoke-TinyStrataInstaller {
    param($SourceDirectory,$WorkingDirectory,$ApiPort,$DoNotStart,$DoNotDownload,$InspectOnly,$BuildAgain,$RefreshSource,
          $Family,$Backend,$HipDirectory,$HipArchitectures,$IncludeUI,$UiPort,$Revision,$FamilyExplicit,$UiPortExplicit)
    return [pscustomobject]@{ Family=$Family; IncludeUI=$IncludeUI; DoNotDownload=$DoNotDownload; UiPort=$UiPort;
                             FamilyExplicit=$FamilyExplicit; UiPortExplicit=$UiPortExplicit; Revision=$Revision }
}
'@
try {
    $source = Join-Path $fixtureRoot 'source'
    $work = Join-Path $fixtureRoot 'work'
    [IO.Directory]::CreateDirectory($source) | Out-Null
    [IO.Directory]::CreateDirectory((Join-Path $work 'runtime\ui')) | Out-Null
    $entry = [scriptblock]::Create($header + "`n" + $familyDefinition + "`n" + $capture + "`n" + $tail)
    $fresh = & $entry -SourceDir $source -WorkDir $work -WithUI
    Assert-Flow ($fresh.IncludeUI -and $fresh.DoNotDownload -and $fresh.Family -eq 'qwen36') 'fresh UI installs no model by default'
    $explicit = & $entry -SourceDir $source -WorkDir $work -WithUI -ModelFamily lfm2
    Assert-Flow ($explicit.DoNotDownload -and $explicit.Family -eq 'lfm2' -and $explicit.FamilyExplicit) 'explicit UI family also does not trigger a model transfer'
    $api = & $entry -SourceDir $source -WorkDir $work
    Assert-Flow (-not $api.IncludeUI -and -not $api.DoNotDownload) 'API-only installer retains its intentional model installation behavior'
    [IO.File]::WriteAllText((Join-Path $work 'runtime\ui\settings.json'), '{"model_family":"gemma4","port":3001}', (New-Object Text.UTF8Encoding($false)))
    $existing = & $entry -SourceDir $source -WorkDir $work -WithUI
    Assert-Flow ($existing.Family -eq 'gemma4' -and -not $existing.FamilyExplicit -and $existing.UiPort -eq 3001) 'ordinary repeated UI command preserves saved model and UI port'
    $override = & $entry -SourceDir $source -WorkDir $work -WithUI -ModelFamily clover -UiPort 3002
    Assert-Flow ($override.Family -eq 'clover' -and $override.UiPort -eq 3002 -and $override.FamilyExplicit -and $override.UiPortExplicit) 'explicit family and port remain deliberate overrides'

    # Evaluate the actual installer function, but reuse a harmless mock API so
    # no preparation branch, process creation, native probe or network can run.
    foreach ($name in @('Invoke-TinyStrataInstaller', 'Get-TinyStrataConfig', 'Get-TinyStrataModelFamily', 'Test-TinyStrataInferenceReady')) {
        $definition = $ast.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name }, $true)
        Invoke-Expression $definition.Extent.Text
    }
    function Test-TinyStrataSource { param($Directory) return $true }
    function Get-TinyStrataBackend { param($Requested,$Configuration) return 'hip' }
    function Get-TinyStrataSavedProcess { param($StatePath,$ConfigPath,$SourcePath,$ApiPort) return [pscustomobject]@{ Id=71; HasExited=$false } }
    function Get-NetTCPConnection { param($LocalPort,$State,$ErrorAction) return [pscustomobject]@{ OwningProcess=71 } }
    function Test-TinyStrataPortOwner { param($ApiPort,$RootProcessId) return $true }
    function Start-Process { throw 'Fixture must not start a process.' }
    function Initialize-TinyStrataOptionalUI { param($SourceRoot,$WorkRoot,$ApiPort,$UiPort,$Family,[switch]$ConfigureOnly) $script:uiPrepared++ }
    function Invoke-TinyStrataApi {
        param($BaseUrl,$Path,$Method,$Headers)
        if ($Path -eq '/health') { return [pscustomobject]@{ status='ok'; loaded=($script:transfers -gt 0); images=$false } }
        if ($Path -eq '/v1/model/download' -and $Method -eq 'POST') { $script:transfers++; return [pscustomobject]@{ state='ready' } }
        if ($Path -eq '/v1/model/download') { return [pscustomobject]@{ state='ready' } }
        throw "Unexpected fixture endpoint $Path"
    }
    [IO.Directory]::CreateDirectory((Join-Path $work 'runtime\.tinystrata')) | Out-Null
    [IO.File]::WriteAllText((Join-Path $work 'runtime\.tinystrata\config.json'), '{"arch":"qwen35moe","vision_mode":"no","port":18080}', (New-Object Text.UTF8Encoding($false)))
    $script:transfers=0; $script:uiPrepared=0
    Invoke-TinyStrataInstaller -SourceDirectory $source -WorkingDirectory $work -ApiPort 18080 -IncludeUI $true -DoNotDownload $false -DoNotStart $false
    Assert-Flow ($script:transfers -eq 0 -and $script:uiPrepared -eq 1) 'an unloaded API opens the UI without calling model download, even if a caller passes DoNotDownload=false'
    Invoke-TinyStrataInstaller -SourceDirectory $source -WorkingDirectory $work -ApiPort 18080 -IncludeUI $false -DoNotDownload $false -DoNotStart $false
    Assert-Flow ($script:transfers -eq 1) 'API-only readiness still requests its intentional fixed-model installation'
    [pscustomobject]@{ Result='passed'; Checks=$checks; RealNetworkRequests=0; NativeProcesses=0; RealModelTransfers=0 } | ConvertTo-Json -Compress
} finally {
    $target=[IO.Path]::GetFullPath($fixtureRoot)
    $temporary=[IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\')+'\'
    if (-not $target.StartsWith($temporary,[StringComparison]::OrdinalIgnoreCase) -or -not ([IO.Path]::GetFileName($target)).StartsWith('tinystrata-ui-flow-')) { throw 'Invalid private flow cleanup target.' }
    if(Test-Path -LiteralPath $target){Remove-Item -LiteralPath $target -Recurse -Force}
}
