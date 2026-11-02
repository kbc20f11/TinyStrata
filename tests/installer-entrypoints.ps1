#Requires -Version 5.1
<#
Run with powershell.exe or pwsh.exe -NoProfile -File tests/installer-entrypoints.ps1.
Exercises installer parameter binding and the UI bootstrap without installing,
starting processes, querying a GPU, or making network requests. The real
headers are parsed from source; the UI's HTTP calls return a harmless stub.
#>
$ErrorActionPreference = 'Stop'
$entrypointRoot = Split-Path -Parent $PSScriptRoot
$entrypointChecks = 0

function Assert-Entrypoint {
    param([bool]$Passed, [string]$Name)
    if (-not $Passed) { throw "Installer entrypoint regression: $Name" }
    $script:entrypointChecks++
}

function Assert-EntrypointThrows {
    param([scriptblock]$Action, [string]$Name)
    $caught = $false
    try { $null = & $Action } catch { $caught = $true }
    Assert-Entrypoint -Passed $caught -Name $Name
}

function Get-EntrypointHeader {
    param([string]$Path)
    $tokens = $null
    $parseErrors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$tokens, [ref]$parseErrors)
    if ($parseErrors.Count -or -not $ast.ParamBlock) { throw "Invalid installer syntax: $Path" }
    $text = [IO.File]::ReadAllText($Path)
    return $text.Substring(0, $ast.ParamBlock.Extent.EndOffset)
}

$entrypointCapture = @'

[pscustomobject]@{
    Reached = $true
    HipArchitectures = $HipArchitectures
    Backend = $Backend
    ModelFamily = $ModelFamily
    Port = $Port
    UiPort = $UiPort
    CheckOnly = [bool]$CheckOnly
    WithUI = [bool](Get-Variable -Name WithUI -ValueOnly -ErrorAction SilentlyContinue)
}
'@

function Invoke-EntrypointHeader {
    param([string]$Header, [ValidateSet('Iex', 'Call')][string]$Invocation = 'Iex',
          [hashtable]$Arguments = @{}, [hashtable]$Ambient = @{})
    & {
        param($probeHeader, $probeInvocation, $probeArguments, $probeAmbient, $probeCapture)
        foreach ($name in $probeAmbient.Keys) { Set-Variable -Name $name -Value $probeAmbient[$name] -Scope Local }
        $probeText = $probeHeader + $probeCapture
        if ($probeInvocation -eq 'Iex') { Invoke-Expression -Command $probeText }
        else { & ([scriptblock]::Create($probeText)) @probeArguments }
    } $Header $Invocation $Arguments $Ambient $entrypointCapture
}

$entrypointCorePath = Join-Path $entrypointRoot 'install.ps1'
$entrypointUiPath = Join-Path $entrypointRoot 'install-ui.ps1'
$entrypointCoreHeader = Get-EntrypointHeader -Path $entrypointCorePath
$entrypointUiHeader = Get-EntrypointHeader -Path $entrypointUiPath

foreach ($header in @($entrypointCoreHeader, $entrypointUiHeader)) {
    $result = Invoke-EntrypointHeader -Header $header
    Assert-Entrypoint ($result.Reached -and [string]::IsNullOrEmpty($result.HipArchitectures)) 'fresh IEX permits omitted architecture'
    Assert-Entrypoint ($result.Backend -eq 'auto' -and $result.ModelFamily -eq 'qwen36' -and $result.Port -eq 0 -and $result.UiPort -eq 3000) 'IEX preserves valid defaults'

    $result = Invoke-EntrypointHeader -Header $header -Ambient @{ HipArchitectures = '' }
    Assert-Entrypoint ($result.Reached -and [string]::IsNullOrEmpty($result.HipArchitectures)) 'IEX with an existing empty architecture'

    $result = Invoke-EntrypointHeader -Header $header -Ambient @{ Backend = 'vulkan'; ModelFamily = 'other'; Port = -1; UiPort = 0 }
    Assert-Entrypoint ($result.Backend -eq 'auto' -and $result.ModelFamily -eq 'qwen36' -and $result.Port -eq 0 -and $result.UiPort -eq 3000) 'defaults replace unrelated ambient values'

    foreach ($architecture in @('', 'gfx1200', 'gfx1200;gfx1201')) {
        $result = Invoke-EntrypointHeader -Header $header -Invocation Call -Arguments @{ HipArchitectures = $architecture; CheckOnly = $true }
        Assert-Entrypoint ($result.Reached -and $result.HipArchitectures -eq $architecture -and $result.CheckOnly) 'valid explicit architecture and CheckOnly bind'
    }
    foreach ($architecture in @('gfx', 'gfx1200;', 'gfx1200;;gfx1201', 'not-gpu')) {
        Assert-EntrypointThrows { Invoke-EntrypointHeader -Header $header -Invocation Call -Arguments @{ HipArchitectures = $architecture } } 'malformed architecture rejected'
    }
    foreach ($invalid in @(@{ Backend = 'vulkan' }, @{ ModelFamily = 'other' }, @{ Port = 65536 }, @{ UiPort = 0 })) {
        Assert-EntrypointThrows { Invoke-EntrypointHeader -Header $header -Invocation Call -Arguments $invalid } 'other ValidateSet/Range rejects invalid explicit input'
    }

    # The reported failure can leave a typed variable in this same scope.
    # Retrying the fixed header must work without opening another PS window.
    $retry = & {
        param($fixedHeader, $capture)
        $legacyHeader = $fixedHeader.Replace('^(gfx[0-9a-f]+(;gfx[0-9a-f]+)*)?$', '^gfx[0-9a-f]+(;gfx[0-9a-f]+)*$')
        if ($legacyHeader -eq $fixedHeader) { throw 'Cannot reconstruct the original strict installer header' }
        $legacyFailed = $false
        try { $null = Invoke-Expression ($legacyHeader + "`n'legacy body'") } catch { $legacyFailed = $true }
        if (-not $legacyFailed) { throw 'Legacy IEX reproducer unexpectedly passed' }
        $first = Invoke-Expression ($fixedHeader + $capture)
        $second = Invoke-Expression ($fixedHeader + $capture)
        return [pscustomobject]@{ LegacyFailed = $legacyFailed; First = $first; Second = $second }
    } $header $entrypointCapture
    Assert-Entrypoint ($retry.LegacyFailed -and $retry.First.Reached -and $retry.Second.Reached) 'failed legacy IEX followed by two fixed retries in same scope'
}

function Invoke-MockedUiBootstrap {
    param([string]$UiText, [string]$CoreHeader, [hashtable]$Arguments = @{}, [switch]$Iex)
    & {
        param($probeUi, $probeCore, $probeArguments, $probeIex, $probeCapture)
        $probeRequests = New-Object 'System.Collections.Generic.List[string]'
        $probeCoreStub = $probeCore + $probeCapture
        # Force the downloaded-ScriptBlock route regardless of the caller's
        # PSScriptRoot. No real installer body or HTTP command may run here.
        function Test-Path { param($LiteralPath, $Path, $PathType) return $false }
        function Invoke-RestMethod {
            param($Uri, $Headers, $TimeoutSec)
            $probeRequests.Add([string]$Uri)
            return [pscustomobject]@{ sha = ('1' * 40) }
        }
        function Invoke-WebRequest {
            param($Uri, [switch]$UseBasicParsing, $TimeoutSec)
            $probeRequests.Add([string]$Uri)
            return [pscustomobject]@{ Content = $probeCoreStub }
        }
        $forwarded = if ($probeIex) { Invoke-Expression $probeUi }
                     else { & ([scriptblock]::Create($probeUi)) @probeArguments }
        return [pscustomobject]@{ Forwarded = $forwarded; Requests = $probeRequests.ToArray() }
    } $UiText $CoreHeader $Arguments ([bool]$Iex) $entrypointCapture
}

$entrypointUiText = [IO.File]::ReadAllText($entrypointUiPath)
$result = Invoke-MockedUiBootstrap -UiText $entrypointUiText -CoreHeader $entrypointCoreHeader -Iex
Assert-Entrypoint ($result.Forwarded.Reached -and $result.Forwarded.WithUI -and $result.Forwarded.Backend -eq 'auto') 'full UI IEX defaults forward WithUI'
Assert-Entrypoint ($result.Requests.Count -eq 2 -and $result.Requests[1] -like ('*/' + ('1' * 40) + '/install.ps1')) 'UI bootstrap uses the resolved commit'
$result = Invoke-MockedUiBootstrap -UiText $entrypointUiText -CoreHeader $entrypointCoreHeader -Arguments @{
    CheckOnly = $true; Backend = 'hip'; HipArchitectures = 'gfx1200;gfx1201'; ModelFamily = 'gemma4'; Port = 8080; UiPort = 3001
}
Assert-Entrypoint ($result.Forwarded.WithUI -and $result.Forwarded.CheckOnly -and $result.Forwarded.Port -eq 8080 -and $result.Forwarded.UiPort -eq 3001) 'UI forwards CheckOnly and requested ports'
Assert-Entrypoint ($result.Forwarded.Backend -eq 'hip' -and $result.Forwarded.ModelFamily -eq 'gemma4' -and $result.Forwarded.HipArchitectures -eq 'gfx1200;gfx1201') 'UI forwards backend family and multiple architectures'

# Evaluate only the pure readiness predicate, never the installer body.
$readinessTokens = $null
$readinessErrors = $null
$readinessAst = [System.Management.Automation.Language.Parser]::ParseFile($entrypointCorePath, [ref]$readinessTokens, [ref]$readinessErrors)
$readinessFunction = $readinessAst.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Test-TinyStrataInferenceReady' }, $true)
if (-not $readinessFunction) { throw 'Installer readiness helper is missing.' }
Invoke-Expression $readinessFunction.Extent.Text
foreach ($mode in @('gpu', 'cpu')) {
    $configuration = [pscustomobject]@{ arch = 'clover_moe'; vision_mode = $mode }
    Assert-Entrypoint (-not (Test-TinyStrataInferenceReady ([pscustomobject]@{ loaded = $true; images = $false }) $configuration)) 'Clover image mode requires image readiness'
    Assert-Entrypoint (Test-TinyStrataInferenceReady ([pscustomobject]@{ loaded = $true; images = $true }) $configuration) 'Clover image mode becomes ready with encoder'
}
foreach ($arch in @('clover_moe', 'lfm2moe')) {
    Assert-Entrypoint (Test-TinyStrataInferenceReady ([pscustomobject]@{ loaded = $true; images = $false }) ([pscustomobject]@{ arch = $arch; vision_mode = 'no' })) 'explicit text-only mode remains ready without encoder'
}
Assert-Entrypoint (-not (Test-TinyStrataInferenceReady ([pscustomobject]@{ loaded = $false; images = $true }) ([pscustomobject]@{ arch = 'clover_moe'; vision_mode = 'gpu' }))) 'Clover readiness still requires loaded language engine'

# Parse only the compiler-version predicates; never execute nvcc or install.
foreach ($name in @('Get-TinyStrataCudaCompilerVersion', 'Test-TinyStrataCudaCompilerVersion')) {
    $versionFunction = $readinessAst.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name }, $true)
    if (-not $versionFunction) { throw "Installer CUDA version helper is missing: $name" }
    Invoke-Expression $versionFunction.Extent.Text
}
foreach ($fixture in @(
    @{ Text = 'Cuda compilation tools, release 13.2, V13.2.51'; Safe = $false },
    @{ Text = 'Cuda compilation tools, release 13.2, V13.2.78'; Safe = $false },
    @{ Text = 'Cuda compilation tools, release 13.2, V13.2.85'; Safe = $false },
    @{ Text = 'Cuda compilation tools, release 13.2, V13.2.86'; Safe = $true },
    @{ Text = 'Cuda compilation tools, release 13.3, V13.3.0'; Safe = $true },
    @{ Text = 'Cuda compilation tools, release 14.0, V14.0.0'; Safe = $true },
    @{ Text = 'Cuda compilation tools, release 13.1, V13.1.115'; Safe = $false },
    @{ Text = 'Cuda compilation tools, release 13.2'; Safe = $false },
    @{ Text = ''; Safe = $false }
)) {
    $compilerVersion = Get-TinyStrataCudaCompilerVersion -Text $fixture.Text
    Assert-Entrypoint ((Test-TinyStrataCudaCompilerVersion -Version $compilerVersion) -eq $fixture.Safe) ('CUDA compiler patch-level validation: ' + $fixture.Text)
}

[pscustomobject]@{
    Result = 'passed'
    PowerShell = $PSVersionTable.PSVersion.ToString()
    Checks = $entrypointChecks
    Installed = $false
    NetworkRequests = 0
    NativeProcesses = 0
} | ConvertTo-Json -Compress
