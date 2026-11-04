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

# Evaluate UI dependency helpers with a function in place of Python. No native
# interpreter/pip is invoked: every invocation and exit code is a fixture.
$uiDependenciesFunction = $readinessAst.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Initialize-TinyStrataUiDependencies' }, $true)
$optionalUiFunction = $readinessAst.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Initialize-TinyStrataOptionalUI' }, $true)
if (-not $uiDependenciesFunction -or -not $optionalUiFunction) { throw 'Optional UI dependency helpers are missing.' }
function Invoke-MockedUiDependencies {
    param([string]$FunctionText, [bool]$PinsHealthy = $true, [bool]$Private = $true, [bool]$InstallFails = $false)
    & {
        param($probeFunction, $probePins, $probePrivate, $probeInstallFails)
        $probeCalls = New-Object 'System.Collections.Generic.List[object]'
        function Test-Path { param($LiteralPath, $PathType) return $true }
        function Invoke-MockedUiPython {
            $probeArguments = @($args)
            $probeCalls.Add($probeArguments)
            if ($probeArguments -contains '-c') { $global:LASTEXITCODE = $(if ($probePrivate) { 0 } else { 1 }) }
            elseif ($probeArguments -contains 'install') { $global:LASTEXITCODE = $(if ($probeInstallFails) { 1 } else { 0 }) }
            elseif ($probeArguments -contains 'check') { $global:LASTEXITCODE = 0 }
            else { $global:LASTEXITCODE = $(if ($probePins) { 0 } else { 1 }) }
        }
        Invoke-Expression $probeFunction
        $caught = $null
        try { Initialize-TinyStrataUiDependencies -SourceRoot 'C:\UI source' -WorkRoot 'C:\UI work' -Python 'Invoke-MockedUiPython' }
        catch { $caught = $_.Exception.Message }
        return [pscustomobject]@{ Calls = $probeCalls.ToArray(); Error = $caught }
    } $FunctionText $PinsHealthy $Private $InstallFails
}
$result = Invoke-MockedUiDependencies -FunctionText $uiDependenciesFunction.Extent.Text
Assert-Entrypoint (-not $result.Error -and $result.Calls.Count -eq 3) 'healthy UI dependencies are checked and reused'
Assert-Entrypoint (@($result.Calls | Where-Object { $_ -contains 'install' }).Count -eq 0) 'healthy UI does not run pip install'
Assert-Entrypoint ($result.Calls[0][-1] -eq 'C:\UI work\runtime\.venv') 'private environment identity checks the installation venv'
$result = Invoke-MockedUiDependencies -FunctionText $uiDependenciesFunction.Extent.Text -PinsHealthy $false
$installCalls = @($result.Calls | Where-Object { $_ -contains 'install' })
Assert-Entrypoint (-not $result.Error -and $installCalls.Count -eq 1) 'missing pinned UI dependency takes install path once'
Assert-Entrypoint ($installCalls[0][-1] -eq 'C:\UI source\ui\requirements.txt' -and $installCalls[0] -contains '--require-virtualenv') 'UI install uses only UI requirements and requires a venv'
Assert-Entrypoint ($installCalls[0] -contains '--isolated' -and $installCalls[0] -contains '--only-binary=:all:' -and $installCalls[0] -contains 'https://pypi.org/simple') 'UI install preserves isolated binary-only official-index policy'
$result = Invoke-MockedUiDependencies -FunctionText $uiDependenciesFunction.Extent.Text -Private $false
Assert-Entrypoint ($result.Error -like '*private Python virtual environment*' -and $result.Calls.Count -eq 1) 'global interpreter is refused before dependency or pip commands'
$result = Invoke-MockedUiDependencies -FunctionText $uiDependenciesFunction.Extent.Text -PinsHealthy $false -InstallFails $true
Assert-Entrypoint ($result.Error -like '*dependency installation failed*' -and $result.Calls.Count -eq 3) 'UI install failure prevents launch and later verification'

# Intercept only the launcher call in the actual optional-UI function. Source
# and dependencies remain separate mocked steps, for both start and NoStart.
$optionalProbeText = $optionalUiFunction.Extent.Text.Replace('& $python @arguments', 'Invoke-MockedUiLauncher @arguments')
if ($optionalProbeText -eq $optionalUiFunction.Extent.Text) { throw 'Could not safely intercept the optional UI launcher.' }
foreach ($configureOnly in @($false, $true)) {
    $result = & {
        param($probeText, $probeConfigure)
        $probeSteps = New-Object 'System.Collections.Generic.List[string]'
        $probeCapture = @{ Arguments = @() }
        function Test-Path { param($LiteralPath, $PathType) return $true }
        function Initialize-TinyStrataUiSource { param($SourceRoot, $WorkRoot) $probeSteps.Add('source') }
        function Initialize-TinyStrataUiDependencies { param($SourceRoot, $WorkRoot, $Python) $probeSteps.Add('dependencies') }
        function Initialize-TinyStrataDesktopShortcut { param($SourceRoot, $WorkRoot) $probeSteps.Add('shortcut') }
        function Invoke-MockedUiLauncher { $probeSteps.Add('launch'); $probeCapture.Arguments = @($args); $global:LASTEXITCODE = 0 }
        Invoke-Expression $probeText
        Initialize-TinyStrataOptionalUI -SourceRoot 'C:\UI source' -WorkRoot 'C:\UI work' -ApiPort 8080 -ConfigureOnly:$probeConfigure
        return [pscustomobject]@{ Steps = $probeSteps.ToArray(); Arguments = $probeCapture.Arguments }
    } $optionalProbeText $configureOnly
    Assert-Entrypoint (($result.Steps -join ',') -eq 'source,dependencies,launch,shortcut') 'UI dependencies precede launch and both start modes create a shortcut after success'
    Assert-Entrypoint (($configureOnly -and $result.Arguments -contains '--configure-only') -or (-not $configureOnly -and $result.Arguments -contains '--no-api-start')) 'UI start and configure-only keep their original launch contracts'
}
$result = & {
    param($probeText)
    $probeCapture = @{ Shortcut = $false }
    function Test-Path { param($LiteralPath, $PathType) return $true }
    function Initialize-TinyStrataUiSource { param($SourceRoot, $WorkRoot) }
    function Initialize-TinyStrataUiDependencies { param($SourceRoot, $WorkRoot, $Python) }
    function Initialize-TinyStrataDesktopShortcut { param($SourceRoot, $WorkRoot) $probeCapture.Shortcut = $true }
    function Invoke-MockedUiLauncher { $global:LASTEXITCODE = 1 }
    Invoke-Expression $probeText
    $caught = $false
    try { Initialize-TinyStrataOptionalUI -SourceRoot 'C:\UI source' -WorkRoot 'C:\UI work' -ApiPort 8080 }
    catch { $caught = $_.Exception.Message -like '*could not be prepared or started*' }
    return [pscustomobject]@{ Caught = $caught; Shortcut = $probeCapture.Shortcut }
} $optionalProbeText
Assert-Entrypoint ($result.Caught -and -not $result.Shortcut) 'failed UI preparation does not create a misleading success shortcut'

# Exercise actual Shell Link serialization, but only in a private temporary
# desktop. No launcher, interpreter, application or production desktop is used.
$shortcutFunction = $readinessAst.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Initialize-TinyStrataDesktopShortcut' }, $true)
if (-not $shortcutFunction) { throw 'Desktop shortcut helper is missing.' }
Invoke-Expression $shortcutFunction.Extent.Text
$shortcutTempRoot = [IO.Path]::GetFullPath((Join-Path ([IO.Path]::GetTempPath()) ('tinystrata-shortcut-test-' + [Guid]::NewGuid().ToString('N'))))
try {
    $shortcutUnicode = [string][char]0x65E5 + [string][char]0x672C
    $shortcutSource = Join-Path $shortcutTempRoot ("source ' quoted & " + $shortcutUnicode)
    $shortcutWork = Join-Path $shortcutTempRoot ("work ' quoted & " + $shortcutUnicode)
    $shortcutDesktop = Join-Path $shortcutTempRoot 'private desktop'
    $shortcutWeb = Join-Path $shortcutSource 'ui\web'
    New-Item -ItemType Directory -Path $shortcutWeb -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $entrypointRoot 'ui\web\terminal-logo.ico') -Destination (Join-Path $shortcutWeb 'terminal-logo.ico')
    Initialize-TinyStrataDesktopShortcut -SourceRoot $shortcutSource -WorkRoot $shortcutWork -DesktopDirectory $shortcutDesktop
    $shortcutPath = Join-Path $shortcutDesktop 'TinyStrata.lnk'
    $shortcutHelper = Join-Path $shortcutWork 'runtime\ui\desktop-launch.ps1'
    $shortcutBytes = [IO.File]::ReadAllBytes($shortcutHelper)
    Assert-Entrypoint ($shortcutBytes[0] -eq 239 -and $shortcutBytes[1] -eq 187 -and $shortcutBytes[2] -eq 191) 'desktop launcher keeps Unicode paths readable by Windows PowerShell 5.1'
    $shortcutShell = New-Object -ComObject WScript.Shell
    $shortcut = $shortcutShell.CreateShortcut($shortcutPath)
    Assert-Entrypoint ($shortcut.TargetPath -eq (Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe')) 'shortcut uses the stable system PowerShell executable'
    Assert-Entrypoint ($shortcut.Arguments -eq ('-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}"' -f $shortcutHelper)) 'shortcut quotes its persistent helper path and starts hidden'
    Assert-Entrypoint ($shortcut.WorkingDirectory -eq $shortcutSource -and $shortcut.IconLocation -eq ((Join-Path $shortcutWeb 'terminal-logo.ico') + ',0')) 'shortcut has the installation working directory and app icon'
    $shortcutHelperTokens = $null; $shortcutHelperErrors = $null
    $shortcutHelperAst = [System.Management.Automation.Language.Parser]::ParseFile($shortcutHelper, [ref]$shortcutHelperTokens, [ref]$shortcutHelperErrors)
    Assert-Entrypoint ($shortcutHelperErrors.Count -eq 0) 'generated launcher parses with apostrophes spaces ampersands and Unicode paths'
    foreach ($fixture in @(@{ Variable = 'tinyStrataSource'; Expected = $shortcutSource }, @{ Variable = 'tinyStrataWork'; Expected = $shortcutWork })) {
        $assignment = $shortcutHelperAst.Find({ param($node) $node -is [System.Management.Automation.Language.AssignmentStatementAst] -and $node.Left.VariablePath.UserPath -eq $fixture.Variable }, $true)
        Assert-Entrypoint ($assignment.Right.Expression.Value -eq $fixture.Expected) 'generated launcher preserves an exact directory without shell interpolation'
    }
    $shortcutHelperText = [IO.File]::ReadAllText($shortcutHelper)
    Assert-Entrypoint ($shortcutHelperText -notmatch '--model-family|--api-port|--port|--no-api-start' -and $shortcutHelperText -match '--source-dir' -and $shortcutHelperText -match '--work-dir') 'shortcut reads saved model and ports and can restart stopped services'
    Assert-Entrypoint ($shortcutHelperText -match '-WindowStyle Hidden' -and $shortcutHelperText -match '-RedirectStandardOutput' -and $shortcutHelperText -match '-RedirectStandardError') 'desktop launch stays hidden and keeps diagnostics'
    $quoteFunction = $shortcutHelperAst.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'ConvertTo-TinyStrataWindowsArgument' }, $true)
    Invoke-Expression $quoteFunction.Extent.Text
    Assert-Entrypoint ((ConvertTo-TinyStrataWindowsArgument 'C:\space path\') -eq '"C:\space path\\"') 'Windows argument quoting doubles trailing slashes before a closing quote'
    Assert-Entrypoint ((ConvertTo-TinyStrataWindowsArgument 'embedded"quote') -eq '"embedded\"quote"') 'Windows argument quoting escapes embedded quotes'
    Initialize-TinyStrataDesktopShortcut -SourceRoot $shortcutSource -WorkRoot $shortcutWork -DesktopDirectory $shortcutDesktop
    Assert-Entrypoint (@(Get-ChildItem -LiteralPath $shortcutDesktop -File).Count -eq 1 -and [IO.File]::ReadAllText($shortcutHelper) -eq $shortcutHelperText) 'reinstall refreshes one shortcut idempotently without duplicate launchers'
    Remove-Item -LiteralPath (Join-Path $shortcutWeb 'terminal-logo.ico')
    Initialize-TinyStrataDesktopShortcut -SourceRoot $shortcutSource -WorkRoot $shortcutWork -DesktopDirectory $shortcutDesktop
    Assert-Entrypoint ($shortcutShell.CreateShortcut($shortcutPath).IconLocation -eq ((Join-Path $env:WINDIR 'System32\shell32.dll') + ',1')) 'older UI assets retain a working shortcut with a fallback icon'
    Remove-Item -LiteralPath $shortcutPath
    New-Item -ItemType Directory -Path $shortcutPath | Out-Null
    Assert-EntrypointThrows { Initialize-TinyStrataDesktopShortcut -SourceRoot $shortcutSource -WorkRoot $shortcutWork -DesktopDirectory $shortcutDesktop } 'an existing directory at the shortcut path is never overwritten'
    Assert-Entrypoint ((Test-Path -LiteralPath $shortcutPath -PathType Container) -and [IO.File]::ReadAllText($shortcutHelper) -eq $shortcutHelperText) 'blocked shortcut creation preserves unrelated directory and helper contents'
} finally {
    $shortcutTempParent = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (-not $shortcutTempRoot.StartsWith($shortcutTempParent, [StringComparison]::OrdinalIgnoreCase) -or
        -not ([IO.Path]::GetFileName($shortcutTempRoot)).StartsWith('tinystrata-shortcut-test-')) { throw 'Refusing to remove an unexpected shortcut test directory.' }
    if (Test-Path -LiteralPath $shortcutTempRoot) { Remove-Item -LiteralPath $shortcutTempRoot -Recurse -Force }
}
$installerBody = $readinessAst.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Invoke-TinyStrataInstaller' }, $true)
$inspectBranch = $installerBody.Find({ param($node) $node -is [System.Management.Automation.Language.IfStatementAst] -and $node.Clauses[0].Item1.Extent.Text -eq '$InspectOnly' }, $true)
Assert-Entrypoint ($inspectBranch -and $inspectBranch.Extent.Text -notmatch 'Initialize-TinyStrata' -and $inspectBranch.Extent.Text -match '\breturn\b') 'CheckOnly remains an inspection path without dependency installation'
$rootRequirements = [IO.File]::ReadAllText((Join-Path $entrypointRoot 'requirements.txt'))
$uiRequirements = [IO.File]::ReadAllText((Join-Path $entrypointRoot 'ui\requirements.txt'))
Assert-Entrypoint ($rootRequirements -notmatch '(?im)^ddgs' -and $uiRequirements -match '(?m)^ddgs==9\.16\.1\s*$') 'DDGS is pinned in UI-only requirements'

[pscustomobject]@{
    Result = 'passed'
    PowerShell = $PSVersionTable.PSVersion.ToString()
    Checks = $entrypointChecks
    Installed = $false
    NetworkRequests = 0
    NativeProcesses = 0
} | ConvertTo-Json -Compress
