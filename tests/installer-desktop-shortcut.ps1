#Requires -Version 5.1
<# Private file/COM fixtures only: no real Desktop writes, UI, model or service starts. #>
$ErrorActionPreference = 'Stop'
$checks = 0
function Assert-Shortcut { param([bool]$Value, [string]$Message) if (-not $Value) { throw $Message }; $script:checks++ }
$installer = Join-Path (Split-Path -Parent $PSScriptRoot) 'install.ps1'
$tokens = $null; $errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile($installer, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw 'Installer parse failed.' }
function Read-InstallerFunction {
    param([string]$Name)
    $node = $ast.Find({ param($item) $item -is [Management.Automation.Language.FunctionDefinitionAst] -and $item.Name -eq $Name }, $true)
    if (-not $node) { throw "Missing installer function: $Name" }
    return $node.Extent.Text
}
$shortcutText = Read-InstallerFunction 'Initialize-TinyStrataDesktopShortcut'
$optionalText = Read-InstallerFunction 'Initialize-TinyStrataOptionalUI'
$root = [IO.Path]::GetFullPath((Join-Path ([IO.Path]::GetTempPath()) ('tinystrata-desktop-' + [Guid]::NewGuid().ToString('N'))))
try {
    $unicode = [string][char]0x8cc7 + [string][char]0x6599
    $source = Join-Path $root ("source [literal] ' & " + $unicode)
    $work = Join-Path $root ("work [literal] ' & " + $unicode)
    $desktop = Join-Path $root ("OneDrive ' & " + $unicode + '\Redirected Desktop')
    [IO.Directory]::CreateDirectory((Join-Path $source 'ui')) | Out-Null
    [IO.Directory]::CreateDirectory($desktop) | Out-Null
    [IO.Directory]::CreateDirectory((Join-Path $work 'runtime\ui')) | Out-Null
    $settings = Join-Path $work 'runtime\ui\settings.json'
    [IO.File]::WriteAllText($settings, '{"model_family":"gemma4","api_port":18082,"port":3000}', [Text.Encoding]::UTF8)
    $settingsBefore = [Convert]::ToBase64String([IO.File]::ReadAllBytes($settings))
    $unrelated = Join-Path $desktop 'keep.txt'
    [IO.File]::WriteAllText($unrelated, 'unrelated desktop file', [Text.Encoding]::UTF8)
    Invoke-Expression $shortcutText
    Initialize-TinyStrataDesktopShortcut -SourceRoot $source -WorkRoot $work -DesktopDirectory $desktop
    $link = Join-Path $desktop 'TinyStrata.lnk'
    $helper = Join-Path $work 'runtime\ui\desktop-launch.ps1'
    Assert-Shortcut ((Test-Path -LiteralPath $link -PathType Leaf)) 'Redirected private Desktop receives a real Shell Link.'
    $shell = New-Object -ComObject WScript.Shell
    $saved = $shell.CreateShortcut($link)
    Assert-Shortcut ($saved.TargetPath -eq (Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe')) 'Shortcut uses system PowerShell.'
    Assert-Shortcut ($saved.Arguments -eq ('-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}"' -f $helper)) 'Hidden launcher quotes the persistent Unicode helper.'
    Assert-Shortcut ($saved.WorkingDirectory -eq $source) 'Shortcut keeps the exact Unicode source path.'
    $bytes = [IO.File]::ReadAllBytes($helper)
    Assert-Shortcut ($bytes[0] -eq 239 -and $bytes[1] -eq 187 -and $bytes[2] -eq 191) 'Helper is UTF-8 with BOM for Windows PowerShell.'
    $helperTokens = $null; $helperErrors = $null
    $helperAst = [Management.Automation.Language.Parser]::ParseFile($helper, [ref]$helperTokens, [ref]$helperErrors)
    Assert-Shortcut ($helperErrors.Count -eq 0) 'Helper parses with apostrophes, ampersands, brackets and Unicode paths.'
    $helperBody = [IO.File]::ReadAllText($helper)
    Assert-Shortcut ($helperBody.Contains("runtime\.venv\Scripts\python.exe")) 'Helper uses this installation private Python.'
    Assert-Shortcut ($helperBody.Contains('--source-dir') -and $helperBody.Contains('--work-dir')) 'Helper carries exact installation pointers.'
    Assert-Shortcut ($helperBody -notmatch '--model-family|--model|--api-port|qwen36|--download') 'Shortcut follows saved/current selection without forcing a model or download.'
    Assert-Shortcut ($helperBody -match 'WindowStyle Hidden' -and $helperBody -match 'RedirectStandardError') 'Python child starts hidden with persistent error output.'
    Assert-Shortcut ($shortcutText.Contains("[Environment]::GetFolderPath('DesktopDirectory')")) 'Default Desktop is the Windows shell KnownFolder, including redirection.'
    Assert-Shortcut ([Convert]::ToBase64String([IO.File]::ReadAllBytes($settings)) -eq $settingsBefore) 'Existing selected family and ports stay unchanged.'
    Initialize-TinyStrataDesktopShortcut -SourceRoot $source -WorkRoot $work -DesktopDirectory $desktop
    Assert-Shortcut ((Get-ChildItem -LiteralPath $desktop -Filter '*.lnk').Count -eq 1) 'Repeated install replaces only the named shortcut, without staging leftovers.'
    Assert-Shortcut ([IO.File]::ReadAllText($unrelated) -eq 'unrelated desktop file') 'Unrelated Desktop files are preserved.'
    $blockedDesktop = Join-Path $root 'blocked desktop'
    [IO.Directory]::CreateDirectory((Join-Path $blockedDesktop 'TinyStrata.lnk')) | Out-Null
    $blocked = $false
    try { Initialize-TinyStrataDesktopShortcut -SourceRoot $source -WorkRoot $work -DesktopDirectory $blockedDesktop }
    catch { $blocked = $_.Exception.Message -like '*left unchanged*' }
    Assert-Shortcut $blocked 'A conflicting directory is reported and left unchanged.'

    # Exercise the production OptionalUI body, intercepting only its native call.
    $probe = $optionalText.Replace('& $python @arguments', 'Invoke-TestUiLauncher @arguments')
    Assert-Shortcut ($probe -ne $optionalText) 'Optional UI native interception is exact.'
    foreach ($configure in @($true, $false)) {
        foreach ($exitCode in @(0, 7)) {
            $result = & {
                param($Body, $Configure, $ExitCode, $Root)
                $events = New-Object 'System.Collections.Generic.List[string]'
                $capture = @{ Arguments = @() }
                function Test-Path { param($LiteralPath, $PathType) return $true }
                function Initialize-TinyStrataUiSource { param($SourceRoot, $WorkRoot) $events.Add('source') }
                function Initialize-TinyStrataUiDependencies { param($SourceRoot, $WorkRoot, $Python) $events.Add('dependencies') }
                function Initialize-TinyStrataDesktopShortcut { param($SourceRoot, $WorkRoot) $events.Add('shortcut') }
                function Invoke-TestUiLauncher { $events.Add('launch'); $capture.Arguments = @($args); $global:LASTEXITCODE = $ExitCode }
                Invoke-Expression $Body
                $failure = ''
                try { Initialize-TinyStrataOptionalUI -SourceRoot $Root -WorkRoot $Root -ApiPort 18082 -Family gemma4 -ConfigureOnly:$Configure }
                catch { $failure = $_.Exception.Message }
                return @{ Events = $events.ToArray(); Arguments = $capture.Arguments; Failure = $failure }
            } $probe $configure $exitCode $root
            Assert-Shortcut (($result.Events -join ',') -eq 'source,dependencies,shortcut,launch') 'Shortcut exists before configure/start, including launcher failure.'
            Assert-Shortcut ($result.Arguments -contains 'gemma4') 'Optional UI keeps the actual selected family.'
            Assert-Shortcut (($configure -and $result.Arguments -contains '--configure-only') -or (-not $configure -and $result.Arguments -contains '--no-api-start')) 'NoStart and normal launch retain their distinct contracts.'
            if ($exitCode) {
                Assert-Shortcut ($result.Failure -like '*code 7*launch.log*ui.stderr.log*desktop shortcut*') 'Launch failure gives its code, concrete logs and retry entry point.'
            } else { Assert-Shortcut (-not $result.Failure) 'Successful mocked UI preparation stays successful.' }
        }
    }
    $result = & {
        param($Body, $Root)
        function Test-Path { param($LiteralPath, $PathType) return $true }
        function Initialize-TinyStrataUiSource { param($SourceRoot, $WorkRoot) }
        function Initialize-TinyStrataUiDependencies { param($SourceRoot, $WorkRoot, $Python) }
        function Initialize-TinyStrataDesktopShortcut { param($SourceRoot, $WorkRoot) throw 'fixture Desktop is not writable' }
        function Invoke-TestUiLauncher { $global:LASTEXITCODE = 0 }
        Invoke-Expression $Body
        Initialize-TinyStrataOptionalUI -SourceRoot $Root -WorkRoot $Root -ApiPort 18080 -WarningAction SilentlyContinue
    } $probe $root
    $failureLog = Join-Path $root 'runtime\ui\shortcut-install.log'
    Assert-Shortcut ((Test-Path -LiteralPath $failureLog -PathType Leaf)) 'Shortcut failure leaves a persistent installation diagnostic.'
    Assert-Shortcut ([IO.File]::ReadAllText($failureLog) -like '*not writable*run-ui.bat*') 'Shortcut diagnostic includes the cause and a usable fallback.'
    Write-Host "Desktop shortcut fixtures: $checks checks PASS"
} finally {
    $temporaryRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (-not $root.StartsWith($temporaryRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe fixture cleanup path.' }
    if (Test-Path -LiteralPath $root) { Remove-Item -LiteralPath $root -Recurse -Force }
}
