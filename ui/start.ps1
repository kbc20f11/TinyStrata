# Unicode-safe bootstrap for run-ui.bat, shipped with the optional ui/** bundle.
# Paths stay in .NET/Win32 strings; never send them through CMD's FOR /F stdout.
$ErrorActionPreference = 'Stop'

function ConvertTo-TinyStrataWindowsArgument {
    param([AllowEmptyString()][string]$Value)
    # Quote for the native CRT/CommandLineToArgvW parser, not another shell.
    return '"' + [regex]::Replace([regex]::Replace($Value, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
}

$process = $null
$code = 1
try {
    $sourceRoot = [IO.Directory]::GetParent($PSScriptRoot).FullName
    $workRoot = Join-Path ([IO.Directory]::GetParent($sourceRoot).FullName) ((Split-Path -Leaf $sourceRoot) + '-work')
    $locationPath = Join-Path $workRoot 'runtime\ui\location.json'
    if (Test-Path -LiteralPath $locationPath -PathType Leaf) {
        try {
            $location = ConvertFrom-Json -InputObject (Get-Content -LiteralPath $locationPath -Raw -Encoding UTF8)
            if ([IO.Path]::GetFullPath($location.source_dir) -eq $sourceRoot) {
                $workRoot = [IO.Path]::GetFullPath($location.work_dir)
            }
        } catch { } # Invalid/foreign pointers retain the usual sibling work dir.
    }
    $python = Join-Path $workRoot 'runtime\.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        $python = Join-Path $sourceRoot '.venv\Scripts\python.exe'
    }
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        $available = Get-Command python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -eq $available) {
            throw 'Python was not found. Run the TinyStrata installer, or make python available on PATH.'
        }
        $python = $available.Source
    }
    $launcher = Join-Path $sourceRoot 'ui\launch.py'
    if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
        throw ('The optional UI launcher was not found: ' + $launcher + '. Update TinyStrata with the UI installer.')
    }
    $pythonArguments = @('-B', $launcher) + @($args)
    $start = New-Object Diagnostics.ProcessStartInfo
    $start.FileName = $python
    $start.Arguments = ($pythonArguments | ForEach-Object { ConvertTo-TinyStrataWindowsArgument ([string]$_) }) -join ' '
    $start.WorkingDirectory = $sourceRoot
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $start
    if (-not $process.Start()) { throw 'Could not start the TinyStrata UI launcher.' }
    # Copy raw bytes concurrently, avoiding PowerShell's legacy text decoding
    # and blocked stderr pipes. Python configures its UTF-8 diagnostics itself.
    $stdoutCopy = $process.StandardOutput.BaseStream.CopyToAsync([Console]::OpenStandardOutput())
    $stderrCopy = $process.StandardError.BaseStream.CopyToAsync([Console]::OpenStandardError())
    $process.WaitForExit()
    $null = $stdoutCopy.GetAwaiter().GetResult()
    $null = $stderrCopy.GetAwaiter().GetResult()
    $code = $process.ExitCode
} catch {
    # A UTF-8 writer changes only this diagnostic stream, not the console or
    # system code page. Keep the complete Unicode path in redirected errors.
    $writer = [IO.StreamWriter]::new([Console]::OpenStandardError(), [Text.UTF8Encoding]::new($false))
    try {
        $writer.WriteLine('TinyStrata UI: ' + $_.Exception.Message)
        $writer.Flush()
    } finally { $writer.Dispose() }
    $code = 1
} finally {
    if ($null -ne $process) { $process.Dispose() }
}
exit $code
