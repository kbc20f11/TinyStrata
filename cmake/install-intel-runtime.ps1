#Requires -Version 5.1
# Pure app-local import helpers. Loading this file never starts the installer,
# executes a binary, searches an SDK, changes PATH, or installs a driver.

function Assert-TinyStrataIntelUnlinkedPath {
    param([Parameter(Mandatory=$true)][string]$Path)
    $itemPath = [IO.Path]::GetFullPath($Path)
    while ($itemPath) {
        if (Test-Path -LiteralPath $itemPath) {
            $item = Get-Item -LiteralPath $itemPath -Force -ErrorAction Stop
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Intel runtime path is linked: $itemPath" }
        }
        $parent = Split-Path -Parent $itemPath
        if ($parent -eq $itemPath) { break }
        $itemPath = $parent
    }
}

function Get-TinyStrataIntelStagePlan {
    param([Parameter(Mandatory=$true)][string]$StageDirectory)
    $root = [IO.Path]::GetFullPath($StageDirectory).TrimEnd('\')
    Assert-TinyStrataIntelUnlinkedPath $root
    if (-not (Test-Path -LiteralPath $root -PathType Container)) { throw 'IntelRuntimeDirectory must be an existing staged runtime directory.' }
    $manifestPath = Join-Path $root '.tinystrata-intel-runtime.json'
    Assert-TinyStrataIntelUnlinkedPath $manifestPath
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw 'Intel runtime stage manifest is missing.' }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($flag in @('Experimental','ModelsIncluded','DriverIncluded','GlobalPathChanged','MklEnabled')) {
        if ($manifest.$flag -isnot [bool]) { throw 'Intel stage scope/runtime flags must be JSON booleans.' }
    }
    if ($manifest.Schema -ne 1 -or $manifest.Backend -ne 'sycl' -or $manifest.State -ne 'staged' -or
        $manifest.Experimental -ne $true -or @($manifest.ModelFamilies).Count -ne 0 -or
        $manifest.ModelsIncluded -ne $false -or $manifest.DriverIncluded -ne $false -or
        $manifest.GlobalPathChanged -ne $false) { throw 'Invalid experimental Intel stage manifest.' }
    $files = @($manifest.Files)
    if ($files.Count -lt 1 -or $files.Count -gt 4096) { throw 'Invalid Intel runtime inventory size.' }
    $names = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $components = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $licenses = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $payloads = @()
    foreach ($file in $files) {
        $relative = [string]$file.RelativePath
        $relative = $relative.Replace('\', '/')
        if (-not $relative -or [IO.Path]::IsPathRooted($relative) -or $relative -match '[<>:"|?*\x00-\x1f]' -or
            $relative -eq '.tinystrata-intel-runtime.json') { throw 'Invalid Intel stage relative path.' }
        foreach ($part in $relative.Split('/')) {
            if (-not $part -or $part -in @('.', '..') -or $part -match '[. ]$' -or
                $part -match '^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)') { throw 'Invalid Intel stage relative path component.' }
        }
        if (-not $names.Add($relative)) { throw 'Duplicate Intel stage relative path.' }
        $component = [string]$file.Component
        if ($component -notin @('tinystrata', 'compiler', 'mkl', 'tbb', 'umf', 'tcm') -or
            -not [string]$file.RedistributionEvidence) { throw 'Intel runtime license/component evidence is missing.' }
        if ($relative -match '^licenses/') { [void]$licenses.Add($component) }
        else {
            [void]$components.Add($component)
            if ($relative.Contains('/') -or $relative -notmatch '\.(exe|dll)$' -or
                [IO.Path]::GetFileName($relative) -match '^(?i:ze_loader|intelocl64|igc[^.]*|igdfcl[^.]*|ocloc[^.]*)\.dll$') {
                throw 'Intel runtime stage contains an unexpected nested binary or driver/compiler payload.'
            }
        }
        if ([string]$file.Sha256 -notmatch '^[a-fA-F0-9]{64}$' -or [long]$file.Bytes -le 0) { throw 'Intel stage hash/size is invalid.' }
        $source = [IO.Path]::GetFullPath((Join-Path $root $relative))
        if (-not $source.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Intel stage path escaped its directory.' }
        Assert-TinyStrataIntelUnlinkedPath $source
        if (-not (Test-Path -LiteralPath $source -PathType Leaf) -or (Get-Item -LiteralPath $source).Length -ne [long]$file.Bytes -or
            (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne $file.Sha256) { throw "Intel stage file hash/size mismatch: $relative" }
        $payloads += [pscustomobject]@{ Source=$source; RelativePath=$relative; Sha256=[string]$file.Sha256 }
    }
    foreach ($component in $components) {
        if (-not $licenses.Contains($component)) { throw "Intel runtime license inventory is missing for $component." }
    }
    foreach ($required in @('tinystrata-intel.exe', 'licenses/TinyStrata/LICENSE', 'licenses/TinyStrata/NOTICE',
                            'licenses/TinyStrata/third_party/ggml/LICENSE', 'licenses/TinyStrata/third_party/licenses/gufo.txt',
                            'licenses/compiler/credist.txt')) {
        if (-not $names.Contains($required)) { throw "Intel stage required payload/license is missing: $required" }
    }
    foreach ($seed in @($manifest.DynamicRuntimeSeeds)) {
        if (-not $seed -or -not $names.Contains([string]$seed) -or ([string]$seed).Contains('/') -or ([string]$seed).Contains('\')) {
            throw 'Intel stage dynamic-runtime seed is absent from its inventory.'
        }
    }
    foreach ($item in Get-ChildItem -LiteralPath $root -Recurse -Force -ErrorAction Stop) {
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Intel stage contains a linked path.' }
        if (-not $item.PSIsContainer -and $item.FullName -ne $manifestPath) {
            $relative = $item.FullName.Substring($root.Length + 1).Replace('\', '/')
            if (-not $names.Contains($relative)) { throw "Intel stage file is absent from its manifest: $relative" }
        }
    }
    return [pscustomobject]@{ Root=$root; ManifestPath=$manifestPath; Manifest=$manifest; Files=$payloads;
        ManifestSha256=(Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant() }
}

function Install-TinyStrataIntelRuntime {
    param([Parameter(Mandatory=$true)][string]$StageDirectory,
          [Parameter(Mandatory=$true)][string]$Destination)
    $plan = Get-TinyStrataIntelStagePlan $StageDirectory
    $targetRoot = [IO.Path]::GetFullPath($Destination).TrimEnd('\')
    Assert-TinyStrataIntelUnlinkedPath $targetRoot
    if ($targetRoot -eq [IO.Path]::GetPathRoot($targetRoot).TrimEnd('\') -or $targetRoot -eq $plan.Root -or
        $targetRoot.StartsWith($plan.Root + '\', [StringComparison]::OrdinalIgnoreCase) -or
        $plan.Root.StartsWith($targetRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Intel runtime destination overlaps the stage or filesystem root.' }
    if (Test-Path -LiteralPath $targetRoot) {
        if (-not (Test-Path -LiteralPath $targetRoot -PathType Container) -or @(Get-ChildItem -LiteralPath $targetRoot -Force).Count) {
            throw 'Intel runtime generation must be new or empty.'
        }
    }
    [IO.Directory]::CreateDirectory($targetRoot) | Out-Null
    $created = New-Object 'System.Collections.Generic.List[string]'
    $createdParents = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    try {
        foreach ($file in $plan.Files) {
            $target = [IO.Path]::GetFullPath((Join-Path $targetRoot $file.RelativePath))
            if (-not $target.StartsWith($targetRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Intel generation path escaped its directory.' }
            Assert-TinyStrataIntelUnlinkedPath $target
            [IO.Directory]::CreateDirectory((Split-Path -Parent $target)) | Out-Null
            $parent = Split-Path -Parent $target
            while ($parent -and $parent -ne $targetRoot) { [void]$createdParents.Add($parent); $parent = Split-Path -Parent $parent }
            $sourceStream = $targetStream = $null
            try {
                $sourceStream = [IO.File]::Open($file.Source, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
                $targetStream = [IO.File]::Open($target, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
                $created.Add($target)
                $sourceStream.CopyTo($targetStream)
            } finally { if ($targetStream) { $targetStream.Dispose() }; if ($sourceStream) { $sourceStream.Dispose() } }
            if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $file.Sha256) { throw 'Intel stage changed while copying; no generation manifest was published.' }
        }
        if ((Get-FileHash -LiteralPath $plan.ManifestPath -Algorithm SHA256).Hash -ne $plan.ManifestSha256) { throw 'Intel stage manifest changed while importing.' }
        $manifest = $plan.Manifest
        $manifest.Destination = $targetRoot
        $manifest | Add-Member -NotePropertyName ImportedFromManifestSha256 -NotePropertyValue $plan.ManifestSha256 -Force
        $manifestPath = Join-Path $targetRoot '.tinystrata-intel-runtime.json'
        $stream = [IO.File]::Open($manifestPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        try {
            $created.Add($manifestPath)
            $bytes = (New-Object Text.UTF8Encoding($false)).GetBytes(($manifest | ConvertTo-Json -Depth 12))
            $stream.Write($bytes, 0, $bytes.Length)
        } finally { $stream.Dispose() }
        return [pscustomobject]@{ Exe=(Join-Path $targetRoot 'tinystrata-intel.exe'); DllDirectories=@($targetRoot);
            Manifest=$manifestPath; FileCount=$plan.Files.Count; StageManifestSha256=$plan.ManifestSha256 }
    } catch {
        # Remove only files this invocation created. Do not recursively delete a
        # caller's directory or any unlisted/racing file.
        foreach ($path in $created) {
            Assert-TinyStrataIntelUnlinkedPath $path
            if (Test-Path -LiteralPath $path -PathType Leaf) { [IO.File]::Delete($path) }
        }
        foreach ($directory in @($createdParents | Sort-Object Length -Descending)) {
            Assert-TinyStrataIntelUnlinkedPath $directory
            if ((Test-Path -LiteralPath $directory -PathType Container) -and -not @(Get-ChildItem -LiteralPath $directory -Force).Count) { [IO.Directory]::Delete($directory) }
        }
        throw
    }
}
