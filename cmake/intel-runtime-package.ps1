#Requires -Version 5.1
# Loading these definitions performs no network, native execution or installation.

function Get-TinyStrataIntelRuntimeCatalog {
    param([string]$Path)
    Assert-TinyStrataIntelUnlinkedPath $Path
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf) -or (Get-Item -LiteralPath $Path).Length -gt 1048576) { throw 'Intel runtime catalog is missing or exceeds its bound.' }
    $catalog = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($catalog.schema -ne 1 -or $catalog.backend -ne 'sycl' -or $catalog.experimental -ne $true -or
        $catalog.availability -notin @('pending_release','published') -or @($catalog.supported_devices).Count -eq 0) { throw 'Invalid experimental Intel runtime catalog.' }
    foreach ($device in @($catalog.supported_devices)) {
        if ($device.vendor_id -ne 32902 -or $device.integrated -ne $true -or $device.architecture -ne 'panther_lake_xe3' -or
            $device.profile -notin @('b390-qwen36','b390-tiel-coder') -or $device.aot_target -ne 'intel_gpu_ptl_h' -or @($device.device_ids).Count -eq 0) { throw 'Intel runtime catalog contains an unvalidated device/profile/target.' }
        foreach ($id in @($device.device_ids)) { if ($id -isnot [int] -and $id -isnot [long] -or $id -lt 1 -or $id -gt 65535) { throw 'Invalid Intel catalog device ID.' } }
    }
    return $catalog
}

function Get-TinyStrataIntelRuntimeArtifact {
    param([string]$CatalogPath, [string]$Profile = 'b390-qwen36')
    $catalog = Get-TinyStrataIntelRuntimeCatalog $CatalogPath
    if ($catalog.availability -ne 'published') { throw 'The experimental Intel runtime package has not been published. Use a verified -IntelRuntimeDirectory offline stage; no SDK was installed.' }
    $model = switch($Profile) { 'b390-qwen36' {'qwen3.6-35b-a3b'} 'b390-tiel-coder' {'tiel-coder-35b-a3b'} default {throw 'Unknown experimental Intel runtime profile.'} }
    $matchingArtifacts = @($catalog.artifacts | Where-Object { $_.profile -eq $Profile })
    if ($matchingArtifacts.Count -ne 1) { throw 'No unique published Intel runtime exists for the requested profile.' }
    $artifact = $matchingArtifacts[0]
    $uri = $null
    if ($artifact.id -notmatch '^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$' -or $artifact.model -ne $model -or
        $artifact.aot_target -ne 'intel_gpu_ptl_h' -or $artifact.experimental -ne $true -or
        [string]$artifact.sha256 -notmatch '^[a-fA-F0-9]{64}$' -or [string]$artifact.manifest_sha256 -notmatch '^[a-fA-F0-9]{64}$' -or
        [string]$artifact.executable_sha256 -notmatch '^[a-fA-F0-9]{64}$' -or [long]$artifact.bytes -lt 1 -or [long]$artifact.bytes -gt 4294967296 -or
        -not [Uri]::TryCreate([string]$artifact.url,[UriKind]::Absolute,[ref]$uri) -or $uri.Scheme -ne 'https' -or
        $uri.Host -ne 'github.com' -or $uri.UserInfo -or $uri.Query -or $uri.Fragment -or
        $uri.AbsolutePath -notmatch '^/kbc20f11/TinyStrata/releases/download/[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+\.zip$') { throw 'Invalid pinned Intel runtime download descriptor.' }
    return $artifact
}

function Assert-TinyStrataIntelDevice {
    param($Report, [string]$CatalogPath, [string]$Profile = 'b390-qwen36', [string]$AotTarget)
    $catalog = Get-TinyStrataIntelRuntimeCatalog $CatalogPath
    if ($Report.schema -ne 1 -or $Report.backend -ne 'sycl' -or $Report.device_index -ne 0 -or
        $Report.vendor_id -ne 32902 -or $Report.integrated -ne $true -or $Report.architecture -ne 'panther_lake_xe3') { throw 'The selected native device is not a validated experimental Intel integrated GPU.' }
    $matching = @($catalog.supported_devices | Where-Object { $_.profile -eq $Profile -and @($_.device_ids) -contains $Report.device_id })
    if ($matching.Count -ne 1 -or ($AotTarget -and $matching[0].aot_target -ne $AotTarget)) { throw 'The selected Intel device ID/profile is not validated for this runtime target; no generic fallback was selected.' }
    return $Report
}

function Expand-TinyStrataIntelRuntimeArchive {
    param([string]$Archive, [string]$Destination)
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $root = [IO.Path]::GetFullPath($Destination).TrimEnd('\')
    Assert-TinyStrataIntelUnlinkedPath $root
    if ($root -eq [IO.Path]::GetPathRoot($root).TrimEnd('\') -or (Test-Path -LiteralPath $root)) { throw 'Intel archive destination must be a new private directory.' }
    $zip = [IO.Compression.ZipFile]::OpenRead($Archive)
    $created = New-Object 'System.Collections.Generic.List[string]'
    $parents = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    try {
        if ($zip.Entries.Count -lt 1 -or $zip.Entries.Count -gt 4096) { throw 'Intel runtime archive exceeds the entry bound.' }
        $names = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
        $entries = @(); $total = [long]0
        foreach ($entry in $zip.Entries) {
            $relative = $entry.FullName.Replace('\','/')
            $directory = $relative.EndsWith('/')
            $relative = $relative.TrimEnd('/')
            if (-not $relative -or [IO.Path]::IsPathRooted($relative) -or $relative -match '[<>:"|?*\x00-\x1f]' -or
                -not $names.Add($relative) -or ($entry.ExternalAttributes -band 0x400) -or
                (($entry.ExternalAttributes -shr 16) -band 0xf000) -eq 0xa000) { throw 'Intel archive contains an invalid/duplicate/linked entry.' }
            foreach ($part in $relative.Split('/')) {
                if (-not $part -or $part -in @('.','..') -or $part -match '[. ]$' -or $part -match '^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)') { throw 'Intel archive contains an unsafe path.' }
            }
            if ($directory) {
                if ($relative -ne 'licenses' -and -not $relative.StartsWith('licenses/',[StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected Intel archive directory.' }
                continue
            }
            if ($relative -ne '.tinystrata-intel-runtime.json' -and $relative -notmatch '^licenses/' -and ($relative.Contains('/') -or $relative -notmatch '\.(exe|dll)$')) { throw 'Unexpected Intel archive file.' }
            if ($entry.Length -le 0 -or $entry.Length -gt 2147483648) { throw 'Intel archive file exceeds its size bound.' }
            $total += $entry.Length
            if ($total -gt 8589934592) { throw 'Intel archive exceeds its uncompressed size bound.' }
            $target = [IO.Path]::GetFullPath((Join-Path $root $relative))
            if (-not $target.StartsWith($root+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Intel archive path escaped its destination.' }
            $entries += [pscustomobject]@{ Entry=$entry; Target=$target }
        }
        [IO.Directory]::CreateDirectory($root) | Out-Null
        foreach ($item in $entries) {
            Assert-TinyStrataIntelUnlinkedPath $item.Target
            $parent = Split-Path -Parent $item.Target
            [IO.Directory]::CreateDirectory($parent) | Out-Null
            while ($parent -and $parent -ne $root) { [void]$parents.Add($parent); $parent=Split-Path -Parent $parent }
            $sourceStream = $targetStream = $null
            try {
                $sourceStream=$item.Entry.Open()
                $targetStream=[IO.File]::Open($item.Target,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
                $created.Add($item.Target); $sourceStream.CopyTo($targetStream)
            } finally { if($targetStream){$targetStream.Dispose()}; if($sourceStream){$sourceStream.Dispose()} }
        }
        return $root
    } catch {
        foreach($path in $created){Assert-TinyStrataIntelUnlinkedPath $path;if(Test-Path -LiteralPath $path -PathType Leaf){[IO.File]::Delete($path)}}
        foreach($path in @($parents|Sort-Object Length -Descending)){
            Assert-TinyStrataIntelUnlinkedPath $path
            if((Test-Path -LiteralPath $path -PathType Container)-and -not @(Get-ChildItem -LiteralPath $path -Force).Count){[IO.Directory]::Delete($path)}
        }
        if((Test-Path -LiteralPath $root -PathType Container)-and -not @(Get-ChildItem -LiteralPath $root -Force).Count){[IO.Directory]::Delete($root)}
        throw
    } finally { $zip.Dispose() }
}

function Receive-TinyStrataIntelRuntime {
    param([string]$CatalogPath, [string]$CacheRoot, [string]$Profile = 'b390-qwen36')
    $artifact = Get-TinyStrataIntelRuntimeArtifact -CatalogPath $CatalogPath -Profile $Profile
    # Keep extraction paths within Windows PowerShell5/.NET Framework limits.
    # Cache filenames still retain/verify the full SHA; folder prefixes are not
    # trusted as identity and two profiles share the same payload cache.
    $cache = Join-Path ([IO.Path]::GetFullPath($CacheRoot)) ('intel-' + $artifact.sha256.ToLowerInvariant().Substring(0,16))
    Assert-TinyStrataIntelUnlinkedPath $cache
    $archive = Join-Path $cache ($artifact.sha256.ToLowerInvariant() + '.zip')
    $null = Get-VerifiedDownload -Uri $artifact.url -Destination $archive -Sha256 $artifact.sha256
    if ((Get-Item -LiteralPath $archive).Length -ne [long]$artifact.bytes) { throw 'Intel runtime archive size does not match its pinned descriptor.' }
    $destination = Join-Path $cache ('stage-' + [Guid]::NewGuid().ToString('N'))
    $null = Expand-TinyStrataIntelRuntimeArchive -Archive $archive -Destination $destination
    $plan = Get-TinyStrataIntelStagePlan $destination
    $exe = @($plan.Files | Where-Object {$_.RelativePath -eq 'tinystrata-intel.exe'})
    if ($plan.ManifestSha256 -ne $artifact.manifest_sha256 -or $exe.Count -ne 1 -or $exe[0].Sha256 -ne $artifact.executable_sha256) { throw 'Intel runtime archive manifest/executable does not match the pinned descriptor.' }
    return [pscustomobject]@{ StageDirectory=$destination; Artifact=$artifact }
}
