#Requires -Version 5.1
<#
Stages an already built experimental Intel executable and selected redistributable
runtimes. It does not run the executable, load a GPU, install a driver or edit PATH.
Use -PlanOnly to inspect the dependency/license/hash inventory without copying.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$ExecutablePath,
    [Parameter(Mandatory=$true)][string]$Destination,
    [string]$CompilerRoot = 'C:\Program Files (x86)\Intel\oneAPI\compiler\latest',
    [string]$MklRoot = 'C:\Program Files (x86)\Intel\oneAPI\mkl\latest',
    [string]$TbbRoot = 'C:\Program Files (x86)\Intel\oneAPI\tbb\latest',
    [string]$TcmRoot = 'C:\Program Files (x86)\Intel\oneAPI\tcm\latest',
    [string]$UmfRoot = 'C:\Program Files (x86)\Intel\oneAPI\umf\latest',
    [string]$VcRuntimeDirectory,
    [string]$VcRedistributionNotice,
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'

function Get-IntelPayloadLength {
    param([Parameter(Mandatory=$true)][string]$Path)
    # Some oneAPI redistributable entries are symbolic links. On Windows,
    # FileInfo.Length can describe the zero-byte link, while hashing and copying
    # follow its actual DLL payload. Record the same payload that we distribute.
    $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
    try { return $stream.Length }
    finally { $stream.Dispose() }
}

function Get-IntelPeImports {
    param([Parameter(Mandatory=$true)][string]$Path)
    # Read headers/import descriptors only; never map/load an SDK DLL or copy a
    # large oneMKL/SYCL JIT DLL into managed memory just to inspect its imports.
    $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
    $reader = New-Object IO.BinaryReader($stream)
    try {
        if ($stream.Length -lt 64 -or $reader.ReadUInt16() -ne 0x5a4d) { throw "Invalid PE image: $Path" }
        $stream.Position = 60; $pe = $reader.ReadUInt32()
        if ([uint64]$pe + 24 -gt $stream.Length) { throw "Invalid PE header: $Path" }
        $stream.Position = $pe
        if ($reader.ReadUInt32() -ne 0x4550 -or $reader.ReadUInt16() -ne 0x8664) { throw "Expected an x64 PE image: $Path" }
        $sectionCount = $reader.ReadUInt16()
        $stream.Position = $pe + 20; $optionalBytes = $reader.ReadUInt16()
        $optional = [uint64]$pe + 24
        if ($optionalBytes -lt 224 -or $optional + $optionalBytes -gt $stream.Length) { throw "Invalid PE optional header: $Path" }
        $stream.Position = $optional
        if ($reader.ReadUInt16() -ne 0x20b) { throw "Expected a PE32+ image: $Path" }
        $stream.Position = $optional + 24; $imageBase = $reader.ReadUInt64()
        $stream.Position = $optional + 108; $directories = $reader.ReadUInt32()
        $imports = 0; $delayed = 0
        if ($directories -gt 1) { $stream.Position = $optional + 120; $imports = $reader.ReadUInt32() }
        if ($directories -gt 13) { $stream.Position = $optional + 216; $delayed = $reader.ReadUInt32() }
        $sections = @()
        if ($sectionCount -gt 96) { throw "Invalid PE section count: $Path" }
        for ($i = 0; $i -lt $sectionCount; $i++) {
            $stream.Position = $optional + $optionalBytes + $i * 40 + 8
            $virtualSize = $reader.ReadUInt32(); $virtual = $reader.ReadUInt32()
            $rawSize = $reader.ReadUInt32(); $raw = $reader.ReadUInt32()
            $sections += [pscustomobject]@{ Virtual = [uint64]$virtual; Size = [uint64][Math]::Max($virtualSize, $rawSize); Raw = [uint64]$raw; RawSize = [uint64]$rawSize }
        }
        $toOffset = {
            param([uint64]$Rva)
            foreach ($section in $sections) {
                if ($Rva -ge $section.Virtual -and $Rva -lt $section.Virtual + $section.Size) {
                    $delta = $Rva - $section.Virtual
                    if ($delta -ge $section.RawSize -or $section.Raw + $delta -ge $stream.Length) { break }
                    return [long]($section.Raw + $delta)
                }
            }
            throw "Invalid import RVA in $Path"
        }
        $names = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
        foreach ($table in @(@($imports, 20, 12), @($delayed, 32, 4))) {
            if (-not $table[0]) { continue }
            $offset = & $toOffset ([uint64]$table[0])
            $terminated = $false
            for ($i = 0; $i -lt 4096; $i++) {
                $descriptor = $offset + $i * $table[1]
                if ($descriptor + $table[1] -gt $stream.Length) { throw "Truncated import table: $Path" }
                $stream.Position = $descriptor; $attributes = $reader.ReadUInt32()
                $stream.Position = $descriptor + $table[2]; $nameRva = [uint64]$reader.ReadUInt32()
                if (-not $nameRva) { $terminated = $true; break }
                if ($table[1] -eq 32 -and -not ($attributes -band 1)) {
                    if ($nameRva -lt $imageBase) { throw "Invalid delay-import VA: $Path" }
                    $nameRva -= $imageBase
                }
                $stream.Position = & $toOffset $nameRva
                $bytes = New-Object 'System.Collections.Generic.List[byte]'
                for ($n = 0; $n -lt 512; $n++) {
                    $value = $reader.ReadByte()
                    if (-not $value) { break }
                    $bytes.Add($value)
                }
                $name = [Text.Encoding]::ASCII.GetString($bytes.ToArray())
                if ($n -eq 512 -or $name -notmatch '^[a-zA-Z0-9_.-]+\.dll$') { throw "Invalid imported DLL name: $Path" }
                $null = $names.Add($name)
            }
            if (-not $terminated) { throw "Unterminated import table: $Path" }
        }
        return @($names | Sort-Object)
    } finally { $reader.Dispose() }
}

function Assert-IntelStageDestination {
    param([string]$Path, [string[]]$SourceRoots)
    $full = [IO.Path]::GetFullPath($Path).TrimEnd('\')
    if ($full -eq [IO.Path]::GetPathRoot($full).TrimEnd('\')) { throw 'A drive root cannot be a runtime destination.' }
    foreach ($root in $SourceRoots) {
        if (-not $root) { continue }
        $source = [IO.Path]::GetFullPath($root).TrimEnd('\')
        if ($full -eq $source -or $full.StartsWith($source + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'The destination must be outside the source executable/SDK directories.' }
    }
    $existing = $full
    while ($existing) {
        if (Test-Path -LiteralPath $existing) {
            $item = Get-Item -LiteralPath $existing -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Linked runtime destination paths are not accepted.' }
        }
        $parent = Split-Path -Parent $existing
        if ($parent -eq $existing) { break }
        $existing = $parent
    }
    if (Test-Path -LiteralPath $full) {
        if (-not (Test-Path -LiteralPath $full -PathType Container) -or @(Get-ChildItem -LiteralPath $full -Force).Count) { throw 'The runtime destination must be new or empty; existing files were left unchanged.' }
    }
    return $full
}

function Get-IntelRuntimePlan {
    param([string]$ExecutablePath, [string]$Destination, [string]$CompilerRoot, [string]$MklRoot, [string]$TbbRoot,
          [string]$TcmRoot = 'C:\Program Files (x86)\Intel\oneAPI\tcm\latest',
          [string]$UmfRoot = 'C:\Program Files (x86)\Intel\oneAPI\umf\latest',
          [string]$SystemDirectory = [Environment]::SystemDirectory,
          [string]$VcRuntimeDirectory, [string]$VcRedistributionNotice)
    $mkl = $null; $tbb = $null; $tcm = $null; $umf = $null
    $exe = (Get-Item -LiteralPath $ExecutablePath -ErrorAction Stop).FullName
    $compiler = (Get-Item -LiteralPath $CompilerRoot -ErrorAction Stop).FullName
    $destinationPath = Assert-IntelStageDestination $Destination @((Split-Path -Parent $exe), $compiler, $MklRoot, $TbbRoot, $TcmRoot, $UmfRoot, $VcRuntimeDirectory)
    $redist = Join-Path $compiler 'share\doc\compiler\credist.txt'
    $agreement = Join-Path $compiler 'share\doc\compiler\licensing\c\LICENSE.rtf'
    if (-not (Test-Path -LiteralPath $redist -PathType Leaf) -or -not (Test-Path -LiteralPath $agreement -PathType Leaf)) { throw 'The compiler redistribution list and license are required.' }
    $approved = New-Object 'System.Collections.Generic.Dictionary[string,object]' ([StringComparer]::OrdinalIgnoreCase)
    $vcSeeds = @()
    if ([bool]$VcRuntimeDirectory -ne [bool]$VcRedistributionNotice) { throw 'App-local VC runtime requires both its directory and redistribution notice.' }
    if ($VcRuntimeDirectory) {
        $VcRuntimeDirectory = (Get-Item -LiteralPath $VcRuntimeDirectory -ErrorAction Stop).FullName
        $VcRedistributionNotice = (Get-Item -LiteralPath $VcRedistributionNotice -ErrorAction Stop).FullName
        if (-not (Test-Path -LiteralPath $VcRuntimeDirectory -PathType Container) -or
            -not (Test-Path -LiteralPath $VcRedistributionNotice -PathType Leaf)) { throw 'Invalid VC runtime payload or notice.' }
        foreach ($name in @('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll')) {
            if (-not (Test-Path -LiteralPath (Join-Path $VcRuntimeDirectory $name) -PathType Leaf)) { throw "Missing app-local VC runtime: $name" }
        }
        # Only release x64 CRT/OpenMP files from a separately verified Microsoft
        # redistributable payload. Never search PATH, System32, or DebugCRT.
        foreach ($file in Get-ChildItem -LiteralPath $VcRuntimeDirectory -Filter '*.dll' -File) {
            if ($file.Name -notmatch '^(msvcp140(_[12]|_atomic_wait|_codecvt_ids)?|vcruntime140(_1|_threads)?|vcomp140|concrt140|vccorlib140)\.dll$') { continue }
            $approved[$file.Name] = [pscustomobject]@{ Path=$file.FullName; Component='microsoft-vc'; Evidence='licenses/microsoft-vc/redistribution.txt' }
            $vcSeeds += $file.Name
        }
    }
    foreach ($line in Get-Content -LiteralPath $redist) {
        if ($line.Trim() -match '^<installdir>/bin/([^/]+\.dll)$') {
            $name = $matches[1]; $file = Join-Path $compiler ('bin\' + $name)
            # OCLOC/IGC and the graphics device driver belong to the installed
            # driver/build tools, never this app-local inference generation.
            if ($name -match '^(igc|igdfcl|ocloc|ze_intel_gpu)') { continue }
            if (Test-Path -LiteralPath $file -PathType Leaf) { $approved[$name] = [pscustomobject]@{ Path = $file; Component = 'compiler'; Evidence = 'compiler/credist.txt' } }
        }
    }
    $nativeImports = @(Get-IntelPeImports $exe)
    $sycl = @($nativeImports | Where-Object { $_ -match '^sycl[0-9]+\.dll$' })
    if ($sycl.Count -ne 1 -or $nativeImports -notcontains 'ze_loader.dll') { throw 'Expected a native Windows Intel SYCL executable importing ze_loader.dll and one SYCL runtime.' }
    $mklBlas = @($nativeImports | Where-Object { $_ -match '^mkl_sycl_blas\.[0-9]+\.dll$' })
    if ($mklBlas.Count -gt 1) { throw 'The executable imports multiple oneMKL BLAS runtime generations.' }
    $mklEnabled = $mklBlas.Count -eq 1
    $seeds = New-Object 'System.Collections.Generic.List[string]'
    foreach ($name in $vcSeeds) { $seeds.Add($name) }
    foreach ($name in @($sycl[0], 'ur_win_proxy_loader.dll', 'ur_loader_shared.dll', 'ur_adapter_level_zero.dll', 'ur_adapter_level_zero_v2.dll', 'umf.dll', 'sycl-jit.dll')) { $seeds.Add($name) }
    $licenseRoots = @([pscustomobject]@{ Root = (Join-Path $compiler 'share\doc\compiler\licensing'); Component = 'compiler' })
    if ($mklEnabled) {
        $mkl = (Get-Item -LiteralPath $MklRoot -ErrorAction Stop).FullName
        $tbb = (Get-Item -LiteralPath $TbbRoot -ErrorAction Stop).FullName
        foreach ($pair in @(@($mkl, 'mkl'), @($tbb, 'tbb'))) {
            $license = Join-Path $pair[0] ('share\doc\' + $pair[1] + '\licensing')
            $terms = Join-Path $license $(if ($pair[1] -eq 'mkl') { 'license.rtf' } else { 'license.txt' })
            if (-not (Test-Path -LiteralPath $terms -PathType Leaf) -or (Get-Content -LiteralPath $terms -Raw) -notmatch 'Intel Simplified Software License') { throw "Missing expected binary redistribution terms: $terms" }
            $licenseRoots += [pscustomobject]@{ Root = $license; Component = $pair[1] }
        }
        foreach ($file in Get-ChildItem -LiteralPath (Join-Path $mkl 'bin') -Filter '*.dll' -File) {
            if ($file.Name -match '^mkl_[a-zA-Z0-9_]+\.[0-9]+\.dll$') {
                if ($approved.ContainsKey($file.Name)) { throw "Ambiguous SDK runtime: $($file.Name)" }
                $approved[$file.Name] = [pscustomobject]@{ Path = $file.FullName; Component = 'mkl'; Evidence = 'mkl/licensing/license.rtf' }
            }
        }
        foreach ($file in Get-ChildItem -LiteralPath (Join-Path $tbb 'bin') -Filter '*.dll' -File | Where-Object { $_.Name -notmatch '_debug\.dll$' }) {
            if ($file.Name -notmatch '^tbb[a-zA-Z0-9_]*\.dll$') { continue }
            if ($approved.ContainsKey($file.Name)) { throw "Ambiguous SDK runtime: $($file.Name)" }
            $approved[$file.Name] = [pscustomobject]@{ Path = $file.FullName; Component = 'tbb'; Evidence = 'tbb/licensing/license.txt' }
            # BLAS needs the TBB runtime; allocator companions may be loaded
            # dynamically. Optional tbbbind NUMA-affinity plugins additionally
            # require hwloc (not supplied by this SDK) and are not this runtime.
            if ($file.Name -match '^tbb([0-9]+|malloc|malloc_proxy)\.dll$') { $seeds.Add($file.Name) }
        }
        if (-not $approved.ContainsKey($mklBlas[0])) { throw 'The selected oneMKL SDK does not contain the imported BLAS runtime.' }
        $core = @(Get-IntelPeImports $approved[$mklBlas[0]].Path | Where-Object { $_ -match '^mkl_core\.([0-9]+)\.dll$' })
        if ($core.Count -ne 1) { throw 'Cannot determine the oneMKL CPU-dispatch generation.' }
        $major = [regex]::Match($core[0], '^mkl_core\.([0-9]+)\.dll$').Groups[1].Value
        # CPU ISA/VML/thread dispatch libraries are loaded dynamically by MKL.
        foreach ($name in @($approved.Keys)) {
            if ($name -match ('^mkl_(avx[0-9]+|def|mc[0-9]+|vml_[a-z0-9_]+|tbb_thread|intel_thread|sequential)\.' + $major + '\.dll$')) { $seeds.Add($name) }
        }
    }
    $chosen = New-Object 'System.Collections.Generic.Dictionary[string,object]' ([StringComparer]::OrdinalIgnoreCase)
    $external = New-Object 'System.Collections.Generic.Dictionary[string,object]' ([StringComparer]::OrdinalIgnoreCase)
    $queue = New-Object 'System.Collections.Generic.Queue[object]'
    $queue.Enqueue([pscustomobject]@{ Path = $exe; Component = 'tinystrata'; Evidence = 'selected executable' })
    foreach ($name in $seeds) {
        if (-not $approved.ContainsKey($name)) { throw "Required dynamic runtime is absent from the selected SDK/redist list: $name" }
        $queue.Enqueue($approved[$name])
    }
    while ($queue.Count) {
        $entry = $queue.Dequeue(); $name = [IO.Path]::GetFileName($entry.Path)
        if ($chosen.ContainsKey($name)) { continue }
        $chosen[$name] = $entry
        foreach ($dependency in Get-IntelPeImports $entry.Path) {
            # The 2026.2 compiler's UMF DLL imports hwloc from the separately
            # installed Threading Composability Manager component. Its own
            # redistribution terms/notices are required, not an arbitrary PATH hit.
            if ($dependency -match '^libhwloc-[0-9]+\.dll$' -and -not $approved.ContainsKey($dependency)) {
                $tcm = (Get-Item -LiteralPath $TcmRoot -ErrorAction Stop).FullName
                $tcmLicense = Join-Path $tcm 'share\doc\tcm\licensing'
                if ((Get-Content -LiteralPath (Join-Path $tcmLicense 'license.txt') -Raw) -notmatch 'Intel Simplified Software License' -or
                    (Get-Content -LiteralPath (Join-Path $tcmLicense 'third-party-programs.txt') -Raw) -notmatch 'hwloc') { throw 'TCM hwloc redistribution terms/notices are missing.' }
                $hwloc = Join-Path $tcm ('bin\' + $dependency)
                if (-not (Test-Path -LiteralPath $hwloc -PathType Leaf)) { throw "The selected TCM component does not provide $dependency" }
                $approved[$dependency] = [pscustomobject]@{ Path = $hwloc; Component = 'tcm'; Evidence = 'tcm/licensing/license.txt and third-party-programs.txt' }
                if (-not @($licenseRoots | Where-Object { $_.Component -eq 'tcm' }).Count) { $licenseRoots += [pscustomobject]@{ Root = $tcmLicense; Component = 'tcm' } }
            }
            if ($approved.ContainsKey($dependency)) { $queue.Enqueue($approved[$dependency]); continue }
            $systemFile = Join-Path $SystemDirectory $dependency
            if ($dependency -notmatch '^(api-ms-win-|ext-ms-win-)' -and -not (Test-Path -LiteralPath $systemFile -PathType Leaf)) { throw "Unresolved dependency (not an approved SDK file or system prerequisite): $dependency required by $($entry.Path)" }
            $external[$dependency] = [pscustomobject]@{ Name = $dependency; Provider = 'Windows / installed graphics driver / Microsoft VC++ runtime'; Bundled = $false }
        }
    }
    $files = @()
    foreach ($entry in $chosen.Values) {
        $info = Get-Item -LiteralPath $entry.Path
        $files += [pscustomobject]@{ Source = $info.FullName; RelativePath = $info.Name; Component = $entry.Component; RedistributionEvidence = $entry.Evidence; Bytes = (Get-IntelPayloadLength $info.FullName); FileVersion = $info.VersionInfo.FileVersion; Sha256 = (Get-FileHash -LiteralPath $info.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
    }
    $notices = @([pscustomobject]@{ Source = $redist; RelativePath = 'licenses/compiler/credist.txt'; Component = 'compiler' })
    if ($VcRuntimeDirectory) { $notices += [pscustomobject]@{ Source=$VcRedistributionNotice; RelativePath='licenses/microsoft-vc/redistribution.txt'; Component='microsoft-vc' } }
    if ($chosen.ContainsKey('umf.dll')) {
        $umf = (Get-Item -LiteralPath $UmfRoot -ErrorAction Stop).FullName
        foreach ($name in @('LICENSE.TXT', 'licensing\third-party-programs.txt')) {
            $notices += [pscustomobject]@{ Source = (Join-Path $umf ('share\doc\umf\' + $name)); RelativePath = ('licenses/umf/' + $name.Replace('\', '/')); Component = 'umf' }
        }
    }
    foreach ($licenseRoot in $licenseRoots) {
        foreach ($file in Get-ChildItem -LiteralPath $licenseRoot.Root -File -Recurse) {
            $relative = $file.FullName.Substring($licenseRoot.Root.Length).TrimStart('\').Replace('\', '/')
            $notices += [pscustomobject]@{ Source = $file.FullName; RelativePath = ('licenses/' + $licenseRoot.Component + '/' + $relative); Component = $licenseRoot.Component }
        }
    }
    $sourceRoot = Split-Path $PSScriptRoot -Parent
    foreach ($file in @('LICENSE', 'NOTICE', 'third_party/ggml/LICENSE', 'third_party/licenses/gufo.txt')) {
        $notices += [pscustomobject]@{ Source = (Join-Path $sourceRoot $file); RelativePath = ('licenses/TinyStrata/' + $file); Component = 'tinystrata' }
    }
    foreach ($entry in $notices) {
        $info = Get-Item -LiteralPath $entry.Source -ErrorAction Stop
        $files += [pscustomobject]@{ Source = $info.FullName; RelativePath = $entry.RelativePath; Component = $entry.Component; RedistributionEvidence = 'license / attribution'; Bytes = (Get-IntelPayloadLength $info.FullName); FileVersion = $null; Sha256 = (Get-FileHash -LiteralPath $info.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
    }
    return [pscustomobject][ordered]@{
        Schema = 1; Backend = 'sycl'; Experimental = $true; ModelFamilies = @(); ModelsIncluded = $false
        CreatedUtc = [DateTime]::UtcNow.ToString('o'); Destination = $destinationPath; ExecutablePath = $exe
        CompilerRoot = $compiler; MklEnabled = $mklEnabled; MklRoot = $(if ($mklEnabled) { $mkl } else { $null }); TbbRoot = $(if ($mklEnabled) { $tbb } else { $null })
        TcmRoot = $tcm; UmfRoot = $umf
        VcRuntimeDirectory = $VcRuntimeDirectory; VcRedistributionNotice = $VcRedistributionNotice
        DriverIncluded = $false; GlobalPathChanged = $false; GpuValidation = 'not performed by staging helper'
        DynamicRuntimeSeeds = @($seeds | Sort-Object -Unique); ExternalPrerequisites = @($external.Values | Sort-Object Name)
        Files = @($files | Sort-Object RelativePath); State = 'planned'
    }
}

function Write-IntelRuntimeStage {
    param([Parameter(Mandatory=$true)]$Plan)
    $root = Assert-IntelStageDestination $Plan.Destination @((Split-Path -Parent $Plan.ExecutablePath), $Plan.CompilerRoot, $Plan.MklRoot, $Plan.TbbRoot, $Plan.TcmRoot, $Plan.UmfRoot, $Plan.VcRuntimeDirectory)
    $null = [IO.Directory]::CreateDirectory($root)
    foreach ($file in $Plan.Files) {
        $target = [IO.Path]::GetFullPath((Join-Path $root $file.RelativePath))
        if (-not $target.StartsWith($root.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid staged relative path.' }
        $null = [IO.Directory]::CreateDirectory((Split-Path -Parent $target))
        # Never overwrite a file, including one added after the empty-dir check.
        [IO.File]::Copy($file.Source, $target, $false)
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.Sha256) { throw "Runtime changed while staging: $($file.Source). No completed manifest was published." }
    }
    $Plan.State = 'staged'
    $manifest = Join-Path $root '.tinystrata-intel-runtime.json'
    [IO.File]::WriteAllText($manifest, ($Plan | ConvertTo-Json -Depth 8), (New-Object Text.UTF8Encoding($false)))
    return $manifest
}

if ($MyInvocation.InvocationName -ne '.') {
    $plan = Get-IntelRuntimePlan -ExecutablePath $ExecutablePath -Destination $Destination -CompilerRoot $CompilerRoot -MklRoot $MklRoot -TbbRoot $TbbRoot -TcmRoot $TcmRoot -UmfRoot $UmfRoot -VcRuntimeDirectory $VcRuntimeDirectory -VcRedistributionNotice $VcRedistributionNotice
    if ($PlanOnly) { $plan | ConvertTo-Json -Depth 8 }
    else { Write-Output (Write-IntelRuntimeStage $plan) }
}
