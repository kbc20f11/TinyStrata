#Requires -Version 5.1
# Installer preparation only. The caller owns the installation guard and cleanup.
# No work runs when these definitions are loaded; models are never downloaded.

function Prepare-TinyStrataIntelUiRuntimes {
    param([string]$SourceRoot, [string]$WorkRoot, [string]$Python, [string]$CatalogPath,
          [string]$CacheRoot, [string]$SelectedModel, [string]$GuardToken,
          [AllowEmptyString()][string]$AotTarget, [bool]$Offline,
          [hashtable]$RuntimeCache,
          [System.Collections.Generic.List[object]]$Configurations,
          [System.Collections.Generic.List[string]]$Generations)
    if ($Offline) {
        Write-Host 'Offline Intel stage selected: other model runtimes are left unchanged; no additional runtime download.'
        return
    }
    $catalog = Get-TinyStrataIntelRuntimeCatalog $CatalogPath
    if ($catalog.availability -ne 'published') {
        Write-Host 'Other Intel model runtimes are not published; install their dedicated runtime before selecting them.'
        return
    }
    foreach ($name in @('qwen36','tiel','gemma4','lfm2','clover')) {
        $family = Get-TinyStrataModelFamily -Name $name -Backend sycl
        if ($family.Model -eq $SelectedModel) { continue }
        $path = Join-Path (Join-Path $WorkRoot 'runtime\.tinystrata') $family.ConfigFile
        Assert-TinyStrataIntelUnlinkedPath $path
        $cfg = Get-TinyStrataConfig -Path $path
        $profile = if ($cfg -and $cfg.intel_profile -eq 'generic') { 'generic' } elseif ($AotTarget) { $family.IntelProfile } else { 'generic' }
        $target = if ($profile -eq 'generic') { '' } else { $AotTarget }
        $entries = @($catalog.artifacts | Where-Object { $_.model -eq $family.Model -and $_.profile -eq $profile -and [string]$_.aot_target -eq $target })
        if ($entries.Count -eq 0) {
            Write-Host "No published dedicated Intel runtime for $($family.Model); its configuration is left unchanged."
            continue
        }
        # Malformed/duplicate descriptors and a failed download are errors, not
        # an unavailable-model skip. Nothing has been published at this point.
        $artifact = Get-TinyStrataIntelRuntimeArtifact -CatalogPath $CatalogPath -Profile $profile -Model $family.Model -AotTarget $target
        if ($cfg -and ($cfg.backend -ne 'sycl' -or $cfg.model_name -ne $family.Model -or $cfg.arch -ne $family.Arch -or
            ($cfg.intel_profile -and $cfg.intel_profile -notin @('generic',$family.IntelProfile)))) {
            throw 'An existing Intel model configuration has a different backend or model; it was left unchanged.'
        }
        if ($cfg -and $cfg.exe) {
            $existingExe = [string]$cfg.exe
            if (-not [IO.Path]::IsPathRooted($existingExe)) { $existingExe = Join-Path $(if ($cfg.cwd) { $cfg.cwd } else { $SourceRoot }) $existingExe }
            $existingExe = [IO.Path]::GetFullPath($existingExe)
            $directory = Split-Path -Parent $existingExe
            $native = [IO.Path]::GetFullPath((Join-Path $WorkRoot 'native')).TrimEnd('\')
            if (-not [string]::Equals((Split-Path -Parent $directory),$native,[StringComparison]::OrdinalIgnoreCase) -or
                (Split-Path -Leaf $directory) -notmatch '^sycl-[a-f0-9]{32}$' -or
                (Split-Path -Leaf $existingExe) -ne 'tinystrata-intel.exe') {
                Write-Host "Existing custom Intel runtime for $($family.Model) is left unchanged; install that family explicitly to replace it."
                continue
            }
        }
        Assert-TinyStrataIntelManagedDeviceSelection -Configuration $cfg
        $key = $artifact.manifest_sha256.ToLowerInvariant() + ':' + $artifact.executable_sha256.ToLowerInvariant()
        $runtime = $RuntimeCache[$key]
        if (-not $runtime -and (Test-TinyStrataIntelRuntimeCurrent -Configuration $cfg -WorkRoot $WorkRoot -Artifact $artifact)) {
            $runtime = [pscustomobject]@{ Exe=$cfg.exe; DllDirectories=@((Split-Path -Parent $cfg.exe)) }
            $RuntimeCache[$key] = $runtime
        }
        if (-not $runtime) {
            $received = Receive-TinyStrataIntelRuntime -CatalogPath $CatalogPath -CacheRoot $CacheRoot -Profile $profile -Model $family.Model -AotTarget $target
            if ($received.Artifact.manifest_sha256 -ne $artifact.manifest_sha256 -or $received.Artifact.executable_sha256 -ne $artifact.executable_sha256) { throw 'Intel runtime descriptor changed during preparation.' }
            $generation = Join-Path (Join-Path $WorkRoot 'native') ('sycl-' + [Guid]::NewGuid().ToString('N'))
            $Generations.Add($generation)
            $imported = Install-TinyStrataIntelRuntime -StageDirectory $received.StageDirectory -Destination $generation
            $runtime = [pscustomobject]@{ Exe=$imported.Exe; DllDirectories=@($imported.DllDirectories) }
            $RuntimeCache[$key] = $runtime
        }
        $null = Assert-TinyStrataIntelCapabilities -Python $Python -SourceRoot $SourceRoot -Executable $runtime.Exe `
            -DllDirectories $runtime.DllDirectories -ExpectedSha256 $artifact.executable_sha256 -Profile $profile `
            -Model $family.Model -AllowLegacy $false -CatalogPath $CatalogPath -ExpectedAotTarget $target
        $candidate = $path + '.prepare-' + [Guid]::NewGuid().ToString('N')
        $Configurations.Add([pscustomobject]@{ Path=$path; Candidate=$candidate })
        if (Test-Path -LiteralPath $path -PathType Leaf) { [IO.File]::Copy($path, $candidate, $false) }
        $arguments = @('-B', (Join-Path $SourceRoot 'run.py'), '--model-family', $family.Name,
            '--config', $candidate, '--exe', $runtime.Exe, '--backend', 'sycl', '--intel-profile', $profile, '--configure-only')
        foreach ($directory in $runtime.DllDirectories) { $arguments += @('--lib-dir', $directory) }
        $previous = [Environment]::GetEnvironmentVariable('TINYSTRATA_INSTALLER_TRANSACTION','Process')
        try {
            [Environment]::SetEnvironmentVariable('TINYSTRATA_INSTALLER_TRANSACTION',$GuardToken,'Process')
            & $Python @arguments
            if ($LASTEXITCODE -ne 0) { throw 'Additional Intel model configuration failed before publication.' }
        } finally {
            if ($null -eq $previous) { Remove-Item -LiteralPath Env:\TINYSTRATA_INSTALLER_TRANSACTION -ErrorAction SilentlyContinue }
            else { [Environment]::SetEnvironmentVariable('TINYSTRATA_INSTALLER_TRANSACTION',$previous,'Process') }
        }
        $prepared = Get-TinyStrataConfig -Path $candidate
        if (-not $prepared -or $prepared.backend -ne 'sycl' -or $prepared.model_name -ne $family.Model -or
            $prepared.arch -ne $family.Arch -or $prepared.intel_profile -ne $profile -or $prepared.exe -ne $runtime.Exe -or
            $prepared.vision_mode -ne 'no' -or $prepared.vision -or @($prepared.lib_dirs).Count -ne 1 -or
            $prepared.lib_dirs[0] -ne (Split-Path -Parent $runtime.Exe)) { throw 'Prepared Intel UI model settings do not match their dedicated runtime.' }
        Write-Host "Prepared dedicated experimental Intel runtime for $($family.Model); its model file is downloaded only when requested."
    }
}

function Publish-TinyStrataIntelConfigurations {
    param([System.Collections.Generic.List[object]]$Configurations)
    # Snapshot only the small owned configuration/backup files, then replace.
    # A later publish failure restores earlier settings and their existing .bak.
    $records = New-Object 'System.Collections.Generic.List[object]'
    $paths = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    $restored = $true
    try {
        foreach ($entry in $Configurations) {
            $path = [IO.Path]::GetFullPath($entry.Path)
            $candidate = [IO.Path]::GetFullPath($entry.Candidate)
            Assert-TinyStrataIntelUnlinkedPath $path
            Assert-TinyStrataIntelUnlinkedPath $candidate
            if (-not $paths.Add($path) -or -not $candidate.StartsWith($path+'.prepare-',[StringComparison]::OrdinalIgnoreCase) -or
                -not (Test-Path -LiteralPath $candidate -PathType Leaf)) { throw 'Invalid or duplicate prepared Intel configuration.' }
            $rollback = $path + '.prepare-rollback-' + [Guid]::NewGuid().ToString('N')
            $record = [pscustomobject]@{Path=$path;Candidate=$candidate;Rollback=$rollback;HadOriginal=(Test-Path -LiteralPath $path -PathType Leaf);
                HadBackup=(Test-Path -LiteralPath ($path+'.bak') -PathType Leaf);Published=$false}
            $records.Add($record)
            if ($record.HadOriginal) { [IO.File]::Copy($path,$rollback,$false) }
            if ($record.HadBackup) { Assert-TinyStrataIntelUnlinkedPath ($path+'.bak');[IO.File]::Copy(($path+'.bak'),($rollback+'.bak'),$false) }
        }
        foreach ($record in $records) {
            if ($record.HadOriginal) { [IO.File]::Replace($record.Candidate,$record.Path,($record.Path+'.bak')) }
            else { [IO.File]::Move($record.Candidate,$record.Path) }
            $record.Published=$true
        }
    } catch {
        $failure=$_
        foreach ($record in @($records | Sort-Object -Property Path -Descending)) {
            if (-not $record.Published) { continue }
            try {
                if ($record.HadOriginal) { [IO.File]::Replace($record.Rollback,$record.Path,[NullString]::Value) }
                else { [IO.File]::Delete($record.Path) }
                if ($record.HadBackup) { [IO.File]::Copy(($record.Rollback+'.bak'),($record.Path+'.bak'),$true) }
                elseif (Test-Path -LiteralPath ($record.Path+'.bak') -PathType Leaf) { [IO.File]::Delete(($record.Path+'.bak')) }
            } catch { $restored=$false;Write-Warning "Intel configuration rollback needs repair from $($record.Rollback): $_" }
        }
        if (-not $restored) { $failure.Exception.Data['IntelConfigurationRollbackFailed']=$true }
        throw $failure
    } finally {
        if ($restored) {
            foreach ($record in $records) {
                foreach ($temporary in @($record.Rollback,($record.Rollback+'.bak'))) {
                    if (Test-Path -LiteralPath $temporary -PathType Leaf) { [IO.File]::Delete($temporary) }
                }
            }
        }
    }
}
