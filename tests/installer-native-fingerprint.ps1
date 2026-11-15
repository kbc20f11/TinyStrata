#Requires -Version 5.1
# Real fingerprint/dependency-receipt/stamp functions, tiny inert file fixtures.
# No native process, compiler, network, GPU, source refresh or installer startup.
$ErrorActionPreference='Stop'
$repository=Split-Path -Parent $PSScriptRoot
$tokens=$null;$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $repository 'install.ps1'),[ref]$tokens,[ref]$errors)
if($errors.Count){throw 'Installer parse failed'}
foreach($name in @('Get-TinyStrataSourceFingerprint','Get-TinyStrataNativeFlagFingerprint','Get-TinyStrataNativeBuildInputs','Write-TinyStrataNativeBuildStamp','Test-TinyStrataNativeBuildStamp')){
    $node=$ast.Find({param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name},$true)
    Invoke-Expression $node.Extent.Text
}
$script:checks=0
function Assert-Fingerprint([bool]$Pass,[string]$Why){if(-not $Pass){throw "Native fingerprint fixture: $Why"};$script:checks++}
function Write-Fixture([string]$Path,[string]$Text){[IO.Directory]::CreateDirectory((Split-Path -Parent $Path))|Out-Null;[IO.File]::WriteAllText($Path,$Text,(New-Object Text.UTF8Encoding($false)))}
$root=Join-Path ([IO.Path]::GetTempPath()) ('tinystrata-fingerprint-v2-'+[Guid]::NewGuid().ToString('N'))
$originalLocation=Get-Location
$originalCxxFlags=[Environment]::GetEnvironmentVariable('CXXFLAGS','Process')
$originalLib=[Environment]::GetEnvironmentVariable('LIB','Process')
try{
    $source=Join-Path $root 'source';$build=Join-Path $root 'build';$sdk=Join-Path $root 'sdk'
    foreach($relative in @('CMakeLists.txt','src\intel\qwen_ops.cpp','src\qwen35\engine.cu','src\unrecognized\future.cpp','include\strata\intel\quant.hpp','include\strata\common\gguf.hpp','third_party\ggml\quants.h','cmake\sycl_backend.cmake','cmake\future-policy.cmake','ui\web\app.js')){Write-Fixture (Join-Path $source $relative) $relative}
    Copy-Item -LiteralPath (Join-Path $repository 'install.ps1') -Destination (Join-Path $source 'install.ps1')
    $base=@{}
    foreach($backend in @('cuda','hip')){$base[$backend]=Get-TinyStrataSourceFingerprint -Directory $source -NativeOnly -Backend $backend}
    Assert-Fingerprint ($base.cuda -ne $base.hip) 'backend is part of the versioned key'
    foreach($relative in @('src\intel\qwen_ops.cpp','include\strata\intel\quant.hpp','cmake\sycl_backend.cmake','ui\web\app.js')){
        $path=Join-Path $source $relative;$before=[IO.File]::ReadAllText($path);Write-Fixture $path ($before+' changed')
        foreach($backend in @('cuda','hip')){Assert-Fingerprint ((Get-TinyStrataSourceFingerprint -Directory $source -NativeOnly -Backend $backend) -eq $base[$backend]) "$relative does not rebuild $backend"}
        Write-Fixture $path $before
    }
    foreach($relative in @('src\qwen35\engine.cu','src\unrecognized\future.cpp','include\strata\common\gguf.hpp','third_party\ggml\quants.h','cmake\future-policy.cmake','CMakeLists.txt')){
        $path=Join-Path $source $relative;$before=[IO.File]::ReadAllText($path);Write-Fixture $path ($before+' changed')
        foreach($backend in @('cuda','hip')){Assert-Fingerprint ((Get-TinyStrataSourceFingerprint -Directory $source -NativeOnly -Backend $backend) -ne $base[$backend]) "$relative conservatively invalidates $backend"}
        Write-Fixture $path $before
    }
    $installer=Join-Path $source 'install.ps1';$original=[IO.File]::ReadAllText($installer)
    foreach($replacement in @(@("'-DSTRATA_PORTABLE=ON'","'-DSTRATA_PORTABLE=OFF'"),@("'gfx1201'","'gfx1202'"))){
        Write-Fixture $installer ($original.Replace($replacement[0],$replacement[1]))
        Assert-Fingerprint ((Get-TinyStrataSourceFingerprint -Directory $source -NativeOnly -Backend hip) -ne $base.hip) 'build flag/transitive architecture helper policy invalidates'
        Write-Fixture $installer $original
    }
    foreach($name in @('compiler.exe','gpu-compiler.exe','cmake.exe','devcmd.bat','external.hpp','external.lib','oldnames.lib','libcmt.lib','libcpmt.lib')){Write-Fixture (Join-Path $sdk $name) ('inert '+$name)}
    Write-Fixture (Join-Path $sdk 'nvvm\libdevice\libdevice.10.bc') 'inert CUDA implicit bitcode'
    [Environment]::SetEnvironmentVariable('LIB',$sdk,'Process')
    foreach($name in @('tinystrata.exe','tinystrata-lfm2.exe','tinystrata-gemma4.exe','tinystrata-vision.exe','strata_kq.dll','app-local-dependency.dll')){Write-Fixture (Join-Path $build $name) ('inert '+$name)}
    Write-Fixture (Join-Path $build 'CMakeCache.txt') ("CMAKE_CXX_COMPILER:FILEPATH=$(Join-Path $sdk 'compiler.exe')`nCMAKE_CUDA_COMPILER:FILEPATH=$(Join-Path $sdk 'gpu-compiler.exe')`nCMAKE_HIP_COMPILER:FILEPATH=$(Join-Path $sdk 'gpu-compiler.exe')`n")
    Write-Fixture (Join-Path $build 'build.ninja') 'actual fixture graph'
    Write-Fixture (Join-Path $build 'link args.rsp') ('"'+(Join-Path $sdk 'external.lib')+'"')
    Write-Fixture (Join-Path $root 'Invoke-FixtureNinja') 'inert alias target for a PowerShell function, never executable'
    $script:fixtureDeps=@('unit.obj: #deps 2, deps mtime 1 (VALID)',('    '+(Join-Path $source 'include\strata\common\gguf.hpp')),('    '+(Join-Path $sdk 'external.hpp')))
    $script:fixtureCommands=@('lib.exe /nologo /out:static-output.lib @"link args.rsp"',
                             'link.exe /out:program.exe /implib:program.lib /implib:"import output.lib" @"link args.rsp"')
    function Invoke-FixtureNinja {param($C,$t) $global:LASTEXITCODE=0;if($t -eq 'deps'){return $script:fixtureDeps};if($t -eq 'commands'){return $script:fixtureCommands};throw 'Unexpected Ninja fixture invocation'}
    Set-Location -LiteralPath $root
    $tools=[pscustomobject]@{Ninja='Invoke-FixtureNinja';CMake=(Join-Path $sdk 'cmake.exe');DevCmd=(Join-Path $sdk 'devcmd.bat')}
    $proof=Get-TinyStrataNativeBuildInputs -Directory $build -SourceRoot $source -Tools $tools -Backend cuda -ToolkitRoot $sdk -Architectures '86;89'
    Assert-Fingerprint ($proof.complete -eq $true) ('actual receipt collector resolves tiny dependency/command/response graph: '+$proof.reason)
    Assert-Fingerprint (@($proof.inputs|Where-Object {$_.path -eq (Join-Path $sdk 'external.hpp')}).Count -eq 1) 'external transitive header is bound'
    Assert-Fingerprint (@($proof.inputs|Where-Object {$_.path -eq (Join-Path $sdk 'external.lib')}).Count -eq 1) 'response-file import library is bound'
    Assert-Fingerprint (@($proof.inputs|Where-Object {$_.path -eq (Join-Path $sdk 'oldnames.lib')}).Count -eq 1) 'implicit CRT library is bound'
    Assert-Fingerprint (@($proof.inputs|Where-Object {$_.path -eq (Join-Path $sdk 'nvvm\libdevice\libdevice.10.bc')}).Count -eq 1) 'implicit GPU bitcode is bound'
    Write-TinyStrataNativeBuildStamp -Directory $build -SourceRoot $source -Backend cuda -BuildInputs $proof
    Assert-Fingerprint (Test-TinyStrataNativeBuildStamp -Directory $build -Fingerprint $base.cuda -Backend cuda -Architectures '86;89' -ToolkitRoot $sdk) 'matching v2 proof reuses generation'
    Assert-Fingerprint (Test-TinyStrataNativeBuildStamp -Directory $build -Fingerprint $base.cuda -Backend cuda -Tools $tools -RequireSelectedTools) 'selected toolchain identity matches'
    $changedTools=[pscustomobject]@{Ninja=$tools.Ninja;CMake=$tools.CMake;DevCmd=$tools.DevCmd;Version='different VS version'}
    Assert-Fingerprint (-not(Test-TinyStrataNativeBuildStamp -Directory $build -Fingerprint $base.cuda -Backend cuda -Tools $changedTools -RequireSelectedTools)) 'new selected toolchain version rejects'
    Assert-Fingerprint (-not(Test-TinyStrataNativeBuildStamp -Directory $build -Fingerprint $base.cuda -Backend cuda -RequireSelectedTools)) 'unknown selected toolchain cannot certify reuse'
    Assert-Fingerprint (-not(Test-TinyStrataNativeBuildStamp -Directory $build -Fingerprint $base.cuda -Backend cuda -Architectures '120')) 'different resolved architecture rejects'
    Assert-Fingerprint (-not(Test-TinyStrataNativeBuildStamp -Directory $build -Fingerprint $base.cuda -Backend cuda -ToolkitRoot (Join-Path $root 'different-sdk'))) 'different selected SDK rejects'
    Assert-Fingerprint (-not(Test-TinyStrataNativeBuildStamp -Directory $build -Fingerprint $base.cuda -Backend hip)) 'backend mismatch rejects'
    foreach($path in @((Join-Path $sdk 'compiler.exe'),(Join-Path $sdk 'gpu-compiler.exe'),(Join-Path $sdk 'external.hpp'),(Join-Path $sdk 'external.lib'),(Join-Path $sdk 'oldnames.lib'),(Join-Path $sdk 'nvvm\libdevice\libdevice.10.bc'),(Join-Path $build 'CMakeCache.txt'),(Join-Path $build 'build.ninja'),(Join-Path $build 'link args.rsp'),(Join-Path $build 'tinystrata.exe'),(Join-Path $build 'strata_kq.dll'),(Join-Path $build 'app-local-dependency.dll'))){
        $before=[IO.File]::ReadAllText($path);Write-Fixture $path ($before+' changed')
        Assert-Fingerprint (-not(Test-TinyStrataNativeBuildStamp -Directory $build -Fingerprint $base.cuda -Backend cuda)) 'compiler/header/library/flags/graph/response/exe/DLL byte change rejects'
        Write-Fixture $path $before
    }
    [Environment]::SetEnvironmentVariable('CXXFLAGS','private-fixture-changed','Process')
    Assert-Fingerprint (-not(Test-TinyStrataNativeBuildStamp -Directory $build -Fingerprint $base.cuda -Backend cuda)) 'changed effective flag environment rejects'
    [Environment]::SetEnvironmentVariable('CXXFLAGS',$originalCxxFlags,'Process')
    foreach($name in @('NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS','NVCC_CCBIN','CXX','CUDAHOSTCXX','HIPHOSTCXX')){
        $before=[Environment]::GetEnvironmentVariable($name,'Process')
        try{[Environment]::SetEnvironmentVariable($name,'private-fixture-input-changed','Process');Assert-Fingerprint (-not(Test-TinyStrataNativeBuildStamp -Directory $build -Fingerprint $base.cuda -Backend cuda)) "$name invalidates reuse"}
        finally{[Environment]::SetEnvironmentVariable($name,$before,'Process')}
    }
    Write-Fixture (Join-Path $build 'newly-injected.dll') 'extra library'
    Assert-Fingerprint (-not(Test-TinyStrataNativeBuildStamp -Directory $build -Fingerprint $base.cuda -Backend cuda)) 'new unrecorded app-local DLL rejects'
    Remove-Item -LiteralPath (Join-Path $build 'newly-injected.dll')
    # If a shared backend actually starts including Intel headers, the actual
    # dependency receipt still invalidates despite their normal source exclusion.
    $script:fixtureDeps+=('    '+(Join-Path $source 'include\strata\intel\quant.hpp'))
    $cross=Get-TinyStrataNativeBuildInputs -Directory $build -SourceRoot $source -Tools $tools -Backend cuda -ToolkitRoot $sdk -Architectures '86;89'
    Write-TinyStrataNativeBuildStamp -Directory $build -SourceRoot $source -Backend cuda -BuildInputs $cross
    $path=Join-Path $source 'include\strata\intel\quant.hpp';$before=[IO.File]::ReadAllText($path);Write-Fixture $path ($before+' cross dependency changed')
    Assert-Fingerprint (-not(Test-TinyStrataNativeBuildStamp -Directory $build -Fingerprint $base.cuda -Backend cuda)) 'actual cross-backend header dependency is never excluded'
    Write-Fixture $path $before
    foreach($kind in @('null','missing','empty')){
        $invalid=Get-Content -LiteralPath (Join-Path $build '.tinystrata-native-build.json') -Raw|ConvertFrom-Json
        if($kind -eq 'missing'){$invalid.build_inputs.PSObject.Properties.Remove('inputs')}
        elseif($kind -eq 'empty'){$invalid.build_inputs.inputs=@()}
        else{$invalid.build_inputs.inputs=$null}
        Write-Fixture (Join-Path $build '.tinystrata-native-build.json') ($invalid|ConvertTo-Json -Depth 8)
        Assert-Fingerprint (-not(Test-TinyStrataNativeBuildStamp -Directory $build -Fingerprint $base.cuda -Backend cuda)) "missing dependency proof ($kind) rejects"
        Write-TinyStrataNativeBuildStamp -Directory $build -SourceRoot $source -Backend cuda -BuildInputs $cross
    }
    $script:fixtureDeps=@()
    $unknown=Get-TinyStrataNativeBuildInputs -Directory $build -SourceRoot $source -Tools $tools -Backend cuda -ToolkitRoot $sdk -Architectures '86;89'
    Assert-Fingerprint ($unknown.complete -eq $false) 'unknown dependency coverage is conservative'
    Write-TinyStrataNativeBuildStamp -Directory $build -SourceRoot $source -Backend cuda -BuildInputs $unknown
    Assert-Fingerprint (-not(Test-TinyStrataNativeBuildStamp -Directory $build -Fingerprint $base.cuda -Backend cuda)) 'incomplete evidence never reuses'
    $stamp=Join-Path $build '.tinystrata-native-build.json';$legacy=Get-Content -LiteralPath $stamp -Raw|ConvertFrom-Json;$legacy.schema=1
    Write-Fixture $stamp ($legacy|ConvertTo-Json -Depth 8)
    Assert-Fingerprint (-not(Test-TinyStrataNativeBuildStamp -Directory $build -Fingerprint $base.cuda -Backend cuda)) 'old stamp cannot weaken v2 proof'
    Write-Output "Native fingerprint v2 CPU fixtures passed: $script:checks"
}finally{
    Set-Location $originalLocation
    [Environment]::SetEnvironmentVariable('CXXFLAGS',$originalCxxFlags,'Process')
    [Environment]::SetEnvironmentVariable('LIB',$originalLib,'Process')
    $resolved=[IO.Path]::GetFullPath($root)
    $tempRoot=[IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\')+'\'
    if($resolved.StartsWith($tempRoot,[StringComparison]::OrdinalIgnoreCase) -and (Split-Path -Leaf $resolved) -like 'tinystrata-fingerprint-v2-*' -and (Test-Path -LiteralPath $resolved)){Remove-Item -LiteralPath $resolved -Recurse -Force}
}
