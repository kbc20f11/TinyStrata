# Staging the experimental Intel native runtime

`stage-intel-runtime.ps1` packages an already built x64 Intel executable into an explicitly supplied **new or empty directory**. It does not build or execute the engine, change global PATH, install a graphics driver, copy a model, or modify model profiles, keys or history. This is a native-runtime staging tool, not the normal TinyStrata/UI installer.

Inspect the plan before staging:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\cmake\stage-intel-runtime.ps1 `
  -ExecutablePath ..\tinystrata-work\build-intel\tinystrata-intel.exe `
  -Destination ..\tinystrata-work\runtime-intel-new -PlanOnly
```

Omit `-PlanOnly` to copy the selected files and publish `.tinystrata-intel-runtime.json`. The manifest records every file's SHA-256, size, file version, component and license evidence, plus external prerequisites and the experimental status. A completed manifest is written only after copied hashes match. A failed copy may leave a partial destination; the tool refuses to overwrite it. Choose a new destination or inspect the partial directory yourself.

The default component roots are the locally installed oneAPI `compiler`, `mkl`, `tbb`, `tcm` and `umf` `latest` directories. Each can be selected explicitly with `-CompilerRoot`, `-MklRoot`, `-TbbRoot`, `-TcmRoot` and `-UmfRoot`. The helper never searches arbitrary PATH directories to resolve a missing SDK DLL.

The selection includes normal and delay-loaded PE imports, plus components not discoverable through PE imports alone:

- The matching SYCL runtime, Unified Runtime proxy/loader, both Level Zero adapters, UMF and the separate SYCL JIT library. Compiler files must be named in the selected SDK's `share/doc/compiler/credist.txt`.
- If the executable imports oneMKL SYCL BLAS, its matching oneMKL core/thread/ISA/VML dispatch generation and oneTBB core/allocator runtime. Optional TBB NUMA-affinity plugins are not included.
- UMF's `libhwloc-*.dll` dependency from the selected Threading Composability Manager component, with that component's redistribution terms and hwloc notices. The compiler's UMF DLL does not imply that its complete dependency chain lives in the compiler's `bin` directory.
- Compiler EULA/redist list/third-party notices, oneMKL and oneTBB binary license/notices, UMF Apache-with-LLVM-exceptions license/notices, TCM/hwloc license/notices, and TinyStrata/GGML/Gufo attribution.

oneMKL and the installed oneTBB/TCM binaries carry the Intel Simplified Software License; the helper retains their installed notices. It does not infer redistribution rights for every file in an SDK `bin` directory. OCLOC, IGC and graphics-driver binaries are excluded.

The target machine must already have a compatible Intel graphics driver, its system `ze_loader.dll`, Windows system libraries and the Microsoft Visual C++ x64 runtime. These remain external prerequisites; this helper never copies files out of System32 or DriverStore. It also does not package Python, HTTP/UI dependencies or model data.

Staging is not a GPU compatibility or answer-quality test. First check `--capabilities` in a child environment whose PATH does not include the SDK, then run the selected-device probe, numerical checks and actual API/model tests separately. Keep `model_families=[]` / experimental status until the relevant acceptance gates pass.

Primary references:

- [Intel compiler redistributable library guidance](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2026-0/redistributable-library-considerations.html)
- [Intel SYCL redistribution guidance](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2025-0/redistribute-your-sycl-application.html) (generic guidance; the installed 2026.2 redist list determines actual Windows filenames)
- [oneMKL DLL usage](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2026-2/using-dlls.html)
- [oneMKL Windows binary directory](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2026-0/contents-of-the-bin-directory.html)
