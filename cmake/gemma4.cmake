# Native Gemma 4 MoE engine. Shares GGML's quantization kernels, never its graph/model execution.
if(NOT STRATA_NATIVE_EXPERTS)
  message(FATAL_ERROR "Gemma 4 hybrid inference requires STRATA_NATIVE_EXPERTS=ON")
endif()

set(_gemma_gg "${STRATA_LLAMA_SOURCE_DIR}/ggml")
set(_gemma_cuda "${_gemma_gg}/src/ggml-cuda")
set(_gemma_mmq_sources src/gemma4/mmq.cu src/prefill/ggml_cuda_host.cu
  "${_gemma_cuda}/quantize.cu")
set(_gemma_mmq_types q4_0 q4_k q6_k q8_0)
# gpt-oss's MXFP4 experts. OFF retains the native matvec prompt fallback on HIP.
option(STRATA_HIP_MXFP4_MMQ "Build the MXFP4 MMQ instance for gpt-oss on HIP" ON)
set(_gemma_mmq_mxfp4 OFF)
if(NOT STRATA_ENABLE_HIP OR STRATA_HIP_MXFP4_MMQ)
  list(APPEND _gemma_mmq_types mxfp4)
  set(_gemma_mmq_mxfp4 ON)
endif()
foreach(_type IN LISTS _gemma_mmq_types)
  list(APPEND _gemma_mmq_sources "${_gemma_cuda}/template-instances/mmq-instance-${_type}.cu")
endforeach()
if(STRATA_ENABLE_HIP)
  # Same pinned GGML HIP MMQ configuration as Strata/MoMoE's prompt path.
  set_source_files_properties(${_gemma_mmq_sources} PROPERTIES LANGUAGE HIP)
endif()
add_library(tinystrata_gemma4_mmq STATIC ${_gemma_mmq_sources})
if(_gemma_mmq_mxfp4)
  target_compile_definitions(tinystrata_gemma4_mmq PRIVATE STRATA_MMQ_MXFP4=1)
endif()
target_include_directories(tinystrata_gemma4_mmq PRIVATE
  "${_gemma_gg}/include" "${_gemma_gg}/src" "${_gemma_cuda}" include)
target_compile_options(tinystrata_gemma4_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
if(MSVC)
  target_compile_options(tinystrata_gemma4_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
endif()
if(STRATA_ENABLE_HIP)
  target_compile_definitions(tinystrata_gemma4_mmq PRIVATE GGML_USE_HIP=1)
  # GGML's vendors/hip.h supplies its own CUDA compatibility surface. Do not
  # force TinyStrata's intrinsics shim into those translation units as well.
  target_link_libraries(tinystrata_gemma4_mmq PUBLIC ggml-base hip::host ${_strata_gpu_blas_target})
else()
  target_link_libraries(tinystrata_gemma4_mmq PUBLIC ggml-base CUDA::cudart)
endif()
add_library(tinystrata_gemma4 STATIC src/gemma4/engine.cpp src/gemma4/gemma4.cpp src/gemma4/gptoss.cpp
  src/gemma4/cpu_experts.cpp src/gemma4/kernels.cu src/gemma4/matvec.cu src/gemma4/sampling.cu)
if(STRATA_ENABLE_CUDA)
  set_source_files_properties(src/gemma4/matvec.cu PROPERTIES COMPILE_OPTIONS "--use_fast_math")
endif()
target_include_directories(tinystrata_gemma4 PRIVATE include third_party/ggml "${_gemma_gg}/include")
target_compile_options(tinystrata_gemma4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_link_libraries(tinystrata_gemma4 PRIVATE strata_artifact strata_kernels tinystrata_gemma4_mmq
  strata_kernels_cpu ggml-cpu ggml-base ${_strata_gpu_runtime_target} ${_strata_gpu_blas_target} Threads::Threads)
if(STRATA_ENABLE_CUDA)
  target_link_libraries(tinystrata_gemma4 PRIVATE CUDA::cuda_driver)
endif()
target_link_libraries(tinystrata_gemma4 PRIVATE tinystrata_fp8_attention)
add_executable(tinystrata-gemma4 src/gemma4/main.cpp)
target_include_directories(tinystrata-gemma4 PRIVATE include)
target_link_libraries(tinystrata-gemma4 PRIVATE tinystrata_gemma4 tinystrata_conversations ${_strata_gpu_runtime_target})
install(TARGETS tinystrata-gemma4 RUNTIME DESTINATION bin)

# Opt-in numerical regression for cached Expert batching; not part of user builds.
add_executable(gemma_cached_slots_test EXCLUDE_FROM_ALL tests/gemma-cached-slots.cu)
target_include_directories(gemma_cached_slots_test PRIVATE include)
target_link_libraries(gemma_cached_slots_test PRIVATE tinystrata_gemma4 ${_strata_gpu_runtime_target})
if(STRATA_ENABLE_HIP)
  set_source_files_properties(tests/gemma-cached-slots.cu PROPERTIES LANGUAGE HIP)
else()
  target_compile_options(gemma_cached_slots_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
  if(MSVC)
    target_compile_options(gemma_cached_slots_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor;-Xcompiler=/utf-8>)
  endif()
endif()

# Opt-in GPT-OSS fusion parity; no model files required, excluded from normal builds.
add_executable(gptoss_fused_test EXCLUDE_FROM_ALL tests/gptoss-fused.cu)
target_include_directories(gptoss_fused_test PRIVATE include)
target_link_libraries(gptoss_fused_test PRIVATE tinystrata_gemma4 ${_strata_gpu_runtime_target})
if(STRATA_ENABLE_HIP)
  set_source_files_properties(tests/gptoss-fused.cu PROPERTIES LANGUAGE HIP)
elseif(MSVC)
  target_compile_options(gptoss_fused_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor;-Xcompiler=/utf-8>)
endif()

# Model-backed state/lookup-window regression; run explicitly with a GPT-OSS GGUF.
add_executable(gptoss_state_test EXCLUDE_FROM_ALL tests/gptoss-state.cpp)
target_include_directories(gptoss_state_test PRIVATE include src/gemma4)
target_link_libraries(gptoss_state_test PRIVATE tinystrata_gemma4 ${_strata_gpu_runtime_target})

# Model-free and saved-logit full-vocabulary categorical regression; no model load is performed.
add_executable(gptoss_sampling_test EXCLUDE_FROM_ALL tests/gptoss-sampling.cu)
target_include_directories(gptoss_sampling_test PRIVATE include)
target_link_libraries(gptoss_sampling_test PRIVATE tinystrata_gemma4 ${_strata_gpu_runtime_target})
if(STRATA_ENABLE_HIP)
  set_source_files_properties(tests/gptoss-sampling.cu PROPERTIES LANGUAGE HIP)
elseif(MSVC)
  target_compile_options(gptoss_sampling_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor;-Xcompiler=/utf-8>)
endif()

# Explicit model-backed regression for device sampling and lazy CPU logit access.
add_executable(gptoss_device_sampling_test EXCLUDE_FROM_ALL tests/gptoss-device-sampling.cpp)
target_include_directories(gptoss_device_sampling_test PRIVATE include src/gemma4)
target_link_libraries(gptoss_device_sampling_test PRIVATE tinystrata_gemma4 ${_strata_gpu_runtime_target})
