# Native Gemma 4 MoE engine. Shares GGML's quantization kernels, never its graph/model execution.
if(NOT STRATA_NATIVE_EXPERTS)
  message(FATAL_ERROR "Gemma 4 hybrid inference requires STRATA_NATIVE_EXPERTS=ON")
endif()
set(_gemma_gg "${STRATA_LLAMA_SOURCE_DIR}/ggml")
set(_gemma_cuda "${_gemma_gg}/src/ggml-cuda")
add_library(tinystrata_gemma4_mmq STATIC src/gemma4/mmq.cu src/prefill/ggml_cuda_host.cu
  "${_gemma_cuda}/quantize.cu")
foreach(_type q4_0 q4_k q6_k q8_0)
  target_sources(tinystrata_gemma4_mmq PRIVATE "${_gemma_cuda}/template-instances/mmq-instance-${_type}.cu")
endforeach()
target_include_directories(tinystrata_gemma4_mmq PRIVATE
  "${_gemma_gg}/include" "${_gemma_gg}/src" "${_gemma_cuda}" include)
target_compile_options(tinystrata_gemma4_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
if(MSVC)
  target_compile_options(tinystrata_gemma4_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
endif()
target_link_libraries(tinystrata_gemma4_mmq PUBLIC ggml-base CUDA::cudart)
add_library(tinystrata_gemma4 STATIC src/gemma4/engine.cpp src/gemma4/gemma4.cpp
  src/gemma4/cpu_experts.cpp src/gemma4/kernels.cu src/gemma4/matvec.cu)
set_source_files_properties(src/gemma4/matvec.cu PROPERTIES COMPILE_OPTIONS "--use_fast_math")
target_include_directories(tinystrata_gemma4 PRIVATE include third_party/ggml "${_gemma_gg}/include")
target_compile_options(tinystrata_gemma4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_link_libraries(tinystrata_gemma4 PRIVATE strata_artifact strata_kernels tinystrata_gemma4_mmq
  ggml-cpu ggml-base CUDA::cudart CUDA::cuda_driver CUDA::cublas Threads::Threads)
add_executable(tinystrata-gemma4 src/gemma4/main.cpp)
target_include_directories(tinystrata-gemma4 PRIVATE include)
target_link_libraries(tinystrata-gemma4 PRIVATE tinystrata_gemma4 CUDA::cudart)
install(TARGETS tinystrata-gemma4 RUNTIME DESTINATION bin)
