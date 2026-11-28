# Adapted from Strata's Inkling 2 Mini SYCL backend (MIT).
# Generic SPIR-V works with the same Intel devices as the native backend.
add_library(strata_inkling_sycl STATIC ink_sycl_backend.cpp)
target_compile_options(strata_inkling_sycl PRIVATE -fsycl)
if(MSVC)
  target_compile_options(strata_inkling_sycl PRIVATE /clang:-fno-fast-math)
else()
  target_compile_options(strata_inkling_sycl PRIVATE -fno-fast-math)
endif()
target_link_options(tinystrata-inkling2 PRIVATE -fsycl)
if(TINYSTRATA_INTEL_AOT_TARGET)
  target_compile_options(strata_inkling_sycl PRIVATE "-fsycl-targets=${TINYSTRATA_INTEL_AOT_TARGET}")
  target_link_options(tinystrata-inkling2 PRIVATE "-fsycl-targets=${TINYSTRATA_INTEL_AOT_TARGET}")
endif()
if(TINYSTRATA_INTEL_MKL)
  target_compile_definitions(strata_inkling_sycl PRIVATE INK_SYCL_USE_ONEMKL=1)
  target_include_directories(strata_inkling_sycl PRIVATE "${TINYSTRATA_MKL_INCLUDE}")
  target_link_libraries(strata_inkling_sycl PRIVATE "${TINYSTRATA_MKL_BLAS}")
endif()
target_link_libraries(strata_inkling_sycl PRIVATE strata_inkling_core)
# Reuse the actual selected-device identity and Windows UMA budget probes.
# The bundled standalone build retains its existing discrete-GPU behavior.
if(TARGET tinystrata_intel)
  target_compile_definitions(strata_inkling_sycl PRIVATE INK_SY_TINYSTRATA_MEMORY=1)
  target_link_libraries(strata_inkling_sycl PRIVATE tinystrata_intel)
  add_executable(intel_inkling_sycl_policy_test
    "${PROJECT_SOURCE_DIR}/tests/intel-inkling-sycl-policy.cpp")
  target_include_directories(intel_inkling_sycl_policy_test PRIVATE "${PROJECT_SOURCE_DIR}/include")
  add_test(NAME intel_inkling_sycl_policy COMMAND intel_inkling_sycl_policy_test)
endif()
target_compile_definitions(tinystrata-inkling2 PRIVATE INKLING_ENABLE_SYCL=1)
target_link_libraries(tinystrata-inkling2 PRIVATE strata_inkling_sycl)

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.gufo"
  DESTINATION share/licenses/TinyStrata/strata-inkling/gufo)

add_executable(intel_inkling_far_attention_policy_test ink_sy_far_attention_test.cpp)
add_test(NAME intel_inkling_far_attention_policy COMMAND intel_inkling_far_attention_policy_test)
