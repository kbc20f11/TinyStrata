# GGML's MXFP4 MMQ uses Blackwell's architecture-specific block-scaled MMA.
# Keep this target selection local to that instance: ordinary MMQ kernels do
# not need to lose their generic CUDA architecture's forward compatibility.
function(strata_cuda_mxfp4_architectures output)
  set(_resolved)
  foreach(_arch IN LISTS ARGN)
    if(_arch STREQUAL "native")
      set(_detected "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
    elseif(_arch STREQUAL "all")
      set(_detected "${CMAKE_CUDA_ARCHITECTURES_ALL}")
    elseif(_arch STREQUAL "all-major")
      set(_detected "${CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR}")
    else()
      set(_detected "${_arch}")
    endif()
    if(NOT _detected MATCHES "^[0-9]+(a|f)?(-real|-virtual)?(;[0-9]+(a|f)?(-real|-virtual)?)*$")
      message(FATAL_ERROR "Cannot resolve CUDA architecture '${_arch}' for MXFP4 MMQ; set CMAKE_CUDA_ARCHITECTURES to explicit architectures")
    endif()
    list(APPEND _resolved ${_detected})
  endforeach()
  # Preserve Ampere/Ada fallbacks, explicit 'a' targets and real/virtual
  # requests. Like pinned GGML, map 12x -> 12xa. MXF4's PTX target exception
  # also requires 'a' rather than the family-specific 12xf target.
  list(TRANSFORM _resolved REPLACE "^(12[0-9])f?$" "\\1a")
  list(TRANSFORM _resolved REPLACE "^(12[0-9])f?-(real|virtual)$" "\\1a-\\2")
  set(${output} "${_resolved}" PARENT_SCOPE)
endfunction()
