# Adapted from Strata's Inkling 2 Mini HIP backend (MIT).
# Use the same compiler, architecture, and runtime selected by TinyStrata.
set(_ink_hip_src ink_hip_backend.hip.cpp)
set_source_files_properties(${_ink_hip_src} PROPERTIES LANGUAGE HIP)
add_library(strata_inkling_hip STATIC ${_ink_hip_src})
target_compile_definitions(strata_inkling_hip PRIVATE INKLING_ENABLE_HIP=1)
target_include_directories(strata_inkling_hip PRIVATE ..)
target_include_directories(strata_inkling_hip BEFORE PRIVATE "${STRATA_HIP_COMPAT_INCLUDE_DIR}")
target_link_libraries(strata_inkling_hip PRIVATE strata_inkling_core hip::host ${CMAKE_DL_LIBS})
target_compile_definitions(tinystrata-inkling2 PRIVATE INKLING_ENABLE_HIP=1)
target_link_libraries(tinystrata-inkling2 PRIVATE strata_inkling_hip hip::host)
