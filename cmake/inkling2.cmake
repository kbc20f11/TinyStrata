# Build the pinned native Inkling engine in its own directory to keep its
# quantization and backend headers isolated from the existing model engines.
add_subdirectory(third_party/strata-inkling/src/inkling2)
set_target_properties(tinystrata-inkling2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
install(FILES third_party/strata-inkling/LICENSE
  DESTINATION share/licenses/TinyStrata/strata-inkling)
install(FILES third_party/strata-inkling/third_party/zstd-dec/LICENSE
  DESTINATION share/licenses/TinyStrata/strata-inkling/zstd)
