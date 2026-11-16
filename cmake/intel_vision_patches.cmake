# Apply reviewed changes only to build-owned copies of pinned source files.
# A caller's STRATA_GGML_DIR checkout is never modified.
function(tinystrata_intel_vision_write_lf destination contents)
  # Insert the source as one substitution so literal @name@ text in upstream
  # C++ cannot itself be interpreted as a configure-file placeholder.
  set(_intel_vision_literal "${contents}")
  set(_template "${CMAKE_BINARY_DIR}/intel-vision-patched/literal.in")
  file(WRITE "${_template}" "@_intel_vision_literal@")
  configure_file("${_template}" "${destination}" @ONLY NEWLINE_STYLE LF)
endfunction()

function(tinystrata_intel_vision_patch_copy root relative patch base_sha patched_sha patch_sha)
  # Check canonical text and patch bytes, independent of Git's Windows EOL
  # policy. Only the build-owned copies are normalized to LF.
  file(READ "${root}/${relative}" _base_contents)
  string(REPLACE "\r\n" "\n" _base_contents "${_base_contents}")
  string(SHA256 _base_actual "${_base_contents}")
  file(READ "${patch}" _patch_contents)
  string(REPLACE "\r\n" "\n" _patch_contents "${_patch_contents}")
  string(SHA256 _patch_actual "${_patch_contents}")
  if(NOT _base_actual STREQUAL base_sha OR NOT _patch_actual STREQUAL patch_sha)
    message(FATAL_ERROR "Intel vision patch does not match its pinned source or reviewed patch hash: ${relative}")
  endif()
  set(_overlay "${CMAKE_BINARY_DIR}/intel-vision-patched")
  set(_destination "${_overlay}/${relative}")
  if(EXISTS "${_destination}")
    file(SHA256 "${_destination}" _existing)
  endif()
  if(NOT _existing STREQUAL patched_sha)
    get_filename_component(_directory "${_destination}" DIRECTORY)
    file(MAKE_DIRECTORY "${_directory}")
    tinystrata_intel_vision_write_lf("${_destination}" "${_base_contents}")
    set(_apply_patch "${_overlay}/reviewed-source.patch")
    tinystrata_intel_vision_write_lf("${_apply_patch}" "${_patch_contents}")
    find_package(Git REQUIRED)
    execute_process(COMMAND "${GIT_EXECUTABLE}" -c core.autocrlf=false -c core.eol=lf apply --check "${_apply_patch}"
      WORKING_DIRECTORY "${_overlay}" RESULT_VARIABLE _check ERROR_VARIABLE _error)
    if(NOT _check EQUAL 0)
      message(FATAL_ERROR "Intel vision source patch check failed: ${_error}")
    endif()
    execute_process(COMMAND "${GIT_EXECUTABLE}" -c core.autocrlf=false -c core.eol=lf apply "${_apply_patch}"
      WORKING_DIRECTORY "${_overlay}" RESULT_VARIABLE _apply ERROR_VARIABLE _error)
    if(NOT _apply EQUAL 0)
      message(FATAL_ERROR "Intel vision source patch failed: ${_error}")
    endif()
    file(READ "${_destination}" _patched_contents)
    string(REPLACE "\r\n" "\n" _patched_contents "${_patched_contents}")
    tinystrata_intel_vision_write_lf("${_destination}" "${_patched_contents}")
  endif()
  file(SHA256 "${_destination}" _patched_actual)
  if(NOT _patched_actual STREQUAL patched_sha)
    message(FATAL_ERROR "Intel vision patched source hash differs: ${relative}")
  endif()
  set_property(GLOBAL APPEND PROPERTY TINYSTRATA_INTEL_VISION_PATCH_HASHES "${_patch_actual}")
endfunction()

function(tinystrata_intel_vision_bounded_attention root)
  set(_relative "ggml/src/ggml-sycl")
  set(_overlay "${CMAKE_BINARY_DIR}/intel-vision-patched/${_relative}")
  tinystrata_intel_vision_patch_copy("${root}" "${_relative}/fattn-common.hpp"
    "${PROJECT_SOURCE_DIR}/cmake/patches/intel-vision-bounded-attention.patch"
    8c0808c976b4c4bc4a11cdd1162deb4ab8d2efd4d51203b287cf52500f88be44
    3f564699bd803113bb71da9be7c2befb9fce48bcc1cbf30ea81282c09e0a4613
    3ea2d06b7246588d392064c98eff6884b394588604b2b27ebfcba261f422e047)
  configure_file("${root}/${_relative}/fattn-tile.hpp" "${_overlay}/fattn-tile.hpp" COPYONLY)
  file(MAKE_DIRECTORY "${_overlay}/template-instances")
  # All three supported projectors use 1152 channels / 16 heads = 72.
  # Leave the remaining upstream head-size implementations untouched.
  set(_instances "${root}/${_relative}/template-instances/fattn-tile-instance-dkq72-dv72.cpp")
  get_target_property(_sources ggml-sycl SOURCES)
  foreach(_source IN LISTS _instances)
    get_filename_component(_name "${_source}" NAME)
    set(_copy "${_overlay}/template-instances/${_name}")
    tinystrata_intel_vision_patch_copy("${root}" "${_relative}/template-instances/${_name}"
      "${PROJECT_SOURCE_DIR}/cmake/patches/intel-vision-attention-fp32.patch"
      144a621a54f6df72d7d73a6f0a9ca0c58f44c23c500862a8536ef3d08282225f
      b123fa8f66e49d4cb0fbd3fe257a3c9ec7f5121ae206f5e077ad10329c173ccc
      b02362847a2a73e0f74258c3a2ce00be5b5a482cacbca37ed22a712d60e7fed5)
    set_property(SOURCE "${_copy}" TARGET_DIRECTORY ggml-sycl
      APPEND PROPERTY COMPILE_DEFINITIONS TINYSTRATA_INTEL_VISION_BOUNDED_FATTN=1)
    list(REMOVE_ITEM _sources "${_source}")
    list(APPEND _sources "${_copy}")
  endforeach()
  set_property(TARGET ggml-sycl PROPERTY SOURCES "${_sources}")
endfunction()
