"""CPU-only CMake regression for the shipping Gemma/GPT-OSS MMQ targets.

No CUDA/HIP compiler or GPU is used. Set CMAKE_COMMAND/CMAKE_MAKE_PROGRAM
when CMake and Ninja are not on PATH; native compilation is a separate check.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CMAKE = os.environ.get('CMAKE_COMMAND') or shutil.which('cmake')
NINJA = os.environ.get('CMAKE_MAKE_PROGRAM') or shutil.which('ninja')

# Use real CMake target commands and the complete production gemma4.cmake.
# Placeholder header-only sources let CMake generate its target graph without
# enabling either GPU language or building anything.
FIXTURE = r'''
cmake_minimum_required(VERSION 3.22)
project(mxfp4_target_probe NONE)
set(CMAKE_CXX_ARCHIVE_CREATE "unused")
set(CMAKE_CXX_ARCHIVE_FINISH "")
set(CMAKE_CXX_LINK_EXECUTABLE "unused")
set(STRATA_NATIVE_EXPERTS ON)
set(STRATA_LLAMA_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/ggml-placeholder")
set(STRATA_ENABLE_CUDA OFF)
foreach(_import CUDA::cudart hip::host Threads::Threads)
  add_library(${_import} INTERFACE IMPORTED)
endforeach()
include("${TINYSTRATA_SOURCE_DIR}/cmake/gemma4.cmake")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/properties.txt"
  "global=${CMAKE_CUDA_ARCHITECTURES}\n")
foreach(_target tinystrata_gemma4_mmq tinystrata_gemma4_mmq_mxfp4)
  if(TARGET ${_target})
    foreach(_property TYPE SOURCES CUDA_ARCHITECTURES COMPILE_DEFINITIONS COMPILE_OPTIONS LINK_LIBRARIES)
      get_target_property(_value ${_target} ${_property})
      file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/properties.txt"
        "${_target}.${_property}=${_value}\n")
    endforeach()
  endif()
endforeach()
get_directory_property(_targets BUILDSYSTEM_TARGETS)
foreach(_target IN LISTS _targets)
  get_target_property(_sources ${_target} SOURCES)
  foreach(_source IN LISTS _sources)
    if(NOT _source MATCHES "^\\$<")
      set_source_files_properties("${_source}" PROPERTIES GENERATED ON HEADER_FILE_ONLY ON)
    endif()
  endforeach()
  set_target_properties(${_target} PROPERTIES LINKER_LANGUAGE CXX)
endforeach()
'''


@unittest.skipUnless(CMAKE and NINJA, 'CMake and Ninja are required (no GPU compiler needed)')
class CudaMxfp4Architectures(unittest.TestCase):
    def configure(self, architectures='86;89;120', **variables):
        with tempfile.TemporaryDirectory(prefix='tinystrata-mxfp4-cmake-') as directory:
            folder = Path(directory)
            (folder / 'CMakeLists.txt').write_text(FIXTURE, encoding='utf-8')
            command = [CMAKE, '-S', str(folder), '-B', str(folder / 'build'),
                       '-G', 'Ninja', f'-DCMAKE_MAKE_PROGRAM={NINJA}',
                       f'-DTINYSTRATA_SOURCE_DIR={ROOT.as_posix()}',
                       f'-DCMAKE_CUDA_ARCHITECTURES={architectures}']
            command.extend(f'-D{key}={value}' for key, value in variables.items())
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    encoding='utf-8', errors='replace', timeout=30)
            data = folder / 'build' / 'properties.txt'
            properties = dict(line.split('=', 1) for line in data.read_text().splitlines()) if data.exists() else {}
            return result, properties

    def test_only_mxfp4_uses_feature_target_and_is_in_existing_archive(self):
        result, props = self.configure()
        self.assertEqual(result.returncode, 0, result.stdout)
        regular = 'tinystrata_gemma4_mmq.'
        fp4 = 'tinystrata_gemma4_mmq_mxfp4.'
        self.assertEqual(props['global'], '86;89;120')
        self.assertEqual(props[regular + 'CUDA_ARCHITECTURES'], '86;89;120')
        self.assertEqual(props[fp4 + 'CUDA_ARCHITECTURES'], '86;89;120a')
        self.assertEqual(props[fp4 + 'TYPE'], 'OBJECT_LIBRARY')
        self.assertTrue(props[fp4 + 'SOURCES'].endswith('src/gemma4/mmq_mxfp4_q8.cu'))
        self.assertNotIn('mmq-instance-mxfp4.cu', props[regular + 'SOURCES'])
        self.assertNotIn('mmq_mxfp4_q8.cu', props[regular + 'SOURCES'])
        self.assertIn('$<TARGET_OBJECTS:tinystrata_gemma4_mmq_mxfp4>', props[regular + 'SOURCES'])
        for quant in ('q4_0', 'q4_k', 'q6_k', 'q8_0'):
            self.assertIn(f'mmq-instance-{quant}.cu', props[regular + 'SOURCES'])
        self.assertIn('STRATA_MMQ_MXFP4=1', props[regular + 'COMPILE_DEFINITIONS'])
        self.assertIn('--extended-lambda', props[fp4 + 'COMPILE_OPTIONS'])
        self.assertIn('ggml-base;CUDA::cudart', props[fp4 + 'LINK_LIBRARIES'])

    def test_explicit_targets_and_real_virtual_requests(self):
        cases = {
            '86;89': '86;89',
            '120': '120a',
            '120-real;121-virtual': '120a-real;121a-virtual',
            '120a;121a-real': '120a;121a-real',
            '120f;121f-real': '120a;121a-real',
            '86-virtual;89-real;120f-virtual': '86-virtual;89-real;120a-virtual',
        }
        for request, expected in cases.items():
            with self.subTest(request=request):
                result, props = self.configure(request)
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertEqual(props['tinystrata_gemma4_mmq_mxfp4.CUDA_ARCHITECTURES'], expected)
                self.assertEqual(props['tinystrata_gemma4_mmq.CUDA_ARCHITECTURES'], request)
                self.assertEqual(props['global'], request)

    def test_symbolic_targets_resolve_or_fail_with_actionable_error(self):
        for request, detected in (('native', 'CMAKE_CUDA_ARCHITECTURES_NATIVE'),
                                  ('all', 'CMAKE_CUDA_ARCHITECTURES_ALL'),
                                  ('all-major', 'CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR')):
            with self.subTest(request=request):
                result, props = self.configure(request, **{detected: '86-real;89-real;120'})
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertEqual(props['tinystrata_gemma4_mmq_mxfp4.CUDA_ARCHITECTURES'],
                                 '86-real;89-real;120a')
                self.assertEqual(props['global'], request)
                result, _ = self.configure(request, **{detected: ''})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('set CMAKE_CUDA_ARCHITECTURES to explicit', ' '.join(result.stdout.split()))

    def test_hip_keeps_existing_instances_and_optional_fallback(self):
        for enabled in ('ON', 'OFF'):
            with self.subTest(mxfp4=enabled):
                # An unresolved symbolic CUDA target must not affect HIP.
                result, props = self.configure('native', STRATA_ENABLE_HIP='ON',
                                               STRATA_HIP_MXFP4_MMQ=enabled)
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertFalse(any(key.startswith('tinystrata_gemma4_mmq_mxfp4.') for key in props))
                sources = props['tinystrata_gemma4_mmq.SOURCES']
                definitions = props['tinystrata_gemma4_mmq.COMPILE_DEFINITIONS']
                self.assertEqual('mmq-instance-mxfp4.cu' in sources, enabled == 'ON')
                self.assertEqual('STRATA_MMQ_MXFP4=1' in definitions, enabled == 'ON')
                self.assertIn('GGML_USE_HIP=1', definitions)


if __name__ == '__main__':
    unittest.main()
