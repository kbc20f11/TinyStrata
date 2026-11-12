"""CPU-only CMake regression for the shipping Gemma MMQ targets.

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
project(gemma_target_probe NONE)
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
get_directory_property(_declared_targets BUILDSYSTEM_TARGETS)
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/properties.txt" "targets=${_declared_targets}\n")
foreach(_target tinystrata_gemma4_mmq tinystrata_gemma4 tinystrata-gemma4 gemma_cached_slots_test)
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
class GemmaBuildSources(unittest.TestCase):
    def configure(self, architectures='86;89;120', **variables):
        with tempfile.TemporaryDirectory(prefix='tinystrata-gemma-cmake-') as directory:
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

    def assert_gemma_only(self, props):
        targets = props['targets'].split(';')
        self.assertEqual(set(targets), {'tinystrata_gemma4_mmq', 'tinystrata_gemma4',
                                        'tinystrata-gemma4', 'gemma_cached_slots_test'})
        self.assertNotIn('mxfp4', '\n'.join(props.values()).lower())
        self.assertNotIn('gptoss', '\n'.join(props.values()).lower())
        regular = props['tinystrata_gemma4_mmq.SOURCES'].split(';')
        instances = sorted(Path(x).name for x in regular if 'mmq-instance-' in x)
        self.assertEqual(instances, [f'mmq-instance-{q}.cu' for q in ('q4_0','q4_k','q6_k','q8_0')])
        self.assertEqual(props['tinystrata_gemma4.SOURCES'].split(';'),
            ['src/gemma4/engine.cpp','src/gemma4/gemma4.cpp','src/gemma4/cpu_experts.cpp',
             'src/gemma4/kernels.cu','src/gemma4/matvec.cu'])
        self.assertIn('strata_native_paths', props['tinystrata-gemma4.LINK_LIBRARIES'])

    def test_cuda_architectures_remain_unmodified(self):
        for request in ('86;89;120', '120', '120-real;121-virtual', 'native', 'all-major'):
            with self.subTest(request=request):
                result, props = self.configure(request)
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assert_gemma_only(props)
                self.assertEqual(props['global'], request)
                self.assertEqual(props['tinystrata_gemma4_mmq.CUDA_ARCHITECTURES'], request)

    def test_hip_uses_only_supported_quant_instances(self):
        result, props = self.configure('native', STRATA_ENABLE_HIP='ON')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assert_gemma_only(props)
        self.assertIn('GGML_USE_HIP=1', props['tinystrata_gemma4_mmq.COMPILE_DEFINITIONS'])

    def test_native_engines_do_not_compile_mxfp4_specializations(self):
        for family in ('gemma4', 'lfm2'):
            source = (ROOT / 'src' / family / 'matvec.cu').read_text(encoding='utf-8')
            self.assertNotIn('MXFP4', source)
            self.assertNotIn('case 39:', source)
        engine = (ROOT / 'src/gemma4/engine.cpp').read_text(encoding='utf-8')
        self.assertNotIn('GptOss', engine)
        self.assertNotIn('oai_', engine)
        self.assertIn('const int requested = forced ? forced : kSlots;', engine)
        self.assertIn('image_text_prefill_eligible', engine)
        self.assertIn('strata::paths::native_path(opt_.profile)', engine)


if __name__ == '__main__':
    unittest.main()
