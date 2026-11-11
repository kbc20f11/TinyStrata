"""Check actual CMake compile commands; never start a compiler, model or GPU.

Set TINYSTRATA_MSVC_BUILD_DIR to a freshly configured Windows CUDA build with
CMAKE_EXPORT_COMPILE_COMMANDS=ON. The small negative-profile fixtures use forced
compiler metadata solely to evaluate CMake's language/compiler expressions.
Their commands are generated, not executed.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("TINYSTRATA_MSVC_BUILD_DIR", ""))


def cmake_tools():
    cmake = shutil.which("cmake")
    if not cmake and os.name == "nt":
        candidates = sorted(Path(os.environ.get("ProgramFiles", "C:/Program Files")).glob(
            "Microsoft Visual Studio/*/*/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"))
        cmake = str(candidates[-1]) if candidates else None
    ninja = None
    if cmake:
        candidate = Path(cmake).parents[2] / "Ninja/ninja.exe"
        ninja = str(candidate) if candidate.is_file() else None
    if not ninja:
        candidate = shutil.which("ninja.exe" if os.name == "nt" else "ninja")
        ninja = candidate
    return cmake, ninja


@unittest.skipUnless((BUILD / "compile_commands.json").is_file(),
                     "Set TINYSTRATA_MSVC_BUILD_DIR to the fresh Windows CUDA build")
class ShippingMsvcCompileCommands(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.commands = json.loads((BUILD / "compile_commands.json").read_text(encoding="utf-8"))

    def selected(self, suffixes, external=False):
        rows = [row for row in self.commands if Path(row["file"]).suffix.lower() in suffixes
                and (not external or "/ggml/" in row["file"].replace("\\", "/"))]
        self.assertTrue(rows, "No matching generated compile commands: " + str((suffixes, external)))
        return rows

    def test_every_c_translation_unit_uses_utf8(self):
        for row in self.selected({".c"}):
            self.assertIn("/utf-8", row["command"], row["file"])

    def test_every_cpp_translation_unit_uses_utf8(self):
        for row in self.selected({".cc", ".cpp", ".cxx"}):
            self.assertIn("/utf-8", row["command"], row["file"])

    def test_every_nvcc_translation_unit_forwards_host_utf8(self):
        for row in self.selected({".cu"}):
            self.assertIn("-Xcompiler=/utf-8", row["command"], row["file"])

    def test_external_ggml_and_vision_inherit_charset(self):
        for suffixes in ({".c"}, {".cpp"}, {".cu"}):
            for row in self.selected(suffixes, external=True):
                expected = "-Xcompiler=/utf-8" if suffixes == {".cu"} else "/utf-8"
                self.assertIn(expected, row["command"], row["file"])


@unittest.skipUnless(os.name == "nt" and all(cmake_tools()), "Windows CMake and Ninja required")
class CharsetCompilerProfiles(unittest.TestCase):
    def generate(self, compiler_id, *, windows=True, hip_as_cxx=False):
        cmake, ninja = cmake_tools()
        with tempfile.TemporaryDirectory(prefix="tinystrata-charset-config-") as folder:
            source = Path(folder)
            # Force identities and bypass compiler checks. Python is a harmless
            # existing path standing in for a compiler; no build is requested.
            prefix = "cmake_minimum_required(VERSION 3.24)\n"
            for language, standard in (("C", 11), ("CXX", 20)):
                version = "19.40" if compiler_id == "MSVC" else "18.0"
                prefix += f'''set(CMAKE_{language}_COMPILER "{Path(sys.executable).as_posix()}")
set(CMAKE_{language}_COMPILER_ID_RUN TRUE)
set(CMAKE_{language}_COMPILER_ID {compiler_id})
set(CMAKE_{language}_COMPILER_VERSION {version})
set(CMAKE_{language}_COMPILER_FORCED TRUE)
set(CMAKE_{language}_COMPILER_WORKS TRUE)
set(CMAKE_{language}_STANDARD_COMPUTED_DEFAULT {standard})
set(CMAKE_{language}_EXTENSIONS_COMPUTED_DEFAULT OFF)
'''
            prefix += "project(CharsetProbe LANGUAGES C CXX)\n"
            if not windows:
                prefix += "set(WIN32 FALSE)\n"
            shipping = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
            charset = shipping.split("project(TinyStrata VERSION 0.1.0 LANGUAGES C CXX)", 1)[1]
            charset = charset.split("set(CMAKE_CXX_STANDARD 20)", 1)[0]
            for name in ("local.c", "local.cpp"):
                (source / name).write_text("int charset_probe;\n", encoding="utf-8")
            extra = ""
            if hip_as_cxx:
                (source / "hip.cu").write_text("int hip_probe;\n", encoding="utf-8")
                extra = "set_source_files_properties(hip.cu PROPERTIES LANGUAGE CXX)\n"
            (source / "CMakeLists.txt").write_text(
                prefix + charset + extra + "add_library(local OBJECT local.c local.cpp"
                + (" hip.cu" if hip_as_cxx else "") + ")\n", encoding="utf-8")
            result = subprocess.run([cmake, "-S", str(source), "-B", str(source / "build"),
                                     "-G", "Ninja", "-DCMAKE_MAKE_PROGRAM=" + ninja,
                                     "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return json.loads((source / "build/compile_commands.json").read_text(encoding="utf-8"))

    def test_msvc_c_and_cpp_positive_profile(self):
        for row in self.generate("MSVC"):
            self.assertIn("/utf-8", row["command"])
            self.assertNotIn("-Xcompiler=", row["command"])

    def test_windows_hip_clang_cxx_does_not_receive_msvc_flags(self):
        rows = self.generate("Clang", hip_as_cxx=True)
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertNotIn("/utf-8", row["command"])
            self.assertNotIn("-Xcompiler=", row["command"])

    def test_non_windows_profile_does_not_receive_msvc_flags(self):
        for row in self.generate("MSVC", windows=False):
            self.assertNotIn("/utf-8", row["command"])


if __name__ == "__main__":
    unittest.main()
