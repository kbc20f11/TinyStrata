"""Regression tests for install.sh with mocked tools: no GPU, CUDA, network or real package manager.

Each test runs a copy of install.sh in a temporary "checkout" with a PATH that holds only stub commands
(cmake, ninja, nvcc, nvidia-smi, python3 with a mocked venv, an apt-get that "installs" ninja and venv
support) and symlinks to the few system utilities the script needs. The stubs record what they were
asked to do, so the tests can also check what was *not* run (dry run, preflight failures).

These checks do not replace the real CUDA build evidence; they cover the installer's preflight and
generator logic only.
"""
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "install.sh"
BASH = shutil.which("bash")

# System utilities install.sh uses; everything else on PATH is a stub, so a real cmake/ninja can never leak in.
SYSTEM_TOOLS = ("sed", "head", "grep", "tr", "sort", "paste", "id", "uname", "mktemp", "rm", "dirname", "nproc",
                "cat", "mkdir", "touch", "chmod", "printf", "env", "cut", "basename")

OS_RELEASES = {
    "debian": 'ID=ubuntu\nID_LIKE=debian\nPRETTY_NAME="Test Ubuntu"\n',
    "arch": 'ID=arch\nPRETTY_NAME="Test Arch"\n',
}

STUB_CMAKE = r"""#!/bin/sh
echo "cmake $*" >> "$FAKE_STATE/cmake.log"
case "$1" in
  --build)
    dir="$2"
    for exe in tinystrata tinystrata-lfm2 tinystrata-gemma4 tinystrata-vision; do : > "$dir/$exe"; chmod +x "$dir/$exe"; done
    exit 0 ;;
esac
# configure: -S src -B dir -G generator ...
dir=""; gen=""
while [ $# -gt 0 ]; do
  case "$1" in -B) dir="$2"; shift ;; -G) gen="$2"; shift ;; esac
  shift
done
mkdir -p "$dir"
if [ -f "$dir/CMakeCache.txt" ]; then
  old="$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "$dir/CMakeCache.txt")"
  if [ "$old" != "$gen" ]; then
    echo "CMake Error: Error: generator : $gen Does not match the generator used previously: $old" >&2
    exit 1
  fi
fi
echo "CMAKE_GENERATOR:INTERNAL=$gen" > "$dir/CMakeCache.txt"
exit 0
"""

STUB_OK = "#!/bin/sh\nexit 0\n"
STUB_NVCC = '#!/bin/sh\necho "Cuda compilation tools, release 13.4, V13.4.92"\n'
STUB_SMI = r"""#!/bin/sh
case "$*" in
  *compute_cap*name*|*name*compute_cap*) echo "Fake GPU, 8.6" ;;
  *compute_cap*) echo "8.6" ;;
esac
"""

# python3: `-m venv DIR` is mocked according to $FAKE_STATE/venv (ok | no_ensurepip | no_pip, default ok, and
# "ok" appears once the fake package manager has run). Everything else exits 0.
STUB_PYTHON = r"""#!/bin/sh
echo "python3 $*" >> "$FAKE_STATE/python.log"
if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
  mode="$(cat "$FAKE_STATE/venv" 2>/dev/null || echo ok)"
  if [ "$mode" = "no_ensurepip" ]; then
    echo "The virtual environment was not created successfully because ensurepip is not available." >&2
    exit 1
  fi
  mkdir -p "$3/bin"
  cat > "$3/bin/python" <<EOF
#!/bin/sh
if [ "\$1" = "-m" ] && [ "\$2" = "pip" ] && [ "\$3" = "--version" ] && [ "$mode" = "no_pip" ]; then exit 1; fi
exit 0
EOF
  chmod +x "$3/bin/python"
  exit 0
fi
exit 0
"""

# apt-get: records the call and makes ninja and venv support available, like the real packages would.
STUB_APT = r"""#!/bin/sh
echo "apt-get $*" >> "$FAKE_STATE/pm.log"
printf '#!/bin/sh\nexit 0\n' > "$FAKE_BIN/ninja"; chmod +x "$FAKE_BIN/ninja"
echo ok > "$FAKE_STATE/venv"
exit 0
"""

STUB_SUDO = '#!/bin/sh\nexec "$@"\n'


def _write(path, text):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@unittest.skipUnless(BASH and sys.platform.startswith("linux"), "install.sh is a Linux bash script")
class InstallScript(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tinystrata-install-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.src = self.tmp / "src"
        self.bin = self.tmp / "bin"
        self.state = self.tmp / "state"
        for d in (self.src, self.bin, self.state):
            d.mkdir()
        shutil.copy(SCRIPT, self.src / "install.sh")
        for name in ("CMakeLists.txt", "run.py", "requirements.txt"):
            (self.src / name).write_text("")
        for name in SYSTEM_TOOLS:
            found = shutil.which(name)
            if found:
                (self.bin / name).symlink_to(found)
        for name in ("g++", "git", "make"):
            _write(self.bin / name, STUB_OK)
        _write(self.bin / "cmake", STUB_CMAKE)
        _write(self.bin / "nvcc", STUB_NVCC)
        _write(self.bin / "nvidia-smi", STUB_SMI)
        _write(self.bin / "python3", STUB_PYTHON)
        _write(self.bin / "sudo", STUB_SUDO)
        _write(self.bin / "apt-get", STUB_APT)
        (self.tmp / "os-release").write_text(OS_RELEASES["debian"])
        (self.state / "venv").write_text("ok")
        self.build = self.tmp / "build"

    # --- helpers ---
    def no_ninja(self):
        (self.bin / "ninja").unlink(missing_ok=True)

    def venv(self, mode):
        (self.state / "venv").write_text(mode)

    def run_script(self, *args):
        env = {"PATH": str(self.bin), "HOME": str(self.tmp), "FAKE_STATE": str(self.state), "FAKE_BIN": str(self.bin),
               "TINYSTRATA_OS_RELEASE": str(self.tmp / "os-release"), "TMPDIR": str(self.tmp)}
        return subprocess.run([BASH, str(self.src / "install.sh"), "--build-dir", str(self.build), *args],
                              env=env, capture_output=True, text=True, timeout=120)

    def log(self, name):
        path = self.state / name
        return path.read_text() if path.exists() else ""

    def configured_generator(self):
        text = (self.build / "CMakeCache.txt").read_text()
        return text.split("CMAKE_GENERATOR:INTERNAL=")[1].splitlines()[0]

    # --- generator across installation and repeat runs ---
    def test_clean_environment_installs_ninja_first_then_builds_with_ninja_and_repeats(self):
        self.no_ninja()
        self.venv("no_ensurepip")
        first = self.run_script("--install-deps")
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertIn("apt-get install", self.log("pm.log"))
        # The generator is chosen after the installation, so the first build already uses Ninja ...
        self.assertEqual(self.configured_generator(), "Ninja")
        # ... and a second run on the same directory does not hit a generator mismatch.
        second = self.run_script("--install-deps")
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertEqual(self.configured_generator(), "Ninja")
        self.assertEqual(self.log("pm.log").count("apt-get install"), 1, "nothing left to install the second time")

    def test_without_ninja_builds_with_makefiles_and_a_later_run_keeps_that_generator(self):
        self.no_ninja()
        first = self.run_script()
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertEqual(self.configured_generator(), "Unix Makefiles")
        # Ninja appears later (installed by hand); the existing directory keeps its generator.
        _write(self.bin / "ninja", STUB_OK)
        second = self.run_script()
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertIn("kept from the existing build directory", second.stdout)
        self.assertEqual(self.configured_generator(), "Unix Makefiles")

    def test_existing_directory_with_a_different_generator_than_preferred(self):
        _write(self.bin / "ninja", STUB_OK)             # Ninja is the preferred generator for a new directory
        self.build.mkdir()
        (self.build / "CMakeCache.txt").write_text("CMAKE_GENERATOR:INTERNAL=Unix Makefiles\n")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.configured_generator(), "Unix Makefiles")
        self.assertNotIn("Does not match the generator", result.stderr)

    def test_existing_ninja_directory_without_ninja_fails_before_the_build(self):
        self.no_ninja()
        self.build.mkdir()
        (self.build / "CMakeCache.txt").write_text("CMAKE_GENERATOR:INTERNAL=Ninja\n")
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ninja is not installed", result.stderr)
        self.assertEqual(self.log("cmake.log"), "", "cmake must not run")

    # --- venv / pip readiness ---
    def test_venv_only_missing_is_reported_and_blocks_the_build_without_install_deps(self):
        _write(self.bin / "ninja", STUB_OK)
        self.venv("no_ensurepip")
        dry = self.run_script("--dry-run")
        self.assertEqual(dry.returncode, 0)
        self.assertIn("Missing: Python venv/pip support", dry.stdout)
        self.assertIn("Install with:", dry.stdout)
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("venv/pip support", result.stderr)
        self.assertEqual(self.log("cmake.log"), "", "the CUDA build must not start")

    def test_venv_only_missing_is_repaired_by_install_deps_and_rechecked(self):
        _write(self.bin / "ninja", STUB_OK)
        self.venv("no_ensurepip")
        result = self.run_script("--install-deps")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("apt-get install", self.log("pm.log"))
        self.assertIn("After installation: all required tools are present.", result.stdout)
        self.assertTrue((self.build / "tinystrata").exists())

    def test_pip_creation_failure_is_detected(self):
        _write(self.bin / "ninja", STUB_OK)
        self.venv("no_pip")                              # the venv is created but has no working pip
        result = self.run_script("--dry-run")
        self.assertIn("Missing: Python venv/pip support", result.stdout)

    def test_install_deps_that_does_not_fix_venv_stops_before_the_build(self):
        _write(self.bin / "ninja", STUB_OK)
        self.venv("no_ensurepip")
        _write(self.bin / "apt-get", "#!/bin/sh\necho \"apt-get $*\" >> \"$FAKE_STATE/pm.log\"\nexit 0\n")
        result = self.run_script("--install-deps")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("still missing after installing packages", result.stderr)
        self.assertEqual(self.log("cmake.log"), "")

    def test_no_venv_ignores_venv_support(self):
        _write(self.bin / "ninja", STUB_OK)
        self.venv("no_ensurepip")
        result = self.run_script("--no-venv")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("venv/pip", result.stdout)
        self.assertEqual(self.log("pm.log"), "")
        self.assertFalse((self.src / ".venv").exists())
        self.assertNotIn("-m venv", self.log("python.log"), "venv support must not even be probed")

    # --- dry run ---
    def test_dry_run_changes_nothing_even_with_install_deps(self):
        self.no_ninja()
        self.venv("no_ensurepip")
        result = self.run_script("--dry-run", "--install-deps")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("(dry run) would run:", result.stdout)
        self.assertIn("Dry run: nothing was changed.", result.stdout)
        self.assertEqual(self.log("pm.log"), "", "no package manager call")
        self.assertEqual(self.log("cmake.log"), "", "no cmake call")
        self.assertFalse(self.build.exists())
        self.assertFalse((self.src / ".venv").exists())
        self.assertFalse((self.bin / "ninja").exists(), "nothing was installed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
