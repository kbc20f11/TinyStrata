#!/usr/bin/env bash
# TinyStrata source build for Linux (NVIDIA CUDA).
#
# Detects the distribution, the NVIDIA GPU and the CUDA toolchain, reports anything that is missing,
# and builds the engine for the detected GPU architecture. It installs nothing system-wide unless
# --install-deps is given, and never installs CUDA itself.
#
# Only the NVIDIA CUDA backend is handled. AMD (HIP) and Intel (SYCL) builds are not covered here.
#
# Usage: ./install.sh [options]
#   --dry-run            Detect and print the plan; change nothing (also: --check)
#   --install-deps       Install missing build packages with the distribution's package manager (uses sudo
#                        unless run as root), then check again before building
#   --arch LIST          CUDA architectures, e.g. 75 or "86;89". Overrides GPU detection.
#   --cuda-root DIR      CUDA toolkit root (default: $CUDA_HOME, $CUDA_PATH, nvcc on PATH, /usr/local/cuda, /opt/cuda)
#   --build-dir DIR      Build directory (default: ./build). An existing directory keeps its CMake generator.
#   --portable           Pass -DSTRATA_PORTABLE=ON (AVX2 floor, for use on other PCs)
#   --jobs N             Parallel build jobs (default: number of CPUs)
#   --no-venv            Do not create the Python virtual environment (and do not require venv/pip support)
#   -h, --help           Show this help
#
# Environment (for tests): TINYSTRATA_OS_RELEASE overrides /etc/os-release.
set -euo pipefail

DRY_RUN=0 INSTALL_DEPS=0 ARCH="" CUDA_ROOT="" BUILD_DIR="" PORTABLE=0 JOBS="" NO_VENV=0

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }
say()  { printf '%s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run|--check) DRY_RUN=1 ;;
    --install-deps) INSTALL_DEPS=1 ;;
    --arch) [ $# -ge 2 ] || die "--arch needs a value"; ARCH="$2"; shift ;;
    --cuda-root) [ $# -ge 2 ] || die "--cuda-root needs a value"; CUDA_ROOT="$2"; shift ;;
    --build-dir) [ $# -ge 2 ] || die "--build-dir needs a value"; BUILD_DIR="$2"; shift ;;
    --portable) PORTABLE=1 ;;
    --jobs) [ $# -ge 2 ] || die "--jobs needs a value"; JOBS="$2"; shift ;;
    --no-venv) NO_VENV=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
  shift
done

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SRC/CMakeLists.txt" ] && [ -f "$SRC/run.py" ] || die "run this script from a TinyStrata checkout"
BUILD_DIR="${BUILD_DIR:-$SRC/build}"
[ "$(uname -s)" = "Linux" ] || die "this script is for Linux (use install.ps1 on Windows)"

# --- distribution and package manager -----------------------------------------------------------
ID="" ID_LIKE="" PRETTY_NAME=""
OS_RELEASE="${TINYSTRATA_OS_RELEASE:-/etc/os-release}"
if [ -r "$OS_RELEASE" ]; then
  # shellcheck disable=SC1090
  . "$OS_RELEASE"
fi
PM="" PKGS=() INSTALL_CMD=""
SUDO="sudo "; [ "$(id -u)" -eq 0 ] && SUDO=""
case " ${ID:-} ${ID_LIKE:-} " in
  *" arch "*)
    PM=pacman; PKGS=(base-devel cmake ninja git python)
    INSTALL_CMD="${SUDO}pacman -S --needed ${PKGS[*]}" ;;
  *" debian "*|*" ubuntu "*)
    PM=apt; PKGS=(build-essential cmake ninja-build git python3 python3-venv)
    INSTALL_CMD="${SUDO}apt-get install -y ${PKGS[*]}" ;;
  *" fedora "*|*" rhel "*)
    PM=dnf; PKGS=(gcc-c++ make cmake ninja-build git python3)
    INSTALL_CMD="${SUDO}dnf install -y ${PKGS[*]}" ;;
esac
say "Distribution: ${PRETTY_NAME:-unknown} (package manager: ${PM:-unknown})"

# --- build tools and the Python environment ---------------------------------------------------------
# Sets `missing` (required) and `optional`. It runs once up front and again after --install-deps, so the
# result that decides the build is always the one measured after any installation.
missing=() optional=()

# True when a virtual environment with pip can really be created here. `import venv` is not enough: on
# Debian/Ubuntu the module imports without python3-venv, and creation then fails for lack of ensurepip.
venv_ready() {
  local tmp
  tmp="$(mktemp -d)" || return 1
  if python3 -m venv "$tmp/venv" >/dev/null 2>&1 && "$tmp/venv/bin/python" -m pip --version >/dev/null 2>&1; then
    rm -rf "$tmp"; return 0
  fi
  rm -rf "$tmp"; return 1
}

detect_tools() {
  missing=() optional=()
  local tool
  for tool in cmake git python3; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
  done
  if ! command -v g++ >/dev/null 2>&1 && ! command -v c++ >/dev/null 2>&1; then missing+=("a C++ compiler (g++)"); fi
  command -v ninja >/dev/null 2>&1 || optional+=("ninja (faster builds)")
  if [ "$NO_VENV" -eq 0 ] && command -v python3 >/dev/null 2>&1 && ! venv_ready; then
    missing+=("Python venv/pip support (python3-venv on Debian/Ubuntu)")
  fi
}

report_tools() {
  if [ "${#missing[@]}" -gt 0 ]; then say "Missing: ${missing[*]}"; fi
  if [ "${#optional[@]}" -gt 0 ]; then say "Optional, not installed: ${optional[*]}"; fi
  if [ "${#missing[@]}" -gt 0 ] || [ "${#optional[@]}" -gt 0 ]; then
    if [ -n "$INSTALL_CMD" ]; then say "Install with: $INSTALL_CMD"; else say "Install the equivalents with your package manager."; fi
  fi
}

detect_tools
report_tools

if [ "$INSTALL_DEPS" -eq 1 ] && { [ "${#missing[@]}" -gt 0 ] || [ "${#optional[@]}" -gt 0 ]; }; then
  [ -n "$INSTALL_CMD" ] || die "--install-deps does not know this distribution; install the packages yourself"
  if [ "$DRY_RUN" -eq 1 ]; then
    say "(dry run) would run: $INSTALL_CMD, then check the tools again"
  else
    say "Running: $INSTALL_CMD"
    $INSTALL_CMD
    detect_tools
    if [ "${#missing[@]}" -gt 0 ]; then die "still missing after installing packages: ${missing[*]}"; fi
    say "After installation: all required tools are present."
  fi
fi

# CMake refuses to reuse a build directory with another generator, so an existing directory keeps its own.
# A new directory gets Ninja when it is available now, i.e. after any installation above.
GENERATOR="" GENERATOR_NOTE="" GENERATOR_ERROR=""
select_generator() {
  local cached=""
  if [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
    cached="$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "$BUILD_DIR/CMakeCache.txt" | head -n1)"
  fi
  if [ -n "$cached" ]; then
    GENERATOR="$cached" GENERATOR_NOTE=" (kept from the existing build directory)"
    case "$cached" in
      Ninja*) command -v ninja >/dev/null 2>&1 || GENERATOR_ERROR="$BUILD_DIR was configured with $cached but ninja is not installed; install ninja or use another --build-dir" ;;
      "Unix Makefiles") command -v make >/dev/null 2>&1 || GENERATOR_ERROR="$BUILD_DIR was configured with Unix Makefiles but make is not installed" ;;
    esac
  elif command -v ninja >/dev/null 2>&1; then
    GENERATOR="Ninja"
  else
    GENERATOR="Unix Makefiles"
  fi
}
select_generator

# --- GPU and CUDA toolkit ------------------------------------------------------------------------
# Select sm_75 when Turing is present, without adding it to the generic build.
AUTO_ARCHS="7.5 8.6 8.9 12.0"
DETECTED=() UNSUPPORTED=()
if command -v nvidia-smi >/dev/null 2>&1; then
  while IFS= read -r cap; do
    cap="${cap//[[:space:]]/}"; [ -n "$cap" ] || continue
    case " $AUTO_ARCHS " in
      *" $cap "*) DETECTED+=("${cap/./}") ;;
      *) UNSUPPORTED+=("$cap") ;;
    esac
  done < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u)
  say "NVIDIA GPU: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | paste -sd ';' - | sed 's/;/; /g')"
else
  say "NVIDIA GPU: nvidia-smi not found (no driver, or no NVIDIA GPU)"
fi

if [ -z "$ARCH" ]; then
  if [ "${#DETECTED[@]}" -gt 0 ]; then
    ARCH="$(printf '%s\n' "${DETECTED[@]}" | sort -un | paste -sd ';' -)"
  elif [ "${#UNSUPPORTED[@]}" -gt 0 ]; then
    say "Detected compute capability ${UNSUPPORTED[*]}, outside automatic targets ($AUTO_ARCHS)."
    say "Pass --arch (for example --arch ${UNSUPPORTED[0]/./}) to try it anyway; this is not officially supported."
    ARCH=""
  else
    ARCH="86;89;120"
    say "No GPU detected; using the installer default of 86;89;120."
  fi
fi

find_nvcc() {
  local candidate
  for candidate in "$CUDA_ROOT" "${CUDA_HOME:-}" "${CUDA_PATH:-}"; do
    [ -n "$candidate" ] && [ -x "$candidate/bin/nvcc" ] && { printf '%s\n' "$candidate/bin/nvcc"; return 0; }
  done
  if command -v nvcc >/dev/null 2>&1; then command -v nvcc; return 0; fi
  for candidate in /usr/local/cuda /opt/cuda; do
    [ -x "$candidate/bin/nvcc" ] && { printf '%s\n' "$candidate/bin/nvcc"; return 0; }
  done
  return 1
}

NVCC="" CUDA_OK=0
if NVCC="$(find_nvcc)"; then
  verline="$("$NVCC" --version | grep -o 'V[0-9]*\.[0-9]*\.[0-9]*' | head -n1 | tr -d V)"
  say "CUDA toolkit: $NVCC (nvcc $verline)"
  # nvcc 13.2.0 / 13.2.1 (13.2.0 to 13.2.85) miscompile IQ kernels on Blackwell; CMakeLists.txt rejects them too.
  IFS=. read -r nmaj nmin npat <<<"$verline"
  if [ "${nmaj:-0}" -eq 13 ] && [ "${nmin:-0}" -eq 2 ] && [ "${npat:-0}" -lt 86 ]; then
    say "This nvcc ($verline) is rejected by CMakeLists.txt. Use CUDA 13.2.2 (nvcc 13.2.86) or newer, or CUDA 13.1."
  else
    CUDA_OK=1
  fi
else
  say "CUDA toolkit: nvcc not found. Install the CUDA Toolkit 13.1 or 13.2.2+ (from NVIDIA or your distribution) and"
  say "pass --cuda-root DIR or put nvcc on PATH. This script does not install CUDA."
fi

# --- plan ----------------------------------------------------------------------------------------
say ""
say "Plan: configure with $GENERATOR$GENERATOR_NOTE, TINYSTRATA_BACKEND=CUDA, CMAKE_CUDA_ARCHITECTURES='${ARCH:-<unset>}'$( [ "$PORTABLE" -eq 1 ] && printf ', STRATA_PORTABLE=ON' )"
say "      build directory: $BUILD_DIR"
[ "$NO_VENV" -eq 1 ] || say "      Python virtual environment: $SRC/.venv (requirements.txt)"
[ -z "$GENERATOR_ERROR" ] || say "      problem: $GENERATOR_ERROR"

if [ "$DRY_RUN" -eq 1 ]; then say "Dry run: nothing was changed."; exit 0; fi

# Everything that can stop the build is checked here, before the expensive part.
[ -n "$ARCH" ] || die "no CUDA architecture selected (see above)"
[ "$CUDA_OK" -eq 1 ] || die "no usable CUDA toolkit (see above)"
[ "${#missing[@]}" -eq 0 ] || die "missing: ${missing[*]} (see above; --install-deps can install packages)"
[ -z "$GENERATOR_ERROR" ] || die "$GENERATOR_ERROR"

# --- build ---------------------------------------------------------------------------------------
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
cmake_args=(-S "$SRC" -B "$BUILD_DIR" -G "$GENERATOR" -DCMAKE_BUILD_TYPE=Release
  -DTINYSTRATA_BACKEND=CUDA "-DCMAKE_CUDA_ARCHITECTURES=$ARCH" "-DCMAKE_CUDA_COMPILER=$NVCC")
[ "$PORTABLE" -eq 1 ] && cmake_args+=(-DSTRATA_PORTABLE=ON)
cmake "${cmake_args[@]}"
cmake --build "$BUILD_DIR" --parallel "$JOBS"
for exe in tinystrata tinystrata-lfm2 tinystrata-gemma4; do
  [ -x "$BUILD_DIR/$exe" ] || die "expected $BUILD_DIR/$exe was not built"
done
say "Built: $BUILD_DIR/tinystrata, tinystrata-lfm2, tinystrata-gemma4"

if [ "$NO_VENV" -eq 0 ]; then
  [ -d "$SRC/.venv" ] || python3 -m venv "$SRC/.venv"
  "$SRC/.venv/bin/python" -m pip install --quiet -r "$SRC/requirements.txt"
  say "Python environment ready: $SRC/.venv"
fi
say "Done. Models are not downloaded by this script."
