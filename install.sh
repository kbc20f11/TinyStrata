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
#   --install-deps       Install missing build packages with the distribution's package manager (uses sudo unless run as root)
#   --arch LIST          CUDA architectures, e.g. 86 or "86;89". Overrides detection (needed for GPUs
#                        outside 86/89/120, e.g. Turing: --arch 75; these are not officially supported)
#   --cuda-root DIR      CUDA toolkit root (default: $CUDA_HOME, $CUDA_PATH, nvcc on PATH, /usr/local/cuda, /opt/cuda)
#   --build-dir DIR      Build directory (default: ./build)
#   --portable           Pass -DSTRATA_PORTABLE=ON (AVX2 floor, for use on other PCs)
#   --jobs N             Parallel build jobs (default: number of CPUs)
#   --no-venv            Do not create the Python virtual environment
#   -h, --help           Show this help
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
ID="" ID_LIKE=""
if [ -r /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
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

# --- build tools --------------------------------------------------------------------------------
missing=()
for tool in cmake git python3; do
  command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if ! command -v g++ >/dev/null 2>&1 && ! command -v c++ >/dev/null 2>&1; then missing+=("a C++ compiler (g++)"); fi
GENERATOR="Unix Makefiles"
if command -v ninja >/dev/null 2>&1; then GENERATOR="Ninja"; else missing+=("ninja (optional, faster builds)"); fi
if [ "${#missing[@]}" -gt 0 ]; then
  say "Missing: ${missing[*]}"
  if [ -n "$INSTALL_CMD" ]; then say "Install with: $INSTALL_CMD"; else say "Install the equivalents with your package manager."; fi
fi
python3 -c 'import venv' >/dev/null 2>&1 || { say "Missing: the Python venv module (python3-venv on Debian/Ubuntu)"; }

# --- GPU and CUDA toolkit ------------------------------------------------------------------------
# GPU architectures the CMake default and the Windows installer accept.
SUPPORTED="8.6 8.9 12.0"
DETECTED=() UNSUPPORTED=()
if command -v nvidia-smi >/dev/null 2>&1; then
  while IFS= read -r cap; do
    cap="${cap//[[:space:]]/}"; [ -n "$cap" ] || continue
    case " $SUPPORTED " in
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
    say "Detected compute capability ${UNSUPPORTED[*]}, which is outside the supported 86 / 89 / 120."
    say "Pass --arch (for example --arch ${UNSUPPORTED[0]/./}) to try it anyway; this is not officially supported."
    ARCH=""
  else
    ARCH="86;89;120"
    say "No GPU detected; using the installer default of 86;89;120 (a build that runs on any supported GPU)."
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
say "Plan: configure with $GENERATOR, TINYSTRATA_BACKEND=CUDA, CMAKE_CUDA_ARCHITECTURES='${ARCH:-<unset>}'$( [ "$PORTABLE" -eq 1 ] && printf ', STRATA_PORTABLE=ON' )"
say "      build directory: $BUILD_DIR"
[ "$NO_VENV" -eq 1 ] || say "      Python virtual environment: $SRC/.venv (requirements.txt)"

if [ "$INSTALL_DEPS" -eq 1 ] && [ "${#missing[@]}" -gt 0 ]; then
  [ -n "$INSTALL_CMD" ] || die "--install-deps does not know this distribution; install the packages yourself"
  if [ "$DRY_RUN" -eq 1 ]; then say "(dry run) would run: $INSTALL_CMD"; else say "Running: $INSTALL_CMD"; $INSTALL_CMD; fi
fi

if [ "$DRY_RUN" -eq 1 ]; then say "Dry run: nothing was changed."; exit 0; fi

[ -n "$ARCH" ] || die "no CUDA architecture selected (see above)"
[ "$CUDA_OK" -eq 1 ] || die "no usable CUDA toolkit (see above)"
for tool in cmake git python3; do command -v "$tool" >/dev/null 2>&1 || die "$tool is required (see above)"; done
command -v g++ >/dev/null 2>&1 || command -v c++ >/dev/null 2>&1 || die "a C++ compiler is required (see above)"

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
