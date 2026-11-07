"""Resolve Windows HIP metadata probes against the native executable's runtime.

Resolution only reads files. The managed-install migration copies an immutable
local generation; neither path loads a DLL, starts a process or initializes HIP. Managed bundles fail closed; legacy installs retain Windows' app-local,
System32, then configured/PATH lookup order for the actual PE import name.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
import copy
import shutil
import tempfile
import uuid
from pathlib import Path
import re
import stat
import struct


MANIFEST = ".tinystrata-hip-runtime.json"
RUNTIME_ENV = "TINYSTRATA_HIP_RUNTIME_DLL"
RUNTIME_DIRS_ENV = "TINYSTRATA_HIP_RUNTIME_DIRS"
COMGR_ENV = "TINYSTRATA_HIP_COMGR_DLLS"
MANAGED_HASH_ENV = "TINYSTRATA_HIP_MANAGED_RUNTIME_SHA256"
FINGERPRINT_ENV = "TINYSTRATA_HIP_RUNTIME_FINGERPRINT"
PROOF = ".tinystrata-hip-vmm.txt"
_RUNTIME = re.compile(r"amdhip64(?:_[0-9]+)?\.dll", re.I)
_BUNDLE_FILE = re.compile(r"(?:amdhip64(?:_[0-9]+)?|amd_comgr[0-9_]*|rocm_kpack)\.dll", re.I)
_MATH_FILES = {"hipblas.dll", "rocblas.dll", "rocsolver.dll", "libhipblaslt.dll", "libtensilelite-host.dll"}
_SDK_DISCOVERY = ("HIP_PATH", "ROCM_PATH", "ROCM_HOME", "HIP_CLANG_PATH", "HIP_DEVICE_LIB_PATH", "LLVM_PATH")


class HipRuntimeError(ValueError):
    """A selected managed HIP bundle is incomplete, changed or incompatible."""


@dataclass(frozen=True)
class HipRuntime:
    dll: str
    directories: tuple[str, ...]
    managed: bool


def _identity(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    if not path.is_file():
        raise HipRuntimeError(f"HIP runtime file is not a regular file: {path}")
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _present_file(path: Path) -> bool:
    try:
        info = path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    if not stat.S_ISREG(info.st_mode):
        raise HipRuntimeError(f"HIP runtime path is not a regular file: {path}")
    return True


@lru_cache(maxsize=256)
def _digest(path: str, identity: tuple[int, ...]) -> str:
    with open(path, "rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if _identity(Path(path)) != identity:
        raise HipRuntimeError(f"HIP runtime changed while it was being checked: {path}. Retry after the update finishes.")
    return actual


def _verified(path: str, identity: tuple[int, ...], expected: str) -> None:
    if _digest(path, identity) != expected:
        raise HipRuntimeError(f"Managed HIP runtime hash mismatch: {path}. Run the TinyStrata installer again.")


@lru_cache(maxsize=64)
def _dependencies(path: str, identity: tuple[int, ...]) -> frozenset[str]:
    raw = Path(path).read_bytes()
    return frozenset(name.lower() for encoding in ("ascii", "utf-16-le")
                     for name in re.findall(r"(?:rocm_kpack|amd_comgr[0-9_]*)\.dll",
                                            raw.decode(encoding, "ignore"), re.I))


def _unique_directories(values) -> tuple[str, ...]:
    result, seen = [], set()
    for value in values:
        directory = str(Path(value).expanduser().resolve())
        key = os.path.normcase(directory)
        if key not in seen:
            seen.add(key)
            result.append(directory)
    return tuple(result)


@lru_cache(maxsize=64)
def _manifest(path: str, identity: tuple[int, ...]) -> dict:
    if identity[2] > 65536:
        raise HipRuntimeError(f"Managed HIP runtime manifest is too large: {path}")
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise HipRuntimeError(f"Cannot read managed HIP runtime manifest {path}: {error}") from error
    if _identity(Path(path)) != identity:
        raise HipRuntimeError(f"HIP runtime manifest changed during the check: {path}")
    if not isinstance(data, dict) or type(data.get("schema")) is not int or data["schema"] != 1:
        raise HipRuntimeError(f"Unsupported managed HIP runtime manifest: {path}")
    runtime, directories, files = data.get("runtime_dll"), data.get("lib_dirs"), data.get("files")
    sdk_root = data.get("sdk_root")
    if sdk_root is not None and (not isinstance(sdk_root, str) or not sdk_root or not Path(sdk_root).is_absolute()):
        raise HipRuntimeError(f"Managed HIP SDK root must be an absolute path: {path}")
    if not isinstance(runtime, str) or not _RUNTIME.fullmatch(runtime):
        raise HipRuntimeError(f"Invalid HIP runtime DLL name in {path}")
    if not isinstance(directories, list) or len(directories) > 64 or any(
            not isinstance(d, str) or not d or not Path(d).is_absolute() for d in directories):
        raise HipRuntimeError(f"Managed HIP library directories must be absolute paths: {path}")
    if not isinstance(files, list) or not 1 <= len(files) <= 64:
        raise HipRuntimeError(f"Invalid managed HIP runtime file set: {path}")
    names = set()
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not _BUNDLE_FILE.fullmatch(entry["name"]):
            raise HipRuntimeError(f"Invalid managed HIP runtime file name: {path}")
        name, digest = entry["name"].lower(), entry.get("sha256")
        if name in names or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise HipRuntimeError(f"Invalid or duplicate managed HIP runtime file: {path}")
        names.add(name)
    if runtime.lower() not in names:
        raise HipRuntimeError(f"Managed HIP runtime DLL is absent from its file set: {path}")
    # COMGR is required by both legacy and current Windows HIP SDKs. Kpack is
    # required when this particular runtime references it (older SDKs lack it).
    if not any(name.startswith("amd_comgr") for name in names):
        raise HipRuntimeError(f"Managed HIP runtime has no matching COMGR file: {path}")
    dependencies = data.get("dependencies", [])
    if not isinstance(dependencies, list) or len(dependencies) > len(_MATH_FILES):
        raise HipRuntimeError(f"Invalid managed HIP math dependencies: {path}")
    seen = set()
    for entry in dependencies:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise HipRuntimeError(f"Invalid managed HIP math dependency: {path}")
        target, digest = Path(entry["path"]), entry.get("sha256")
        name = target.name.lower()
        if not target.is_absolute() or name not in _MATH_FILES or name in seen or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise HipRuntimeError(f"Invalid managed HIP math dependency identity: {path}")
        seen.add(name)
    return data


@lru_cache(maxsize=64)
def _imports(path: str, identity: tuple[int, ...]) -> tuple[str, ...]:
    data = Path(path).read_bytes()
    # Old unit fixtures and non-PE launchers do not describe native imports.
    if len(data) < 64 or data[:2] != b"MZ":
        return ()
    try:
        def read(fmt, offset):
            return struct.unpack_from(fmt, data, offset)[0]
        pe = read("<I", 60)
        if pe + 24 > len(data) or data[pe:pe + 4] != b"PE\0\0":
            raise ValueError("invalid PE signature")
        sections, optional_size = read("<H", pe + 6), read("<H", pe + 20)
        optional = pe + 24
        magic = read("<H", optional)
        directory = {0x20b: 112, 0x10b: 96}.get(magic)
        table = optional + optional_size
        if directory is None or optional_size < directory + 16 or sections > 96 or table + 40 * sections > len(data):
            raise ValueError("invalid PE headers")
        import_rva = read("<I", optional + directory + 8)
        if not import_rva:
            return ()
        def offset_of(rva):
            for index in range(sections):
                section = table + 40 * index
                address, size, raw = read("<I", section + 12), read("<I", section + 16), read("<I", section + 20)
                if address <= rva < address + size:
                    offset = raw + rva - address
                    if offset >= len(data):
                        raise ValueError("invalid PE section data")
                    return offset
            raise ValueError("unmapped import RVA")
        descriptor, names = offset_of(import_rva), []
        for _ in range(4096):
            if descriptor + 20 > len(data):
                raise ValueError("truncated import table")
            name_rva = read("<I", descriptor + 12)
            if not name_rva:
                return tuple(dict.fromkeys(names))
            start = offset_of(name_rva)
            end = data.find(b"\0", start, min(len(data), start + 257))
            if end < start:
                raise ValueError("invalid import name")
            name = data[start:end].decode("ascii", "strict")
            if _RUNTIME.fullmatch(name):
                names.append(name)
            descriptor += 20
        raise ValueError("unterminated import table")
    except (ValueError, struct.error) as error:
        raise HipRuntimeError(f"Cannot inspect HIP imports in {path}: {error}") from error


def hip_imports(executable: str | Path) -> tuple[str, ...]:
    path = Path(executable).expanduser().resolve()
    try:
        return _imports(str(path), _identity(path))
    except OSError as error:
        raise HipRuntimeError(f"Cannot read native executable {path}: {error}") from error


def _system_directory() -> Path | None:
    root = os.environ.get("SystemRoot") or os.environ.get("WINDIR")
    return Path(root) / "System32" if root else None


def resolve_hip_runtime(executable: str | Path | None, directories=()) -> HipRuntime | None:
    """Read the native runtime choice; managed bundle failures never try another SDK."""
    if not executable:
        return None
    path = Path(executable).expanduser().resolve()
    manifest_path = path.parent / MANIFEST
    try:
        try:
            manifest_path.lstat()
            have_manifest = True
        except (FileNotFoundError, NotADirectoryError):
            have_manifest = False
        if have_manifest:
            data = _manifest(str(manifest_path), _identity(manifest_path))
            if data.get("sdk_root") and not Path(data["sdk_root"]).is_dir():
                raise HipRuntimeError(f"Managed HIP SDK root is missing: {data['sdk_root']}")
            selected_dirs = _unique_directories(data["lib_dirs"])
            for directory in selected_dirs:
                if not Path(directory).is_dir():
                    raise HipRuntimeError(f"Managed HIP library directory is missing: {directory}. Run the TinyStrata installer again.")
            selected_files = {entry["name"].lower(): entry for entry in data["files"]}
            for entry in selected_files.values():
                target = path.parent / entry["name"]
                _verified(str(target), _identity(target), entry["sha256"].lower())
            dll = path.parent / data["runtime_dll"]
            for name in _dependencies(str(dll), _identity(dll)):
                if name not in selected_files:
                    raise HipRuntimeError(f"Managed HIP runtime dependency is absent from the bundle: {name}")
            for dependency in data.get("dependencies", []):
                expected = Path(dependency["path"])
                if os.path.normcase(str(expected.parent.resolve())) not in {os.path.normcase(d) for d in selected_dirs}:
                    raise HipRuntimeError(f"HIP math dependency is outside the selected SDK directories: {expected}")
                digest = dependency["sha256"].lower()
                _verified(str(expected), _identity(expected), digest)
                system = _system_directory()
                for directory in (path.parent, *([system] if system else []), *map(Path, selected_dirs)):
                    shadow = directory / expected.name
                    if _present_file(shadow):
                        if _digest(str(shadow), _identity(shadow)) != digest:
                            raise HipRuntimeError(f"Conflicting HIP math DLL would override the selected SDK: {shadow}. "
                                                  "Remove the conflicting copy or run the matching TinyStrata installer.")
                        break  # First existing DLL is what Windows will select.
            _verify_native_proof(data, path.parent)
            imports = hip_imports(path) if _present_file(path) else ()
            if any(name.lower() != data["runtime_dll"].lower() for name in imports):
                raise HipRuntimeError(f"Managed HIP runtime does not match the imports of {path.name}")
            return HipRuntime(str(dll), selected_dirs, True)
        if not _present_file(path):
            return None
        imports = hip_imports(path)
        if not imports:
            return None
        if len(imports) != 1:
            raise HipRuntimeError(f"Native executable imports multiple HIP runtimes: {path}")
        selected_dirs = _unique_directories(directories)
        system = _system_directory()
        candidates = [path.parent, *([system] if system else []), *map(Path, selected_dirs)]
        candidates.extend(Path(d) for d in os.environ.get("PATH", "").split(os.pathsep) if d)
        for directory in candidates:
            dll = directory / imports[0]
            if _present_file(dll):
                return HipRuntime(str(dll.resolve()), selected_dirs, False)
        raise HipRuntimeError(f"Cannot find {imports[0]} required by {path.name}. Run the TinyStrata installer again.")
    except OSError as error:
        raise HipRuntimeError(f"Managed HIP runtime files are missing or unreadable beside {path}: {error}") from error



def _native_proof(data: dict, directory: Path) -> bytes:
    """ASCII proof for the native guard; an unsupported older SDK gets no grant."""
    entries = {entry["name"].lower(): entry for entry in data["files"]}
    runtime = data["runtime_dll"].lower()
    if "rocm_kpack.dll" not in entries:
        return b""
    comgr = [name for name in entries if name.startswith("amd_comgr")]
    referenced = _dependencies(str(directory / entries[runtime]["name"]),
                               _identity(directory / entries[runtime]["name"]))
    chosen = [name for name in comgr if name in referenced]
    if not chosen and len(comgr) == 1:
        chosen = comgr
    if len(chosen) != 1:
        return b""
    names = sorted((runtime, chosen[0], "rocm_kpack.dll"))
    return ("TINYSTRATA_HIP_VMM_V1\n" + "".join(
        name + " " + entries[name]["sha256"].lower() + "\n" for name in names)).encode("ascii")


def _verify_native_proof(data: dict, directory: Path) -> None:
    path = directory / PROOF
    if not _present_file(path):
        return  # Older managed generation: native keeps the coarse fallback.
    if path.stat().st_size > 2048:
        raise HipRuntimeError(f"Managed HIP native proof is too large: {path}")
    expected = _native_proof(data, directory)
    if expected and not {"hipblas.dll", "rocblas.dll"} <= {
            Path(entry["path"]).name.lower() for entry in data.get("dependencies", [])}:
        raise HipRuntimeError(f"Managed HIP native proof lacks verified HIP/rocBLAS dependencies: {path}")
    if path.read_bytes().replace(b"\r\n", b"\n") != expected:
        raise HipRuntimeError(f"Managed HIP native proof does not match its verified bundle: {path}. Run the installer again.")



def runtime_fingerprint(runtime: HipRuntime) -> str:
    """Bind a core runtime to its COMGR/math set and SDK asset directories."""
    directory = Path(runtime.dll).parent
    identity = {"runtime": _digest(runtime.dll, _identity(Path(runtime.dll))),
                "directories": [os.path.normcase(path) for path in runtime.directories]}
    if runtime.managed:
        data = _manifest(str(directory / MANIFEST), _identity(directory / MANIFEST))
        identity["core"] = sorted((entry["name"].lower(), entry["sha256"].lower()) for entry in data["files"])
        identity["math"] = sorted((Path(entry["path"]).name.lower(), entry["sha256"].lower())
                                  for entry in data.get("dependencies", []))
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


def runtime_environment(cfg: dict, env: dict) -> dict:
    """Apply the executable's runtime to an environment without changing saved settings."""
    result = dict(env)
    if cfg.get("backend") != "hip" or os.name != "nt":
        return result
    # A parent process or saved custom env cannot grant managed status.
    for key in (MANAGED_HASH_ENV, FINGERPRINT_ENV, RUNTIME_ENV, RUNTIME_DIRS_ENV, COMGR_ENV):
        result.pop(key, None)
    executable = cfg.get("exe")
    if executable and not Path(executable).is_absolute():
        executable = Path(cfg.get("cwd") or ".") / executable
    runtime = resolve_hip_runtime(executable, cfg.get("lib_dirs") or ())
    if runtime is not None:
        result[RUNTIME_ENV] = runtime.dll
        result[FINGERPRINT_ENV] = runtime_fingerprint(runtime)
        directories = (str(Path(runtime.dll).parent), *runtime.directories)
        directories = tuple(dict.fromkeys(directories))
        result[RUNTIME_DIRS_ENV] = json.dumps(directories)
        result["PATH"] = os.pathsep.join((*directories, result.get("PATH", "")))
        if runtime.managed:
            result[MANAGED_HASH_ENV] = _digest(runtime.dll, _identity(Path(runtime.dll)))
            manifest = Path(runtime.dll).parent / MANIFEST
            data = _manifest(str(manifest), _identity(manifest))
            for name in _SDK_DISCOVERY:
                result.pop(name, None)
            if data.get("sdk_root"):
                root = str(Path(data["sdk_root"]).resolve())
                result["HIP_PATH"] = result["ROCM_PATH"] = root
            referenced = _dependencies(runtime.dll, _identity(Path(runtime.dll)))
            comgr = [entry["name"] for entry in data["files"] if entry["name"].lower().startswith("amd_comgr")]
            chosen = [name for name in comgr if name.lower() in referenced]
            if not chosen and len(comgr) == 1:
                chosen = comgr
            result[COMGR_ENV] = json.dumps([str(Path(runtime.dll).parent / name) for name in chosen])
        else:
            result.pop(COMGR_ENV, None)
    return result



def _installed_pinned_sdk(work: Path) -> Path | None:
    """Read package metadata without importing/running a possibly different SDK."""
    pin = "10.2.0a20260930"
    packages = ("rocm", "rocm_sdk_core", "rocm_sdk_devel", "rocm_sdk_libraries")
    for environment in (work / "rocm-venv", work / ("rocm-venv-" + pin)):
        site = environment / "Lib/site-packages"
        complete = True
        for package in packages:
            records = list(site.glob(package + "-*.dist-info/METADATA"))
            if len(records) != 1:
                complete = False
                break
            fields = dict(line.split(": ", 1) for line in records[0].read_text(encoding="utf-8").splitlines()
                          if line.startswith(("Name: ", "Version: ")))
            if fields.get("Version") != pin or fields.get("Name", "").lower().replace("-", "_") != package:
                complete = False
                break
        sdk = site / "_rocm_sdk_devel"
        if complete and (sdk / "include/hip/hip_runtime.h").is_file():
            return sdk
    return None


def migrate_installed_runtime(cfg: dict, config_path: str | Path | None) -> dict:
    """Move a legacy managed installation to a verified local SDK generation.

    Only conventional installed configs and owned native/source paths qualify.
    Nothing is downloaded, loaded or overwritten. The caller atomically saves
    the returned config after its normal validation; custom configs stay legacy.
    """
    if os.name != "nt" or cfg.get("backend") != "hip" or not cfg.get("exe") or not config_path:
        return cfg
    config_path = Path(config_path).expanduser().resolve()
    if (not config_path.name.lower().startswith("config") or config_path.suffix.lower() != ".json" or
            config_path.parent.name.lower() != ".tinystrata" or
            config_path.parent.parent.name.lower() != "runtime"):
        return cfg
    work = config_path.parent.parent.parent
    executable = Path(cfg["exe"]).expanduser()
    if not executable.is_absolute():
        executable = Path(cfg.get("cwd") or ".") / executable
    executable = executable.resolve()
    application = Path(__file__).resolve().parent.parent
    if not any(root == executable.parent or root in executable.parents for root in (work, application)):
        return cfg
    runtime = resolve_hip_runtime(executable, cfg.get("lib_dirs") or ())
    if runtime is not None and runtime.managed:
        data = _manifest(str(executable.parent / MANIFEST), _identity(executable.parent / MANIFEST))
        if (executable.parent / PROOF).is_file() or not _native_proof(data, executable.parent):
            return cfg
    if not executable.is_file():
        return cfg
    imports = hip_imports(executable)
    if len(imports) != 1:
        return cfg
    # Choose a complete explicit SDK, never a mixture filled from System32/PATH.
    sdk = _installed_pinned_sdk(work)
    if sdk is None:
        for directory in cfg.get("lib_dirs") or ():
            directory = Path(directory).expanduser().resolve()
            for root in (directory, *list(directory.parents)[:3]):
                if (root / "include/hip/hip_runtime.h").is_file():
                    sdk = root
                    break
            if sdk is not None:
                break
    if sdk is None:
        return cfg  # No local SDK: preserve the old runtime and coarse native guard.
    directories = [path for path in (sdk / "bin", sdk / "lib", sdk / "lib/llvm/bin") if path.is_dir()]
    core_directory = next((path for path in directories if (path / imports[0]).is_file()), None)
    if core_directory is None:
        raise HipRuntimeError(f"The configured SDK does not provide {imports[0]}; run the TinyStrata installer again.")
    core = {imports[0]: core_directory / imports[0]}
    for path in core_directory.glob("amd_comgr*.dll"):
        core[path.name] = path
    if not any(name.lower().startswith("amd_comgr") for name in core):
        raise HipRuntimeError(f"The selected SDK has no matching COMGR library: {sdk}")
    kpack = core_directory / "rocm_kpack.dll"
    if kpack.is_file():
        core[kpack.name] = kpack
    for name in _dependencies(str(core[imports[0]]), _identity(core[imports[0]])):
        if name not in {entry.lower() for entry in core}:
            raise HipRuntimeError(f"The selected SDK is missing {name}; run the TinyStrata installer again.")
    dependencies = []
    for name in sorted(_MATH_FILES):
        path = next((folder / name for folder in directories if (folder / name).is_file()), None)
        if path is not None:
            dependencies.append({"path": str(path), "sha256": _digest(str(path), _identity(path))})
    if not {"hipblas.dll", "rocblas.dll"} <= {Path(entry["path"]).name.lower() for entry in dependencies}:
        raise HipRuntimeError(f"The selected SDK has no complete HIP/rocBLAS runtime: {sdk}")
    files = [{"name": name, "sha256": _digest(str(path), _identity(path))} for name, path in sorted(core.items())]
    native_root = work / "native"
    native_root.mkdir(parents=True, exist_ok=True)
    if native_root.is_symlink() or native_root.resolve() != native_root:
        raise HipRuntimeError("The managed native directory is linked; run the TinyStrata installer to repair it.")
    stage = Path(tempfile.mkdtemp(prefix=".hip-prepare-", dir=native_root))
    final = native_root / ("hip-" + uuid.uuid4().hex)
    try:
        for old in executable.parent.iterdir():
            if old.is_file() and re.fullmatch(r"tinystrata(?:-[a-z0-9]+)?\.exe|strata[^\\/]*\.dll|(?:msvcp140[^\\/]*|vcruntime140[^\\/]*|vcomp140|concrt140|vccorlib140)\.dll", old.name, re.I):
                shutil.copy2(old, stage / old.name)
        if not (stage / executable.name).is_file():
            raise HipRuntimeError("The selected TinyStrata executable could not be copied.")
        for old in stage.glob("*.exe"):
            if any(name.lower() != imports[0].lower() for name in hip_imports(old)):
                raise HipRuntimeError("Installed native engines use different HIP runtimes; rebuild them together.")
        for name, old in core.items():
            shutil.copy2(old, stage / name)
        manifest = {"schema": 1, "runtime_dll": imports[0], "sdk_root": str(sdk),
                    "lib_dirs": list(map(str, directories)), "files": files, "dependencies": dependencies}
        (stage / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (stage / PROOF).write_bytes(_native_proof(manifest, stage))
        # The resolver rechecks source/copied hashes, dependency shadows and PE
        # imports before any generation or config is published.
        resolve_hip_runtime(stage / executable.name, directories)
        stage.rename(final)
        result = copy.deepcopy(cfg)
        result["exe"] = str(final / executable.name)
        result["lib_dirs"] = list(map(str, directories))
        for holder, key in ((result, "vision_exe"), (result.get("vision") or {}, "exe")):
            value = holder.get(key)
            if value:
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = Path(cfg.get("cwd") or ".") / path
                if path.resolve().parent == executable.parent and (final / path.name).is_file():
                    holder[key] = str(final / path.name)
        return result
    except BaseException:
        # Only the unpublished generated directory is ours to remove.
        if stage.parent == native_root and stage.name.startswith(".hip-prepare-") and stage.exists():
            shutil.rmtree(stage)
        raise
