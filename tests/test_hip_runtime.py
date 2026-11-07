"""Windows HIP runtime selection uses files only; every metadata subprocess is mocked."""
import hashlib
import contextlib
import ctypes
import io
import json
import os
import shutil
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
from types import ModuleType
import unittest
from unittest.mock import patch

from serve import backend, hip_runtime, windows_amd_telemetry
from serve.hip_runtime import (COMGR_ENV, HipRuntimeError, MANIFEST, RUNTIME_DIRS_ENV, RUNTIME_ENV,
                               resolve_hip_runtime, runtime_environment)


def write_pe(path, runtime="amdhip64_7.dll", *, pe32=False):
    """An import-only PE image. No generated fixture is ever executed."""
    data = bytearray(1024)
    struct.pack_into("<H", data, 0, 0x5a4d)
    struct.pack_into("<I", data, 60, 128)
    data[128:132] = b"PE\0\0"
    struct.pack_into("<H", data, 134, 1)
    struct.pack_into("<H", data, 148, 224 if pe32 else 240)
    struct.pack_into("<H", data, 152, 0x10b if pe32 else 0x20b)
    directory = 96 if pe32 else 112
    struct.pack_into("<I", data, 152 + directory + 8, 0x1000)
    section = 152 + (224 if pe32 else 240)
    struct.pack_into("<III", data, section + 12, 0x1000, 512, 512)
    struct.pack_into("<I", data, 524, 0x1040)
    name = runtime.encode("ascii")
    data[576:576 + len(name)] = name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RuntimeFiles(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="tinystrata-hip-runtime-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.engine = self.root / "engine"
        self.sdk = self.root / "new SDK"
        self.old_sdk = self.root / "old SDK"
        self.windows = self.root / "Windows"
        self.system = self.windows / "System32"
        for directory in (self.engine, self.sdk, self.old_sdk, self.system):
            directory.mkdir(parents=True)
        self.exe = self.engine / "tinystrata.exe"
        write_pe(self.exe)
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {"SystemRoot": str(self.windows), "PATH": str(self.old_sdk)}, clear=True).start()

    def bundle(self, *, dependencies=True):
        values = {"amdhip64_7.dll": b"MZ selected SDK amd_comgr.dll rocm_kpack.dll",
                  "amd_comgr.dll": b"selected COMGR", "rocm_kpack.dll": b"selected KPACK"}
        files = []
        for name, value in values.items():
            path = self.engine / name
            path.write_bytes(value)
            files.append({"name": name, "sha256": sha(path)})
        manifest = {"schema": 1, "runtime_dll": "amdhip64_7.dll", "lib_dirs": [str(self.sdk)], "files": files}
        if dependencies:
            manifest["dependencies"] = []
            for name in ("hipblas.dll", "rocblas.dll"):
                path = self.sdk / name
                path.write_bytes(("selected " + name).encode())
                manifest["dependencies"].append({"path": str(path), "sha256": sha(path)})
        self.save_manifest(manifest)
        return manifest

    def save_manifest(self, manifest):
        (self.engine / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")

    def test_managed_bundle_wins_over_old_sdk_and_system_runtime(self):
        self.bundle()
        for directory in (self.old_sdk, self.system):
            (directory / "amdhip64_7.dll").write_bytes(b"wrong old runtime")
        runtime = resolve_hip_runtime(self.exe, [self.old_sdk])
        self.assertTrue(runtime.managed)
        self.assertEqual(runtime.dll, str(self.engine / "amdhip64_7.dll"))
        self.assertEqual(runtime.directories, (str(self.sdk),))

    def test_native_proof_is_exactly_bound_to_verified_core_set(self):
        manifest = self.bundle()
        proof = self.engine / hip_runtime.PROOF
        wanted = hip_runtime._native_proof(manifest, self.engine)
        self.assertEqual(wanted.splitlines()[0], b"TINYSTRATA_HIP_VMM_V1")
        self.assertEqual(len(wanted.splitlines()), 4)
        proof.write_bytes(wanted.replace(b"\n", b"\r\n"))
        self.assertTrue(resolve_hip_runtime(self.exe).managed)
        proof.write_bytes(wanted.replace(b"amd_comgr.dll ", b"rocm_kpack.dll ", 1))
        with self.assertRaisesRegex(HipRuntimeError, "native proof"):
            resolve_hip_runtime(self.exe)
        proof.write_bytes(b"TINYSTRATA_HIP_VMM_V1\n")
        with self.assertRaisesRegex(HipRuntimeError, "native proof"):
            resolve_hip_runtime(self.exe)

    @unittest.skipUnless(os.name == "nt", "Windows environment policy")
    def test_managed_hash_marker_is_recreated_not_inherited_by_legacy(self):
        self.bundle()
        marker = hip_runtime.MANAGED_HASH_ENV
        result = runtime_environment({"backend": "hip", "exe": str(self.exe)}, {marker: "spoofed"})
        self.assertEqual(result[marker], sha(self.engine / "amdhip64_7.dll"))
        (self.engine / MANIFEST).unlink()
        result = runtime_environment({"backend": "hip", "exe": str(self.exe)}, {marker: "stale"})
        self.assertNotIn(marker, result)

    def migration_fixture(self):
        manifest = self.bundle()
        work = self.root / "installed-work"
        old = work / "build-hip"
        old.mkdir(parents=True)
        for name in ("tinystrata.exe", "tinystrata-gemma4.exe", "tinystrata-lfm2.exe", "tinystrata-vision.exe"):
            write_pe(old / name)
        (self.system / "amdhip64_7.dll").write_bytes(b"legacy system core")
        (self.sdk / "include/hip").mkdir(parents=True)
        (self.sdk / "include/hip/hip_runtime.h").write_text("fixture", encoding="ascii")
        selected = self.sdk / "bin"
        selected.mkdir()
        for entry in manifest["files"]:
            shutil.copy2(self.engine / entry["name"], selected / entry["name"])
        for name in ("hipblas.dll", "rocblas.dll"):
            shutil.copy2(self.sdk / name, selected / name)
        config = work / "runtime/.tinystrata/config.json"
        config.parent.mkdir(parents=True)
        config.write_text('{"old":true}', encoding="ascii")
        cfg = {"backend": "hip", "exe": str(old / "tinystrata.exe"), "lib_dirs": [str(selected)],
               "args": ["--mtp", "3", "--lookup-draft", "7"], "mtp_auto": 0,
               "vision": {"exe": str(old / "tinystrata-vision.exe"), "gpu": True},
               "sampling": {"temperature": 0.7}, "api_key": "fixture-preserved"}
        return cfg, config, old, work

    @unittest.skipUnless(os.name == "nt", "Windows installed generation migration")
    def test_ordinary_legacy_launch_publishes_immutable_verified_generation(self):
        cfg, config, old, work = self.migration_fixture()
        previous = {p.name: p.read_bytes() for p in old.iterdir()}
        result = hip_runtime.migrate_installed_runtime(cfg, config)
        folder = Path(result["exe"]).parent
        self.assertEqual(folder.parent, work / "native")
        self.assertTrue(resolve_hip_runtime(result["exe"]).managed)
        self.assertEqual((folder / hip_runtime.PROOF).read_bytes(),
                         hip_runtime._native_proof(json.loads((folder / MANIFEST).read_text()), folder))
        self.assertEqual(result["vision"]["exe"], str(folder / "tinystrata-vision.exe"))
        self.assertEqual(result["args"], cfg["args"])
        self.assertEqual(result["sampling"], cfg["sampling"])
        self.assertEqual(result["api_key"], cfg["api_key"])
        self.assertEqual(cfg["exe"], str(old / "tinystrata.exe"))
        self.assertEqual(config.read_text(), '{"old":true}')  # caller commits config atomically
        self.assertEqual(previous, {p.name: p.read_bytes() for p in old.iterdir()})
        self.assertIs(hip_runtime.migrate_installed_runtime(result, config), result)

    @unittest.skipUnless(os.name == "nt", "Windows installed generation migration")
    def test_unmanaged_old_profile_prefers_locally_verified_private_pin(self):
        cfg, config, old, work = self.migration_fixture()
        site = work / "rocm-venv/Lib/site-packages"
        pinned = site / "_rocm_sdk_devel"
        shutil.copytree(self.sdk, pinned)
        for package in ("rocm", "rocm_sdk_core", "rocm_sdk_devel", "rocm_sdk_libraries"):
            info = site / (package + "-10.2.0a20260930.dist-info")
            info.mkdir(parents=True)
            (info / "METADATA").write_text("Name: " + package.replace("_", "-") + "\nVersion: 10.2.0a20260930\n", encoding="utf-8")
        result = hip_runtime.migrate_installed_runtime(cfg, config)
        manifest = json.loads((Path(result["exe"]).parent / MANIFEST).read_text(encoding="utf-8"))
        self.assertEqual(manifest["sdk_root"], str(pinned))
        self.assertEqual(result["lib_dirs"][0], str(pinned / "bin"))
        # A mixed or wrong package pin must never masquerade as the tested SDK.
        metadata = site / "rocm-10.2.0a20260930.dist-info/METADATA"
        metadata.write_text("Name: rocm\nVersion: 7.2\n", encoding="utf-8")
        self.assertIsNone(hip_runtime._installed_pinned_sdk(work))

    @unittest.skipUnless(os.name == "nt", "Windows installed generation migration")
    def test_migration_failure_preserves_prior_native_config_and_removes_unpublished_stage(self):
        cfg, config, old, work = self.migration_fixture()
        original = (old / "tinystrata.exe").read_bytes()
        with patch.object(hip_runtime.shutil, "copy2", side_effect=OSError("copy denied")):
            with self.assertRaisesRegex(OSError, "copy denied"):
                hip_runtime.migrate_installed_runtime(cfg, config)
        self.assertEqual((old / "tinystrata.exe").read_bytes(), original)
        self.assertEqual(config.read_text(), '{"old":true}')
        self.assertEqual(list((work / "native").iterdir()), [])

    @unittest.skipUnless(os.name == "nt", "Windows installed generation migration")
    def test_custom_config_and_no_sdk_keep_coarse_legacy_route(self):
        cfg, config, old, work = self.migration_fixture()
        self.assertIs(hip_runtime.migrate_installed_runtime(cfg, self.root / "private.json"), cfg)
        no_sdk = {**cfg, "lib_dirs": [str(self.old_sdk)]}
        self.assertIs(hip_runtime.migrate_installed_runtime(no_sdk, config), no_sdk)
        self.assertFalse((work / "native").exists())

    def test_earlier_selected_directory_cannot_shadow_verified_math(self):
        manifest = self.bundle()
        (self.old_sdk / "hipblas.dll").write_bytes(b"different earlier math")
        manifest["lib_dirs"] = [str(self.old_sdk), str(self.sdk)]
        self.save_manifest(manifest)
        with self.assertRaisesRegex(HipRuntimeError, "override the selected SDK"):
            resolve_hip_runtime(self.exe)

    def test_manifest_swap_changes_selected_sdk_without_old_config_winning(self):
        manifest = self.bundle()
        replacement = self.root / "replacement SDK"
        replacement.mkdir()
        manifest["lib_dirs"] = [str(replacement)]
        manifest.pop("dependencies")
        self.save_manifest(manifest)
        runtime = resolve_hip_runtime(self.exe, [self.old_sdk])
        self.assertEqual(runtime.directories, (str(replacement),))

    def test_missing_or_changed_core_never_falls_back(self):
        self.bundle()
        (self.old_sdk / "amdhip64_7.dll").write_bytes(b"old runtime")
        resolve_hip_runtime(self.exe)
        (self.engine / "rocm_kpack.dll").unlink()
        with self.assertRaises(HipRuntimeError):
            resolve_hip_runtime(self.exe, [self.old_sdk])
        self.bundle()
        resolve_hip_runtime(self.exe)
        (self.engine / "amdhip64_7.dll").write_bytes(b"changed SDK runtime")
        with self.assertRaisesRegex(HipRuntimeError, "hash mismatch"):
            resolve_hip_runtime(self.exe, [self.old_sdk])

    def test_removed_sdk_directory_is_not_replaced_with_old_config(self):
        self.bundle(dependencies=False)
        self.sdk.rmdir()
        with self.assertRaisesRegex(HipRuntimeError, "directory is missing"):
            resolve_hip_runtime(self.exe, [self.old_sdk])

    def test_dependency_missing_from_manifest_is_rejected_even_if_local_file_exists(self):
        manifest = self.bundle()
        manifest["files"] = [entry for entry in manifest["files"] if entry["name"] != "rocm_kpack.dll"]
        self.save_manifest(manifest)
        with self.assertRaisesRegex(HipRuntimeError, "dependency is absent"):
            resolve_hip_runtime(self.exe)

    def test_invalid_manifest_fields_and_path_traversal(self):
        for field, value in (("schema", True), ("schema", 2), ("runtime_dll", "../amdhip64_7.dll"),
                             ("lib_dirs", ["relative SDK"]), ("files", [{"name": "../amd_comgr.dll", "sha256": "0" * 64}])):
            with self.subTest(field=field, value=value):
                manifest = self.bundle()
                manifest[field] = value
                self.save_manifest(manifest)
                with self.assertRaises(HipRuntimeError):
                    resolve_hip_runtime(self.exe)

    def test_duplicate_files_and_wrong_import_basename_are_rejected(self):
        manifest = self.bundle()
        manifest["files"].append({**manifest["files"][0], "name": "AMDHIP64_7.DLL"})
        self.save_manifest(manifest)
        with self.assertRaisesRegex(HipRuntimeError, "duplicate"):
            resolve_hip_runtime(self.exe)
        self.bundle()
        write_pe(self.exe, "amdhip64.dll")
        with self.assertRaisesRegex(HipRuntimeError, "does not match"):
            resolve_hip_runtime(self.exe)

    def test_invalid_json_does_not_revert_to_legacy(self):
        (self.engine / MANIFEST).write_text("{bad", encoding="utf-8")
        (self.system / "amdhip64_7.dll").write_bytes(b"old")
        with self.assertRaisesRegex(HipRuntimeError, "manifest"):
            resolve_hip_runtime(self.exe)

    def test_hashes_are_cached_but_file_replacement_invalidates_cache(self):
        self.bundle()
        with patch.object(hip_runtime.hashlib, "file_digest", wraps=hashlib.file_digest) as digest:
            resolve_hip_runtime(self.exe)
            count = digest.call_count
            self.assertGreaterEqual(count, 5)
            resolve_hip_runtime(self.exe)
            self.assertEqual(digest.call_count, count)
            changed = self.engine / "amd_comgr.dll"
            replacement = self.engine / "new-comgr"
            replacement.write_bytes(changed.read_bytes() + b" altered")
            replacement.replace(changed)
            with self.assertRaisesRegex(HipRuntimeError, "hash mismatch"):
                resolve_hip_runtime(self.exe)
            self.assertGreater(digest.call_count, count)

    def test_math_identity_change_or_missing_source_fails_closed(self):
        self.bundle()
        resolve_hip_runtime(self.exe)
        (self.sdk / "hipblas.dll").write_bytes(b"SDK was updated independently")
        with self.assertRaisesRegex(HipRuntimeError, "hash mismatch"):
            resolve_hip_runtime(self.exe)
        self.bundle()
        (self.sdk / "rocblas.dll").unlink()
        with self.assertRaises(HipRuntimeError):
            resolve_hip_runtime(self.exe)

    def test_old_system_or_app_local_math_cannot_shadow_sdk(self):
        for directory in (self.system, self.engine):
            with self.subTest(directory=directory):
                self.bundle()
                shadow = directory / "hipblas.dll"
                shadow.write_bytes(b"old unrelated math DLL")
                with self.assertRaisesRegex(HipRuntimeError, "Conflicting HIP math DLL"):
                    resolve_hip_runtime(self.exe)
                shadow.write_bytes((self.sdk / "hipblas.dll").read_bytes())
                self.assertTrue(resolve_hip_runtime(self.exe).managed)
                shadow.unlink()

    def test_unallowlisted_math_identity_is_rejected(self):
        manifest = self.bundle()
        manifest["dependencies"] = [{"path": str(self.sdk / "unrelated.dll"), "sha256": "0" * 64}]
        self.save_manifest(manifest)
        with self.assertRaisesRegex(HipRuntimeError, "math dependency identity"):
            resolve_hip_runtime(self.exe)

    def test_math_source_outside_selected_sdk_and_missing_sdk_root_are_rejected(self):
        manifest = self.bundle()
        other = self.old_sdk / "hipblas.dll"
        other.write_bytes(b"wrong SDK math")
        manifest["dependencies"][0] = {"path": str(other), "sha256": sha(other)}
        self.save_manifest(manifest)
        with self.assertRaisesRegex(HipRuntimeError, "outside the selected SDK"):
            resolve_hip_runtime(self.exe)
        manifest = self.bundle()
        manifest["sdk_root"] = str(self.root / "removed SDK")
        self.save_manifest(manifest)
        with self.assertRaisesRegex(HipRuntimeError, "SDK root is missing"):
            resolve_hip_runtime(self.exe)

    def test_unreadable_legacy_candidate_does_not_fall_through_to_sdk(self):
        with patch.object(hip_runtime, "_present_file", side_effect=PermissionError("access denied")):
            with self.assertRaises(HipRuntimeError):
                resolve_hip_runtime(self.exe, [self.sdk])

    def fake_probe(self, *, wrong_loaded_runtime=False):
        """Run the probe text against fake ctypes functions, never a Windows/HIP DLL."""
        self.bundle()
        dll, comgr = str(self.engine / "amdhip64_7.dll"), str(self.engine / "amd_comgr.dll")
        loads, directories = [], []
        paths = {1: comgr, 2: str(self.old_sdk / "amdhip64_7.dll") if wrong_loaded_runtime else dll}
        def module_path(handle, buffer, size):
            buffer.value = paths[handle]
            return len(buffer.value)
        def count(pointer):
            pointer._obj.value = 1
            return 0
        def name(buffer, size, index):
            buffer.value = b"Discrete fixture"
            return 0
        def total(pointer, index):
            pointer._obj.value = 16 << 30
            return 0
        def attribute(pointer, key, index):
            pointer._obj.value = 0
            return 0
        hip = SimpleNamespace(_handle=2, hipGetDeviceCount=count, hipDeviceGetName=name,
                              hipDeviceTotalMem=total, hipDeviceGetAttribute=attribute, hipDeviceGetLuid=None)
        def load(path, **kwargs):
            loads.append((path, kwargs))
            return SimpleNamespace(_handle=1) if path == comgr else hip
        fake = ModuleType("ctypes")
        fake.__dict__.update(ctypes.__dict__)
        fake.CDLL = load
        fake.WinDLL = lambda *args, **kwargs: SimpleNamespace(GetModuleFileNameW=module_path)
        env = {RUNTIME_ENV: dll, RUNTIME_DIRS_ENV: json.dumps([str(self.engine), str(self.sdk)]),
               COMGR_ENV: json.dumps([comgr]), "PATH": str(self.old_sdk)}
        output = io.StringIO()
        with patch.dict(sys.modules, {"ctypes": fake}), patch.dict(os.environ, env, clear=True), \
                patch.object(os, "add_dll_directory", side_effect=lambda path: directories.append(path), create=True), \
                contextlib.redirect_stdout(output):
            exec(compile(windows_amd_telemetry._HIP_PROBE, "<fake-metadata-only>", "exec"), {})
        return json.loads(output.getvalue()), loads, directories

    def test_probe_preloads_selected_comgr_and_reports_actual_paths_using_fake_ctypes(self):
        cards, loads, directories = self.fake_probe()
        comgr, dll = str(self.engine / "amd_comgr.dll"), str(self.engine / "amdhip64_7.dll")
        self.assertEqual([path for path, _ in loads], [comgr, dll])
        self.assertTrue(all(kwargs["winmode"] == 0x1100 for _, kwargs in loads))
        self.assertEqual(directories, [str(self.engine), str(self.sdk)])
        self.assertNotIn(str(self.old_sdk), directories)
        self.assertEqual(cards[0]["runtime_dll"], dll)
        self.assertEqual(cards[0]["comgr_dlls"], [comgr])
        self.assertIs(cards[0]["integrated"], False)

    def test_probe_refuses_a_different_loaded_module_using_fake_ctypes(self):
        with self.assertRaisesRegex(RuntimeError, "different library"):
            self.fake_probe(wrong_loaded_runtime=True)

    def test_legacy_app_local_then_system_then_matching_sdk(self):
        for directory in (self.engine, self.system, self.sdk):
            (directory / "amdhip64_7.dll").write_bytes(str(directory).encode())
        self.assertEqual(resolve_hip_runtime(self.exe, [self.sdk]).dll, str(self.engine / "amdhip64_7.dll"))
        (self.engine / "amdhip64_7.dll").unlink()
        self.assertEqual(resolve_hip_runtime(self.exe, [self.sdk]).dll, str(self.system / "amdhip64_7.dll"))
        (self.system / "amdhip64_7.dll").unlink()
        self.assertEqual(resolve_hip_runtime(self.exe, [self.sdk]).dll, str(self.sdk / "amdhip64_7.dll"))
        self.assertFalse(resolve_hip_runtime(self.exe, [self.sdk]).managed)

    def test_legacy_multiple_sdks_use_import_basename_not_first_glob(self):
        (self.old_sdk / "amdhip64.dll").write_bytes(b"wrong basename")
        (self.sdk / "amdhip64_7.dll").write_bytes(b"correct basename")
        self.assertEqual(resolve_hip_runtime(self.exe, [self.old_sdk, self.sdk]).dll, str(self.sdk / "amdhip64_7.dll"))
        (self.sdk / "amdhip64_7.dll").unlink()
        with self.assertRaisesRegex(HipRuntimeError, "Cannot find amdhip64_7.dll"):
            resolve_hip_runtime(self.exe, [self.old_sdk, self.sdk])

    def test_pe32_and_malformed_real_pe_and_small_old_fixture(self):
        write_pe(self.exe, "amdhip64.dll", pe32=True)
        self.assertEqual(hip_runtime.hip_imports(self.exe), ("amdhip64.dll",))
        self.exe.write_bytes(b"MZ" + b"\0" * 126)
        with self.assertRaisesRegex(HipRuntimeError, "PE"):
            resolve_hip_runtime(self.exe)
        self.exe.write_bytes(b"amdhip64 fixture")
        self.assertIsNone(resolve_hip_runtime(self.exe))

    @unittest.skipUnless(os.name == "nt", "Windows environment policy")
    def test_environment_uses_manifest_dirs_and_marker_without_mutating_settings(self):
        self.bundle()
        cfg = {"backend": "hip", "exe": str(self.exe), "lib_dirs": [str(self.old_sdk)]}
        env = {"PATH": str(self.old_sdk), RUNTIME_ENV: "stale marker", "HIP_VISIBLE_DEVICES": "1"}
        result = runtime_environment(cfg, env)
        self.assertEqual(result[RUNTIME_ENV], str(self.engine / "amdhip64_7.dll"))
        self.assertEqual(result["PATH"].split(os.pathsep)[:2], [str(self.engine), str(self.sdk)])
        self.assertEqual(json.loads(result[RUNTIME_DIRS_ENV]), [str(self.engine), str(self.sdk)])
        self.assertEqual(json.loads(result[COMGR_ENV]), [str(self.engine / "amd_comgr.dll")])
        self.assertEqual(result["HIP_VISIBLE_DEVICES"], "1")
        self.assertEqual(cfg["lib_dirs"], [str(self.old_sdk)])
        self.assertEqual(env[RUNTIME_ENV], "stale marker")
        self.assertEqual(runtime_environment({**cfg, "backend": "cuda"}, env), env)

    @unittest.skipUnless(os.name == "nt", "Windows environment policy")
    def test_managed_discovery_variables_align_but_explicit_tuning_survives(self):
        manifest = self.bundle()
        manifest["sdk_root"] = str(self.sdk)
        self.save_manifest(manifest)
        env = {name: str(self.old_sdk) for name in hip_runtime._SDK_DISCOVERY}
        env.update(PATH=str(self.old_sdk), STRATA_POOL_FUSED="0", HIP_VISIBLE_DEVICES="1", AMD_SERIALIZE_KERNEL="3")
        result = runtime_environment({"backend": "hip", "exe": str(self.exe)}, env)
        self.assertEqual(result["HIP_PATH"], str(self.sdk))
        self.assertEqual(result["ROCM_PATH"], str(self.sdk))
        for name in ("ROCM_HOME", "HIP_CLANG_PATH", "HIP_DEVICE_LIB_PATH", "LLVM_PATH"):
            self.assertNotIn(name, result)
        for name in ("STRATA_POOL_FUSED", "HIP_VISIBLE_DEVICES", "AMD_SERIALIZE_KERNEL"):
            self.assertEqual(result[name], env[name])
        manifest.pop("sdk_root")
        self.save_manifest(manifest)
        result = runtime_environment({"backend": "hip", "exe": str(self.exe)}, env)
        self.assertFalse(any(name in result for name in hip_runtime._SDK_DISCOVERY))

    @unittest.skipUnless(os.name == "nt", "Windows metadata probe API")
    def test_ordinal_and_cards_use_the_resolved_dll_and_no_real_probe(self):
        self.bundle()
        cards = [{"index": 0, "name": "integrated", "integrated": True, "mem_total": 40 << 30},
                 {"index": 1, "name": "discrete", "integrated": False, "mem_total": 16 << 30}]
        with patch.object(windows_amd_telemetry, "hip_cards", return_value=cards) as probe:
            self.assertEqual(backend.hip_ordinal([self.old_sdk], executable=self.exe), 1)
            self.assertEqual(probe.call_args.args[0][RUNTIME_ENV], str(self.engine / "amdhip64_7.dll"))
            self.assertEqual(backend.hip_ordinal([self.old_sdk]), 1)  # public one-argument API remains valid
        cards[1]["integrated"] = None
        with patch.object(windows_amd_telemetry, "hip_cards", return_value=cards):
            self.assertIsNone(backend.hip_ordinal())
        with patch.dict(os.environ, {"HIP_VISIBLE_DEVICES": "1"}), patch.object(windows_amd_telemetry, "hip_cards") as probe:
            self.assertIsNone(backend.hip_ordinal(executable=self.exe))
            probe.assert_not_called()
        env = runtime_environment({"backend": "hip", "exe": str(self.exe)}, {})
        response = SimpleNamespace(returncode=0, stdout=json.dumps([{**cards[0], "runtime_dll": env[RUNTIME_ENV]}]))
        with patch.object(windows_amd_telemetry.subprocess, "run", return_value=response) as process:
            actual = windows_amd_telemetry.hip_cards(env)
            self.assertEqual(actual[0]["runtime_dll"], env[RUNTIME_ENV])
            self.assertEqual(process.call_args.kwargs["env"][RUNTIME_ENV], env[RUNTIME_ENV])
        compile(windows_amd_telemetry._HIP_PROBE, "<metadata-probe-syntax-only>", "exec")


if __name__ == "__main__":
    unittest.main()
