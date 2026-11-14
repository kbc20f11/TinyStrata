"""CPU-only staging contracts using tiny synthetic PE files and SDK fixtures."""
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "cmake" / "stage-intel-runtime.ps1"


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def pe(path, imports=(), delayed=()):
    data = bytearray(8192)
    struct.pack_into("<H", data, 0, 0x5A4D)
    struct.pack_into("<I", data, 60, 0x80)
    struct.pack_into("<IHHIIIHH", data, 0x80, 0x4550, 0x8664, 1, 0, 0, 0, 240, 0x22)
    optional = 0x80 + 24
    struct.pack_into("<H", data, optional, 0x20B)
    struct.pack_into("<Q", data, optional + 24, 0x180000000)
    struct.pack_into("<I", data, optional + 108, 16)
    struct.pack_into("<IIII", data, optional + 240 + 8, 7680, 0x1000, 7680, 512)
    name_offset = 0xA00
    for names, directory_offset, table_offset, stride, name_field in (
            (imports, 120, 0x200, 20, 12), (delayed, 216, 0x400, 32, 4)):
        if not names:
            continue
        struct.pack_into("<II", data, optional + directory_offset, table_offset + 0xE00, (len(names) + 1) * stride)
        for index, name in enumerate(names):
            if stride == 32:
                struct.pack_into("<I", data, table_offset + index * stride, 1)
            struct.pack_into("<I", data, table_offset + index * stride + name_field, name_offset + 0xE00)
            encoded = name.encode("ascii") + b"\0"
            data[name_offset:name_offset + len(encoded)] = encoded
            name_offset += len(encoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@unittest.skipUnless(os.name == "nt" and shutil.which("powershell.exe"), "requires Windows PowerShell")
class IntelRuntimeStage(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="tinystrata-stage-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.compiler, self.mkl, self.tbb = (self.root / name for name in ("compiler", "mkl", "tbb"))
        self.umf, self.tcm = self.root / "umf", self.root / "tcm"
        umf_docs = self.umf / "share" / "doc" / "umf"
        (umf_docs / "licensing").mkdir(parents=True)
        (umf_docs / "LICENSE.TXT").write_text("Apache License with LLVM Exceptions")
        (umf_docs / "licensing" / "third-party-programs.txt").write_text("UMF third-party notices")
        self.system = self.root / "system"
        self.system.mkdir()
        (self.system / "ze_loader.dll").write_text("external system loader; must not be copied")
        self.exe = self.root / "build" / "tinystrata-intel.exe"
        self.destination = self.root / "stage"
        self.names = ["sycl9.dll", "ur_win_proxy_loader.dll", "ur_loader_shared.dll", "ur_adapter_level_zero.dll",
                      "ur_adapter_level_zero_v2.dll", "umf.dll", "sycl-jit.dll"]
        for name in self.names:
            pe(self.compiler / "bin" / name)
        licensing = self.compiler / "share" / "doc" / "compiler" / "licensing" / "c"
        licensing.mkdir(parents=True)
        (licensing / "LICENSE.rtf").write_text("compiler license fixture")
        (licensing / "third-party-programs.txt").write_text("compiler notices fixture")
        redist = licensing.parent.parent / "credist.txt"
        redist.write_text("\n".join("<installdir>/bin/" + name for name in self.names))
        pe(self.exe, ["sycl9.dll", "ze_loader.dll"])

    def powershell(self, command):
        return subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                               "-Command", "$ErrorActionPreference='Stop'; . " + quote(SCRIPT) +
                               " -ExecutablePath unused -Destination unused; " + command],
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)

    def plan_command(self):
        return "Get-IntelRuntimePlan " + " ".join(
            "-" + key + " " + quote(value) for key, value in (
                ("ExecutablePath", self.exe), ("Destination", self.destination), ("CompilerRoot", self.compiler),
                ("MklRoot", self.mkl), ("TbbRoot", self.tbb), ("TcmRoot", self.tcm),
                ("UmfRoot", self.umf), ("SystemDirectory", self.system)))

    def plan(self):
        result = self.powershell(self.plan_command() + " | ConvertTo-Json -Depth 8")
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_normal_and_delay_imports_are_read_without_loading_image(self):
        pe(self.exe, ["sycl9.dll"], ["mkl_tbb_thread.3.dll"])
        result = self.powershell("@(Get-IntelPeImports " + quote(self.exe) + ") | ConvertTo-Json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(json.loads(result.stdout)), {"sycl9.dll", "mkl_tbb_thread.3.dll"})

    def test_plan_includes_dynamic_adapters_and_not_system_loader(self):
        plan = self.plan()
        names = {file["RelativePath"] for file in plan["Files"]}
        self.assertTrue(set(self.names).issubset(names))
        self.assertIn("licenses/compiler/credist.txt", names)
        self.assertNotIn("ze_loader.dll", names)
        self.assertFalse(self.destination.exists())
        self.assertFalse(plan["DriverIncluded"])

    def test_manifest_records_payload_size_when_sdk_alias_metadata_is_zero(self):
        # Windows FileInfo reports Length=0 for the installed SDK's umf.dll
        # symbolic link. Emulate that metadata without requiring link privileges;
        # hashing/copying still see the actual nonempty payload, as in the SDK.
        alias = self.compiler / "bin" / "umf.dll"
        command = """
function Get-Item {
    [CmdletBinding()] param([string]$LiteralPath, [switch]$Force)
    $item = Microsoft.PowerShell.Management\\Get-Item @PSBoundParameters
    if ($LiteralPath -eq ALIAS) {
        return [pscustomobject]@{FullName=$item.FullName; Name=$item.Name;
            Length=0; VersionInfo=$item.VersionInfo}
    }
    return $item
}
""".replace("ALIAS", quote(alias))
        result = self.powershell(command + self.plan_command() + " | ConvertTo-Json -Depth 8")
        self.assertEqual(result.returncode, 0, result.stderr)
        entry = next(x for x in json.loads(result.stdout)["Files"] if x["RelativePath"] == "umf.dll")
        self.assertEqual(entry["Bytes"], alias.stat().st_size)
        self.assertGreater(entry["Bytes"], 0)

    def test_mkl_dynamic_dispatch_and_tbb_are_included(self):
        for root, component, filename in ((self.mkl, "mkl", "license.rtf"), (self.tbb, "tbb", "license.txt")):
            licensing = root / "share" / "doc" / component / "licensing"
            licensing.mkdir(parents=True)
            (licensing / filename).write_text("Intel Simplified Software License")
        pe(self.exe, ["sycl9.dll", "ze_loader.dll", "mkl_sycl_blas.6.dll"])
        pe(self.mkl / "bin" / "mkl_sycl_blas.6.dll", ["sycl9.dll", "mkl_core.3.dll"], ["mkl_tbb_thread.3.dll"])
        pe(self.mkl / "bin" / "mkl_core.3.dll")
        pe(self.mkl / "bin" / "mkl_tbb_thread.3.dll", delayed=["tbb12.dll"])
        pe(self.mkl / "bin" / "mkl_avx2.3.dll")
        pe(self.mkl / "bin" / "mkl_vml_avx512.3.dll")
        pe(self.tbb / "bin" / "tbb12.dll")
        names = {file["RelativePath"] for file in self.plan()["Files"]}
        self.assertTrue({"mkl_tbb_thread.3.dll", "mkl_avx2.3.dll", "mkl_vml_avx512.3.dll", "tbb12.dll"}.issubset(names))

    def test_sdk_dll_not_in_redist_list_is_refused(self):
        pe(self.compiler / "bin" / "unlisted.dll")
        pe(self.exe, ["sycl9.dll", "ze_loader.dll", "unlisted.dll"])
        result = self.powershell(self.plan_command())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unresolved dependency", result.stderr)
        self.assertFalse(self.destination.exists())

    def test_selected_app_local_vc_runtime_closes_imports_and_keeps_notice(self):
        runtime = self.root / "vc-runtime"
        for name in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140_atomic_wait.dll"):
            pe(runtime / name)
        pe(runtime / "vcruntime140d.dll")
        pe(runtime / "unlisted.dll")
        notice = self.root / "vc-redistribution.txt"
        notice.write_text("Microsoft redistributable package provenance fixture")
        pe(self.exe, ["sycl9.dll", "ze_loader.dll", "msvcp140.dll", "vcruntime140.dll"])
        command = self.plan_command() + " -VcRuntimeDirectory " + quote(runtime) + " -VcRedistributionNotice " + quote(notice)
        result = self.powershell("$plan = " + command + "; Write-IntelRuntimeStage $plan")
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads((self.destination / ".tinystrata-intel-runtime.json").read_text(encoding="utf-8"))
        names = {file["RelativePath"] for file in plan["Files"]}
        self.assertTrue({"msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140_atomic_wait.dll",
                         "licenses/microsoft-vc/redistribution.txt"}.issubset(names))
        self.assertNotIn("vcruntime140d.dll", names)
        self.assertNotIn("unlisted.dll", names)
        self.assertNotIn("msvcp140.dll", {x["Name"] for x in plan["ExternalPrerequisites"]})

    def test_app_local_vc_runtime_requires_redistribution_notice(self):
        result = self.powershell(self.plan_command() + " -VcRuntimeDirectory " + quote(self.root / "vc-runtime"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires both", result.stderr)
        self.assertFalse(self.destination.exists())

    def test_umf_hwloc_dependency_requires_selected_tcm_notices(self):
        pe(self.compiler / "bin" / "umf.dll", ["libhwloc-15.dll"])
        licensing = self.tcm / "share" / "doc" / "tcm" / "licensing"
        licensing.mkdir(parents=True)
        (licensing / "license.txt").write_text("Intel Simplified Software License")
        (licensing / "third-party-programs.txt").write_text("Portable Hardware Locality (hwloc) notices")
        pe(self.tcm / "bin" / "libhwloc-15.dll")
        names = {file["RelativePath"] for file in self.plan()["Files"]}
        self.assertIn("libhwloc-15.dll", names)
        self.assertIn("licenses/tcm/license.txt", names)
        self.assertIn("licenses/umf/LICENSE.TXT", names)

    def test_existing_user_files_are_preserved(self):
        self.destination.mkdir()
        marker = self.destination / "history.sqlite"
        marker.write_bytes(b"existing user data")
        result = self.powershell(self.plan_command())
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(marker.read_bytes(), b"existing user data")

    def test_stage_copies_only_fixture_inventory_and_publishes_hash_manifest(self):
        result = self.powershell("$plan = " + self.plan_command() + "; Write-IntelRuntimeStage $plan")
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads((self.destination / ".tinystrata-intel-runtime.json").read_text(encoding="utf-8"))
        self.assertEqual(plan["State"], "staged")
        self.assertFalse((self.destination / "ze_loader.dll").exists())
        for file in plan["Files"]:
            self.assertEqual((self.destination / file["RelativePath"]).stat().st_size, file["Bytes"])


if __name__ == "__main__":
    unittest.main()
