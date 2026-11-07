"""Installer startup gate fixtures: no services, subprocesses, GPU, or real owners."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import psutil
from serve import install_guard as guard


class InstallationGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tinystrata-install-guard-")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name) / "work"
        self.work.mkdir()
        self.marker = self.work / guard.GUARD_NAME
        self.config = self.work / "runtime/.tinystrata/config-clover.json"
        self.value = {"schema": 1, "pid": 1234, "started_unix": 1000.125, "token": "fixture-private-token"}
        self.environment = mock.patch.dict(os.environ, {guard.TRANSACTION_ENV: ""})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def write_marker(self, value=None):
        self.marker.write_text(json.dumps(self.value if value is None else value), encoding="utf-8")
        return self.marker.read_bytes()

    def test_absent_marker_never_queries_a_process(self):
        with mock.patch.object(guard, "_owner_state") as owner:
            guard.assert_installation_ready(config_path=self.config)
        owner.assert_not_called()

    def test_standard_config_derives_work_and_blocks_active_owner(self):
        before = self.write_marker()
        with mock.patch.object(guard, "_owner_state", return_value="alive") as owner, \
             self.assertRaisesRegex(guard.UpdateInProgressError, "being updated"):
            guard.assert_installation_ready(config_path=self.config)
        owner.assert_called_once_with(1234, 1000.125)
        self.assertEqual(self.marker.read_bytes(), before)

    def test_all_managed_profile_names_and_case_insensitive_shape(self):
        self.write_marker()
        for name in ("config.json", "config-gemma4.json", "config-tiel-coder.json", "CONFIG-LFM2.JSON"):
            with self.subTest(name=name), mock.patch.object(guard, "_owner_state", return_value="alive"), \
                 self.assertRaises(guard.UpdateInProgressError):
                guard.assert_installation_ready(config_path=self.config.with_name(name))

    def test_private_config_has_no_guessed_installation(self):
        self.write_marker()
        for path in (self.work / "bench/clover/config.json", self.work / "config.json",
                     self.config.with_name("settings.json"), None):
            with self.subTest(path=path), mock.patch.object(guard, "_owner_state") as owner:
                guard.assert_installation_ready(config_path=path)
            owner.assert_not_called()

    def test_explicit_work_root_is_authoritative(self):
        self.write_marker()
        with mock.patch.object(guard, "_owner_state", return_value="alive"), \
             self.assertRaises(guard.UpdateInProgressError):
            guard.assert_installation_ready(config_path=self.work / "private/config.json", work_root=self.work)

    def test_only_authorized_configure_child_can_bypass_alive_guard(self):
        before = self.write_marker()
        with mock.patch.object(guard, "_owner_state", return_value="alive"):
            with mock.patch.dict(os.environ, {guard.TRANSACTION_ENV: self.value["token"]}):
                guard.assert_installation_ready(work_root=self.work, allow_installer=True)
                with self.assertRaises(guard.UpdateInProgressError):
                    guard.assert_installation_ready(work_root=self.work)
            for token in ("", "different-token"):
                with mock.patch.dict(os.environ, {guard.TRANSACTION_ENV: token}), self.assertRaises(guard.UpdateInProgressError):
                    guard.assert_installation_ready(work_root=self.work, allow_installer=True)
        self.assertEqual(self.marker.read_bytes(), before)

    def test_dead_or_unknown_owner_blocks_even_with_matching_token(self):
        before = self.write_marker()
        for state, message in (("dead", "interrupted"), ("unknown", "cannot verify")):
            with self.subTest(state=state), mock.patch.object(guard, "_owner_state", return_value=state), \
                 mock.patch.dict(os.environ, {guard.TRANSACTION_ENV: self.value["token"]}), \
                 self.assertRaisesRegex(guard.UpdateInProgressError, message):
                guard.assert_installation_ready(work_root=self.work, allow_installer=True)
            self.assertEqual(self.marker.read_bytes(), before)

    def test_malformed_markers_are_not_removed_or_queried(self):
        values = [[], {}, {**self.value, "schema": True}, {**self.value, "schema": 2},
                  {**self.value, "pid": True}, {**self.value, "pid": 0},
                  {**self.value, "pid": "1234"}, {**self.value, "pid": 1 << 32},
                  {**self.value, "started_unix": None}, {**self.value, "started_unix": True},
                  {**self.value, "started_unix": float("nan")}, {**self.value, "started_unix": -1},
                  {**self.value, "token": ""}, {**self.value, "token": "unsafe\nvalue"},
                  {**self.value, "token": "x" * 257}]
        for value in values:
            before = self.write_marker(value)
            with self.subTest(value=value), mock.patch.object(guard, "_owner_state") as owner, \
                 self.assertRaisesRegex(guard.UpdateInProgressError, "incomplete or invalid"):
                guard.assert_installation_ready(work_root=self.work)
            owner.assert_not_called()
            self.assertEqual(self.marker.read_bytes(), before)

    def test_truncated_invalid_utf8_and_oversized_markers_block(self):
        for data in (b"", b'{"schema":', b"\xff", b"x" * 65537):
            self.marker.write_bytes(data)
            with self.subTest(length=len(data)), self.assertRaises(guard.UpdateInProgressError):
                guard.assert_installation_ready(work_root=self.work)
            self.assertEqual(self.marker.read_bytes(), data)

    def test_unreadable_marker_blocks_without_owner_probe(self):
        self.write_marker()
        with mock.patch.object(Path, "open", side_effect=PermissionError("fixture")), \
             mock.patch.object(guard, "_owner_state") as owner, \
             self.assertRaisesRegex(guard.UpdateInProgressError, "cannot be read"):
            guard.assert_installation_ready(work_root=self.work)
        owner.assert_not_called()

    def test_owner_identity_detects_reused_pid_and_zombie(self):
        process = mock.Mock()
        process.is_running.return_value = True
        process.status.return_value = psutil.STATUS_RUNNING
        process.create_time.return_value = 1000.125
        with mock.patch.object(psutil, "Process", return_value=process):
            self.assertEqual(guard._owner_state(1234, 1000.125), "alive")
            self.assertEqual(guard._owner_state(1234, 999.0), "dead")
            process.status.return_value = psutil.STATUS_ZOMBIE
            self.assertEqual(guard._owner_state(1234, 1000.125), "dead")

    def test_missing_and_unqueryable_process_are_distinguished(self):
        with mock.patch.object(psutil, "Process", side_effect=psutil.NoSuchProcess(1234)):
            self.assertEqual(guard._owner_state(1234, 1000.125), "dead")
        with mock.patch.object(psutil, "Process", side_effect=psutil.AccessDenied(1234)):
            self.assertEqual(guard._owner_state(1234, 1000.125), "unknown")

    def test_ui_blocks_before_creating_state_directory_or_starting_anything(self):
        from ui import launch
        args = argparse.Namespace(api_port=18080, port=3000, model_family="clover", configure_only=False)
        with mock.patch.object(guard, "assert_installation_ready", side_effect=guard.UpdateInProgressError("updating")) as check, \
             self.assertRaises(guard.UpdateInProgressError):
            launch.Launcher(args, Path(self.temporary.name) / "source", self.work)
        check.assert_called_once_with(work_root=self.work, allow_installer=False)
        self.assertFalse((self.work / "runtime").exists())

    def test_ui_configure_only_explicitly_requests_installer_permission(self):
        from ui import launch
        args = argparse.Namespace(api_port=18080, port=3000, model_family="clover", configure_only=True)
        with mock.patch.object(guard, "assert_installation_ready") as check:
            launch.Launcher(args, Path(self.temporary.name) / "source", self.work)
        check.assert_called_once_with(work_root=self.work, allow_installer=True)


if __name__ == "__main__":
    unittest.main()
