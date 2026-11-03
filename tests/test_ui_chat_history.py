"""Isolated SQLite/UI protocol tests. No existing database, API or GPU is used."""
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ui.chat_history import HistoryError, HistoryStore, handle_history, MAX_REQUEST_BYTES


def chat(id="cfixture", text="hello"):
    return {"id": id, "title": "", "time": 1234, "messages": [{"role": "user", "text": text}]}


class Handler:
    def __init__(self, store, method="GET", path="/local-ui/chats", body=None, headers=None):
        data = json.dumps(body).encode() if body is not None else b""
        self.server = SimpleNamespace(chat_history=store); self.command = method; self.path = path
        self.headers = {"Content-Length": str(len(data)), **(headers or {})}
        self.rfile, self.wfile = io.BytesIO(data), io.BytesIO(); self.close_connection = False
        self.response_headers = {}
    def send_response(self, status): self.status = status
    def send_header(self, key, value): self.response_headers[key] = value
    def end_headers(self): pass
    def result(self): return json.loads(self.wfile.getvalue())


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="tinystrata-history-test-")
        self.path = Path(self.directory.name) / "history.sqlite3"
        self.store = HistoryStore(self.path)
    def tearDown(self): self.directory.cleanup()

    def test_wal_roundtrip_and_optimistic_revision(self):
        first = self.store.put(chat()); self.assertEqual(first["_revision"], 1)
        self.assertEqual(self.store.list()["chats"], [first])
        second = self.store.put(chat(text="second"), 1); self.assertEqual(second["_revision"], 2)
        with self.assertRaises(HistoryError) as cm: self.store.put(chat(text="stale"), 1)
        self.assertEqual(cm.exception.status, 409)
        self.assertEqual(cm.exception.details["current"], second)
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("PRAGMA journal_mode").fetchone()[0], "wal")

    def test_concurrent_writers_never_silently_overwrite(self):
        self.store.put(chat())
        def write(text):
            try: return self.store.put(chat(text=text), 1)
            except HistoryError as error: return error.status
        with ThreadPoolExecutor(2) as pool: results = list(pool.map(write, ["A", "B"]))
        self.assertEqual(sum(isinstance(value, dict) for value in results), 1)
        self.assertIn(409, results)
        self.assertEqual(self.store.list()["chats"][0]["_revision"], 2)

    def test_attachments_survive_restart_and_are_deduplicated(self):
        value = chat(); value["messages"][0].update(images=[{"name": "x.png", "url": "data:image/png;base64," + base64.b64encode(b"fixture pixels").decode()}], files=[{"name": "note.txt", "text": "日本語\ncontent"}])
        a = self.store.put(value); other = {**value, "id": "cother"}; self.store.put(other)
        self.assertEqual(HistoryStore(self.path).list()["chats"][0]["messages"], a["messages"])
        with closing(sqlite3.connect(self.path)) as db: self.assertEqual(db.execute("SELECT count(*) FROM attachments").fetchone()[0], 2)
        self.store.remove(value["id"], 1)
        with closing(sqlite3.connect(self.path)) as db: self.assertEqual(db.execute("SELECT count(*) FROM attachments").fetchone()[0], 2)
        self.store.remove("cother", 1)
        with closing(sqlite3.connect(self.path)) as db: self.assertEqual(db.execute("SELECT count(*) FROM attachments").fetchone()[0], 0)

    def test_atomic_import_backup_receipt_and_retry_dedup(self):
        original = [chat(), chat("csecond", "another")]
        receipt = self.store.import_browser(original)
        self.assertTrue(receipt["confirmed"]); self.assertEqual(receipt["imported"], 2)
        self.assertEqual(self.store.import_backup(receipt["backup_id"]), original)
        again = self.store.import_browser(original)
        self.assertTrue(again["replayed"]); self.assertEqual(len(self.store.list()["chats"]), 2)
        with closing(sqlite3.connect(self.path)) as db: self.assertEqual(db.execute("SELECT count(*) FROM browser_imports").fetchone()[0], 1)

    def test_conflicting_import_preserves_both_and_retries_without_duplicates(self):
        self.store.put(chat(text="server version"))
        receipt = self.store.import_browser([chat(text="browser version")])
        self.assertNotEqual(receipt["mappings"][0]["id"], "cfixture")
        self.assertEqual(len(self.store.list()["chats"]), 2)
        self.store.import_browser([chat(text="browser version")])
        self.assertEqual(len(self.store.list()["chats"]), 2)
        self.assertEqual({c["messages"][0]["text"] for c in self.store.list()["chats"]}, {"server version", "browser version"})

    def test_import_over_quota_is_fully_rolled_back(self):
        limited = HistoryStore(Path(self.directory.name) / "limited.sqlite3", max_bytes=100)
        with self.assertRaises(HistoryError) as cm: limited.import_browser([chat(), chat("csecond")])
        self.assertEqual(cm.exception.status, 507); self.assertEqual(limited.list()["chats"], [])
        with closing(sqlite3.connect(limited.path)) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM browser_imports").fetchone()[0], 0)

    def test_deleted_chats_are_not_resurrected_by_stale_browser(self):
        self.store.put(chat()); self.store.remove("cfixture", 1)
        receipt = self.store.import_browser([chat()])
        self.assertTrue(receipt["mappings"][0]["deleted"]); self.assertEqual(self.store.list()["chats"], [])
        with self.assertRaises(HistoryError) as cm: self.store.put(chat(), 0)
        self.assertEqual(cm.exception.status, 409)
        self.assertEqual(self.store.put(chat(), 2)["_revision"], 3)  # explicit Undo uses the tombstone revision.

    def test_safe_ids_bad_json_and_internal_refs_rejected(self):
        for bad in ("../escape", "a/b", "%2f", "", None):
            with self.subTest(bad=bad), self.assertRaises(HistoryError): self.store.put(chat(bad))
        for value in ({**chat(), "time": 10**1000}, {**chat(), "title": "\ud800"}, {**chat(), "time": float("nan")}):
            with self.assertRaises(HistoryError): self.store.put(value)
        value = chat(); value["messages"][0]["images"] = [{"_local_blob": "a" * 64}]
        with self.assertRaises(HistoryError): self.store.put(value)
        self.assertEqual(self.store.list()["chats"], [])

    def test_pagination_returns_every_chat_once(self):
        for index in range(5): self.store.put(chat(f"c{index}"))
        after = ""; ids = []
        while True:
            page = self.store.list(after=after, limit=2); ids.extend(c["id"] for c in page["chats"])
            if not page["next"]: break
            after = page["next"]
        self.assertEqual(ids, ["c0", "c1", "c2", "c3", "c4"])

    def test_endpoint_delegation_and_conflict_status(self):
        h = Handler(self.store, "PUT", "/local-ui/chats/cfixture", {"chat": chat(), "expected_revision": 0})
        self.assertTrue(handle_history(h, h.path)); self.assertEqual(h.status, 200)
        h = Handler(self.store, "PUT", "/local-ui/chats/cfixture", {"chat": chat(), "expected_revision": 0})
        handle_history(h, h.path); self.assertEqual(h.status, 409); self.assertEqual(h.result()["revision"], 1)
        h = Handler(self.store, "POST", body={"chat": {"title": "", "time": 0, "messages": []}})
        handle_history(h, h.path); self.assertEqual(h.status, 200); self.assertTrue(h.result()["chat"]["id"].startswith("c"))
        self.assertFalse(handle_history(h, "/health"))

    def test_oversized_endpoint_body_is_rejected_without_read(self):
        h = Handler(self.store, "PUT", "/local-ui/chats/cfixture", headers={"Content-Length": str(MAX_REQUEST_BYTES + 1)})
        handle_history(h, h.path)
        self.assertEqual(h.status, 413); self.assertTrue(h.close_connection)
        self.assertEqual(h.rfile.tell(), 0)


if __name__ == "__main__": unittest.main()
