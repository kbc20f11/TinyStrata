"""Local UI chat persistence. Independent implementation using only sqlite3.

The database belongs to the UI process, outside the source/install directory.
This module never reads model profiles or calls the inference API. Browser
imports and their original JSON backup commit together in one transaction.
"""
from __future__ import annotations

import base64
import binascii
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import parse_qs, urlsplit
import uuid

MAX_CHAT_BYTES = 32 * 1024 * 1024
MAX_REQUEST_BYTES = 64 * 1024 * 1024
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
MAX_STORE_BYTES = 1024 * 1024 * 1024
MAX_CHATS = 10000
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
IMAGE = re.compile(r"data:(image/(?:png|jpeg|webp|gif|bmp|avif));base64,([A-Za-z0-9+/=\r\n]+)\Z")


class HistoryError(Exception):
    def __init__(self, status, message, **details):
        self.status, self.message, self.details = status, message, details
        super().__init__(message)


def _json(value):
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        text.encode("utf-8")
        return text
    except (ValueError, TypeError, RecursionError, UnicodeError) as error:
        raise HistoryError(400, "Chat must contain finite JSON values") from error


def _revision(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**53 - 1:
        raise HistoryError(400, "expected_revision must be a nonnegative integer")
    return value


def _id(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise HistoryError(400, "Invalid chat ID")
    return value


def _bounded_tree(value):
    stack = [(value, 0)]; nodes = 0
    while stack:
        item, depth = stack.pop(); nodes += 1
        if depth > 40 or nodes > 200000:
            raise HistoryError(413, "Chat structure is too large")
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise HistoryError(400, "JSON object keys must be strings")
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
        elif item is not None and not isinstance(item, (str, bool, int, float)):
            raise HistoryError(400, "Unsupported chat value")
        elif isinstance(item, float) and not math.isfinite(item):
            raise HistoryError(400, "Nonfinite chat value")


def _prepare(value, *, import_id=False):
    if not isinstance(value, dict):
        raise HistoryError(400, "Expected a chat object")
    _bounded_tree(value)
    # Storage bookkeeping is never part of the conversation/content identity.
    chat = {key: val for key, val in value.items() if key not in
            {"_revision", "_pending", "_deleted", "_local_token", "_server_id"}}
    raw = _json(chat).encode("utf-8")
    if len(raw) > MAX_CHAT_BYTES:
        raise HistoryError(413, "Chat exceeds the 32 MiB limit")
    chat = json.loads(raw)
    if import_id and (not isinstance(chat.get("id"), str) or not ID.fullmatch(chat["id"])):
        chat["id"] = "c" + hashlib.sha256(raw).hexdigest()[:32]
    _id(chat.get("id"))
    if not isinstance(chat.get("title", ""), str) or len(chat.get("title", "")) > 512:
        raise HistoryError(400, "Invalid chat title")
    messages = chat.get("messages")
    if not isinstance(messages, list) or len(messages) > 20000 or any(not isinstance(m, dict) for m in messages):
        raise HistoryError(400, "Expected a bounded message list")
    when = chat.get("time", 0)
    if isinstance(when, bool) or not isinstance(when, (int, float)) or not 0 <= when <= 2**53 - 1 or not math.isfinite(when):
        raise HistoryError(400, "Invalid chat time")
    blobs = {}
    for message in messages:
        for field, content_key in (("images", "url"), ("files", "text")):
            items = message.get(field, [])
            if not isinstance(items, list) or len(items) > 256 or any(not isinstance(item, dict) for item in items):
                raise HistoryError(400, "Invalid attachment list")
            for attachment in items:
                if "_local_blob" in attachment:
                    raise HistoryError(400, "Internal attachment references are not accepted")
                content = attachment.get(content_key)
                if content is None:
                    continue  # Older browser records can have names only.
                if not isinstance(content, str):
                    raise HistoryError(400, "Invalid attachment content")
                if field == "images":
                    match = IMAGE.fullmatch(content)
                    if not match:
                        raise HistoryError(400, "Saved image must be a supported base64 data URL")
                    mime = match[1]
                    try:
                        data = base64.b64decode(match[2].replace("\r", "").replace("\n", ""), validate=True)
                    except (ValueError, binascii.Error) as error:
                        raise HistoryError(400, "Invalid image base64") from error
                else:
                    mime, data = "text/plain;charset=utf-8", content.encode("utf-8")
                if len(data) > MAX_ATTACHMENT_BYTES:
                    raise HistoryError(413, "Attachment exceeds the 20 MiB limit")
                digest = hashlib.sha256(mime.encode() + b"\0" + data).hexdigest()
                blobs[digest] = (mime, data)
                del attachment[content_key]
                attachment["_local_blob"] = digest
    payload = _json(chat)
    fingerprint = hashlib.sha256(payload.encode()).hexdigest()
    return chat, payload, fingerprint, blobs


class HistoryStore:
    """One process/thread-safe SQLite store; open a fresh connection per call."""
    def __init__(self, path, *, max_bytes=MAX_STORE_BYTES, max_chats=MAX_CHATS):
        self.path = Path(path).expanduser().resolve()
        self.max_bytes, self.max_chats = int(max_bytes), int(max_chats)
        if self.max_bytes < 1 or self.max_chats < 1:
            raise ValueError("History limits must be positive")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise HistoryError(503, "Unsupported chat database schema; database was left unchanged")
            db.executescript("""
              CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY, revision INTEGER NOT NULL, updated_ms INTEGER NOT NULL,
                payload TEXT NOT NULL, fingerprint TEXT NOT NULL, deleted INTEGER NOT NULL DEFAULT 0);
              CREATE TABLE IF NOT EXISTS attachments (
                hash TEXT PRIMARY KEY, mime TEXT NOT NULL, data BLOB NOT NULL, size INTEGER NOT NULL);
              CREATE TABLE IF NOT EXISTS chat_attachments (
                chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                hash TEXT NOT NULL REFERENCES attachments(hash), PRIMARY KEY(chat_id,hash));
              CREATE TABLE IF NOT EXISTS browser_imports (
                digest TEXT PRIMARY KEY, source TEXT NOT NULL, created_ms INTEGER NOT NULL,
                original BLOB NOT NULL, receipt TEXT NOT NULL);
              PRAGMA user_version=1;
            """)

    @contextmanager
    def _connect(self, *, write=False):
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA busy_timeout=5000")
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if write:
                db.commit()
        except BaseException:
            if write:
                db.rollback()
            raise
        finally:
            db.close()

    def _hydrate(self, db, row):
        if not row or row["deleted"]:
            return None
        chat = json.loads(row["payload"])
        blobs = {r["hash"]: r for r in db.execute("SELECT a.* FROM attachments a JOIN chat_attachments c ON a.hash=c.hash WHERE c.chat_id=?", (row["id"],))}
        for message in chat["messages"]:
            for field, content_key in (("images", "url"), ("files", "text")):
                for attachment in message.get(field, []):
                    digest = attachment.pop("_local_blob", None)
                    if digest:
                        blob = blobs.get(digest)
                        if blob is None:
                            raise HistoryError(503, "Chat attachment is missing; database was left unchanged")
                        data = bytes(blob["data"])
                        attachment[content_key] = ("data:" + blob["mime"] + ";base64," + base64.b64encode(data).decode()) if field == "images" else data.decode("utf-8")
        chat["_revision"] = row["revision"]
        return chat

    def _limits(self, db):
        count = db.execute("SELECT count(*) FROM chats").fetchone()[0]
        size = db.execute("SELECT coalesce(sum(length(cast(payload AS BLOB))),0) FROM chats").fetchone()[0]
        size += db.execute("SELECT coalesce(sum(size),0) FROM attachments").fetchone()[0]
        size += db.execute("SELECT coalesce(sum(length(original)),0) FROM browser_imports").fetchone()[0]
        if count > self.max_chats or size > self.max_bytes:
            raise HistoryError(507, "Local chat storage limit reached; existing chats were kept")

    def _write(self, db, chat, payload, fingerprint, blobs, revision):
        db.execute("INSERT INTO chats(id,revision,updated_ms,payload,fingerprint,deleted) VALUES(?,?,?,?,?,0) ON CONFLICT(id) DO UPDATE SET revision=excluded.revision,updated_ms=excluded.updated_ms,payload=excluded.payload,fingerprint=excluded.fingerprint,deleted=0",
                   (chat["id"], revision, int(time.time() * 1000), payload, fingerprint))
        db.execute("DELETE FROM chat_attachments WHERE chat_id=?", (chat["id"],))
        for digest, (mime, data) in blobs.items():
            db.execute("INSERT OR IGNORE INTO attachments(hash,mime,data,size) VALUES(?,?,?,?)", (digest, mime, data, len(data)))
            db.execute("INSERT INTO chat_attachments(chat_id,hash) VALUES(?,?)", (chat["id"], digest))

    @staticmethod
    def _gc(db):
        db.execute("DELETE FROM attachments WHERE hash NOT IN (SELECT hash FROM chat_attachments)")

    def put(self, value, expected_revision=0):
        expected_revision = _revision(expected_revision)
        chat, payload, fingerprint, blobs = _prepare(value)
        with self._connect(write=True) as db:
            old = db.execute("SELECT * FROM chats WHERE id=?", (chat["id"],)).fetchone()
            actual = old["revision"] if old else 0
            if actual != expected_revision:
                raise HistoryError(409, "Chat changed in another tab; your local copy was kept", revision=actual, current=self._hydrate(db, old), deleted=bool(old and old["deleted"]))
            self._write(db, chat, payload, fingerprint, blobs, actual + 1)
            self._gc(db); self._limits(db)
            return self._hydrate(db, db.execute("SELECT * FROM chats WHERE id=?", (chat["id"],)).fetchone())

    def remove(self, chat_id, expected_revision):
        chat_id, expected_revision = _id(chat_id), _revision(expected_revision)
        with self._connect(write=True) as db:
            old = db.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
            actual = old["revision"] if old else 0
            if expected_revision != actual:
                raise HistoryError(409, "Chat changed in another tab; deletion was not applied", revision=actual, current=self._hydrate(db, old), deleted=bool(old and old["deleted"]))
            if old is None:
                return {"id": chat_id, "revision": 0, "deleted": True}
            revision = actual + 1
            db.execute("UPDATE chats SET revision=?,payload='',fingerprint='',deleted=1,updated_ms=? WHERE id=?", (revision, int(time.time() * 1000), chat_id))
            db.execute("DELETE FROM chat_attachments WHERE chat_id=?", (chat_id,)); self._gc(db)
            return {"id": chat_id, "revision": revision, "deleted": True}

    def list(self, *, after="", limit=50):
        if after:
            _id(after)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise HistoryError(400, "Invalid history page size")
        chats, size, last = [], 0, after
        with self._connect() as db:
            for row in db.execute("SELECT * FROM chats WHERE deleted=0 AND id>? ORDER BY id LIMIT ?", (after, limit + 1)):
                if len(chats) == limit:
                    return {"chats": chats, "next": last, "storage": "sqlite", "schema_version": 1}
                chat = self._hydrate(db, row); count = len(_json(chat).encode())
                if chats and size + count > MAX_REQUEST_BYTES - 1024:
                    return {"chats": chats, "next": last, "storage": "sqlite", "schema_version": 1}
                chats.append(chat); size += count; last = row["id"]
        return {"chats": chats, "next": None, "storage": "sqlite", "schema_version": 1}

    def import_browser(self, values, *, source="indexeddb-strata-v1"):
        if not isinstance(values, list) or len(values) > MAX_CHATS or not isinstance(source, str) or not 1 <= len(source) <= 128:
            raise HistoryError(400, "Invalid browser import")
        _bounded_tree(values)
        original = _json(values).encode()
        if len(original) > MAX_REQUEST_BYTES:
            raise HistoryError(413, "Browser import exceeds the 64 MiB limit; original browser data was kept")
        digest = hashlib.sha256(original).hexdigest()
        prepared = [_prepare(chat, import_id=True) for chat in values]
        with self._connect(write=True) as db:
            previous = db.execute("SELECT receipt FROM browser_imports WHERE digest=?", (digest,)).fetchone()
            if previous:
                return {**json.loads(previous[0]), "replayed": True}
            mappings, imported, skipped = [], 0, 0
            seen = set()
            for chat, payload, fingerprint, blobs in prepared:
                original_id = chat["id"]
                if original_id in seen:
                    raise HistoryError(400, "Duplicate chat IDs in browser import")
                seen.add(original_id)
                old = db.execute("SELECT * FROM chats WHERE id=?", (original_id,)).fetchone()
                if old and old["deleted"]:
                    mappings.append({"from": original_id, "id": original_id, "revision": old["revision"], "deleted": True}); skipped += 1; continue
                if old and old["fingerprint"] != fingerprint:
                    chat["id"] = "c" + hashlib.sha256((original_id + "\0" + fingerprint).encode()).hexdigest()[:32]
                    payload = _json(chat); fingerprint = hashlib.sha256(payload.encode()).hexdigest()
                    old = db.execute("SELECT * FROM chats WHERE id=?", (chat["id"],)).fetchone()
                    if old and old["fingerprint"] != fingerprint and not old["deleted"]:
                        raise HistoryError(409, "Import ID collision; original browser data was kept")
                if old:
                    skipped += 1
                else:
                    self._write(db, chat, payload, fingerprint, blobs, 1); imported += 1
                mappings.append({"from": original_id, "id": chat["id"], "revision": old["revision"] if old else 1, "deleted": bool(old and old["deleted"])})
            receipt = {"backup_id": digest, "imported": imported, "deduplicated": skipped, "mappings": mappings, "confirmed": True}
            db.execute("INSERT INTO browser_imports VALUES(?,?,?,?,?)", (digest, source, int(time.time() * 1000), original, _json(receipt)))
            self._gc(db); self._limits(db)
            return receipt

    def import_backup(self, digest):
        if not isinstance(digest, str) or not HASH.fullmatch(digest):
            raise HistoryError(400, "Invalid browser backup ID")
        with self._connect() as db:
            row = db.execute("SELECT original FROM browser_imports WHERE digest=?", (digest,)).fetchone()
            if row is None:
                raise HistoryError(404, "Browser backup not found")
            return json.loads(bytes(row[0]))


def _send(handler, status, value):
    data = _json(value).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    if handler.command != "HEAD":
        handler.wfile.write(data)


def handle_history(handler, path):
    """Return False for other paths. Caller must enforce loopback Host/Origin.

    Wire `server.chat_history = HistoryStore(repo_external_db_path)` and call
    this before generic assets/proxy routing for GET/POST/PUT/DELETE/HEAD.
    """
    if path != "/local-ui/chats" and not path.startswith("/local-ui/chats/"):
        return False
    try:
        store = getattr(handler.server, "chat_history", None)
        if store is None:
            raise HistoryError(503, "Local chat storage is unavailable")
        method = handler.command
        body = None
        if method in {"POST", "PUT", "DELETE"}:
            if handler.headers.get("Transfer-Encoding", "identity").lower() != "identity":
                raise HistoryError(400, "Content-Length is required")
            try:
                length = int(handler.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise HistoryError(413 if length > MAX_REQUEST_BYTES else 400, "Invalid history request length")
                if hasattr(handler, "connection"):
                    handler.connection.settimeout(30)
                raw = handler.rfile.read(length)
                if len(raw) != length:
                    raise HistoryError(400, "Incomplete history request")
                body = json.loads(raw)
            except (ValueError, UnicodeError, RecursionError) as error:
                raise HistoryError(400, "Expected JSON history request") from error
            if not isinstance(body, dict):
                raise HistoryError(400, "Expected a history request object")
        parts = path.removeprefix("/local-ui/chats").strip("/").split("/") if path != "/local-ui/chats" else []
        if not parts and method in {"GET", "HEAD"}:
            query = parse_qs(urlsplit(handler.path).query)
            try:
                limit = int(query.get("limit", ["50"])[0])
            except ValueError as error:
                raise HistoryError(400, "Invalid history page size") from error
            result = store.list(after=query.get("after", [""])[0], limit=limit)
        elif not parts and method == "POST":
            chat = body.get("chat")
            if not isinstance(chat, dict):
                raise HistoryError(400, "Expected a chat object")
            chat = {**chat, "id": chat.get("id") or "c" + uuid.uuid4().hex}
            result = {"chat": store.put(chat, body.get("expected_revision", 0))}
        elif parts == ["import"] and method == "POST":
            result = store.import_browser(body.get("chats"), source=body.get("source", "indexeddb-strata-v1"))
        elif len(parts) == 2 and parts[0] == "imports" and method in {"GET", "HEAD"}:
            result = {"chats": store.import_backup(parts[1])}
        elif len(parts) == 1 and method == "PUT":
            chat = body.get("chat")
            if not isinstance(chat, dict) or chat.get("id") != parts[0]:
                raise HistoryError(400, "Chat ID does not match request path")
            result = {"chat": store.put(chat, body.get("expected_revision", 0))}
        elif len(parts) == 1 and method == "DELETE":
            result = store.remove(parts[0], body.get("expected_revision"))
        else:
            raise HistoryError(405, "Unsupported local chat operation")
        _send(handler, 200, result)
    except HistoryError as error:
        if error.status in {400, 405, 413}:
            handler.close_connection = True  # Never interpret unread body as a second HTTP request.
        _send(handler, error.status, {"error": error.message, **error.details})
    except (sqlite3.Error, OSError):
        handler.close_connection = True
        _send(handler, 503, {"error": "Local chat storage is unavailable; browser data was kept"})
    return True
