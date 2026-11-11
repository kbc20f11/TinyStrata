"""Serve TinyStrata's optional browser UI and relay its API over loopback."""
from __future__ import annotations

import argparse
import http.client
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import select
import socket
import sys
import threading
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ui.chat_history import HistoryStore, handle_history
from ui.launch import backend_revision
from ui.sandbox_runtime import RuntimeCache
from ui.tool_settings import ToolSettings
from ui.web_search import SearchError, SearchService

APP_ID = "tinystrata-ui"
WEB = Path(__file__).resolve().parent / "web"
HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
               "te", "trailer", "transfer-encoding", "upgrade", "proxy-connection"}
API_ROUTES = {"/health", "/metrics", "/mcp", "/settings", "/config", "/load", "/unload",
              "/props", "/slots", "/models"}


def port_number(value):
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("expected a port from 1 to 65535")
    return port


class UIServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, port, api_port, data_dir=None):
        self.api_port = api_port
        # Freeze this at startup: edited on-disk sources must not make an old
        # Python process claim that its loaded backend is already up to date.
        self.backend_revision = backend_revision(ROOT)
        self.data_dir = Path(data_dir or ROOT.with_name(ROOT.name + "-work") / "runtime" / "ui").resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.chat_history = HistoryStore(self.data_dir / "chat-history.sqlite3")
        self.tool_settings = ToolSettings(self.data_dir / "search-settings.json")
        self.search = SearchService()
        self.sandbox_runtime = RuntimeCache(self.data_dir / "python-runtime")
        super().__init__(("127.0.0.1", port), Handler)

    def server_close(self):
        try:
            self.search.close()
        finally:
            super().server_close()


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "TinyStrataUI/1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def local_request(self):
        hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if self.headers.get("Host", "").lower() not in hosts:
            self.send_error(403, "Use this UI's loopback address")
            return False
        origin = self.headers.get("Origin", "").rstrip("/").lower()
        if origin and origin not in {"http://" + host for host in hosts}:
            self.send_error(403, "Use this UI's own page")
            return False
        return True

    def dispatch(self):
        if not self.local_request():
            return
        parsed = urlsplit(self.path)
        if parsed.scheme or parsed.netloc:
            self.send_error(400, "Expected a local request path")
            return
        path = parsed.path
        if path.startswith("/local-ui/") or path.startswith("/sandbox-runtime/"):
            self.connection.settimeout(20)
            if self.command in {"POST", "PUT", "PATCH", "DELETE"} and self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                self.close_connection = True
                self.json_response(415, {"error": {"message": "Expected application/json"}})
                return
            try:
                if handle_history(self, path):
                    return
                if self.local_tools(path):
                    return
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.timeout):
                self.close_connection = True
                return
            self.send_error(404)
            return
        if path == "/local-ui-health" and self.command in {"GET", "HEAD"}:
            body = json.dumps({"app": APP_ID, "pid": os.getpid(), "status": "ok",
                               "api_port": self.server.api_port,
                               "backend_revision": self.server.backend_revision,
                               "source_dir": str(ROOT), "data_dir": str(self.server.data_dir),
                               "api": f"http://127.0.0.1:{self.server.api_port}/v1"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        elif path in {"/monitor", "/monitor.html"} and self.command in {"GET", "HEAD"}:
            self.send_response(302)
            self.send_header("Location", "/#monitor")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif path in API_ROUTES or path.startswith(("/v1/", "/api/", "/tool-media/")):
            self.relay()
        elif self.command in {"GET", "HEAD"}:
            self.asset(path)
        else:
            self.send_error(405)

    def json_response(self, status, value, extra_headers=None):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, str(value))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def json_body(self, maximum=131072):
        if self.headers.get("Transfer-Encoding", "identity").lower() != "identity":
            self.close_connection = True
            raise ValueError("Content-Length is required")
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= maximum:
                raise ValueError()
            data = self.rfile.read(length)
            if len(data) != length:
                raise ValueError()
            body = json.loads(data)
        except (ValueError, UnicodeError, RecursionError):
            self.close_connection = True
            raise ValueError("Invalid JSON request") from None
        if not isinstance(body, dict):
            raise ValueError("Expected a JSON object")
        return body

    def cancellable(self, function):
        """Notice a browser abort while a bounded provider/download is running."""
        cancel, box = threading.Event(), {}
        def work():
            try:
                box["value"] = function(cancel)
            except Exception as error:
                box["error"] = error
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        try:
            while worker.is_alive():
                worker.join(0.2)
                if not worker.is_alive():
                    break
                readable, _, _ = select.select([self.connection], [], [], 0)
                if readable and not self.connection.recv(1, socket.MSG_PEEK):
                    self.close_connection = True
                    cancel.set()
                    return None
            if "error" in box:
                raise box["error"]
            return box.get("value")
        finally:
            cancel.set()

    def local_tools(self, path):
        try:
            if path == "/local-ui/settings":
                if self.command in {"GET", "HEAD"}:
                    self.json_response(200, self.server.tool_settings.public())
                elif self.command == "PUT":
                    self.json_response(200, self.server.tool_settings.update(self.json_body()))
                else:
                    self.send_error(405)
            elif path == "/local-ui/search" and self.command == "POST":
                body = self.json_body(4096)
                if set(body) - {"query"}:
                    raise ValueError("Expected only a search query")
                result = self.cancellable(lambda cancel: self.server.search.search(
                    body.get("query"), settings=self.server.tool_settings.read(), cancel=cancel))
                if result is not None:
                    self.json_response(200, result)
            elif path == "/local-ui/sandbox" and self.command in {"GET", "HEAD"}:
                self.json_response(200, self.server.sandbox_runtime.status())
            elif path == "/local-ui/sandbox/prepare" and self.command == "POST":
                self.json_body(256)
                result = self.cancellable(self.server.sandbox_runtime.prepare)
                if result is not None:
                    self.json_response(200, result)
            elif path.startswith("/sandbox-runtime/") and self.command in {"GET", "HEAD"}:
                parts = path.removeprefix("/sandbox-runtime/").split("/")
                target = self.server.sandbox_runtime.asset(*parts) if len(parts) == 2 else None
                if target is None:
                    self.send_error(404)
                else:
                    with target.open("rb") as source:
                        self.send_response(200)
                        self.send_header("Content-Type", "application/wasm" if target.suffix == ".wasm" else self.guess_type(str(target)))
                        self.send_header("Content-Length", str(target.stat().st_size))
                        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
                        self.end_headers()
                        if self.command != "HEAD":
                            self.copyfile(source, self.wfile)
            else:
                return False
        except SearchError as error:
            self.json_response(429 if error.code in {"rate_limited", "busy", "quota_exceeded"} else 400 if error.code == "invalid_query" else 503,
                               {"error": error.to_dict()}, {"Retry-After": error.retry_after} if error.retry_after is not None else None)
        except ValueError as error:
            self.json_response(400, {"error": {"message": str(error)}})
        except (OSError, TimeoutError):
            self.json_response(503, {"error": {"message": "The local tool is unavailable; try again later"}})
        return True

    def asset(self, path):
        if path == "/":
            relative = "index.html"
        elif path == "/sw.js":
            relative = "sw.js"       # Retire the earlier local chat UI's root worker.
        elif path.startswith("/web/"):
            relative = unquote(path[5:])
        elif path.startswith("/fonts/"):
            relative = "fonts/" + unquote(path[7:])
        else:
            self.send_error(404)
            return
        # Open the checked path directly: SimpleHTTPRequestHandler would decode
        # a second time, which could turn an encoded filename into traversal.
        try:
            target = (WEB / relative).resolve()
            if not target.is_relative_to(WEB.resolve()) or not target.is_file():
                self.send_error(404)
                return
            source = target.open("rb")
        except (OSError, ValueError):
            self.send_error(404)
            return
        with source:
            try:
                self.send_response(200)
                self.send_header("Content-Type", self.guess_type(str(target)))
                if path == "/sw.js":
                    self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(os.fstat(source.fileno()).st_size))
                self.end_headers()
                if self.command != "HEAD":
                    self.copyfile(source, self.wfile)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                self.close_connection = True

    def relay(self):
        if self.headers.get("Transfer-Encoding", "identity").lower() != "identity":
            self.send_error(400, "Content-Length is required")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= 64 * 1024 * 1024:
                raise ValueError
        except ValueError:
            self.send_error(400, "Invalid Content-Length")
            return
        self.connection.settimeout(900)
        body = self.rfile.read(length) if length else None
        if body is not None and len(body) != length:
            self.close_connection = True
            return
        nominated = {part.strip().lower() for part in self.headers.get("Connection", "").split(",")}
        blocked = HOP_HEADERS | nominated | {"content-length"}
        headers = {key: value for key, value in self.headers.items() if key.lower() not in blocked}
        # The API's mutation checks require the page's Host and Origin to agree.
        headers["Host"] = self.headers["Host"]
        headers["Connection"] = "close"
        if body is not None:
            headers["Content-Length"] = str(len(body))
        upstream = http.client.HTTPConnection("127.0.0.1", self.server.api_port, timeout=900)
        response, sent = None, False
        try:
            upstream.request(self.command, self.path, body=body, headers=headers)
            response = upstream.getresponse()
            self.send_response(response.status, response.reason)
            nominated = {part.strip().lower() for part in response.getheader("Connection", "").split(",")}
            for key, value in response.getheaders():
                if key.lower() not in HOP_HEADERS | nominated | {"server", "date"}:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            sent = True
            if self.command != "HEAD":
                # read1 returns available SSE bytes rather than waiting for a
                # full buffer, so the UI can update while generation is active.
                while chunk := response.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True
        except (OSError, http.client.HTTPException) as error:
            self.log_error("TinyStrata relay: %s", error)
            if not sent:
                self.send_error(502, "TinyStrata is unavailable; run run-ui.bat")
            self.close_connection = True
        finally:
            if response is not None:
                response.close()
            upstream.close()

    do_GET = do_HEAD = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = dispatch


def main():
    from ui.stdio import configure_utf8_stdio

    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=port_number, default=3000)
    parser.add_argument("--api-port", type=port_number, default=18080)
    parser.add_argument("--data-dir", type=Path, help="Private UI history, settings and Python runtime directory")
    args = parser.parse_args()
    if args.port == args.api_port:
        parser.error("the UI and API must use different ports")
    with UIServer(args.port, args.api_port, args.data_dir) as server:
        print(f"TinyStrata: http://127.0.0.1:{server.server_port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
