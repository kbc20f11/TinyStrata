"""Serve TinyStrata's optional browser UI and relay its API over loopback."""
from __future__ import annotations

import argparse
import http.client
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from urllib.parse import unquote, urlsplit

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

    def __init__(self, port, api_port):
        self.api_port = api_port
        super().__init__(("127.0.0.1", port), Handler)


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
        if path == "/local-ui-health" and self.command in {"GET", "HEAD"}:
            body = json.dumps({"app": APP_ID, "pid": os.getpid(), "status": "ok",
                               "api_port": self.server.api_port,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=port_number, default=3000)
    parser.add_argument("--api-port", type=port_number, default=18080)
    args = parser.parse_args()
    if args.port == args.api_port:
        parser.error("the UI and API must use different ports")
    with UIServer(args.port, args.api_port) as server:
        print(f"TinyStrata: http://127.0.0.1:{server.server_port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
