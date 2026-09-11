"""A small, locked-down HTTP adapter over the demo service.

Standard library only (the GPU host's environment is pinned; no web
framework is installed there). The server serves patient-level data, so it
is deliberately strict:

- binds to loopback only and is reached through an SSH tunnel;
- rejects any ``Host`` header other than the loopback names it was bound
  for (defends against DNS rebinding from a page in another tab);
- API calls must carry ``X-Odyssey-Demo: 1``, which a cross-site form
  cannot set, and no CORS headers are ever sent;
- every response is ``Cache-Control: no-store`` (patient JSON must not land
  in the laptop's browser cache) with a same-origin CSP;
- static files are served from an allowlisted directory with path
  containment, nothing else on disk is reachable.
"""

import json
import logging
import re
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit

from apps.clinician_demo.config import LOOPBACK_HOSTS
from apps.clinician_demo.schemas import to_jsonable


logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
API_HEADER = "X-Odyssey-Demo"
MAX_BODY_BYTES = 64 * 1024
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json",
    ".png": "image/png",
    ".ico": "image/x-icon",
}
SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
    ),
}


class DemoAPI(Protocol):
    """What the server needs from the service (a fake implements it in tests)."""

    def meta(self) -> object:
        """Return the static facts the UI needs."""

    def gallery(self) -> object:
        """Return the curated gallery."""

    def patient(self, subject_id: int) -> object:
        """Return one patient's header facts and admissions."""

    def trace(self, subject_id: int, visit_id: int) -> object:
        """Return the replay view of one visit."""

    def presets(self) -> object:
        """Return the what-if controls."""

    def whatif(self, subject_id: int, visit_id: int, body: dict[str, Any]) -> object:
        """Compare the factual and edited forecast."""

    def evidence(self, subject_id: int, visit_id: int, body: dict[str, Any]) -> object:
        """Start or reuse an evidence job."""

    def job(self, job_id: str) -> object:
        """Return an evidence job's state."""

    def scorecard(self) -> object:
        """Return the model's report card."""


class HttpError(Exception):
    """An error with an HTTP status and a message safe to show the user."""

    def __init__(self, status: HTTPStatus, message: str) -> None:
        """Carry the status and message."""
        super().__init__(message)
        self.status = status
        self.message = message


Handler = Callable[[DemoAPI, re.Match[str], dict[str, Any]], tuple[HTTPStatus, object]]


def _ids(match: re.Match[str]) -> tuple[int, int]:
    return int(match["sid"]), int(match["vid"])


ROUTES: list[tuple[str, re.Pattern[str], Handler]] = [
    ("GET", re.compile(r"/api/meta"), lambda api, m, b: (HTTPStatus.OK, api.meta())),
    (
        "GET",
        re.compile(r"/api/gallery"),
        lambda api, m, b: (HTTPStatus.OK, api.gallery()),
    ),
    (
        "GET",
        re.compile(r"/api/whatif/presets"),
        lambda api, m, b: (HTTPStatus.OK, api.presets()),
    ),
    (
        "GET",
        re.compile(r"/api/scorecard"),
        lambda api, m, b: (HTTPStatus.OK, api.scorecard()),
    ),
    (
        "GET",
        re.compile(r"/api/patients/(?P<sid>\d{1,18})"),
        lambda api, m, b: (HTTPStatus.OK, api.patient(int(m["sid"]))),
    ),
    (
        "GET",
        re.compile(r"/api/patients/(?P<sid>\d{1,18})/visits/(?P<vid>-?\d{1,18})/trace"),
        lambda api, m, b: (HTTPStatus.OK, api.trace(*_ids(m))),
    ),
    (
        "POST",
        re.compile(
            r"/api/patients/(?P<sid>\d{1,18})/visits/(?P<vid>-?\d{1,18})/whatif"
        ),
        lambda api, m, b: (HTTPStatus.OK, api.whatif(*_ids(m), b)),
    ),
    (
        "POST",
        re.compile(
            r"/api/patients/(?P<sid>\d{1,18})/visits/(?P<vid>-?\d{1,18})/evidence"
        ),
        lambda api, m, b: (HTTPStatus.ACCEPTED, api.evidence(*_ids(m), b)),
    ),
    (
        "GET",
        re.compile(r"/api/jobs/(?P<job>[A-Za-z0-9_-]{1,32})"),
        lambda api, m, b: (HTTPStatus.OK, api.job(m["job"])),
    ),
]


def resolve_static(path: str, root: Path = STATIC_DIR) -> Path | None:
    """Return the file under ``root`` a URL path names, or ``None`` if unservable.

    ``/`` is ``index.html``; everything else must live under ``/static/``,
    resolve inside ``root`` (no ``..`` or symlink escapes), exist, and have
    an allowlisted extension.
    """
    if path in ("/", "/index.html"):
        relative = "index.html"
    elif path.startswith("/static/"):
        relative = unquote(path[len("/static/") :])
    else:
        return None
    root = root.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        return None
    return candidate if candidate.suffix in CONTENT_TYPES else None


def allowed_hosts(port: int) -> frozenset[str]:
    """Return the ``Host`` header values a loopback server on ``port`` accepts."""
    names = {"localhost", "127.0.0.1", "[::1]"}
    return frozenset({*names, *(f"{n}:{port}" for n in names)})


class DemoRequestHandler(BaseHTTPRequestHandler):
    """Route requests to the API or the static directory."""

    server_version = "OdysseyDemo"
    sys_version = ""
    api: DemoAPI
    static_root: Path = STATIC_DIR

    def do_GET(self) -> None:  # noqa: N802 -- stdlib hook name
        """Handle GET."""
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802 -- stdlib hook name
        """Handle POST."""
        self._dispatch("POST")

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 -- stdlib signature
        """Route access logs through :mod:`logging` (they stay on the host)."""
        logger.info("%s %s", self.address_string(), format % args)

    def _dispatch(self, method: str) -> None:
        try:
            port = int(getattr(self.server, "server_port", 0))
            if self.headers.get("Host", "") not in allowed_hosts(port):
                raise HttpError(HTTPStatus.FORBIDDEN, "unexpected Host header")
            path = urlsplit(self.path).path
            if path.startswith("/api/"):
                self._api(method, path)
            elif method == "GET":
                self._static(path)
            else:
                raise HttpError(HTTPStatus.METHOD_NOT_ALLOWED, "method not allowed")
        except HttpError as err:
            self._send_json(err.status, {"error": err.message})

    def _api(self, method: str, path: str) -> None:
        if self.headers.get(API_HEADER) != "1":
            raise HttpError(HTTPStatus.FORBIDDEN, f"missing {API_HEADER} header")
        path_matched = False
        for route_method, pattern, handler in ROUTES:
            match = pattern.fullmatch(path)
            if match is None:
                continue
            path_matched = True
            if route_method != method:
                continue
            body = self._read_body() if method == "POST" else {}
            status, payload = self._call(handler, match, body)
            self._send_json(status, to_jsonable(payload))
            return
        if path_matched:
            raise HttpError(HTTPStatus.METHOD_NOT_ALLOWED, "method not allowed")
        raise HttpError(HTTPStatus.NOT_FOUND, "no such endpoint")

    def _call(
        self, handler: Handler, match: re.Match[str], body: dict[str, Any]
    ) -> tuple[HTTPStatus, object]:
        # Imported here only for the exception types, to keep this module
        # importable (and testable) without torch.
        from apps.clinician_demo.service import (  # noqa: PLC0415
            BadRequestError,
            NotFoundError,
        )

        try:
            return handler(self.api, match, body)
        except NotFoundError as exc:
            raise HttpError(HTTPStatus.NOT_FOUND, str(exc)) from exc
        except BadRequestError as exc:
            raise HttpError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        except Exception as exc:
            logger.exception("[server] %s failed", self.path)
            raise HttpError(
                HTTPStatus.INTERNAL_SERVER_ERROR, "internal error; see the server log"
            ) from exc

    def _read_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise HttpError(HTTPStatus.BAD_REQUEST, "bad Content-Length") from exc
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise HttpError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body too large"
            )
        try:
            body = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HttpError(HTTPStatus.BAD_REQUEST, "body must be JSON") from exc
        if not isinstance(body, dict):
            raise HttpError(HTTPStatus.BAD_REQUEST, "body must be a JSON object")
        return body

    def _static(self, path: str) -> None:
        file = resolve_static(path, self.static_root)
        if file is None:
            raise HttpError(HTTPStatus.NOT_FOUND, "not found")
        data = file.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", CONTENT_TYPES[file.suffix])
        self.send_header("Content-Length", str(len(data)))
        self._security_headers()
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        data = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self._security_headers()
        self.end_headers()
        self.wfile.write(data)

    def _security_headers(self) -> None:
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)


def make_server(
    api: DemoAPI,
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    static_root: Path = STATIC_DIR,
) -> ThreadingHTTPServer:
    """Create a threaded loopback server for ``api`` and ``static_root``.

    Raises
    ------
    ValueError
        If ``host`` is not a loopback address.
    """
    if host not in LOOPBACK_HOSTS:
        raise ValueError(
            f"refusing to bind {host!r}: the demo serves patient data on loopback only"
        )
    handler = type(
        "BoundDemoHandler",
        (DemoRequestHandler,),
        {"api": api, "static_root": static_root},
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


__all__ = [
    "API_HEADER",
    "CONTENT_TYPES",
    "ROUTES",
    "SECURITY_HEADERS",
    "STATIC_DIR",
    "DemoAPI",
    "DemoRequestHandler",
    "HttpError",
    "allowed_hosts",
    "make_server",
    "resolve_static",
]
