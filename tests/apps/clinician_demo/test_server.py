"""Server: routing, status codes and every security rule, over a real socket."""

import http.client
import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from apps.clinician_demo.server import (
    API_HEADER,
    MAX_BODY_BYTES,
    SECURITY_HEADERS,
    allowed_hosts,
    make_server,
    resolve_static,
    static_files,
)
from apps.clinician_demo.service import BadRequestError, NotFoundError


@dataclass(frozen=True)
class Echo:
    """A dataclass payload, to check schema serialization."""

    value: float


class FakeAPI:
    """Implements the DemoAPI protocol with canned answers and failures."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    def meta(self) -> object:
        """Return a payload holding a NaN."""
        return {"ok": True, "nan": float("nan")}

    def gallery(self) -> object:
        """Return an empty gallery."""
        return {"sections": []}

    def patient(self, subject_id: int) -> object:
        """Return a patient, or fail on the magic ids 404 and 500."""
        if subject_id == 404:
            raise NotFoundError("patient 404 is not in the loaded data")
        if subject_id == 500:
            raise RuntimeError("secret internal detail")
        return {"subject_id": subject_id}

    def trace(self, subject_id: int, visit_id: int) -> object:
        """Return a dataclass payload."""
        return Echo(value=visit_id + 0.123456)

    def presets(self) -> object:
        """Return no presets."""
        return []

    def whatif(self, subject_id: int, visit_id: int, body: dict[str, Any]) -> object:
        """Echo the body, or fail when it asks to."""
        self.bodies.append(body)
        if body.get("bad"):
            raise BadRequestError("t_hours must be a number")
        return {"echo": body}

    def evidence(self, subject_id: int, visit_id: int, body: dict[str, Any]) -> object:
        """Return a job."""
        return {"job_id": "ev1"}

    def job(self, job_id: str) -> object:
        """Echo the job id."""
        return {"job_id": job_id}

    def scorecard(self) -> object:
        """Return an empty scorecard."""
        return {"cells": []}


@pytest.fixture
def static_root(tmp_path: Path) -> Path:
    root = tmp_path / "static"
    (root / "js").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html><title>demo</title>")
    (root / "js" / "app.js").write_text("export {};")
    (root / "notes.txt").write_text("not allowlisted")
    (tmp_path / "secret.json").write_text('{"secret": 1}')
    (root / "escape.json").symlink_to(tmp_path / "secret.json")
    return root


@pytest.fixture
def served(static_root: Path) -> Iterator[tuple[int, FakeAPI]]:
    api = FakeAPI()
    server = make_server(api, "127.0.0.1", 0, static_root=static_root)  # type: ignore[arg-type]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, api
    finally:
        server.shutdown()
        server.server_close()


def _request(
    port: int,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    api_header: bool = True,
    host: str | None = None,
    extra: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Host": host or f"127.0.0.1:{port}"}
    if api_header:
        headers[API_HEADER] = "1"
    if body is not None:
        headers["Content-Type"] = "application/json"
    headers.update(extra or {})
    conn.request(method, path, body=body, headers=headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, dict(response.getheaders()), data


def test_api_success_serializes_through_the_schema_layer(
    served: tuple[int, FakeAPI],
) -> None:
    port, _ = served
    status, headers, data = _request(port, "GET", "/api/meta")
    assert status == 200 and headers["Content-Type"] == "application/json"
    assert json.loads(data) == {"ok": True, "nan": None}
    status, _, data = _request(port, "GET", "/api/patients/7/visits/-3/trace")
    assert status == 200 and json.loads(data) == {"value": -2.87654}


def test_every_response_carries_the_security_headers(
    served: tuple[int, FakeAPI],
) -> None:
    port, _ = served
    for path, api in [
        ("/api/meta", True),
        ("/", False),
        ("/api/nope", True),
        ("/api/meta", False),
    ]:
        _, headers, _ = _request(port, "GET", path, api_header=api)
        for name, value in SECURITY_HEADERS.items():
            assert headers.get(name) == value, (path, name)
        assert "Access-Control-Allow-Origin" not in headers


@pytest.mark.parametrize(
    "host", ["evil.example", "evil.example:80", "127.0.0.1:1", "localhost.evil.com"]
)
def test_foreign_host_headers_are_refused_everywhere(
    served: tuple[int, FakeAPI], host: str
) -> None:
    port, _ = served
    assert _request(port, "GET", "/api/meta", host=host)[0] == 403
    assert _request(port, "GET", "/", host=host)[0] == 403


def test_loopback_host_variants_are_accepted(served: tuple[int, FakeAPI]) -> None:
    port, _ = served
    for host in (f"localhost:{port}", f"127.0.0.1:{port}", "localhost"):
        assert _request(port, "GET", "/api/meta", host=host)[0] == 200
    assert allowed_hosts(8765) >= {"localhost:8765", "[::1]:8765", "127.0.0.1"}


def test_api_calls_without_the_custom_header_are_refused(
    served: tuple[int, FakeAPI],
) -> None:
    port, _ = served
    status, _, data = _request(port, "GET", "/api/meta", api_header=False)
    assert status == 403 and "X-Odyssey-Demo" in json.loads(data)["error"]
    assert (
        _request(port, "GET", "/api/meta", extra={API_HEADER: "yes"}, api_header=False)[
            0
        ]
        == 403
    )


def test_status_codes_for_errors(served: tuple[int, FakeAPI]) -> None:
    port, _ = served
    assert _request(port, "GET", "/api/patients/404")[0] == 404
    status, _, data = _request(port, "GET", "/api/patients/500")
    assert status == 500 and "secret" not in data.decode()
    assert _request(port, "GET", "/api/patients/abc")[0] == 404
    assert _request(port, "GET", "/api/nothing")[0] == 404
    assert _request(port, "POST", "/api/meta", body=b"{}")[0] == 405
    assert _request(port, "GET", "/api/patients/1/visits/2/whatif")[0] == 405
    assert _request(port, "DELETE", "/api/meta")[0] == 501  # stdlib: unsupported method


def test_post_bodies_are_validated(served: tuple[int, FakeAPI]) -> None:
    port, api = served
    path = "/api/patients/1/visits/2/whatif"
    status, _, data = _request(
        port, "POST", path, body=json.dumps({"t_hours": 3}).encode()
    )
    assert status == 200 and json.loads(data) == {"echo": {"t_hours": 3}}
    assert _request(port, "POST", path, body=b"not json")[0] == 400
    assert _request(port, "POST", path, body=b"[1, 2]")[0] == 400
    assert _request(port, "POST", path, body=b"\xff\xfe")[0] == 400
    status, _, data = _request(
        port, "POST", path, body=json.dumps({"bad": True}).encode()
    )
    assert status == 400 and json.loads(data)["error"] == "t_hours must be a number"
    big = b'{"x": "' + b"a" * MAX_BODY_BYTES + b'"}'
    assert _request(port, "POST", path, body=big)[0] == 413
    assert (
        _request(port, "POST", path, body=b"", extra={"Content-Length": "0"})[0] == 200
    )
    assert (
        _request(port, "POST", "/api/patients/1/visits/2/evidence", body=b"{}")[0]
        == 202
    )
    assert api.bodies[0] == {"t_hours": 3}


def test_static_files_are_served_with_the_right_types(
    served: tuple[int, FakeAPI],
) -> None:
    port, _ = served
    status, headers, data = _request(port, "GET", "/", api_header=False)
    assert (
        status == 200
        and headers["Content-Type"].startswith("text/html")
        and b"demo" in data
    )
    status, headers, _ = _request(port, "GET", "/static/js/app.js", api_header=False)
    assert status == 200 and headers["Content-Type"].startswith("text/javascript")
    assert _request(port, "GET", "/static/notes.txt", api_header=False)[0] == 404
    assert _request(port, "GET", "/static/../secret.json", api_header=False)[0] == 404
    assert (
        _request(port, "GET", "/static/%2e%2e/secret.json", api_header=False)[0] == 404
    )
    assert _request(port, "GET", "/static/escape.json", api_header=False)[0] == 404
    assert _request(port, "GET", "/etc/passwd", api_header=False)[0] == 404
    assert _request(port, "POST", "/", body=b"{}", api_header=False)[0] == 405


def test_resolve_static_containment(static_root: Path) -> None:
    files = static_files(static_root)
    index = (static_root / "index.html").resolve()
    assert resolve_static("/", files) == index
    assert resolve_static("/index.html", files) == index
    assert resolve_static("/static/index.html", files) == index
    assert resolve_static("/static/js/app.js", files) is not None
    for bad in (
        "/static/",
        "/static/js",
        "/static/../secret.json",
        "/static/escape.json",  # symlink out of the root
        "/static/notes.txt",  # extension not allowlisted
        "/other.js",
        "/static/missing.js",
    ):
        assert resolve_static(bad, files) is None, bad


def test_static_map_is_empty_without_an_index(tmp_path: Path) -> None:
    (tmp_path / "a.css").write_text("x")
    files = static_files(tmp_path)
    assert set(files) == {"/static/a.css"}
    assert resolve_static("/", files) is None


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.com"])
def test_the_server_refuses_to_bind_off_loopback(host: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        make_server(FakeAPI(), host, 0)  # type: ignore[arg-type]


def test_the_packaged_static_dir_has_an_index() -> None:
    from apps.clinician_demo.server import STATIC_DIR  # noqa: PLC0415

    assert (STATIC_DIR / "index.html").is_file()
