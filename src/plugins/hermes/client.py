"""HTTP client used by the plugin to communicate with the local service."""

from __future__ import annotations

import json
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib import error, request


_DEFAULT_TIMEOUT_SECONDS = 5.0
_ALLOWED_LOOPBACK_HOSTS = {
    "127.0.0.1",
    "localhost",
    "::1",
    "[::1]",
}


class LedgerMindClientError(RuntimeError):
    """Base client error."""


class LedgerMindNetworkError(LedgerMindClientError):
    """Transport-level or temporary server error from the local service."""


class LedgerMindUnauthorizedError(LedgerMindClientError):
    """Authentication failed against the local service."""


class LedgerMindConflictError(LedgerMindClientError):
    """Upstream returned idempotency conflict for a delivery payload."""


class LedgerMindResponseError(LedgerMindClientError):
    """Unexpected but non-retriable HTTP response from the local service."""


class LedgerMindSchemaError(LedgerMindResponseError):
    """Server payload does not match expected schema."""


@dataclass(frozen=True, slots=True)
class SearchResponse:
    items: list[dict[str, Any]]
    raw: dict[str, Any]


def _read_token(path: Path) -> str | None:
    if not path.exists():
        return None
    payload = path.read_text(encoding="utf-8").strip()
    return payload or None


def _join_path(base: str, suffix: str) -> str:
    return base.rstrip("/") + suffix


def _validate_service_url(url: str, *, allow_remote: bool) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("service_url must start with http:// or https://")

    host = parsed.hostname.lower() if parsed.hostname else ""
    if not host:
        raise ValueError("service_url must include a host")

    if not allow_remote and host not in _ALLOWED_LOOPBACK_HOSTS:
        raise ValueError("service_url must be loopback host")

    return url.rstrip("/")


class LedgerMindClient:
    """Small HTTP helper with token refresh semantics."""

    def __init__(
        self,
        *,
        service_url: str,
        token_file: str,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        allow_remote_service: bool = False,
    ) -> None:
        self._service_url = _validate_service_url(
            service_url,
            allow_remote=allow_remote_service,
        )
        self._token_file = Path(token_file).expanduser()
        self._timeout = timeout
        self._token = _read_token(self._token_file)

    def search_context(
        self,
        *,
        memory_space_id: str,
        query: str,
        limit: int,
        timeout: float | None = None,
    ) -> SearchResponse:
        payload = {
            "api_version": "1",
            "memory_space_id": memory_space_id,
            "query": query,
            "limit": int(limit),
        }
        response = self._request_json(
            method="POST",
            path="/v1/context/search",
            payload=payload,
            timeout=timeout,
            allow_unauthorized_retry=True,
        )
        items = response.get("items")
        if not isinstance(items, list):
            raise LedgerMindSchemaError("search_context response is missing items list")
        return SearchResponse(items=items, raw=response)

    def health(self, *, timeout: float | None = None) -> dict[str, Any]:
        return self._request_json(
            method="GET",
            path="/v1/health/ready",
            payload=None,
            timeout=timeout,
            allow_unauthorized_retry=True,
        )

    def ingest_atom(self, payload: Mapping[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        return self._request_json(
            method="POST",
            path="/v1/atoms",
            payload=dict(payload),
            timeout=timeout,
            allow_unauthorized_retry=True,
        )

    def reload_token(self) -> str | None:
        self._token = _read_token(self._token_file)
        return self._token

    def _request_json(
        self,
        *,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None,
        timeout: float | None,
        allow_unauthorized_retry: bool,
    ) -> dict[str, Any]:
        request_timeout = timeout if timeout is not None else self._timeout
        try:
            return self._request_json_once(
                method=method,
                path=path,
                payload=payload,
                timeout=request_timeout,
            )
        except LedgerMindUnauthorizedError:
            if not allow_unauthorized_retry:
                raise
            self.reload_token()
            return self._request_json_once(
                method=method,
                path=path,
                payload=payload,
                timeout=request_timeout,
            )

    def _request_json_once(
        self,
        *,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None,
        timeout: float,
    ) -> dict[str, Any]:
        if payload is None:
            request_payload: bytes | None = None
        else:
            request_payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        headers = {"accept": "application/json"}
        if request_payload is not None:
            headers["content-type"] = "application/json"
        if self._token:
            headers["authorization"] = f"Bearer {self._token}"

        req = request.Request(
            url=_join_path(self._service_url, path),
            data=request_payload,
            headers=headers,
            method=method,
        )

        try:
            with request.urlopen(req, timeout=timeout) as response:
                status = response.getcode()
                raw_body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            raw_body = exc.read().decode("utf-8", errors="replace")
            status = exc.code
            if status == 401:
                raise LedgerMindUnauthorizedError("authentication failed")
            if status == 409:
                raise LedgerMindConflictError(f"idempotency conflict: {raw_body}")
            if status >= 500:
                raise LedgerMindNetworkError(f"remote server error: {status}")
            raise LedgerMindResponseError(f"unexpected status {status}: {raw_body}")
        except OSError as exc:
            raise LedgerMindNetworkError(str(exc))

        if status not in {200, 201}:
            raise LedgerMindResponseError(f"unexpected status {status}: {raw_body}")

        if not raw_body:
            return {}
        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise LedgerMindSchemaError(f"invalid JSON response: {exc}")

        if not isinstance(data, dict):
            raise LedgerMindSchemaError("response body is not a JSON object")
        return data
