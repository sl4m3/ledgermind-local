"""Tests for the Hermes HTTP client wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from ledgermind_local.plugins.hermes.client import (
    LedgerMindClient,
    LedgerMindConflictError,
    LedgerMindNetworkError,
    LedgerMindUnauthorizedError,
    _validate_service_url,
)


class _ScriptedClient(LedgerMindClient):
    def __init__(self, service_url: str, responses: list[Exception | dict[str, object]], token_file: str) -> None:
        super().__init__(service_url=service_url, token_file=token_file, timeout=1.0)
        self._responses = list(responses)
        self.requests = 0
        self.reload_calls = 0

    def reload_token(self) -> str | None:
        self.reload_calls += 1
        return super().reload_token()

    def _request_json_once(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, object] | None,
        timeout: float,
    ) -> dict[str, object]:
        del method, path, payload, timeout
        if not self._responses:
            raise RuntimeError("no scripted responses")
        self.requests += 1
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_validate_service_url_forbids_non_loopback_host() -> None:
    with pytest.raises(ValueError):
        _validate_service_url("https://example.com/api", allow_remote=False)


def test_client_reloads_token_once_after_unauthorized() -> None:
    client = _ScriptedClient(
        service_url="http://127.0.0.1:8000",
        token_file=str(Path("/tmp/unused-token")),
        responses=[
            LedgerMindUnauthorizedError("expired"),
            {"status": "ok"},
        ],
    )

    response = client._request_json(
        method="GET",
        path="/v1/health/ready",
        payload=None,
        timeout=1.0,
        allow_unauthorized_retry=True,
    )

    assert response == {"status": "ok"}
    assert client.requests == 2
    assert client.reload_calls == 1


def test_client_marks_unauthorized_and_stops_after_single_retry() -> None:
    client = _ScriptedClient(
        service_url="http://127.0.0.1:8000",
        token_file=str(Path("/tmp/unused-token")),
        responses=[LedgerMindUnauthorizedError("expired"), LedgerMindUnauthorizedError("still-expired")],
    )

    with pytest.raises(LedgerMindUnauthorizedError):
        client._request_json(
            method="GET",
            path="/v1/health/ready",
            payload=None,
            timeout=1.0,
            allow_unauthorized_retry=True,
        )

    assert client.requests == 2
    assert client.reload_calls == 1


def test_client_propagates_conflict_and_network_errors() -> None:
    client = _ScriptedClient(
        service_url="http://127.0.0.1:8000",
        token_file=str(Path("/tmp/unused-token")),
        responses=[LedgerMindConflictError("conflict"), LedgerMindNetworkError("offline")],
    )

    with pytest.raises(LedgerMindConflictError):
        client._request_json(
            method="POST",
            path="/v1/atoms",
            payload={"value": 1},
            timeout=1.0,
            allow_unauthorized_retry=True,
        )

    with pytest.raises(LedgerMindNetworkError):
        client._request_json(
            method="POST",
            path="/v1/atoms",
            payload={"value": 2},
            timeout=1.0,
            allow_unauthorized_retry=False,
        )
