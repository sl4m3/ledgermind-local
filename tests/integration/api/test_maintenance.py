from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ledgermind_local.api.maintenance import create_maintenance_router


class _Runtime:
    def __init__(self) -> None:
        self.limits: list[int] = []

    def retry_failed_user_semantic(self, *, limit: int) -> dict[str, object]:
        self.limits.append(limit)
        return {
            "status": "completed",
            "requeued_normalization_commands": 2,
            "retried_failed_user_semantic": 3,
        }


def test_replay_failed_is_authenticated_explicit_and_bounded() -> None:
    runtime = _Runtime()
    app = FastAPI()
    app.include_router(create_maintenance_router(lambda: "test-token", runtime))
    client = TestClient(app)

    response = client.post("/maintenance/replay-failed?limit=5")

    assert response.status_code == 200
    assert response.json() == {
        "status": "completed",
        "requeued_normalization_commands": 2,
        "retried_failed_user_semantic": 3,
    }
    assert runtime.limits == [5]
    assert client.post("/maintenance/replay-failed?limit=0").status_code == 422
