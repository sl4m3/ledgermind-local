from __future__ import annotations

import json
import sqlite3

import pytest

from ledgermind_local.inference.broker import (
    InferenceBroker,
    InferenceInputTooLargeError,
    InferenceProfileDisabledError,
    InferenceResponseValidationError,
    ModelTask,
)
from ledgermind_local.inference.profile_store import InferenceProfileStore
from ledgermind_local.inference.profiles import InferenceProfile
from ledgermind_local.inference.providers.base import (
    ModelRequest,
    ModelResponse,
    ProviderAuthenticationError,
)
from ledgermind_local.inference.secrets import SecretStore
from ledgermind_local.persistence import rounds_migrations as migrations
from ledgermind_local.processing.models import NormalizedRound


class _Provider:
    provider_kind = "openai_compatible"

    def __init__(self, content: str, *, error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.requests = []

    def complete_json(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return ModelResponse(
            content=self.content,
            model=request.model,
            attempts=1,
            request_bytes=len(request.encoded_payload()),
            response_bytes=len(self.content.encode()),
            status_code=200,
        )

    def close(self) -> None:
        return None


def _round() -> NormalizedRound:
    return NormalizedRound(
        memory_space_id="space",
        source_system="hermes",
        source_instance_id="instance",
        source_profile_id="profile",
        source_session_id="session",
        source_round_id="round",
        started_at="2026-08-03T00:00:00Z",
        completed_at="2026-08-03T00:01:00Z",
        user_text="request",
        assistant_text="answer",
        transcript="user: request\nassistant: answer",
        tool_interactions=(),
        normalized_digest="sha256:" + "a" * 64,
        source_event_ids=("event-1", "event-2"),
    )


def _setup(tmp_path, *, enabled: bool = True, output_tokens: int = 800):
    tmp_path.mkdir(parents=True, exist_ok=True)
    database = tmp_path / "local.db"
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    migrations.apply_migrations(connection)
    connection.execute(
        "INSERT INTO memory_spaces(memory_space_id, source_client, created_at, updated_at) VALUES (?, ?, ?, ?)",
        ("space", "tests", "2026-08-03T00:00:00Z", "2026-08-03T00:00:00Z"),
    )
    store = InferenceProfileStore(connection)
    store.upsert(
        InferenceProfile(
            profile_id="hypothesis-default",
            base_url="https://provider.example/v1",
            model="test-model",
            secret_ref="provider-main",
            max_input_tokens=10_000,
            max_output_tokens=output_tokens,
            enabled=enabled,
        )
    )
    connection.commit()
    connection.close()
    secrets = SecretStore(tmp_path / "secrets.json")
    secrets.put("provider-main", "TOP_SECRET")
    return database, secrets


def test_broker_generates_candidates_and_records_payload_free_audit(tmp_path) -> None:
    database, secrets = _setup(tmp_path)
    provider = _Provider(
        json.dumps(
            {
                "hypotheses": [
                    {
                        "title": "Title",
                        "target": "ops",
                        "statement": "Statement",
                        "rationale": "Reason",
                        "result": "Result",
                        "artifacts": [],
                        "source_event_ids": ["event-1"],
                    }
                ]
            }
        )
    )
    broker = InferenceBroker(
        database_path=database,
        secret_store=secrets,
        provider_factory=lambda profile, secret: provider,
    )

    candidates = broker.generate_hypotheses(_round(), "hypothesis-default")

    assert len(candidates) == 1
    assert candidates[0].source_event_ids == ("event-1",)
    assert provider.requests[0].model == "test-model"
    assert "TOP_SECRET" not in provider.requests[0].model_dump_json()
    connection = sqlite3.connect(database)
    row = connection.execute("SELECT * FROM egress_audit").fetchone()
    assert row is not None
    assert "TOP_SECRET" not in " ".join(str(value) for value in row)
    assert row[6] == "success"
    connection.close()


def test_broker_rejects_extra_fields_and_unknown_source_events(tmp_path) -> None:
    database, secrets = _setup(tmp_path)
    provider = _Provider(
        '{"hypotheses":[{"title":"T","target":"O","statement":"S","rationale":"","result":"","artifacts":[],"source_event_ids":["missing"],"phase":"pattern"}]}'
    )
    broker = InferenceBroker(
        database_path=database,
        secret_store=secrets,
        provider_factory=lambda profile, secret: provider,
    )

    with pytest.raises(InferenceResponseValidationError):
        broker.generate_hypotheses(_round(), "hypothesis-default")

    assert "phase" not in str(provider.error or "")


def test_broker_records_provider_failure_without_secret(tmp_path) -> None:
    database, secrets = _setup(tmp_path)
    provider = _Provider(
        "", error=ProviderAuthenticationError("provider authentication failed")
    )
    broker = InferenceBroker(
        database_path=database,
        secret_store=secrets,
        provider_factory=lambda profile, secret: provider,
    )

    with pytest.raises(ProviderAuthenticationError) as error:
        broker.generate_hypotheses(_round(), "hypothesis-default")

    assert "TOP_SECRET" not in str(error.value)
    connection = sqlite3.connect(database)
    assert (
        connection.execute("SELECT status FROM egress_audit").fetchone()[0] == "error"
    )
    connection.close()


def test_broker_rejects_disabled_profile_and_missing_secret(tmp_path) -> None:
    database, secrets = _setup(tmp_path, enabled=False)
    broker = InferenceBroker(database_path=database, secret_store=secrets)
    with pytest.raises(InferenceProfileDisabledError):
        broker.generate_hypotheses(_round(), "hypothesis-default")

    database, secrets = _setup(tmp_path / "missing", enabled=True)
    secrets.delete("provider-main")
    broker = InferenceBroker(database_path=database, secret_store=secrets)
    with pytest.raises(KeyError, match="provider-main"):
        broker.generate_hypotheses(_round(), "hypothesis-default")


def test_broker_limits_input_before_provider_call(tmp_path) -> None:
    database, secrets = _setup(tmp_path, output_tokens=800)
    provider = _Provider('{"hypotheses": []}')
    broker = InferenceBroker(
        database_path=database,
        secret_store=secrets,
        provider_factory=lambda profile, secret: provider,
        max_input_chars=20,
    )

    with pytest.raises(InferenceInputTooLargeError):
        broker.generate_hypotheses(_round(), "hypothesis-default")
    assert provider.requests == []


def test_model_task_is_strict_and_broker_exposes_execution_boundary(tmp_path) -> None:
    _setup(tmp_path)
    provider = _Provider('{"hypotheses": []}')
    request = provider.requests
    assert request == []
    task = ModelTask(
        memory_space_id="space",
        operation="hypothesis",
        request=ModelRequest.from_messages(
            model="test-model",
            system_prompt="system",
            user_prompt="user",
            max_output_tokens=10,
        ),
    )
    assert task.operation == "hypothesis"
