from __future__ import annotations

import json
import sqlite3

import pytest

from ledgermind_local.inference.broker import (
    InferenceBroker,
    InferenceResponseValidationError,
)
from ledgermind_local.inference.profile_store import InferenceProfileStore
from ledgermind_local.inference.profiles import InferenceProfile
from ledgermind_local.inference.providers.base import ModelResponse
from ledgermind_local.inference.secrets import SecretStore
from ledgermind_local.persistence import rounds_migrations as migrations


class _Provider:
    def __init__(self, content: str) -> None:
        self.content = content
        self.requests = []

    def complete_json(self, request):
        self.requests.append(request)
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


def _setup(tmp_path):
    database = tmp_path / "local.db"
    connection = sqlite3.connect(database)
    migrations.apply_migrations(connection)
    connection.execute(
        "INSERT INTO memory_spaces(memory_space_id, source_client, created_at, updated_at) VALUES (?, ?, ?, ?)",
        ("space", "tests", "2026-08-03T00:00:00Z", "2026-08-03T00:00:00Z"),
    )
    InferenceProfileStore(connection).upsert(
        InferenceProfile(
            profile_id="merge-default",
            base_url="https://provider.example/v1",
            model="merge-model",
            secret_ref="provider-main",
            max_input_tokens=10_000,
            max_output_tokens=800,
            enabled=True,
        )
    )
    connection.commit()
    connection.close()
    secrets = SecretStore(tmp_path / "secrets.json")
    secrets.put("provider-main", "TOP_SECRET")
    return database, secrets


def _proposal_payload(*, extra: bool = False) -> dict[str, object]:
    payload: dict[str, object] = {
        "title": "Merged knowledge",
        "target": "ops",
        "statement": "One merged statement",
        "rationale": "The sources agree",
        "preserved_references": ["knowledge-a", "knowledge-b"],
        "preserved_constraints": ["keep source"],
    }
    if extra:
        payload["phase"] = "canonical"
    return payload


def test_broker_generates_strict_merge_proposal_and_audits_operation(tmp_path) -> None:
    database, secrets = _setup(tmp_path)
    provider = _Provider(json.dumps(_proposal_payload()))
    broker = InferenceBroker(
        database_path=database,
        secret_store=secrets,
        provider_factory=lambda profile, secret: provider,
    )

    proposal = broker.generate_merge_proposal(
        memory_space_id="space",
        model_input={"items": [{"reference": "knowledge-a"}]},
        profile_id="merge-default",
    )

    assert proposal.title == "Merged knowledge"
    assert proposal.preserved_references == ("knowledge-a", "knowledge-b")
    assert provider.requests[0].model == "merge-model"
    assert "TOP_SECRET" not in provider.requests[0].model_dump_json()
    connection = sqlite3.connect(database)
    assert connection.execute("SELECT operation FROM egress_audit").fetchone()[0] == "merge"
    connection.close()


def test_broker_rejects_extra_merge_fields_before_returning_proposal(tmp_path) -> None:
    database, secrets = _setup(tmp_path)
    provider = _Provider(json.dumps(_proposal_payload(extra=True)))
    broker = InferenceBroker(
        database_path=database,
        secret_store=secrets,
        provider_factory=lambda profile, secret: provider,
    )

    with pytest.raises(InferenceResponseValidationError):
        broker.generate_merge_proposal(
            memory_space_id="space",
            model_input={"items": []},
            profile_id="merge-default",
        )
