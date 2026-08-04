from __future__ import annotations

import sqlite3

import pytest

from ledgermind_local.bootstrap import build_round_processing_worker
from ledgermind_local.inference import InferenceBroker, SecretStore
from ledgermind_local.inference.broker import InferenceProfileNotFoundError
from ledgermind_local.inference.profile_store import InferenceProfileStore
from ledgermind_local.inference.profiles import InferenceProfile
from ledgermind_local.persistence import rounds_migrations as migrations
from ledgermind_local.processing import BrokerHypothesisGenerator


def _database(tmp_path):
    path = tmp_path / "local.db"
    connection = sqlite3.connect(path)
    migrations.apply_migrations(connection)
    InferenceProfileStore(connection).upsert(
        InferenceProfile(
            profile_id="hypothesis-default",
            base_url="https://provider.example/v1",
            model="model",
            secret_ref="provider-main",
        )
    )
    connection.commit()
    connection.close()
    return path


def test_processing_requires_profile_when_no_generator_is_injected(tmp_path) -> None:
    with pytest.raises(ValueError, match="hypothesis_profile_id"):
        build_round_processing_worker(database_path=tmp_path / "local.db")


def test_processing_builds_broker_generator_and_validates_profile(tmp_path) -> None:
    database = _database(tmp_path)
    secrets = SecretStore(tmp_path / "secrets.json")
    secrets.put("provider-main", "TOP_SECRET")
    broker = InferenceBroker(database_path=database, secret_store=secrets)

    worker = build_round_processing_worker(
        database_path=database,
        broker=broker,
        hypothesis_profile_id="hypothesis-default",
    )

    assert isinstance(worker.generator, BrokerHypothesisGenerator)
    assert worker.generator.model == "model"


def test_processing_refuses_unknown_profile_at_startup(tmp_path) -> None:
    database = _database(tmp_path)
    secrets = SecretStore(tmp_path / "secrets.json")
    broker = InferenceBroker(database_path=database, secret_store=secrets)

    with pytest.raises(InferenceProfileNotFoundError):
        build_round_processing_worker(
            database_path=database,
            broker=broker,
            hypothesis_profile_id="missing",
        )
