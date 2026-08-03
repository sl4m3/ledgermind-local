"""Explicit temporary bridge from hypotheses to the existing core atom flow."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

from ledgermind_core.application.mappers import IngestAtomCommand
from ledgermind_core.domain import AtomContent, ExtractionInfo, SourceReference

from ..persistence import RawRoundRecord
from .generator import HypothesisDraft


def _draft_material(raw_round: RawRoundRecord, draft: HypothesisDraft) -> dict[str, object]:
    return {
        "raw_round_id": raw_round.raw_round_id,
        "memory_space_id": raw_round.memory_space_id,
        "title": draft.title,
        "target": draft.target,
        "statement": draft.statement,
        "rationale": draft.rationale,
        "result": draft.result,
        "artifacts": list(draft.artifacts),
    }


def _digest(raw_round: RawRoundRecord, draft: HypothesisDraft) -> str:
    payload = json.dumps(
        _draft_material(raw_round, draft),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class CoreHypothesisBridge:
    """Map semantic drafts to the pre-existing core ``IngestAtomHandler``."""

    def __init__(
        self,
        ingest_handler: Callable[[IngestAtomCommand], object] | object,
        *,
        provider: str = "local",
        model: str = "hypothesis-generator",
        prompt_version: int = 1,
        schema_version: int = 1,
    ) -> None:
        self._ingest_handler = ingest_handler
        self.provider = provider
        self.model = model
        self.prompt_version = max(int(prompt_version), 1)
        self.schema_version = max(int(schema_version), 1)

    def publish(self, raw_round: RawRoundRecord, draft: HypothesisDraft) -> object:
        digest = _digest(raw_round, draft)
        source_system = "hermes" if raw_round.source_system == "hermes" else "legacy_import"
        command = IngestAtomCommand(
            idempotency_key=digest,
            request_hash=digest,
            memory_space_id=raw_round.memory_space_id,
            source=SourceReference(
                source_system=source_system,
                source_instance_id=raw_round.source_instance_id,
                source_profile_id=raw_round.source_profile_id,
                source_session_id=raw_round.source_session_id,
                source_round_id=f"{raw_round.source_round_id}:hypothesis:{digest[-12:]}",
                first_message_id=None,
                final_message_id=None,
                message_ids=(),
                source_digest=raw_round.payload_digest,
                source_schema_version=raw_round.capture_schema_version,
                resolver_version=1,
            ),
            content=AtomContent(
                title=draft.title,
                target=draft.target,
                statement=draft.statement,
                rationale=draft.rationale,
                result=draft.result,
                artifacts=tuple(draft.artifacts),
            ),
            extraction=ExtractionInfo(
                host="local",
                provider=self.provider,
                model=self.model,
                prompt_version=self.prompt_version,
                schema_version=self.schema_version,
                purpose="ledgermind.hypothesis.bridge",
            ),
        )
        handler = self._ingest_handler
        if hasattr(handler, "handle"):
            return handler.handle(command)
        return handler(command)  # type: ignore[operator]
