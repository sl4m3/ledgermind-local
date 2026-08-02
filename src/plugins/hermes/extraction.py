"""Atom extraction orchestration for a completed turn."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .extraction_schema import (
    ATOM_EXTRACTION_SCHEMA_V1,
    EXTRACTION_INSTRUCTIONS_V1,
    EXTRACTION_SYSTEM_PROMPT_V1,
    validate_extraction_payload,
)

try:
    from application.digests import (
        calculate_idempotency_key,
        calculate_source_round_key,
    )
except Exception:  # pragma: no cover - defensive fallback for isolated environments

    def calculate_source_round_key(payload: Mapping[str, Any]) -> str:
        source = dict(payload)
        raw = "\n".join(
            (
                str(source.get("source_system", "")),
                str(source.get("source_instance_id", "")),
                str(source.get("source_profile_id", "")),
                str(source.get("source_session_id", "")),
                str(source.get("source_round_id", "")),
            )
        )
        return f"sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"

    def calculate_idempotency_key(
        source_round_key: str,
        extraction_prompt_version: int,
        extraction_schema_version: int,
    ) -> str:
        payload = {
            "source_round_key": source_round_key,
            "prompt_version": extraction_prompt_version,
            "schema_version": extraction_schema_version,
        }
        body = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(body).hexdigest()}"


_MAX_CONTENT_CHARS = 12_000


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    has_knowledge: bool
    title: str
    target: str
    statement: str
    rationale: str
    result: str
    artifacts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExtractionCallMetadata:
    provider: str
    model: str
    usage_cache_read_tokens: int | None


@dataclass(frozen=True, slots=True)
class ExtractionPlan:
    payload: dict[str, Any]
    metadata: ExtractionCallMetadata


def _sanitize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _truncate_large_text(value: str, *, max_chars: int = _MAX_CONTENT_CHARS) -> str:
    if len(value) <= max_chars:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    tail = max_chars // 2
    head = max_chars - tail
    return (
        f"{value[:head]}\n[... truncated ...]\n"
        f"size={len(value)} sha256={digest}\n"
        f"{value[-tail:]}"
    )


def _normalize_tool_args(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, dict, int, float, bool)):
        return json.dumps(value, ensure_ascii=False)
    return _sanitize_text(value)


def _normalize_message_for_model(message: Mapping[str, Any]) -> dict[str, Any]:
    role = _sanitize_text(message.get("role", "")).strip().lower()
    content = _sanitize_text(message.get("content", ""))
    tool_calls = []
    for entry in message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []:
        call_id = _sanitize_text(entry.get("id") or entry.get("call_id"))
        name = _sanitize_text(entry.get("name") or entry.get("tool_name"))
        arguments = _normalize_tool_args(entry.get("arguments"))
        tool_calls.append(
            {
                "id": call_id,
                "name": name,
                "arguments": _truncate_large_text(arguments),
            }
        )
    return {
        "role": role,
        "content": _truncate_large_text(content),
        "tool_name": _sanitize_text(message.get("tool_name")) or None,
        "tool_call_id": _sanitize_text(message.get("tool_call_id")) or None,
        "tool_calls": tool_calls,
    }


def serialize_round_for_extraction(
    messages: list[dict[str, Any]],
    *,
    platform: str,
    model: str,
) -> str:
    payload = {
        "platform": _sanitize_text(platform),
        "model": _sanitize_text(model),
        "round": [_normalize_message_for_model(message) for message in messages],
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _supports_param(callable_obj: Any, name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    parameters = signature.parameters
    if name in parameters:
        return True
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


def _build_call_kwargs(
    *,
    complete_structured: Any,
    platform: str,
    extraction_payload: str,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "instructions": EXTRACTION_INSTRUCTIONS_V1,
        "input": [{"type": "text", "text": extraction_payload}],
        "json_schema": ATOM_EXTRACTION_SCHEMA_V1,
        "schema_name": "ledgermind.atom.v1",
        "system_prompt": EXTRACTION_SYSTEM_PROMPT_V1,
        "temperature": 0.0,
        "max_tokens": 2048,
        "timeout": 120,
        "purpose": "ledgermind.atom.extract",
    }
    if platform and _supports_param(complete_structured, "platform"):
        kwargs["platform"] = platform
    return kwargs


def _supports_reasoning_effort(callable_obj: Any) -> bool:
    return _supports_param(callable_obj, "reasoning_effort")


def _read_extraction_response(payload: Any) -> tuple[dict[str, Any], ExtractionCallMetadata]:
    parsed = getattr(payload, "parsed", None)
    if parsed is None and isinstance(payload, Mapping):
        parsed = payload.get("parsed")
    if parsed is None and isinstance(payload, Mapping):
        parsed = dict(payload)
    if not isinstance(parsed, Mapping):
        raise ValueError("extraction response has no parsed payload")

    normalized = {
        "has_knowledge": bool(parsed.get("has_knowledge")),
        "title": _sanitize_text(parsed.get("title", ""))[:240],
        "target": _sanitize_text(parsed.get("target", ""))[:240],
        "statement": _sanitize_text(parsed.get("statement", ""))[:20_000],
        "rationale": _sanitize_text(parsed.get("rationale", ""))[:40_000],
        "result": _sanitize_text(parsed.get("result", ""))[:20_000],
        "artifacts": tuple(parsed.get("artifacts") or ()),
    }
    if not isinstance(normalized["artifacts"], tuple):
        if isinstance(normalized["artifacts"], list):
            normalized["artifacts"] = tuple(str(item) for item in normalized["artifacts"])
        else:
            normalized["artifacts"] = tuple()
    if not normalized["has_knowledge"]:
        normalized["title"] = ""
        normalized["target"] = ""
        normalized["statement"] = ""
        normalized["rationale"] = ""
        normalized["result"] = ""
        normalized["artifacts"] = tuple()

    if not validate_extraction_payload(normalized):
        raise ValueError("extraction payload violates schema")

    usage = getattr(payload, "usage", None)
    cache_tokens = None
    if isinstance(usage, Mapping):
        cache_value = usage.get("cache_read_tokens")
        if cache_value is not None:
            try:
                cache_tokens = int(cache_value)
            except (TypeError, ValueError):
                cache_tokens = None

    return (
        {
            "has_knowledge": normalized["has_knowledge"],
            "title": normalized["title"],
            "target": normalized["target"],
            "statement": normalized["statement"],
            "rationale": normalized["rationale"],
            "result": normalized["result"],
            "artifacts": list(normalized["artifacts"]),
        },
        ExtractionCallMetadata(
            provider=_sanitize_text(getattr(payload, "provider", ""))[:200],
            model=_sanitize_text(getattr(payload, "model", ""))[:300],
            usage_cache_read_tokens=cache_tokens,
        ),
    )


def extract_atom(
    *,
    llm: Any,
    messages: list[dict[str, Any]],
    platform: str,
    model: str,
    extraction_prompt_version: int,
    extraction_schema_version: int,
    source_round_key: str,
) -> tuple[ExtractionResult, ExtractionCallMetadata]:
    if llm is None or not hasattr(llm, "complete_structured"):
        raise ValueError("llm client does not expose complete_structured")

    serialized = serialize_round_for_extraction(messages, platform=platform, model=model)
    kwargs = _build_call_kwargs(
        complete_structured=llm.complete_structured,
        platform=platform,
        extraction_payload=serialized,
    )
    if _supports_reasoning_effort(llm.complete_structured):
        kwargs["reasoning_effort"] = "none"

    raw = llm.complete_structured(**kwargs)
    payload, metadata = _read_extraction_response(raw)

    result = ExtractionResult(
        has_knowledge=bool(payload["has_knowledge"]),
        title=payload["title"],
        target=payload["target"],
        statement=payload["statement"],
        rationale=payload["rationale"],
        result=payload["result"],
        artifacts=tuple(payload["artifacts"]),
    )
    return result, metadata


def build_empty_result() -> ExtractionResult:
    return ExtractionResult(False, "", "", "", "", "", ())


def build_request_from_extraction(
    *,
    memory_space_id: str,
    extraction_payload: dict[str, Any],
    source_reference: dict[str, Any],
    extraction_prompt_version: int,
    extraction_schema_version: int,
    provider: str,
    model: str,
    source_round_key: str,
) -> dict[str, Any]:
    source = dict(source_reference)
    source.setdefault("message_ids", [])

    request: dict[str, Any] = {
        "api_version": "1",
        "idempotency_key": calculate_idempotency_key(
            source_round_key=source_round_key,
            extraction_prompt_version=extraction_prompt_version,
            extraction_schema_version=extraction_schema_version,
        ),
        "memory_space_id": str(memory_space_id),
        "source": source,
        "atom": {
            "title": str(extraction_payload.get("title", ""))[:240],
            "target": str(extraction_payload.get("target", ""))[:240],
            "statement": str(extraction_payload.get("statement", ""))[:20_000],
            "rationale": str(extraction_payload.get("rationale", ""))[:40_000],
            "result": str(extraction_payload.get("result", ""))[:20_000],
            "artifacts": [
                str(item) for item in (extraction_payload.get("artifacts") or [])
            ],
        },
        "extraction": {
            "host": "hermes",
            "provider": _sanitize_text(provider)[:200],
            "model": _sanitize_text(model)[:300],
            "prompt_version": int(extraction_prompt_version),
            "schema_version": int(extraction_schema_version),
            "purpose": "ledgermind.atom.extract",
        },
    }
    if request["atom"]["title"] == "" and request["atom"]["target"] == "" and request["atom"]["statement"] == "":
        request["atom"]["artifacts"] = []

    if len(request["atom"]["artifacts"]) > 500:
        request["atom"]["artifacts"] = request["atom"]["artifacts"][:500]
    return request


def build_extraction_payload(
    *,
    source_reference: Mapping[str, Any],
    extraction: ExtractionResult,
    memory_space_id: str,
    extraction_prompt_version: int,
    extraction_schema_version: int,
) -> dict[str, Any]:
    source_round_key = calculate_source_round_key(source_reference)
    return build_request_from_extraction(
        memory_space_id=memory_space_id,
        extraction_payload={
            "has_knowledge": extraction.has_knowledge,
            "title": extraction.title,
            "target": extraction.target,
            "statement": extraction.statement,
            "rationale": extraction.rationale,
            "result": extraction.result,
            "artifacts": list(extraction.artifacts),
        },
        source_reference=dict(source_reference),
        extraction_prompt_version=extraction_prompt_version,
        extraction_schema_version=extraction_schema_version,
        provider="",
        model="",
        source_round_key=source_round_key,
    )


def extract_request_payload(
    *,
    llm: Any,
    messages: list[dict[str, Any]],
    platform: str,
    model: str,
    source_reference: Mapping[str, Any],
    extraction_prompt_version: int,
    extraction_schema_version: int,
    memory_space_id: str,
) -> tuple[dict[str, Any] | None, ExtractionCallMetadata | None, ExtractionResult]:
    source_round_key = calculate_source_round_key(source_reference)
    extraction, metadata = extract_atom(
        llm=llm,
        messages=messages,
        platform=platform,
        model=model,
        extraction_prompt_version=extraction_prompt_version,
        extraction_schema_version=extraction_schema_version,
        source_round_key=source_round_key,
    )
    if not extraction.has_knowledge:
        return None, metadata, extraction

    payload = build_request_from_extraction(
        memory_space_id=memory_space_id,
        extraction_payload={
            "has_knowledge": extraction.has_knowledge,
            "title": extraction.title,
            "target": extraction.target,
            "statement": extraction.statement,
            "rationale": extraction.rationale,
            "result": extraction.result,
            "artifacts": list(extraction.artifacts),
        },
        source_reference=dict(source_reference),
        extraction_prompt_version=extraction_prompt_version,
        extraction_schema_version=extraction_schema_version,
        provider=metadata.provider,
        model=metadata.model,
        source_round_key=source_round_key,
    )
    return payload, metadata, extraction
