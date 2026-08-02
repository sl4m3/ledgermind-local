"""Tests for Hermes extraction payload composition."""

from __future__ import annotations

from ledgermind_local.plugins.hermes.extraction import extract_request_payload


class _FakeResponse:
    def __init__(self, payload: dict[str, object], provider: str = "provider-x", model: str = "model-y"):
        self.parsed = payload
        self.provider = provider
        self.model = model


class _FakeLLM:
    def __init__(self, payload: dict[str, object], provider: str = "provider-x", model: str = "model-y"):
        self.payload = payload
        self.provider = provider
        self.model = model
        self.calls: list[dict[str, object]] = []

    def complete_structured(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return _FakeResponse(dict(self.payload), provider=self.provider, model=self.model)


class _FakeLLMWithReasoning:
    def __init__(self, payload: dict[str, object], provider: str = "provider-x", model: str = "model-y"):
        self.payload = payload
        self.provider = provider
        self.model = model
        self.calls: list[dict[str, object]] = []

    def complete_structured(
        self,
        *,
        instructions: object,
        input: object,
        json_schema: object,
        schema_name: object,
        system_prompt: object,
        temperature: object,
        max_tokens: object,
        timeout: object,
        purpose: object,
        reasoning_effort: str,
        **kwargs: object,
    ) -> object:
        del instructions, input, json_schema, schema_name, system_prompt, temperature
        del max_tokens, timeout, purpose, kwargs
        self.calls.append({"reasoning_effort": reasoning_effort})
        return _FakeResponse(dict(self.payload), provider=self.provider, model=self.model)


def _build_source_reference() -> dict[str, object]:
    return {
        "source_system": "hermes",
        "source_instance_id": "src_01J",
        "source_profile_id": "default",
        "source_session_id": "session-1",
        "source_round_id": "round-1",
        "first_message_id": None,
        "final_message_id": None,
        "message_ids": [],
        "source_digest": "sha256:feedface",
        "source_schema_version": 1,
        "resolver_version": 1,
    }


def test_extraction_request_does_not_pass_model_argument() -> None:
    llm = _FakeLLM(
        payload={
            "has_knowledge": True,
            "title": "title",
            "target": "target",
            "statement": "statement",
            "rationale": "rationale",
            "result": "result",
            "artifacts": [],
        }
    )

    payload, metadata, extraction = extract_request_payload(
        llm=llm,
        messages=[{"role": "user", "content": "q"}],
        platform="unit",
        model="model-overridden",
        source_reference=_build_source_reference(),
        extraction_prompt_version=1,
        extraction_schema_version=1,
        memory_space_id="hermes:src_01J:default",
    )

    assert payload is not None
    assert extraction.has_knowledge
    assert llm.calls
    call = llm.calls[0]
    assert "model" not in call
    assert "provider" not in call
    assert "json_schema" in call
    assert "schema_name" in call
    assert call["purpose"] == "ledgermind.atom.extract"
    assert call["temperature"] == 0.0
    assert metadata is not None


def test_extraction_request_returns_none_when_no_knowledge() -> None:
    llm = _FakeLLM(
        payload={
            "has_knowledge": False,
            "title": "",
            "target": "",
            "statement": "",
            "rationale": "",
            "result": "",
            "artifacts": [],
        }
    )

    payload, _metadata, extraction = extract_request_payload(
        llm=llm,
        messages=[{"role": "user", "content": "q"}],
        platform="unit",
        model="ignored",
        source_reference=_build_source_reference(),
        extraction_prompt_version=1,
        extraction_schema_version=1,
        memory_space_id="hermes:src_01J:default",
    )

    assert payload is None
    assert extraction.has_knowledge is False


def test_extraction_request_forwards_platform_and_limits_tool_args() -> None:
    huge_text = "x" * 20_000
    llm = _FakeLLM(
        payload={
            "has_knowledge": True,
            "title": "title",
            "target": "target",
            "statement": "statement",
            "rationale": "rationale",
            "result": "result",
            "artifacts": [],
        }
    )

    extract_request_payload(
        llm=llm,
        messages=[
            {
                "role": "user",
                "content": "what",
                "tool_calls": [
                    {
                        "id": "tool-1",
                        "name": "calc",
                        "arguments": huge_text,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [],
            },
        ],
        platform="unit-test",
        model="ignored",
        source_reference=_build_source_reference(),
        extraction_prompt_version=1,
        extraction_schema_version=1,
        memory_space_id="hermes:src_01J:default",
    )

    request = llm.calls[0]
    assert request["platform"] == "unit-test"
    assert request["purpose"] == "ledgermind.atom.extract"
    payload_text = str(request["input"][0]["text"])
    assert "[... truncated ...]" in payload_text


def test_extraction_rejects_extra_fields_in_schema() -> None:
    llm = _FakeLLM(
        payload={
            "has_knowledge": True,
            "title": "title",
            "target": "target",
            "statement": "statement",
            "rationale": "rationale",
            "result": "result",
            "artifacts": [],
            "phase": "pattern",
        }
    )

    try:
        extract_request_payload(
            llm=llm,
            messages=[{"role": "user", "content": "q"}],
            platform="unit",
            model="ignored",
            source_reference=_build_source_reference(),
            extraction_prompt_version=1,
            extraction_schema_version=1,
            memory_space_id="hermes:src_01J:default",
        )
    except ValueError:
        return
    raise AssertionError("expected extraction to reject extra fields")


def test_extraction_request_sets_reasoning_effort_when_supported() -> None:
    llm = _FakeLLMWithReasoning(
        payload={
            "has_knowledge": True,
            "title": "title",
            "target": "target",
            "statement": "statement",
            "rationale": "rationale",
            "result": "result",
            "artifacts": [],
        }
    )

    payload, _metadata, extraction = extract_request_payload(
        llm=llm,
        messages=[{"role": "user", "content": "q"}],
        platform="unit",
        model="ignored",
        source_reference={
            "source_system": "hermes",
            "source_instance_id": "src_01J",
            "source_profile_id": "default",
            "source_session_id": "session-1",
            "source_round_id": "round-1",
            "first_message_id": None,
            "final_message_id": None,
            "message_ids": [],
            "source_digest": "sha256:feedface",
            "source_schema_version": 1,
            "resolver_version": 1,
        },
        extraction_prompt_version=1,
        extraction_schema_version=1,
        memory_space_id="hermes:src_01J:default",
    )

    assert payload is not None
    assert llm.calls[0]["reasoning_effort"] == "none"
    assert extraction.has_knowledge
