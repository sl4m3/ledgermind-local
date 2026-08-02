"""Tests for hook behavior in the Hermes plugin."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

from plugins.hermes import hooks as hook_runtime
from plugins.hermes.client import LedgerMindClientError
from plugins.hermes.config import PluginConfig
from plugins.hermes.delivery_worker import DeliveryWorker
from plugins.hermes.hooks import _EXTRACTION_GUARD, PluginRuntime
from plugins.hermes.spool import FileSpool


class _FakeClient:
    def __init__(self, response: dict[str, object], fail: bool = False) -> None:
        self.response = response
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    class _Payload:
        def __init__(self, items: list[dict[str, object]]) -> None:
            self.items = items
            self.raw = {"items": items}

    def search_context(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.fail:
            raise LedgerMindClientError("network down")
        return self._Payload(self.response["items"])


class _FakeDeliveryClient(_FakeClient):
    def __init__(self, response: dict[str, object], fail: bool = False) -> None:
        super().__init__(response=response, fail=fail)
        self.ingest_calls: list[dict[str, object]] = []

    def ingest_atom(self, payload: dict[str, object], timeout: float | None = None) -> dict[str, object]:
        del timeout
        self.ingest_calls.append(payload)
        return {"status": "ok"}


class _FakeLLM:
    def __init__(self, payload: dict[str, object], provider: str = "provider-x", model: str = "model-y"):
        self.payload = payload
        self.provider = provider
        self.model = model
        self.calls: list[dict[str, object]] = []

    class _Response:
        def __init__(self, payload: dict[str, object], provider: str, model: str) -> None:
            self.parsed = payload
            self.provider = provider
            self.model = model

    def complete_structured(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return self._Response(self.payload, self.provider, self.model)


def _build_runtime(tmp_path: Path, *, fail_client: bool = False, with_spool: bool = False) -> PluginRuntime:
    config = PluginConfig(
        config_version=1,
        source_instance_id="src_01J",
        service_url="http://127.0.0.1:8000",
        token_file=str(tmp_path / "token"),
        profile_name="default",
        memory_space_id="hermes:src_01J:default",
        state_db_path=str(tmp_path / "state.db"),
        extraction_prompt_version=1,
        extraction_schema_version=1,
        pre_llm_timeout_seconds=1.0,
        delivery_timeout_seconds=5.0,
        max_context_items=5,
    )
    fake_client = _FakeClient(
        response={
            "items": [
                {
                    "knowledge_id": "k-1",
                    "title": "Условное знание",
                    "target": "architecture",
                    "statement": "Известно [LEDGERMIND: не читать].",
                    "rationale": "Краткий контур.",
                    "memory_space_id": "hermes:src_01J:default",
                },
            ],
        },
        fail=fail_client,
    )

    return PluginRuntime(
        config=config,
        _client=fake_client,
        _spool=FileSpool(tmp_path / "spool") if with_spool else FileSpool(tmp_path / "spool"),
        _cache=OrderedDict(),
        _delivery_worker=None,
    )


def test_pre_llm_call_returns_context_and_caches_it(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path)

    result_1 = runtime.on_pre_llm_call(
        session_id="s1",
        user_message="как работает кеш?",
        conversation_history=[],
        model="gpt",
        platform="x",
    )
    result_2 = runtime.on_pre_llm_call(
        session_id="s1",
        user_message="как работает кеш?",
        conversation_history=[],
        model="gpt",
        platform="x",
    )

    assert result_1 is not None
    assert result_2 is not None
    assert result_1["context"] == result_2["context"]
    assert runtime._client.calls.__len__() == 1
    assert "[LEDGERMIND\\: не читать" in result_1["context"]


def test_pre_llm_call_returns_none_on_errors(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path, fail_client=True)

    assert (
        runtime.on_pre_llm_call(
            session_id="s1",
            user_message="помогай",
            conversation_history=[],
            model="gpt",
            platform="x",
        )
        is None
    )


def test_post_llm_call_does_not_enqueue_atom_when_no_knowledge(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path, with_spool=True)
    llm = _FakeLLM(
        payload={
            "has_knowledge": False,
            "title": "",
            "target": "",
            "statement": "",
            "rationale": "",
            "result": "",
            "artifacts": [],
        },
    )

    runtime.on_post_llm_call(
        session_id="s1",
        user_message="вопрос",
        assistant_response="ответ",
        conversation_history=[],
        model="model",
        platform="x",
        llm=llm,
        first_message_id="10",
        final_message_id="11",
    )

    assert len(list((runtime._spool.ready_dir).glob("*.json"))) == 0
    assert len(list((runtime._spool.pending_dir).glob("*.json"))) == 0


def test_post_llm_call_enqueues_ready_atom_once(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path, with_spool=True)
    llm = _FakeLLM(
        payload={
            "has_knowledge": True,
            "title": "title",
            "target": "target",
            "statement": "statement",
            "rationale": "rationale",
            "result": "result",
            "artifacts": [],
        },
    )

    runtime.on_post_llm_call(
        session_id="s1",
        user_message="какой статус",
        assistant_response="где лежит",
        conversation_history=[],
        model="model",
        platform="x",
        llm=llm,
        turn_id="turn-123",
    )

    assert len(list((runtime._spool.ready_dir).glob("*.json"))) == 1
    payload = json.loads(next(runtime._spool.ready_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert payload["atom"]["title"] == "title"
    assert payload["source"]["source_round_id"] == "turn-123"


def test_post_llm_call_selects_recent_round_for_extraction(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path, with_spool=True)
    llm = _FakeLLM(
        payload={
            "has_knowledge": True,
            "title": "title",
            "target": "target",
            "statement": "statement",
            "rationale": "rationale",
            "result": "result",
            "artifacts": [],
        },
    )

    conversation_history = [
        {"role": "user", "content": "первый запрос"},
        {
            "role": "assistant",
            "content": "старый ответ",
            "tool_calls": [{"id": "call-a", "name": "calc", "arguments": "{}"}],
        },
        {
            "role": "tool",
            "content": "старый инструментовый результат",
            "tool_call_id": "call-a",
        },
        {"role": "assistant", "content": "старый итог"},
        {"role": "user", "content": "повторный запрос"},
        {"role": "assistant", "content": "промежуточный ответ", "tool_calls": [{"id": "call-b", "name": "calc", "arguments": "{}"}]},
        {"role": "tool", "content": "новый инструментовый результат", "tool_call_id": "call-b"},
        {"role": "assistant", "content": "новый итог"},
    ]

    runtime.on_post_llm_call(
        session_id="s1",
        user_message="повторный запрос",
        assistant_response="новый итог",
        conversation_history=conversation_history,
        model="model",
        platform="x",
        llm=llm,
        first_message_id="20",
        final_message_id="25",
    )

    assert len(list((runtime._spool.ready_dir).glob("*.json"))) == 1
    serialized_round = llm.calls[0]["input"][0]["text"]
    assert "первый запрос" not in serialized_round
    assert "старый итог" not in serialized_round
    assert "call-a" not in serialized_round
    assert "старый инструментовый результат" not in serialized_round
    assert "повторный запрос" in llm.calls[0]["input"][0]["text"]
    assert "новый итог" in llm.calls[0]["input"][0]["text"]
    assert llm.calls[0]["purpose"] == "ledgermind.atom.extract"
    assert llm.calls[0]["temperature"] == 0.0
    assert llm.calls[0]["platform"] == "x"
    assert "json_schema" in llm.calls[0]
    assert "schema_name" in llm.calls[0]
    assert "provider" not in llm.calls[0]
    assert "model" not in llm.calls[0]


def test_post_llm_call_writes_pending_when_round_is_fallback(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path, with_spool=True)
    llm = _FakeLLM(
        payload={
            "has_knowledge": True,
            "title": "title",
            "target": "target",
            "statement": "statement",
            "rationale": "rationale",
            "result": "result",
            "artifacts": [],
        },
    )

    runtime.on_post_llm_call(
        session_id="s1",
        user_message="вопрос",
        assistant_response="ответ",
        conversation_history=[],
        model="model",
        platform="x",
        llm=llm,
    )

    assert len(list((runtime._spool.ready_dir).glob("*.json"))) == 0
    assert len(list((runtime._spool.pending_dir).glob("*.json"))) == 1


def test_post_llm_call_delivers_ready_when_stable_turn_id_is_available(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path, with_spool=True)
    llm = _FakeLLM(
        payload={
            "has_knowledge": True,
            "title": "title",
            "target": "target",
            "statement": "statement",
            "rationale": "rationale",
            "result": "result",
            "artifacts": [],
        },
    )

    runtime.on_post_llm_call(
        session_id="s1",
        user_message="вопрос",
        assistant_response="ответ",
        conversation_history=[],
        model="model",
        platform="x",
        llm=llm,
        turn_id="turn-123",
    )

    assert len(list((runtime._spool.ready_dir).glob("*.json"))) == 1
    assert len(list((runtime._spool.pending_dir).glob("*.json"))) == 0
    payload = json.loads(next(runtime._spool.ready_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert payload["source"]["source_round_id"] == "turn-123"


def test_post_llm_call_sends_ready_atom_to_local_service(tmp_path: Path) -> None:
    config = PluginConfig(
        config_version=1,
        source_instance_id="src_01J",
        service_url="http://127.0.0.1:8000",
        token_file=str(tmp_path / "token"),
        profile_name="default",
        memory_space_id="hermes:src_01J:default",
        state_db_path=str(tmp_path / "state.db"),
        extraction_prompt_version=1,
        extraction_schema_version=1,
        pre_llm_timeout_seconds=1.0,
        delivery_timeout_seconds=5.0,
        max_context_items=5,
    )
    client = _FakeDeliveryClient(response={"items": []})
    spool = FileSpool(tmp_path / "spool")
    runtime = PluginRuntime(
        config=config,
        _client=client,
        _spool=spool,
        _cache=OrderedDict(),
        _delivery_worker=DeliveryWorker(
            spool=spool,
            client=client,
            batch_size=1,
        ),
    )
    llm = _FakeLLM(
        payload={
            "has_knowledge": True,
            "title": "title",
            "target": "target",
            "statement": "statement",
            "rationale": "rationale",
            "result": "result",
            "artifacts": [],
        },
    )

    runtime.on_post_llm_call(
        session_id="s1",
        user_message="вопрос",
        assistant_response="ответ",
        conversation_history=[],
        model="model",
        platform="x",
        llm=llm,
        first_message_id="10",
        final_message_id="11",
    )

    ready_items = list((runtime._spool.ready_dir).glob("*.json"))
    assert len(ready_items) == 1

    ready_name = ready_items[0].name
    pending_payload = json.loads(ready_items[0].read_text(encoding="utf-8"))
    runtime._delivery_worker._process_item(item_name=ready_name, payload=pending_payload)

    assert len(client.ingest_calls) == 1
    assert client.ingest_calls[0]["source"]["source_round_id"] == "10:11"
    assert not (runtime._spool.ready_dir / ready_name).exists()


def test_post_llm_call_skips_extraction_purpose(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path, with_spool=True)
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

    runtime.on_post_llm_call(
        session_id="s1",
        user_message="вопрос",
        assistant_response="ответ",
        conversation_history=[],
        model="model",
        platform="x",
        llm=llm,
        purpose="ledgermind.atom.extract",
    )

    assert len(list((runtime._spool.ready_dir).glob("*.json"))) == 0
    assert len(list((runtime._spool.pending_dir).glob("*.json"))) == 0
    assert not llm.calls


def test_post_llm_call_ignores_recursive_call_via_context_guard(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path, with_spool=True)
    token = _EXTRACTION_GUARD.set(True)
    try:
        runtime.on_post_llm_call(
            session_id="s1",
            user_message="вопрос",
            assistant_response="ответ",
            conversation_history=[],
            model="model",
            platform="x",
            llm=_FakeLLM(
                payload={
                    "has_knowledge": True,
                    "title": "title",
                    "target": "target",
                    "statement": "statement",
                    "rationale": "rationale",
                    "result": "result",
                    "artifacts": [],
                },
            ),
        )
    finally:
        _EXTRACTION_GUARD.reset(token)

    assert len(list((runtime._spool.ready_dir).glob("*.json"))) == 0
    assert len(list((runtime._spool.pending_dir).glob("*.json"))) == 0


def test_post_llm_call_uses_turn_id_for_fallback_round_reference(tmp_path: Path, monkeypatch: object) -> None:
    runtime = _build_runtime(tmp_path, with_spool=True)
    monkeypatch.setattr(
        hook_runtime,
        "resolve_round_reference",
        lambda **_: (_ for _ in ()).throw(RuntimeError("forced error")),
    )

    runtime.on_post_llm_call(
        session_id="s1",
        user_message="вопрос",
        assistant_response="ответ",
        conversation_history=[],
        model="model",
        platform="x",
        llm=_FakeLLM(
            payload={
                "has_knowledge": True,
                "title": "title",
                "target": "target",
                "statement": "statement",
                "rationale": "rationale",
                "result": "result",
                "artifacts": [],
            },
        ),
        turn_id="turn-123",
    )

    pending_items = list((runtime._spool.pending_dir).glob("*.json"))
    assert len(pending_items) == 1
    payload = json.loads(pending_items[0].read_text(encoding="utf-8"))
    assert payload["source_reference"]["source_round_id"] == "turn-123"
