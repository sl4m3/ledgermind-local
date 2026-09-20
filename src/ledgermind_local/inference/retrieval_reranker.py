"""Local-only reranking and complete-item packing of Core retrieval candidates."""

from __future__ import annotations

import math
import multiprocessing
import re
import sys
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

import httpx

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"


class Reranker(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]: ...


class ApiReranker:
    """Bounded HTTP reranker supporting Cohere and NVIDIA NIM envelopes."""

    def __init__(self, endpoint: str, token: str, model: str, *, timeout_seconds: float) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.token = token
        self.model = model
        self.timeout_seconds = timeout_seconds

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        operation = self.endpoint.casefold().rsplit("/", 1)[-1]
        ranking_api = operation in {"ranking", "reranking"}
        payload: dict[str, object]
        if ranking_api:
            payload = {
                "model": self.model,
                "query": {"text": query},
                "passages": [{"text": document} for document in documents],
            }
        else:
            payload = {
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": len(documents),
            }
        try:
            response = httpx.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError("reranker API request failed") from exc
        if not isinstance(body, Mapping):
            raise TypeError("reranker API response must be an object")
        rows = body.get("rankings" if ranking_api else "results")
        if not isinstance(rows, list):
            raise TypeError("reranker API response has no ranked results")
        scores: list[float | None] = [None] * len(documents)
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError("reranker API result is invalid")
            index = row.get("index")
            score = row.get("logit" if ranking_api else "relevance_score")
            if not isinstance(index, int) or isinstance(index, bool):
                raise TypeError("reranker API result index is invalid")
            if index < 0 or index >= len(scores) or scores[index] is not None:
                raise ValueError("reranker API result index is out of range or duplicated")
            if not isinstance(score, (float, int)) or isinstance(score, bool):
                raise TypeError("reranker API result score is invalid")
            numeric = float(score)
            if not math.isfinite(numeric):
                raise ValueError("reranker API result score is not finite")
            scores[index] = numeric
        if any(score is None for score in scores):
            raise ValueError("reranker API returned incomplete scores")
        return [float(score) for score in scores if score is not None]

    def close(self) -> None:
        """Keep the lifecycle identical to the isolated local worker."""


def candidate_document(item: Mapping[str, Any]) -> str:
    """Only public, Core-admitted candidate fields are visible to the model."""
    fields = (
        ("object", item.get("object_name")),
        ("facet", item.get("facet")),
        ("content", item.get("content")),
        ("conditions", "; ".join(
            str(row.get("surface_text", "")) for row in item.get("conditions", ())
            if isinstance(row, Mapping) and row.get("surface_text")
        )),
        ("scope", item.get("scope_text")
         or " / ".join(str(part) for part in item.get("target_breadcrumb", ()) or ())
         or item.get("target_name")),
    )
    return "\n".join(f"{key}: {value}" for key, value in fields if value)


def rank_items(query: str, items: list[dict[str, Any]], scorer: Reranker) -> list[dict[str, Any]]:
    ids = [str(item["value_id"]) for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate Core candidate ID")
    scores = scorer.score(query, [candidate_document(item) for item in items])
    if len(scores) != len(items) or any(
        not isinstance(score, (float, int)) or isinstance(score, bool)
        or not math.isfinite(float(score)) for score in scores
    ):
        raise ValueError("reranker returned incomplete or non-finite scores")
    return [items[index] for index in sorted(
        range(len(items)), key=lambda index: (-float(scores[index]), index)
    )]


def render_injection(items: list[dict[str, Any]], original: Mapping[str, Any],
                     language: str | None) -> dict[str, Any]:
    if not items:
        return {"format": "facet_legend", "text": "", "legend": [], "item_count": 0}
    original_text = str(original.get("text", ""))
    heading = "\n".join(original_text.splitlines()[:2])
    legends = {str(line).split(" — ", 1)[0]: str(line)
               for line in original.get("legend", ()) if " — " in str(line)}
    facets = sorted({str(item["facet"]) for item in items})
    if not heading or any(facet not in legends for facet in facets):
        raise ValueError("Core injection legend is incomplete")
    locale = (language or "en").replace("_", "-").split("-", 1)[0].lower()
    marker = {"ru": "при условии", "es": "cuando"}.get(locale, "when")
    lines = []
    for index, item in enumerate(items, 1):
        line = f"M{index}: [{item['facet']}] {item['object_name']}: {str(item['content']).strip()}"
        conditions = [str(row.get("surface_text", "")).strip()
                      for row in item.get("conditions", ()) if isinstance(row, Mapping)
                      and str(row.get("surface_text", "")).strip()]
        if conditions:
            line += f" | {marker} " + "; ".join(conditions)
        lines.append(line)
    legend = [legends[facet] for facet in facets]
    return {"format": "facet_legend", "text": "\n".join([heading, *legend, "", *lines]),
            "legend": legend, "item_count": len(items)}


def pack_items(items: list[dict[str, Any]], original: Mapping[str, Any],
               language: str | None, *, min_k: int = 6,
               soft_budget: int = 400) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    if min_k < 1 or soft_budget < 1:
        raise ValueError("invalid injection policy")
    selected = list(items[:min_k])
    for item in items[min_k:]:
        trial = [*selected, item]
        if len(TOKEN_RE.findall(render_injection(trial, original, language)["text"])) <= soft_budget:
            selected = trial
    injection = render_injection(selected, original, language)
    return selected, injection, len(TOKEN_RE.findall(injection["text"]))


class QwenLocalReranker:
    """Pinned Qwen model in an explicitly configured local CPU/CUDA/ROCm runtime.

    The path must already contain a verified, locally installed model. No
    Hugging Face download or external inference is possible in this class.
    """

    def __init__(self, model_path: str | Path, device: str) -> None:
        path = Path(model_path).expanduser().resolve(strict=True)
        if path.name != MODEL_REVISION or path.parent.name != "snapshots":
            raise ValueError("reranker requires the pinned local Qwen snapshot")
        if device not in {"cpu", "cuda", "rocm"}:
            raise ValueError("unsupported reranker device")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if device in {"cuda", "rocm"} else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("configured reranker GPU is unavailable")
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, dtype=torch.float32 if device == "cpu" else torch.float16
        ).eval().to(self.device)
        self.false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.true_id = self.tokenizer.convert_tokens_to_ids("yes")
        if self.false_id is None or self.true_id is None or self.false_id == self.true_id:
            raise ValueError("reranker yes/no tokens are unavailable")
        self.prefix = self.tokenizer.encode(
            '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
            'and the Instruct provided. Note that the answer can only be "yes" or "no".'
            '<|im_end|>\n<|im_start|>user\n', add_special_tokens=False,
        )
        self.suffix = self.tokenizer.encode(
            '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',
            add_special_tokens=False,
        )
        self._lock = threading.Lock()

    def score(self, query: str, documents: list[str]) -> list[float]:
        scores = []
        max_pair = 2048 - len(self.prefix) - len(self.suffix)
        with self._lock, self.torch.inference_mode():
            for document in documents:
                pair = ("<Instruct>: Given a web search query, retrieve relevant passages that "
                        f"answer the query\n<Query>: {query}\n<Document>: {document}")
                middle = self.tokenizer.encode(pair, add_special_tokens=False)
                if len(middle) > max_pair:
                    raise ValueError("reranker pair exceeds the 2048-token window")
                tokens = self.prefix + middle + self.suffix
                inputs = self.torch.tensor([tokens], dtype=self.torch.long, device=self.device)
                logits = self.model(input_ids=inputs).logits[0, -1]
                binary = logits[[self.false_id, self.true_id]].float()
                scores.append(float(self.torch.softmax(binary, dim=0)[1]))
        return scores


def _worker_main(connection: Any, model_path: str, device: str,
                 runtime_path: str | None = None) -> None:
    """Inference child: never opens sockets or receives Core credentials."""
    try:
        if runtime_path:
            packages = Path(runtime_path).resolve(strict=True) / "site-packages"
            if not packages.is_dir():
                raise OSError("reranker runtime site-packages missing")
            sys.path.insert(0, str(packages))
        model = QwenLocalReranker(model_path, device)
        connection.send(("ready", None))
        while True:
            message = connection.recv()
            if message is None:
                break
            query, documents = message
            try:
                connection.send(("scores", model.score(query, documents)))
            except (RuntimeError, ValueError) as exc:
                connection.send(("error", type(exc).__name__))
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        try:
            connection.send(("error", type(exc).__name__))
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


class QwenWorkerReranker:
    """Bounded, restartable local model process, separate from the Local API."""

    def __init__(
        self, model_path: str, device: str, *,
        startup_timeout: float = 60, score_timeout: float = 30,
        worker_target: Callable[..., None] = _worker_main,
        runtime_path: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.startup_timeout = startup_timeout
        self.score_timeout = score_timeout
        self.worker_target = worker_target
        self.runtime_path = runtime_path
        self._process: multiprocessing.Process | None = None
        self._connection: Any | None = None
        self._lock = threading.Lock()

    def _stop(self) -> None:
        process, connection = self._process, self._connection
        self._process, self._connection = None, None
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        if connection is not None:
            connection.close()

    def close(self) -> None:
        with self._lock:
            self._stop()

    def _ensure_started(self) -> Any:
        if self._process is not None and self._process.is_alive():
            return self._connection
        self._stop()
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        args = (child, self.model_path, self.device)
        if self.runtime_path is not None:
            args += (self.runtime_path,)
        process = context.Process(target=self.worker_target, args=args, daemon=True)
        process.start()
        child.close()
        self._process, self._connection = process, parent
        if not parent.poll(self.startup_timeout):
            self._stop()
            raise TimeoutError("local reranker startup timeout")
        try:
            status, detail = parent.recv()
        except EOFError as exc:
            self._stop()
            raise RuntimeError("local reranker exited during startup") from exc
        if status != "ready":
            self._stop()
            raise RuntimeError(f"local reranker startup failed: {detail}")
        return parent

    def score(self, query: str, documents: list[str]) -> list[float]:
        with self._lock:
            connection = self._ensure_started()
            try:
                connection.send((query, documents))
                if not connection.poll(self.score_timeout):
                    self._stop()
                    raise TimeoutError("local reranker scoring timeout")
                status, payload = connection.recv()
            except TimeoutError:
                raise
            except (EOFError, OSError) as exc:
                self._stop()
                raise RuntimeError("local reranker process exited") from exc
            if status != "scores":
                raise RuntimeError(f"local reranker scoring failed: {payload}")
            return payload
