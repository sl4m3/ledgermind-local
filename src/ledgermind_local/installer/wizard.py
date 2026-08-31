"""Guided terminal installer backed by the same strict config model."""

from __future__ import annotations

import getpass
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TextIO, cast

from .errors import UserCancelledError
from .models import (
    AdvancedConfig,
    EmbeddingApiConfig,
    EmbeddingConfig,
    GenerationConfig,
    InstallerConfig,
    IntegrationConfig,
    LocalEmbeddingConfig,
    RuntimeConfig,
)
from .provider_profiles import GenerationProviderProfileId

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
REFERENCE_GENERATION_MODEL = "deepseek/deepseek-v4-flash-0731"
REFERENCE_NVIDIA_GENERATION_MODEL = "nvidia/nemotron-3-super-120b-a12b"
REFERENCE_EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b:free"
REFERENCE_EMBEDDING_DIMENSIONS = 2048


@dataclass(frozen=True, slots=True)
class _Choice:
    value: str
    label: str
    detail: str = ""


class _TerminalWizard:
    def __init__(
        self,
        *,
        input_fn: Callable[[str], str],
        secret_fn: Callable[[str], str],
        output: TextIO,
    ) -> None:
        self.input_fn = input_fn
        self.secret_fn = secret_fn
        self.output = output
        self.color = bool(getattr(output, "isatty", lambda: False)())

    def _style(self, text: str, code: str) -> str:
        if not self.color:
            return text
        return f"\033[{code}m{text}\033[0m"

    def line(self, text: str = "") -> None:
        print(text, file=self.output)

    def banner(self) -> None:
        self.line()
        self.line(self._style("  LEDGERMIND SETUP", "1;36"))
        self.line("  Private memory for your local agents")
        self.line("  " + "─" * 48)

    def section(self, number: int, title: str, detail: str) -> None:
        self.line()
        self.line(self._style(f"  {number}/5  {title}", "1;34"))
        self.line(f"  {detail}")

    def ask(self, prompt: str, *, default: str | None = None) -> str:
        suffix = f" [{default}]" if default else ""
        value = self.input_fn(f"  {prompt}{suffix}: ").strip()
        return value or (default or "")

    def required(self, prompt: str, *, default: str | None = None) -> str:
        while True:
            value = self.ask(prompt, default=default)
            if value:
                return value
            self.line(self._style("  A value is required.", "31"))

    def secret(self, prompt: str) -> str:
        while True:
            value = self.secret_fn(f"  {prompt}: ").strip()
            if value:
                return value
            self.line(self._style("  A token is required.", "31"))

    def choose(
        self,
        prompt: str,
        choices: Sequence[_Choice],
        *,
        default: int = 1,
    ) -> str:
        self.line(f"  {prompt}")
        for index, choice in enumerate(choices, start=1):
            marker = self._style(str(index), "1;36")
            detail = f" — {choice.detail}" if choice.detail else ""
            self.line(f"    {marker}. {choice.label}{detail}")
        while True:
            raw = self.ask("Select", default=str(default)).lower()
            if raw.isdigit() and 1 <= int(raw) <= len(choices):
                return choices[int(raw) - 1].value
            for choice in choices:
                if raw == choice.value.lower():
                    return choice.value
            self.line(self._style("  Select one of the listed options.", "31"))

    def confirm(self, prompt: str, *, default: bool = True) -> bool:
        answer = self.ask(prompt, default="yes" if default else "no").lower()
        while answer not in {"yes", "no", "y", "n"}:
            self.line(self._style("  Enter yes or no.", "31"))
            answer = self.ask(prompt, default="yes" if default else "no").lower()
        return answer in {"yes", "y"}

    def integrations(self, discovered: Sequence[tuple[str, str]]) -> tuple[str, ...]:
        if not discovered:
            self.line("  No supported agent installations were detected.")
            return ()
        self.line("  Detected agents")
        for index, (_, label) in enumerate(discovered, start=1):
            self.line(f"    {self._style(str(index), '1;36')}. {label}")
        self.line("  Enter comma-separated numbers, 'all', or 'none'.")
        while True:
            raw = self.ask("Connect", default="all").lower()
            if raw == "all":
                return tuple(identifier for identifier, _ in discovered)
            if raw == "none":
                return ()
            try:
                indexes = {int(item.strip()) for item in raw.split(",") if item.strip()}
            except ValueError:
                indexes = set()
            if indexes and all(1 <= index <= len(discovered) for index in indexes):
                return tuple(
                    identifier
                    for index, (identifier, _) in enumerate(discovered, start=1)
                    if index in indexes
                )
            self.line(
                self._style("  Use numbers such as 1,3, or choose all/none.", "31")
            )


def build_interactive_config(
    *,
    input_fn: Callable[[str], str] = input,
    secret_fn: Callable[[str], str] = getpass.getpass,
    output: TextIO = sys.stderr,
    existing_config: InstallerConfig | None = None,
) -> InstallerConfig:
    ui = _TerminalWizard(input_fn=input_fn, secret_fn=secret_fn, output=output)
    try:
        ui.banner()
        if existing_config is None:
            ui.section(1, "LANGUAGE", "Language used by semantic memory")
            semantic_language = ui.choose(
                "Choose a semantic language",
                (
                    _Choice("ru", "Russian"),
                    _Choice("en", "English"),
                    _Choice("es", "Spanish"),
                    _Choice("pt", "Portuguese"),
                    _Choice("fr", "French"),
                    _Choice("de", "German"),
                    _Choice("uk", "Ukrainian"),
                ),
                default=1,
            )
        else:
            semantic_language = existing_config.semantic_language
            ui.line(
                "  Reconfiguring providers; memory layout and agents are preserved."
            )

        ui.section(2, "GENERATION", "Semantic extraction and knowledge resolution")
        generation_provider = ui.choose(
            "Generation provider",
            (
                _Choice("openrouter", "OpenRouter", "tested reference configuration"),
                _Choice("nvidia_nim", "NVIDIA NIM", "native guided JSON"),
                _Choice("custom", "Custom OpenAI-compatible endpoint"),
            ),
        )
        if generation_provider == "openrouter":
            endpoint = OPENROUTER_BASE_URL
            ui.line(f"  API base: {endpoint}")
            route = None
            fallback_routes: tuple[str, ...] = ()
            default_generation_model = REFERENCE_GENERATION_MODEL
        elif generation_provider == "nvidia_nim":
            endpoint = NVIDIA_NIM_BASE_URL
            route = None
            fallback_routes = ()
            default_generation_model = REFERENCE_NVIDIA_GENERATION_MODEL
            ui.line(f"  API base: {endpoint}")
        else:
            endpoint = ui.required("API base URL (for example https://host.example/v1)")
            route = None
            fallback_routes = ()
            default_generation_model = None
        token = ui.secret("Generation token")
        model = ui.required("Generation model", default=default_generation_model)
        object_resolution_model = ui.required("Object Resolution model", default=model)
        if generation_provider == "openrouter":
            route_count = ui.choose(
                "OpenRouter routing",
                (
                    _Choice("one", "One provider", "no fallback"),
                    _Choice("two", "Primary + fallback", "two providers maximum"),
                ),
            )
            route = ui.required(
                "Primary OpenRouter provider (for example provider/variant)"
            )
            if route_count == "two":
                while True:
                    fallback = ui.required(
                        "Fallback OpenRouter provider (for example provider/variant)"
                    )
                    if fallback != route:
                        fallback_routes = (fallback,)
                        break
                    ui.line(
                        ui._style(
                            "  Fallback provider must differ from the primary provider.",
                            "31",
                        )
                    )

        ui.section(3, "EMBEDDINGS", "Retrieval model and vector dimensions")
        embedding_mode = ui.choose(
            "Embedding source",
            (
                _Choice("api", "API endpoint", "quickest setup"),
                _Choice("local", "Local signed model", "private and offline"),
            ),
        )
        if embedding_mode == "api":
            embedding_endpoint = ui.required("Embedding API base", default=endpoint)
            reuse_token = embedding_endpoint.rstrip("/") == endpoint.rstrip(
                "/"
            ) and ui.confirm("Reuse the generation token", default=True)
            embedding_token = token if reuse_token else ui.secret("Embedding token")
            default_embedding_model = (
                REFERENCE_EMBEDDING_MODEL
                if "openrouter.ai" in embedding_endpoint
                else None
            )
            embedding_model = ui.required(
                "Embedding model", default=default_embedding_model
            )
            default_dimensions = (
                REFERENCE_EMBEDDING_DIMENSIONS
                if embedding_model.removesuffix(":free")
                == REFERENCE_EMBEDDING_MODEL.removesuffix(":free")
                else 1536
            )
            dimensions = int(
                ui.required("Embedding dimensions", default=str(default_dimensions))
            )
            embedding = EmbeddingConfig(
                mode="api",
                api=EmbeddingApiConfig(
                    endpoint=embedding_endpoint,
                    token=embedding_token,
                    model=embedding_model,
                    dimensions=dimensions,
                ),
            )
        else:
            catalog_id = ui.required("Signed local model id")
            device = ui.choose(
                "Compute device",
                (
                    _Choice("auto", "Automatic"),
                    _Choice("cpu", "CPU"),
                    _Choice("cuda", "NVIDIA CUDA"),
                    _Choice("rocm", "AMD ROCm"),
                ),
            )
            embedding = EmbeddingConfig(
                mode="local",
                local=LocalEmbeddingConfig(
                    catalog_id=catalog_id,
                    device=cast(Literal["auto", "cpu", "cuda", "rocm"], device),
                ),
            )

        if existing_config is None:
            ui.section(4, "MEMORY", "Choose whether connected agents share knowledge")
            memory_mode = ui.choose(
                "Memory layout",
                (
                    _Choice("per_agent", "Separate per agent", "strong isolation"),
                    _Choice("shared", "Shared by all agents", "one knowledge space"),
                ),
            )
            customize_runtime = ui.confirm(
                "Customize runtime lifecycle settings", default=False
            )
            if customize_runtime:
                runtime = RuntimeConfig(
                    idle_shutdown_seconds=float(
                        ui.required("Idle shutdown seconds", default="60")
                    ),
                    lease_ttl_seconds=float(
                        ui.required("Lease TTL seconds", default="30")
                    ),
                    heartbeat_seconds=float(
                        ui.required("Heartbeat seconds", default="10")
                    ),
                )
            else:
                runtime = RuntimeConfig()

            ui.section(5, "AGENTS", "Select installed agents to connect")
            from .targets.registry import get_target_adapter, target_ids

            discovered: list[tuple[str, str]] = []
            for target_id in target_ids():
                adapter = get_target_adapter(target_id)
                if adapter.discover().detected:
                    discovered.append((target_id, adapter.label))
            selected = ui.integrations(discovered)
            integrations = tuple(
                IntegrationConfig.model_validate({"id": target_id, "enabled": True})
                for target_id in selected
            )
        else:
            memory_mode = existing_config.memory_mode
            runtime = existing_config.runtime
            integrations = existing_config.integrations

        ui.line()
        ui.line(ui._style("  REVIEW", "1;34"))
        ui.line(f"  Provider:   {generation_provider}")
        ui.line(f"  Generation: {model}")
        if route:
            route_chain = " → ".join((route, *fallback_routes))
            ui.line(f"  Routes:     {route_chain} (restricted)")
        ui.line(f"  Embeddings:  {embedding.mode}")
        if embedding.api is not None:
            ui.line(
                f"               {embedding.api.model} ({embedding.api.dimensions}d)"
            )
        ui.line(f"  Memory:      {memory_mode}")
        ui.line(
            "  Agents:      " + (", ".join(item.id for item in integrations) or "none")
        )
        confirmation = (
            "Apply provider configuration"
            if existing_config is not None
            else "Start installation"
        )
        if not ui.confirm(confirmation, default=True):
            raise UserCancelledError("installation cancelled by user")

        return InstallerConfig(
            semantic_language=cast(
                Literal["ru", "en", "es", "pt", "fr", "de", "uk"],
                semantic_language,
            ),
            integrations=integrations,
            memory_mode=cast(Literal["shared", "per_agent"], memory_mode),
            generation=GenerationConfig(
                provider_profile=cast(
                    GenerationProviderProfileId,
                    "openai_compatible"
                    if generation_provider == "custom"
                    else generation_provider,
                ),
                endpoint=endpoint,
                route=route,
                fallback_routes=fallback_routes,
                token=token,
                model=model,
                object_resolution_model=object_resolution_model,
            ),
            embedding=embedding,
            runtime=runtime,
            memory_data_path=(
                existing_config.memory_data_path
                if existing_config is not None
                else None
            ),
            advanced=(
                existing_config.advanced
                if existing_config is not None
                else AdvancedConfig()
            ),
        )
    except (EOFError, KeyboardInterrupt) as exc:
        raise UserCancelledError("installation cancelled by user") from exc


def choose_existing_install_action(
    *, input_fn: Callable[[str], str] = input, output: TextIO = sys.stderr
) -> str:
    """Select one safe operation when the installer finds an existing deployment."""

    ui = _TerminalWizard(input_fn=input_fn, secret_fn=getpass.getpass, output=output)
    ui.banner()
    ui.section(
        1, "EXISTING INSTALLATION", "Your memory and provider settings are preserved"
    )
    return ui.choose(
        "What would you like to do?",
        (
            _Choice("add-agent", "Add an agent", "does not change model profiles"),
            _Choice("repair", "Repair installation", "restores runtime and hooks"),
            _Choice("reconfigure", "Reconfigure providers", "keeps memory and agents"),
        ),
    )


def choose_integrations_to_connect(
    config: InstallerConfig,
    *,
    input_fn: Callable[[str], str] = input,
    output: TextIO = sys.stderr,
) -> tuple[str, ...]:
    """Choose detected agents that are not already connected."""

    from .targets.registry import get_target_adapter, target_ids

    connected = {item.id for item in config.integrations}
    available: list[tuple[str, str]] = []
    for target_id in target_ids():
        adapter = get_target_adapter(target_id)
        if target_id not in connected and adapter.discover().detected:
            available.append((target_id, adapter.label))
    ui = _TerminalWizard(input_fn=input_fn, secret_fn=getpass.getpass, output=output)
    ui.section(
        2, "AGENTS", "Existing integrations and inference profiles stay unchanged"
    )
    return ui.integrations(available)


__all__ = [
    "NVIDIA_NIM_BASE_URL",
    "OPENROUTER_BASE_URL",
    "REFERENCE_EMBEDDING_DIMENSIONS",
    "REFERENCE_EMBEDDING_MODEL",
    "REFERENCE_GENERATION_MODEL",
    "REFERENCE_NVIDIA_GENERATION_MODEL",
    "build_interactive_config",
    "choose_existing_install_action",
    "choose_integrations_to_connect",
]
