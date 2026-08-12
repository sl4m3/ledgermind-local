"""Stable machine-readable installer results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .errors import ExitCode


@dataclass(slots=True)
class ResultStep:
    name: str
    status: str
    detail: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"name": self.name, "status": self.status}
        if self.detail:
            result["detail"] = self.detail
        if self.data:
            result["data"] = dict(self.data)
        return result


@dataclass(slots=True)
class InstallResult:
    operation: str
    exit_code: ExitCode = ExitCode.SUCCESS
    status: str = "success"
    steps: list[ResultStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    paths: dict[str, Any] = field(default_factory=dict)
    profiles: list[dict[str, Any]] = field(default_factory=list)
    runtime: dict[str, Any] = field(default_factory=dict)
    smoke_test: dict[str, Any] = field(default_factory=dict)

    def step(
        self,
        name: str,
        status: str,
        detail: str | None = None,
        **data: Any,
    ) -> None:
        self.steps.append(ResultStep(name, status, detail, data))

    def warning(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)
        self.status = "failed"

    def fail(self, code: ExitCode, message: str) -> InstallResult:
        self.exit_code = code
        self.error(message)
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "operation": self.operation,
            "exit_code": int(self.exit_code),
            "steps": [step.as_dict() for step in self.steps],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "paths": dict(self.paths),
            "profiles": list(self.profiles),
            "runtime": dict(self.runtime),
            "smoke_test": dict(self.smoke_test),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True)


__all__ = ["InstallResult", "ResultStep"]
