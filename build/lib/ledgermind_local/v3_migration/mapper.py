from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any

from ledgermind_local.v3_migration.models import LegacyRecord

_PHASE_MAP = {
    "pattern": "PATTERN",
    "emergent": "EMERGENT",
    "canonical": "CANONICAL",
}

_VITALITY_MAP = {
    "active": "ACTIVE",
    "draft": "ACTIVE",
    "enriched": "ACTIVE",
    "pending_merge": "ACTIVE",
    "deprecated": "DEPRECATED",
    "superseded": "SUPERSEDED",
    "accepted": "ARCHIVED",
    "rejected": "ARCHIVED",
    "fulfilled": "ARCHIVED",
    "falsified": "ARCHIVED",
    "deleted": "DELETED",
}

_FRONTMATTER_DELIMITERS = (("---", "---"), ("+++", "+++"))


def _normalize_space(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _phase_from(value: str | None) -> tuple[str | None, list[str]]:
    warnings: list[str] = []
    normalized = _normalize_space(value).lower()
    if not normalized:
        return "PATTERN", ["empty_phase_defaulted_to_pattern"]
    phase = _PHASE_MAP.get(normalized)
    if phase:
        return phase, warnings
    warnings.append(f"unknown_phase:{normalized}")
    return "PATTERN", warnings


def _vitality_from(value: str | None) -> tuple[str | None, list[str]]:
    warnings: list[str] = []
    normalized = _normalize_space(value).lower()
    if not normalized:
        return "ACTIVE", ["empty_vitality_defaulted_to_active"]
    vitality = _VITALITY_MAP.get(normalized)
    if vitality:
        return vitality, warnings
    warnings.append(f"unknown_vitality:{normalized}")
    return "ACTIVE", warnings


def _extract_frontmatter(raw: str | None) -> tuple[dict[str, Any], str, list[str]]:
    warnings: list[str] = []
    if not raw:
        return {}, "", warnings
    text = raw.strip()
    if not text.startswith("---") and not text.startswith("+++"):
        return {}, text, warnings
    for start_delim, end_delim in _FRONTMATTER_DELIMITERS:
        if text.startswith(start_delim):
            end = text.find(end_delim, len(start_delim))
            if end == -1:
                warnings.append("unclosed_frontmatter")
                return {}, text, warnings
            body = text[end + len(end_delim) :].strip()
            chunk = text[len(start_delim) : end].strip()
            metadata: dict[str, Any] = {}
            current_key = "body"
            current_value_lines: list[str] = []
            for line in chunk.splitlines():
                if line.startswith((" ", "\t")) and current_key is not None:
                    current_value_lines.append(line)
                    continue
                if current_key is not None and current_value_lines:
                    metadata[current_key] = "\n".join(current_value_lines).strip()
                if ":" in line:
                    key, value = line.split(":", 1)
                    current_key = key.strip().lower()
                    current_value_lines = [value.strip()]
                else:
                    current_key = "body"
                    current_value_lines = [line]
            if current_key is not None and current_value_lines:
                metadata[current_key] = "\n".join(current_value_lines).strip()
            return metadata, body, warnings
    return {}, text, warnings


def map_record(record: LegacyRecord) -> dict[str, Any]:
    frontmatter: dict[str, Any] = {}
    body = record.raw_markdown or ""
    warnings = list(record.warnings)

    if record.parsed_frontmatter is not None:
        frontmatter = deepcopy(record.parsed_frontmatter)
    elif record.markdown_exists and record.raw_markdown:
        frontmatter, body, fm_warnings = _extract_frontmatter(record.raw_markdown)
        warnings.extend(fm_warnings)

    metadata = record.metadata_row or {}
    body = _normalize_space(body)
    title = _normalize_space(frontmatter.get("title") or metadata.get("title") or record.fid)
    target = _normalize_space(
        frontmatter.get("target") or metadata.get("target") or f"legacy/{record.fid}"
    )
    rationale = _normalize_space(
        frontmatter.get("rationale")
        or frontmatter.get("compressive_rationale")
        or metadata.get("rationale")
        or metadata.get("content")
        or body
    )
    statement = _normalize_space(
        frontmatter.get("statement")
        or metadata.get("rationale")
        or metadata.get("statement")
        or rationale
        or body
    )
    legacy_status = _normalize_space(
        frontmatter.get("status")
        or metadata.get("status")
    )
    phase_value = _normalize_space(frontmatter.get("phase") or metadata.get("phase"))
    phase, phase_warnings = _phase_from(phase_value)
    vitality, vitality_warnings = _vitality_from(legacy_status)
    warnings.extend(phase_warnings)
    warnings.extend(vitality_warnings)

    if not title or not target:
        warnings.append("missing_critical_fields")
    if not statement:
        warnings.append("missing_statement")

    old_fid = record.fid
    legacy_source_hash = hashlib.sha256(
        f"{record.raw_markdown or ''}\n{record.metadata_row or {}}".encode()
    ).hexdigest()

    memory_space_id = "legacy:{}:{}".format(
        legacy_source_hash[:12],
        _normalize_space(metadata.get("namespace") or metadata.get("profile") or "default"),
    )

    artifacts: list[str] = []
    for key in ("artifacts", "attachments", "files"):
        value = frontmatter.get(key) or metadata.get(key)
        if not value:
            continue
        if isinstance(value, list):
            artifacts.extend(str(item) for item in value)
        else:
            artifacts.extend(str(value).splitlines())

    supersedes: list[str] = []
    for key in ("supersedes", "superseded"):
        value = frontmatter.get(key) or metadata.get(key)
        if not value:
            continue
        if isinstance(value, list):
            supersedes.extend(str(item) for item in value)
        else:
            supersedes.extend(str(value).splitlines())

    return {
        "source_system": "legacy_import",
        "source_instance_id": legacy_source_hash,
        "source_profile_id": _normalize_space(metadata.get("namespace") or metadata.get("profile") or "default"),
        "source_session_id": "legacy-v3",
        "source_round_id": old_fid,
        "source_digest": f"sha256:{legacy_source_hash}",
        "extraction_host": "ledgermind-v3",
        "extraction_provider": "legacy",
        "extraction_model": _normalize_space(metadata.get("extraction_model") or ""),
        "prompt_version": 1,
        "schema_version": 1,
        "memory_space_id": memory_space_id,
        "title": title,
        "target": target,
        "statement": statement,
        "rationale": rationale,
        "artifacts": artifacts,
        "supersedes": supersedes,
        "phase": phase,
        "vitality": vitality,
        "legacy_status": legacy_status,
        "warnings": warnings,
        "markdown_body": body,
    }
