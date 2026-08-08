"""Closed technical profile slots and their resolution to configured profiles."""

from __future__ import annotations

from enum import Enum

from .profile_store import InferenceProfileStore
from .profiles import InferenceProfile, MemorySpaceInferenceProfiles


class ProfileSlot(str, Enum):
    """Closed set of technical slots selecting a configured inference profile.

    A slot is a selection of a configured profile, not a domain facet.
    """

    OPERATIONAL = "operational"
    BACKGROUND = "background"
    EMBEDDING = "embedding"

    def __str__(self) -> str:
        return self.value


class MissingProfileError(RuntimeError):
    """Structured failure raised when a slot has no resolvable profile."""

    code = "profile_missing"

    def __init__(
        self,
        *,
        slot: ProfileSlot,
        memory_space_id: str,
        profile_id: str | None = None,
        reason: str = "profile_not_found",
    ) -> None:
        self.slot = slot
        self.memory_space_id = memory_space_id
        self.profile_id = profile_id
        self.reason = reason
        super().__init__(
            f"no {slot.value} profile for memory space {memory_space_id!r}"
            + (f" (profile {profile_id!r} {reason})" if profile_id else f" ({reason})")
        )


class ProfileResolver:
    """Service interface selecting a configured profile for a technical slot."""

    def resolve_profile(
        self, memory_space_id: str, slot: ProfileSlot
    ) -> InferenceProfile:
        """Return the configured profile for ``slot`` or raise MissingProfileError."""
        raise NotImplementedError


class StoreBackedProfileResolver(ProfileResolver):
    """Base resolution over the existing inference profile store.

    Extension point for C2: real storage/settings wiring can subclass this
    resolver (or implement ProfileResolver directly) without touching A3 code.
    """

    def __init__(
        self,
        profile_store: InferenceProfileStore,
        *,
        embedding_profile_id: str | None = None,
    ) -> None:
        self._profile_store = profile_store
        self._embedding_profile_id = embedding_profile_id

    def resolve_profile(
        self, memory_space_id: str, slot: ProfileSlot
    ) -> InferenceProfile:
        binding = self._profile_store.get_binding(memory_space_id)
        if binding is None:
            raise MissingProfileError(
                slot=slot,
                memory_space_id=memory_space_id,
                reason="binding_missing",
            )
        profile_id = self._profile_id_for_slot(binding, slot)
        if profile_id is None:
            raise MissingProfileError(
                slot=slot,
                memory_space_id=memory_space_id,
                reason="profile_id_missing",
            )
        profile = self._profile_store.get(profile_id)
        if profile is None:
            raise MissingProfileError(
                slot=slot,
                memory_space_id=memory_space_id,
                profile_id=profile_id,
                reason="profile_not_found",
            )
        if not profile.enabled:
            raise MissingProfileError(
                slot=slot,
                memory_space_id=memory_space_id,
                profile_id=profile_id,
                reason="profile_disabled",
            )
        return profile

    def _profile_id_for_slot(
        self, binding: MemorySpaceInferenceProfiles, slot: ProfileSlot
    ) -> str | None:
        if slot is ProfileSlot.OPERATIONAL:
            return binding.hypothesis_profile_id
        if slot is ProfileSlot.BACKGROUND:
            return binding.merge_profile_id
        if slot is ProfileSlot.EMBEDDING:
            return self._embedding_profile_id
        return None


__all__ = [
    "MissingProfileError",
    "ProfileResolver",
    "ProfileSlot",
    "StoreBackedProfileResolver",
]
