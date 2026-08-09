"""Local inference profiles, secrets, providers, and broker."""

from .profile_slots import (
    DatabaseBackedProfileResolver,
    MissingProfileError,
    ProfileSlot,
    StoreBackedProfileResolver,
)
from .profile_store import InferenceProfileStore
from .profiles import InferenceProfile, MemorySpaceInferenceProfiles, ProviderKind
from .secrets import SecretNotFoundError, SecretStore

__all__ = [
    "DatabaseBackedProfileResolver",
    "InferenceBroker",
    "InferenceBrokerError",
    "InferenceInputTooLargeError",
    "InferenceProfile",
    "InferenceProfileDisabledError",
    "InferenceProfileNotFoundError",
    "InferenceProfileStore",
    "InferenceResponseValidationError",
    "MemorySpaceInferenceProfiles",
    "MissingProfileError",
    "ModelTask",
    "ProfileSlot",
    "ProviderKind",
    "SecretNotFoundError",
    "SecretStore",
    "StoreBackedProfileResolver",
]


def __getattr__(name: str) -> object:
    """Keep the legacy broker import lazy until D2 removes it."""

    if name in {
        "InferenceBroker",
        "InferenceBrokerError",
        "InferenceInputTooLargeError",
        "InferenceProfileDisabledError",
        "InferenceProfileNotFoundError",
        "InferenceResponseValidationError",
        "ModelTask",
    }:
        from . import broker

        return getattr(broker, name)
    raise AttributeError(name)
