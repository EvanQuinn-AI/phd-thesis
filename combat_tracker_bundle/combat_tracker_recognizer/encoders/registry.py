"""Named-encoder registry. Encoders self-register at import time."""

from __future__ import annotations

from typing import Type

from combat_tracker_recognizer.config import EncoderConfig
from combat_tracker_recognizer.encoders.protocol import Encoder

_registry: dict[str, Type[Encoder]] = {}


def register_encoder(name: str, cls: Type[Encoder]) -> None:
    _registry[name] = cls


def get_encoder(name: str, config: EncoderConfig) -> Encoder:
    if name not in _registry:
        raise KeyError(f"unknown encoder {name!r}; registered: {sorted(_registry)}")
    return _registry[name](config)


def list_encoders() -> list[str]:
    return sorted(_registry)
