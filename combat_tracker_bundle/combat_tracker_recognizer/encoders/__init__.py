from combat_tracker_recognizer.encoders.handcrafted import HandcraftedEncoder
from combat_tracker_recognizer.encoders.protocol import Encoder
from combat_tracker_recognizer.encoders.registry import (
    get_encoder,
    list_encoders,
    register_encoder,
)

# Self-register the default encoder at import time.
register_encoder("handcrafted_v1", HandcraftedEncoder)

__all__ = [
    "Encoder",
    "HandcraftedEncoder",
    "get_encoder",
    "list_encoders",
    "register_encoder",
]
