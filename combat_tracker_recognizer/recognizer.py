"""SubclassActionRecognizer: the public plugin API."""

from __future__ import annotations

import pickle
from datetime import datetime
from typing import Optional

from combat_tracker_recognizer.bank import PrototypeBank
from combat_tracker_recognizer.config import RecognizerConfig
from combat_tracker_recognizer.encoders import get_encoder
from combat_tracker_recognizer.encoders.protocol import Encoder
from combat_tracker_recognizer.gate import NoveltyGate
from combat_tracker_recognizer.store import ClipStore
from combat_tracker_recognizer.types import (
    Clip,
    Embedding,
    GateDecision,
    PoseWindow,
    SubclassResult,
)


class SubclassActionRecognizer:
    def __init__(
        self,
        config: RecognizerConfig,
        clipstore: Optional[ClipStore] = None,
        encoder: Optional[Encoder] = None,
        bank: Optional[PrototypeBank] = None,
    ):
        self.config = config
        self.encoder = encoder or get_encoder(config.encoder.name, config.encoder)
        self.clipstore = clipstore or ClipStore(
            config.store.db_path, pose_dtype=config.store.pose_dtype
        )
        self.bank = bank or PrototypeBank()
        self.gate = NoveltyGate(self.bank, config.gate)

    # ---- Inference -----------------------------------------------------

    def observe(
        self,
        parent_class: str,
        pose_window: PoseWindow,
        source_track_id: Optional[int] = None,
        video_ref: Optional[str] = None,
        session_id: str = "default",
    ) -> SubclassResult:
        embedding_vec = self.encoder.encode(pose_window)
        decision, subclass, confidence, top_matches = self.gate.route(
            parent_class, embedding_vec
        )

        if decision == GateDecision.NOISE:
            return SubclassResult(
                decision=decision, subclass=None, confidence=confidence,
                clip_id=None, top_matches=top_matches,
            )
        if decision == GateDecision.KNOWN:
            return SubclassResult(
                decision=decision, subclass=subclass, confidence=confidence,
                clip_id=None, top_matches=top_matches,
            )

        # AMBIGUOUS or UNKNOWN: persist for review.
        clip = Clip(
            parent_class=parent_class,
            pose=pose_window,
            embedding=Embedding(
                vector=embedding_vec, encoder_version=self.encoder.version,
                created_at=datetime.utcnow(),
            ),
            session_id=session_id,
            video_ref=video_ref,
            source_track_id=source_track_id,
            similarity_scores={sc: float(d) for sc, d in top_matches},
        )
        clip_id = self.clipstore.store_clip(clip, decision)
        return SubclassResult(
            decision=decision, subclass=None, confidence=confidence,
            clip_id=clip_id, top_matches=top_matches,
        )

    # ---- Ops -----------------------------------------------------------

    def pending_review_count(
        self,
        session_id: Optional[str] = None,
        parent: Optional[str] = None,
    ) -> int:
        return len(self.clipstore.get_unlabeled(parent=parent, session_id=session_id))

    def bank_status(self) -> dict:
        snapshots = self.clipstore.list_bank_snapshots()
        return {
            "encoder_version": self.encoder.version,
            "subclass_keys": [list(k) for k in self.bank.all_subclasses()],
            "num_prototypes": self.bank.num_prototypes(),
            "last_snapshot": snapshots[0]["created_at"] if snapshots else None,
            "num_snapshots": len(snapshots),
        }

    def rebuild_bank(self, encoder_name: Optional[str] = None,
                     note: str = "rebuild") -> None:
        before = pickle.dumps(self.bank._store)
        self.clipstore.save_bank_snapshot(before, self.encoder.version,
                                          note=f"pre-{note}")
        if encoder_name and encoder_name != self.encoder.version:
            self.encoder = get_encoder(encoder_name, self.config.encoder)
        labeled = self.clipstore.get_labeled()

        class _LabeledStub:
            def __init__(self, clip, label):
                self.parent_class = clip.parent_class
                self.subclass = label
                self.pose = clip.pose

        stubs = [_LabeledStub(c, lbl) for c, lbl in labeled]
        self.bank.rebuild_from_clips(stubs, self.encoder)
        after = pickle.dumps(self.bank._store)
        self.clipstore.save_bank_snapshot(after, self.encoder.version,
                                          note=f"post-{note}")
