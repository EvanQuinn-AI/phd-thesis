"""ClipStore: SQLite persistence for clips, labels, and bank snapshots."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from datetime import datetime
from typing import Iterable, Optional

import lz4.frame
import numpy as np

from combat_tracker_recognizer.store.migrations import run_migrations
from combat_tracker_recognizer.types import (
    Clip,
    Embedding,
    GateDecision,
    KeypointFormat,
    PoseWindow,
)


def _compress_array(arr: np.ndarray, dtype: str = "float16") -> bytes:
    return lz4.frame.compress(arr.astype(dtype, copy=False).tobytes())


def _decompress_array(blob: bytes, dtype: str, shape: tuple) -> np.ndarray:
    raw = lz4.frame.decompress(blob)
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


class ClipStore:
    def __init__(self, db_path: str, pose_dtype: str = "float16"):
        self.db_path = db_path
        self.pose_dtype = pose_dtype
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys=ON;")
        run_migrations(self._conn)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ClipStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ---- Clips ---------------------------------------------------------

    def store_clip(self, clip: Clip, gate_decision: GateDecision) -> int:
        T, K = clip.pose.points.shape[:2]
        cur = self._conn.execute(
            """
            INSERT INTO clips (
                parent_class, pose_blob, pose_scores_blob, pose_shape,
                embedding_blob, encoder_version, video_ref, frame_start,
                frame_end, fps, source_track_id, session_id,
                similarity_scores_json, gate_decision, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clip.parent_class,
                _compress_array(clip.pose.points, self.pose_dtype),
                _compress_array(clip.pose.scores, self.pose_dtype),
                json.dumps({"T": int(T), "K": int(K),
                            "format": clip.pose.keypoint_format.value}),
                clip.embedding.vector.astype(np.float32).tobytes(),
                clip.embedding.encoder_version,
                clip.video_ref,
                clip.pose.frame_start,
                clip.pose.frame_end,
                clip.pose.fps,
                clip.source_track_id,
                clip.session_id,
                json.dumps(clip.similarity_scores) if clip.similarity_scores else None,
                gate_decision.value,
                clip.created_at.isoformat(),
            ),
        )
        self._conn.commit()
        clip.id = int(cur.lastrowid)
        return clip.id

    def _row_to_clip(self, row: sqlite3.Row) -> Clip:
        shape = json.loads(row["pose_shape"])
        T, K = shape["T"], shape["K"]
        fmt = KeypointFormat(shape.get("format", KeypointFormat.MEDIAPIPE_13.value))
        points = _decompress_array(row["pose_blob"], self.pose_dtype, (T, K, 2)).astype(np.float32)
        scores = _decompress_array(row["pose_scores_blob"], self.pose_dtype, (T, K)).astype(np.float32)
        embedding_vec = np.frombuffer(row["embedding_blob"], dtype=np.float32).copy()
        pose = PoseWindow(
            points=points, scores=scores, fps=row["fps"],
            frame_start=row["frame_start"], frame_end=row["frame_end"],
            keypoint_format=fmt,
        )
        embedding = Embedding(
            vector=embedding_vec,
            encoder_version=row["encoder_version"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
        sim_json = row["similarity_scores_json"]
        return Clip(
            id=row["id"],
            parent_class=row["parent_class"],
            pose=pose,
            embedding=embedding,
            session_id=row["session_id"],
            video_ref=row["video_ref"],
            source_track_id=row["source_track_id"],
            similarity_scores=json.loads(sim_json) if sim_json else {},
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def get_clip(self, clip_id: int) -> Clip:
        self._conn.row_factory = sqlite3.Row
        row = self._conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
        if row is None:
            raise KeyError(f"clip {clip_id} not found")
        return self._row_to_clip(row)

    def get_unlabeled(
        self,
        parent: Optional[str] = None,
        session_id: Optional[str] = None,
        decision: Optional[GateDecision] = None,
        limit: Optional[int] = None,
    ) -> list[Clip]:
        self._conn.row_factory = sqlite3.Row
        # "Unlabeled" = needs review = no row in `labels` at all (neither
        # a real label nor a discard mark). Once a clip has been triaged
        # in either direction, it leaves the review queue.
        sql = """
            SELECT * FROM clips
            WHERE id NOT IN (SELECT clip_id FROM labels)
        """
        params: list = []
        if parent:
            sql += " AND parent_class = ?"
            params.append(parent)
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        if decision:
            sql += " AND gate_decision = ?"
            params.append(decision.value)
        sql += " ORDER BY id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_clip(r) for r in rows]

    def get_labeled(
        self,
        parent: Optional[str] = None,
        subclass: Optional[str] = None,
    ) -> list[tuple[Clip, str]]:
        self._conn.row_factory = sqlite3.Row
        sql = """
            SELECT c.*, l.subclass AS label_subclass
            FROM clips c
            JOIN labels l ON l.clip_id = c.id
            WHERE l.is_discarded = 0
        """
        params: list = []
        if parent:
            sql += " AND c.parent_class = ?"
            params.append(parent)
        if subclass:
            sql += " AND l.subclass = ?"
            params.append(subclass)
        # Latest label per clip wins.
        sql += " ORDER BY c.id, l.id DESC"
        rows = self._conn.execute(sql, params).fetchall()
        seen: set[int] = set()
        out: list[tuple[Clip, str]] = []
        for r in rows:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            out.append((self._row_to_clip(r), r["label_subclass"]))
        return out

    # ---- Labels --------------------------------------------------------

    def label_clip(self, clip_id: int, subclass: str, parent_class: str,
                   labeled_by: Optional[str] = None,
                   session_id: Optional[str] = None,
                   note: Optional[str] = None) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO labels (clip_id, subclass, parent_class, labeled_at,
                                labeled_by, session_id, note, is_discarded)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (clip_id, subclass, parent_class, datetime.utcnow().isoformat(),
             labeled_by, session_id, note),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def discard_clip(self, clip_id: int, note: Optional[str] = None,
                     labeled_by: Optional[str] = None) -> int:
        # Look up parent_class from clips for the labels FK row.
        row = self._conn.execute(
            "SELECT parent_class FROM clips WHERE id = ?", (clip_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"clip {clip_id} not found")
        cur = self._conn.execute(
            """
            INSERT INTO labels (clip_id, subclass, parent_class, labeled_at,
                                labeled_by, session_id, note, is_discarded)
            VALUES (?, ?, ?, ?, ?, NULL, ?, 1)
            """,
            (clip_id, "__discarded__", row[0], datetime.utcnow().isoformat(),
             labeled_by, note),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def relabel_clip(self, clip_id: int, new_subclass: str, parent_class: str,
                     labeled_by: Optional[str] = None) -> int:
        return self.label_clip(clip_id, new_subclass, parent_class, labeled_by=labeled_by)

    # ---- Bank snapshots ------------------------------------------------

    def save_bank_snapshot(self, bank_blob: bytes, encoder_version: str,
                           note: Optional[str] = None) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO prototype_versions (encoder_version, bank_blob, note, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (encoder_version, bank_blob, note, datetime.utcnow().isoformat()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def load_bank_snapshot(self, snapshot_id: int) -> bytes:
        row = self._conn.execute(
            "SELECT bank_blob FROM prototype_versions WHERE id = ?", (snapshot_id,)
        ).fetchone()
        if row is None:
            raise KeyError(snapshot_id)
        return row[0]

    def list_bank_snapshots(self) -> list[dict]:
        self._conn.row_factory = sqlite3.Row
        rows = self._conn.execute(
            "SELECT id, encoder_version, note, created_at FROM prototype_versions ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- Import / export ----------------------------------------------

    def export(self, path: str) -> None:
        self._conn.commit()
        shutil.copy(self.db_path, path)

    def import_(self, path: str, merge: bool = False) -> None:
        if merge:
            raise NotImplementedError("merge=True is not yet implemented; use merge=False to replace")
        self.close()
        shutil.copy(path, self.db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys=ON;")
        run_migrations(self._conn)
