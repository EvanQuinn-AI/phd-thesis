"""Bank snapshots + rollback. Phase 1: pickle to disk; Phase 2: SQLite."""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from combat_tracker_recognizer.bank.bank import PrototypeBank


_SNAPSHOT_DIR = ".recognizer_snapshots"


@dataclass
class BankSnapshot:
    snapshot_id: int
    created_at: datetime
    note: str
    encoder_version: str
    path: str


def _ensure_dir(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def save_snapshot(bank: PrototypeBank, note: str = "",
                  encoder_version: str = "",
                  directory: str = _SNAPSHOT_DIR) -> BankSnapshot:
    _ensure_dir(directory)
    snapshot_id = int(time.time() * 1000)
    path = os.path.join(directory, f"snapshot_{snapshot_id}.pkl")
    payload = {
        "store": bank._store,
        "encoder_version": encoder_version,
        "note": note,
        "created_at": datetime.utcnow(),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return BankSnapshot(snapshot_id=snapshot_id, created_at=payload["created_at"],
                        note=note, encoder_version=encoder_version, path=path)


def load_snapshot(snapshot_id: int, directory: str = _SNAPSHOT_DIR) -> PrototypeBank:
    path = os.path.join(directory, f"snapshot_{snapshot_id}.pkl")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    bank = PrototypeBank()
    bank._store = payload["store"]
    return bank


def list_snapshots(directory: str = _SNAPSHOT_DIR) -> list[BankSnapshot]:
    if not os.path.isdir(directory):
        return []
    out = []
    for fn in sorted(os.listdir(directory)):
        if not (fn.startswith("snapshot_") and fn.endswith(".pkl")):
            continue
        path = os.path.join(directory, fn)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        sid = int(fn[len("snapshot_"):-len(".pkl")])
        out.append(BankSnapshot(
            snapshot_id=sid, created_at=payload["created_at"],
            note=payload.get("note", ""),
            encoder_version=payload.get("encoder_version", ""),
            path=path,
        ))
    return out


def rollback(bank: PrototypeBank, snapshot_id: int,
             directory: str = _SNAPSHOT_DIR) -> None:
    """Replace ``bank`` state with the snapshot in place."""
    snapshot_bank = load_snapshot(snapshot_id, directory)
    bank._store = snapshot_bank._store
