CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_class TEXT NOT NULL,
    pose_blob BLOB NOT NULL,
    pose_scores_blob BLOB NOT NULL,
    pose_shape TEXT NOT NULL,
    embedding_blob BLOB NOT NULL,
    encoder_version TEXT NOT NULL,
    video_ref TEXT,
    frame_start INTEGER,
    frame_end INTEGER,
    fps REAL,
    source_track_id INTEGER,
    session_id TEXT NOT NULL,
    similarity_scores_json TEXT,
    gate_decision TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_clips_session ON clips(session_id);
CREATE INDEX IF NOT EXISTS idx_clips_parent ON clips(parent_class);
CREATE INDEX IF NOT EXISTS idx_clips_decision ON clips(gate_decision);

CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_id INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    subclass TEXT NOT NULL,
    parent_class TEXT NOT NULL,
    labeled_at TEXT NOT NULL,
    labeled_by TEXT,
    session_id TEXT,
    note TEXT,
    is_discarded INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_labels_clip ON labels(clip_id);

CREATE TABLE IF NOT EXISTS prototype_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    encoder_version TEXT NOT NULL,
    bank_blob BLOB NOT NULL,
    note TEXT,
    created_at TEXT NOT NULL
);
