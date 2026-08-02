CREATE TABLE projection_state (
    projection_name TEXT PRIMARY KEY,
    projection_version INTEGER NOT NULL,
    rebuilt_at TEXT,
    item_count INTEGER NOT NULL DEFAULT 0,
    checksum TEXT
);
