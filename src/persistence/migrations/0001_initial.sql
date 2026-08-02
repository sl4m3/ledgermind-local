CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    checksum TEXT NOT NULL,
    applied_at TEXT NOT NULL
);

CREATE TABLE memory_spaces (
    memory_space_id TEXT PRIMARY KEY,
    display_name TEXT,
    source_client TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (length(memory_space_id) BETWEEN 1 AND 200)
);

CREATE TABLE atoms (
    atom_id TEXT PRIMARY KEY,
    memory_space_id TEXT NOT NULL
        REFERENCES memory_spaces(memory_space_id) ON DELETE CASCADE,

    source_system TEXT NOT NULL,
    source_instance_id TEXT NOT NULL,
    source_profile_id TEXT NOT NULL,
    source_session_id TEXT NOT NULL,
    source_round_id TEXT NOT NULL,
    source_round_key TEXT NOT NULL,
    first_message_id TEXT,
    final_message_id TEXT,
    message_ids_json TEXT NOT NULL DEFAULT '[]',
    source_digest TEXT NOT NULL,
    source_schema_version INTEGER NOT NULL,
    resolver_version INTEGER NOT NULL,

    extraction_host TEXT NOT NULL,
    extraction_provider TEXT NOT NULL DEFAULT '',
    extraction_model TEXT NOT NULL DEFAULT '',
    extraction_prompt_version INTEGER NOT NULL,
    extraction_schema_version INTEGER NOT NULL,
    extraction_purpose TEXT NOT NULL,

    title TEXT NOT NULL,
    target TEXT NOT NULL,
    statement TEXT NOT NULL,
    rationale TEXT NOT NULL DEFAULT '',
    result TEXT NOT NULL DEFAULT '',
    artifacts_json TEXT NOT NULL DEFAULT '[]',
    content_digest TEXT NOT NULL,

    supersedes_atom_id TEXT REFERENCES atoms(atom_id),
    created_at TEXT NOT NULL,

    CHECK (source_schema_version >= 1),
    CHECK (resolver_version >= 1),
    CHECK (extraction_prompt_version >= 1),
    CHECK (extraction_schema_version >= 1),
    CHECK (source_digest GLOB 'sha256:*'),
    CHECK (content_digest GLOB 'sha256:*'),
    CHECK (supersedes_atom_id IS NULL OR supersedes_atom_id <> atom_id)
);

CREATE UNIQUE INDEX ux_atoms_source_extraction
ON atoms (
    memory_space_id,
    source_round_key,
    extraction_prompt_version,
    extraction_schema_version
);

CREATE INDEX ix_atoms_source_session
ON atoms (memory_space_id, source_session_id, created_at);

CREATE TABLE knowledge_items (
    knowledge_id TEXT PRIMARY KEY,
    memory_space_id TEXT NOT NULL
        REFERENCES memory_spaces(memory_space_id) ON DELETE CASCADE,

    title TEXT NOT NULL,
    target TEXT NOT NULL,
    statement TEXT NOT NULL,
    rationale TEXT NOT NULL DEFAULT '',
    phase TEXT NOT NULL,
    version INTEGER NOT NULL,

    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    superseded_by_id TEXT REFERENCES knowledge_items(knowledge_id),
    deleted_at TEXT,

    CHECK (phase IN ('pattern', 'emergent', 'canonical')),
    CHECK (version >= 1),
    CHECK (superseded_by_id IS NULL OR superseded_by_id <> knowledge_id)
);

CREATE INDEX ix_knowledge_current_space
ON knowledge_items (memory_space_id, target, phase)
WHERE superseded_by_id IS NULL AND deleted_at IS NULL;

CREATE INDEX ix_knowledge_updated
ON knowledge_items (memory_space_id, updated_at DESC);

CREATE TABLE knowledge_evidence (
    knowledge_id TEXT NOT NULL
        REFERENCES knowledge_items(knowledge_id) ON DELETE CASCADE,
    atom_id TEXT NOT NULL
        REFERENCES atoms(atom_id) ON DELETE RESTRICT,
    relation TEXT NOT NULL,
    created_at TEXT NOT NULL,

    PRIMARY KEY (knowledge_id, atom_id, relation),
    CHECK (relation IN ('origin', 'supports', 'contradicts', 'refines'))
);

CREATE INDEX ix_evidence_atom
ON knowledge_evidence (atom_id, knowledge_id);

CREATE TABLE knowledge_revisions (
    revision_id TEXT PRIMARY KEY,
    knowledge_id TEXT NOT NULL
        REFERENCES knowledge_items(knowledge_id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    cause_atom_id TEXT REFERENCES atoms(atom_id),
    created_at TEXT NOT NULL,

    UNIQUE (knowledge_id, version),
    CHECK (version >= 1)
);

CREATE TABLE idempotency_results (
    idempotency_key TEXT PRIMARY KEY,
    request_hash TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT
);

CREATE TABLE outbox_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    aggregate_id TEXT NOT NULL,
    memory_space_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    available_at TEXT NOT NULL,
    claimed_at TEXT,
    claimed_by TEXT,
    processed_at TEXT,
    last_error TEXT
);

CREATE INDEX ix_outbox_ready
ON outbox_events (available_at, occurred_at)
WHERE processed_at IS NULL;

CREATE TABLE projection_deliveries (
    projection_name TEXT NOT NULL,
    event_id TEXT NOT NULL REFERENCES outbox_events(event_id) ON DELETE CASCADE,
    attempts INTEGER NOT NULL DEFAULT 0,
    available_at TEXT NOT NULL,
    claimed_at TEXT,
    claimed_by TEXT,
    processed_at TEXT,
    last_error TEXT,

    PRIMARY KEY (projection_name, event_id)
);

CREATE INDEX ix_projection_deliveries_ready
ON projection_deliveries (projection_name, available_at)
WHERE processed_at IS NULL;
