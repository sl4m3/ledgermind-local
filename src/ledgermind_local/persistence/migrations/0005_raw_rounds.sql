CREATE TABLE raw_rounds (
    raw_round_id TEXT PRIMARY KEY,
    memory_space_id TEXT NOT NULL
        REFERENCES memory_spaces(memory_space_id) ON DELETE CASCADE,
    source_system TEXT NOT NULL,
    source_instance_id TEXT NOT NULL,
    source_profile_id TEXT NOT NULL,
    source_session_id TEXT NOT NULL,
    source_round_id TEXT NOT NULL,
    source_round_key TEXT NOT NULL,
    capture_schema_version INTEGER NOT NULL,
    adapter_version TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    received_at TEXT NOT NULL,
    retention_expires_at TEXT,
    CHECK (capture_schema_version >= 1),
    CHECK (payload_digest GLOB 'sha256:*'),
    UNIQUE (memory_space_id, source_round_key, capture_schema_version)
);

CREATE INDEX ix_raw_rounds_memory_received
ON raw_rounds (memory_space_id, received_at DESC);

CREATE TABLE round_processing_jobs (
    job_id TEXT PRIMARY KEY,
    raw_round_id TEXT NOT NULL
        REFERENCES raw_rounds(raw_round_id) ON DELETE CASCADE,
    pipeline_version INTEGER NOT NULL,
    normalizer_version INTEGER NOT NULL,
    prompt_version INTEGER NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    available_at TEXT NOT NULL,
    claimed_at TEXT,
    claimed_by TEXT,
    completed_at TEXT,
    last_error TEXT,
    CHECK (pipeline_version >= 1),
    CHECK (normalizer_version >= 1),
    CHECK (prompt_version >= 1),
    CHECK (status IN ('received', 'processing', 'retry_wait', 'completed', 'no_knowledge', 'failed')),
    UNIQUE (raw_round_id, pipeline_version, normalizer_version, prompt_version)
);

CREATE INDEX ix_round_processing_jobs_ready
ON round_processing_jobs (status, available_at)
WHERE status IN ('received', 'retry_wait');

CREATE TABLE hypothesis_attempts (
    attempt_id TEXT PRIMARY KEY,
    raw_round_id TEXT NOT NULL
        REFERENCES raw_rounds(raw_round_id) ON DELETE CASCADE,
    job_id TEXT NOT NULL
        REFERENCES round_processing_jobs(job_id) ON DELETE CASCADE,
    pipeline_version INTEGER NOT NULL,
    normalizer_version INTEGER NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version INTEGER NOT NULL,
    schema_version INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    response_digest TEXT,
    error_code TEXT,
    error_detail TEXT
);

CREATE TABLE hypotheses (
    hypothesis_id TEXT PRIMARY KEY,
    raw_round_id TEXT NOT NULL
        REFERENCES raw_rounds(raw_round_id) ON DELETE CASCADE,
    attempt_id TEXT NOT NULL
        REFERENCES hypothesis_attempts(attempt_id) ON DELETE RESTRICT,
    hypothesis_index INTEGER NOT NULL,
    title TEXT NOT NULL,
    target TEXT NOT NULL,
    statement TEXT NOT NULL,
    rationale TEXT NOT NULL DEFAULT '',
    result TEXT NOT NULL DEFAULT '',
    artifacts_json TEXT NOT NULL DEFAULT '[]',
    content_digest TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (attempt_id, hypothesis_index),
    CHECK (hypothesis_index >= 0),
    CHECK (content_digest GLOB 'sha256:*')
);

CREATE INDEX ix_hypotheses_raw_round
ON hypotheses (raw_round_id, created_at);

ALTER TABLE knowledge_evidence
ADD COLUMN hypothesis_id TEXT REFERENCES hypotheses(hypothesis_id) ON DELETE SET NULL;

CREATE INDEX ix_evidence_hypothesis
ON knowledge_evidence (hypothesis_id);
