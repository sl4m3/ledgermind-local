CREATE TABLE idempotency_results_v2 (
    memory_space_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    PRIMARY KEY (memory_space_id, idempotency_key)
);

INSERT INTO idempotency_results_v2 (
    memory_space_id,
    idempotency_key,
    request_hash,
    response_json,
    created_at,
    expires_at
)
SELECT
    '',
    idempotency_key,
    request_hash,
    response_json,
    created_at,
    expires_at
FROM idempotency_results;

DROP TABLE idempotency_results;
ALTER TABLE idempotency_results_v2 RENAME TO idempotency_results;
