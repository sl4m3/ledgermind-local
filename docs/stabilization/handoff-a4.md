# A4 handoff: runtime search composition

## A4 status

The A4 Local slice provides:

- FTS and vector candidate adapters scoped to `memory_space_id`;
- deterministic candidate deduplication and score merging in `CoreBackedSearch`;
- Core-owned final ranking through `RetrieveContextCommand`;
- Core fallback with the internal `degraded` status marker when Local candidate search is unavailable;
- a one-shot `RawRoundRetentionWorker.process_once` that purges payload bodies in a short write transaction while preserving provenance, normalized rounds, and hypotheses.

No bootstrap, health, configuration, raw-round repository, or scheduler-thread wiring was changed by A4.

## External owner: runtime composition (B2)

`src/ledgermind_local/api/context.py` now accepts the structural `ContextSearch` port. The existing `create_app` composition still passes `application.core_gateway`; wiring `CoreBackedSearch` into the production application is outside the A4-owned paths and should be completed by the B2/runtime owner.

Required follow-up:

1. Build `CoreProjectionSearchAdapter` against the Local FTS projection connection.
2. Optionally build `CoreProjectionVectorSearchAdapter` with the configured vector projection and vectorizer factory.
3. Compose `CoreBackedSearch(fts_search=..., vector_search=..., core_gateway=..., status_callback=...)` and inject it as the context route's `ContextSearch` implementation.
4. Own the SQLite connection, vectorizer, and adapter shutdown lifecycle in the runtime composition layer.
5. Keep `ContextViewResponse` unchanged; candidate IDs/scores and the degraded marker are internal boundary data.

The A4 search path deliberately does not start a background thread. The retention worker is also one-shot; B2 owns scheduling and shutdown integration. Retention uses the existing SQLite transaction boundary and does not claim processing-job leases. Existing processing lease/heartbeat/reclaim tests remain green.

## Contracts and migrations

- No database migrations were added.
- The existing `RetrieveContextCommand.candidate_ids` and `candidate_scores` fields carry Local evidence to Core.
- Retention updates only `raw_round_payloads.payload_json`, `payload_bytes`, and `deleted_at`; durable provenance and derived records remain intact.
