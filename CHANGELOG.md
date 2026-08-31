# Changelog

## 4.0.0b1 - 2026-08-30

- Made repeated installation explicit and non-destructive: users choose to add
  agents, repair the current installation, or reconfigure provider profiles.
  Adding an agent cannot rewrite inference or memory configuration.
- Made signed updates reuse the installed provider identities, credentials,
  memory topology, storage, and agent selection instead of reopening setup.
- Made ordinary uninstall stop the runtime and create a private memory backup
  while preserving knowledge and configuration; destructive purge still
  requires explicit purge flags and confirmation.
- Added actionable terminal summaries for install, provider reconfiguration,
  repair, removal, and integration lifecycle operations.
- Added explicit OpenRouter provider-route selection after model selection:
  one required primary provider and at most one optional fallback, with no
  implicit provider default. Routes remain part of every generation profile
  fingerprint and provider probe.
- Kept agent integration lifecycle operations isolated from inference
  configuration so connecting or toggling an agent cannot rewrite provider
  routes or invalidate verified capabilities.
- Persisted successful installer strict-JSON probes into Local's capability
  cache so the first semantic task uses the exact provider/model/route that
  passed preflight instead of failing before the provider call.
- Made validated reconfiguration rematerialize Local profiles and
  refresh their capability rows instead of updating only the installer JSON.
- Replaced the plain prompt chain with a guided terminal setup, OpenRouter
  reference defaults, consolidated agent selection, endpoint normalization,
  fail-fast provider checks, and actionable embedding probe diagnostics.
- Prevented false generation-probe failures when a reasoning model consumed
  the previous 32-token capability-check budget before emitting strict JSON.
- Accepted OpenRouter's canonical embedding model response when a request uses
  the `:free` routing suffix or its response uses the known
  `private/openrouter/` namespace, while retaining strict mismatch detection.
- Created and preserved the Local API bearer token during installation, added
  authenticated runtime ownership checks, and rejected stale process state.
- Bundled the pinned CPython 3.12 runtime with Local instead of relying on an
  ABI-compatible system `python3`.
- Made secure Bubblewrap isolation enter an unprivileged user namespace before
  creating its network namespace, including on hosts that forbid direct
  unprivileged network namespace creation.
- Prevented API embedding profiles from spawning a local embedding service and
  preserved all managed process records across nested runtime supervisors.
- Documented Local/Core and Local/provider trust boundaries, split capture/full
  readiness, and the sensitivity of backup archives.
- Added release-content assertions for the Local wheel's runtime data and
  generated-artifact/secrets exclusions.
