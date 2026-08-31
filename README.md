# ledgermind-local

`ledgermind-local` — локальная служба LedgerMind. Она принимает
структурные `RawRound`, хранит транспортное состояние и общается с закрытым
Rust Core через IPC. Доменное object-facet состояние и его обработка принадлежат
Core; Local выполняет только технические generic execution tasks.

## Trust boundaries and data ownership

- Local владеет `rounds.db`: raw-round metadata/payload, durable Core command
  delivery, retention state, inference profile metadata, technical profile-slot
  bindings и egress audit.
- Local **никогда не открывает рабочую `knowledge.db`**. Knowledge доступен
  только через supervised `CoreGateway` к отдельному подписанному Rust process
  `ledgermind-core`; SQL Core и его миграции не входят в Local.
- Local не импортирует внутренний код Core и не является владельцем Core
  storage.

## External APIs and egress

Local вызывает только те внешние API, endpoint и модель которых выбрал и
настроил пользователь. Поэтому payload, который Local отправляет в provider
или в выбранный Cloud endpoint, может покинуть машину пользователя. Перед
включением egress нужно проверить endpoint, retention, регион и policy
провайдера. Секреты адресуются через Local secret store/`secret_ref`, не
вшиваются в пакет и не должны попадать в RawRound, обычные журналы или backup.

## Readiness is deliberately split

### Capture-ready

Capture-ready означает, что интеграция может наблюдать завершённый раунд,
создать валидный RawRound и сохранить/доставить его без включённого
модельного провайдера. В этом режиме важны `rounds.db`, durable spool и
повторяемая доставка; отсутствие Core или provider не должно уничтожать
захваченные данные.

### Full-ready

Full-ready дополнительно требует:

- доступный и проверенный подписанный `ledgermind-core` binary с signature и
  public key;
- рабочий Core IPC и локальную `knowledge.db` в каталоге Core;
- настроенный пользователем inference endpoint/profile и соответствующий
  `secret_ref`;
- готовый generic execution worker и технические profile slots.

Capture-ready не следует объявлять full-ready: захват и доставка могут быть
здоровы, пока inference/Core временно недоступны.

## Signed Core комплект

Local не импортирует Python implementation Core. Для process boundary соберите
Rust binary из соседнего приватного `ledgermind-core` workspace и положите
подписанный комплект в `$LEDGERMIND_HOME/../core/bin/`:

```text
ledgermind-core
ledgermind-core.sig   # raw Ed25519 signature over the exact binary
ledgermind-core.pub   # raw 32-byte Ed25519 public key
```

Для release signing используйте `scripts/sign_core_binary.py` с private key,
который не хранится в repository:

```bash
python scripts/sign_core_binary.py \
  --binary /path/to/ledgermind-core \
  --private-key "$LEDGERMIND_CORE_SIGNING_KEY" \
  --signature release/ledgermind-core.sig \
  --public-key release/ledgermind-core.pub
```

## Backup archive sensitivity

Backup archive следует считать чувствительным секретным материалом. В нём
могут находиться raw conversation/tool payloads, inference metadata, egress
audit и opaque Core snapshot artifacts. Передавайте
архив только по доверенному каналу, ограничивайте права файла, шифруйте его
при переносе и хранении, а перед восстановлением проверяйте источник и
целостность. Не коммитьте archive в repository и не включайте в публичный
release artifact.

## Local development

```bash
pip install -e .[dev]
pytest -q
```

## Docker

Сборка выполняется из workspace-каталога `Проекты/ledgermind`, потому что Core,
Local и protocol находятся в соседних каталогах. Rust Core собирается в
отдельном multi-stage builder; старый Python runtime в образ не попадает:

```bash
cd "$HOME/Проекты/ledgermind"
docker build -f ledgermind-local/Dockerfile -t ledgermind-local .
```

Образ намеренно не генерирует подпись. Перед запуском нужно примонтировать
signature/public-key для binary, собранного этим же Docker build. Entrypoint
завершает запуск с ошибкой, если оба файла подписи не переданы;
`verify_core_signature` остаётся включённым.

## Compatibility

Legacy contract payloads are migrated once before Local workers start. Legacy
runtime routes, operation names and aliases are not supported; new capture
arrives through Integrations. Local executes Core-owned generic tasks and does
not own `HypothesisCandidate`, `MemoryObject`, `Target`, `Facet`, Merge, or
Replace semantics. The application Core coordinator owns those decisions;
Local only transports provider work and reports opaque results.

## Install on Linux

The current release supports rootless Linux installation. The installer uses
one configuration engine for the interactive wizard and agent-driven JSON
installs. It accepts only an
OpenAI-compatible generation API and either an OpenAI-compatible embeddings
API or a signed local CPU/GPU model catalog entry:

```bash
curl -fsSL https://github.com/sl4m3/ledgermind/releases/latest/download/install.sh | sh
ledgermind install schema --json
ledgermind integrations discover --json
ledgermind integrations status --json
ledgermind doctor --json
ledgermind runtime status --json
```

Without arguments, `install.sh` starts a navigable terminal wizard (with a
line-mode fallback for restricted terminals and SSH). It checks the Linux
host, lets the user choose English, Spanish, German, French, Russian, or any
custom BCP-47 memory language, detects agents, selects shared or per-agent
memory, and then configures generation and embeddings. One generation model is
used throughout the complete knowledge pipeline; there is no separate Object
Resolution model prompt. The token is entered privately and displayed only as
a masked fingerprint during review. Nothing is installed before confirmation.
Provider credentials and persisted configuration are private (`0600`).

Use a generation model with at least 120B parameters. Its provider endpoint
must implement strict JSON Schema structured outputs; plain JSON mode is not a
substitute. All full provider-backed benchmark and integration runs reported
for the current implementation used the tested reference model
`deepseek/deepseek-v4-flash-0731`. Alternative models are supported only when
they satisfy the same capability requirements. The installer verifies strict
structured-output support with a real provider probe. For OpenRouter it queries
the selected model's current endpoints, lists only providers advertising
strict structured outputs, and lets the user choose one primary route and at
most one fallback. The same restricted route chain and
`require_parameters=true` are used during the probe and every production
request; automatic provider routing is disabled. A capability verified on one
model/route cannot be reused after either changes.

API embedding dimensions are detected from an actual vector response rather
than entered by hand. Releases that include the signed local catalog also
offer `nvidia/Nemotron-3-Embed-1B-BF16` (2048 dimensions) on CPU, NVIDIA CUDA,
or AMD ROCm. Model files are pinned to an upstream revision and verified by
size and SHA-256; the matching device runtime is part of the signed release.

An agent can run the same installation non-interactively. It should first
collect the deployment choices from the user, write the current-schema JSON,
then execute one command and consume its machine-readable result:

```bash
curl -fsSL https://github.com/sl4m3/ledgermind/releases/latest/download/install.sh \
  | sh -s -- install --non-interactive --config /secure/install.json --json
```

Provider credentials should be supplied through `token_env`, `token_stdin`, or
an existing `secret_ref`; the persisted installer config never contains the
token value. One platform installation may connect several agent integrations.
Set `memory_mode` to `shared` to give those agents one knowledge space, or to
`per_agent` to keep their knowledge separate. The interactive wizard asks for
the same choice and defaults to `per_agent` for compatibility.
Their lifecycle is independent from the platform transaction:

```bash
ledgermind integrations discover
ledgermind integrations connect hermes
ledgermind integrations connect codex
ledgermind integrations connect claude-code
ledgermind integrations connect opencode
ledgermind integrations connect openclaw
ledgermind integrations status
ledgermind integrations disable hermes --json
ledgermind integrations enable hermes --json
ledgermind integrations disconnect hermes --json
```

Cursor integration is retained as experimental code but is not part of the
current Linux acceptance matrix; it will be validated separately. The other
listed agents have dedicated lifecycle adapters. Human-readable `discover`,
`status`, and `connect` output identifies the detected installation, hook
verification, memory space, and any activation step. Add `--json` for
automation.

The states are intentionally distinct: `installed` means the LedgerMind
platform exists, `connected` means an adapter is registered, `enabled` means it
will attach to future agent sessions, and `active` means the installed adapter
is enabled and has no remaining activation step. Codex CLI requires the user to
review and trust newly installed hooks with `/hooks`; status reports this step
instead of claiming that the integration is active. A failed agent connection
returns a partial result and never rolls back a successfully verified platform
installation.

Install data follows XDG directories. Signed manifests, bundle artifacts,
Core, model files, and embedding runtimes are verified before `current` is
switched. The default runtime is on-demand: an enabled Hermes integration
acquires a TTL lease before memory work, heartbeats while active, and releases
it on shutdown. Ordinary uninstall preserves user memory, configuration, and
secrets.

### Existing installations

Running the installer again never silently replaces provider settings. The
interactive installer requires one explicit action:

1. **Add an agent** connects one or more detected agents and leaves generation,
   embeddings, memory mode, storage, and existing integrations unchanged.
2. **Update LedgerMind** installs the current signed release while preserving
   provider configuration, agents, and memory.
3. **Repair** restores binaries, links, permissions, and already-selected
   integrations, then runs diagnostics. It does not select new models.
4. **Reconfigure providers** probes and replaces only generation and embedding
   profiles. Memory, agents, language, runtime settings, and storage remain
   unchanged.
5. **Exit** makes no changes.

The same actions are available without prompts:

```bash
ledgermind install --existing-mode add-agent --agent opencode --non-interactive
ledgermind install --existing-mode update --non-interactive
ledgermind install --existing-mode repair --non-interactive
ledgermind install --existing-mode reconfigure \
  --non-interactive --config /secure/providers.json
```

For OpenRouter, generation roles use one ordered route chain: one required
primary provider and, optionally, one fallback. Embeddings are configured by
endpoint and model only; they do not inherit the generation provider chain.

### Updates and removal

An update consumes a signed manifest and bundle but always reuses the installed
provider identities, secrets, memory configuration, and selected agents. Use
explicit provider reconfiguration when those values must change:

```bash
ledgermind update --manifest install-manifest.json --bundle release.tar.zst
ledgermind doctor
```

Default removal disconnects hooks and removes installed release binaries while
preserving configuration, secrets, models, and knowledge data. Before removal
it writes a private backup of the memory directory and prints its location.
Destructive removal must be requested explicitly:

```bash
ledgermind uninstall
ledgermind uninstall --purge-data --purge-config --yes
```
