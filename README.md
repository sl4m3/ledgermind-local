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

## Universal installer

The release installer is rootless and uses one configuration engine for the
interactive wizard and agent-driven JSON installs. It accepts only an
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

Without arguments, `install.sh` starts the human wizard. An agent uses the same
installer non-interactively: it asks the user for deployment choices, writes a
private (`0600`) current-schema config, then executes one command and consumes its
JSON result:

```bash
curl -fsSL https://github.com/sl4m3/ledgermind/releases/latest/download/install.sh \
  | sh -s -- install --non-interactive --config /secure/install.json --json
```

Provider credentials should be supplied through `token_env`, `token_stdin`, or
an existing `secret_ref`; the persisted installer config never contains the
token value. One platform installation may connect several agent integrations.
Their lifecycle is independent from the platform transaction:

```bash
ledgermind integrations connect hermes --json
ledgermind integrations disable hermes --json
ledgermind integrations enable hermes --json
ledgermind integrations disconnect hermes --json
```

The states are intentionally distinct: `installed` means the LedgerMind
platform exists, `connected` means an adapter is registered, `enabled` means it
will attach to future agent sessions, and `active` means a live session holds a
runtime lease. A failed agent connection returns a partial result and never
rolls back a successfully verified platform installation.

Install data follows XDG directories. Signed manifests, bundle artifacts,
Core, model files, and embedding runtimes are verified before `current` is
switched. The default runtime is on-demand: an enabled Hermes integration acquires a TTL lease before
memory work, heartbeats while active, and releases it on shutdown. Ordinary
uninstall preserves user memory, configuration, and secrets.
