# ledgermind-local

`ledgermind-local` — локальная служба LedgerMind 4.0. Она принимает
структурные `RawRound v2`, хранит локальное состояние, выполняет выбранную
пользователем обработку и общается с закрытым Rust Core через IPC.

## Trust boundaries and data ownership

- Local владеет `rounds.db`: raw-round metadata/payload, нормализованные раунды,
  processing jobs, leases, inference attempts, hypothesis records, command
  outbox и локальные projection stores остаются в локальной базе.
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
создать валидный RawRound v2 и сохранить/доставить его без включённого
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
- готовые processing workers, projection workers и наблюдаемую retry policy.

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
могут находиться raw conversation/tool payloads, нормализованные transcript,
inference metadata, egress audit и локальные projection данные. Передавайте
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

Версии v3 и v4 несовместимы. Автоматического импорта или миграции v3 нет;
новая служба принимает RawRound v2 через Integrations.
