# ledgermind-local

Локальная служба LedgerMind 4.0: SQLite как источник истины для Local-owned
rounds, RawRound v2 HTTP API, processing worker и Core-backed проекции.

## Локальная разработка
```bash
pip install -e .[dev]
pytest -q
```

Local не импортирует Python Core. Для запуска process boundary соберите Rust
binary из соседнего приватного `ledgermind-core` workspace и положите подписанный комплект в
`$LEDGERMIND_HOME/../core/bin/`:

```text
ledgermind-core
ledgermind-core.sig   # raw Ed25519 signature over the exact binary
ledgermind-core.pub   # raw 32-byte Ed25519 public key
```

Для release signing используйте `scripts/sign_core_binary.py` с private key,
который не хранится в repository:

```bash
python ledgermind-local/scripts/sign_core_binary.py \
  --binary ledgermind-core/target/release/ledgermind-core \
  --private-key "$LEDGERMIND_CORE_SIGNING_KEY" \
  --signature release/ledgermind-core.sig \
  --public-key release/ledgermind-core.pub
```

## Docker
Сборка выполняется из workspace-каталога `Проекты/ledgermind`, потому что Core,
Local и protocol находятся в соседних каталогах. Rust Core собирается в
отдельном multi-stage builder; Python Core в образ не попадает:

```bash
cd "$HOME/Проекты/ledgermind"
docker build -f ledgermind-local/Dockerfile -t ledgermind-local .
```

Образ намеренно не генерирует подпись. Перед запуском нужно примонтировать
signature/public-key для binary, собранного этим же Docker build:

```bash
docker run --rm \
  -v ledgermind-data:/data \
  -v "$PWD/release/ledgermind-core.sig:/run/secrets/ledgermind-core.sig:ro" \
  -v "$PWD/release/ledgermind-core.pub:/run/secrets/ledgermind-core.pub:ro" \
  -p 127.0.0.1:8765:8765 \
  ledgermind-local
```

Entrypoint завершает запуск с ошибкой, если оба файла подписи не переданы;
`verify_core_signature` остаётся включённым.

## Совместимость
Версии v3 и v4 несовместимы. Автоматического импорта или миграции v3 нет; новая служба принимает RawRound v2 через интеграции.
