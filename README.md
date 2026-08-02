# ledgermind-local

Локальная служба LedgerMind 4.0: SQLite как источник истины, HTTP API, Hermes plugin, проекции.

## Локальная разработка
```bash
pip install -e ../ledgermind-core
pip install -e .[dev]
pytest -q
```

## Docker
Сборка выполняется из корня workspace, чтобы образ получил оба пакета:
```bash
docker build -f ledgermind-local/Dockerfile -t ledgermind-local .
```

## Переход с v3
- Временная команда: `ledgermind-v4`
- Миграция: `ledgermind-v4 migrate-v3 --source <старый storage_path> --dry-run`
- Документация: `docs/migration-v3.md`
