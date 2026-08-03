# ledgermind-local

Локальная служба LedgerMind 4.0: SQLite как источник истины, RawRound v2 HTTP API, processing worker и проекции.

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

## Совместимость
Версии v3 и v4 несовместимы. Автоматического импорта или миграции v3 нет; новая служба принимает RawRound v2 через интеграции.
