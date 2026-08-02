# ledgermind-local

Scaffold for the local LedgerMind service (v4 architecture).

Current stage: `4.4` — expose isolated atom and knowledge reads.

## Быстрый старт

```bash
python -m pip install -e .[dev]
ledgermind --home ~/.ledgermind/local init
ledgermind --home ~/.ledgermind/local status
```

```python
from bootstrap import bootstrap_local_service

paths, config = bootstrap_local_service(home="~/.ledgermind/local")
print(paths.home)
print(config.config_version)
```
