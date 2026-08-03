# Baseline test results

## Repository

- Repository: `ledgermind-local`
- Branch: `refactor/local-core-boundary`
- Baseline commit: `76cf88a`
- Baseline tag: `pre-rust-core-boundary`
- Python: `3.11.15`

## Commands

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$HOME/Проекты/ledgermind/ledgermind/src:$PWD/src" \
/home/stanislav/.hermes/hermes-agent/venv/bin/python -m pytest -q --tb=short

/home/stanislav/.hermes/hermes-agent/venv/bin/python -m build
```

## Results

- Pytest: **230 passed**.
- Build: **passed**; sdist and wheel produced for `ledgermind-local==4.0.0a1`.
- Known warning: Starlette deprecation warning from the installed FastAPI test client integration (`httpx` compatibility); no test failure.

## Known skipped checks

- Rust checks are not applicable to the Python Local repository.
- `cargo deny` is not applicable until the Rust workspace is created in Stage 8.
- Ruff and mypy were not part of the Stage 0 baseline command set; they remain required for subsequent Python changes.
