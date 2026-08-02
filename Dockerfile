FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LEDGERMIND_HOME=/data/ledgermind

WORKDIR /app

COPY ledgermind-core/pyproject.toml /opt/ledgermind-core/pyproject.toml
COPY ledgermind-core/README.md /opt/ledgermind-core/README.md
COPY ledgermind-core/src /opt/ledgermind-core/src
RUN pip install /opt/ledgermind-core

COPY ledgermind-local/pyproject.toml /app/pyproject.toml
COPY ledgermind-local/README.md /app/README.md
COPY ledgermind-local/src /app/src
RUN pip install /app

EXPOSE 8765

CMD ["python", "-m", "ledgermind_local.cli", "--home", "/data/ledgermind", "serve"]
