FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LEDGERMIND_HOME=/data/ledgermind

WORKDIR /app

COPY pyproject.toml .
COPY README.md .
COPY src ./src

RUN pip install .

EXPOSE 8765

CMD ["python", "-m", "cli", "--home", "/data/ledgermind", "serve"]
