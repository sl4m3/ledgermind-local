# ADR 0004: Local никогда не открывает knowledge.db

- **Статус:** accepted
- **Дата:** 2026-08-03

Даже временный Python backend изолируется за `CoreGateway`; единственный файл, имеющий право импортировать переходный Python Core, — `core_gateway/python_backend.py`.
Обычный Local runtime не импортирует внутренние Core repository adapters и не открывает `knowledge.db`.

Публичный ContextView минимален и не раскрывает фазу, evidence count, внутренние scores, rationale или source atom IDs.
