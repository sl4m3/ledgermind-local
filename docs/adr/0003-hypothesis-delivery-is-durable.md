# ADR 0003: доставка Hypothesis в Core долговечна

- **Статус:** accepted
- **Дата:** 2026-08-03

Local сначала сохраняет нормализованный раунд, inference attempt, HypothesisRecord и идемпотентную команду `accept_hypothesis` в `rounds.db` одной транзакцией.
Только после commit отдельный `CoreCommandWorker` доставляет команду через `CoreGateway`.

Обработка раунда не вызывает Core синхронно. Команды имеют lease, retry policy, terminal states и scoped idempotency.
Падение Local после commit не теряет Hypothesis.
