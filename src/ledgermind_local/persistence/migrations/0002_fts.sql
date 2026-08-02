CREATE VIRTUAL TABLE knowledge_fts USING fts5(
    knowledge_id UNINDEXED,
    memory_space_id UNINDEXED,
    title,
    target,
    statement,
    rationale,
    tokenize='unicode61'
);
