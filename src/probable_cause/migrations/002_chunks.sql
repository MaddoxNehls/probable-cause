-- Milestone 2: chunks for the three strategies, with fixed-dimension embeddings.
-- The milestone-1 placeholder table was never populated, so it is replaced outright.

DROP TABLE chunks;

CREATE TABLE chunks (
    id                  bigserial PRIMARY KEY,
    strategy            text NOT NULL CHECK (strategy IN ('fixed', 'section', 'parent_child')),
    ev_id               text NOT NULL REFERENCES events ON DELETE CASCADE,
    document_id         bigint NOT NULL REFERENCES documents ON DELETE CASCADE,
    section             text NOT NULL,          -- documents.section; the citation key part
    heading             text,                   -- sub-heading inside the section, if any
    ordinal             integer NOT NULL,
    body                text NOT NULL,          -- the text that is embedded and indexed
    char_start          integer NOT NULL,       -- offsets into the cleaned document text
    char_end            integer NOT NULL,
    -- parent_child: parents have is_parent = true and no embedding; children point at them.
    is_parent           boolean NOT NULL DEFAULT false,
    parent_id           bigint REFERENCES chunks ON DELETE CASCADE,
    embedding           vector(384)             -- snowflake-arctic-embed-xs
);
CREATE INDEX chunks_strategy_ev_idx ON chunks (strategy, ev_id);
CREATE INDEX chunks_parent_idx ON chunks (parent_id);
-- Rows still waiting for an embedding, so `pc index embed` can resume cheaply.
CREATE INDEX chunks_unembedded_idx ON chunks (strategy, id)
    WHERE embedding IS NULL AND NOT is_parent;

-- One HNSW index per strategy (partial), so a query filtered to one strategy can use it.
-- Built by `pc index build` after embedding: bulk-loading into an existing HNSW index is
-- several times slower than building it once.
