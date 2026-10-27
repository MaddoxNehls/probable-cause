-- Core relational schema for the NTSB avall.mdb snapshot (2008-present).
-- Coded NTSB fields are stored decoded; the raw code is kept only where it is a
-- natural filter key (FAR part, occurrence/finding codes).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE source_snapshots (
    id              bigserial PRIMARY KEY,
    file_name       text        NOT NULL,
    listed_at       timestamp,              -- avdata listing time, no zone given
    size_bytes      bigint      NOT NULL,
    sha256          text        NOT NULL,
    loaded_at       timestamptz NOT NULL DEFAULT now(),
    row_counts      jsonb       NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE events (
    ev_id               text PRIMARY KEY,
    ntsb_no             text,
    event_type          text NOT NULL,          -- Accident | Incident
    ev_date             date NOT NULL,
    ev_time_local       time,
    timezone            text,
    city                text,
    state               text,
    country             text,
    latitude            double precision,
    longitude           double precision,
    airport_id          text,
    airport_name        text,
    highest_injury      text,                   -- Fatal | Serious | Minor | None | Unknown
    fatalities          integer NOT NULL DEFAULT 0,
    serious_injuries    integer NOT NULL DEFAULT 0,
    minor_injuries      integer NOT NULL DEFAULT 0,
    uninjured           integer NOT NULL DEFAULT 0,
    light_condition     text,
    weather_condition   text,                   -- VMC | IMC | Unknown
    carol_mkey          integer,                -- CAROL report key, derivable for post-2020 ids
    docket_url          text NOT NULL,
    report_url          text,
    last_changed        timestamp
);
CREATE INDEX events_date_idx ON events (ev_date);
CREATE INDEX events_ntsb_no_idx ON events (ntsb_no);

CREATE TABLE aircraft (
    ev_id               text NOT NULL REFERENCES events ON DELETE CASCADE,
    aircraft_key        integer NOT NULL,
    registration        text,
    make                text,
    model               text,
    series              text,
    category            text,
    far_part            text,
    far_part_desc       text,
    operation_type      text,
    damage              text,
    amateur_built       boolean,
    num_engines         smallint,
    PRIMARY KEY (ev_id, aircraft_key)
);
CREATE INDEX aircraft_far_part_idx ON aircraft (far_part);

CREATE TABLE injuries (
    ev_id               text NOT NULL,
    aircraft_key        integer NOT NULL,
    person_category     text NOT NULL,
    injury_level        text NOT NULL,
    person_count        integer NOT NULL,
    PRIMARY KEY (ev_id, aircraft_key, person_category, injury_level),
    FOREIGN KEY (ev_id, aircraft_key) REFERENCES aircraft ON DELETE CASCADE
);

CREATE TABLE occurrences (
    ev_id               text NOT NULL,
    aircraft_key        integer NOT NULL,
    occurrence_no       integer NOT NULL,
    occurrence_code     text,
    phase_code          text,
    phase               text,
    event_code          text,
    event               text,
    description         text,
    is_defining         boolean NOT NULL,
    PRIMARY KEY (ev_id, aircraft_key, occurrence_no),
    FOREIGN KEY (ev_id, aircraft_key) REFERENCES aircraft ON DELETE CASCADE
);
CREATE INDEX occurrences_phase_idx ON occurrences (phase);

CREATE TABLE findings (
    ev_id               text NOT NULL,
    aircraft_key        integer NOT NULL,
    finding_no          integer NOT NULL,
    finding_code        text,
    description         text,
    cause_factor        text,                   -- Cause | Factor | NULL (not coded)
    in_probable_cause   boolean NOT NULL,
    PRIMARY KEY (ev_id, aircraft_key, finding_no),
    FOREIGN KEY (ev_id, aircraft_key) REFERENCES aircraft ON DELETE CASCADE
);

CREATE TABLE narratives (
    ev_id               text NOT NULL REFERENCES events ON DELETE CASCADE,
    aircraft_key        integer NOT NULL,
    factual             text,
    analysis            text,
    probable_cause      text,
    PRIMARY KEY (ev_id, aircraft_key)
);

-- Chunkable text with provenance. One row per (event, aircraft, section, source).
CREATE TABLE documents (
    id                  bigserial PRIMARY KEY,
    ev_id               text NOT NULL REFERENCES events ON DELETE CASCADE,
    aircraft_key        integer,
    section             text NOT NULL CHECK (section IN
                            ('factual', 'analysis', 'probable_cause', 'report_pdf', 'docket_pdf')),
    source              text NOT NULL CHECK (source IN ('avall', 'carol_report', 'docket')),
    source_url          text NOT NULL,
    title               text,
    body                text NOT NULL,
    extraction_method   text NOT NULL CHECK (extraction_method IN ('database', 'pdf_text', 'ocr', 'mixed')),
    metadata            jsonb NOT NULL DEFAULT '{}'::jsonb,
    tsv                 tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED,
    UNIQUE NULLS NOT DISTINCT (ev_id, aircraft_key, section, source_url)
);
CREATE INDEX documents_tsv_idx ON documents USING gin (tsv);
CREATE INDEX documents_ev_idx ON documents (ev_id);

-- Filled in milestone 2. The vector dimension is deliberately unset until the
-- embedding model is chosen; an HNSW index is added then with a typed cast.
CREATE TABLE chunks (
    id                  bigserial PRIMARY KEY,
    document_id         bigint NOT NULL REFERENCES documents ON DELETE CASCADE,
    strategy            text NOT NULL,
    ordinal             integer NOT NULL,
    body                text NOT NULL,
    char_start          integer NOT NULL,
    char_end            integer NOT NULL,
    embedding           vector,
    UNIQUE (document_id, strategy, ordinal)
);
