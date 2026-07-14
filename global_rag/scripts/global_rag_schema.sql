-- ============================================================
-- Global RAG System - PostgreSQL / pgvector Schema
-- Bare-bones but extensible schema for:
-- 1) document inventory
-- 2) extracted text / tables
-- 3) chunking + embeddings
-- 4) structured financial staging records
-- 5) deterministic exceptions
-- 6) findings / evidence / report sections
--
-- IMPORTANT:
-- - Raw PDFs / Excel files should remain on disk.
-- - PostgreSQL should store paths, metadata, extracted text, chunks,
--   embeddings, structured rows, exceptions, findings and report outputs.
-- - Default embedding dimension below is 1536 for openAI's embedding model: text-embedding-3-small.
--   If you use OpenAI text-embedding-3-large, change vector(1024) to vector(3072).
-- ============================================================

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_name TEXT,
    run_type TEXT NOT NULL DEFAULT 'ingestion',
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running',
    notes TEXT
);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    corpus_zone TEXT NOT NULL,
    corpus_pack TEXT,
    workstream TEXT,
    source_folder TEXT,
    relative_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_extension TEXT,
    file_size_bytes BIGINT,
    file_checksum_sha256 TEXT,
    document_type TEXT,
    document_title TEXT,
    source_authority TEXT,
    confidentiality_level TEXT DEFAULT 'confidential',
    is_client_confidential BOOLEAN DEFAULT TRUE,
    index_in_rag BOOLEAN DEFAULT TRUE,
    extraction_required BOOLEAN DEFAULT TRUE,
    ingest_status TEXT DEFAULT 'pending',
    extraction_status TEXT DEFAULT 'pending',
    extraction_quality TEXT,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_corpus_zone ON documents (corpus_zone);
CREATE INDEX IF NOT EXISTS idx_documents_corpus_pack ON documents (corpus_pack);
CREATE INDEX IF NOT EXISTS idx_documents_workstream ON documents (workstream);
CREATE INDEX IF NOT EXISTS idx_documents_checksum ON documents (file_checksum_sha256);
CREATE INDEX IF NOT EXISTS idx_documents_relative_path ON documents (relative_path);

CREATE TABLE IF NOT EXISTS extracted_text (
    extracted_text_id BIGSERIAL PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    page_no INTEGER,
    section_heading TEXT,
    extraction_method TEXT,
    extraction_quality TEXT,
    token_count_estimate INTEGER,
    text_content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_extracted_text_document ON extracted_text (document_id);
CREATE INDEX IF NOT EXISTS idx_extracted_text_page ON extracted_text (document_id, page_no);
CREATE INDEX IF NOT EXISTS idx_extracted_text_heading_trgm ON extracted_text USING gin (section_heading gin_trgm_ops);

CREATE TABLE IF NOT EXISTS extracted_tables (
    table_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    table_name TEXT,
    sheet_name TEXT,
    page_no INTEGER,
    extracted_file_path TEXT,
    row_count INTEGER,
    column_count INTEGER,
    extraction_method TEXT,
    extraction_quality TEXT,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_extracted_tables_document ON extracted_tables (document_id);
CREATE INDEX IF NOT EXISTS idx_extracted_tables_sheet ON extracted_tables (sheet_name);

CREATE TABLE IF NOT EXISTS extracted_table_rows (
    row_id BIGSERIAL PRIMARY KEY,
    table_id TEXT NOT NULL REFERENCES extracted_tables(table_id) ON DELETE CASCADE,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    sheet_name TEXT,
    row_number INTEGER,
    row_data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_table_rows_table ON extracted_table_rows (table_id);
CREATE INDEX IF NOT EXISTS idx_table_rows_document ON extracted_table_rows (document_id);
CREATE INDEX IF NOT EXISTS idx_table_rows_data_gin ON extracted_table_rows USING gin (row_data);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    extracted_text_id BIGINT REFERENCES extracted_text(extracted_text_id) ON DELETE SET NULL,
    corpus_zone TEXT NOT NULL,
    corpus_pack TEXT,
    workstream TEXT,
    section_heading TEXT,
    page_start INTEGER,
    page_end INTEGER,
    chunk_index INTEGER,
    chunk_text TEXT NOT NULL,
    token_count_estimate INTEGER,
    source_reference TEXT,
    embedding_model TEXT,
    embedding vector(1536),
    search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', coalesce(chunk_text, ''))) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_corpus_pack ON chunks (corpus_pack);
CREATE INDEX IF NOT EXISTS idx_chunks_workstream ON chunks (workstream);
CREATE INDEX IF NOT EXISTS idx_chunks_search_vector ON chunks USING gin (search_vector);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
ON chunks USING hnsw (embedding vector_cosine_ops);

-- Alternative for older pgvector versions, if HNSW fails:
-- CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivfflat
-- ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE TABLE IF NOT EXISTS financial_records (
    record_id BIGSERIAL PRIMARY KEY,
    document_id TEXT REFERENCES documents(document_id) ON DELETE SET NULL,
    table_id TEXT REFERENCES extracted_tables(table_id) ON DELETE SET NULL,
    source_folder TEXT,
    source_file TEXT,
    source_row_number INTEGER,

    record_type TEXT NOT NULL,
    record_date DATE,
    period TEXT,
    transaction_id TEXT,
    journal_id TEXT,
    invoice_no TEXT,
    po_no TEXT,
    payment_ref TEXT,
    account_code TEXT,
    account_name TEXT,
    entity_name TEXT,
    counterparty_name TEXT,
    counterparty_type TEXT,
    description TEXT,

    debit NUMERIC(20,2),
    credit NUMERIC(20,2),
    amount NUMERIC(20,2),
    currency TEXT DEFAULT 'BHD',

    prepared_by TEXT,
    approved_by TEXT,
    approval_status TEXT,
    support_status TEXT,
    risk_note TEXT,

    raw_row_data JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_fin_records_type ON financial_records (record_type);
CREATE INDEX IF NOT EXISTS idx_fin_records_date ON financial_records (record_date);
CREATE INDEX IF NOT EXISTS idx_fin_records_counterparty ON financial_records USING gin (counterparty_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_fin_records_amount ON financial_records (amount);
CREATE INDEX IF NOT EXISTS idx_fin_records_doc ON financial_records (document_id);
CREATE INDEX IF NOT EXISTS idx_fin_records_raw_json ON financial_records USING gin (raw_row_data);

CREATE TABLE IF NOT EXISTS sanctions_entities (
    sanctions_entity_id BIGSERIAL PRIMARY KEY,
    source_list TEXT NOT NULL,
    entity_type TEXT,
    primary_name TEXT NOT NULL,
    aliases TEXT[],
    countries TEXT[],
    identifiers JSONB,
    addresses JSONB,
    programs TEXT[],
    first_seen DATE,
    last_seen DATE,
    last_changed DATE,
    raw_record JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sanctions_primary_name_trgm ON sanctions_entities USING gin (primary_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_sanctions_aliases ON sanctions_entities USING gin (aliases);
CREATE INDEX IF NOT EXISTS idx_sanctions_raw ON sanctions_entities USING gin (raw_record);

CREATE TABLE IF NOT EXISTS screening_results (
    screening_id BIGSERIAL PRIMARY KEY,
    screened_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    screened_name TEXT NOT NULL,
    screened_type TEXT,
    source_record_id BIGINT REFERENCES financial_records(record_id) ON DELETE SET NULL,
    matched_entity_id BIGINT REFERENCES sanctions_entities(sanctions_entity_id) ON DELETE SET NULL,
    source_list TEXT,
    match_score NUMERIC(6,4),
    match_status TEXT DEFAULT 'pending_review',
    review_notes TEXT,
    reviewed_by TEXT,
    reviewed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_screening_name_trgm ON screening_results USING gin (screened_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_screening_status ON screening_results (match_status);

CREATE TABLE IF NOT EXISTS exceptions (
    exception_id TEXT PRIMARY KEY,
    run_id UUID REFERENCES ingest_runs(run_id) ON DELETE SET NULL,
    test_name TEXT NOT NULL,
    workstream TEXT NOT NULL,
    exception_category TEXT,
    exception_date DATE,
    counterparty_name TEXT,
    amount NUMERIC(20,2),
    currency TEXT DEFAULT 'BHD',
    severity_hint TEXT,
    confidence_hint TEXT,
    exception_description TEXT NOT NULL,
    source_reference TEXT,
    source_record_ids BIGINT[],
    supporting_document_ids TEXT[],
    raw_exception_data JSONB,
    status TEXT DEFAULT 'open',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_exceptions_workstream ON exceptions (workstream);
CREATE INDEX IF NOT EXISTS idx_exceptions_category ON exceptions (exception_category);
CREATE INDEX IF NOT EXISTS idx_exceptions_counterparty ON exceptions USING gin (counterparty_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_exceptions_severity ON exceptions (severity_hint);

CREATE TABLE IF NOT EXISTS evidence_register (
    evidence_id TEXT PRIMARY KEY,
    exception_id TEXT REFERENCES exceptions(exception_id) ON DELETE SET NULL,
    finding_id TEXT,
    evidence_type TEXT NOT NULL,
    document_id TEXT REFERENCES documents(document_id) ON DELETE SET NULL,
    chunk_id TEXT REFERENCES chunks(chunk_id) ON DELETE SET NULL,
    record_id BIGINT REFERENCES financial_records(record_id) ON DELETE SET NULL,
    table_id TEXT REFERENCES extracted_tables(table_id) ON DELETE SET NULL,
    source_file TEXT,
    source_page INTEGER,
    source_sheet TEXT,
    source_row_number INTEGER,
    source_reference TEXT,
    evidence_summary TEXT NOT NULL,
    evidence_value TEXT,
    amount NUMERIC(20,2),
    currency TEXT DEFAULT 'BHD',
    confidence TEXT DEFAULT 'medium',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_evidence_exception ON evidence_register (exception_id);
CREATE INDEX IF NOT EXISTS idx_evidence_finding ON evidence_register (finding_id);
CREATE INDEX IF NOT EXISTS idx_evidence_document ON evidence_register (document_id);
CREATE INDEX IF NOT EXISTS idx_evidence_record ON evidence_register (record_id);

CREATE TABLE IF NOT EXISTS findings_register (
    finding_id TEXT PRIMARY KEY,
    run_id UUID REFERENCES ingest_runs(run_id) ON DELETE SET NULL,
    workstream TEXT NOT NULL,
    finding_category TEXT,
    rag_rating TEXT NOT NULL,
    confidence TEXT DEFAULT 'medium',
    observation TEXT NOT NULL,
    evidence_summary TEXT,
    source_reference TEXT,
    financial_exposure NUMERIC(20,2),
    currency TEXT DEFAULT 'BHD',
    entities_involved TEXT[],
    risk_implication TEXT,
    recommended_next_step TEXT,
    limitation TEXT,
    exception_ids TEXT[],
    status TEXT DEFAULT 'draft',
    human_review_notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_findings_workstream ON findings_register (workstream);
CREATE INDEX IF NOT EXISTS idx_findings_rating ON findings_register (rag_rating);
CREATE INDEX IF NOT EXISTS idx_findings_category ON findings_register (finding_category);
CREATE INDEX IF NOT EXISTS idx_findings_entities ON findings_register USING gin (entities_involved);

CREATE TABLE IF NOT EXISTS llm_call_log (
    llm_call_id BIGSERIAL PRIMARY KEY,
    run_id UUID REFERENCES ingest_runs(run_id) ON DELETE SET NULL,
    task_name TEXT NOT NULL,
    model_name TEXT,
    prompt_hash_sha256 TEXT,
    prompt_text TEXT,
    response_text TEXT,
    input_token_estimate INTEGER,
    output_token_estimate INTEGER,
    status TEXT DEFAULT 'completed',
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_llm_call_task ON llm_call_log (task_name);
CREATE INDEX IF NOT EXISTS idx_llm_call_run ON llm_call_log (run_id);

CREATE TABLE IF NOT EXISTS report_sections (
    report_section_id BIGSERIAL PRIMARY KEY,
    run_id UUID REFERENCES ingest_runs(run_id) ON DELETE SET NULL,
    report_name TEXT NOT NULL,
    section_order INTEGER NOT NULL,
    section_key TEXT NOT NULL,
    section_title TEXT NOT NULL,
    section_markdown TEXT NOT NULL,
    source_finding_ids TEXT[],
    status TEXT DEFAULT 'draft',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_report_sections_report ON report_sections (report_name);
CREATE INDEX IF NOT EXISTS idx_report_sections_order ON report_sections (report_name, section_order);

CREATE OR REPLACE VIEW v_open_exceptions AS
SELECT
    exception_id,
    test_name,
    workstream,
    exception_category,
    exception_date,
    counterparty_name,
    amount,
    currency,
    severity_hint,
    confidence_hint,
    exception_description,
    source_reference,
    status,
    created_at
FROM exceptions
WHERE status = 'open';

CREATE OR REPLACE VIEW v_findings_summary AS
SELECT
    finding_id,
    workstream,
    finding_category,
    rag_rating,
    confidence,
    financial_exposure,
    currency,
    observation,
    recommended_next_step,
    status,
    created_at
FROM findings_register
ORDER BY
    CASE rag_rating
        WHEN 'Red' THEN 1
        WHEN 'Amber' THEN 2
        WHEN 'Grey' THEN 3
        WHEN 'Green' THEN 4
        ELSE 5
    END,
    financial_exposure DESC NULLS LAST;

CREATE OR REPLACE VIEW v_document_ingestion_status AS
SELECT
    corpus_zone,
    corpus_pack,
    ingest_status,
    extraction_status,
    COUNT(*) AS document_count,
    SUM(file_size_bytes) AS total_size_bytes
FROM documents
GROUP BY corpus_zone, corpus_pack, ingest_status, extraction_status
ORDER BY corpus_zone, corpus_pack, ingest_status, extraction_status;

CREATE TABLE IF NOT EXISTS build_document_inventory(
s_no BIGSERIAL PRIMARY KEY,
document_id TEXT,
source_group TEXT,
corpus_pack TEXT,
file_name TEXT,
file_extension TEXT,
relative_path TEXT,
absolute_path TEXT,
file_size_bytes BIGINT,
last_modified_datetime TEXT,
sha256_checksum TEXT,
supported_file_type TEXT,
extraction_method_hint TEXT,
index_in_rag TEXT,
ingest_status TEXT,
notes TEXT,
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMIT;

-- Sanity checks after running:
SELECT
*
FROM
information_schema.tables
WHERE
table_schema = 'public'
ORDER BY
table_name;
SELECT * FROM public.v_document_ingestion_status;
SELECT * FROM documents;
SELECT * FROM chunks;
SELECT * FROM financial_records;
SELECT
attname AS column_name,
format_type(atttypid, atttypmod) AS data_type
FROM pg_attribute
WHERE attrelid = 'public.chunks'::regclass
AND attname = 'embedding';
SELECT * FROM build_document_inventory;