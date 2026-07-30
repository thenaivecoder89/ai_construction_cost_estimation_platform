-- AI Construction Cost Estimation Platform
-- Takeoff layer: audited measurements ready for BOQ v2 consumption.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS takeoff_layer (
    takeoff_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    takeoff_run_id TEXT NOT NULL,
    run_status TEXT NOT NULL DEFAULT 'processing',
    project_id TEXT NOT NULL,

    division_code TEXT NOT NULL,
    section_code TEXT,
    item_code TEXT,
    item_description TEXT NOT NULL,
    takeoff_category TEXT NOT NULL,

    quantity NUMERIC(24,8),
    uom TEXT NOT NULL,
    applicable_area TEXT,
    element_type TEXT,
    element_mark TEXT,
    level_name TEXT,
    zone_name TEXT,

    calculation_method TEXT NOT NULL,
    calculation_formula TEXT,
    calculation_inputs JSONB NOT NULL DEFAULT '{}'::jsonb,
    calculation_logic TEXT NOT NULL,
    theoretical_basis TEXT NOT NULL,

    drawing_scale_text TEXT,
    drawing_scale_ratio NUMERIC(18,6),
    geometry_source TEXT,
    geometry_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    symbol_name TEXT,
    symbol_count INTEGER,

    source_document_id TEXT REFERENCES documents(document_id) ON DELETE SET NULL,
    source_file_name TEXT,
    source_relative_path TEXT,
    source_page INTEGER,
    source_sheet TEXT,
    source_table_id TEXT REFERENCES extracted_tables(table_id) ON DELETE SET NULL,
    source_row_numbers INTEGER[],
    source_reference TEXT NOT NULL,
    evidence_text TEXT,

    extraction_method TEXT NOT NULL,
    parser_name TEXT NOT NULL,
    model_name TEXT,
    confidence_score NUMERIC(6,5) NOT NULL,
    confidence_threshold NUMERIC(6,5) NOT NULL,
    confidence_basis TEXT NOT NULL,

    validation_status TEXT NOT NULL DEFAULT 'pending',
    is_boq_ready BOOLEAN NOT NULL DEFAULT FALSE,
    rejection_reason TEXT,
    raw_extraction JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT chk_takeoff_quantity_nonnegative
        CHECK (quantity IS NULL OR quantity >= 0),
    CONSTRAINT chk_takeoff_confidence
        CHECK (confidence_score >= 0 AND confidence_score <= 1),
    CONSTRAINT chk_takeoff_threshold
        CHECK (confidence_threshold >= 0 AND confidence_threshold <= 1),
    CONSTRAINT chk_takeoff_validation_status
        CHECK (
            validation_status IN (
                'accepted',
                'below_confidence_threshold',
                'rejected_validation',
                'pending',
                'superseded'
            )
        ),
    CONSTRAINT chk_takeoff_run_status
        CHECK (run_status IN ('processing', 'completed', 'failed', 'superseded')),
    CONSTRAINT chk_takeoff_boq_ready
        CHECK (
            is_boq_ready = FALSE
            OR (
                validation_status = 'accepted'
                AND quantity IS NOT NULL
                AND quantity > 0
                AND confidence_score >= confidence_threshold
            )
        )
);

CREATE INDEX IF NOT EXISTS idx_takeoff_layer_project_ready
    ON takeoff_layer (project_id, is_boq_ready, division_code);

CREATE INDEX IF NOT EXISTS idx_takeoff_layer_run
    ON takeoff_layer (takeoff_run_id, run_status);

CREATE INDEX IF NOT EXISTS idx_takeoff_layer_source_document
    ON takeoff_layer (source_document_id, source_page);

CREATE INDEX IF NOT EXISTS idx_takeoff_layer_item
    ON takeoff_layer (project_id, division_code, section_code, item_code);

CREATE INDEX IF NOT EXISTS idx_takeoff_layer_raw_extraction
    ON takeoff_layer USING gin (raw_extraction);

CREATE INDEX IF NOT EXISTS idx_takeoff_layer_calculation_inputs
    ON takeoff_layer USING gin (calculation_inputs);

COMMIT;
