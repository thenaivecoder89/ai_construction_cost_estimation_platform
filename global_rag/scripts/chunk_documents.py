# File_name: chunk_documents.py
# Purpose:
# Chunk extracted text and extracted table rows into the existing chunks table.
#
# Input tables:
# 1. documents
# 2. extracted_text
# 3. extracted_tables
# 4. extracted_table_rows
#
# Output table:
# 1. chunks
#
# Notes:
# - Bare-bones POC version.
# - No classes.
# - No embeddings are generated here.
# - Existing chunks table is used.
# - search_vector is populated using PostgreSQL to_tsvector().
# - embedding remains NULL and will be populated later by embed_chunks.py.

import json
import pandas as pd
from sqlalchemy import create_engine, text

from global_rag.scripts import config


def clean_text(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    value = str(value).replace("\x00", " ")
    lines = [line.strip() for line in value.splitlines()]
    lines = [line for line in lines if line]

    return "\n".join(lines)


def estimate_tokens(value):
    value = clean_text(value)

    if value == "":
        return 0

    # Bare-bones approximation: 1 token ~= 4 characters
    return max(1, int(len(value) / 4))


def nullable_int(value):
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def split_text_into_chunks(text_content, chunk_size_tokens, chunk_overlap_tokens):
    text_content = clean_text(text_content)

    if text_content == "":
        return []

    max_chars = int(chunk_size_tokens) * 4
    overlap_chars = int(chunk_overlap_tokens) * 4

    if max_chars <= 0:
        max_chars = 4000

    if overlap_chars < 0:
        overlap_chars = 0

    if overlap_chars >= max_chars:
        overlap_chars = int(max_chars * 0.15)

    chunks = []
    start = 0
    text_length = len(text_content)

    while start < text_length:
        end = min(start + max_chars, text_length)
        candidate = text_content[start:end]

        # Try not to cut the chunk in the middle of a sentence or paragraph.
        if end < text_length:
            min_break_position = int(len(candidate) * 0.60)

            break_points = [
                candidate.rfind("\n\n", min_break_position),
                candidate.rfind("\n", min_break_position),
                candidate.rfind(". ", min_break_position),
                candidate.rfind(" ", min_break_position),
            ]

            best_break = max(break_points)

            if best_break > 0:
                end = start + best_break + 1
                candidate = text_content[start:end]

        candidate = clean_text(candidate)

        if candidate:
            chunks.append(candidate)

        if end >= text_length:
            break

        start = max(0, end - overlap_chars)

    return chunks


def row_data_to_text(row_data):
    if row_data is None:
        return ""

    if isinstance(row_data, dict):
        row_dict = row_data
    else:
        row_text = clean_text(row_data)

        if row_text == "":
            return ""

        try:
            row_dict = json.loads(row_text)
        except Exception:
            return row_text

    parts = []

    for key, value in row_dict.items():
        clean_key = clean_text(key).replace("\n", " ")
        clean_value = clean_text(value).replace("\n", " ")

        if clean_key and clean_value:
            parts.append(f"{clean_key}: {clean_value}")

    return " | ".join(parts)


def check_chunks_table(engine):
    required_columns = [
        "chunk_id",
        "document_id",
        "extracted_text_id",
        "corpus_zone",
        "corpus_pack",
        "workstream",
        "section_heading",
        "page_start",
        "page_end",
        "chunk_index",
        "chunk_text",
        "token_count_estimate",
        "source_reference",
        "embedding_model",
        "embedding",
        "search_vector",
        "created_at",
    ]

    sql = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'chunks';
    """

    existing_columns_df = pd.read_sql(text(sql), engine)
    existing_columns = set(existing_columns_df["column_name"].tolist())

    missing_columns = []

    for col in required_columns:
        if col not in existing_columns:
            missing_columns.append(col)

    if missing_columns:
        raise RuntimeError(
            "The chunks table is missing these required columns: "
            + ", ".join(missing_columns)
        )


def chunk_documents(rebuild_inventory: str = "Y"):
    config_base = config.config_base()
    rebuild_inventory = rebuild_inventory.strip().upper()

    if rebuild_inventory not in ["Y", "N"]:
        raise ValueError("rebuild_inventory must be 'Y' or 'N'.")

    engine = create_engine(
        url=config_base["db_url"],
        pool_pre_ping=True
    )

    chunk_size_tokens = int(config_base["chunk_size_tokens"])
    chunk_overlap_tokens = int(config_base["chunk_overlap_tokens"])
    embedding_model = config_base["embedding_model"]

    check_chunks_table(engine)

    new_documents_filter = ""
    if rebuild_inventory == "N":
        new_documents_filter = """
          AND NOT EXISTS (
              SELECT 1
              FROM chunks c
              WHERE c.document_id = d.document_id
          )
        """

    # ---------------------------------------------------------------------
    # 1. Read narrative extracted text.
    # ---------------------------------------------------------------------
    # Only documents marked index_in_rag = TRUE are chunked as normal text.
    text_sql = f"""
        SELECT
            et.extracted_text_id,
            et.document_id,
            d.corpus_zone,
            d.corpus_pack,
            d.workstream,
            et.page_no,
            et.section_heading,
            et.text_content
        FROM extracted_text et
        JOIN documents d
            ON d.document_id = et.document_id
        WHERE d.index_in_rag = TRUE
          AND et.text_content IS NOT NULL
          AND LENGTH(TRIM(et.text_content)) > 0
          {new_documents_filter}
        ORDER BY
            et.document_id,
            et.page_no NULLS LAST,
            et.extracted_text_id;
    """

    text_df = pd.read_sql(text(text_sql), engine)

    # ---------------------------------------------------------------------
    # 2. Read table rows.
    # ---------------------------------------------------------------------
    # CSV / Excel files may not be directly indexed in RAG, but their rows
    # may still contain analytical evidence. So table rows use extraction_required.
    table_sql = f"""
        SELECT
            tr.table_id,
            tr.document_id,
            tr.row_number,
            tr.row_data,
            t.table_name,
            t.sheet_name,
            t.page_no,
            d.corpus_zone,
            d.corpus_pack,
            d.workstream
        FROM extracted_table_rows tr
        JOIN extracted_tables t
            ON t.table_id = tr.table_id
        JOIN documents d
            ON d.document_id = tr.document_id
        WHERE d.extraction_required = TRUE
          {new_documents_filter}
        ORDER BY
            tr.document_id,
            tr.table_id,
            tr.row_number;
    """

    table_rows_df = pd.read_sql(text(table_sql), engine)

    chunk_rows = []
    doc_chunk_counter = {}

    def next_chunk_index(document_id):
        document_id = str(document_id)
        current_value = doc_chunk_counter.get(document_id, 0) + 1
        doc_chunk_counter[document_id] = current_value
        return current_value

    # ---------------------------------------------------------------------
    # 3. Create chunks from extracted_text.
    # ---------------------------------------------------------------------
    for _, row in text_df.iterrows():
        document_id = str(row["document_id"])
        extracted_text_id = int(row["extracted_text_id"])

        corpus_zone = clean_text(row["corpus_zone"])
        corpus_pack = clean_text(row["corpus_pack"])
        workstream = clean_text(row["workstream"])

        page_no = nullable_int(row["page_no"])
        section_heading = clean_text(row["section_heading"])

        source_text_chunks = split_text_into_chunks(
            text_content=row["text_content"],
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens
        )

        for source_text_chunk in source_text_chunks:
            chunk_index = next_chunk_index(document_id)
            chunk_id = f"{document_id}_CHUNK_{chunk_index:06d}"

            source_reference = (
                f"source_type=extracted_text; "
                f"extracted_text_id={extracted_text_id}; "
                f"page_no={page_no}; "
                f"section_heading={section_heading}"
            )

            chunk_text = clean_text(
                f"""
                Source type: extracted_text
                Document ID: {document_id}
                Corpus zone: {corpus_zone}
                Corpus pack: {corpus_pack}
                Workstream: {workstream}
                Page number: {page_no}
                Section heading: {section_heading}

                {source_text_chunk}
                """
            )

            chunk_rows.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "extracted_text_id": extracted_text_id,
                    "corpus_zone": corpus_zone,
                    "corpus_pack": corpus_pack,
                    "workstream": workstream,
                    "section_heading": section_heading,
                    "page_start": page_no,
                    "page_end": page_no,
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                    "token_count_estimate": estimate_tokens(chunk_text),
                    "source_reference": source_reference,
                    "embedding_model": embedding_model,
                }
            )

    # ---------------------------------------------------------------------
    # 4. Create chunks from extracted table rows.
    # ---------------------------------------------------------------------
    current_table_id = None
    current_document_id = None
    current_table_name = None
    current_sheet_name = None
    current_page_no = None
    current_corpus_zone = None
    current_corpus_pack = None
    current_workstream = None
    current_row_start = None
    current_row_end = None
    current_lines = []
    current_tokens = 0

    def flush_table_chunk():
        nonlocal current_table_id
        nonlocal current_document_id
        nonlocal current_table_name
        nonlocal current_sheet_name
        nonlocal current_page_no
        nonlocal current_corpus_zone
        nonlocal current_corpus_pack
        nonlocal current_workstream
        nonlocal current_row_start
        nonlocal current_row_end
        nonlocal current_lines
        nonlocal current_tokens

        if not current_lines:
            return

        chunk_index = next_chunk_index(current_document_id)
        chunk_id = f"{current_document_id}_CHUNK_{chunk_index:06d}"

        section_heading = f"table: {current_table_name}"

        source_reference = (
            f"source_type=extracted_table_rows; "
            f"table_id={current_table_id}; "
            f"table_name={current_table_name}; "
            f"sheet_name={current_sheet_name}; "
            f"page_no={current_page_no}; "
            f"row_start={current_row_start}; "
            f"row_end={current_row_end}"
        )

        table_body = clean_text("\n".join(current_lines))

        chunk_text = clean_text(
            f"""
            Source type: extracted_table_rows
            Document ID: {current_document_id}
            Corpus zone: {current_corpus_zone}
            Corpus pack: {current_corpus_pack}
            Workstream: {current_workstream}
            Table ID: {current_table_id}
            Table name: {current_table_name}
            Sheet name: {current_sheet_name}
            Page number: {current_page_no}
            Row range: {current_row_start} to {current_row_end}

            {table_body}
            """
        )

        chunk_rows.append(
            {
                "chunk_id": chunk_id,
                "document_id": current_document_id,
                "extracted_text_id": None,
                "corpus_zone": current_corpus_zone,
                "corpus_pack": current_corpus_pack,
                "workstream": current_workstream,
                "section_heading": section_heading,
                "page_start": current_page_no,
                "page_end": current_page_no,
                "chunk_index": chunk_index,
                "chunk_text": chunk_text,
                "token_count_estimate": estimate_tokens(chunk_text),
                "source_reference": source_reference,
                "embedding_model": embedding_model,
            }
        )

        current_lines = []
        current_tokens = 0
        current_row_start = None
        current_row_end = None

    for _, row in table_rows_df.iterrows():
        table_id = str(row["table_id"])
        document_id = str(row["document_id"])
        row_number = int(row["row_number"])

        row_text = row_data_to_text(row["row_data"])

        if row_text == "":
            continue

        table_name = clean_text(row["table_name"])
        sheet_name = clean_text(row["sheet_name"])
        page_no = nullable_int(row["page_no"])

        corpus_zone = clean_text(row["corpus_zone"])
        corpus_pack = clean_text(row["corpus_pack"])
        workstream = clean_text(row["workstream"])

        line_text = f"Row {row_number}: {row_text}"
        line_tokens = estimate_tokens(line_text)

        if current_table_id is None:
            current_table_id = table_id
            current_document_id = document_id
            current_table_name = table_name
            current_sheet_name = sheet_name
            current_page_no = page_no
            current_corpus_zone = corpus_zone
            current_corpus_pack = corpus_pack
            current_workstream = workstream
            current_row_start = row_number

        table_changed = table_id != current_table_id

        chunk_full = False
        if current_lines:
            if current_tokens + line_tokens > chunk_size_tokens:
                chunk_full = True

        if table_changed or chunk_full:
            flush_table_chunk()

            current_table_id = table_id
            current_document_id = document_id
            current_table_name = table_name
            current_sheet_name = sheet_name
            current_page_no = page_no
            current_corpus_zone = corpus_zone
            current_corpus_pack = corpus_pack
            current_workstream = workstream
            current_row_start = row_number

        current_lines.append(line_text)
        current_tokens += line_tokens
        current_row_end = row_number

    flush_table_chunk()

    # ---------------------------------------------------------------------
    # 5. Load chunks into existing chunks table.
    # ---------------------------------------------------------------------
    with engine.begin() as conn:
        if rebuild_inventory == "Y":
            # Full refresh for POC simplicity.
            conn.execute(text("DELETE FROM chunks;"))

        if len(chunk_rows) > 0:
            insert_sql = text("""
                INSERT INTO chunks (
                    chunk_id,
                    document_id,
                    extracted_text_id,
                    corpus_zone,
                    corpus_pack,
                    workstream,
                    section_heading,
                    page_start,
                    page_end,
                    chunk_index,
                    chunk_text,
                    token_count_estimate,
                    source_reference,
                    embedding_model,
                    created_at
                )
                VALUES (
                    :chunk_id,
                    :document_id,
                    :extracted_text_id,
                    :corpus_zone,
                    :corpus_pack,
                    :workstream,
                    :section_heading,
                    :page_start,
                    :page_end,
                    :chunk_index,
                    :chunk_text,
                    :token_count_estimate,
                    :source_reference,
                    :embedding_model,
                    NOW()
                );
            """)

            conn.execute(insert_sql, chunk_rows)

    text_chunks_count = 0
    table_chunks_count = 0
    documents_with_chunks = 0

    if len(chunk_rows) > 0:
        chunks_df = pd.DataFrame(chunk_rows)

        text_chunks_count = int(chunks_df["extracted_text_id"].notna().sum())
        table_chunks_count = int(chunks_df["extracted_text_id"].isna().sum())
        documents_with_chunks = int(chunks_df["document_id"].nunique())

    return {
        "message": "Document chunking completed.",
        "mode": "rebuild" if rebuild_inventory == "Y" else "update",
        "output_table": "chunks",
        "chunk_size_tokens": chunk_size_tokens,
        "chunk_overlap_tokens": chunk_overlap_tokens,
        "embedding_model": embedding_model,
        "text_source_rows": int(len(text_df)),
        "table_source_rows": int(len(table_rows_df)),
        "text_chunks": text_chunks_count,
        "table_chunks": table_chunks_count,
        "total_chunks": int(len(chunk_rows)),
        "documents_with_chunks": documents_with_chunks,
        "next_step": "Run embed_chunks.py to populate chunks.embedding where embedding IS NULL.",
    }


if __name__ == "__main__":
    print(chunk_documents(rebuild_inventory="Y"))
