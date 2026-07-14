# File_name: embed_chunks.py
# Purpose:
# Generate embeddings for existing chunks and update chunks.embedding.
#
# Input table:
# 1. chunks
#
# Output update:
# 1. chunks.embedding
# 2. chunks.embedding_model
#
# Notes:
# - Bare-bones POC version.
# - No classes.
# - Uses OpenAI text-embedding-3-small from config.py.
# - Assumes pgvector extension is enabled and chunks.embedding is vector(1536).
# - search_vector is NOT manually updated because it is a generated column.
# - This version avoids upstream errors by processing small batches per API call.

import time
import pandas as pd
from sqlalchemy import create_engine, text
from openai import OpenAI, RateLimitError

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


def vector_to_pgvector(embedding):
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


def check_chunks_ready(engine):
    sql = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'chunks';
    """

    columns_df = pd.read_sql(text(sql), engine)
    existing_columns = set(columns_df["column_name"].tolist())

    required_columns = [
        "chunk_id",
        "chunk_text",
        "embedding_model",
        "embedding",
    ]

    missing_columns = []

    for col in required_columns:
        if col not in existing_columns:
            missing_columns.append(col)

    if missing_columns:
        raise RuntimeError(
            "The chunks table is missing these required columns: "
            + ", ".join(missing_columns)
        )


def get_remaining_chunks(engine):
    remaining_sql = """
        SELECT COUNT(*) AS remaining_chunks
        FROM chunks
        WHERE chunk_text IS NOT NULL
          AND LENGTH(TRIM(chunk_text)) > 0
          AND embedding IS NULL;
    """

    remaining_df = pd.read_sql(text(remaining_sql), engine)
    return int(remaining_df.loc[0, "remaining_chunks"])


def embed_chunks(rebuild_inventory: str = "Y"):
    config_base = config.config_base()
    rebuild_inventory = rebuild_inventory.strip().upper()

    if rebuild_inventory not in ["Y", "N"]:
        raise ValueError("rebuild_inventory must be 'Y' or 'N'.")

    db_url = config_base["db_url"]
    openai_api_key = config_base["openai_api_key"]
    embedding_model = config_base["embedding_model"]
    embedding_dimension = int(config_base["embedding_dimension"])

    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in Railway environment variables.")

    engine = create_engine(
        url=db_url,
        pool_pre_ping=True
    )

    client = OpenAI(api_key=openai_api_key)

    check_chunks_ready(engine)

    # Per OpenAI API request. Keep this below the per-run cap to reduce
    # rate-limit risk and make partial progress if a later batch is throttled.
    batch_size = 100

    # Each API call to /embed_chunks will process only this many batches.
    # Re-run the endpoint until remaining_chunks_without_embedding = 0.
    max_batches_per_run = 10

    # Small pause between successful batches.
    sleep_seconds_between_batches = 6

    limit_chunks = batch_size * max_batches_per_run

    remaining_before = get_remaining_chunks(engine)

    embedding_filter = "embedding IS NULL"
    if rebuild_inventory == "Y":
        embedding_filter = """
                embedding IS NULL
                OR embedding_model IS DISTINCT FROM :embedding_model
        """

    chunks_sql = f"""
        SELECT
            chunk_id,
            chunk_text
        FROM chunks
        WHERE chunk_text IS NOT NULL
          AND LENGTH(TRIM(chunk_text)) > 0
          AND (
                {embedding_filter}
              )
        ORDER BY
            document_id,
            chunk_index,
            chunk_id
        LIMIT :limit_chunks;
    """

    chunks_df = pd.read_sql(
        text(chunks_sql),
        engine,
        params={
            "embedding_model": embedding_model,
            "limit_chunks": limit_chunks
        }
    )

    chunks_selected_this_run = len(chunks_df)
    chunks_embedded = 0
    batches_processed = 0

    update_sql = text("""
        UPDATE chunks
        SET
            embedding = CAST(:embedding AS vector),
            embedding_model = :embedding_model
        WHERE chunk_id = :chunk_id;
    """)

    for start_index in range(0, chunks_selected_this_run, batch_size):
        batch_df = chunks_df.iloc[start_index:start_index + batch_size].copy()

        input_texts = []

        for _, row in batch_df.iterrows():
            chunk_text = clean_text(row["chunk_text"])

            # Safety cap. Your chunking is already around 1,000 tokens,
            # so this should normally not cut anything.
            chunk_text = chunk_text[:30000]

            input_texts.append(chunk_text.replace("\n", " "))

        try:
            response = client.embeddings.create(
                model=embedding_model,
                input=input_texts
            )

        except RateLimitError as e:
            remaining_after_rate_limit = get_remaining_chunks(engine)

            return {
                "message": "Embedding paused because OpenAI rate limit was reached.",
                "status": "rate_limited",
                "mode": "rebuild" if rebuild_inventory == "Y" else "update",
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
                "chunks_per_openai_request": batch_size,
                "max_chunks_per_endpoint_run": limit_chunks,
                "remaining_chunks_before_run": remaining_before,
                "chunks_selected_this_run": int(chunks_selected_this_run),
                "chunks_embedded_this_run": int(chunks_embedded),
                "batches_processed_this_run": int(batches_processed),
                "remaining_chunks_without_embedding": remaining_after_rate_limit,
                "recommended_action": "Wait around 60 seconds and call /embed_chunks again.",
                "error_message": str(e)[:1000],
            }

        update_rows = []

        for item in response.data:
            row = batch_df.iloc[item.index]
            embedding = item.embedding

            if len(embedding) != embedding_dimension:
                raise RuntimeError(
                    f"Embedding dimension mismatch for chunk_id={row['chunk_id']}. "
                    f"Expected {embedding_dimension}, got {len(embedding)}."
                )

            update_rows.append(
                {
                    "chunk_id": str(row["chunk_id"]),
                    "embedding": vector_to_pgvector(embedding),
                    "embedding_model": embedding_model,
                }
            )

        with engine.begin() as conn:
            conn.execute(update_sql, update_rows)

        chunks_embedded += len(update_rows)
        batches_processed += 1

        if batches_processed < max_batches_per_run:
            time.sleep(sleep_seconds_between_batches)

    remaining_after = get_remaining_chunks(engine)

    return {
        "message": "Document embedding batch completed.",
        "status": "ok",
        "mode": "rebuild" if rebuild_inventory == "Y" else "update",
        "input_table": "chunks",
        "updated_table": "chunks",
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "chunks_per_openai_request": batch_size,
        "max_batches_per_run": max_batches_per_run,
        "max_chunks_per_endpoint_run": limit_chunks,
        "remaining_chunks_before_run": remaining_before,
        "chunks_selected_this_run": int(chunks_selected_this_run),
        "chunks_embedded_this_run": int(chunks_embedded),
        "batches_processed_this_run": int(batches_processed),
        "remaining_chunks_without_embedding": remaining_after,
        "next_step": "Re-run /embed_chunks until remaining_chunks_without_embedding is 0.",
    }


if __name__ == "__main__":
    print(embed_chunks(rebuild_inventory="Y"))
