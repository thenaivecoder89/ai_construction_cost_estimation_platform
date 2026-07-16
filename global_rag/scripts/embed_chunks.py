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
import os
import pandas as pd
from sqlalchemy import create_engine, text
from openai import OpenAI, RateLimitError, BadRequestError

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


def positive_int_env(name, default_value):
    try:
        value = int(os.getenv(name, default_value))
        if value > 0:
            return value
    except Exception:
        pass

    return default_value


def non_negative_float_env(name, default_value):
    try:
        value = float(os.getenv(name, default_value))
        if value >= 0:
            return value
    except Exception:
        pass

    return default_value


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
    final_status = None

    for status_update in iter_embed_chunks(rebuild_inventory=rebuild_inventory):
        final_status = status_update

    if final_status is None:
        raise RuntimeError("Embedding did not produce a final status.")

    return final_status


def iter_embed_chunks(rebuild_inventory: str = "Y"):
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
    batch_size = positive_int_env("EMBED_BATCH_SIZE", 100)

    # Each API call to /embed_chunks will process only this many batches.
    # Re-run the endpoint until remaining_chunks_without_embedding = 0.
    max_batches_per_run = positive_int_env("EMBED_MAX_BATCHES_PER_RUN", 10)

    # Small pause between successful batches.
    sleep_seconds_between_batches = non_negative_float_env("EMBED_SLEEP_SECONDS_BETWEEN_BATCHES", 6)
    max_embedding_input_chars = positive_int_env("EMBED_MAX_INPUT_CHARS_PER_CHUNK", 8000)
    rate_limit_sleep_seconds = non_negative_float_env("EMBED_RATE_LIMIT_SLEEP_SECONDS", 60)
    max_rate_limit_retries = positive_int_env("EMBED_MAX_RATE_LIMIT_RETRIES", 10)

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
    limit_chunks_per_cycle = batch_size * max_batches_per_run
    chunks_selected_this_run = 0
    chunks_embedded = 0
    batches_processed = 0
    cycles_processed = 0

    update_sql = text("""
        UPDATE chunks
        SET
            embedding = CAST(:embedding AS vector),
            embedding_model = :embedding_model
        WHERE chunk_id = :chunk_id;
    """)

    while True:
        chunks_df = pd.read_sql(
            text(chunks_sql),
            engine,
            params={
                "embedding_model": embedding_model,
                "limit_chunks": limit_chunks_per_cycle
            }
        )

        chunks_selected_this_cycle = len(chunks_df)

        if chunks_selected_this_cycle == 0:
            break

        chunks_selected_this_run += chunks_selected_this_cycle
        cycles_processed += 1

        for start_index in range(0, chunks_selected_this_cycle, batch_size):
            batch_df = chunks_df.iloc[start_index:start_index + batch_size].copy()

            input_texts = []

            for _, row in batch_df.iterrows():
                chunk_text = clean_text(row["chunk_text"])

                # Safety cap to keep each OpenAI embedding request under the
                # model's aggregate token limit when a batch has long chunks.
                chunk_text = chunk_text[:max_embedding_input_chars]

                input_texts.append(chunk_text.replace("\n", " "))

            rate_limit_retries = 0

            while True:
                try:
                    response = client.embeddings.create(
                        model=embedding_model,
                        input=input_texts
                    )
                    break

                except RateLimitError as e:
                    rate_limit_retries += 1
                    remaining_after_rate_limit = get_remaining_chunks(engine)

                    yield {
                        "message": "OpenAI rate limit reached. Waiting before retrying the same batch.",
                        "status": "rate_limited_retrying",
                        "mode": "rebuild" if rebuild_inventory == "Y" else "update",
                        "embedding_model": embedding_model,
                        "embedding_dimension": embedding_dimension,
                        "chunks_per_openai_request": batch_size,
                        "max_batches_per_cycle": max_batches_per_run,
                        "max_chunks_per_cycle": limit_chunks_per_cycle,
                        "max_embedding_input_chars_per_chunk": max_embedding_input_chars,
                        "remaining_chunks_before_run": remaining_before,
                        "chunks_selected_this_run": int(chunks_selected_this_run),
                        "chunks_embedded_this_run": int(chunks_embedded),
                        "batches_processed_this_run": int(batches_processed),
                        "cycles_processed_this_run": int(cycles_processed),
                        "rate_limit_retries_for_current_batch": rate_limit_retries,
                        "remaining_chunks_without_embedding": remaining_after_rate_limit,
                        "retry_sleep_seconds": rate_limit_sleep_seconds,
                        "error_message": str(e)[:1000],
                    }

                    if rate_limit_retries >= max_rate_limit_retries:
                        yield {
                            "message": "Embedding stopped after reaching max rate-limit retries.",
                            "status": "rate_limited_stopped",
                            "mode": "rebuild" if rebuild_inventory == "Y" else "update",
                            "embedding_model": embedding_model,
                            "embedding_dimension": embedding_dimension,
                            "chunks_per_openai_request": batch_size,
                            "max_batches_per_cycle": max_batches_per_run,
                            "max_chunks_per_cycle": limit_chunks_per_cycle,
                            "max_embedding_input_chars_per_chunk": max_embedding_input_chars,
                            "remaining_chunks_before_run": remaining_before,
                            "chunks_selected_this_run": int(chunks_selected_this_run),
                            "chunks_embedded_this_run": int(chunks_embedded),
                            "batches_processed_this_run": int(batches_processed),
                            "cycles_processed_this_run": int(cycles_processed),
                            "remaining_chunks_without_embedding": remaining_after_rate_limit,
                            "recommended_action": "Wait and call /embed_chunks again.",
                            "error_message": str(e)[:1000],
                        }
                        return

                    if rate_limit_sleep_seconds > 0:
                        time.sleep(rate_limit_sleep_seconds)

                except BadRequestError as e:
                    # Batch is still too large for OpenAI's aggregate request limit.
                    # Stop explicitly so the operator can reduce EMBED_BATCH_SIZE or
                    # EMBED_MAX_INPUT_CHARS_PER_CHUNK instead of silently looping.
                    remaining_after_bad_request = get_remaining_chunks(engine)

                    yield {
                        "message": "Embedding stopped because OpenAI rejected the request size.",
                        "status": "request_too_large_stopped",
                        "mode": "rebuild" if rebuild_inventory == "Y" else "update",
                        "embedding_model": embedding_model,
                        "embedding_dimension": embedding_dimension,
                        "chunks_per_openai_request": batch_size,
                        "max_batches_per_cycle": max_batches_per_run,
                        "max_chunks_per_cycle": limit_chunks_per_cycle,
                        "max_embedding_input_chars_per_chunk": max_embedding_input_chars,
                        "remaining_chunks_before_run": remaining_before,
                        "chunks_selected_this_run": int(chunks_selected_this_run),
                        "chunks_embedded_this_run": int(chunks_embedded),
                        "batches_processed_this_run": int(batches_processed),
                        "cycles_processed_this_run": int(cycles_processed),
                        "remaining_chunks_without_embedding": remaining_after_bad_request,
                        "recommended_action": "Reduce EMBED_BATCH_SIZE or EMBED_MAX_INPUT_CHARS_PER_CHUNK and call /embed_chunks again.",
                        "error_message": str(e)[:1000],
                    }
                    return

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
            remaining_after_batch = get_remaining_chunks(engine)

            yield {
                "message": "Embedding batch completed.",
                "status": "batch_completed",
                "mode": "rebuild" if rebuild_inventory == "Y" else "update",
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
                "chunks_per_openai_request": batch_size,
                "max_batches_per_cycle": max_batches_per_run,
                "max_chunks_per_cycle": limit_chunks_per_cycle,
                "max_embedding_input_chars_per_chunk": max_embedding_input_chars,
                "remaining_chunks_before_run": remaining_before,
                "chunks_selected_this_run": int(chunks_selected_this_run),
                "chunks_embedded_this_run": int(chunks_embedded),
                "chunks_embedded_this_batch": int(len(update_rows)),
                "batches_processed_this_run": int(batches_processed),
                "cycles_processed_this_run": int(cycles_processed),
                "remaining_chunks_without_embedding": remaining_after_batch,
            }

            if sleep_seconds_between_batches > 0:
                time.sleep(sleep_seconds_between_batches)

    remaining_after = get_remaining_chunks(engine)

    yield {
        "message": "Document embedding batch completed.",
        "status": "ok",
        "mode": "rebuild" if rebuild_inventory == "Y" else "update",
        "input_table": "chunks",
        "updated_table": "chunks",
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "chunks_per_openai_request": batch_size,
        "max_batches_per_cycle": max_batches_per_run,
        "max_chunks_per_cycle": limit_chunks_per_cycle,
        "max_embedding_input_chars_per_chunk": max_embedding_input_chars,
        "remaining_chunks_before_run": remaining_before,
        "chunks_selected_this_run": int(chunks_selected_this_run),
        "chunks_embedded_this_run": int(chunks_embedded),
        "batches_processed_this_run": int(batches_processed),
        "cycles_processed_this_run": int(cycles_processed),
        "remaining_chunks_without_embedding": remaining_after,
        "next_step": "Embedding is complete." if remaining_after == 0 else "Re-run /embed_chunks if the run was interrupted.",
    }


if __name__ == "__main__":
    print(embed_chunks(rebuild_inventory="Y"))
