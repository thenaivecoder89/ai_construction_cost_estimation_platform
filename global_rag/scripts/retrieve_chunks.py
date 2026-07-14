# File_name: retrieve_chunks.py
# Purpose:
# Test retrieval from embedded chunks using pgvector and/or PostgreSQL full-text search.
#
# Input table:
# 1. chunks
#
# Retrieval modes:
# 1. vector  - semantic retrieval using chunks.embedding
# 2. keyword - exact/keyword retrieval using chunks.search_vector
# 3. hybrid  - combines vector + keyword using simple reciprocal-rank fusion
#
# Notes:
# - Bare-bones POC version.
# - No classes.
# - Does not call the LLM.
# - Returns retrieved chunks with source traceability.

import pandas as pd
from sqlalchemy import create_engine, text
from openai import OpenAI

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


def nullable_value(value):
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def vector_to_pgvector(embedding):
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


def check_chunks_ready(engine):
    required_columns = [
        "chunk_id",
        "document_id",
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
    ]

    sql = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'chunks';
    """

    columns_df = pd.read_sql(text(sql), engine)
    existing_columns = set(columns_df["column_name"].tolist())

    missing_columns = []

    for col in required_columns:
        if col not in existing_columns:
            missing_columns.append(col)

    if missing_columns:
        raise RuntimeError(
            "The chunks table is missing these required columns: "
            + ", ".join(missing_columns)
        )


def build_filter_sql(corpus_zone=None, corpus_pack=None, document_id=None):
    conditions = [
        "chunk_text IS NOT NULL",
        "LENGTH(TRIM(chunk_text)) > 0"
    ]

    params = {}

    if corpus_zone:
        conditions.append("corpus_zone = :corpus_zone")
        params["corpus_zone"] = corpus_zone

    if corpus_pack:
        conditions.append("corpus_pack = :corpus_pack")
        params["corpus_pack"] = corpus_pack

    if document_id:
        conditions.append("document_id = :document_id")
        params["document_id"] = document_id

    where_sql = " AND ".join(conditions)

    return where_sql, params


def embed_query(client, query_text, embedding_model, embedding_dimension):
    query_text = clean_text(query_text)

    if query_text == "":
        raise RuntimeError("Query text is empty.")

    response = client.embeddings.create(
        model=embedding_model,
        input=query_text
    )

    query_embedding = response.data[0].embedding

    if len(query_embedding) != embedding_dimension:
        raise RuntimeError(
            f"Query embedding dimension mismatch. "
            f"Expected {embedding_dimension}, got {len(query_embedding)}."
        )

    return vector_to_pgvector(query_embedding)


def dataframe_to_results(df, max_chunk_chars):
    results = []

    for _, row in df.iterrows():
        chunk_text = clean_text(row["chunk_text"])

        if max_chunk_chars is not None:
            if int(max_chunk_chars) > 0:
                if len(chunk_text) > int(max_chunk_chars):
                    chunk_text = chunk_text[:int(max_chunk_chars)] + "\n...[truncated]"

        result = {
            "chunk_id": clean_text(row["chunk_id"]),
            "document_id": clean_text(row["document_id"]),
            "corpus_zone": clean_text(row["corpus_zone"]),
            "corpus_pack": clean_text(row["corpus_pack"]),
            "workstream": clean_text(row["workstream"]),
            "section_heading": clean_text(row["section_heading"]),
            "page_start": nullable_value(row["page_start"]),
            "page_end": nullable_value(row["page_end"]),
            "chunk_index": nullable_value(row["chunk_index"]),
            "token_count_estimate": nullable_value(row["token_count_estimate"]),
            "source_reference": clean_text(row["source_reference"]),
            "chunk_text": chunk_text,
        }

        if "similarity_score" in df.columns:
            result["similarity_score"] = float(row["similarity_score"])

        if "distance" in df.columns:
            result["distance"] = float(row["distance"])

        if "keyword_score" in df.columns:
            result["keyword_score"] = float(row["keyword_score"])

        results.append(result)

    return results


def run_vector_search(
    engine,
    query_embedding,
    top_k,
    corpus_zone=None,
    corpus_pack=None,
    document_id=None,
    max_chunk_chars=3000
):
    where_sql, params = build_filter_sql(
        corpus_zone=corpus_zone,
        corpus_pack=corpus_pack,
        document_id=document_id
    )

    where_sql = where_sql + " AND embedding IS NOT NULL"

    params["query_embedding"] = query_embedding
    params["top_k"] = int(top_k)

    sql = f"""
        SELECT
            chunk_id,
            document_id,
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
            embedding <=> CAST(:query_embedding AS vector) AS distance,
            1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity_score
        FROM chunks
        WHERE {where_sql}
        ORDER BY embedding <=> CAST(:query_embedding AS vector)
        LIMIT :top_k;
    """

    df = pd.read_sql(text(sql), engine, params=params)

    return dataframe_to_results(df, max_chunk_chars)


def run_keyword_search(
    engine,
    query_text,
    top_k,
    corpus_zone=None,
    corpus_pack=None,
    document_id=None,
    max_chunk_chars=3000
):
    where_sql, params = build_filter_sql(
        corpus_zone=corpus_zone,
        corpus_pack=corpus_pack,
        document_id=document_id
    )

    where_sql = where_sql + """
        AND search_vector @@ plainto_tsquery('english', :query_text)
    """

    params["query_text"] = query_text
    params["top_k"] = int(top_k)

    sql = f"""
        SELECT
            chunk_id,
            document_id,
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
            ts_rank(search_vector, plainto_tsquery('english', :query_text)) AS keyword_score
        FROM chunks
        WHERE {where_sql}
        ORDER BY keyword_score DESC
        LIMIT :top_k;
    """

    df = pd.read_sql(text(sql), engine, params=params)

    return dataframe_to_results(df, max_chunk_chars)


def run_hybrid_search(
    engine,
    client,
    query_text,
    embedding_model,
    embedding_dimension,
    top_k,
    corpus_zone=None,
    corpus_pack=None,
    document_id=None,
    max_chunk_chars=3000
):
    query_embedding = embed_query(
        client=client,
        query_text=query_text,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension
    )

    # Pull a larger candidate pool from both methods, then combine.
    candidate_k = max(int(top_k) * 3, 10)

    vector_results = run_vector_search(
        engine=engine,
        query_embedding=query_embedding,
        top_k=candidate_k,
        corpus_zone=corpus_zone,
        corpus_pack=corpus_pack,
        document_id=document_id,
        max_chunk_chars=max_chunk_chars
    )

    keyword_results = run_keyword_search(
        engine=engine,
        query_text=query_text,
        top_k=candidate_k,
        corpus_zone=corpus_zone,
        corpus_pack=corpus_pack,
        document_id=document_id,
        max_chunk_chars=max_chunk_chars
    )

    combined = {}

    # Reciprocal-rank fusion. Simple and stable for a POC.
    rrf_k = 60

    for rank, item in enumerate(vector_results, start=1):
        chunk_id = item["chunk_id"]

        if chunk_id not in combined:
            combined[chunk_id] = item

        combined[chunk_id]["vector_rank"] = rank
        combined[chunk_id]["hybrid_score"] = combined[chunk_id].get("hybrid_score", 0) + (1 / (rrf_k + rank))

    for rank, item in enumerate(keyword_results, start=1):
        chunk_id = item["chunk_id"]

        if chunk_id not in combined:
            combined[chunk_id] = item

        combined[chunk_id]["keyword_rank"] = rank
        combined[chunk_id]["keyword_score"] = item.get("keyword_score")
        combined[chunk_id]["hybrid_score"] = combined[chunk_id].get("hybrid_score", 0) + (1 / (rrf_k + rank))

    combined_results = list(combined.values())

    combined_results = sorted(
        combined_results,
        key=lambda x: x.get("hybrid_score", 0),
        reverse=True
    )

    return combined_results[:int(top_k)]


def retrieve_chunks(
    query_text,
    top_k=10,
    mode="hybrid",
    corpus_zone=None,
    corpus_pack=None,
    document_id=None,
    max_chunk_chars=3000
):
    config_base = config.config_base()

    db_url = config_base["db_url"]
    openai_api_key = config_base["openai_api_key"]
    embedding_model = config_base["embedding_model"]
    embedding_dimension = int(config_base["embedding_dimension"])

    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in Railway environment variables.")

    query_text = clean_text(query_text)

    if query_text == "":
        raise RuntimeError("Query text is empty.")

    top_k = int(top_k)

    if top_k <= 0:
        top_k = 10

    if top_k > 50:
        top_k = 50

    mode = clean_text(mode).lower()

    if mode not in ["vector", "keyword", "hybrid"]:
        mode = "hybrid"

    engine = create_engine(
        url=db_url,
        pool_pre_ping=True
    )

    client = OpenAI(api_key=openai_api_key)

    check_chunks_ready(engine)

    if mode == "vector":
        query_embedding = embed_query(
            client=client,
            query_text=query_text,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension
        )

        results = run_vector_search(
            engine=engine,
            query_embedding=query_embedding,
            top_k=top_k,
            corpus_zone=corpus_zone,
            corpus_pack=corpus_pack,
            document_id=document_id,
            max_chunk_chars=max_chunk_chars
        )

    elif mode == "keyword":
        results = run_keyword_search(
            engine=engine,
            query_text=query_text,
            top_k=top_k,
            corpus_zone=corpus_zone,
            corpus_pack=corpus_pack,
            document_id=document_id,
            max_chunk_chars=max_chunk_chars
        )

    else:
        results = run_hybrid_search(
            engine=engine,
            client=client,
            query_text=query_text,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            top_k=top_k,
            corpus_zone=corpus_zone,
            corpus_pack=corpus_pack,
            document_id=document_id,
            max_chunk_chars=max_chunk_chars
        )

    return {
        "message": "Retrieval completed.",
        "query_text": query_text,
        "retrieval_mode": mode,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "top_k_requested": top_k,
        "results_returned": len(results),
        "filters": {
            "corpus_zone": corpus_zone,
            "corpus_pack": corpus_pack,
            "document_id": document_id,
        },
        "results": results,
        "next_step": "Review retrieved chunks for relevance before connecting retrieval to final report generation.",
    }


if __name__ == "__main__":
    print(
        retrieve_chunks(
            query_text="What are the key investment risks?",
            top_k=10,
            mode="hybrid"
        )
    )