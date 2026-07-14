import global_rag.scripts.build_document_inventory as bdi
import global_rag.scripts.config as cnfg
import global_rag.scripts.extract_documents as ed
import global_rag.scripts.chunk_documents as cd
import global_rag.scripts.embed_chunks as emb
import global_rag.scripts.retrieve_chunks as ret
import global_rag.scripts.report_generation as rg
import global_rag.scripts.wb_scraper as wb
import global_rag.scripts.country_macro_llm_call as cmllm
import global_rag.scripts.country_arima_llm_call as arimallm
from global_rag.scripts import investment_chatbot as chatbot

from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse
from pathlib import Path

app = FastAPI()

config_base = cnfg.config_base()
origins = config_base["allowed_origins"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/health", status_code=200)
def health_check():
    return {"status": "ok", "message": "FastAPI service is running"}

@app.get(path="/build_document_inventory", status_code=200)
def build_doc_inv(client_data: str, rebuild_inventory: str = "Y"):
    build_document_inventory_output = bdi.build_document_inventory(
        client_data=client_data,
        rebuild_inventory=rebuild_inventory
    )
    api_response = JSONResponse(
        {
            "status": "ok",
            "output": build_document_inventory_output
        }
    )
    return api_response

@app.get("/debug_paths")
def debug_paths():
    base = Path("/app")

    return {
        "app_exists": base.exists(),
        "app_children": [str(p) for p in base.iterdir()] if base.exists() else [],
        "cwd": str(Path.cwd()),
    }

@app.get(path="/extract_documents", status_code=200)
def extract_docs(client_data: str, rebuild_inventory: str = "Y"):
    extract_documents_output = ed.extract_documents(
        client_data=client_data,
        rebuild_inventory=rebuild_inventory
    )

    api_response = JSONResponse(
        {
            "status": "ok",
            "output": extract_documents_output
        }
    )

    return api_response

@app.get(path="/chunk_documents", status_code=200)
def chunk_docs(rebuild_inventory: str = "Y"):
    chunk_documents_output = cd.chunk_documents(
        rebuild_inventory=rebuild_inventory
    )

    api_response = JSONResponse(
        {
            "status": "ok",
            "output": chunk_documents_output
        }
    )

    return api_response

@app.get(path="/embed_chunks", status_code=200)
def embed_chunks(rebuild_inventory: str = "Y"):
    embed_documents_output = emb.embed_chunks(
        rebuild_inventory=rebuild_inventory
    )

    api_response = JSONResponse(
        {
            "status": "ok",
            "output": embed_documents_output
        }
    )

    return api_response

@app.get(path="/scrape_world_bank_wdi", status_code=200)
def scrape_world_bank_wdi(
    country_codes: Optional[str] = None,
    start_year: int = 2010,
    end_year: int = 2024
):
    parsed_country_codes = None
    if country_codes:
        parsed_country_codes = [
            country_code.strip()
            for country_code in country_codes.split(",")
            if country_code.strip()
        ]

    scrape_output = wb.scrape_world_bank_wdi(
        country_codes=parsed_country_codes,
        start_year=start_year,
        end_year=end_year
    )

    api_response = JSONResponse(
        {
            "status": "ok",
            "output": scrape_output
        }
    )

    return api_response

@app.get(path="/retrieve_chunks", status_code=200)
def retrieve_chunks_api(
    q: str,
    top_k: int = 10,
    mode: str = "hybrid",
    corpus_zone: Optional[str] = None,
    corpus_pack: Optional[str] = None,
    document_id: Optional[str] = None,
    max_chunk_chars: int = 3000
):
    retrieval_output = ret.retrieve_chunks(
        query_text=q,
        top_k=top_k,
        mode=mode,
        corpus_zone=corpus_zone,
        corpus_pack=corpus_pack,
        document_id=document_id,
        max_chunk_chars=max_chunk_chars
    )

    api_response = JSONResponse(
        {
            "status": "ok",
            "output": retrieval_output
        }
    )

    return api_response

@app.get(path="/generate_ic_review_report", status_code=200)
def generate_ic_review_report_api(
    transaction_id: str,
    use_llm_summary: bool = True,
    write_audit: bool = True
):
    report_generation_output = rg.generate_investment_ic_review_report(
        transaction_id=transaction_id,
        use_llm_summary=use_llm_summary,
        write_audit=write_audit
    )

    api_response = JSONResponse(
        content=jsonable_encoder(
            {
                "status": "ok",
                "output": report_generation_output
            }
        )
    )

    return api_response

@app.get(path="/country_macro_llm_call", status_code=200)
def country_macro_llm_call_api(
    n_clusters: int = 4,
    schema: str = "public",
    table_name: str = "country_features_raw",
    focus_country: str = "UAE",
    include_graphs_in_llm: bool = True
):
    country_macro_llm_output = cmllm.llm_call(
        n_clusters=n_clusters,
        schema=schema,
        table_name=table_name,
        focus_country=focus_country,
        include_graphs_in_llm=include_graphs_in_llm
    )

    api_response = JSONResponse(
        content=jsonable_encoder(
            {
                "status": "ok",
                "output": country_macro_llm_output
            }
        )
    )

    return api_response

@app.get(path="/country_arima_llm_call", status_code=200)
def country_arima_llm_call_api(
    forecast_years: int = 3,
    schema: str = "public",
    country_codes: Optional[str] = None,
    focus_country: str = "ARE",
    include_graphs_in_llm: bool = True,
    max_graphs_to_send: int = 27,
    max_output_tokens: int = 16000
):
    parsed_country_codes = None

    if country_codes:
        parsed_country_codes = [
            country_code.strip().upper()
            for country_code in country_codes.split(",")
            if country_code.strip()
        ]

    country_arima_llm_output = arimallm.llm_call(
        forecast_years=forecast_years,
        schema=schema,
        country_codes=parsed_country_codes,
        focus_country=focus_country,
        include_graphs_in_llm=include_graphs_in_llm,
        max_graphs_to_send=max_graphs_to_send,
        max_output_tokens=max_output_tokens
    )

    api_response = JSONResponse(
        content=jsonable_encoder(
            {
                "status": "ok",
                "output": country_arima_llm_output
            }
        )
    )

    return api_response

@app.get(path="/investment_chatbot", status_code=200)
def investment_chatbot_api(
    q: str,
    client_data_pack: Optional[str] = None,
    top_k: int = 8,
    workstream: Optional[str] = None,
    corpus_pack_filter: Optional[str] = None,
    max_output_tokens: int = 1200
):
    try:
        chatbot_output = chatbot.answer_question(
            question=q,
            client_data_pack=client_data_pack,
            chat_history=None,
            top_k=top_k,
            workstream=workstream,
            corpus_pack_filter=corpus_pack_filter,
            max_output_tokens=max_output_tokens
        )

        api_response = JSONResponse(
            content=jsonable_encoder(
                {
                    "status": "ok",
                    "output": chatbot_output
                }
            )
        )

        return api_response

    except Exception as e:
        api_response = JSONResponse(
            status_code=500,
            content=jsonable_encoder(
                {
                    "status": "error",
                    "message": "investment_chatbot failed",
                    "error": str(e)
                }
            )
        )

        return api_response