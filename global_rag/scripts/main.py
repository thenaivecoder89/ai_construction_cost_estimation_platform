import global_rag.scripts.build_document_inventory as bdi
import global_rag.scripts.config as cnfg
import global_rag.scripts.extract_documents as ed
import global_rag.scripts.chunk_documents as cd
import global_rag.scripts.embed_chunks as emb
import global_rag.scripts.retrieve_chunks as ret
import global_rag.scripts.report_generation as rg
import global_rag.scripts.boq_generation as boq
import global_rag.scripts.boq_generation_v2 as boqv2
import global_rag.scripts.boq_takeoff_layer as takeoff
import global_rag.scripts.boq_analysis as boqa
import global_rag.scripts.wb_scraper as wb
import global_rag.scripts.country_macro_llm_call as cmllm
import global_rag.scripts.country_arima_llm_call as arimallm
from global_rag.scripts import investment_chatbot as chatbot

import json
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from pathlib import Path

app = FastAPI()
background_job_executor = ThreadPoolExecutor(max_workers=1)
background_jobs = {}
background_jobs_lock = Lock()

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


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def elapsed_seconds_for_job(job_record):
    end_time = job_record.get("completed_time") or time.time()
    return int(end_time - job_record["started_time"])


def public_job_record(job_id, include_result=False):
    with background_jobs_lock:
        job_record = background_jobs.get(job_id)

        if job_record is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        public_record = {
            key: value
            for key, value in job_record.items()
            if key not in ["future", "started_time", "completed_time", "result"]
        }
        public_record["elapsed_seconds"] = elapsed_seconds_for_job(job_record)

        if include_result:
            public_record["result"] = job_record.get("result")

        return public_record


def run_background_job(job_id, operation_func, operation_kwargs):
    with background_jobs_lock:
        background_jobs[job_id].update(
            {
                "status": "running",
                "message": "Job is running.",
                "started_at": utc_now_iso(),
                "updated_at": utc_now_iso(),
            }
        )

    try:
        result = operation_func(**operation_kwargs)
        operation_status = (
            result.get("status", "ok")
            if isinstance(result, dict)
            else "ok"
        )

        with background_jobs_lock:
            background_jobs[job_id].update(
                {
                    "status": "completed",
                    "message": (
                        f"Background job completed. Operation status: {operation_status}."
                    ),
                    "operation_status": operation_status,
                    "result": result,
                    "completed_at": utc_now_iso(),
                    "updated_at": utc_now_iso(),
                    "completed_time": time.time(),
                }
            )

    except Exception as exc:
        with background_jobs_lock:
            background_jobs[job_id].update(
                {
                    "status": "failed",
                    "message": "Job failed.",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:4000],
                    "traceback": traceback.format_exc()[-8000:],
                    "completed_at": utc_now_iso(),
                    "updated_at": utc_now_iso(),
                    "completed_time": time.time(),
                }
            )


def start_background_job(operation_name, operation_func, **operation_kwargs):
    job_id = f"JOB_{uuid.uuid4().hex[:12]}"
    created_at = utc_now_iso()

    with background_jobs_lock:
        background_jobs[job_id] = {
            "job_id": job_id,
            "operation": operation_name,
            "status": "queued",
            "message": "Job has been queued.",
            "parameters": operation_kwargs,
            "created_at": created_at,
            "started_at": None,
            "completed_at": None,
            "updated_at": created_at,
            "started_time": time.time(),
            "completed_time": None,
            "result": None,
            "error_type": None,
            "error_message": None,
            "traceback": None,
        }

        future = background_job_executor.submit(
            run_background_job,
            job_id,
            operation_func,
            operation_kwargs,
        )
        background_jobs[job_id]["future"] = future

    return {
        "status": "queued",
        "output": {
            "job_id": job_id,
            "operation": operation_name,
            "message": "Job queued. Poll the status URL until status is completed or failed.",
            "parameters": operation_kwargs,
            "status_url": f"/pipeline_jobs/{job_id}",
            "result_url": f"/pipeline_jobs/{job_id}/result",
        },
    }


@app.get(path="/pipeline_jobs/{job_id}", status_code=200)
def pipeline_job_status(job_id: str):
    return {
        "status": "ok",
        "output": public_job_record(job_id=job_id, include_result=False),
    }


@app.get(path="/pipeline_jobs/{job_id}/result", status_code=200)
def pipeline_job_result(job_id: str, download_file: bool = False):
    job_record = public_job_record(job_id=job_id, include_result=True)

    if job_record["status"] not in ["completed", "failed"]:
        return {
            "status": "running",
            "output": job_record,
        }

    if download_file and job_record["status"] == "completed":
        result = job_record.get("result") or {}
        output_files = result.get("output_files") or {}
        xlsx_path = output_files.get("xlsx_path")

        if job_record.get("operation") in ["generate_boq", "generate_boq_v2"] and xlsx_path:
            xlsx_path = Path(xlsx_path)
            if xlsx_path.exists():
                return FileResponse(
                    path=str(xlsx_path),
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    filename=xlsx_path.name,
                )

            raise HTTPException(
                status_code=404,
                detail=f"Generated BOQ workbook is not available on this Railway container: {xlsx_path}",
            )

    return {
        "status": "ok" if job_record["status"] == "completed" else "error",
        "output": job_record,
    }


@app.get(path="/build_document_inventory/start", status_code=202)
def start_build_doc_inv(client_data: str, rebuild_inventory: str = "Y"):
    return start_background_job(
        operation_name="build_document_inventory",
        operation_func=bdi.build_document_inventory,
        client_data=client_data,
        rebuild_inventory=rebuild_inventory,
    )


@app.get(path="/extract_documents/start", status_code=202)
def start_extract_docs(client_data: str, rebuild_inventory: str = "Y"):
    return start_background_job(
        operation_name="extract_documents",
        operation_func=ed.extract_documents,
        client_data=client_data,
        rebuild_inventory=rebuild_inventory,
    )


@app.get(path="/chunk_documents/start", status_code=202)
def start_chunk_docs(rebuild_inventory: str = "Y"):
    return start_background_job(
        operation_name="chunk_documents",
        operation_func=cd.chunk_documents,
        rebuild_inventory=rebuild_inventory,
    )


@app.get(path="/embed_chunks/start", status_code=202)
def start_embed_chunks(rebuild_inventory: str = "Y"):
    return start_background_job(
        operation_name="embed_chunks",
        operation_func=emb.embed_chunks,
        rebuild_inventory=rebuild_inventory,
    )


@app.get(path="/generate_boq/start", status_code=202)
def start_generate_boq(
    project_id: str,
    write_workbook: bool = True,
):
    return start_background_job(
        operation_name="generate_boq",
        operation_func=boq.generate_boq,
        project_id=project_id,
        write_workbook=write_workbook,
    )


@app.get(path="/generate_boq_v2/start", status_code=202)
def start_generate_boq_v2(
    project_id: str,
    write_workbook: bool = True,
    max_items_per_division: int = 50,
    text_row_limit: int = 2000,
    table_row_limit: int = 5000,
):
    return start_background_job(
        operation_name="generate_boq_v2",
        operation_func=boqv2.generate_boq_v2,
        project_id=project_id,
        write_workbook=write_workbook,
        max_items_per_division=max_items_per_division,
        text_row_limit=text_row_limit,
        table_row_limit=table_row_limit,
    )


@app.get(path="/generate_boq_takeoff/start", status_code=202)
def start_generate_boq_takeoff(project_id: str):
    return start_background_job(
        operation_name="generate_boq_takeoff",
        operation_func=takeoff.generate_boq_takeoff,
        project_id=project_id,
    )


@app.get(path="/analyze_boq/start", status_code=202)
@app.get(path="/boq_analysis/start", status_code=202)
def start_analyze_boq(
    project_id: str,
    boq_run_id: Optional[str] = None,
    write_report: bool = True,
    max_items_for_prompt: int = 300,
):
    return start_background_job(
        operation_name="analyze_boq",
        operation_func=boqa.generate_boq_analysis,
        project_id=project_id,
        boq_run_id=boq_run_id,
        write_report=write_report,
        max_items_for_prompt=max_items_for_prompt,
    )


def stream_pipeline_call(operation_name, operation_func, heartbeat_seconds=15, **operation_kwargs):
    start_time = time.time()

    yield json.dumps(
        {
            "status": "started",
            "output": {
                "message": f"{operation_name} started.",
                "operation": operation_name,
                "elapsed_seconds": 0,
                "parameters": operation_kwargs,
            },
        },
        ensure_ascii=False,
        default=str,
    ) + "\n"

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(operation_func, **operation_kwargs)

        while not future.done():
            time.sleep(heartbeat_seconds)

            if future.done():
                break

            elapsed_seconds = int(time.time() - start_time)
            yield json.dumps(
                {
                    "status": "running",
                    "output": {
                        "message": f"{operation_name} is still running.",
                        "operation": operation_name,
                        "elapsed_seconds": elapsed_seconds,
                        "heartbeat_seconds": heartbeat_seconds,
                    },
                },
                ensure_ascii=False,
                default=str,
            ) + "\n"

        try:
            operation_output = future.result()
        except Exception as exc:
            elapsed_seconds = int(time.time() - start_time)
            yield json.dumps(
                {
                    "status": "error",
                    "output": {
                        "message": f"{operation_name} failed.",
                        "operation": operation_name,
                        "elapsed_seconds": elapsed_seconds,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:2000],
                    },
                },
                ensure_ascii=False,
                default=str,
            ) + "\n"
            return

        elapsed_seconds = int(time.time() - start_time)
        yield json.dumps(
            {
                "status": "ok",
                "output": operation_output,
                "elapsed_seconds": elapsed_seconds,
            },
            ensure_ascii=False,
            default=str,
        ) + "\n"

@app.get(path="/build_document_inventory", status_code=200)
def build_doc_inv(client_data: str, rebuild_inventory: str = "Y", stream: bool = True):
    if stream:
        return JSONResponse(
            status_code=202,
            content=jsonable_encoder(
                start_background_job(
                    operation_name="build_document_inventory",
                    operation_func=bdi.build_document_inventory,
                    client_data=client_data,
                    rebuild_inventory=rebuild_inventory,
                )
            ),
        )

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
def extract_docs(client_data: str, rebuild_inventory: str = "Y", stream: bool = True):
    if stream:
        return JSONResponse(
            status_code=202,
            content=jsonable_encoder(
                start_background_job(
                    operation_name="extract_documents",
                    operation_func=ed.extract_documents,
                    client_data=client_data,
                    rebuild_inventory=rebuild_inventory,
                )
            ),
        )

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
def chunk_docs(rebuild_inventory: str = "Y", stream: bool = True):
    if stream:
        return JSONResponse(
            status_code=202,
            content=jsonable_encoder(
                start_background_job(
                    operation_name="chunk_documents",
                    operation_func=cd.chunk_documents,
                    rebuild_inventory=rebuild_inventory,
                )
            ),
        )

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
def embed_chunks(rebuild_inventory: str = "Y", stream: bool = True, live_stream: bool = False):
    if stream:
        if not live_stream:
            return JSONResponse(
                status_code=202,
                content=jsonable_encoder(
                    start_background_job(
                        operation_name="embed_chunks",
                        operation_func=emb.embed_chunks,
                        rebuild_inventory=rebuild_inventory,
                    )
                ),
            )

        def event_stream():
            for status_update in emb.iter_embed_chunks(rebuild_inventory=rebuild_inventory):
                yield json.dumps(
                    {
                        "status": status_update.get("status", "ok"),
                        "output": status_update,
                    },
                    ensure_ascii=False,
                    default=str,
                ) + "\n"

        return StreamingResponse(
            event_stream(),
            media_type="application/x-ndjson",
        )

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


@app.get(path="/generate_boq", status_code=200)
def generate_boq_api(
    project_id: str,
    write_workbook: bool = True,
    stream: bool = True,
):
    if stream:
        return JSONResponse(
            status_code=202,
            content=jsonable_encoder(
                start_background_job(
                    operation_name="generate_boq",
                    operation_func=boq.generate_boq,
                    project_id=project_id,
                    write_workbook=write_workbook,
                )
            ),
        )

    boq_output = boq.generate_boq(
        project_id=project_id,
        write_workbook=write_workbook,
    )

    return JSONResponse(
        content=jsonable_encoder(
            {
                "status": "ok",
                "output": boq_output,
            }
        )
    )


@app.get(path="/generate_boq_v2", status_code=200)
def generate_boq_v2_api(
    project_id: str,
    write_workbook: bool = True,
    max_items_per_division: int = 50,
    text_row_limit: int = 2000,
    table_row_limit: int = 5000,
    stream: bool = True,
):
    if stream:
        return JSONResponse(
            status_code=202,
            content=jsonable_encoder(
                start_background_job(
                    operation_name="generate_boq_v2",
                    operation_func=boqv2.generate_boq_v2,
                    project_id=project_id,
                    write_workbook=write_workbook,
                    max_items_per_division=max_items_per_division,
                    text_row_limit=text_row_limit,
                    table_row_limit=table_row_limit,
                )
            ),
        )

    boq_output = boqv2.generate_boq_v2(
        project_id=project_id,
        write_workbook=write_workbook,
        max_items_per_division=max_items_per_division,
        text_row_limit=text_row_limit,
        table_row_limit=table_row_limit,
    )

    return JSONResponse(
        content=jsonable_encoder(
            {
                "status": "ok",
                "output": boq_output,
            }
        )
    )


@app.get(path="/generate_boq_takeoff", status_code=200)
def generate_boq_takeoff_api(
    project_id: str,
    stream: bool = True,
):
    if stream:
        return JSONResponse(
            status_code=202,
            content=jsonable_encoder(
                start_background_job(
                    operation_name="generate_boq_takeoff",
                    operation_func=takeoff.generate_boq_takeoff,
                    project_id=project_id,
                )
            ),
        )

    takeoff_output = takeoff.generate_boq_takeoff(project_id=project_id)
    return JSONResponse(
        content=jsonable_encoder(
            {
                "status": "ok",
                "output": takeoff_output,
            }
        )
    )


@app.get(path="/analyze_boq", status_code=200)
@app.get(path="/boq_analysis", status_code=200)
def analyze_boq_api(
    project_id: str,
    boq_run_id: Optional[str] = None,
    write_report: bool = True,
    max_items_for_prompt: int = 300,
    stream: bool = True,
):
    if stream:
        return JSONResponse(
            status_code=202,
            content=jsonable_encoder(
                start_background_job(
                    operation_name="analyze_boq",
                    operation_func=boqa.generate_boq_analysis,
                    project_id=project_id,
                    boq_run_id=boq_run_id,
                    write_report=write_report,
                    max_items_for_prompt=max_items_for_prompt,
                )
            ),
        )

    analysis_output = boqa.generate_boq_analysis(
        project_id=project_id,
        boq_run_id=boq_run_id,
        write_report=write_report,
        max_items_for_prompt=max_items_for_prompt,
    )

    return JSONResponse(
        content=jsonable_encoder(
            {
                "status": "ok",
                "output": analysis_output,
            }
        )
    )


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

# @app.get(path="/retrieve_chunks", status_code=200)
# def retrieve_chunks_api(
#     q: str,
#     top_k: int = 10,
#     mode: str = "hybrid",
#     corpus_zone: Optional[str] = None,
#     corpus_pack: Optional[str] = None,
#     document_id: Optional[str] = None,
#     max_chunk_chars: int = 3000
# ):
#     retrieval_output = ret.retrieve_chunks(
#         query_text=q,
#         top_k=top_k,
#         mode=mode,
#         corpus_zone=corpus_zone,
#         corpus_pack=corpus_pack,
#         document_id=document_id,
#         max_chunk_chars=max_chunk_chars
#     )

#     api_response = JSONResponse(
#         {
#             "status": "ok",
#             "output": retrieval_output
#         }
#     )

#     return api_response

@app.get(path="/generate_review_report", status_code=200)
def generate_ic_review_report_api(
    project_id: str,
    use_llm_summary: bool = True,
    write_audit: bool = True
):
    report_generation_output = rg.generate_construction_cost_estimation_report(
        project_id=project_id,
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

@app.get(path="/ai_cost_estimation_chatbot", status_code=200)
def ai_cost_estimation_chatbot_api(
    q: str = "Summarize the key construction cost risks and mitigants for this project.",
    project_name: Optional[str] = "synthetic_construction_cost_rag_pack",
    body_of_knowledge: Optional[str] = "All",
    client_data_pack: Optional[str] = None,
    top_k: int = 8,
    workstream: Optional[str] = None,
    corpus_pack_filter: Optional[str] = None,
    max_output_tokens: int = 8000
):
    try:
        chatbot_output = chatbot.answer_question(
            question=q,
            project_name=project_name,
            body_of_knowledge=body_of_knowledge,
            client_data_pack=client_data_pack,
            top_k=top_k,
            workstream=workstream,
            corpus_pack_filter=corpus_pack_filter,
            max_output_tokens=max_output_tokens,
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
