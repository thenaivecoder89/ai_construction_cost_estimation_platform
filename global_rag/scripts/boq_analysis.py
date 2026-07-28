# File_name: boq_analysis.py
# Purpose: Analyze a generated BOQ against corpus guidance and RAG evidence
# using the configured LLM. This program does not generate or alter BOQ rows.

import json
import re
import hashlib
from pathlib import Path
from datetime import datetime, timezone

from openai import OpenAI

from global_rag.scripts import config
import global_rag.scripts.retrieve_chunks as ret


BOQ_ANALYSIS_VERSION = "boq_analysis_v1"
REPORT_TYPE = "ai_boq_analysis"
CLASSIFICATION = "Confidential External"


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def clean_text(value):
    if value is None:
        return ""

    value = str(value).replace("\x00", " ")
    value = value.replace("\r", " ").replace("\n", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def make_run_id(project_id):
    run_label = re.sub(r"[^A-Za-z0-9_]+", "_", clean_text(project_id)).strip("_")
    if not run_label:
        run_label = "PROJECT"

    raw_value = f"{run_label}_{utc_now_iso()}_{BOQ_ANALYSIS_VERSION}"
    digest = hashlib.sha256(raw_value.encode("utf-8")).hexdigest()[:12]
    return f"BOQA_{run_label}_{digest}"


def get_config_pack(project_id):
    config_base = config.config_base()
    config_paths = config.config_paths(client_data=project_id)

    boq_output_dir = Path(config_paths["output_dir"]) / "boq_generation"
    analysis_output_dir = Path(config_paths["output_dir"]) / "boq_analysis"
    analysis_output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "config_base": config_base,
        "config_paths": config_paths,
        "boq_output_dir": boq_output_dir,
        "analysis_output_dir": analysis_output_dir,
    }


def find_boq_json_path(boq_output_dir, project_id, boq_run_id=None):
    boq_output_dir = Path(boq_output_dir)

    if boq_run_id:
        candidate = boq_output_dir / f"{boq_run_id}.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"BOQ JSON output not found for boq_run_id={boq_run_id}: {candidate}")

    pattern = f"BOQ_{re.sub(r'[^A-Za-z0-9_]+', '_', project_id).strip('_')}_*.json"
    candidates = sorted(
        boq_output_dir.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No BOQ JSON outputs found for project_id={project_id} in {boq_output_dir}."
        )

    return candidates[0]


def load_boq_output(boq_json_path):
    return json.loads(Path(boq_json_path).read_text(encoding="utf-8"))


def run_retrieval(query_text, corpus_zone=None, corpus_pack=None, top_k=12):
    output = ret.retrieve_chunks(
        query_text=query_text,
        top_k=top_k,
        mode="hybrid",
        corpus_zone=corpus_zone,
        corpus_pack=corpus_pack,
        max_chunk_chars=8000,
    )
    return output.get("results", [])


def collect_analysis_evidence(project_id):
    retrieval_plan = {
        "measurement_rules": {
            "query": (
                "bill of quantities measurement rules NRM CESMM RICS quantity surveying "
                "method of measurement construction work sections preliminaries exclusions"
            ),
            "corpus_zone": "corpus_data",
            "corpus_pack": None,
            "top_k": 16,
        },
        "cost_benchmarking": {
            "query": (
                "UAE construction cost benchmark concrete masonry finishes MEP electrical "
                "plumbing HVAC price book cost guide unit rates"
            ),
            "corpus_zone": "corpus_data",
            "corpus_pack": None,
            "top_k": 16,
        },
        "client_scope_evidence": {
            "query": (
                "Building 1 drawings schedules BOQ quantities rates wall partitions structure "
                "architecture MEP electrical plumbing HVAC scope"
            ),
            "corpus_zone": "client_data",
            "corpus_pack": project_id,
            "top_k": 16,
        },
    }

    evidence_packets = {}
    for key, query_config in retrieval_plan.items():
        try:
            results = run_retrieval(
                query_text=query_config["query"],
                corpus_zone=query_config.get("corpus_zone"),
                corpus_pack=query_config.get("corpus_pack"),
                top_k=query_config.get("top_k", 12),
            )
            evidence_packets[key] = {
                "status": "ok",
                "query": query_config["query"],
                "results_returned": len(results),
                "results": results,
            }
        except Exception as exc:
            evidence_packets[key] = {
                "status": "failed",
                "query": query_config["query"],
                "results_returned": 0,
                "results": [],
                "error": f"{type(exc).__name__}: {str(exc)}",
            }

    return evidence_packets


def compact_boq_for_prompt(boq_output, max_items=300):
    items = boq_output.get("items", [])
    compact_items = []

    for item in items[:max_items]:
        compact_items.append(
            {
                "division_code": item.get("division_code"),
                "division_name": item.get("division_name"),
                "section_code": item.get("section_code"),
                "item_code": item.get("item_code"),
                "description": item.get("description"),
                "unit": item.get("unit"),
                "quantity": item.get("quantity"),
                "unit_rate_aed": item.get("unit_rate_aed"),
                "amount_aed": item.get("amount_aed"),
                "confidence": item.get("confidence"),
            }
        )

    return {
        "run_id": boq_output.get("run_id"),
        "project_id": boq_output.get("project_id"),
        "summary": boq_output.get("summary", {}),
        "direct_table_status": boq_output.get("direct_table_status", {}),
        "quality_notes": boq_output.get("quality_notes", []),
        "items_sample": compact_items,
        "items_sample_count": len(compact_items),
        "total_items": len(items),
    }


def compact_evidence_for_prompt(evidence_packets, max_chars=30000):
    parts = []
    for packet_key, packet in evidence_packets.items():
        parts.append(f"\n[{packet_key}] status={packet.get('status')} results={packet.get('results_returned')}")
        if packet.get("error"):
            parts.append(f"Error: {packet.get('error')}")

        for result in packet.get("results", [])[:8]:
            parts.append(
                " | ".join(
                    [
                        f"chunk_id={result.get('chunk_id')}",
                        f"document_id={result.get('document_id')}",
                        f"zone={result.get('corpus_zone')}",
                        f"pack={result.get('corpus_pack')}",
                        clean_text(result.get("chunk_text"))[:1200],
                    ]
                )
            )

    return "\n".join(parts)[:max_chars]


def extract_response_text(response):
    if hasattr(response, "output_text") and response.output_text:
        return clean_text(response.output_text)

    try:
        return clean_text(response.output[0].content[0].text)
    except Exception:
        return clean_text(str(response))


def generate_llm_analysis(config_base, project_id, boq_output, evidence_packets, max_items_for_prompt=300):
    openai_api_key = config_base.get("openai_api_key")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for boq_analysis.")

    client = OpenAI(api_key=openai_api_key)
    model = config_base.get("llm_model", "gpt-4.1-mini")

    boq_prompt_data = compact_boq_for_prompt(
        boq_output=boq_output,
        max_items=max_items_for_prompt,
    )
    evidence_prompt_text = compact_evidence_for_prompt(evidence_packets)

    prompt = f"""
You are a senior quantity surveying reviewer. Analyze the generated BOQ for {project_id}.

Use the BOQ output as the primary generated commercial document. Use corpus evidence to assess
measurement discipline, classification, completeness, pricing support and estimator-review gaps.
Use client evidence to check whether generated BOQ scope appears grounded in source documents.

Do not alter or regenerate BOQ rows. Do not invent missing quantities or rates.

Return a clear professional analysis report with these sections:
1. Executive view
2. BOQ completeness assessment
3. Division-level observations
4. Quantity and rate reconciliation observations
5. Measurement and classification observations against corpus guidance
6. Critical gaps and estimator actions
7. Source/evidence limitations

Generated BOQ data:
{json.dumps(boq_prompt_data, ensure_ascii=False, default=str)}

Retrieved RAG evidence:
{evidence_prompt_text}
"""

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You produce evidence-grounded quantity surveying analysis. "
                    "Be explicit about limitations and never invent quantities, rates or source support."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_output_tokens=12000,
        store=False,
    )

    return extract_response_text(response)


def generate_boq_analysis(
    project_id,
    boq_run_id=None,
    write_report=True,
    max_items_for_prompt=300,
):
    pack = get_config_pack(project_id)
    run_id = make_run_id(project_id)

    boq_json_path = find_boq_json_path(
        boq_output_dir=pack["boq_output_dir"],
        project_id=project_id,
        boq_run_id=boq_run_id,
    )
    boq_output = load_boq_output(boq_json_path)
    evidence_packets = collect_analysis_evidence(project_id)

    analysis_markdown = generate_llm_analysis(
        config_base=pack["config_base"],
        project_id=project_id,
        boq_output=boq_output,
        evidence_packets=evidence_packets,
        max_items_for_prompt=max_items_for_prompt,
    )

    output_files = {}
    analysis_payload = {
        "run_id": run_id,
        "project_id": project_id,
        "boq_run_id": boq_output.get("run_id"),
        "boq_json_path": str(boq_json_path),
        "report_type": REPORT_TYPE,
        "classification": CLASSIFICATION,
        "boq_analysis_version": BOQ_ANALYSIS_VERSION,
        "generated_at": utc_now_iso(),
        "boq_summary": boq_output.get("summary", {}),
        "analysis_markdown": analysis_markdown,
        "evidence_status": {
            key: {
                "status": packet.get("status"),
                "query": packet.get("query"),
                "results_returned": packet.get("results_returned"),
                "error": packet.get("error"),
            }
            for key, packet in evidence_packets.items()
        },
    }

    if write_report:
        json_path = pack["analysis_output_dir"] / f"{run_id}.json"
        markdown_path = pack["analysis_output_dir"] / f"{run_id}.md"
        json_path.write_text(json.dumps(analysis_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        markdown_path.write_text(analysis_markdown, encoding="utf-8")
        output_files["json_path"] = str(json_path)
        output_files["markdown_path"] = str(markdown_path)

    return {
        "message": "BOQ analysis completed.",
        "status": "ok",
        "run_id": run_id,
        "project_id": project_id,
        "boq_run_id": boq_output.get("run_id"),
        "boq_json_path": str(boq_json_path),
        "output_files": output_files,
        "boq_summary": boq_output.get("summary", {}),
        "analysis_markdown": analysis_markdown,
        "evidence_status": analysis_payload["evidence_status"],
    }


if __name__ == "__main__":
    print(
        json.dumps(
            generate_boq_analysis(
                project_id="ai_construction_cost_estimation_platform",
                boq_run_id=None,
                write_report=True,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
