# File_name: boq_generation_v2.py
# Purpose: Generate a BOQ from client drawings/specifications/schedules even
# when no client-authored BOQ workbook is available.

import json
import re
import hashlib
from difflib import SequenceMatcher
from pathlib import Path
from datetime import datetime, timezone

from openai import OpenAI
from sqlalchemy import create_engine, text

from global_rag.scripts import config
import global_rag.scripts.retrieve_chunks as ret
import global_rag.scripts.boq_generation as boq_v1


BOQ_GENERATION_V2_VERSION = "boq_generation_v2_scope_takeoff"
REPORT_TYPE = "ai_boq_generation_v2"
CLASSIFICATION = "Confidential External"
WORKSTREAM = "ai_construction_cost_estimation_platform"

DIVISIONS = boq_v1.DIVISIONS
DIVISION_BY_CODE = boq_v1.DIVISION_BY_CODE

DIVISION_QUERY_HINTS = {
    "03": "concrete structural drawings foundation slab column beam rebar reinforcement concrete volume",
    "04": "masonry blockwork wall partition cement board ECFS partitions precast cladding wall types",
    "05": "metal steel balustrade railing ladder grating metal fabrication",
    "06": "wood plastic composite joinery millwork carpentry cabinets counters",
    "07": "waterproofing insulation thermal moisture protection roof wet area membrane",
    "08": "doors windows glazing openings frames ironmongery hardware door schedule window schedule",
    "09": "finishes plaster paint tile floor wall ceiling finish schedule room finish",
    "10": "specialties signage accessories toilet partitions lockers miscellaneous specialties",
    "12": "furnishings furniture counters blinds curtains fit out furnishings",
    "14": "elevator lift escalator conveying equipment schedule",
    "21": "fire suppression sprinkler fire fighting pump hydrant hose reel",
    "22": "plumbing sanitary water supply drainage fixtures piping",
    "23": "HVAC mechanical ventilation air conditioning duct pipe FCU AHU diffuser grille",
    "26": "electrical power lighting containment cable tray conduit distribution board",
    "27": "communications data telecom ELV structured cabling network",
    "28": "security access control CCTV fire alarm electronic safety",
}

PROJECT_FACT_QUERY_TERMS = [
    "residential units apartment count number of units unit schedule total units flats",
    "built-up area bua gfa gross floor area total built up area construction area 5214 5,214",
    "area schedule floor area schedule accommodation schedule room schedule apartment schedule",
    "door schedule window schedule opening schedule quantity count size type",
    "finish schedule room finish floor finish wall finish ceiling finish",
    "lift elevator schedule number of lifts lift core",
]

HIGH_VALUE_EVIDENCE_TERMS = [
    "quantity",
    "qty",
    "count",
    "number of",
    "no.",
    "schedule",
    "unit schedule",
    "apartment",
    "residential unit",
    "flat",
    "built-up",
    "built up",
    "bua",
    "gfa",
    "gross floor",
    "total area",
    "5214",
    "5,214",
    "door schedule",
    "window schedule",
    "opening schedule",
    "finish schedule",
    "room schedule",
    "lift",
    "elevator",
    "dimension",
    "area",
]

STANDARD_SCOPE_TEMPLATES = [
    ("03", "033000", "Cast-in-place concrete to structural elements", "m3"),
    ("03", "032000", "Reinforcement steel to concrete elements", "ton"),
    ("04", "042000", "Internal masonry/blockwork or cement-board partitions", "m2"),
    ("04", "044000", "External precast or masonry cladding panels", "m2"),
    ("05", "055000", "Metal fabrications, railings, ladders and grating", "Item"),
    ("06", "064000", "Architectural woodwork and joinery", "Item"),
    ("07", "071000", "Waterproofing to wet areas, roofs and exposed surfaces", "m2"),
    ("07", "072000", "Thermal insulation and moisture protection", "m2"),
    ("08", "081000", "Doors, frames, ironmongery and related openings", "No."),
    ("08", "084000", "Glazing, windows and curtain wall/opening assemblies", "m2"),
    ("09", "092000", "Plaster and wall/ceiling substrate finishes", "m2"),
    ("09", "096000", "Floor finishes", "m2"),
    ("09", "099000", "Painting and coating systems", "m2"),
    ("10", "101000", "Specialties, signs and accessories", "Item"),
    ("12", "123000", "Furnishings and built-in fittings", "Item"),
    ("14", "142000", "Elevators and conveying equipment", "L.S"),
    ("21", "211000", "Fire suppression systems", "Item"),
    ("22", "220000", "Plumbing systems and sanitary fixtures", "Item"),
    ("23", "230000", "HVAC systems, equipment, ducts and accessories", "Item"),
    ("26", "260000", "Electrical systems, lighting, power and containment", "Item"),
    ("27", "270000", "Communications and structured cabling systems", "Item"),
    ("28", "280000", "Electronic safety and security systems", "Item"),
]

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def clean_text(value):
    return boq_v1.clean_text(value)


def normalize_text(value):
    return boq_v1.normalize_text(value)


def normalize_division_code(value):
    value_text = clean_text(value)
    if value_text.isdigit():
        return value_text.zfill(2)

    number = boq_v1.safe_float(value_text)
    if number is not None and float(number).is_integer():
        return str(int(number)).zfill(2)

    return value_text


def make_run_id(project_id):
    run_label = re.sub(r"[^A-Za-z0-9_]+", "_", clean_text(project_id)).strip("_")
    if not run_label:
        run_label = "PROJECT"

    raw_value = f"{run_label}_{utc_now_iso()}_{BOQ_GENERATION_V2_VERSION}"
    digest = hashlib.sha256(raw_value.encode("utf-8")).hexdigest()[:12]
    return f"BOQV2_{run_label}_{digest}"


def config_pack(project_id):
    config_base = config.config_base()
    config_paths = config.config_paths(client_data=project_id)
    output_dir = Path(config_paths["output_dir"]) / "boq_generation_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "config_base": config_base,
        "config_paths": config_paths,
        "output_dir": output_dir,
    }


def get_source_reference(result):
    return {
        "chunk_id": result.get("chunk_id"),
        "document_id": result.get("document_id"),
        "corpus_zone": result.get("corpus_zone"),
        "corpus_pack": result.get("corpus_pack"),
        "section_heading": result.get("section_heading"),
        "page_start": result.get("page_start"),
        "page_end": result.get("page_end"),
        "source_reference": clean_text(result.get("source_reference")),
    }


def run_retrieval(query_text, corpus_zone=None, corpus_pack=None, top_k=12, max_chunk_chars=6000):
    output = ret.retrieve_chunks(
        query_text=query_text,
        top_k=top_k,
        mode="hybrid",
        corpus_zone=corpus_zone,
        corpus_pack=corpus_pack,
        max_chunk_chars=max_chunk_chars,
    )
    return output.get("results", [])


def fetch_client_text_sections(project_id, db_url, row_limit=600):
    engine = create_engine(db_url, pool_pre_ping=True)
    sql = text(
        """
        SELECT
            d.document_id,
            d.file_name,
            d.relative_path,
            et.extracted_text_id,
            et.section_heading,
            et.page_no AS page_start,
            NULL AS page_end,
            et.text_content
        FROM extracted_text et
        JOIN documents d
            ON d.document_id = et.document_id
        WHERE d.corpus_zone = 'client_data'
          AND d.workstream = :workstream
          AND d.corpus_pack = :project_id
          AND et.text_content IS NOT NULL
          AND LENGTH(TRIM(et.text_content)) > 0
        ORDER BY d.document_id, et.page_no NULLS LAST, et.extracted_text_id
        LIMIT :row_limit;
        """
    )

    with engine.begin() as conn:
        rows = conn.execute(
            sql,
            {
                "workstream": WORKSTREAM,
                "project_id": project_id,
                "row_limit": int(row_limit),
            },
        ).mappings().all()

    return [dict(row) for row in rows]


def fetch_client_table_rows(project_id, db_url, row_limit=2500):
    engine = create_engine(db_url, pool_pre_ping=True)
    sql = text(
        """
        SELECT
            d.document_id,
            d.file_name,
            d.relative_path,
            t.table_id,
            t.table_name,
            t.sheet_name,
            tr.row_number,
            tr.row_data
        FROM extracted_table_rows tr
        JOIN extracted_tables t
            ON t.table_id = tr.table_id
        JOIN documents d
            ON d.document_id = tr.document_id
        WHERE d.corpus_zone = 'client_data'
          AND d.workstream = :workstream
          AND d.corpus_pack = :project_id
        ORDER BY d.document_id, t.table_id, tr.row_number
        LIMIT :row_limit;
        """
    )

    with engine.begin() as conn:
        rows = conn.execute(
            sql,
            {
                "workstream": WORKSTREAM,
                "project_id": project_id,
                "row_limit": int(row_limit),
            },
        ).mappings().all()

    return [dict(row) for row in rows]


def fetch_cost_database_rows(db_url, row_limit=10000):
    engine = create_engine(db_url, pool_pre_ping=True)
    sql = text(
        """
        SELECT
            division_code,
            section_code,
            item_code,
            section_heading,
            description,
            unit,
            unit_rate_aed,
            source_workbook,
            sheet_name,
            row_number
        FROM cost_database
        WHERE unit_rate_aed IS NOT NULL
        LIMIT :row_limit;
        """
    )

    with engine.begin() as conn:
        rows = conn.execute(sql, {"row_limit": int(row_limit)}).mappings().all()

    normalized_rows = []
    for row in rows:
        rate = first_supported_number(row.get("unit_rate_aed"))
        if rate is None:
            continue

        division_code = normalize_division_code(row.get("division_code"))

        normalized_rows.append(
            {
                "division_code": division_code,
                "section_code": clean_text(row.get("section_code")),
                "item_code": clean_text(row.get("item_code")),
                "section_heading": clean_text(row.get("section_heading")),
                "description": clean_text(row.get("description")),
                "unit": canonical_unit(row.get("unit")),
                "unit_rate_aed": float(rate),
                "source_workbook": clean_text(row.get("source_workbook")),
                "sheet_name": clean_text(row.get("sheet_name")),
                "row_number": row.get("row_number"),
            }
        )

    return normalized_rows


def classify_evidence_record(record):
    text_blob = " ".join(
        clean_text(record.get(key))
        for key in ["file_name", "relative_path", "sheet_name", "table_name", "section_heading", "text_content"]
    )
    lowered = text_blob.lower()

    labels = []
    checks = [
        ("boq_or_cost", ["boq", "bill of quantities", "unit rate", "total amount", "cost loading"]),
        ("door_window_schedule", ["door schedule", "window schedule", "opening schedule", "ironmongery"]),
        ("finish_schedule", ["finish schedule", "room finish", "floor finish", "wall finish", "ceiling finish"]),
        ("wall_partition", ["partition", "blockwork", "masonry", "wall type", "ecfs", "cement board"]),
        ("structural", ["structural", "foundation", "slab", "beam", "column", "reinforcement", "rebar"]),
        ("mep", ["hvac", "plumbing", "electrical", "fire fighting", "sprinkler", "duct", "cable", "pipe"]),
        ("drawing_plan", ["plan", "elevation", "section", "detail", "drawing"]),
        ("specification", ["specification", "material", "method", "standard", "approved", "requirement"]),
    ]

    for label, terms in checks:
        if any(term in lowered for term in terms):
            labels.append(label)

    if not labels:
        labels.append("general_client_evidence")

    return labels


def row_data_to_text(row_data):
    row_dict = boq_v1.row_data_to_dict(row_data)
    values = [clean_text(value) for value in boq_v1.ordered_row_values(row_dict)]
    return " | ".join(value for value in values if value)


def evidence_priority_score(value):
    text_value = normalize_text(value)
    score = 0
    for term in HIGH_VALUE_EVIDENCE_TERMS:
        if normalize_text(term) in text_value:
            score += 1

    if re.search(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\s*(?:m2|m3|sqm|sq\.m|no\.|nos|units?)\b", text_value):
        score += 3

    return score


def sort_evidence_rows(rows, text_getter):
    indexed_rows = []
    for index, row in enumerate(rows):
        text_value = text_getter(row)
        indexed_rows.append((evidence_priority_score(text_value), index, row))

    indexed_rows.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in indexed_rows]


def compact_text_sections(text_sections, max_records=240, max_chars=56000):
    parts = []
    prioritized_rows = sort_evidence_rows(
        text_sections,
        lambda row: " ".join(
            [
                clean_text(row.get("file_name")),
                clean_text(row.get("relative_path")),
                clean_text(row.get("section_heading")),
                clean_text(row.get("text_content")),
            ]
        ),
    )

    for row in prioritized_rows[:max_records]:
        labels = classify_evidence_record(row)
        parts.append(
            " | ".join(
                [
                    f"document_id={row.get('document_id')}",
                    f"file={clean_text(row.get('file_name'))}",
                    f"path={clean_text(row.get('relative_path'))}",
                    f"section={clean_text(row.get('section_heading'))}",
                    f"pages={row.get('page_start')}-{row.get('page_end')}",
                    f"labels={','.join(labels)}",
                    clean_text(row.get("text_content"))[:1600],
                ]
            )
        )

    return "\n".join(parts)[:max_chars]


def compact_table_rows(table_rows, max_records=420, max_chars=72000):
    parts = []
    table_rows_with_text = []
    for row in table_rows:
        row_text = row_data_to_text(row.get("row_data"))
        if row_text:
            table_rows_with_text.append((row, row_text))

    table_rows_with_text.sort(
        key=lambda item: (
            -evidence_priority_score(
                " ".join(
                    [
                        clean_text(item[0].get("file_name")),
                        clean_text(item[0].get("sheet_name")),
                        clean_text(item[0].get("table_name")),
                        item[1],
                    ]
                )
            ),
            int(item[0].get("row_number") or 0),
        )
    )

    for row, row_text in table_rows_with_text[:max_records]:
        labels = classify_evidence_record(
            {
                **row,
                "text_content": row_text,
            }
        )
        parts.append(
            " | ".join(
                [
                    f"document_id={row.get('document_id')}",
                    f"file={clean_text(row.get('file_name'))}",
                    f"sheet={clean_text(row.get('sheet_name'))}",
                    f"table={clean_text(row.get('table_name'))}",
                    f"row={row.get('row_number')}",
                    f"labels={','.join(labels)}",
                    row_text[:1200],
                ]
            )
        )

    return "\n".join(parts)[:max_chars]


def collect_retrieval_evidence(project_id):
    evidence_packets = {}
    retrieval_plan = {
        "project_core_facts": {
            "query": " ".join(PROJECT_FACT_QUERY_TERMS),
            "corpus_zone": "client_data",
            "corpus_pack": project_id,
            "top_k": 50,
        },
        "client_scope": {
            "query": (
                "architectural structural drawings schedules quantities dimensions wall partition door window "
                "finish MEP electrical plumbing HVAC fire scope"
            ),
            "corpus_zone": "client_data",
            "corpus_pack": project_id,
            "top_k": 40,
        },
        "client_schedule_takeoff": {
            "query": (
                "door schedule window schedule finish schedule room schedule wall type schedule equipment schedule "
                "quantity unit size type count"
            ),
            "corpus_zone": "client_data",
            "corpus_pack": project_id,
            "top_k": 50,
        },
        "client_area_unit_schedule": {
            "query": (
                "residential units apartment count total units unit schedule built-up area BUA GFA gross floor area "
                "total built up area area schedule accommodation schedule"
            ),
            "corpus_zone": "client_data",
            "corpus_pack": project_id,
            "top_k": 50,
        },
        "client_opening_finish_schedule": {
            "query": (
                "door schedule window schedule opening schedule finish schedule room finish floor finish wall finish "
                "ceiling finish quantity count size type"
            ),
            "corpus_zone": "client_data",
            "corpus_pack": project_id,
            "top_k": 50,
        },
        "measurement_rules": {
            "query": (
                "bill of quantities quantity takeoff measurement rules wall door window finish concrete MEP "
                "NRM CESMM RICS method of measurement"
            ),
            "corpus_zone": "corpus_data",
            "corpus_pack": None,
            "top_k": 24,
        },
    }

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
                "results": results,
                "sources": [get_source_reference(item) for item in results],
                "evidence_blob": " ".join(clean_text(item.get("chunk_text")) for item in results),
            }
        except Exception as exc:
            evidence_packets[key] = {
                "status": "failed",
                "query": query_config["query"],
                "results": [],
                "sources": [],
                "evidence_blob": "",
                "error": f"{type(exc).__name__}: {str(exc)}",
            }

    return evidence_packets


def collect_division_retrieval_evidence(project_id, division_code, division_name):
    query_hint = DIVISION_QUERY_HINTS.get(division_code, division_name)
    query_text = (
        f"{division_name} {query_hint} quantity schedule count dimensions size type area "
        "client drawing client schedule measured quantity takeoff"
    )
    packet_key = f"division_{division_code}_targeted"

    try:
        results = run_retrieval(
            query_text=query_text,
            corpus_zone="client_data",
            corpus_pack=project_id,
            top_k=24,
        )
        return {
            packet_key: {
                "status": "ok",
                "query": query_text,
                "results": results,
                "sources": [get_source_reference(item) for item in results],
                "evidence_blob": " ".join(clean_text(item.get("chunk_text")) for item in results),
            }
        }
    except Exception as exc:
        return {
            packet_key: {
                "status": "failed",
                "query": query_text,
                "results": [],
                "sources": [],
                "evidence_blob": "",
                "error": f"{type(exc).__name__}: {str(exc)}",
            }
        }


def compact_retrieval_evidence(evidence_packets, max_chars=64000, max_results_per_packet=16, max_result_chars=1800):
    parts = []
    for packet_key, packet in evidence_packets.items():
        parts.append(f"\n[{packet_key}] status={packet.get('status')} query={packet.get('query')}")
        if packet.get("error"):
            parts.append(f"error={packet.get('error')}")

        for result in packet.get("results", [])[: int(max_results_per_packet)]:
            parts.append(
                " | ".join(
                    [
                        f"chunk_id={result.get('chunk_id')}",
                        f"document_id={result.get('document_id')}",
                        f"section={clean_text(result.get('section_heading'))}",
                        f"pages={result.get('page_start')}-{result.get('page_end')}",
                        f"zone={result.get('corpus_zone')}",
                        f"pack={result.get('corpus_pack')}",
                        clean_text(result.get("chunk_text"))[: int(max_result_chars)],
                    ]
                )
            )

    return "\n".join(parts)[:max_chars]


def build_project_evidence(project_id, config_base, text_row_limit=600, table_row_limit=2500):
    text_sections = []
    table_rows = []
    direct_boq_items = []
    cost_database_rows = []
    db_status = {
        "status": "not_attempted",
        "text_rows": 0,
        "table_rows": 0,
        "direct_boq_items": 0,
        "cost_database_rates": 0,
    }

    try:
        text_sections = fetch_client_text_sections(
            project_id=project_id,
            db_url=config_base["db_url"],
            row_limit=text_row_limit,
        )
        table_rows = fetch_client_table_rows(
            project_id=project_id,
            db_url=config_base["db_url"],
            row_limit=table_row_limit,
        )
        direct_rows = boq_v1.fetch_client_boq_table_rows(
            project_id=project_id,
            db_url=config_base["db_url"],
        )
        direct_boq_items = boq_v1.extract_boq_items_from_table_rows(direct_rows)
        cost_database_rows = fetch_cost_database_rows(
            db_url=config_base["db_url"],
        )
        db_status = {
            "status": "ok",
            "text_rows": len(text_sections),
            "table_rows": len(table_rows),
            "direct_boq_items": len(direct_boq_items),
            "cost_database_rates": len(cost_database_rows),
        }
    except Exception as exc:
        db_status = {
            "status": "failed",
            "text_rows": len(text_sections),
            "table_rows": len(table_rows),
            "direct_boq_items": len(direct_boq_items),
            "cost_database_rates": len(cost_database_rows),
            "error": f"{type(exc).__name__}: {str(exc)}",
        }

    return {
        "text_sections": text_sections,
        "table_rows": table_rows,
        "direct_boq_items": direct_boq_items,
        "cost_database_rows": cost_database_rows,
        "db_status": db_status,
        "text_prompt": compact_text_sections(text_sections),
        "table_prompt": compact_table_rows(table_rows),
    }


def extract_response_text(response):
    if hasattr(response, "output_text") and response.output_text:
        return clean_text(response.output_text)

    try:
        return clean_text(response.output[0].content[0].text)
    except Exception:
        return clean_text(str(response))


def parse_json_from_response_text(response_text):
    response_text = clean_text(response_text)
    if response_text.startswith("```"):
        response_text = re.sub(r"^```(?:json)?", "", response_text, flags=re.IGNORECASE).strip()
        response_text = re.sub(r"```$", "", response_text).strip()

    try:
        return json.loads(response_text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM response did not contain a JSON object.")

    return json.loads(match.group(0))


def template_items_for_division(division_code):
    items = []
    counter = 1
    for div, section, description, unit in STANDARD_SCOPE_TEMPLATES:
        if div != division_code:
            continue
        items.append(
            boq_v1.make_boq_item(
                division_code=division_code,
                section_code=section,
                item_code=f"V2-{counter:03d}",
                description=description,
                unit=unit,
                quantity=0,
                unit_rate=0,
                source="v2 standard scope template; no project-specific quantity/rate extracted",
                confidence="very_low_assumed_scope",
            )
        )
        counter += 1

    return items


def canonical_unit(unit):
    unit_text = clean_text(unit)
    lowered = unit_text.lower().replace(" ", "")

    if lowered in ["m2", "m²", "sqm", "sq.m", "sq.m."]:
        return "m2"
    if lowered in ["m3", "m³", "cum", "cu.m", "cu.m."]:
        return "m3"
    if lowered in ["lm", "l.m", "l.m.", "m", "linear metre", "linear meter"]:
        return "Lm"
    if lowered in ["no", "no.", "nr", "nos", "number"]:
        return "No."
    if lowered in ["ls", "l.s", "l.s.", "lump sum", "lumpsum"]:
        return "L.S"
    if lowered in ["kg", "kilogram"]:
        return "kg"
    if lowered in ["ton", "tonne", "t"]:
        return "ton"
    if lowered in ["item", "sum"]:
        return "Item"

    return unit_text or "Item"


def first_supported_number(value):
    number = boq_v1.safe_float(value)
    if number is not None:
        return number

    return None


def normalized_description_tokens(value):
    text_value = normalize_text(value)
    tokens = re.findall(r"[a-z0-9]+", text_value)
    stop_words = {
        "and", "or", "to", "the", "of", "for", "with", "including", "include",
        "complete", "all", "as", "shown", "specified", "required", "where",
        "work", "works", "supply", "install", "fix", "provide",
    }
    return {token for token in tokens if len(token) > 2 and token not in stop_words}


def description_similarity(left, right):
    left_text = normalize_text(left)
    right_text = normalize_text(right)

    if not left_text or not right_text:
        return 0.0

    left_tokens = normalized_description_tokens(left_text)
    right_tokens = normalized_description_tokens(right_text)
    token_score = 0.0
    if left_tokens and right_tokens:
        token_score = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    sequence_score = SequenceMatcher(None, left_text[:240], right_text[:240]).ratio()
    return (token_score * 0.65) + (sequence_score * 0.35)


def find_cost_database_rate(item, cost_database_rows):
    if not cost_database_rows:
        return None

    division_code = normalize_division_code(item.get("division_code"))

    unit = canonical_unit(item.get("unit"))
    description = clean_text(item.get("description"))
    section_code = clean_text(item.get("section_code"))

    candidates = []
    for row in cost_database_rows:
        if row.get("division_code") != division_code:
            continue

        unit_match = row.get("unit") == unit
        if not unit_match and unit not in ["Item", "L.S"] and row.get("unit") not in ["Item", "L.S"]:
            continue

        score = description_similarity(description, row.get("description"))
        if section_code and section_code == clean_text(row.get("section_code")):
            score += 0.12
        if unit_match:
            score += 0.18

        candidates.append((score, row))

    if not candidates:
        return None

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    best_score, best_row = candidates[0]
    if best_score < 0.22:
        return None

    return {
        "unit_rate_aed": float(best_row["unit_rate_aed"]),
        "match_score": round(float(best_score), 4),
        "matched_description": best_row.get("description"),
        "matched_unit": best_row.get("unit"),
        "matched_division_code": best_row.get("division_code"),
        "matched_section_code": best_row.get("section_code"),
        "matched_item_code": best_row.get("item_code"),
        "source": (
            f"cost_database: {best_row.get('source_workbook')}; "
            f"sheet={best_row.get('sheet_name')}; row={best_row.get('row_number')}"
        ),
    }


def compact_cost_database_for_division(cost_database_rows, division_code, max_rows=35):
    compact_rows = []
    for row in cost_database_rows:
        if row.get("division_code") != division_code:
            continue
        compact_rows.append(
            {
                "section_code": row.get("section_code"),
                "item_code": row.get("item_code"),
                "description": row.get("description"),
                "unit": row.get("unit"),
                "unit_rate_aed": row.get("unit_rate_aed"),
            }
        )
        if len(compact_rows) >= max_rows:
            break

    return compact_rows


def complete_numeric_estimates(items, cost_database_rows=None):
    completed_items = []
    cost_database_rows = cost_database_rows or []

    for raw_item in items:
        item = dict(raw_item)
        quantity = first_supported_number(item.get("quantity"))
        existing_confidence = clean_text(item.get("confidence"))
        estimate_notes = []

        if quantity is None or quantity == 0:
            quantity = 0
            estimate_notes.append(
                "Quantity set to 0 because client_data did not provide a directly auditable quantity."
            )

        cost_match = find_cost_database_rate(item, cost_database_rows)
        if cost_match:
            unit_rate = cost_match["unit_rate_aed"]
            item["cost_database_match"] = cost_match
            estimate_notes.append(
                "Unit rate selected from cost_database "
                f"(match_score={cost_match['match_score']}; "
                f"matched_item={cost_match.get('matched_item_code')}; "
                f"matched_description={cost_match.get('matched_description')}; "
                f"source={cost_match.get('source')}): AED {unit_rate}."
            )
        else:
            unit_rate = 0
            estimate_notes.append(
                "Unit rate set to 0 because no comparable cost_database rate was found."
            )

        item["quantity"] = float(quantity)
        item["unit_rate_aed"] = float(unit_rate)
        item["amount_aed"] = float(quantity) * float(unit_rate)

        if estimate_notes:
            existing_source = clean_text(item.get("source"))
            item["source"] = " | ".join(part for part in [existing_source, *estimate_notes] if part)
            if item.get("cost_database_match"):
                if existing_confidence in ["", "low_scope_assumption", "very_low_assumed_scope", "needs_estimator_review"]:
                    item["confidence"] = "estimated_with_cost_database_rate_requires_review"
                else:
                    item["confidence"] = f"{existing_confidence}_with_cost_database_rate"

            measurement_basis = clean_text(item.get("measurement_basis"))
            item["measurement_basis"] = " | ".join(
                part for part in [measurement_basis, "BOQ v2 uses client_data quantities only and cost_database unit rates only; missing values remain 0."] if part
            )

        completed_items.append(item)

    return completed_items


def compact_direct_items_for_division(direct_items, division_code, max_items=80):
    compact_items = []
    for item in direct_items:
        if clean_text(item.get("division_code")) != division_code:
            continue
        compact_items.append(
            {
                "section_code": item.get("section_code"),
                "item_code": item.get("item_code"),
                "description": item.get("description"),
                "unit": item.get("unit"),
                "quantity": item.get("quantity"),
                "confidence": item.get("confidence"),
                "source": item.get("source"),
            }
        )
        if len(compact_items) >= max_items:
            break

    return compact_items


def generate_llm_items_for_division(
    client,
    model,
    project_id,
    division_code,
    division_name,
    evidence,
    retrieval_prompt,
    max_items_per_division=50,
):
    query_hint = DIVISION_QUERY_HINTS.get(division_code, division_name)
    direct_items = compact_direct_items_for_division(evidence["direct_boq_items"], division_code)
    cost_database_items = compact_cost_database_for_division(
        evidence.get("cost_database_rows", []),
        division_code,
    )

    prompt = f"""
Generate BOQ line items for project {project_id}, Division {division_code} - {division_name}.

Objective:
- Produce a practical BOQ draft even when no client-authored BOQ exists.
- Use client data as project facts.
- Use corpus data only for measurement method, item templates, classification and assumptions.
- Populate quantity and unit_rate_aed with the best defensible numeric value available.
- Quantities must come only from client_data: directly extracted BOQ rows, schedules, visible counts, dimensions or calculations based on client evidence.
- If client_data does not support a quantity, set quantity to 0 and explain the gap in measurement_basis.
- Unit rates must come only from the provided cost_database rows for comparable items in the same division/unit.
- If no comparable cost_database rate is available, set unit_rate_aed to 0 and explain the gap in rate_basis.
- Do not estimate quantities or use generic corpus rate estimates.

Allowed confidence/source modes:
- high_client_boq_row: directly extracted from a client BOQ/schedule table with quantity.
- medium_client_schedule_calculated: calculated from client schedule/dimension evidence.
- low_scope_assumption: scope inferred from drawings/specifications, quantity/rate not proven.
- needs_estimator_review: item is likely required but evidence is incomplete.

Return JSON only with this schema:
{{
  "division_code": "{division_code}",
  "source_mode": "direct_boq_mode | schedule_takeoff_mode | drawing_scope_mode | insufficient_evidence",
  "assumptions": ["..."],
  "items": [
    {{
      "section_code": "string",
      "item_code": "string",
      "description": "string",
      "unit": "m2|m3|Lm|No.|kg|ton|L.S|Item",
      "quantity": 0,
      "unit_rate_aed": 0,
      "measurement_basis": "string",
      "rate_basis": "string",
      "source": "document/chunk/table references for supported values; evidence-gap note for unsupported values",
      "confidence": "one allowed confidence/source mode"
    }}
  ]
}}

Limit items to {int(max_items_per_division)}. Prefer specific schedule-derived rows over generic scope rows.
Every item must have numeric quantity and numeric unit_rate_aed values. Use 0 where client_data lacks quantity support or cost_database lacks a comparable unit rate.

Direct BOQ rows already extracted for this division, if any:
{json.dumps(direct_items, ensure_ascii=False, default=str)}

Cost database unit-rate rows for this division:
{json.dumps(cost_database_items, ensure_ascii=False, default=str)}

Client extracted text evidence:
{evidence["text_prompt"]}

Client extracted table/schedule evidence:
{evidence["table_prompt"]}

Retrieved client/corpus evidence:
{retrieval_prompt}

Division-specific query hint:
{query_hint}
"""

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a senior quantity surveyor generating auditable BOQ drafts. "
                    "Return strict JSON. Use client_data only for quantities and cost_database only for unit rates. "
                    "Set unsupported quantities or unit rates to 0 and explain the evidence gap in the basis fields."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_output_tokens=10000,
        store=False,
    )

    parsed = parse_json_from_response_text(extract_response_text(response))
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON root must be an object.")

    parsed_items = parsed.get("items", [])
    if not isinstance(parsed_items, list):
        raise ValueError("LLM response JSON items must be a list.")

    items = []
    for index, raw_item in enumerate(parsed_items[: int(max_items_per_division)], start=1):
        if not isinstance(raw_item, dict):
            continue

        description = clean_text(raw_item.get("description"))
        unit = clean_text(raw_item.get("unit"))
        if not description or not unit:
            continue

        section_code = clean_text(raw_item.get("section_code")) or f"{division_code}0000"
        item_code = clean_text(raw_item.get("item_code")) or f"V2-{index:03d}"
        measurement_basis = clean_text(raw_item.get("measurement_basis"))
        rate_basis = clean_text(raw_item.get("rate_basis"))
        source = clean_text(raw_item.get("source"))
        confidence = clean_text(raw_item.get("confidence")) or "needs_estimator_review"

        source_parts = [part for part in [source, measurement_basis, rate_basis] if part]
        item = boq_v1.make_boq_item(
            division_code=division_code,
            section_code=section_code,
            item_code=item_code,
            description=description,
            unit=unit,
            quantity=raw_item.get("quantity", 0),
            unit_rate=raw_item.get("unit_rate_aed", 0),
            source=" | ".join(source_parts) or "LLM v2 generated from RAG evidence",
            confidence=confidence,
        )
        item["source_mode"] = clean_text(parsed.get("source_mode")) or "drawing_scope_mode"
        item["measurement_basis"] = measurement_basis
        item["rate_basis"] = rate_basis
        items.append(item)

    return {
        "division_code": division_code,
        "source_mode": clean_text(parsed.get("source_mode")) or "drawing_scope_mode",
        "assumptions": parsed.get("assumptions", []) if isinstance(parsed.get("assumptions", []), list) else [],
        "items": items,
    }


def generate_scope_items_with_llm(project_id, config_base, evidence, retrieval_evidence, max_items_per_division=50):
    openai_api_key = config_base.get("openai_api_key")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for boq_generation_v2.")

    client = OpenAI(api_key=openai_api_key)
    model = config_base.get("llm_model", "gpt-4.1-mini")
    global_retrieval_prompt = compact_retrieval_evidence(retrieval_evidence)

    all_items = []
    division_status = []
    assumptions = []

    for division_code, division_name, _ in DIVISIONS:
        try:
            division_retrieval_evidence = collect_division_retrieval_evidence(
                project_id=project_id,
                division_code=division_code,
                division_name=division_name,
            )
            division_retrieval_packet = next(iter(division_retrieval_evidence.values()))
            division_retrieval_prompt = compact_retrieval_evidence(
                division_retrieval_evidence,
                max_chars=24000,
                max_results_per_packet=12,
                max_result_chars=1800,
            )
            retrieval_prompt = "\n".join(
                part for part in [global_retrieval_prompt, "\nDivision-targeted retrieval evidence:", division_retrieval_prompt] if part
            )

            output = generate_llm_items_for_division(
                client=client,
                model=model,
                project_id=project_id,
                division_code=division_code,
                division_name=division_name,
                evidence=evidence,
                retrieval_prompt=retrieval_prompt,
                max_items_per_division=max_items_per_division,
            )
            division_items = output["items"]
            if not division_items:
                division_items = template_items_for_division(division_code)

            all_items.extend(division_items)
            assumptions.extend(clean_text(item) for item in output.get("assumptions", []) if clean_text(item))
            division_status.append(
                {
                    "division_code": division_code,
                    "division_name": division_name,
                    "status": "ok",
                    "source_mode": output.get("source_mode"),
                    "items_generated": len(division_items),
                    "targeted_retrieval_status": division_retrieval_packet.get("status"),
                    "targeted_retrieval_results": len(division_retrieval_packet.get("results", [])),
                }
            )
        except Exception as exc:
            fallback_items = template_items_for_division(division_code)
            all_items.extend(fallback_items)
            division_status.append(
                {
                    "division_code": division_code,
                    "division_name": division_name,
                    "status": "failed_fallback_template_used",
                    "source_mode": "insufficient_evidence",
                    "items_generated": len(fallback_items),
                    "error": f"{type(exc).__name__}: {str(exc)}",
                }
            )

    return {
        "items": all_items,
        "division_status": division_status,
        "assumptions": assumptions,
    }


def build_generation_mode(evidence, items):
    direct_count = int(evidence.get("db_status", {}).get("direct_boq_items") or 0)
    priced_count = sum(
        1
        for item in items
        if isinstance(item.get("quantity"), (int, float))
        and isinstance(item.get("unit_rate_aed"), (int, float))
        and item.get("quantity") != 0
        and item.get("unit_rate_aed") != 0
    )

    if direct_count > 0:
        return "direct_boq_augmented_mode"
    if priced_count > 0:
        return "schedule_takeoff_or_cost_database_mode"
    if items:
        return "drawing_scope_mode"
    return "insufficient_evidence"


def write_v2_audit_sheet(output_path, generation_payload):
    # The core workbook is written by boq_generation.py. v2 stores richer audit
    # fields in JSON to avoid bloating the estimator-facing Excel tabs.
    return output_path


def fetch_boq_ready_takeoff_rows(project_id, db_url):
    """
    Return accepted measurements from the latest successful takeoff run.

    BOQ v2 deliberately does not read drawings, OCR text, extracted text,
    extracted tables or RAG chunks. Project quantities and their measurement
    bases enter this program exclusively through takeoff_layer.
    """
    engine = create_engine(db_url, pool_pre_ping=True)
    sql = text(
        """
        WITH latest_run AS (
            SELECT takeoff_run_id
            FROM takeoff_layer
            WHERE project_id = :project_id
              AND run_status = 'completed'
              AND validation_status = 'accepted'
              AND is_boq_ready = TRUE
            ORDER BY created_at DESC
            LIMIT 1
        )
        SELECT
            t.takeoff_id,
            t.takeoff_run_id,
            t.division_code,
            t.section_code,
            t.item_code,
            t.item_description,
            t.takeoff_category,
            t.quantity,
            t.uom,
            t.applicable_area,
            t.element_type,
            t.element_mark,
            t.level_name,
            t.zone_name,
            t.calculation_method,
            t.calculation_formula,
            t.calculation_inputs,
            t.calculation_logic,
            t.theoretical_basis,
            t.drawing_scale_text,
            t.drawing_scale_ratio,
            t.geometry_source,
            t.geometry_data,
            t.symbol_name,
            t.symbol_count,
            t.source_document_id,
            t.source_file_name,
            t.source_relative_path,
            t.source_page,
            t.source_sheet,
            t.source_table_id,
            t.source_row_numbers,
            t.source_reference,
            t.evidence_text,
            t.extraction_method,
            t.parser_name,
            t.model_name,
            t.confidence_score,
            t.confidence_threshold,
            t.confidence_basis
        FROM takeoff_layer t
        JOIN latest_run lr
          ON lr.takeoff_run_id = t.takeoff_run_id
        WHERE t.project_id = :project_id
          AND t.run_status = 'completed'
          AND t.validation_status = 'accepted'
          AND t.is_boq_ready = TRUE
          AND t.quantity > 0
          AND t.confidence_score >= t.confidence_threshold
        ORDER BY
            t.division_code,
            t.section_code NULLS LAST,
            t.item_code NULLS LAST,
            t.created_at,
            t.takeoff_id
        """
    )
    with engine.begin() as connection:
        rows = connection.execute(sql, {"project_id": project_id}).mappings().all()
    return [dict(row) for row in rows]


def takeoff_rows_to_boq_items(takeoff_rows):
    items = []
    counters = {}
    for row in takeoff_rows:
        division_code = normalize_division_code(row.get("division_code"))
        if division_code not in DIVISION_BY_CODE:
            continue
        counters[division_code] = counters.get(division_code, 0) + 1
        sequence = counters[division_code]
        section_code = clean_text(row.get("section_code")) or f"{division_code}0000"
        item_code = clean_text(row.get("item_code")) or f"TO-{division_code}-{sequence:04d}"
        quantity = first_supported_number(row.get("quantity"))
        if quantity is None or quantity <= 0:
            continue

        source_parts = [
            f"takeoff_id={row.get('takeoff_id')}",
            f"takeoff_run_id={row.get('takeoff_run_id')}",
            clean_text(row.get("source_reference")),
            f"calculation={clean_text(row.get('calculation_formula'))}",
            f"logic={clean_text(row.get('calculation_logic'))}",
            f"theoretical_basis={clean_text(row.get('theoretical_basis'))}",
            f"confidence={row.get('confidence_score')}",
        ]
        item = boq_v1.make_boq_item(
            division_code=division_code,
            section_code=section_code,
            item_code=item_code,
            description=clean_text(row.get("item_description")),
            unit=canonical_unit(row.get("uom")),
            quantity=quantity,
            unit_rate=0,
            source=" | ".join(part for part in source_parts if part),
            confidence="high_takeoff_layer_accepted",
        )
        item["source_mode"] = "takeoff_layer"
        item["takeoff_id"] = str(row.get("takeoff_id"))
        item["takeoff_run_id"] = clean_text(row.get("takeoff_run_id"))
        item["measurement_basis"] = clean_text(row.get("calculation_logic"))
        item["rate_basis"] = ""
        item["theoretical_basis"] = clean_text(row.get("theoretical_basis"))
        item["calculation_formula"] = clean_text(row.get("calculation_formula"))
        item["calculation_inputs"] = row.get("calculation_inputs") or {}
        item["applicable_area"] = clean_text(row.get("applicable_area"))
        item["confidence_score"] = float(row.get("confidence_score"))
        items.append(item)
    return items


def generate_boq_v2(
    project_id,
    write_workbook=True,
    max_items_per_division=50,
    text_row_limit=2000,
    table_row_limit=5000,
):
    pack = config_pack(project_id)
    config_base = pack["config_base"]
    run_id = make_run_id(project_id)

    # The existing API parameters remain accepted for front-end compatibility.
    # max_items_per_division/text_row_limit/table_row_limit are intentionally not
    # used: takeoff-layer validation and row readiness control BOQ eligibility.
    takeoff_rows = fetch_boq_ready_takeoff_rows(
        project_id=project_id,
        db_url=config_base["db_url"],
    )
    if not takeoff_rows:
        raise RuntimeError(
            f"No BOQ-ready takeoff rows found for project_id={project_id}. "
            "Call /generate_boq_takeoff first and review accepted takeoff_layer rows."
        )

    cost_database_rows = fetch_cost_database_rows(db_url=config_base["db_url"])
    takeoff_items = takeoff_rows_to_boq_items(takeoff_rows)
    items = complete_numeric_estimates(
        takeoff_items,
        cost_database_rows=cost_database_rows,
    )
    summary = boq_v1.summarize_items(items)
    generation_mode = "takeoff_layer_mode"
    takeoff_run_id = clean_text(takeoff_rows[0].get("takeoff_run_id"))

    quality_notes = [
        "BOQ v2 consumes project quantities only from accepted takeoff_layer rows.",
        "Drawing rendering, OCR, schedule parsing, dimension/scale recognition, symbol counting and CAD/BIM ingestion are performed upstream by boq_takeoff_layer.py.",
        "Only rows marked is_boq_ready with confidence_score at or above their confidence_threshold are included.",
        "Every quantity retains its calculation formula, inputs, logic, theoretical basis, source reference and confidence in takeoff_layer.",
        "Unit rates are populated only from the cost_database table; where no comparable cost_database rate is found, BOQ v2 leaves unit_rate_aed as 0.",
        "The legacy BOQ v2 document-text/RAG inference path is not used by this generation run.",
    ]

    output_files = {}
    if write_workbook:
        xlsx_path = pack["output_dir"] / f"{run_id}.xlsx"
        boq_v1.write_boq_workbook(
            output_path=xlsx_path,
            run_id=run_id,
            project_id=project_id,
            items=items,
            evidence_packets={},
            quality_notes=quality_notes,
        )
        write_v2_audit_sheet(xlsx_path, {})
        output_files["xlsx_path"] = str(xlsx_path)

    json_path = pack["output_dir"] / f"{run_id}.json"
    output_payload = {
        "run_id": run_id,
        "project_id": project_id,
        "report_type": REPORT_TYPE,
        "classification": CLASSIFICATION,
        "boq_generation_version": BOQ_GENERATION_V2_VERSION,
        "generated_at": utc_now_iso(),
        "generation_mode": generation_mode,
        "takeoff_run_id": takeoff_run_id,
        "takeoff_rows_consumed": len(takeoff_rows),
        "summary": summary,
        "items": items,
        "quality_notes": quality_notes,
        "db_status": {
            "status": "ok",
            "takeoff_rows": len(takeoff_rows),
            "cost_database_rates": len(cost_database_rows),
        },
        "division_generation_status": [],
        "evidence_status": {},
        "output_files": output_files,
    }
    json_path.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    output_files["json_path"] = str(json_path)

    return {
        "message": "BOQ v2 generation completed.",
        "status": "ok",
        "run_id": run_id,
        "project_id": project_id,
        "generation_mode": generation_mode,
        "takeoff_run_id": takeoff_run_id,
        "takeoff_rows_consumed": len(takeoff_rows),
        "summary": summary,
        "db_status": output_payload["db_status"],
        "division_generation_status": [],
        "evidence_status": {},
        "quality_notes": quality_notes,
        "output_files": output_files,
    }


if __name__ == "__main__":
    print(
        json.dumps(
            generate_boq_v2(
                project_id="ai_construction_cost_estimation_platform",
                write_workbook=True,
            ),
            indent=2,
            default=str,
        )
    )
