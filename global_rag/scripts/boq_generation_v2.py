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

BENCHMARK_RATE_AED = {
    "03": {"m2": 180, "m3": 850, "ton": 4200, "Item": 25000, "L.S": 50000},
    "04": {"m2": 190, "Lm": 120, "Item": 15000, "L.S": 25000},
    "05": {"m2": 450, "Lm": 380, "kg": 14, "ton": 12500, "No.": 2500, "Item": 30000, "L.S": 50000},
    "06": {"m2": 750, "Lm": 650, "No.": 3500, "Item": 35000, "L.S": 60000},
    "07": {"m2": 95, "Lm": 85, "Item": 15000, "L.S": 30000},
    "08": {"m2": 850, "No.": 2800, "Item": 30000, "L.S": 75000},
    "09": {"m2": 110, "Lm": 75, "No.": 1200, "Item": 20000, "L.S": 40000},
    "10": {"No.": 1200, "Item": 15000, "L.S": 30000},
    "12": {"No.": 2500, "Lm": 850, "m2": 900, "Item": 25000, "L.S": 50000},
    "14": {"No.": 180000, "Item": 180000, "L.S": 180000},
    "21": {"No.": 850, "Lm": 140, "Item": 50000, "L.S": 120000},
    "22": {"No.": 1200, "Lm": 180, "Item": 45000, "L.S": 100000},
    "23": {"No.": 2500, "Lm": 220, "m2": 150, "Item": 65000, "L.S": 150000},
    "26": {"No.": 1800, "Lm": 95, "Item": 75000, "L.S": 180000},
    "27": {"No.": 900, "Lm": 65, "Item": 35000, "L.S": 80000},
    "28": {"No.": 1400, "Lm": 80, "Item": 45000, "L.S": 100000},
}

PARAMETRIC_QUANTITY_ALLOWANCE = {
    "03": {"m2": 500, "m3": 150, "ton": 18, "Item": 1, "L.S": 1},
    "04": {"m2": 650, "Lm": 250, "Item": 1, "L.S": 1},
    "05": {"m2": 75, "Lm": 120, "kg": 1200, "ton": 3, "No.": 8, "Item": 1, "L.S": 1},
    "06": {"m2": 80, "Lm": 120, "No.": 10, "Item": 1, "L.S": 1},
    "07": {"m2": 450, "Lm": 180, "Item": 1, "L.S": 1},
    "08": {"m2": 120, "No.": 35, "Item": 1, "L.S": 1},
    "09": {"m2": 900, "Lm": 250, "No.": 30, "Item": 1, "L.S": 1},
    "10": {"No.": 20, "Item": 1, "L.S": 1},
    "12": {"No.": 12, "Lm": 80, "m2": 50, "Item": 1, "L.S": 1},
    "14": {"No.": 1, "Item": 1, "L.S": 1},
    "21": {"No.": 80, "Lm": 300, "Item": 1, "L.S": 1},
    "22": {"No.": 45, "Lm": 350, "Item": 1, "L.S": 1},
    "23": {"No.": 35, "Lm": 450, "m2": 300, "Item": 1, "L.S": 1},
    "26": {"No.": 120, "Lm": 700, "Item": 1, "L.S": 1},
    "27": {"No.": 80, "Lm": 500, "Item": 1, "L.S": 1},
    "28": {"No.": 60, "Lm": 350, "Item": 1, "L.S": 1},
}


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
            et.section_id,
            et.section_heading,
            et.page_start,
            et.page_end,
            et.text_content
        FROM extracted_text et
        JOIN documents d
            ON d.document_id = et.document_id
        WHERE d.corpus_zone = 'client_data'
          AND d.workstream = :workstream
          AND d.corpus_pack = :project_id
          AND et.text_content IS NOT NULL
          AND LENGTH(TRIM(et.text_content)) > 0
        ORDER BY d.document_id, et.page_start NULLS LAST, et.section_id
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


def compact_text_sections(text_sections, max_records=120, max_chars=28000):
    parts = []
    for row in text_sections[:max_records]:
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
                    clean_text(row.get("text_content"))[:1200],
                ]
            )
        )

    return "\n".join(parts)[:max_chars]


def compact_table_rows(table_rows, max_records=220, max_chars=36000):
    parts = []
    for row in table_rows[:max_records]:
        row_text = row_data_to_text(row.get("row_data"))
        if not row_text:
            continue

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
                    row_text[:900],
                ]
            )
        )

    return "\n".join(parts)[:max_chars]


def collect_retrieval_evidence(project_id):
    evidence_packets = {}
    retrieval_plan = {
        "client_scope": {
            "query": (
                "architectural structural drawings schedules quantities dimensions wall partition door window "
                "finish MEP electrical plumbing HVAC fire scope"
            ),
            "corpus_zone": "client_data",
            "corpus_pack": project_id,
            "top_k": 32,
        },
        "client_schedule_takeoff": {
            "query": (
                "door schedule window schedule finish schedule room schedule wall type schedule equipment schedule "
                "quantity unit size type count"
            ),
            "corpus_zone": "client_data",
            "corpus_pack": project_id,
            "top_k": 32,
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
        "rate_benchmarks": {
            "query": (
                "UAE construction cost benchmark unit rates concrete masonry partitions finishes doors windows "
                "MEP electrical plumbing HVAC price book"
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


def compact_retrieval_evidence(evidence_packets, max_chars=32000):
    parts = []
    for packet_key, packet in evidence_packets.items():
        parts.append(f"\n[{packet_key}] status={packet.get('status')} query={packet.get('query')}")
        if packet.get("error"):
            parts.append(f"error={packet.get('error')}")

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


def benchmark_rate_for_item(item):
    division_code = normalize_division_code(item.get("division_code"))
    unit = canonical_unit(item.get("unit"))
    division_rates = BENCHMARK_RATE_AED.get(division_code, {})

    if unit in division_rates:
        return float(division_rates[unit])

    if unit in ["Item", "L.S"]:
        return float(division_rates.get("Item") or division_rates.get("L.S") or 25000)

    return float(division_rates.get("Item") or 1000)


def quantity_allowance_for_item(item):
    division_code = normalize_division_code(item.get("division_code"))
    unit = canonical_unit(item.get("unit"))
    division_quantities = PARAMETRIC_QUANTITY_ALLOWANCE.get(division_code, {})

    if unit in division_quantities:
        return float(division_quantities[unit])

    if unit in ["Item", "L.S"]:
        return 1.0

    return 1.0


def complete_numeric_estimates(items, cost_database_rows=None):
    completed_items = []
    cost_database_rows = cost_database_rows or []

    for raw_item in items:
        item = dict(raw_item)
        quantity = first_supported_number(item.get("quantity"))
        unit_rate = first_supported_number(item.get("unit_rate_aed"))
        existing_confidence = clean_text(item.get("confidence"))
        estimate_notes = []

        if quantity is None or quantity == 0:
            quantity = quantity_allowance_for_item(item)
            estimate_notes.append(
                f"Quantity estimated by BOQ v2 parametric allowance for Division {item.get('division_code')} / unit {canonical_unit(item.get('unit'))}: {quantity}."
            )

        cost_match = find_cost_database_rate(item, cost_database_rows)
        can_replace_rate_with_cost_database = existing_confidence not in ["high_client_boq_row"]
        if cost_match and can_replace_rate_with_cost_database:
            unit_rate = cost_match["unit_rate_aed"]
            item["cost_database_match"] = cost_match
            estimate_notes.append(
                "Unit rate selected from cost_database "
                f"(match_score={cost_match['match_score']}; "
                f"matched_item={cost_match.get('matched_item_code')}; "
                f"matched_description={cost_match.get('matched_description')}; "
                f"source={cost_match.get('source')}): AED {unit_rate}."
            )
        elif unit_rate is None or unit_rate == 0:
            unit_rate = benchmark_rate_for_item(item)
            estimate_notes.append(
                f"Unit rate estimated by BOQ v2 benchmark fallback for Division {item.get('division_code')} / unit {canonical_unit(item.get('unit'))}: AED {unit_rate}."
            )

        item["quantity"] = float(quantity)
        item["unit_rate_aed"] = float(unit_rate)
        item["amount_aed"] = float(quantity) * float(unit_rate)

        if estimate_notes:
            existing_source = clean_text(item.get("source"))
            item["source"] = " | ".join(part for part in [existing_source, *estimate_notes] if part)
            if item.get("cost_database_match"):
                if existing_confidence in ["", "low_scope_assumption", "very_low_assumed_scope", "needs_estimator_review", "estimated_parametric_requires_review"]:
                    item["confidence"] = "estimated_with_cost_database_rate_requires_review"
                else:
                    item["confidence"] = f"{existing_confidence}_with_cost_database_rate"
            elif existing_confidence in ["", "low_scope_assumption", "very_low_assumed_scope", "needs_estimator_review"]:
                item["confidence"] = "estimated_parametric_requires_review"
            else:
                item["confidence"] = f"{existing_confidence}_with_parametric_estimate"

            measurement_basis = clean_text(item.get("measurement_basis"))
            item["measurement_basis"] = " | ".join(
                part for part in [measurement_basis, "BOQ v2 inserted numeric estimate because extracted evidence did not provide a directly auditable quantity/rate."] if part
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
                "unit_rate_aed": item.get("unit_rate_aed"),
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
- Use corpus data only for measurement method, item templates, benchmark rate context and assumptions.
- Populate quantity and unit_rate_aed with the best defensible numeric value available.
- Use directly extracted quantities/rates first.
- For unit rates, prefer the provided cost_database rates for similar items in the same division/unit before using generic corpus benchmarks.
- If exact measured quantities are unavailable, provide a conservative estimate from schedules, visible counts, dimensions, corpus measurement guidance or a clearly stated parametric allowance.
- If project-specific rates are unavailable, provide a benchmark unit rate estimate from corpus/rate context and state that it requires estimator validation.
- Do not present estimated values as exact measured quantities. The measurement_basis and rate_basis fields must say whether each number is direct, calculated, benchmarked or assumed.

Allowed confidence/source modes:
- high_client_boq_row: directly extracted from a client BOQ/schedule table with quantity and rate.
- medium_client_schedule_calculated: calculated from client schedule/dimension evidence.
- medium_benchmark_rate: client quantity with rate inferred from corpus/rate benchmark.
- low_scope_assumption: scope inferred from drawings/specifications, quantity/rate not proven.
- estimated_parametric_requires_review: numeric quantity/rate estimated from parametric allowances because evidence is incomplete.
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
      "source": "document/chunk/table references or explicit assumption",
      "confidence": "one allowed confidence/source mode"
    }}
  ]
}}

Limit items to {int(max_items_per_division)}. Prefer specific schedule-derived rows over generic scope rows.
Every item must have numeric quantity and numeric unit_rate_aed values. Use 0 only if an item is genuinely non-priced or excluded; otherwise estimate and label it.

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
                    "Return strict JSON. Use cost database rates for comparable items where available. "
                    "Estimated quantities/rates must be explicitly labelled as estimates in the basis fields."
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
    retrieval_prompt = compact_retrieval_evidence(retrieval_evidence)

    all_items = []
    division_status = []
    assumptions = []

    for division_code, division_name, _ in DIVISIONS:
        try:
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
        return "schedule_takeoff_or_benchmark_mode"
    if items:
        return "drawing_scope_mode"
    return "insufficient_evidence"


def write_v2_audit_sheet(output_path, generation_payload):
    # The core workbook is written by boq_generation.py. v2 stores richer audit
    # fields in JSON to avoid bloating the estimator-facing Excel tabs.
    return output_path


def generate_boq_v2(
    project_id,
    write_workbook=True,
    max_items_per_division=50,
    text_row_limit=600,
    table_row_limit=2500,
):
    pack = config_pack(project_id)
    config_base = pack["config_base"]
    run_id = make_run_id(project_id)

    evidence = build_project_evidence(
        project_id=project_id,
        config_base=config_base,
        text_row_limit=text_row_limit,
        table_row_limit=table_row_limit,
    )
    retrieval_evidence = collect_retrieval_evidence(project_id)
    llm_output = generate_scope_items_with_llm(
        project_id=project_id,
        config_base=config_base,
        evidence=evidence,
        retrieval_evidence=retrieval_evidence,
        max_items_per_division=max_items_per_division,
    )

    items = complete_numeric_estimates(
        boq_v1.dedupe_boq_items(llm_output["items"]),
        cost_database_rows=evidence.get("cost_database_rows", []),
    )
    summary = boq_v1.summarize_items(items)
    generation_mode = build_generation_mode(evidence, items)

    quality_notes = [
        "BOQ v2 can generate a scope/takeoff draft without a client-authored BOQ file.",
        "Client documents remain the only source for project-specific facts. Corpus evidence is used for measurement rules, item templates, classification and rate benchmark context.",
        "Where direct quantities/rates are unavailable, BOQ v2 now inserts numeric parametric or benchmark estimates instead of zero values.",
        "Unit rates are selected from the cost_database table before the generic benchmark fallback is used.",
        "Rows marked estimated_parametric_requires_review, low_scope_assumption, very_low_assumed_scope or needs_estimator_review require estimator validation before commercial use.",
        "This is not a replacement for a full CAD/BIM quantity takeoff where drawing geometry, scale and dimensions are unavailable in extracted text/tables.",
    ]
    for assumption in llm_output["assumptions"][:30]:
        quality_notes.append(f"LLM assumption: {assumption}")

    output_files = {}
    if write_workbook:
        xlsx_path = pack["output_dir"] / f"{run_id}.xlsx"
        boq_v1.write_boq_workbook(
            output_path=xlsx_path,
            run_id=run_id,
            project_id=project_id,
            items=items,
            evidence_packets=retrieval_evidence,
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
        "summary": summary,
        "items": items,
        "quality_notes": quality_notes,
        "db_status": evidence["db_status"],
        "division_generation_status": llm_output["division_status"],
        "evidence_status": {
            key: {
                "status": packet.get("status"),
                "query": packet.get("query"),
                "results_returned": len(packet.get("results", [])),
                "error": packet.get("error"),
            }
            for key, packet in retrieval_evidence.items()
        },
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
        "summary": summary,
        "db_status": evidence["db_status"],
        "division_generation_status": llm_output["division_status"],
        "evidence_status": output_payload["evidence_status"],
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
