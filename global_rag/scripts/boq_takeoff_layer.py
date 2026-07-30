"""
Audited construction takeoff layer.

This module is intentionally separate from BOQ generation. It renders drawings,
uses OpenAI vision to detect schedules/dimensions/symbols, validates returned
calculations, and persists both accepted and rejected observations. BOQ v2 may
consume only rows marked is_boq_ready = TRUE.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import math
import operator
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import fitz
from openai import OpenAI
from sqlalchemy import create_engine, inspect, text

from global_rag.scripts import config


WORKSTREAM = "ai_construction_cost_estimation_platform"
TAKEOFF_VERSION = "takeoff_layer_v1"
DEFAULT_CONFIDENCE_THRESHOLD = 0.80
DEFAULT_RENDER_DPI = 220
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
CAD_EXTENSIONS = {".dxf"}
BIM_EXTENSIONS = {".ifc"}
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS | CAD_EXTENSIONS | BIM_EXTENSIONS

DIVISION_NAMES = {
    "03": "Concrete",
    "04": "Masonry",
    "05": "Metals",
    "06": "Wood, Plastics and Composites",
    "07": "Thermal and Moisture Protection",
    "08": "Openings",
    "09": "Finishes",
    "10": "Specialties",
    "12": "Furnishings",
    "14": "Conveying Equipment",
    "21": "Fire Suppression",
    "22": "Plumbing",
    "23": "HVAC",
    "26": "Electrical",
    "27": "Communications",
    "28": "Electronic Safety and Security",
}

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def clean_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\x00", " ")).strip()


def normalize_division_code(value):
    value = clean_text(value)
    match = re.search(r"\b(\d{1,2})\b", value)
    return match.group(1).zfill(2) if match else ""


def canonical_uom(value):
    normalized = clean_text(value).lower().replace(" ", "")
    aliases = {
        "m²": "m2", "sqm": "m2", "sq.m": "m2", "m2": "m2",
        "m³": "m3", "cum": "m3", "cu.m": "m3", "m3": "m3",
        "lm": "Lm", "l.m": "Lm", "linearmeter": "Lm", "linearmetre": "Lm",
        "no": "No.", "no.": "No.", "nos": "No.", "number": "No.",
        "kg": "kg", "kilogram": "kg",
        "t": "ton", "ton": "ton", "tonne": "ton",
        "ls": "L.S", "l.s": "L.S", "lumpsum": "L.S",
        "item": "Item",
    }
    return aliases.get(normalized, clean_text(value))


def make_run_id(project_id):
    safe_project = re.sub(r"[^A-Za-z0-9_]+", "_", clean_text(project_id)).strip("_")
    digest = hashlib.sha256(
        f"{safe_project}|{utc_now_iso()}|{TAKEOFF_VERSION}".encode("utf-8")
    ).hexdigest()[:12]
    return f"TAKEOFF_{safe_project or 'PROJECT'}_{digest}"


def extract_response_text(response):
    if getattr(response, "output_text", None):
        return response.output_text
    try:
        return response.output[0].content[0].text
    except Exception:
        return str(response)


def parse_json_response(value):
    value = str(value or "").strip()
    value = re.sub(r"^```(?:json)?", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"```$", "", value).strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.DOTALL)
        if not match:
            raise ValueError("OpenAI takeoff response did not contain a JSON object.")
        return json.loads(match.group(0))


def evaluate_formula(expression, inputs):
    """Evaluate an arithmetic expression containing only named numeric inputs."""
    if not clean_text(expression):
        raise ValueError("calculation_formula is required.")
    numeric_inputs = {}
    for key, value in (inputs or {}).items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key)):
            raise ValueError(f"Unsafe calculation input name: {key}")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Calculation input must be numeric: {key}")
        numeric_inputs[str(key)] = float(value)

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in numeric_inputs:
            return numeric_inputs[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPERATORS:
            return SAFE_OPERATORS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPERATORS:
            return SAFE_OPERATORS[type(node.op)](visit(node.operand))
        raise ValueError("Formula contains an unsupported operation or unknown input.")

    result = float(visit(ast.parse(expression, mode="eval")))
    if not math.isfinite(result):
        raise ValueError("Formula result is not finite.")
    return result


def validate_observation(raw, confidence_threshold):
    errors = []
    division_code = normalize_division_code(raw.get("division_code"))
    description = clean_text(raw.get("item_description"))
    uom = canonical_uom(raw.get("uom"))
    formula = clean_text(raw.get("calculation_formula"))
    inputs = raw.get("calculation_inputs") or {}
    confidence = raw.get("confidence_score")
    reported_quantity = raw.get("quantity")

    if division_code not in DIVISION_NAMES:
        errors.append("division_code is missing or unsupported")
    if not description:
        errors.append("item_description is required")
    if not uom:
        errors.append("uom is required")
    if not clean_text(raw.get("calculation_logic")):
        errors.append("calculation_logic is required")
    if not clean_text(raw.get("theoretical_basis")):
        errors.append("theoretical_basis is required")
    if not clean_text(raw.get("source_reference")):
        errors.append("source_reference is required")
    if not clean_text(raw.get("confidence_basis")):
        errors.append("confidence_basis is required")

    try:
        confidence = float(confidence)
        if not 0 <= confidence <= 1:
            raise ValueError
    except (TypeError, ValueError):
        confidence = 0.0
        errors.append("confidence_score must be between 0 and 1")

    calculated_quantity = None
    try:
        calculated_quantity = evaluate_formula(formula, inputs)
        if calculated_quantity <= 0:
            errors.append("calculated quantity must be greater than zero")
    except Exception as exc:
        errors.append(f"formula validation failed: {exc}")

    try:
        reported_quantity = float(reported_quantity)
        if calculated_quantity is not None and not math.isclose(
            reported_quantity, calculated_quantity, rel_tol=0.005, abs_tol=0.001
        ):
            errors.append(
                f"reported quantity {reported_quantity} does not match validated formula result "
                f"{calculated_quantity}"
            )
    except (TypeError, ValueError):
        errors.append("quantity must be numeric")

    scale_ratio = raw.get("drawing_scale_ratio")
    if scale_ratio not in [None, ""]:
        try:
            scale_ratio = float(scale_ratio)
            if scale_ratio <= 0:
                raise ValueError
        except (TypeError, ValueError):
            scale_ratio = None
            errors.append("drawing_scale_ratio must be positive")

    calculation_method = clean_text(raw.get("calculation_method")).lower()
    geometry_source = clean_text(raw.get("geometry_source")).lower()
    geometry_data = raw.get("geometry_data") or {}
    if calculation_method == "scaled_geometry" or geometry_source == "scaled_pdf":
        required_calibration_fields = {
            "measured_pixels",
            "calibration_pixels",
            "calibration_length_m",
        }
        if scale_ratio is None:
            errors.append("scaled geometry requires a legible drawing_scale_ratio")
        if not required_calibration_fields.issubset(geometry_data):
            errors.append(
                "scaled geometry requires measured_pixels, calibration_pixels and "
                "calibration_length_m in geometry_data"
            )
        if "do not scale" in clean_text(raw.get("evidence_text")).lower():
            errors.append("scaled geometry is prohibited by the drawing note")

    if calculation_method == "symbol_count":
        try:
            symbol_count = int(raw.get("symbol_count"))
            if symbol_count <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append("symbol_count method requires a positive integer symbol_count")
        if not clean_text(raw.get("symbol_name")):
            errors.append("symbol_count method requires symbol_name")

    if errors:
        validation_status = "rejected_validation"
        rejection_reason = "; ".join(errors)
        is_boq_ready = False
    elif confidence < confidence_threshold:
        validation_status = "below_confidence_threshold"
        rejection_reason = (
            f"confidence_score {confidence:.5f} is below threshold "
            f"{confidence_threshold:.5f}"
        )
        is_boq_ready = False
    else:
        validation_status = "accepted"
        rejection_reason = None
        is_boq_ready = True

    return {
        **raw,
        "division_code": division_code,
        "item_description": description,
        "uom": uom,
        "quantity": calculated_quantity if calculated_quantity is not None else reported_quantity,
        "confidence_score": confidence,
        "confidence_threshold": confidence_threshold,
        "drawing_scale_ratio": scale_ratio,
        "validation_status": validation_status,
        "is_boq_ready": is_boq_ready,
        "rejection_reason": rejection_reason,
    }


def takeoff_prompt(project_id, source_file_name, page_no):
    divisions = ", ".join(f"{code} {name}" for code, name in DIVISION_NAMES.items())
    return f"""
You are a senior quantity surveyor performing an auditable takeoff from one rendered
construction drawing or schedule page for project {project_id}.

Page source: {source_file_name}, page {page_no}.

Perform OCR and visual table detection. Detect structured door, window, opening and
finish schedules; written dimensions and drawing scales; repeated symbols/marks and
counts; areas, lengths, volumes, item counts and applicable locations.

Rules:
- Use only visible evidence on this page. Never invent or estimate an unreadable value.
- Written dimensions override scaled measurements.
- Do not scale a drawing explicitly marked "do not scale".
- A scaled measurement is allowed only when a legible scale is present and calibration
  is internally consistent. Explain the calibration in calculation_logic.
- Use net or gross measurement in accordance with the stated drawing/schedule and name
  the applicable measurement principle in theoretical_basis.
- Counts must identify the symbol/mark counted and the counted region.
- Each quantity must be reproducible using calculation_formula and calculation_inputs.
- calculation_formula must use only arithmetic operators and variable names supplied in
  calculation_inputs, for example length_m * width_m, count, or
  wall_length_m * wall_height_m - openings_m2.
- Use SI units. Convert millimetres to metres explicitly in the inputs/formula.
- Confidence above 0.80 requires legible source values, unambiguous scope/location, a
  reproducible calculation, and no conflict with drawing notes.
- Include schedule classifications even when no accepted quantity is possible only if a
  formula and numeric quantity can be produced. Do not emit qualitative scope rows.
- Allowed divisions: {divisions}.

Return JSON only:
{{
  "page_classification": "schedule|plan|section|elevation|detail|other",
  "ocr_text": "concise transcription of relevant visible text",
  "detected_scale": {{
    "scale_text": "1:100 or empty",
    "scale_ratio": 100,
    "scale_confidence": 0.0
  }},
  "observations": [
    {{
      "division_code": "08",
      "section_code": "08 11 00",
      "item_code": "D-01",
      "item_description": "Door type D-01",
      "takeoff_category": "door_schedule",
      "quantity": 12,
      "uom": "No.",
      "applicable_area": "Ground to sixth floor",
      "element_type": "door",
      "element_mark": "D-01",
      "level_name": "Ground to sixth floor",
      "zone_name": "",
      "calculation_method": "schedule_count|symbol_count|written_dimensions|scaled_geometry",
      "calculation_formula": "count",
      "calculation_inputs": {{"count": 12}},
      "calculation_logic": "Exact explanation of how the visible figures produce the quantity.",
      "theoretical_basis": "Relevant quantity-surveying measurement basis and why the UOM applies.",
      "drawing_scale_text": "",
      "drawing_scale_ratio": null,
      "geometry_source": "written_dimension|schedule|scaled_pdf|symbol_detection",
      "geometry_data": {{
        "measured_pixels": 0,
        "calibration_pixels": 0,
        "calibration_length_m": 0
      }},
      "symbol_name": "D-01",
      "symbol_count": 12,
      "source_reference": "Visible schedule/plan region and identifiers",
      "evidence_text": "Exact short visible evidence supporting the inputs",
      "confidence_score": 0.92,
      "confidence_basis": "Why this score is warranted"
    }}
  ]
}}
""".strip()


def analyze_rendered_image(
    client,
    model,
    image_bytes,
    mime_type,
    project_id,
    source_file_name,
    page_no,
):
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": takeoff_prompt(project_id, source_file_name, page_no),
                    },
                    {
                        "type": "input_image",
                        "image_url": (
                            f"data:{mime_type};base64,"
                            f"{base64.b64encode(image_bytes).decode('ascii')}"
                        ),
                        "detail": "high",
                    },
                ],
            }
        ],
        max_output_tokens=int(os.getenv("TAKEOFF_OPENAI_MAX_OUTPUT_TOKENS", "12000")),
        store=False,
    )
    return parse_json_response(extract_response_text(response))


def render_pdf_pages(file_path, render_dpi):
    document = fitz.open(str(file_path))
    try:
        requested_scale = max(1.0, float(render_dpi) / 72.0)
        max_dimension_pixels = int(os.getenv("TAKEOFF_MAX_RENDER_DIMENSION", "6000"))
        for page_index, page in enumerate(document, start=1):
            longest_page_dimension = max(float(page.rect.width), float(page.rect.height), 1.0)
            capped_scale = float(max_dimension_pixels) / longest_page_dimension
            scale = max(1.0, min(requested_scale, capped_scale))
            matrix = fitz.Matrix(scale, scale)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            yield page_index, pixmap.tobytes("png"), "image/png"
    finally:
        document.close()


def image_payload(file_path):
    suffix = file_path.suffix.lower()
    mime_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else (
        "image/webp" if suffix == ".webp" else "image/png"
    )
    yield 1, file_path.read_bytes(), mime_type


def dxf_group_pairs(file_path):
    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return [
        (lines[index].strip(), lines[index + 1].strip())
        for index in range(0, len(lines) - 1, 2)
    ]


def analyze_dxf(file_path):
    """
    Deterministic ASCII-DXF geometry ingestion. Geometry is stored in metres only
    when $INSUNITS declares millimetres (4) or metres (6); otherwise observations
    are retained as rejected audit rows rather than assumed to be metric.
    """
    pairs = dxf_group_pairs(file_path)
    insunits = None
    for index, pair in enumerate(pairs):
        if pair == ("9", "$INSUNITS") and index + 1 < len(pairs):
            insunits = pairs[index + 1][1]
            break
    unit_factor = {"4": 0.001, "6": 1.0}.get(insunits)

    entity_counts = {}
    line_length = 0.0
    current = None
    attrs = {}

    def flush():
        nonlocal line_length
        if not current:
            return
        entity_counts[current] = entity_counts.get(current, 0) + 1
        if current == "LINE" and unit_factor:
            try:
                x1, y1 = float(attrs["10"]), float(attrs["20"])
                x2, y2 = float(attrs["11"]), float(attrs["21"])
                line_length += math.hypot(x2 - x1, y2 - y1) * unit_factor
            except (KeyError, ValueError):
                pass

    for code, value in pairs:
        if code == "0":
            flush()
            current = value.upper()
            attrs = {}
        elif current:
            attrs[code] = value
    flush()

    observations = []
    if line_length > 0:
        observations.append(
            {
                "division_code": "",
                "item_description": "Unclassified DXF line geometry",
                "takeoff_category": "cad_geometry",
                "quantity": line_length,
                "uom": "Lm",
                "calculation_method": "cad_geometry",
                "calculation_formula": "total_line_length_m",
                "calculation_inputs": {"total_line_length_m": line_length},
                "calculation_logic": (
                    "Sum of DXF LINE entity lengths converted using the file $INSUNITS value."
                ),
                "theoretical_basis": (
                    "Linear work is measured by centre-line length only after CAD units are explicit."
                ),
                "geometry_source": "dxf_entities",
                "geometry_data": {
                    "insunits": insunits,
                    "entity_counts": entity_counts,
                },
                "source_reference": "DXF ENTITIES section",
                "evidence_text": f"$INSUNITS={insunits}",
                "confidence_score": 0.70,
                "confidence_basis": (
                    "Geometry is deterministic, but layer/entity classification is required "
                    "before it can become a BOQ item."
                ),
            }
        )
    return {
        "page_classification": "cad",
        "ocr_text": "",
        "detected_scale": {},
        "observations": observations,
        "diagnostics": {"insunits": insunits, "entity_counts": entity_counts},
    }


def analyze_ifc(file_path):
    try:
        import ifcopenshell
        import ifcopenshell.util.element
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "IFC file found but ifcopenshell is not installed. Install ifcopenshell "
            "to enable BIM quantity ingestion; no BIM quantities were assumed."
        ) from exc

    model = ifcopenshell.open(str(file_path))
    observations = []
    for element in model.by_type("IfcElement"):
        quantities = ifcopenshell.util.element.get_psets(element, qtos_only=True)
        for qto_name, values in quantities.items():
            for name, value in values.items():
                if name == "id" or not isinstance(value, (int, float)) or value <= 0:
                    continue
                lowered = name.lower()
                if "area" in lowered:
                    uom = "m2"
                elif "volume" in lowered:
                    uom = "m3"
                elif "length" in lowered:
                    uom = "Lm"
                elif "count" in lowered:
                    uom = "No."
                else:
                    continue
                observations.append(
                    {
                        "division_code": "",
                        "item_description": f"{element.is_a()} {name}",
                        "takeoff_category": "bim_quantity",
                        "quantity": float(value),
                        "uom": uom,
                        "element_type": element.is_a(),
                        "element_mark": clean_text(getattr(element, "Tag", "")),
                        "calculation_method": "ifc_base_quantity",
                        "calculation_formula": "ifc_quantity",
                        "calculation_inputs": {"ifc_quantity": float(value)},
                        "calculation_logic": f"Direct IFC quantity {qto_name}.{name}.",
                        "theoretical_basis": (
                            "Native BIM base quantity used without geometric inference."
                        ),
                        "geometry_source": "ifc_quantity_set",
                        "geometry_data": {"qto_name": qto_name, "quantity_name": name},
                        "source_reference": f"IFC GlobalId={element.GlobalId}; {qto_name}.{name}",
                        "evidence_text": f"{name}={value}",
                        "confidence_score": 0.85,
                        "confidence_basis": (
                            "Native IFC quantity; classification remains mandatory."
                        ),
                    }
                )
    return {
        "page_classification": "bim",
        "ocr_text": "",
        "detected_scale": {},
        "observations": observations,
    }


def fetch_source_documents(engine, project_id):
    query = text(
        """
        SELECT
            document_id,
            file_name,
            file_extension,
            relative_path,
            absolute_path
        FROM build_document_inventory
        WHERE source_group = 'client_data'
          AND corpus_pack = :project_id
          AND LOWER(file_extension) IN (
              '.pdf', '.png', '.jpg', '.jpeg', '.webp', '.dxf', '.ifc'
          )
        ORDER BY document_id
        """
    )
    with engine.begin() as connection:
        return [
            dict(row)
            for row in connection.execute(
                query,
                {"project_id": project_id},
            ).mappings()
        ]


def ensure_takeoff_table(engine):
    if not inspect(engine).has_table("takeoff_layer"):
        raise RuntimeError(
            "takeoff_layer table does not exist. Run "
            "global_rag/scripts/takeoff_layer_schema.sql before calling the takeoff API."
        )


def observation_to_row(
    observation,
    analysis,
    document,
    takeoff_run_id,
    project_id,
    page_no,
    parser_name,
    model_name,
    confidence_threshold,
):
    detected_scale = analysis.get("detected_scale") or {}
    raw = {
        **observation,
        "drawing_scale_text": (
            observation.get("drawing_scale_text")
            or detected_scale.get("scale_text")
            or ""
        ),
        "drawing_scale_ratio": (
            observation.get("drawing_scale_ratio")
            or detected_scale.get("scale_ratio")
        ),
    }
    validated = validate_observation(raw, confidence_threshold)
    return {
        "takeoff_run_id": takeoff_run_id,
        "run_status": "processing",
        "project_id": project_id,
        "division_code": validated.get("division_code") or "00",
        "section_code": clean_text(validated.get("section_code")) or None,
        "item_code": clean_text(validated.get("item_code")) or None,
        "item_description": validated.get("item_description") or "Unclassified observation",
        "takeoff_category": clean_text(validated.get("takeoff_category")) or "general",
        "quantity": validated.get("quantity"),
        "uom": validated.get("uom") or "Item",
        "applicable_area": clean_text(validated.get("applicable_area")) or None,
        "element_type": clean_text(validated.get("element_type")) or None,
        "element_mark": clean_text(validated.get("element_mark")) or None,
        "level_name": clean_text(validated.get("level_name")) or None,
        "zone_name": clean_text(validated.get("zone_name")) or None,
        "calculation_method": clean_text(validated.get("calculation_method")) or "unknown",
        "calculation_formula": clean_text(validated.get("calculation_formula")) or None,
        "calculation_inputs": json.dumps(validated.get("calculation_inputs") or {}),
        "calculation_logic": clean_text(validated.get("calculation_logic")) or "Not provided",
        "theoretical_basis": clean_text(validated.get("theoretical_basis")) or "Not provided",
        "drawing_scale_text": clean_text(validated.get("drawing_scale_text")) or None,
        "drawing_scale_ratio": validated.get("drawing_scale_ratio"),
        "geometry_source": clean_text(validated.get("geometry_source")) or None,
        "geometry_data": json.dumps(validated.get("geometry_data") or {}),
        "symbol_name": clean_text(validated.get("symbol_name")) or None,
        "symbol_count": (
            int(validated.get("symbol_count"))
            if isinstance(validated.get("symbol_count"), (int, float))
            else None
        ),
        "source_document_id": document.get("document_id"),
        "source_file_name": document.get("file_name"),
        "source_relative_path": document.get("relative_path"),
        "source_page": page_no,
        "source_sheet": clean_text(validated.get("source_sheet")) or None,
        "source_table_id": None,
        "source_row_numbers": None,
        "source_reference": clean_text(validated.get("source_reference")) or (
            f"{document.get('file_name')} page {page_no}"
        ),
        "evidence_text": clean_text(validated.get("evidence_text")) or None,
        "extraction_method": "openai_vision_ocr" if parser_name == "openai_vision" else parser_name,
        "parser_name": parser_name,
        "model_name": model_name,
        "confidence_score": validated.get("confidence_score", 0),
        "confidence_threshold": confidence_threshold,
        "confidence_basis": clean_text(validated.get("confidence_basis")) or "Not provided",
        "validation_status": validated["validation_status"],
        "is_boq_ready": validated["is_boq_ready"],
        "rejection_reason": validated.get("rejection_reason"),
        "raw_extraction": json.dumps(
            {
                "observation": observation,
                "page_classification": analysis.get("page_classification"),
                "detected_scale": detected_scale,
            },
            default=str,
        ),
    }


INSERT_SQL = text(
    """
    INSERT INTO takeoff_layer (
        takeoff_run_id, run_status, project_id, division_code, section_code, item_code,
        item_description, takeoff_category, quantity, uom, applicable_area,
        element_type, element_mark, level_name, zone_name, calculation_method,
        calculation_formula, calculation_inputs, calculation_logic, theoretical_basis,
        drawing_scale_text, drawing_scale_ratio, geometry_source, geometry_data,
        symbol_name, symbol_count, source_document_id, source_file_name,
        source_relative_path, source_page, source_sheet, source_table_id,
        source_row_numbers, source_reference, evidence_text, extraction_method,
        parser_name, model_name, confidence_score, confidence_threshold,
        confidence_basis, validation_status, is_boq_ready, rejection_reason,
        raw_extraction, created_at, updated_at
    ) VALUES (
        :takeoff_run_id, :run_status, :project_id, :division_code, :section_code, :item_code,
        :item_description, :takeoff_category, :quantity, :uom, :applicable_area,
        :element_type, :element_mark, :level_name, :zone_name, :calculation_method,
        :calculation_formula, CAST(:calculation_inputs AS JSONB), :calculation_logic,
        :theoretical_basis, :drawing_scale_text, :drawing_scale_ratio,
        :geometry_source, CAST(:geometry_data AS JSONB), :symbol_name, :symbol_count,
        :source_document_id, :source_file_name, :source_relative_path, :source_page,
        :source_sheet, :source_table_id, :source_row_numbers, :source_reference,
        :evidence_text, :extraction_method, :parser_name, :model_name,
        :confidence_score, :confidence_threshold, :confidence_basis,
        :validation_status, :is_boq_ready, :rejection_reason,
        CAST(:raw_extraction AS JSONB), NOW(), NOW()
    )
    """
)


def generate_boq_takeoff(project_id):
    """
    Render and analyze all eligible client drawings for a project.

    The API intentionally exposes only project_id. Operational values are
    environment-backed/hardcoded so the BOQ v2 API contract remains unchanged.
    """
    project_id = clean_text(project_id)
    if not project_id:
        raise ValueError("project_id must be provided.")

    base = config.config_base()
    paths = config.config_paths(client_data=project_id)
    if not base.get("openai_api_key"):
        raise RuntimeError("OPENAI_API_KEY is required for the takeoff layer.")

    engine = create_engine(base["db_url"], pool_pre_ping=True)
    ensure_takeoff_table(engine)
    documents = fetch_source_documents(engine, project_id)
    if not documents:
        raise RuntimeError(
            f"No eligible client drawing files found in build_document_inventory for {project_id}."
        )

    confidence_threshold = float(
        os.getenv("TAKEOFF_CONFIDENCE_THRESHOLD", str(DEFAULT_CONFIDENCE_THRESHOLD))
    )
    if not 0 < confidence_threshold <= 1:
        raise ValueError("TAKEOFF_CONFIDENCE_THRESHOLD must be greater than 0 and at most 1.")
    render_dpi = int(os.getenv("TAKEOFF_RENDER_DPI", str(DEFAULT_RENDER_DPI)))
    max_pages = int(os.getenv("TAKEOFF_MAX_PAGES_PER_RUN", "500"))
    model = os.getenv(
        "OPENAI_TAKEOFF_MODEL",
        os.getenv("OPENAI_PDF_OCR_MODEL", base.get("llm_model")),
    )
    client = OpenAI(api_key=base["openai_api_key"])
    takeoff_run_id = make_run_id(project_id)

    rows_written = 0
    accepted_rows = 0
    rejected_rows = 0
    pages_processed = 0
    file_results = []

    for document in documents:
        file_path = Path(document.get("absolute_path") or "")
        if not file_path.exists():
            file_path = paths["project_root"] / str(document.get("relative_path") or "")
        suffix = file_path.suffix.lower()
        file_result = {
            "document_id": document.get("document_id"),
            "file_name": document.get("file_name"),
            "status": "pending",
            "pages_processed": 0,
            "rows_written": 0,
        }
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"Source file not found: {file_path}")
            if suffix in PDF_EXTENSIONS:
                payloads = render_pdf_pages(file_path, render_dpi)
                parser_name = "openai_vision"
            elif suffix in IMAGE_EXTENSIONS:
                payloads = image_payload(file_path)
                parser_name = "openai_vision"
            elif suffix in CAD_EXTENSIONS:
                payloads = [(1, None, None)]
                parser_name = "dxf_ascii_geometry"
            elif suffix in BIM_EXTENSIONS:
                payloads = [(1, None, None)]
                parser_name = "ifc_quantity_sets"
            else:
                continue

            for page_no, image_bytes, mime_type in payloads:
                if pages_processed >= max_pages:
                    raise RuntimeError(
                        f"TAKEOFF_MAX_PAGES_PER_RUN={max_pages} reached. "
                        "Increase the configured limit and rerun."
                    )
                if suffix in CAD_EXTENSIONS:
                    analysis = analyze_dxf(file_path)
                elif suffix in BIM_EXTENSIONS:
                    analysis = analyze_ifc(file_path)
                else:
                    analysis = analyze_rendered_image(
                        client=client,
                        model=model,
                        image_bytes=image_bytes,
                        mime_type=mime_type,
                        project_id=project_id,
                        source_file_name=document.get("file_name"),
                        page_no=page_no,
                    )
                rows = [
                    observation_to_row(
                        observation=observation,
                        analysis=analysis,
                        document=document,
                        takeoff_run_id=takeoff_run_id,
                        project_id=project_id,
                        page_no=page_no,
                        parser_name=parser_name,
                        model_name=model if parser_name == "openai_vision" else None,
                        confidence_threshold=confidence_threshold,
                    )
                    for observation in (analysis.get("observations") or [])
                    if isinstance(observation, dict)
                ]
                if rows:
                    with engine.begin() as connection:
                        connection.execute(INSERT_SQL, rows)
                    rows_written += len(rows)
                    file_result["rows_written"] += len(rows)
                    accepted_rows += sum(row["is_boq_ready"] for row in rows)
                    rejected_rows += sum(not row["is_boq_ready"] for row in rows)
                pages_processed += 1
                file_result["pages_processed"] += 1
            file_result["status"] = "ok"
        except Exception as exc:
            file_result["status"] = "failed"
            file_result["error"] = f"{type(exc).__name__}: {str(exc)}"
        file_results.append(file_result)

    failed_files = sum(result.get("status") == "failed" for result in file_results)
    run_is_publishable = accepted_rows > 0 and failed_files == 0

    with engine.begin() as connection:
        if run_is_publishable:
            connection.execute(
                text(
                    """
                    UPDATE takeoff_layer
                    SET validation_status = 'superseded',
                        run_status = 'superseded',
                        is_boq_ready = FALSE,
                        updated_at = NOW()
                    WHERE project_id = :project_id
                      AND takeoff_run_id <> :takeoff_run_id
                      AND run_status = 'completed'
                    """
                ),
                {"project_id": project_id, "takeoff_run_id": takeoff_run_id},
            )
            connection.execute(
                text(
                    """
                    UPDATE takeoff_layer
                    SET run_status = 'completed',
                        updated_at = NOW()
                    WHERE takeoff_run_id = :takeoff_run_id
                    """
                ),
                {"takeoff_run_id": takeoff_run_id},
            )
        else:
            connection.execute(
                text(
                    """
                    UPDATE takeoff_layer
                    SET run_status = 'failed',
                        is_boq_ready = FALSE,
                        updated_at = NOW()
                    WHERE takeoff_run_id = :takeoff_run_id
                    """
                ),
                {"takeoff_run_id": takeoff_run_id},
            )

    return {
        "message": "BOQ takeoff layer processing completed.",
        "status": "ok" if run_is_publishable else "validation_failed",
        "takeoff_run_id": takeoff_run_id,
        "project_id": project_id,
        "confidence_threshold": confidence_threshold,
        "render_dpi": render_dpi,
        "model": model,
        "documents_selected": len(documents),
        "pages_processed": pages_processed,
        "rows_written": rows_written,
        "accepted_boq_ready_rows": accepted_rows,
        "rejected_or_below_threshold_rows": rejected_rows,
        "failed_files": failed_files,
        "run_published_for_boq": run_is_publishable,
        "file_results": file_results,
        "next_step": (
            "Call the existing /generate_boq_v2 API after reviewing takeoff_layer accepted rows."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(generate_boq_takeoff("ai_construction_cost_estimation_platform"), indent=2))
