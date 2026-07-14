# File_name: report_generation.py
# Purpose: Generate a SOW-driven AI first-line IC review pack for investment submissions.

# The script:
# 1. Uses retrieve_chunks.py for staged evidence retrieval
# 2. Builds SOW-aligned review modules
# 3. Clearly marks unavailable SOW areas as Grey / Not assessed
# 4. Produces JSON and Markdown outputs
# 5. Optionally writes an audit record to PostgreSQL

# Key controls:
# - No investment recommendation is made.
# - Historical IC learning is not claimed if historical logs are unavailable.
# - EV/EBITDA external benchmarking is not claimed unless external evidence exists.
# - Macro / FX / GDP conclusions are not fabricated where source data is unavailable.
# - Every finding is traceable through chunk_id / document_id / source_reference.

# Assumptions:
# - global_rag.scripts.config contains config_base() and config_paths()
# - global_rag.scripts.retrieve_chunks contains retrieve_chunks()
# - chunks table and embeddings are already populated

import json
import re
import hashlib
from pathlib import Path
from datetime import datetime, timezone

from sqlalchemy import create_engine, text
from openai import OpenAI

from global_rag.scripts import config
import global_rag.scripts.retrieve_chunks as ret


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

REPORT_GENERATION_VERSION = "report_generation_stage2_asset_overlay_v1"
REPORT_TYPE = "ai_first_line_ic_review"
CLASSIFICATION = "Confidential External"

STATUS_GREEN = "Green"
STATUS_AMBER = "Amber"
STATUS_RED = "Red"
STATUS_GREY = "Grey"

READINESS_PASS = "Pass"
READINESS_PARTIAL = "Partial"
READINESS_MISSING = "Missing"
READINESS_WEAK = "Weak"
READINESS_NOT_ASSESSED = "Not assessed"


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def clean_text(value):
    if value is None:
        return ""

    value = str(value)
    value = value.replace("\x00", " ")
    value = value.replace("\r", " ")
    value = value.replace("\n", " ")
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_text(value):
    return clean_text(value).lower()


def safe_float(value):
    if value is None:
        return None

    try:
        value = str(value)
        value = value.replace(",", "")
        value = value.replace("%", "")
        value = value.replace("x", "")
        value = value.replace("X", "")
        value = value.strip()
        return float(value)
    except Exception:
        return None


def has_any(text_value, terms):
    text_value = normalize_text(text_value)

    for term in terms:
        if normalize_text(term) in text_value:
            return True

    return False


def has_all(text_value, terms):
    text_value = normalize_text(text_value)

    for term in terms:
        if normalize_text(term) not in text_value:
            return False

    return True


def make_run_id(transaction_id):
    raw_value = f"{transaction_id}_{utc_now_iso()}_{REPORT_GENERATION_VERSION}"
    digest = hashlib.sha256(raw_value.encode("utf-8")).hexdigest()[:12]
    return f"RPT_{transaction_id}_{digest}"


def get_config_pack(client_data):
    config_base = config.config_base()
    config_paths = config.config_paths(client_data=client_data)

    draft_report_dir = Path(config_paths["draft_report_dir"])
    draft_report_dir.mkdir(parents=True, exist_ok=True)

    return {
        "config_base": config_base,
        "config_paths": config_paths,
        "draft_report_dir": draft_report_dir,
    }


def get_evidence_blob(results):
    text_parts = []

    for item in results:
        text_parts.append(clean_text(item.get("chunk_text")))
        text_parts.append(clean_text(item.get("section_heading")))
        text_parts.append(clean_text(item.get("source_reference")))
        text_parts.append(clean_text(item.get("document_id")))
        text_parts.append(clean_text(item.get("corpus_zone")))
        text_parts.append(clean_text(item.get("corpus_pack")))

    return " ".join(text_parts)


def get_source_reference(result):
    source_reference = clean_text(result.get("source_reference"))

    table_id = None
    table_match = re.search(r"table_id=([^;]+)", source_reference)
    if table_match:
        table_id = table_match.group(1).strip()

    page_no = None
    page_match = re.search(r"page_no=([^;]+)", source_reference)
    if page_match:
        page_no = page_match.group(1).strip()

    row_start = None
    row_end = None

    row_start_match = re.search(r"row_start=([^;]+)", source_reference)
    row_end_match = re.search(r"row_end=([^;]+)", source_reference)

    if row_start_match:
        row_start = row_start_match.group(1).strip()

    if row_end_match:
        row_end = row_end_match.group(1).strip()

    return {
        "chunk_id": result.get("chunk_id"),
        "document_id": result.get("document_id"),
        "corpus_zone": result.get("corpus_zone"),
        "corpus_pack": result.get("corpus_pack"),
        "section_heading": result.get("section_heading"),
        "page_start": result.get("page_start"),
        "page_end": result.get("page_end"),
        "page_no": page_no,
        "table_id": table_id,
        "row_start": row_start,
        "row_end": row_end,
        "source_reference": source_reference,
    }


def find_source_chunks(results, terms, max_sources=5):
    matched_sources = []
    seen_chunk_ids = set()

    for item in results:
        chunk_text = clean_text(item.get("chunk_text"))
        chunk_id = item.get("chunk_id")

        if chunk_id in seen_chunk_ids:
            continue

        if has_any(chunk_text, terms):
            matched_sources.append(get_source_reference(item))
            seen_chunk_ids.add(chunk_id)

        if len(matched_sources) >= max_sources:
            break

    return matched_sources


def short_evidence_text(results, terms, max_chars=600):
    for item in results:
        chunk_text = clean_text(item.get("chunk_text"))

        if has_any(chunk_text, terms):
            return chunk_text[:max_chars]

    return ""


def split_evidence_by_terms(results, positive_terms, weak_terms):
    positive_matches = []
    weak_matches = []

    for item in results:
        chunk_text = clean_text(item.get("chunk_text"))

        if not has_any(chunk_text, positive_terms):
            continue

        if weak_terms and has_any(chunk_text, weak_terms):
            weak_matches.append(item)
        else:
            positive_matches.append(item)

    return positive_matches, weak_matches


def dedupe_results(results):
    seen_signatures = set()
    deduped = []

    for item in results:
        chunk_text = normalize_text(item.get("chunk_text"))
        signature_text = chunk_text[:900]
        signature_text = re.sub(r"\s+", " ", signature_text)
        signature = hashlib.sha256(signature_text.encode("utf-8")).hexdigest()

        if signature in seen_signatures:
            continue

        seen_signatures.add(signature)
        deduped.append(item)

    return deduped


# ---------------------------------------------------------------------
# Source classification
# ---------------------------------------------------------------------

def classify_evidence_type(result):
    """
    Classify evidence using stable metadata only.

    Do not classify a chunk as workbook/memo/deck based on words inside chunk_text,
    because a deck can mention a workbook and a memo can mention a benchmark.
    """

    corpus_zone = clean_text(result.get("corpus_zone"))
    document_id = clean_text(result.get("document_id"))
    section_heading = normalize_text(result.get("section_heading"))
    source_reference = normalize_text(result.get("source_reference"))
    corpus_pack = normalize_text(result.get("corpus_pack"))

    if corpus_zone == "client_data":
        if "memorandum" in source_reference or "memo" in source_reference:
            return "client_memo"

        if "deck" in source_reference or section_heading.startswith("slide_"):
            return "client_deck"

        if section_heading.startswith("page_"):
            return "client_pdf"

        if "docx_table" in source_reference or section_heading == "docx_document_text":
            return "client_supporting_note"

        if "financial" in source_reference or "assumption" in source_reference:
            return "client_workbook_control"

        if section_heading.startswith("table:"):
            return "client_table_evidence"

        return "client_evidence"

    if corpus_zone == "corpus_data":
        combined_metadata = normalize_text(
            f"{corpus_pack} {document_id} {section_heading} {source_reference}"
        )

        if "benchmark" in combined_metadata or "nrel" in combined_metadata or "utility-scale" in combined_metadata:
            return "external_benchmark"

        if "seed" in combined_metadata or "risk" in combined_metadata:
            return "risk_taxonomy"

        return "public_context"

    return "unknown"


def infer_source_label(result):
    evidence_type = classify_evidence_type(result)

    if evidence_type == "client_workbook_control":
        return "workbook"
    if evidence_type == "client_memo":
        return "memo"
    if evidence_type == "client_deck":
        return "deck"
    if evidence_type == "client_pdf":
        return "pdf_deck"
    if evidence_type == "client_supporting_note":
        return "client_supporting_note"
    if evidence_type == "client_table_evidence":
        return "client_table_evidence"

    return "unknown"


# ---------------------------------------------------------------------
# Evidence selection scoring
# ---------------------------------------------------------------------

def get_default_module_boost_terms(transaction_profile=None):
    transaction_profile = transaction_profile or {}
    asset_classes = transaction_profile.get("asset_classes") or [transaction_profile.get("primary_asset_class") or transaction_profile.get("asset_class")]
    asset_classes = [normalize_text(asset_class) for asset_class in asset_classes if clean_text(asset_class)]
    revenue_models = transaction_profile.get("revenue_models") or [transaction_profile.get("revenue_model")]
    revenue_models = [normalize_text(revenue_model) for revenue_model in revenue_models if clean_text(revenue_model)]
    asset_class = normalize_text(transaction_profile.get("primary_asset_class") or transaction_profile.get("asset_class"))
    revenue_model = " ".join(revenue_models)
    profile_terms = get_profile_terms(transaction_profile).split() if transaction_profile else []

    terms = {
        "risk_review": [
            "risk",
            "rating",
            "mitigation",
            "grid cost",
            "offtaker credit",
            "cybersecurity",
            "legal",
            "permit",
        ] + profile_terms,
        "conditions_open_items": [
            "conditions",
            "final documentation",
            "metric reconciliation",
            "offtaker-credit",
            "decommissioning",
            "open items",
        ] + profile_terms,
        "financial_reconciliation": [
            "project irr",
            "equity irr",
            "npv",
            "minimum dscr",
            "ev / ebitda",
            "total project cost",
            "total capex",
            "metrics differ",
        ],
        "sensitivity_review": [
            "sensitivity",
            "combined downside",
            "equity irr",
            "minimum dscr",
            "included_in_submission",
            "capex",
            "cod delay",
            "availability",
        ] + profile_terms,
        "market_offtake_revenue": [
            "revenue",
            "offtake",
            "payment security",
            "counterparty",
            "tariff",
            "capacity payment",
        ] + profile_terms,
        "external_cost_benchmark": [
            "overnight capital cost",
            "grid connection cost",
            "fixed operating expenses",
            "technology cost benchmark",
        ] + profile_terms,
    }

    if any(item in asset_classes for item in ["solar_pv", "onshore_wind", "offshore_wind"]):
        terms["market_offtake_revenue"].extend(["ppa", "cfd", "merchant", "power price"])
        terms["sensitivity_review"].extend(["generation", "resource"])

    if "bess" in asset_classes:
        terms["market_offtake_revenue"].extend(["tolling", "ancillary services", "capacity", "shifting"])
        terms["sensitivity_review"].extend(["augmentation", "degradation", "cycling"])

    if "hydrogen" in asset_classes:
        terms["market_offtake_revenue"].extend(["hydrogen offtake", "ammonia", "lcoh", "take-or-pay"])
        terms["sensitivity_review"].extend(["utilisation", "electricity price", "electrolyser"])

    if "regulated_grid_distribution" in asset_classes or "regulated" in revenue_model:
        terms["market_offtake_revenue"].extend(["regulated return", "tariff", "allowed revenue", "rab"])
        terms["risk_review"].extend(["regulatory", "allowed return", "grid modernisation"])

    return terms


def score_result_for_module(result, module_key, review_config=None):
    score = 0.0

    similarity_score = result.get("similarity_score")
    hybrid_score = result.get("hybrid_score")
    keyword_score = result.get("keyword_score")

    if isinstance(similarity_score, (float, int)):
        score += float(similarity_score) * 10

    if isinstance(hybrid_score, (float, int)):
        score += float(hybrid_score) * 100

    if isinstance(keyword_score, (float, int)):
        score += float(keyword_score)

    section_heading = normalize_text(result.get("section_heading"))
    chunk_text = normalize_text(result.get("chunk_text"))
    evidence_type = classify_evidence_type(result)
    token_count = result.get("token_count_estimate") or 0

    if section_heading.startswith("table:"):
        score += 2.5

    if evidence_type == "client_workbook_control":
        if module_key in [
            "financial_reconciliation",
            "market_offtake_revenue",
            "sensitivity_review",
            "client_valuation",
        ]:
            score += 3.0

    if evidence_type in ["client_memo", "client_deck", "client_pdf", "client_evidence"]:
        if module_key in [
            "completeness_readiness",
            "strategy_fit",
            "risk_review",
            "conditions_open_items",
        ]:
            score += 1.5

    if evidence_type == "external_benchmark":
        if module_key in ["external_cost_benchmark", "external_ev_ebitda_benchmark"]:
            score += 4.0

    if token_count < 80 and not section_heading.startswith("table:"):
        score -= 3.0

    review_config = review_config or {}
    module_boost_terms = review_config.get("module_boost_terms") or get_default_module_boost_terms()
    if module_key == "risk_review":
        module_boost_terms = dict(module_boost_terms)
        module_boost_terms["risk_review"] = (
            module_boost_terms.get("risk_review", [])
            + get_risk_taxonomy_terms(review_config)
        )

    for term in module_boost_terms.get(module_key, []):
        if term in chunk_text:
            score += 0.75

    return score


def select_evidence(results, module_key, top_n=10, review_config=None):
    results = dedupe_results(results)

    scored_results = []

    for item in results:
        scored_results.append(
            {
                "score": score_result_for_module(item, module_key, review_config),
                "item": item,
            }
        )

    scored_results = sorted(scored_results, key=lambda x: x["score"], reverse=True)

    selected = []
    for row in scored_results[:top_n]:
        selected.append(row["item"])

    return selected


# ---------------------------------------------------------------------
# Config objects
# ---------------------------------------------------------------------

def merge_config_lists(*configs):
    merged = {}

    for config_item in configs:
        if not config_item:
            continue

        for key, value in config_item.items():
            if isinstance(value, list):
                merged.setdefault(key, [])
                merged[key].extend(value)
            elif isinstance(value, dict):
                merged.setdefault(key, {})
                merged[key].update(value)
            else:
                merged[key] = value

    return merged


def get_profile_terms(transaction_profile):
    asset_classes = " ".join(transaction_profile.get("asset_classes", []))
    revenue_models = " ".join(transaction_profile.get("revenue_models", []))
    asset_class = normalize_text(transaction_profile.get("primary_asset_class") or transaction_profile.get("asset_class"))
    revenue_model = normalize_text(transaction_profile.get("revenue_model"))
    energy_value_chain = normalize_text(transaction_profile.get("energy_value_chain"))
    technology_subtypes = " ".join(transaction_profile.get("technology_subtypes", []))

    return clean_text(
        f"{asset_class} {asset_classes} {revenue_model} {revenue_models} {energy_value_chain} {technology_subtypes}"
    )


def get_risk_taxonomy_terms(review_config):
    terms = []
    for risk_item in review_config.get("asset_class_risk_taxonomy", []):
        terms.extend(risk_item.get("terms", []))
        terms.append(risk_item.get("risk", ""))
    return [term for term in terms if clean_text(term)]


def get_asset_class_risk_overlay_library(transaction_profile):
    asset_classes = transaction_profile.get("asset_classes") or [transaction_profile.get("primary_asset_class") or transaction_profile.get("asset_class")]
    asset_classes = [normalize_text(asset_class) for asset_class in asset_classes if clean_text(asset_class)]
    libraries = {
        "solar_pv": [
            {
                "risk_id": "SOLAR_CURTAILMENT",
                "risk": "Solar curtailment and grid dispatch risk",
                "asset_class": "solar_pv",
                "terms": ["curtailment", "dispatch", "grid constraint", "export limit", "congestion"],
                "reviewer_challenge": "Has curtailment risk been adequately stress-tested?",
            },
            {
                "risk_id": "SOLAR_RESOURCE_DEGRADATION",
                "risk": "Solar resource, degradation and availability",
                "asset_class": "solar_pv",
                "terms": ["irradiation", "p50", "p90", "degradation", "availability", "module warranty"],
                "reviewer_challenge": "Are irradiation, degradation and availability assumptions supported by independent technical evidence?",
            },
            {
                "risk_id": "SOLAR_EPC_MODULE_SUPPLY",
                "risk": "EPC, module supply and equipment warranty",
                "asset_class": "solar_pv",
                "terms": ["epc", "module", "inverter", "supplier", "warranty", "liquidated damages"],
                "reviewer_challenge": "Are module, inverter and EPC obligations bankable and reflected in contingency assumptions?",
            },
        ],
        "onshore_wind": [
            {
                "risk_id": "ONSHORE_WIND_RESOURCE",
                "risk": "Wind resource and production uncertainty",
                "asset_class": "onshore_wind",
                "terms": ["wind resource", "p50", "p90", "wake loss", "energy yield", "turbine availability"],
                "reviewer_challenge": "Has wind resource uncertainty and wake loss been adequately stress-tested?",
            },
            {
                "risk_id": "ONSHORE_WIND_TURBINE_OEM",
                "risk": "Turbine OEM, warranty and availability",
                "asset_class": "onshore_wind",
                "terms": ["turbine", "oem", "availability warranty", "blade", "gearbox", "service agreement"],
                "reviewer_challenge": "Are turbine OEM warranties, service obligations and availability assumptions sufficient for the return case?",
            },
            {
                "risk_id": "ONSHORE_WIND_PERMIT_COMMUNITY",
                "risk": "Permitting, land access and community risk",
                "asset_class": "onshore_wind",
                "terms": ["permit", "land lease", "community", "noise", "setback", "environmental approval"],
                "reviewer_challenge": "Are land, community and permitting dependencies closed or reflected in schedule contingency?",
            },
        ],
        "offshore_wind": [
            {
                "risk_id": "OFFSHORE_CONSTRUCTION_INSTALLATION",
                "risk": "Offshore construction and marine installation",
                "asset_class": "offshore_wind",
                "terms": ["offshore construction", "marine installation", "vessel", "foundation", "weather window"],
                "reviewer_challenge": "Is offshore construction risk sufficiently reflected in contingency assumptions?",
            },
            {
                "risk_id": "OFFSHORE_GRID_EXPORT",
                "risk": "Export cable, grid connection and offshore transmission",
                "asset_class": "offshore_wind",
                "terms": ["export cable", "offshore substation", "grid connection", "transmission", "interarray cable"],
                "reviewer_challenge": "Are export cable and grid interface risks allocated and supported by credible schedule assumptions?",
            },
            {
                "risk_id": "OFFSHORE_SEABED_CONSENT",
                "risk": "Seabed, consenting and environmental interfaces",
                "asset_class": "offshore_wind",
                "terms": ["seabed", "consent", "marine permit", "environmental impact", "lease"],
                "reviewer_challenge": "Are seabed, consent and environmental interfaces sufficiently advanced for the approval stage?",
            },
        ],
        "bess": [
            {
                "risk_id": "BESS_DEGRADATION_AUGMENTATION",
                "risk": "Battery degradation and augmentation",
                "asset_class": "bess",
                "terms": ["degradation", "augmentation", "cycle", "state of health", "warranty"],
                "reviewer_challenge": "Are degradation, augmentation and warranty assumptions reflected in the downside case?",
            },
            {
                "risk_id": "BESS_REVENUE_STACK",
                "risk": "Storage revenue stack and dispatch assumptions",
                "asset_class": "bess",
                "terms": ["ancillary services", "capacity", "arbitrage", "dispatch", "tolling"],
                "reviewer_challenge": "Is the storage revenue stack supported by contracted revenue or independently validated market assumptions?",
            },
            {
                "risk_id": "BESS_SAFETY_INTERCONNECTION",
                "risk": "Battery safety, fire protection and interconnection",
                "asset_class": "bess",
                "terms": ["fire", "safety", "thermal runaway", "interconnection", "ems", "scada"],
                "reviewer_challenge": "Are fire safety, EMS / SCADA and interconnection risks adequately mitigated?",
            },
        ],
        "hydrogen": [
            {
                "risk_id": "HYDROGEN_OFFTAKE_FIRMNESS",
                "risk": "Hydrogen offtake firmness and utilisation",
                "asset_class": "hydrogen",
                "terms": ["hydrogen offtake", "take-or-pay", "utilisation", "customer", "ammonia offtake"],
                "reviewer_challenge": "Has hydrogen offtake firmness been reconciled with electrolyser utilisation assumptions?",
            },
            {
                "risk_id": "HYDROGEN_POWER_COST",
                "risk": "Power supply, electricity price and LCOH",
                "asset_class": "hydrogen",
                "terms": ["electricity price", "power supply", "lcoh", "renewable power", "ppa"],
                "reviewer_challenge": "Are power price, renewable supply and LCOH assumptions robust under downside cases?",
            },
            {
                "risk_id": "HYDROGEN_TECH_EXECUTION",
                "risk": "Electrolyser technology, construction and ramp-up",
                "asset_class": "hydrogen",
                "terms": ["electrolyser", "electrolyzer", "ramp-up", "availability", "stack replacement"],
                "reviewer_challenge": "Are electrolyser availability, stack replacement and ramp-up risks reflected in contingency and returns?",
            },
        ],
    }
    risk_items = []
    for asset_class in asset_classes:
        risk_items.extend(libraries.get(asset_class, []))
    return {"asset_class_risk_taxonomy": risk_items}


def get_legal_risk_overlay_library(transaction_profile):
    return {
        "asset_class_risk_taxonomy": [
            {
                "risk_id": "R08_LEGAL_TRANSACTION",
                "risk": "Legal transaction risk",
                "asset_class": "cross_asset",
                "terms": ["title", "change of control", "consent", "permit", "license", "material contracts", "sanctions", "aml"],
                "reviewer_challenge": "Are title, transfer consents, permits, contracts, sanctions and AML diligence complete?",
            },
            {
                "risk_id": "LEGAL_CONTRACT_ENFORCEABILITY",
                "risk": "Contract enforceability and termination rights",
                "asset_class": "cross_asset",
                "terms": ["termination", "default", "force majeure", "step-in", "assignment", "governing law"],
                "reviewer_challenge": "Are termination, default, assignment and step-in rights acceptable for the proposed investment structure?",
            },
        ]
    }


def get_blind_spot_overlay_library(transaction_profile):
    return {
        "asset_class_risk_taxonomy": [
            {
                "risk_id": "R09_BLIND_SPOT_RECONCILIATION",
                "risk": "Past-transaction blind spots",
                "asset_class": "cross_asset",
                "terms": ["inconsistent", "reconcile", "terminal value", "contingency", "cod", "benchmark", "missing"],
                "reviewer_challenge": "Have common IC blind spots such as unreconciled values, optimistic COD and weak contingency been challenged?",
            },
            {
                "risk_id": "BLIND_SPOT_MODEL_PACK_CONFLICT",
                "risk": "Model, memo and deck inconsistency",
                "asset_class": "cross_asset",
                "terms": ["metrics differ", "different values", "model", "deck", "memo", "reconciliation"],
                "reviewer_challenge": "Have all material model, memo and deck conflicts been reconciled before approval?",
            },
        ],
        "common_ic_questions": [
            {
                "question_id": "ICQ_BLIND_SPOT_CONTINGENCY",
                "theme": "Contingency and optimism bias",
                "question": "Has contingency been benchmarked against execution risk rather than set as a balancing item?",
                "coverage_terms": ["contingency", "benchmark", "cost overrun", "overrun"],
                "weak_terms": ["not quantified", "not benchmarked", "not included"],
            },
        ],
    }


def get_market_geography_overlay_library(transaction_profile):
    geography = normalize_text(transaction_profile.get("geography"))
    terms = ["country risk", "fx", "inflation", "interest rate", "regulation", "tax", "political risk"]
    if geography not in ["not_identified", ""]:
        terms.append(geography)
    return {
        "asset_class_risk_taxonomy": [
            {
                "risk_id": "GEO_MARKET_MACRO",
                "risk": "Geography, market and macro risk",
                "asset_class": "cross_asset",
                "terms": terms,
                "reviewer_challenge": "Are country, FX, inflation, tax and regulatory risks reflected in valuation and downside assumptions?",
            }
        ],
        "common_ic_questions": [
            {
                "question_id": "ICQ_GEO_MARKET",
                "theme": "Market and geography risk",
                "question": "Do the downside cases reflect market, country, FX, inflation and regulatory risks for the transaction geography?",
                "coverage_terms": terms,
                "weak_terms": ["not assessed", "not included", "not modelled", "pending"],
            }
        ],
    }


def get_revenue_model_overlay_library(transaction_profile):
    revenue_models = transaction_profile.get("revenue_models") or [transaction_profile.get("revenue_model")]
    revenue_models = [normalize_text(revenue_model) for revenue_model in revenue_models if clean_text(revenue_model)]
    overlays = {
        "merchant": {
            "asset_class_risk_taxonomy": [
                {
                    "risk_id": "REV_MERCHANT_PRICE",
                    "risk": "Merchant price exposure",
                    "asset_class": "revenue_model",
                    "terms": ["merchant", "merchant price", "price curve", "capture price"],
                    "reviewer_challenge": "Is merchant price exposure supported by independent curves and downside capture-price cases?",
                }
            ]
        },
        "ppa": {
            "asset_class_risk_taxonomy": [
                {
                    "risk_id": "REV_PPA_COUNTERPARTY",
                    "risk": "PPA counterparty and contract terms",
                    "asset_class": "revenue_model",
                    "terms": ["ppa", "offtaker", "termination", "credit", "payment security"],
                    "reviewer_challenge": "Are PPA counterparty credit, termination rights and payment security adequate?",
                }
            ]
        },
        "regulated_return": {
            "asset_class_risk_taxonomy": [
                {
                    "risk_id": "REV_REGULATED_RETURN",
                    "risk": "Regulated return and tariff reset",
                    "asset_class": "revenue_model",
                    "terms": ["regulated return", "tariff", "allowed revenue", "rab", "reset"],
                    "reviewer_challenge": "Are tariff reset, allowed return and regulatory true-up assumptions adequately supported?",
                }
            ]
        },
        "regulated_grid_distribution": {
            "asset_class_risk_taxonomy": [
                {
                    "risk_id": "REV_GRID_DISTRIBUTION_TARIFF",
                    "risk": "Distribution tariff and allowed revenue",
                    "asset_class": "revenue_model",
                    "terms": ["distribution tariff", "allowed revenue", "regulated asset base", "network capex"],
                    "reviewer_challenge": "Is the distribution tariff framework reconciled with grid modernisation capex and allowed revenue assumptions?",
                }
            ]
        },
        "hydrogen_offtake": {
            "asset_class_risk_taxonomy": [
                {
                    "risk_id": "REV_HYDROGEN_OFFTAKE",
                    "risk": "Hydrogen offtake and customer credit",
                    "asset_class": "revenue_model",
                    "terms": ["hydrogen offtake", "customer", "take-or-pay", "credit", "ammonia"],
                    "reviewer_challenge": "Is hydrogen offtake firmness, tenor and customer credit sufficient for the utilisation case?",
                }
            ]
        },
    }
    merged_overlay = {}
    for revenue_model in revenue_models:
        merged_overlay = merge_config_lists(merged_overlay, overlays.get(revenue_model, {}))
    return merged_overlay


def get_sensitivity_overlay_library(transaction_profile):
    asset_classes = transaction_profile.get("asset_classes") or [transaction_profile.get("primary_asset_class") or transaction_profile.get("asset_class")]
    asset_classes = [normalize_text(asset_class) for asset_class in asset_classes if clean_text(asset_class)]
    themes = {
        "solar_pv": {
            "required": ["Irradiation - P90", "Generation -5%", "Curtailment downside"],
            "recommended": ["Module degradation"],
            "optional": [],
        },
        "onshore_wind": {
            "required": ["Wind resource P90", "Wake loss downside", "Turbine availability downside"],
            "recommended": ["COD delay 6 months"],
            "optional": [],
        },
        "offshore_wind": {
            "required": ["Offshore construction delay", "Vessel/weather downtime", "Capex +15%"],
            "recommended": ["Export cable failure"],
            "optional": [],
        },
        "bess": {
            "required": ["BESS degradation / augmentation", "Cycling downside", "Availability downside"],
            "recommended": ["Ancillary services price downside"],
            "optional": [],
        },
        "hydrogen": {
            "required": ["Electrolyser utilisation downside", "Electricity price upside", "Hydrogen offtake volume downside"],
            "recommended": ["LCOH downside"],
            "optional": [],
        },
        "regulated_grid_distribution": {
            "required": ["Allowed revenue reduction", "Tariff reset downside", "Network capex overrun"],
            "recommended": ["Regulatory delay"],
            "optional": [],
        },
    }
    tiered_themes = {"required": [], "recommended": [], "optional": []}
    for asset_class in asset_classes:
        asset_themes = themes.get(asset_class, {"required": [], "recommended": [], "optional": []})
        for tier_name in ["required", "recommended", "optional"]:
            for theme in asset_themes.get(tier_name, []):
                if theme not in tiered_themes[tier_name]:
                    tiered_themes[tier_name].append(theme)
    return {
        "sensitivity_theme_library": (
            tiered_themes.get("required", [])
            + tiered_themes.get("recommended", [])
            + tiered_themes.get("optional", [])
        ),
        "sensitivity_theme_tiers": tiered_themes,
    }


def get_benchmark_overlay_library(transaction_profile):
    return {
        "benchmark_term_library": get_asset_benchmark_terms(transaction_profile),
        "benchmark_whitelist": [
            {
                "source_name": "Asset-class benchmark overlay",
                "allowed_use": "Use only where retrieved benchmark evidence matches the inferred asset class and metric category.",
            }
        ],
    }


def dedupe_config_list(config_items, key_name):
    output = []
    seen = set()
    for item in config_items:
        item_key = item.get(key_name) if isinstance(item, dict) else None
        if item_key and item_key in seen:
            continue
        if item_key:
            seen.add(item_key)
        output.append(item)
    return output


def finalize_review_config(config_pack):
    config_pack["asset_class_risk_taxonomy"] = dedupe_config_list(config_pack.get("asset_class_risk_taxonomy", []), "risk_id")
    config_pack["common_ic_questions"] = dedupe_config_list(config_pack.get("common_ic_questions", []), "question_id")
    config_pack["benchmark_whitelist"] = dedupe_config_list(config_pack.get("benchmark_whitelist", []), "source_name")
    return config_pack


def checklist_item_is_applicable(item, transaction_profile):
    asset_classes = transaction_profile.get("asset_classes") or [transaction_profile.get("primary_asset_class") or transaction_profile.get("asset_class")]
    revenue_models = transaction_profile.get("revenue_models") or [transaction_profile.get("revenue_model")]
    applicable_asset_classes = item.get("applicable_asset_classes", [])
    applicable_revenue_models = item.get("applicable_revenue_models", [])

    asset_applicable = not applicable_asset_classes or bool(set(asset_classes).intersection(set(applicable_asset_classes)))
    revenue_applicable = not applicable_revenue_models or bool(set(revenue_models).intersection(set(applicable_revenue_models)))
    return asset_applicable and revenue_applicable


def apply_checklist_applicability_defaults(checklist):
    for item in checklist:
        item.setdefault("applicable_asset_classes", [])
        item.setdefault("applicable_revenue_models", [])
        item.setdefault("not_applicable_status", STATUS_GREY)
    return checklist


def load_review_config(transaction_profile=None):
    transaction_profile = transaction_profile or {}
    profile_terms = get_profile_terms(transaction_profile)

    completeness_checklist = [
        {
            "item_id": "C01_APPROVAL_REQUEST",
            "required_item": "Clear approval request and transaction overview",
            "evidence_terms": ["approval", "approve progression", "requested approval", "transaction overview"],
            "weak_terms": [],
        },
        {
            "item_id": "C02_FINANCIAL_METRICS",
            "required_item": "Core financial metrics and return case",
            "evidence_terms": ["project irr", "equity irr", "npv", "minimum dscr", "ev / ebitda"],
            "weak_terms": ["metrics differ", "different values", "reconciliation"],
        },
        {
            "item_id": "C03_REVENUE_OFFTAKE",
            "required_item": "Revenue model, offtake and market assumptions",
            "evidence_terms": ["revenue", "offtake", "ppa", "merchant", "cfd", "tolling", "capacity payment", "regulated", "hydrogen offtake"],
            "weak_terms": ["not appended", "under negotiation", "no independent", "referenced but not included"],
        },
        {
            "item_id": "C04_SENSITIVITY_ANALYSIS",
            "required_item": "Sensitivity analysis and downside cases",
            "evidence_terms": ["sensitivity", "downside", "capex", "cod delay", "combined downside", "price downside", "availability", "utilisation"],
            "weak_terms": ["not shown", "not included", "absent from submitted documents"],
        },
        {
            "item_id": "C05_RISK_REGISTER",
            "required_item": "Risk register / principal risks and mitigants",
            "evidence_terms": ["principal risks", "risk", "mitigation", "mitigants", "rating"],
            "weak_terms": ["not assessed", "under negotiation", "absent", "not modelled"],
        },
        {
            "item_id": "C06_MARKET_PRICE_SUPPORT",
            "required_item": "Independent market price / revenue assumption support",
            "evidence_terms": ["price curve", "market price", "price forecast", "independent price", "revenue assumption"],
            "weak_terms": ["not appended", "not included", "missing", "independent curve absent"],
            "applicable_revenue_models": ["merchant", "hybrid_contract_merchant"],
        },
        {
            "item_id": "C07_OFFTAKER_CREDIT",
            "required_item": "Offtaker credit and payment-security evidence",
            "evidence_terms": ["offtaker", "credit", "payment security", "letter of credit"],
            "weak_terms": ["under discussion", "under negotiation", "not executed", "no independent credit"],
            "applicable_revenue_models": ["ppa", "cfd", "hydrogen_offtake"],
        },
        {
            "item_id": "C08_ESG_LEGAL_REGULATORY",
            "required_item": "ESG, legal and regulatory review",
            "evidence_terms": ["esg", "legal", "regulatory", "environmental", "social"],
            "weak_terms": ["open items", "remain open", "updates", "not assessed"],
        },
        {
            "item_id": "C09_LAND_PERMITTING",
            "required_item": "Land, easements and permits",
            "evidence_terms": ["land", "easement", "permit", "permitting"],
            "weak_terms": ["open", "remain open", "final easements", "updates"],
            "applicable_asset_classes": ["solar_pv", "onshore_wind", "offshore_wind", "bess", "hydrogen"],
        },
        {
            "item_id": "C10_CYBERSECURITY",
            "required_item": "Cybersecurity / EMS / SCADA assessment",
            "evidence_terms": ["cyber", "cybersecurity", "ems", "scada"],
            "weak_terms": ["not assessed", "no dedicated assessment", "no dedicated ems/scada"],
        },
        {
            "item_id": "C11_DECOMMISSIONING",
            "required_item": "Decommissioning / end-of-life reserve",
            "evidence_terms": ["decommissioning", "end-of-life", "disposal", "dismantling", "restoration"],
            "weak_terms": ["not assessed", "no funded", "no reserve"],
            "applicable_asset_classes": ["solar_pv", "onshore_wind", "offshore_wind", "bess", "hydrogen"],
        },
        {
            "item_id": "C12_BENCHMARK_SUPPORT",
            "required_item": "Benchmark support for costs / valuation",
            "evidence_terms": ["benchmark", "upper quartile", "high", "cost per mw", "overnight capital cost"],
            "weak_terms": ["client-provided", "not independent", "not available", "not evidenced"],
        },
        {
            "item_id": "C13_CONDITIONS_PRECEDENT",
            "required_item": "Conditions precedent / open items before final approval",
            "evidence_terms": ["conditions", "open items", "final documentation", "final approval", "approve progression"],
            "weak_terms": ["subject to", "require", "close", "obtain", "reconcile"],
        },
    ]
    completeness_checklist.extend(
        [
            {
                "item_id": "C14_REG_RECOVERY_APPROVAL",
                "required_item": "Regulatory recovery approval",
                "evidence_terms": ["regulatory approval", "recovery approval", "approved recovery", "regulator approval"],
                "weak_terms": ["pending", "not approved", "under review", "not final"],
                "applicable_asset_classes": ["regulated_grid_distribution"],
                "applicable_revenue_models": ["regulated_grid_distribution", "regulated_return"],
            },
            {
                "item_id": "C15_RAB_ALLOWED_REVENUE_TARIFF",
                "required_item": "RAB / allowed revenue / tariff treatment",
                "evidence_terms": ["rab", "regulated asset base", "allowed revenue", "tariff treatment", "distribution tariff"],
                "weak_terms": ["not final", "not approved", "not quantified", "pending"],
                "applicable_asset_classes": ["regulated_grid_distribution"],
                "applicable_revenue_models": ["regulated_grid_distribution", "regulated_return"],
            },
            {
                "item_id": "C16_RECOVERY_LAG_TRUE_UP",
                "required_item": "Recovery lag / true-up mechanism",
                "evidence_terms": ["recovery lag", "true-up", "true up", "pass-through", "regulatory lag"],
                "weak_terms": ["not modelled", "not quantified", "unclear", "pending"],
                "applicable_asset_classes": ["regulated_grid_distribution"],
                "applicable_revenue_models": ["regulated_grid_distribution", "regulated_return"],
            },
            {
                "item_id": "C17_VENDOR_EPC_PRICING_LOCK",
                "required_item": "Vendor / EPC pricing lock",
                "evidence_terms": ["vendor", "epc", "pricing lock", "fixed price", "locked pricing", "supplier"],
                "weak_terms": ["not fixed", "not locked", "under negotiation", "pending"],
                "applicable_asset_classes": ["regulated_grid_distribution"],
            },
            {
                "item_id": "C18_CYBER_OT_ASSESSMENT",
                "required_item": "Cyber / OT assessment",
                "evidence_terms": ["cyber", "ot", "operational technology", "scada", "ems", "cybersecurity"],
                "weak_terms": ["not assessed", "no dedicated assessment", "pending", "not completed"],
                "applicable_asset_classes": ["regulated_grid_distribution"],
            },
            {
                "item_id": "C19_IMPLEMENTATION_ROLLOUT_RISK",
                "required_item": "Implementation rollout risk",
                "evidence_terms": ["rollout", "implementation", "deployment", "milestones", "schedule risk"],
                "weak_terms": ["delayed", "not final", "pending", "not quantified"],
                "applicable_asset_classes": ["regulated_grid_distribution"],
            },
            {
                "item_id": "C20_BENEFIT_REALISATION",
                "required_item": "Benefit-realisation evidence",
                "evidence_terms": ["benefit realisation", "benefit realization", "loss reduction", "efficiency", "customer benefit"],
                "weak_terms": ["not quantified", "not evidenced", "not tracked", "unclear"],
                "applicable_asset_classes": ["regulated_grid_distribution"],
            },
        ]
    )
    completeness_checklist = apply_checklist_applicability_defaults(completeness_checklist)

    strategy_criteria = [
        {
            "criterion_id": "S01_ENERGY_SCOPE",
            "criterion": "Investment is within configured energy-sector scope",
            "configured": True,
            "positive_terms": ["energy", "power", "renewable", "solar", "wind", "storage", "battery", "bess", "hydrogen"] + profile_terms.split(),
            "negative_terms": [],
        },
        {
            "criterion_id": "S02_TECHNOLOGY_SCOPE",
            "criterion": "Technology matches target technologies",
            "configured": True,
            "positive_terms": ["solar", "pv", "wind", "offshore", "onshore", "bess", "battery", "storage", "hydrogen", "electrolyser"] + profile_terms.split(),
            "negative_terms": [],
        },
        {
            "criterion_id": "S03_GEOGRAPHY_PRIORITY",
            "criterion": "Geography is within configured priority geographies",
            "configured": False,
            "positive_terms": [],
            "negative_terms": [],
        },
        {
            "criterion_id": "S04_CONTROL_RIGHTS",
            "criterion": "Ownership / governance structure is aligned with control criteria",
            "configured": False,
            "positive_terms": ["80%", "controlling", "control rights", "directors"],
            "negative_terms": [],
        },
        {
            "criterion_id": "S05_PORTFOLIO_CONCENTRATION",
            "criterion": "Portfolio concentration by geography/counterparty is assessed",
            "configured": True,
            "positive_terms": ["portfolio concentration", "counterparty concentration", "geography", "counterparty"],
            "negative_terms": ["does not quantify", "not quantify", "not assessed"],
        },
        {
            "criterion_id": "S06_RETURN_HURDLE",
            "criterion": "Return hurdle / capital allocation threshold is evidenced",
            "configured": False,
            "positive_terms": ["hurdle", "return hurdle", "capital allocation"],
            "negative_terms": [],
        },
    ]

    common_ic_questions = [
        {
            "question_id": "ICQ01",
            "theme": "Revenue basis",
            "question": "Is the revenue basis supported by executed contracts, regulation, market evidence or independently validated assumptions?",
            "coverage_terms": ["revenue", "offtake", "ppa", "merchant", "cfd", "tolling", "capacity payment", "regulated"],
            "weak_terms": ["not appended", "independent curve absent", "not included", "missing"],
        },
        {
            "question_id": "ICQ02",
            "theme": "Offtaker credit",
            "question": "Has the offtaker's creditworthiness and payment-security package been independently assessed?",
            "coverage_terms": ["offtaker", "credit", "payment security", "letter of credit"],
            "weak_terms": ["under discussion", "under negotiation", "not executed", "no independent"],
        },
        {
            "question_id": "ICQ03",
            "theme": "Downside protection",
            "question": "Has the downside case combined the most material project risks rather than isolated one-factor sensitivities?",
            "coverage_terms": ["combined downside", "sensitivity", "price", "generation", "availability", "capex", "cod delay", "fx"],
            "weak_terms": ["not included", "not shown", "absent"],
        },
        {
            "question_id": "ICQ04",
            "theme": "Technology performance",
            "question": "Are asset performance assumptions explicitly modelled and supported by technical or warranty evidence?",
            "coverage_terms": ["degradation", "availability", "warranty", "resource", "utilisation", "performance"],
            "weak_terms": ["not separately modelled", "not included", "not shown", "term sheet"],
        },
        {
            "question_id": "ICQ05",
            "theme": "Grid cost and schedule",
            "question": "Is grid interconnection cost and schedule confirmed by the utility, and is contingency adequate?",
            "coverage_terms": ["grid", "interconnection", "utility", "schedule", "contingency"],
            "weak_terms": ["ongoing", "under review", "confirmation", "not confirmed"],
        },
        {
            "question_id": "ICQ06",
            "theme": "IRR bridge",
            "question": "Is there an IRR bridge explaining key value drivers versus seller / management assumptions?",
            "coverage_terms": ["irr bridge", "seller", "management assumptions", "business case"],
            "weak_terms": ["not included", "missing", "not evidenced"],
        },
        {
            "question_id": "ICQ07",
            "theme": "Accounting impact / book value",
            "question": "Are accounting impact, book values and consolidation implications addressed?",
            "coverage_terms": ["accounting", "book value", "consolidation"],
            "weak_terms": ["not included", "missing", "not evidenced"],
        },
    ]

    asset_class_risk_taxonomy = [
        {
            "risk_id": "R01_MERCHANT_EXPOSURE",
            "risk": "Revenue model support",
            "asset_class": transaction_profile.get("asset_class", "dynamic"),
            "terms": ["revenue", "merchant", "price", "offtake", "ppa", "cfd", "tolling", "capacity payment", "regulated"],
            "reviewer_challenge": "Is revenue supported by appropriate contract, regulation or independent market evidence?",
        },
        {
            "risk_id": "R02_GRID_COST_SCHEDULE",
            "risk": "Grid cost and schedule",
            "asset_class": transaction_profile.get("asset_class", "dynamic"),
            "terms": ["grid cost and schedule", "grid cost", "grid interconnection", "grid connection", "utility design"],
            "reviewer_challenge": "Is the grid interconnection scope, cost and schedule confirmed by the utility?",
        },
        {
            "risk_id": "R03_TECHNICAL_PERFORMANCE",
            "risk": "Technical performance and availability",
            "asset_class": transaction_profile.get("asset_class", "dynamic"),
            "terms": ["performance", "availability", "degradation", "resource", "utilisation", "warranty"],
            "reviewer_challenge": "Are technical performance assumptions independently supported and stress-tested?",
        },
        {
            "risk_id": "R04_OFFTAKER_CREDIT",
            "risk": "Offtaker credit and payment security",
            "asset_class": transaction_profile.get("asset_class", "dynamic"),
            "terms": ["offtaker credit", "offtaker", "payment security", "letter of credit", "counterparty"],
            "reviewer_challenge": "Has offtaker creditworthiness and payment security been independently verified?",
        },
        {
            "risk_id": "R05_LAND_PERMITTING",
            "risk": "Land, easements and permits",
            "asset_class": transaction_profile.get("asset_class", "dynamic"),
            "terms": ["land and permits", "land", "easement", "permit", "permitting"],
            "reviewer_challenge": "Are all land rights, easements and construction permits fully closed before approval?",
        },
        {
            "risk_id": "R06_CYBERSECURITY",
            "risk": "Cybersecurity / EMS / SCADA",
            "asset_class": transaction_profile.get("asset_class", "dynamic"),
            "terms": ["cybersecurity", "cyber", "ems", "scada"],
            "reviewer_challenge": "Has a dedicated EMS / SCADA cybersecurity assessment been completed?",
        },
        {
            "risk_id": "R07_DECOMMISSIONING",
            "risk": "Decommissioning and end-of-life obligations",
            "asset_class": transaction_profile.get("asset_class", "dynamic"),
            "terms": ["decommissioning", "end-of-life", "battery-disposal", "battery disposal", "dismantling"],
            "reviewer_challenge": "Are end-of-life obligations and reserves reflected in the investment case?",
        },
        {
            "risk_id": "R08_LEGAL_TRANSACTION",
            "risk": "Legal transaction risk",
            "asset_class": "cross_asset",
            "terms": ["title", "change of control", "consent", "permit", "license", "material contracts", "sanctions", "aml"],
            "reviewer_challenge": "Are title, transfer consents, permits, contracts, sanctions and AML diligence complete?",
        },
        {
            "risk_id": "R09_BLIND_SPOT_RECONCILIATION",
            "risk": "Past-transaction blind spots",
            "asset_class": "cross_asset",
            "terms": ["inconsistent", "reconcile", "terminal value", "contingency", "cod", "benchmark", "missing"],
            "reviewer_challenge": "Have common IC blind spots such as unreconciled values, optimistic COD and weak contingency been challenged?",
        },
    ]

    benchmark_whitelist = [
        {
            "source_name": "02_Benchmark_and_Market_Data",
            "allowed_use": "Cost benchmarking only unless valuation-multiple evidence is specifically retrieved",
        },
        {
            "source_name": "NREL / ATB style benchmark workbook",
            "allowed_use": "Technology-specific cost, grid and O&M benchmark categories only where asset-class evidence matches the transaction profile.",
        },
    ]

    base_review_config = {
        "transaction_profile": transaction_profile,
        "module_boost_terms": get_default_module_boost_terms(transaction_profile),
        "completeness_checklist": completeness_checklist,
        "strategy_criteria": strategy_criteria,
        "common_ic_questions": common_ic_questions,
        "asset_class_risk_taxonomy": asset_class_risk_taxonomy,
        "benchmark_whitelist": benchmark_whitelist,
    }

    merged_config = merge_config_lists(
        base_review_config,
        get_legal_risk_overlay_library(transaction_profile),
        get_market_geography_overlay_library(transaction_profile),
        get_blind_spot_overlay_library(transaction_profile),
        get_asset_class_risk_overlay_library(transaction_profile),
        get_revenue_model_overlay_library(transaction_profile),
        get_sensitivity_overlay_library(transaction_profile),
        get_benchmark_overlay_library(transaction_profile),
    )
    merged_config["module_boost_terms"] = get_default_module_boost_terms(transaction_profile)

    return finalize_review_config(merged_config)


# ---------------------------------------------------------------------
# Retrieval plan and transaction profile
# ---------------------------------------------------------------------

def get_profile_detection_terms():
    return {
        "asset_class": {
            "solar_pv": ["solar", "pv", "photovoltaic", "module", "inverter"],
            "onshore_wind": ["onshore wind", "wind farm", "turbine", "wake loss"],
            "offshore_wind": ["offshore wind", "foundation", "export cable", "seabed", "marine installation"],
            "bess": ["bess", "battery", "storage", "augmentation", "cycling", "ancillary services"],
            "hydrogen": ["hydrogen", "electrolyser", "electrolyzer", "lcoh", "ammonia", "green hydrogen"],
            "regulated_grid_distribution": [
                "addc",
                "distribution company",
                "regulated grid",
                "grid distribution",
                "distribution network",
                "grid modernisation",
                "grid modernization",
                "regulated asset base",
                "allowed revenue",
            ],
        },
        "energy_value_chain": {
            "upstream": ["development", "site control", "resource assessment", "permitting", "land"],
            "midstream": ["storage", "transmission", "grid", "pipeline", "transport", "compression"],
            "downstream": ["offtake", "ppa", "customer", "retail", "distribution", "sale"],
            "integrated": ["integrated", "platform", "portfolio", "generation and storage", "hybrid"],
        },
        "project_stage": {
            "development": ["development", "late-stage development", "pre-fid", "permitting"],
            "construction": ["construction", "epc", "notice to proceed", "under construction"],
            "operating": ["operating", "operational", "commercial operation", "cod achieved"],
            "expansion": ["expansion", "repowering", "augmentation", "phase ii"],
            "acquisition": ["acquisition", "stake", "shares", "purchase", "seller"],
        },
        "ownership": {
            "minority": ["minority", "less than 50%", "non-controlling"],
            "majority": ["majority", "more than 50%"],
            "control": ["control", "controlling", "reserved matters", "board control"],
            "platform": ["platform", "portfolio company", "holdco"],
        },
        "revenue_model": {
            "ppa": ["ppa", "power purchase agreement"],
            "merchant": ["merchant", "merchant exposure", "merchant price"],
            "hybrid_contract_merchant": ["contracted share", "merchant share", "hybrid revenue"],
            "cfd": ["cfd", "contract for difference"],
            "tolling": ["tolling", "tolling agreement"],
            "capacity_payment": ["capacity payment", "capacity revenue"],
            "regulated_return": ["regulated return", "tariff", "regulated asset base"],
            "regulated_grid_distribution": ["allowed revenue", "regulated asset base", "distribution tariff", "regulated grid"],
            "hydrogen_offtake": ["hydrogen offtake", "hydrogen sale", "ammonia offtake"],
        },
        "contract_type": {
            "ppa": ["ppa", "power purchase agreement"],
            "cfd": ["cfd", "contract for difference"],
            "tolling": ["tolling"],
            "offtake": ["offtake"],
            "concession": ["concession"],
            "regulated": ["regulated", "tariff"],
            "merchant": ["merchant"],
            "mixed": ["contracted share", "merchant share", "hybrid"],
        },
    }


def infer_category(blob, category_terms, default_value="not_identified"):
    scores = {}
    for category, terms in category_terms.items():
        score = 0
        for term in terms:
            if normalize_text(term) in blob:
                score += 1
        scores[category] = score

    best_category = default_value
    best_score = 0
    for category, score in scores.items():
        if score > best_score:
            best_category = category
            best_score = score

    return best_category, best_score


def infer_categories(blob, category_terms):
    matches = []
    for category, terms in category_terms.items():
        score = 0
        for term in terms:
            if normalize_text(term) in blob:
                score += 1
        if score > 0:
            matches.append({"category": category, "score": score})

    matches = sorted(matches, key=lambda item: item["score"], reverse=True)
    return matches


def infer_geography(blob):
    country_candidates = [
        "Australia",
        "Cambodia",
        "Egypt",
        "India",
        "Jordan",
        "Saudi Arabia",
        "South Africa",
        "Thailand",
        "United Arab Emirates",
        "UAE",
    ]

    for country in country_candidates:
        if normalize_text(country) in blob:
            return country

    return "not_identified"


def infer_technology_subtypes(blob, asset_classes):
    if isinstance(asset_classes, str):
        asset_classes = [asset_classes]

    subtype_terms = {
        "solar_pv": ["utility-scale", "rooftop", "tracker", "fixed tilt", "bifacial"],
        "onshore_wind": ["onshore", "turbine", "wake losses"],
        "offshore_wind": ["fixed-bottom", "floating", "foundation", "export cable"],
        "bess": ["lithium-ion", "standalone", "co-located", "two-hour", "four-hour"],
        "hydrogen": ["electrolyser", "electrolyzer", "alkaline", "pem", "ammonia"],
    }

    output = []
    for asset_class in asset_classes:
        for term in subtype_terms.get(asset_class, []):
            if normalize_text(term) in blob and term not in output:
                output.append(term)

    return output


def infer_transaction_profile(transaction_id, profile_results):
    blob = normalize_text(get_evidence_blob(profile_results))
    detection_terms = get_profile_detection_terms()

    asset_matches = infer_categories(blob, detection_terms["asset_class"])
    revenue_matches = infer_categories(blob, detection_terms["revenue_model"])
    asset_classes = [item["category"] for item in asset_matches]
    revenue_models = [item["category"] for item in revenue_matches]

    has_solar = "solar_pv" in asset_classes
    has_bess = "bess" in asset_classes
    if has_solar and has_bess:
        primary_asset_class = "solar_pv_plus_bess"
        asset_score = max(item["score"] for item in asset_matches)
    else:
        primary_asset_class = asset_classes[0] if asset_classes else "not_identified"
        asset_score = asset_matches[0]["score"] if asset_matches else 0

    energy_value_chain, value_chain_score = infer_category(blob, detection_terms["energy_value_chain"])
    project_stage, stage_score = infer_category(blob, detection_terms["project_stage"])
    ownership, ownership_score = infer_category(blob, detection_terms["ownership"])
    revenue_model = revenue_models[0] if revenue_models else "not_identified"
    revenue_score = revenue_matches[0]["score"] if revenue_matches else 0
    contract_type, contract_score = infer_category(blob, detection_terms["contract_type"])

    detected_scores = [
        asset_score,
        value_chain_score,
        stage_score,
        ownership_score,
        revenue_score,
        contract_score,
    ]
    populated_count = len([score for score in detected_scores if score > 0])

    if asset_score > 1 and populated_count >= 4:
        classification_confidence = "high"
    elif asset_score > 0 and populated_count >= 2:
        classification_confidence = "medium"
    else:
        classification_confidence = "low"

    return {
        "transaction_id": transaction_id,
        "energy_value_chain": energy_value_chain,
        "primary_asset_class": primary_asset_class,
        "asset_class": primary_asset_class,
        "asset_classes": asset_classes,
        "technology_subtypes": infer_technology_subtypes(blob, asset_classes),
        "geography": infer_geography(blob),
        "project_stage": project_stage,
        "ownership": ownership,
        "revenue_model": revenue_model,
        "revenue_models": revenue_models,
        "contract_type": contract_type,
        "development_scope": energy_value_chain,
        "classification_confidence": classification_confidence,
        "source_chunks": [get_source_reference(item) for item in profile_results[:8]],
    }


def run_neutral_profile_retrieval(transaction_id):
    query_config = {
        "query": (
            f"{transaction_id} transaction overview asset class technology project stage geography "
            "ownership revenue model contract type offtake construction operation acquisition "
            "solar wind offshore storage battery hydrogen regulated grid distribution grid modernisation "
            "merchant ppa cfd tolling allowed revenue tariff"
        ),
        "corpus_zone": "client_data",
        "corpus_pack": transaction_id,
        "top_k": 20,
        "mode": "hybrid",
        "max_chunk_chars": 10000,
    }
    retrieval_output = run_single_retrieval(query_config)
    results = retrieval_output.get("results", [])
    selected_results = select_evidence(results, module_key="profile", top_n=12)

    return {
        "retrieval_calls": [
            {
                "query": query_config["query"],
                "corpus_zone": query_config.get("corpus_zone"),
                "corpus_pack": query_config.get("corpus_pack"),
                "results_returned": len(results),
                "status": "ok",
            }
        ],
        "raw_results_count": len(results),
        "selected_results": selected_results,
        "selected_sources": [get_source_reference(item) for item in selected_results],
        "evidence_blob": get_evidence_blob(selected_results),
    }


def build_retrieval_plan(transaction_profile, review_config):
    transaction_id = transaction_profile["transaction_id"]
    client_scope = {
        "corpus_zone": "client_data",
        "corpus_pack": transaction_id,
    }
    asset_class = clean_text(transaction_profile.get("primary_asset_class") or transaction_profile.get("asset_class"))
    asset_classes = clean_text(" ".join(transaction_profile.get("asset_classes", [])))
    revenue_model = clean_text(transaction_profile.get("revenue_model"))
    revenue_models = clean_text(" ".join(transaction_profile.get("revenue_models", [])))
    project_stage = clean_text(transaction_profile.get("project_stage"))
    geography = clean_text(transaction_profile.get("geography"))
    profile_context = clean_text(
        f"{asset_class} {asset_classes} {revenue_model} {revenue_models} {project_stage} {geography}"
    )
    risk_taxonomy_terms = clean_text(" ".join(get_risk_taxonomy_terms(review_config)))
    benchmark_terms = clean_text(" ".join(review_config.get("benchmark_term_library", [])))

    return {
        "profile": [
            {
                "query": (
                    f"{transaction_id} transaction overview asset class geography technology ownership stage "
                    "revenue model contract type project cost return metrics COD"
                ),
                **client_scope,
                "top_k": 15,
                "mode": "hybrid",
                "max_chunk_chars": 8000,
            }
        ],
        "completeness_readiness": [
            {
                "query": (
                    f"{profile_context} required submission components completeness readiness missing weak open items "
                    "conditions precedent final investment approval metric reconciliation revenue support "
                    "counterparty credit combined downside permits legal cyber decommissioning"
                ),
                **client_scope,
                "top_k": 20,
                "mode": "hybrid",
                "max_chunk_chars": 10000,
            }
        ],
        "strategy_fit": [
            {
                "query": (
                    f"{profile_context} strategy fit growth priorities geography technology value chain "
                    "portfolio concentration capital allocation control rights return hurdle"
                ),
                **client_scope,
                "top_k": 15,
                "mode": "hybrid",
                "max_chunk_chars": 8000,
            }
        ],
        "historical_ic_questions": [
            {
                "query": (
                    f"{profile_context} historical IC questions recurring IC themes construction technology risk "
                    "counterparty credit downside protection risk allocation revenue assumptions business case "
                    "seller assumptions IRR bridge accounting impact book values follow up actions"
                ),
                "corpus_zone": "corpus_data",
                "top_k": 15,
                "mode": "hybrid",
                "max_chunk_chars": 8000,
            }
        ],
        "risk_review": [
            {
                "query": (
                    f"{profile_context} principal risks mitigants risk rating revenue risk grid cost schedule "
                    "technical performance counterparty credit land permits legal cybersecurity decommissioning "
                    f"construction technology risk {risk_taxonomy_terms}"
                ),
                **client_scope,
                "top_k": 20,
                "mode": "hybrid",
                "max_chunk_chars": 10000,
            }
        ],
        "financial_reconciliation": [
            {
                "query": (
                    f"{profile_context} main financial metrics returns project cost capex project IRR equity IRR "
                    "NPV DSCR EBITDA LCOE LCOS LCOH availability utilisation metrics differ deck memo workbook"
                ),
                **client_scope,
                "top_k": 20,
                "mode": "hybrid",
                "max_chunk_chars": 12000,
            }
        ],
        "sensitivity_review": [
            {
                "query": (
                    f"{profile_context} sensitivity results downside cases price volume generation availability "
                    "resource capex COD delay FX inflation combined downside utilisation degradation included in submission"
                ),
                **client_scope,
                "top_k": 20,
                "mode": "hybrid",
                "max_chunk_chars": 12000,
            }
        ],
        "market_offtake_revenue": [
            {
                "query": (
                    f"{profile_context} market offtake revenue assumptions contract revenue merchant regulated "
                    "PPA CfD tolling capacity payment payment security counterparty credit revenue projection EBITDA"
                ),
                **client_scope,
                "top_k": 20,
                "mode": "hybrid",
                "max_chunk_chars": 12000,
            }
        ],
        "client_valuation": [
            {
                "query": (
                    f"{profile_context} valuation assumptions NPV EV EBITDA enterprise value first full year EBITDA "
                    "discount rate cost per MW capex equipment grid interconnection overhead contingency benchmark"
                ),
                **client_scope,
                "top_k": 20,
                "mode": "hybrid",
                "max_chunk_chars": 12000,
            }
        ],
        "external_cost_benchmark": [
            {
                "query": (
                    f"{profile_context} external benchmark overnight capital cost grid connection cost "
                    f"fixed operating expense variable operating expense technology cost benchmark {benchmark_terms}"
                ),
                "corpus_zone": "corpus_data",
                "top_k": 20,
                "mode": "hybrid",
                "max_chunk_chars": 12000,
            }
        ],
        "external_ev_ebitda_benchmark": [
            {
                "query": (
                    f"{profile_context} renewable energy infrastructure EV EBITDA enterprise value valuation multiple "
                    "transaction benchmark comparable transactions listed company multiples"
                ),
                "corpus_zone": "corpus_data",
                "top_k": 20,
                "mode": "hybrid",
                "max_chunk_chars": 12000,
            }
        ],
        "conditions_open_items": [
            {
                "query": (
                    f"{profile_context} conditions precedent open items final investment approval metric reconciliation "
                    "revenue support counterparty credit combined downside grid overhead advisory legal land permit cyber "
                    "decommissioning actions"
                ),
                **client_scope,
                "top_k": 20,
                "mode": "hybrid",
                "max_chunk_chars": 10000,
            }
        ],
        "previous_request_accommodation": [
            {
                "query": (
                    "previous request missing data accommodated prior IC request follow up action "
                    "previously requested included in current memo not accommodated"
                ),
                **client_scope,
                "top_k": 15,
                "mode": "hybrid",
                "max_chunk_chars": 8000,
            }
        ],
        "prior_deal_comparison": [
            {
                "query": (
                    f"{profile_context} comparable prior investments historical deal comparison similarities differences "
                    "prior energy investments platform geography stage asset class revenue model"
                ),
                "corpus_zone": "corpus_data",
                "top_k": 15,
                "mode": "hybrid",
                "max_chunk_chars": 8000,
            }
        ],
        "macro_context": [
            {
                "query": (
                    f"{geography} macroeconomic data FX projections exchange rate GDP growth inflation "
                    "energy market country risk"
                ),
                "corpus_zone": "corpus_data",
                "top_k": 15,
                "mode": "hybrid",
                "max_chunk_chars": 8000,
            }
        ],
    }


def get_retrieval_plan(transaction_id):
    profile_packet = run_neutral_profile_retrieval(transaction_id)
    transaction_profile = infer_transaction_profile(
        transaction_id,
        profile_packet.get("selected_results", []),
    )
    review_config = load_review_config(transaction_profile)
    return build_retrieval_plan(transaction_profile, review_config)


def run_single_retrieval(query_config):
    return ret.retrieve_chunks(
        query_text=query_config["query"],
        top_k=query_config.get("top_k", 10),
        mode=query_config.get("mode", "hybrid"),
        corpus_zone=query_config.get("corpus_zone"),
        corpus_pack=query_config.get("corpus_pack"),
        document_id=query_config.get("document_id"),
        max_chunk_chars=query_config.get("max_chunk_chars", 8000),
    )


def run_retrieval_plan(transaction_profile, review_config, profile_packet=None):
    retrieval_plan = build_retrieval_plan(transaction_profile, review_config)
    evidence_packets = {}

    if profile_packet is not None:
        evidence_packets["profile"] = profile_packet

    for module_key, query_configs in retrieval_plan.items():
        if module_key == "profile" and profile_packet is not None:
            continue

        module_results = []
        retrieval_calls = []

        for query_config in query_configs:
            try:
                retrieval_output = run_single_retrieval(query_config)
                results = retrieval_output.get("results", [])
                module_results.extend(results)

                retrieval_calls.append(
                    {
                        "query": query_config["query"],
                        "corpus_zone": query_config.get("corpus_zone"),
                        "corpus_pack": query_config.get("corpus_pack"),
                        "results_returned": len(results),
                        "status": "ok",
                    }
                )

            except Exception as e:
                retrieval_calls.append(
                    {
                        "query": query_config["query"],
                        "corpus_zone": query_config.get("corpus_zone"),
                        "corpus_pack": query_config.get("corpus_pack"),
                        "results_returned": 0,
                        "status": "failed",
                        "error": f"{type(e).__name__}: {str(e)}",
                    }
                )

        selected_results = select_evidence(
            results=module_results,
            module_key=module_key,
            top_n=12,
            review_config=review_config,
        )

        evidence_packets[module_key] = {
            "retrieval_calls": retrieval_calls,
            "raw_results_count": len(module_results),
            "selected_results": selected_results,
            "selected_sources": [get_source_reference(item) for item in selected_results],
            "evidence_blob": get_evidence_blob(selected_results),
        }

    return evidence_packets


# ---------------------------------------------------------------------
# Transaction profile
# ---------------------------------------------------------------------

def identify_transaction_profile(transaction_id, evidence_packets):
    profile_evidence = evidence_packets.get("profile", {}).get("selected_results", [])
    return infer_transaction_profile(transaction_id, profile_evidence)


# ---------------------------------------------------------------------
# Table row parsing utilities
# ---------------------------------------------------------------------

def extract_row_texts(chunk_text):
    """
    Returns row text strings without the leading 'Row n:' label.
    Handles rows embedded in one long chunk.
    """

    chunk_text = clean_text(chunk_text)
    pattern = r"Row\s+\d+:\s*(.*?)(?=\s+Row\s+\d+:|$)"
    rows = re.findall(pattern, chunk_text, flags=re.IGNORECASE)

    return [clean_text(row) for row in rows]


def parse_table_row_cells(row_text):
    """
    Parses rows like:
    0: Total project cost | 1: 289.0 | 2: USD mn
    Unnamed: 0: Total project cost | Unnamed: 1: 289.0 | Unnamed: 2: USD mn

    Returns:
    [
      {"label": "0", "value": "Total project cost"},
      {"label": "1", "value": "289.0"}
    ]
    """

    cells = []
    parts = [part.strip() for part in row_text.split("|")]

    for part in parts:
        match = re.match(r"(?:Unnamed:\s*)?([^:]+):\s*(.*)$", part, flags=re.IGNORECASE)

        if match:
            label = clean_text(match.group(1))
            value = clean_text(match.group(2))
        else:
            label = ""
            value = clean_text(part)

        cells.append({"label": label, "value": value})

    return cells


def canonical_structured_label(label):
    label_norm = normalize_text(label)
    label_norm = re.sub(r"[^a-z0-9]+", "_", label_norm).strip("_")
    mapping = {
        "assumption_name": "assumption_name",
        "assumption": "assumption_name",
        "metric": "metric",
        "metric_name": "metric",
        "value": "value",
        "unit": "unit",
        "case": "case",
        "scenario": "case",
        "year": "year",
        "section": "section",
        "basis_or_commentary": "basis_or_commentary",
        "basis": "basis_or_commentary",
        "commentary": "basis_or_commentary",
        "comment": "basis_or_commentary",
    }
    return mapping.get(label_norm)


def extract_structured_key_value_row(row_text):
    structured = {}
    cells = parse_table_row_cells(row_text)

    for cell in cells:
        label = clean_text(cell.get("label"))
        value = clean_text(cell.get("value"))
        canonical_label = canonical_structured_label(label)

        if canonical_label and value:
            structured[canonical_label] = value
            continue

        embedded_match = re.match(r"([^:|]+):\s*(.+)$", value)
        if embedded_match:
            embedded_label = canonical_structured_label(embedded_match.group(1))
            if embedded_label:
                structured[embedded_label] = clean_text(embedded_match.group(2))

    return structured


def cell_values_from_row(row_text):
    cells = parse_table_row_cells(row_text)
    return [cell["value"] for cell in cells]


def get_cell_value_by_label(row_text, label_candidates):
    """Return the value for the first matching parsed table label."""

    cells = parse_table_row_cells(row_text)
    normalized_candidates = [normalize_text(label) for label in label_candidates]

    for cell in cells:
        label = normalize_text(cell.get("label"))

        for candidate in normalized_candidates:
            if label == candidate or candidate in label:
                return clean_text(cell.get("value"))

    return None


def row_source_is_table(result):
    source_reference = normalize_text(result.get("source_reference"))
    section_heading = normalize_text(result.get("section_heading"))
    return "extracted_table_rows" in source_reference or section_heading.startswith("table:")


def first_number_in_text(text_value):
    match = re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?", clean_text(text_value))
    if match:
        return safe_float(match.group(0))
    return None


# ---------------------------------------------------------------------
# Completeness and readiness
# ---------------------------------------------------------------------

def run_completeness_check(review_config, evidence_packets, financial_reconciliation=None, sensitivity_review=None):
    checklist = review_config["completeness_checklist"]
    transaction_profile = review_config.get("transaction_profile", {})
    evidence_results = evidence_packets.get("completeness_readiness", {}).get("selected_results", [])
    blob = get_evidence_blob(evidence_results)

    external_cost_blob = evidence_packets.get("external_cost_benchmark", {}).get("evidence_blob", "")
    external_ev_blob = evidence_packets.get("external_ev_ebitda_benchmark", {}).get("evidence_blob", "")
    benchmark_terms = review_config.get("benchmark_term_library", [])
    asset_benchmark_match = bool(benchmark_terms) and has_any(external_cost_blob, benchmark_terms)
    generic_benchmark_match = has_any(
        external_cost_blob,
        [
            "overnight capital cost",
            "grid connection cost",
            "fixed operating expenses",
            "variable operating expense",
            "capex benchmark",
            "cost benchmark",
        ],
    )

    external_cost_available = asset_benchmark_match and generic_benchmark_match

    external_ev_available = has_all(external_ev_blob, ["ev", "ebitda"]) and has_any(
        external_ev_blob,
        ["enterprise value", "valuation multiple", "transaction benchmark", "comparable transaction", "listed company"],
    )

    rows = []

    for item in checklist:
        if not checklist_item_is_applicable(item, transaction_profile):
            rows.append(
                {
                    "item_id": item["item_id"],
                    "required_item": item["required_item"],
                    "status": READINESS_NOT_ASSESSED,
                    "traffic_light": item.get("not_applicable_status", STATUS_GREY),
                    "evidence_found": False,
                    "weakness": "Checklist item is not applicable to the inferred asset class / revenue model.",
                    "evidence_text": "",
                    "source_chunks": [],
                    "recommended_action": "No action required unless reviewer determines this item is applicable.",
                }
            )
            continue

        positive_matches, weak_matches = split_evidence_by_terms(
            results=evidence_results,
            positive_terms=item["evidence_terms"],
            weak_terms=item["weak_terms"],
        )
        evidence_found = bool(positive_matches or weak_matches)
        weak_evidence = bool(weak_matches)

        if item["item_id"] == "C02_FINANCIAL_METRICS" and financial_reconciliation:
            inherited_status = financial_reconciliation.get("traffic_light")
            if inherited_status == STATUS_RED:
                status = READINESS_WEAK
                traffic_light = STATUS_RED
                weakness = "Financial reconciliation has Red metric conflicts."
                evidence_found = True
            elif inherited_status == STATUS_AMBER:
                status = READINESS_WEAK
                traffic_light = STATUS_AMBER
                weakness = "Financial reconciliation has Amber metric issues."
                evidence_found = True
            elif evidence_found:
                status = READINESS_PASS
                traffic_light = STATUS_GREEN
                weakness = "Financial metric evidence found and no higher-severity reconciliation issue was reported."
            else:
                status = READINESS_MISSING
                traffic_light = STATUS_RED
                weakness = "Financial metrics not found in retrieved submission materials."
        elif item["item_id"] == "C04_SENSITIVITY_ANALYSIS" and sensitivity_review:
            inherited_status = sensitivity_review.get("traffic_light")
            if sensitivity_review.get("validation_failures"):
                status = READINESS_WEAK
                traffic_light = STATUS_RED if inherited_status == STATUS_RED else STATUS_AMBER
                weakness = "Required sensitivity scenarios are missing or incomplete."
                evidence_found = True
            elif inherited_status == STATUS_AMBER:
                status = READINESS_WEAK
                traffic_light = STATUS_AMBER
                weakness = "Sensitivity review reports Amber gaps."
                evidence_found = True
            elif evidence_found:
                status = READINESS_PASS
                traffic_light = STATUS_GREEN
                weakness = "Sensitivity evidence found and required scenarios are not flagged as missing."
            else:
                status = READINESS_MISSING
                traffic_light = STATUS_RED
                weakness = "Sensitivity evidence not found in retrieved submission materials."
        elif item["item_id"] == "C12_BENCHMARK_SUPPORT":
            if external_cost_available and external_ev_available:
                status = READINESS_PASS
                traffic_light = STATUS_GREEN
                weakness = "External cost and EV/EBITDA benchmark evidence found."
                evidence_found = True
            elif external_cost_available and not external_ev_available:
                status = READINESS_WEAK
                traffic_light = STATUS_AMBER
                weakness = "External cost benchmark evidence is available, but external EV/EBITDA valuation multiple evidence is not available."
                evidence_found = True
            elif generic_benchmark_match and not asset_benchmark_match:
                status = READINESS_WEAK
                traffic_light = STATUS_AMBER
                weakness = "External benchmark corpus evidence exists, but it does not match the inferred asset class."
                evidence_found = True
            elif evidence_found:
                status = READINESS_WEAK
                traffic_light = STATUS_AMBER
                weakness = "Client-side benchmark assertions are evidenced, but matching external benchmark corpus evidence is unavailable."
            else:
                status = READINESS_MISSING
                traffic_light = STATUS_RED
                weakness = "Client failed to provide benchmark evidence and no matching external benchmark corpus evidence was found."
        else:
            if not evidence_found:
                status = READINESS_MISSING
                traffic_light = STATUS_RED
                weakness = "Required evidence not found in retrieved submission materials."
            elif weak_evidence:
                status = READINESS_WEAK
                traffic_light = STATUS_AMBER
                weakness = "Evidence found, but weakness / open issue language is present."
            else:
                status = READINESS_PASS
                traffic_light = STATUS_GREEN
                weakness = "No obvious weakness language detected in retrieved evidence."

        evidence_text = ""
        matched_results = positive_matches + weak_matches
        if matched_results:
            evidence_text = clean_text(matched_results[0].get("chunk_text"))[:700]

        source_chunks = [get_source_reference(result) for result in matched_results[:5]]

        if item["item_id"] == "C12_BENCHMARK_SUPPORT" and external_cost_available:
            external_sources = evidence_packets.get("external_cost_benchmark", {}).get("selected_sources", [])
            source_chunks = source_chunks + external_sources[:3]

        if status == READINESS_PASS:
            recommended_action = "No immediate action identified from retrieved evidence."
        elif status == READINESS_WEAK:
            recommended_action = "Reviewer should request clarification / supporting evidence before final approval."
        else:
            recommended_action = "Reviewer should request missing item or confirm whether it is outside the current submission."

        rows.append(
            {
                "item_id": item["item_id"],
                "required_item": item["required_item"],
                "status": status,
                "traffic_light": traffic_light,
                "evidence_found": evidence_found,
                "weakness": weakness,
                "evidence_text": evidence_text,
                "source_chunks": source_chunks,
                "recommended_action": recommended_action,
            }
        )

    module_status = STATUS_GREEN

    if any(row["traffic_light"] == STATUS_RED for row in rows):
        module_status = STATUS_RED
    elif any(row["traffic_light"] == STATUS_AMBER for row in rows):
        module_status = STATUS_AMBER

    return {
        "module": "Completeness & Readiness Check",
        "traffic_light": module_status,
        "summary": "Submission readiness assessed using deterministic checklist and retrieved evidence.",
        "checklist": rows,
    }


# ---------------------------------------------------------------------
# Strategy and fit assessment
# ---------------------------------------------------------------------

def run_strategy_fit_assessment(review_config, evidence_packets):
    criteria = review_config["strategy_criteria"]
    evidence_results = evidence_packets.get("strategy_fit", {}).get("selected_results", [])
    blob = get_evidence_blob(evidence_results)

    rows = []

    for criterion in criteria:
        if not criterion["configured"]:
            status = READINESS_NOT_ASSESSED
            traffic_light = STATUS_GREY
            finding = "Criterion is not configured for this POC run."
            source_chunks = []
        else:
            positive_found = has_any(blob, criterion["positive_terms"])
            negative_found = has_any(blob, criterion["negative_terms"])

            if positive_found and not negative_found:
                status = READINESS_PASS
                traffic_light = STATUS_GREEN
                finding = "Alignment evidence found."
            elif positive_found and negative_found:
                status = READINESS_PARTIAL
                traffic_light = STATUS_AMBER
                finding = "Some alignment evidence found, but unresolved issue / limitation language is present."
            else:
                status = READINESS_NOT_ASSESSED
                traffic_light = STATUS_GREY
                finding = "Alignment could not be assessed from retrieved evidence."

            source_chunks = find_source_chunks(
                results=evidence_results,
                terms=criterion["positive_terms"] + criterion["negative_terms"],
                max_sources=5,
            )

        rows.append(
            {
                "criterion_id": criterion["criterion_id"],
                "criterion": criterion["criterion"],
                "configured": criterion["configured"],
                "status": status,
                "traffic_light": traffic_light,
                "finding": finding,
                "source_chunks": source_chunks,
            }
        )

    if any(row["traffic_light"] == STATUS_RED for row in rows):
        module_status = STATUS_RED
    elif any(row["traffic_light"] == STATUS_AMBER for row in rows):
        module_status = STATUS_AMBER
    elif any(row["traffic_light"] == STATUS_GREY for row in rows):
        module_status = STATUS_AMBER
    else:
        module_status = STATUS_GREEN

    return {
        "module": "Strategy & Fit Assessment",
        "traffic_light": module_status,
        "summary": "Strategy fit is assessed using explicit configured rules. Unconfigured criteria are marked Grey.",
        "criteria_assessment": rows,
    }


# ---------------------------------------------------------------------
# Historical IC question coverage
# ---------------------------------------------------------------------

def run_historical_ic_question_check(review_config, evidence_packets):
    questions = review_config["common_ic_questions"]
    evidence_results = evidence_packets.get("historical_ic_questions", {}).get("selected_results", [])

    current_submission_results = []
    for module_key in [
        "risk_review",
        "financial_reconciliation",
        "sensitivity_review",
        "market_offtake_revenue",
        "conditions_open_items",
    ]:
        current_submission_results.extend(evidence_packets.get(module_key, {}).get("selected_results", []))

    historical_blob = get_evidence_blob(evidence_results)
    current_blob = get_evidence_blob(current_submission_results)

    historical_logs_available = has_any(
        historical_blob,
        ["ic question", "investment committee", "follow-up action", "historical ic", "past ic"],
    )

    rows = []

    for question in questions:
        coverage_found = has_any(current_blob, question["coverage_terms"])
        weak_found = has_any(current_blob, question["weak_terms"])

        if coverage_found and not weak_found:
            coverage_status = "Fully addressed"
            traffic_light = STATUS_GREEN
            source_chunks = find_source_chunks(
                results=current_submission_results,
                terms=question["coverage_terms"],
                max_sources=5,
            )
            negative_evidence_chunks = []
        elif coverage_found and weak_found:
            coverage_status = "Partially addressed"
            traffic_light = STATUS_AMBER
            source_chunks = find_source_chunks(
                results=current_submission_results,
                terms=question["coverage_terms"] + question["weak_terms"],
                max_sources=5,
            )
            negative_evidence_chunks = find_source_chunks(
                results=current_submission_results,
                terms=question["weak_terms"],
                max_sources=3,
            )
        else:
            coverage_status = "Not addressed"
            traffic_light = STATUS_RED
            source_chunks = []
            negative_evidence_chunks = []

        rows.append(
            {
                "question_id": question["question_id"],
                "theme": question["theme"],
                "likely_ic_question": question["question"],
                "coverage_status": coverage_status,
                "traffic_light": traffic_light,
                "source_chunks": source_chunks,
                "negative_evidence_chunks": negative_evidence_chunks,
            }
        )

    return {
        "module": "Historical IC Question Coverage",
        "traffic_light": STATUS_AMBER,
        "summary": "Historical IC logs are not evidenced; standard IC question library is used for this POC run.",
        "historical_logs_available": historical_logs_available,
        "limitation": (
            "Historical IC Q&A logs / follow-up actions were not clearly evidenced in the current corpus. "
            "For this POC run, the tool uses a standard IC question library and does not claim to have learned "
            "from actual historical IC interactions."
        ),
        "question_coverage": rows,
    }


# ---------------------------------------------------------------------
# Financial metric extraction and reconciliation
# ---------------------------------------------------------------------

def metric_value_is_plausible(metric_key, value, transaction_profile=None):
    if value is None:
        return False
    transaction_profile = transaction_profile or {}
    asset_class = normalize_text(transaction_profile.get("asset_class"))

    if metric_key == "total_capex":
        if asset_class == "offshore_wind":
            return 100 <= value <= 20000
        if asset_class == "hydrogen":
            return 10 <= value <= 15000
        return 100 <= value <= 2000

    if metric_key == "project_irr_pct":
        return 3 <= value <= 50

    if metric_key == "equity_irr_pct":
        return 3 <= value <= 60

    if metric_key == "npv":
        return -1000 <= value <= 1000 and abs(value) >= 3

    if metric_key == "minimum_dscr_x":
        return 0.5 <= value <= 3.5

    if metric_key == "ev_ebitda_x":
        return 2.5 <= value <= 30

    if metric_key == "debt_total_cost_pct":
        return 0 <= value <= 100

    if metric_key == "year_one_revenue":
        if asset_class in ["offshore_wind", "hydrogen"]:
            return 1 <= value <= 5000
        return 1 <= value <= 1000

    if metric_key == "offtake_price":
        if asset_class == "hydrogen":
            return 0.1 <= value <= 10000
        return 1 <= value <= 500

    if metric_key == "capacity_factor_pct":
        return 5 <= value <= 70

    return True


def get_metric_definitions(transaction_profile=None):
    return [
        {
            "metric_key": "total_capex",
            "metric_name": "Total capex / project cost",
            "terms": ["total project cost", "total capex", "capex"],
            "unit": None,
            "direct_patterns": [
                r"(?:total project cost|total capex|capex)[^0-9]{0,50}(?:[a-z]{3}\s*)?([0-9]+(?:\.[0-9]+)?)",
            ],
            "allow_table_names": ["summary", "assumptions", "docx_table_4"],
        },
        {
            "metric_key": "project_irr_pct",
            "metric_name": "Project IRR",
            "terms": ["project irr"],
            "unit": "%",
            "direct_patterns": [r"project irr[^0-9]{0,50}([0-9]+(?:\.[0-9]+)?)"],
            "allow_table_names": ["summary", "assumptions", "docx_table_4"],
        },
        {
            "metric_key": "equity_irr_pct",
            "metric_name": "Equity IRR",
            "terms": ["equity irr"],
            "unit": "%",
            "direct_patterns": [r"equity irr[^0-9]{0,50}([0-9]+(?:\.[0-9]+)?)"],
            "allow_table_names": ["summary", "assumptions", "docx_table_4"],
        },
        {
            "metric_key": "npv",
            "metric_name": "NPV",
            "terms": ["npv"],
            "unit": None,
            "direct_patterns": [r"\bnpv\b[^0-9\-]{0,50}(?:[a-z]{3}\s*)?([-+]?[0-9]+(?:\.[0-9]+)?)"],
            "allow_table_names": ["summary", "assumptions", "docx_table_4"],
        },
        {
            "metric_key": "minimum_dscr_x",
            "metric_name": "Minimum DSCR",
            "terms": ["minimum dscr", "min_dscr", "min dscr"],
            "unit": "x",
            "direct_patterns": [r"(?:minimum dscr|min_dscr|min dscr)[^0-9]{0,50}([0-9]+(?:\.[0-9]+)?)"],
            "allow_table_names": ["summary", "assumptions", "docx_table_4"],
        },
        {
            "metric_key": "ev_ebitda_x",
            "metric_name": "EV / EBITDA",
            "terms": ["ev / ebitda", "ev ebitda", "ev/ebitda"],
            "unit": "x",
            "direct_patterns": [r"ev\s*/?\s*ebitda[^0-9]{0,50}([0-9]+(?:\.[0-9]+)?)"],
            "allow_table_names": ["summary", "assumptions", "docx_table_4"],
        },
        {
            "metric_key": "debt_total_cost_pct",
            "metric_name": "Debt / total cost",
            "terms": ["debt / total cost", "debt total cost", "debt-to-cost"],
            "unit": "%",
            "direct_patterns": [r"(?:debt\s*/\s*total cost|debt total cost|debt-to-cost)[^0-9]{0,50}([0-9]+(?:\.[0-9]+)?)"],
            "allow_table_names": ["summary", "assumptions", "docx_table_4"],
        },
        {
            "metric_key": "year_one_revenue",
            "metric_name": "Year-one revenue",
            "terms": ["year-one revenue", "year one revenue"],
            "unit": None,
            "direct_patterns": [r"(?:year-one revenue|year one revenue)[^0-9]{0,50}(?:[a-z]{3}\s*)?([0-9]+(?:\.[0-9]+)?)"],
            "allow_table_names": ["summary", "assumptions", "docx_table_4"],
        },
        {
            "metric_key": "offtake_price",
            "metric_name": "Offtake / contracted price",
            "terms": ["ppa price", "contracted ppa price", "offtake price", "tariff", "hydrogen price"],
            "unit": None,
            "direct_patterns": [
                r"(?:ppa price|contracted ppa price|offtake price|tariff|hydrogen price)[^0-9]{0,50}([0-9]+(?:\.[0-9]+)?)",
                r"[a-z]{3}\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*(?:mwh|kg|tonne)",
            ],
            "allow_table_names": ["summary", "assumptions"],
        },
        {
            "metric_key": "capacity_factor_pct",
            "metric_name": "Capacity factor",
            "terms": ["capacity factor", "p50 capacity factor"],
            "unit": "%",
            "direct_patterns": [r"capacity factor[^0-9]{0,50}([0-9]+(?:\.[0-9]+)?)"],
            "allow_table_names": ["summary", "assumptions"],
        },
    ]


def metric_source_allowed(result, metric):
    """Restrict metric parsing to proper financial-metric sources."""

    source_label = infer_source_label(result)
    section_heading = normalize_text(result.get("section_heading"))
    source_reference = normalize_text(result.get("source_reference"))
    if source_label not in ["workbook", "memo", "deck", "pdf_deck", "client_table_evidence"]:
        return False

    if metric["metric_key"] == "total_capex":
        if "project costs" in section_heading and "summary" not in section_heading:
            return False

    allowed_names = metric.get("allow_table_names", [])

    if source_label in ["workbook", "memo"] and row_source_is_table(result):
        if not any(name in section_heading or name in source_reference for name in allowed_names):
            return False

    return True


def infer_unit_from_text(text_value, default_unit=None):
    text_norm = normalize_text(text_value)

    unit_patterns = [
        (r"\b(usd|aed|eur|gbp|aud|cad|sar|inr)\s*(mn|m|million|bn|billion)?\b", lambda m: clean_text(" ".join(part for part in m.groups() if part)).upper()),
        (r"\b([a-z]{3})\s*/\s*(mwh|kwh|kg|tonne)\b", lambda m: f"{m.group(1).upper()}/{m.group(2)}"),
        (r"\b%\b", lambda m: "%"),
        (r"\bx\b", lambda m: "x"),
    ]

    for pattern, formatter in unit_patterns:
        match = re.search(pattern, text_norm)
        if match:
            return formatter(match)

    return default_unit


def metric_name_matches(metric, value):
    return has_any(value, metric["terms"])


def extract_metric_from_table_row(row_text, metric, transaction_profile=None):
    row_lower = normalize_text(row_text)
    structured = extract_structured_key_value_row(row_text)
    metric_label = clean_text(
        structured.get("metric")
        or structured.get("assumption_name")
        or structured.get("section")
    )

    if metric_label and metric_name_matches(metric, metric_label):
        value = safe_float(structured.get("value"))
        if value is not None and metric_value_is_plausible(metric["metric_key"], value, transaction_profile):
            unit = clean_text(structured.get("unit")) or infer_unit_from_text(row_text, metric.get("unit"))
            return {
                "value": value,
                "unit": unit,
                "structured_fields": structured,
            }

    if not any(term in row_lower for term in metric["terms"]):
        return None

    if metric["metric_key"] == "total_capex" and not has_any(row_lower, ["total project cost", "total capex", "capex"]):
        return None

    cells = parse_table_row_cells(row_text)
    values = [cell["value"] for cell in cells]

    metric_cell_index = None

    for index, value in enumerate(values):
        if any(term in normalize_text(value) for term in metric["terms"]):
            metric_cell_index = index
            break

    if metric_cell_index is None:
        return None

    candidate_values = []

    for value in values[metric_cell_index + 1:]:
        number_value = first_number_in_text(value)

        if number_value is not None:
            candidate_values.append(number_value)

    for candidate in candidate_values:
        if metric_value_is_plausible(metric["metric_key"], candidate, transaction_profile):
            return {
                "value": candidate,
                "unit": infer_unit_from_text(row_text, metric.get("unit")),
                "structured_fields": structured,
            }

    return None


def extract_metric_from_direct_text(chunk_text, metric, transaction_profile=None):
    text_value = clean_text(chunk_text)

    if re.search(r"Row\s+\d+:", text_value, flags=re.IGNORECASE):
        return None

    for pattern in metric["direct_patterns"]:
        match = re.search(pattern, text_value, flags=re.IGNORECASE)

        if not match:
            continue

        candidate = safe_float(match.group(1))

        if metric_value_is_plausible(metric["metric_key"], candidate, transaction_profile):
            return {
                "value": candidate,
                "unit": infer_unit_from_text(text_value, metric.get("unit")),
                "structured_fields": {},
            }

    return None


def extract_financial_metrics(evidence_results, transaction_profile=None):
    metric_definitions = get_metric_definitions(transaction_profile)
    rows = []
    seen = set()

    for result in evidence_results:
        chunk_text = clean_text(result.get("chunk_text"))
        source_label = infer_source_label(result)

        if source_label not in ["workbook", "memo", "deck", "pdf_deck", "client_table_evidence"]:
            continue

        row_texts = extract_row_texts(chunk_text)

        for metric in metric_definitions:
            if not metric_source_allowed(result, metric):
                continue

            for row_text in row_texts:
                extracted_metric = extract_metric_from_table_row(row_text, metric, transaction_profile)

                if extracted_metric is None:
                    continue
                value = extracted_metric["value"]

                signature = (metric["metric_key"], source_label, round(value, 6), result.get("chunk_id"), "table_row")

                if signature in seen:
                    continue

                seen.add(signature)

                rows.append({
                    "metric_key": metric["metric_key"],
                    "metric_name": metric["metric_name"],
                    "source_label": source_label,
                    "value": value,
                    "unit": extracted_metric.get("unit") or metric.get("unit"),
                    "structured_fields": extracted_metric.get("structured_fields", {}),
                    "source_chunk": get_source_reference(result),
                    "extraction_method": "table_row",
                })

            extracted_metric = extract_metric_from_direct_text(chunk_text, metric, transaction_profile)

            if extracted_metric is None:
                continue
            value = extracted_metric["value"]

            signature = (metric["metric_key"], source_label, round(value, 6), result.get("chunk_id"), "direct_text")

            if signature in seen:
                continue

            seen.add(signature)

            rows.append({
                "metric_key": metric["metric_key"],
                "metric_name": metric["metric_name"],
                "source_label": source_label,
                "value": value,
                "unit": extracted_metric.get("unit") or metric.get("unit"),
                "structured_fields": extracted_metric.get("structured_fields", {}),
                "source_chunk": get_source_reference(result),
                "extraction_method": "direct_text",
            })

    return rows


def unique_float_values(values, tolerance=0.0001):
    unique_values = []
    for value in values:
        if value is None:
            continue
        already_seen = False
        for existing in unique_values:
            if abs(float(existing) - float(value)) <= tolerance:
                already_seen = True
                break
        if not already_seen:
            unique_values.append(value)
    return unique_values


def dedupe_source_chunks(chunks):
    output = []
    seen = set()
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        key = (chunk.get("chunk_id"), chunk.get("source_reference"))
        if key in seen:
            continue
        seen.add(key)
        output.append(chunk)
    return output


def reconcile_financial_metrics(metric_rows):
    by_metric = {}

    for row in metric_rows:
        metric_key = row["metric_key"]
        if metric_key not in by_metric:
            by_metric[metric_key] = {
                "metric_key": metric_key,
                "metric_name": row["metric_name"],
                "unit": row["unit"],
                "values_by_source": {},
                "units_by_source": {},
                "source_chunks_by_source": {},
            }
        source_label = row["source_label"]
        by_metric[metric_key]["values_by_source"].setdefault(source_label, []).append(row["value"])
        by_metric[metric_key]["units_by_source"].setdefault(source_label, [])
        if row.get("unit") and row.get("unit") not in by_metric[metric_key]["units_by_source"][source_label]:
            by_metric[metric_key]["units_by_source"][source_label].append(row.get("unit"))
        by_metric[metric_key]["source_chunks_by_source"].setdefault(source_label, []).append(row["source_chunk"])

    reconciliation_rows = []
    for metric_key, item in by_metric.items():
        cleaned_values_by_source = {}
        source_chunks = []
        for source_label, values in item["values_by_source"].items():
            cleaned_values_by_source[source_label] = unique_float_values(values)
        for chunks in item["source_chunks_by_source"].values():
            source_chunks.extend(chunks)
        source_chunks = dedupe_source_chunks(source_chunks)

        all_values = []
        for values in cleaned_values_by_source.values():
            all_values.extend(values)
        distinct_values = unique_float_values(all_values)

        preferred_value = None
        preferred_source = None
        for source_label in ["workbook", "client_table_evidence", "memo", "deck", "pdf_deck", "unknown"]:
            source_values = cleaned_values_by_source.get(source_label, [])
            if source_values:
                preferred_value = source_values[0]
                preferred_source = source_label
                break

        source_level_conflict = any(len(values) > 1 for values in cleaned_values_by_source.values())
        if len(distinct_values) > 1:
            traffic_light = STATUS_RED
            issue = "Inconsistent values found across retrieved sources."
        elif source_level_conflict:
            traffic_light = STATUS_RED
            issue = "Multiple values found within at least one source category."
        elif len(distinct_values) == 1:
            traffic_light = STATUS_GREEN
            issue = "No inconsistency detected across retrieved values."
        else:
            traffic_light = STATUS_GREY
            issue = "No numeric value extracted."

        reconciliation_rows.append({
            "metric_key": metric_key,
            "metric_name": item["metric_name"],
            "unit": item["unit"],
            "values_by_source": cleaned_values_by_source,
            "units_by_source": item.get("units_by_source", {}),
            "preferred_value_for_review": preferred_value,
            "preferred_source": preferred_source,
            "traffic_light": traffic_light,
            "issue": issue,
            "source_chunks": source_chunks,
        })

    metric_order = [
        "total_capex", "project_irr_pct", "equity_irr_pct", "npv",
        "minimum_dscr_x", "ev_ebitda_x", "debt_total_cost_pct", "year_one_revenue",
        "offtake_price", "capacity_factor_pct",
    ]
    order_map = {metric_key: index for index, metric_key in enumerate(metric_order)}
    return sorted(reconciliation_rows, key=lambda row: order_map.get(row["metric_key"], 999))


def run_financial_reconciliation(evidence_packets, transaction_profile=None):
    evidence_results = evidence_packets.get("financial_reconciliation", {}).get("selected_results", [])
    metric_rows = extract_financial_metrics(evidence_results, transaction_profile)
    reconciliation_rows = reconcile_financial_metrics(metric_rows)

    if any(row["traffic_light"] == STATUS_RED for row in reconciliation_rows):
        module_status = STATUS_RED
    elif reconciliation_rows:
        module_status = STATUS_GREEN
    else:
        module_status = STATUS_GREY

    return {
        "module": "Financial Metrics and Inconsistency Review",
        "traffic_light": module_status,
        "summary": "Financial metrics are extracted using row-aware parsing and reconciled across workbook, memo, deck and PDF evidence.",
        "metric_rows_extracted": metric_rows,
        "reconciliation_table": reconciliation_rows,
    }


# ---------------------------------------------------------------------
# Sensitivity review
# ---------------------------------------------------------------------

def get_sensitivity_scenarios(transaction_profile=None, review_config=None):
    transaction_profile = transaction_profile or {}
    review_config = review_config or {}
    asset_class = normalize_text(transaction_profile.get("asset_class"))
    revenue_model = normalize_text(transaction_profile.get("revenue_model"))

    scenarios = [
        "Base Case",
        "Capex +10%",
        "COD delay 6 months",
        "FX depreciation 10%",
        "Combined downside",
    ]
    scenarios.extend(review_config.get("sensitivity_theme_library", []))

    if "merchant" in revenue_model:
        scenarios.append("Merchant price -15%")

    if asset_class in ["solar_pv", "onshore_wind", "offshore_wind"]:
        scenarios.append("Generation -5%")

    if asset_class in ["bess", "hydrogen"]:
        scenarios.append("Utilisation / availability downside")

    if asset_class == "bess":
        scenarios.append("BESS degradation / augmentation")

    output = []
    seen = set()
    for scenario in scenarios:
        if scenario not in seen:
            output.append(scenario)
            seen.add(scenario)
    return output


def get_required_sensitivity_scenarios(transaction_profile=None, review_config=None):
    review_config = review_config or {}
    tiers = review_config.get("sensitivity_theme_tiers", {})
    tier_required = tiers.get("required", [])
    if tier_required:
        base_required = ["Base Case", "Capex +10%", "COD delay 6 months", "FX depreciation 10%", "Combined downside"]
        output = []
        seen = set()
        for scenario in base_required + tier_required:
            if scenario not in seen:
                output.append(scenario)
                seen.add(scenario)
        return output
    return get_sensitivity_scenarios(transaction_profile, review_config)


def get_all_known_sensitivity_scenarios():
    return [
        "Base Case",
        "Merchant price -15%",
        "Generation -5%",
        "Capex +10%",
        "COD delay 6 months",
        "FX depreciation 10%",
        "Combined downside",
        "Utilisation / availability downside",
        "BESS degradation / augmentation",
        "Irradiation - P90",
        "Capex +15%",
        "Curtailment downside",
        "Module degradation",
        "Wind resource P90",
        "Wake loss downside",
        "Turbine availability downside",
        "Offshore construction delay",
        "Vessel/weather downtime",
        "Export cable failure",
        "Cycling downside",
        "Ancillary services price downside",
        "Electrolyser utilisation downside",
        "Electricity price upside",
        "Hydrogen offtake volume downside",
        "LCOH downside",
        "Allowed revenue reduction",
        "Tariff reset downside",
        "Network capex overrun",
        "Regulatory delay",
    ]


def normalize_sensitivity_scenario(value):
    value_norm = normalize_text(value)
    mapping = {
        "base case": "Base Case",
        "merchant price -15%": "Merchant price -15%",
        "irradiation": "Irradiation - P90",
        "generation -5%": "Generation -5%",
        "capex +10%": "Capex +10%",
        "capex +15%": "Capex +15%",
        "cod delay 6 months": "COD delay 6 months",
        "fx depreciation 10%": "FX depreciation 10%",
        "combined downside": "Combined downside",
        "utilisation": "Utilisation / availability downside",
        "utilization": "Utilisation / availability downside",
        "availability": "Utilisation / availability downside",
        "bess degradation / augmentation": "BESS degradation / augmentation",
        "battery augmentation": "BESS degradation / augmentation",
        "curtailment": "Curtailment downside",
        "module degradation": "Module degradation",
        "wind resource": "Wind resource P90",
        "wake loss": "Wake loss downside",
        "turbine availability": "Turbine availability downside",
        "offshore construction": "Offshore construction delay",
        "weather downtime": "Vessel/weather downtime",
        "export cable": "Export cable failure",
        "cycling": "Cycling downside",
        "ancillary services": "Ancillary services price downside",
        "electrolyser utilisation": "Electrolyser utilisation downside",
        "electrolyzer utilization": "Electrolyser utilisation downside",
        "electricity price": "Electricity price upside",
        "hydrogen offtake volume": "Hydrogen offtake volume downside",
        "lcoh": "LCOH downside",
        "allowed revenue": "Allowed revenue reduction",
        "tariff reset": "Tariff reset downside",
        "network capex": "Network capex overrun",
        "regulatory delay": "Regulatory delay",
    }
    for key, label in mapping.items():
        if key in value_norm:
            return label
    return None


def extract_workbook_sensitivity_case(row_text, result):
    scenario = get_cell_value_by_label(row_text, ["Sensitivity Cases"])
    scenario = normalize_sensitivity_scenario(scenario)
    if not scenario:
        return None
    equity_irr = safe_float(get_cell_value_by_label(row_text, ["6", "Equity_IRR_pct"]))
    minimum_dscr = safe_float(get_cell_value_by_label(row_text, ["7", "Min_DSCR_x"]))
    included_flag = clean_text(get_cell_value_by_label(row_text, ["8", "Included_In_Submission"]))
    if included_flag.lower() == "yes":
        included_flag = "Yes"
    elif included_flag.lower() == "no":
        included_flag = "No"
    else:
        included_flag = None
    if equity_irr is None or minimum_dscr is None:
        return None
    return {
        "scenario": scenario,
        "equity_irr_pct": equity_irr,
        "minimum_dscr_x": minimum_dscr,
        "included_in_submission": included_flag,
        "source_label": infer_source_label(result),
        "source_chunk": get_source_reference(result),
        "extraction_method": "workbook_labelled_table",
    }


def extract_simple_sensitivity_case(row_text, result):
    cells = parse_table_row_cells(row_text)
    values = [cell["value"] for cell in cells]
    scenario = None
    scenario_index = None
    for index, value in enumerate(values):
        scenario = normalize_sensitivity_scenario(value)
        if scenario:
            scenario_index = index
            break
    if not scenario:
        return None
    if has_any(row_text, ["not shown", "not included", "does not include"]):
        return None
    candidate_numbers = []
    for value in values[scenario_index + 1:]:
        number_value = first_number_in_text(value)
        if number_value is not None:
            candidate_numbers.append(number_value)
    if len(candidate_numbers) < 1:
        return None
    equity_irr = None
    minimum_dscr = None
    for number_value in candidate_numbers:
        if equity_irr is None and 3 <= number_value <= 60:
            equity_irr = number_value
            continue
        if minimum_dscr is None and 0.5 <= number_value <= 3.5:
            minimum_dscr = number_value
            continue
    return {
        "scenario": scenario,
        "equity_irr_pct": equity_irr,
        "minimum_dscr_x": minimum_dscr,
        "included_in_submission": None,
        "source_label": infer_source_label(result),
        "source_chunk": get_source_reference(result),
        "extraction_method": "simple_table_row",
    }


def extract_sensitivity_cases(evidence_results):
    sensitivity_cases = []
    seen = set()
    for result in evidence_results:
        chunk_text = clean_text(result.get("chunk_text"))
        row_texts = extract_row_texts(chunk_text)
        for row_text in row_texts:
            if not has_any(row_text, get_all_known_sensitivity_scenarios()):
                continue
            case = None
            case = extract_workbook_sensitivity_case(row_text, result)
            if case is None:
                case = extract_simple_sensitivity_case(row_text, result)
            if case is None:
                continue
            signature = (case.get("scenario"), case.get("equity_irr_pct"), case.get("minimum_dscr_x"), case.get("included_in_submission"), case.get("source_chunk", {}).get("chunk_id"))
            if signature in seen:
                continue
            seen.add(signature)
            sensitivity_cases.append(case)
    return sensitivity_cases


def dedupe_sensitivity_cases(cases, transaction_profile=None, review_config=None):
    by_scenario = {}
    for case in cases:
        scenario = clean_text(case.get("scenario"))
        if not scenario:
            continue
        quality_score = 0
        if case.get("source_label") == "workbook":
            quality_score += 10
        if case.get("included_in_submission") in ["Yes", "No"]:
            quality_score += 6
        if case.get("equity_irr_pct") is not None:
            quality_score += 2
        if case.get("minimum_dscr_x") is not None:
            quality_score += 2
        if case.get("extraction_method") == "workbook_labelled_table":
            quality_score += 4
        if scenario not in by_scenario or quality_score > by_scenario[scenario]["quality_score"]:
            by_scenario[scenario] = {"quality_score": quality_score, "case": case}
    output_cases = []
    for scenario in get_sensitivity_scenarios(transaction_profile, review_config):
        if scenario in by_scenario:
            output_cases.append(by_scenario[scenario]["case"])
    for scenario, data in by_scenario.items():
        if scenario not in get_sensitivity_scenarios(transaction_profile, review_config):
            output_cases.append(data["case"])
    return output_cases


def extract_submitted_pack_sensitivity_gaps(evidence_results):
    gaps = []
    seen = set()
    for result in evidence_results:
        chunk_text = clean_text(result.get("chunk_text"))
        if has_any(chunk_text, ["combined downside not shown", "does not include a combined downside case"]):
            key = "combined_downside_not_shown"
            if key not in seen:
                gaps.append({"gap": "Combined downside not shown in submitted pack.", "source_chunk": get_source_reference(result)})
                seen.add(key)
        if has_any(chunk_text, ["battery augmentation sensitivity not shown", "bess degradation/augmentation sensitivity", "bess degradation and augmentation sensitivity"]):
            key = "bess_augmentation_not_shown"
            if key not in seen:
                gaps.append({"gap": "BESS degradation / augmentation sensitivity not shown in submitted pack.", "source_chunk": get_source_reference(result)})
                seen.add(key)
    return gaps


def validate_sensitivity_cases(cases, transaction_profile=None, review_config=None):
    failures = []
    expected_scenarios = get_required_sensitivity_scenarios(transaction_profile, review_config)
    case_map = {case.get("scenario"): case for case in cases}

    for scenario in expected_scenarios:
        case = case_map.get(scenario)
        if not case:
            failures.append(f"Missing sensitivity coverage theme: {scenario}")
            continue

        if case.get("equity_irr_pct") is None and case.get("minimum_dscr_x") is None:
            failures.append(f"Sensitivity case has no extracted quantitative metric: {scenario}")

    return failures


def run_sensitivity_review(evidence_packets, review_config=None, transaction_profile=None):
    evidence_results = evidence_packets.get("sensitivity_review", {}).get("selected_results", [])
    blob = get_evidence_blob(evidence_results)
    raw_sensitivity_cases = extract_sensitivity_cases(evidence_results)
    sensitivity_cases = dedupe_sensitivity_cases(raw_sensitivity_cases, transaction_profile, review_config)
    submitted_pack_gaps = extract_submitted_pack_sensitivity_gaps(evidence_results)
    validation_failures = validate_sensitivity_cases(sensitivity_cases, transaction_profile, review_config)
    combined_downside_available = any(case.get("scenario") == "Combined downside" for case in sensitivity_cases)
    bess_augmentation_available = any(case.get("scenario") == "BESS degradation / augmentation" for case in sensitivity_cases)
    required_scenarios = get_required_sensitivity_scenarios(transaction_profile, review_config)
    all_scenarios = get_sensitivity_scenarios(transaction_profile, review_config)
    recommended_scenarios = review_config.get("sensitivity_theme_tiers", {}).get("recommended", [])
    optional_scenarios = review_config.get("sensitivity_theme_tiers", {}).get("optional", [])
    submission_gap_flag = any(case.get("scenario") in required_scenarios and case.get("included_in_submission") == "No" for case in sensitivity_cases)
    if validation_failures:
        module_status = STATUS_RED
    elif submission_gap_flag:
        module_status = STATUS_AMBER
    elif not sensitivity_cases and not has_any(blob, ["sensitivity"]):
        module_status = STATUS_RED
    else:
        module_status = STATUS_GREEN
    return {
        "module": "Sensitivity and Downside Protection Review",
        "traffic_light": module_status,
        "summary": "Sensitivity review checks coverage of profile-relevant downside themes and distinguishes submitted cases from workbook-only cases.",
        "required_scenarios": required_scenarios,
        "recommended_scenarios": recommended_scenarios,
        "optional_scenarios": optional_scenarios,
        "all_profile_scenarios": all_scenarios,
        "combined_downside_available_in_evidence": combined_downside_available,
        "bess_augmentation_available_in_evidence": bess_augmentation_available,
        "submission_gap_flag": submission_gap_flag,
        "sensitivity_cases": sensitivity_cases,
        "submitted_pack_gaps": submitted_pack_gaps,
        "validation_failures": validation_failures,
        "raw_case_count_before_deduplication": len(raw_sensitivity_cases),
        "deduped_case_count": len(sensitivity_cases),
        "source_chunks": [get_source_reference(item) for item in evidence_results[:8]],
    }


# ---------------------------------------------------------------------
# Market/offtake/revenue
# ---------------------------------------------------------------------

def run_market_offtake_revenue_review(evidence_packets):
    evidence_results = evidence_packets.get("market_offtake_revenue", {}).get("selected_results", [])
    blob = get_evidence_blob(evidence_results)

    flags = {
        "contracted_revenue_evidenced": has_any(blob, ["contracted share", "contracted revenue", "contract revenue", "offtake", "tariff"]),
        "market_price_support_missing": has_any(blob, ["price curve source not appended", "independent price curve referenced but not appended", "independent price study is referenced but not included", "independent curve absent"]),
        "offtaker_credit_missing": has_any(blob, ["no independent credit", "offtaker-credit support are missing", "offtaker credit support are missing", "independent credit review absent"]),
        "payment_security_pending": has_any(blob, ["payment-security package remains under negotiation", "payment security", "under discussion", "under negotiation", "not executed"]),
        "price_conflict": has_any(blob, ["price conflict", "deck says", "different price", "price inconsistency"]),
        "non_energy_revenue_evidenced": has_any(blob, ["ancillary services", "capacity", "shifting", "availability payment", "allowed revenue"]),
    }

    findings = []

    if flags["contracted_revenue_evidenced"]:
        findings.append(
            {
                "finding": "Contracted or regulated revenue basis is evidenced.",
                "traffic_light": STATUS_AMBER if flags["market_price_support_missing"] else STATUS_GREEN,
                "explanation": "Revenue support is evidenced, but any market-price component should be checked.",
                "source_chunks": find_source_chunks(evidence_results, ["contracted share", "contracted revenue", "offtake", "tariff", "allowed revenue"]),
            }
        )

    if flags["market_price_support_missing"]:
        findings.append(
            {
                "finding": "Independent market-price support is missing or not appended.",
                "traffic_light": STATUS_RED,
                "explanation": "The submission appears to rely on market-price revenue, but the independent curve is not available in the submission evidence.",
                "source_chunks": find_source_chunks(evidence_results, ["price curve", "not appended", "not included", "independent curve absent"]),
            }
        )

    if flags["offtaker_credit_missing"] or flags["payment_security_pending"]:
        findings.append(
            {
                "finding": "Offtaker credit and payment-security package are not fully evidenced.",
                "traffic_light": STATUS_RED,
                "explanation": "Payment security / credit support appears under discussion, not fully executed or independently reviewed.",
                "source_chunks": find_source_chunks(evidence_results, ["offtaker", "credit", "payment security", "under negotiation", "under discussion", "not executed"]),
            }
        )

    if flags["price_conflict"]:
        findings.append(
            {
                "finding": "Revenue price inconsistency should be reconciled.",
                "traffic_light": STATUS_AMBER,
                "explanation": "Retrieved evidence indicates different price references across sources.",
                "source_chunks": find_source_chunks(evidence_results, ["price", "conflict", "different", "inconsistency"]),
            }
        )

    module_status = STATUS_GREEN
    if any(item["traffic_light"] == STATUS_RED for item in findings):
        module_status = STATUS_RED
    elif any(item["traffic_light"] == STATUS_AMBER for item in findings):
        module_status = STATUS_AMBER

    return {
        "module": "Market, Offtake and Revenue Review",
        "traffic_light": module_status,
        "summary": "Market/offtake review flags market-price support, offtaker credit, payment security and price consistency issues.",
        "flags": flags,
        "findings": findings,
    }


def run_revenue_rule_review(evidence_packets, module_name, review_path, rules):
    evidence_results = evidence_packets.get("market_offtake_revenue", {}).get("selected_results", [])
    blob = get_evidence_blob(evidence_results)
    findings = []
    flags = {}

    for rule in rules:
        present = has_any(blob, rule["terms"])
        weak = has_any(blob, rule.get("weak_terms", []))
        flags[rule["flag"]] = present

        if not present and rule.get("required", False):
            findings.append(
                {
                    "finding": rule["missing_finding"],
                    "traffic_light": STATUS_RED,
                    "explanation": rule["missing_explanation"],
                    "source_chunks": [],
                }
            )
            continue

        if present:
            findings.append(
                {
                    "finding": rule["finding"],
                    "traffic_light": STATUS_AMBER if weak else rule.get("traffic_light", STATUS_GREEN),
                    "explanation": rule["weak_explanation"] if weak else rule["explanation"],
                    "source_chunks": find_source_chunks(evidence_results, rule["terms"] + rule.get("weak_terms", [])),
                }
            )

    module_status = STATUS_GREEN
    if any(item["traffic_light"] == STATUS_RED for item in findings):
        module_status = STATUS_RED
    elif any(item["traffic_light"] == STATUS_AMBER for item in findings):
        module_status = STATUS_AMBER
    elif not findings:
        module_status = STATUS_GREY

    return {
        "module": module_name,
        "traffic_light": module_status,
        "summary": f"{module_name} applies revenue checks selected from the inferred transaction profile.",
        "revenue_review_path": review_path,
        "flags": flags,
        "findings": findings,
    }


def run_power_revenue_review(evidence_packets, transaction_profile):
    return run_revenue_rule_review(
        evidence_packets,
        module_name="Power Revenue Review",
        review_path="power",
        rules=[
            {
                "flag": "power_revenue_basis_evidenced",
                "terms": ["ppa", "cfd", "merchant", "power price", "tariff", "offtake"],
                "weak_terms": ["not appended", "not included", "not executed", "under negotiation", "no independent"],
                "required": True,
                "finding": "Power revenue basis is evidenced.",
                "explanation": "Retrieved evidence includes PPA, CfD, merchant, tariff or offtake revenue support.",
                "weak_explanation": "Power revenue basis is evidenced, but support appears incomplete or not executed.",
                "missing_finding": "Power revenue basis is not clearly evidenced.",
                "missing_explanation": "No PPA, CfD, merchant, tariff or offtake support was found in selected evidence.",
            },
            {
                "flag": "merchant_support_evidenced",
                "terms": ["merchant curve", "merchant price", "independent price", "price forecast"],
                "weak_terms": ["not appended", "not included", "independent curve absent"],
                "finding": "Merchant price support is referenced.",
                "explanation": "Merchant price support appears in the selected evidence.",
                "weak_explanation": "Merchant price support is referenced but may not be attached or independently supported.",
            },
        ],
    )


def run_storage_revenue_review(evidence_packets, transaction_profile):
    return run_revenue_rule_review(
        evidence_packets,
        module_name="Storage Revenue Review",
        review_path="storage",
        rules=[
            {
                "flag": "storage_revenue_basis_evidenced",
                "terms": ["tolling", "capacity", "ancillary services", "arbitrage", "shifting", "availability payment"],
                "weak_terms": ["not executed", "under negotiation", "not included", "not modelled"],
                "required": True,
                "finding": "Storage revenue basis is evidenced.",
                "explanation": "Retrieved evidence includes tolling, capacity, ancillary services, arbitrage or availability revenue.",
                "weak_explanation": "Storage revenue is evidenced, but contract status or modelling support appears incomplete.",
                "missing_finding": "Storage revenue basis is not clearly evidenced.",
                "missing_explanation": "No tolling, capacity, ancillary services, arbitrage or availability support was found.",
            },
            {
                "flag": "cycling_degradation_evidenced",
                "terms": ["cycling", "degradation", "augmentation", "availability", "warranty"],
                "weak_terms": ["not shown", "not included", "not separately modelled"],
                "finding": "Storage operating-performance linkage is evidenced.",
                "explanation": "Evidence links storage revenue to cycling, degradation, augmentation, availability or warranty assumptions.",
                "weak_explanation": "Storage operating-performance linkage appears incomplete or not separately modelled.",
            },
        ],
    )


def run_hydrogen_offtake_review(evidence_packets, transaction_profile):
    return run_revenue_rule_review(
        evidence_packets,
        module_name="Hydrogen Offtake Revenue Review",
        review_path="hydrogen",
        rules=[
            {
                "flag": "hydrogen_offtake_evidenced",
                "terms": ["hydrogen offtake", "ammonia offtake", "take-or-pay", "hydrogen sale", "customer contract"],
                "weak_terms": ["under negotiation", "not executed", "non-binding", "not included"],
                "required": True,
                "finding": "Hydrogen offtake basis is evidenced.",
                "explanation": "Retrieved evidence includes hydrogen/ammonia offtake or customer contract support.",
                "weak_explanation": "Hydrogen offtake is referenced, but execution status or binding support appears incomplete.",
                "missing_finding": "Hydrogen offtake basis is not clearly evidenced.",
                "missing_explanation": "No hydrogen/ammonia offtake or customer contract support was found.",
            },
            {
                "flag": "lcoh_power_cost_evidenced",
                "terms": ["lcoh", "electricity price", "power cost", "electrolyser utilisation", "electrolyzer utilisation"],
                "weak_terms": ["not shown", "not included", "not modelled"],
                "finding": "Hydrogen unit economics support is evidenced.",
                "explanation": "Evidence references LCOH, electricity cost or electrolyser utilisation assumptions.",
                "weak_explanation": "Hydrogen unit economics are referenced but support appears incomplete.",
            },
        ],
    )


def run_regulated_revenue_review(evidence_packets, transaction_profile):
    return run_revenue_rule_review(
        evidence_packets,
        module_name="Regulated Revenue Review",
        review_path="regulated",
        rules=[
            {
                "flag": "regulated_revenue_evidenced",
                "terms": ["allowed revenue", "regulated asset base", "rab", "tariff", "regulated return", "distribution tariff"],
                "weak_terms": ["under review", "not approved", "pending", "not final"],
                "required": True,
                "finding": "Regulated revenue framework is evidenced.",
                "explanation": "Retrieved evidence includes tariff, allowed revenue, RAB or regulated-return support.",
                "weak_explanation": "Regulated revenue framework is evidenced, but approval or finality appears unresolved.",
                "missing_finding": "Regulated revenue framework is not clearly evidenced.",
                "missing_explanation": "No tariff, allowed revenue, RAB or regulated-return support was found.",
            },
            {
                "flag": "grid_modernisation_evidenced",
                "terms": ["grid modernisation", "grid modernization", "distribution network", "network capex", "loss reduction"],
                "weak_terms": ["not approved", "pending", "not quantified", "under review"],
                "finding": "Grid modernisation scope is evidenced.",
                "explanation": "Evidence references grid modernisation, distribution network capex or loss-reduction scope.",
                "weak_explanation": "Grid modernisation scope is referenced but approval or quantification appears incomplete.",
            },
        ],
    )


def run_generic_revenue_review(evidence_packets, transaction_profile):
    return run_revenue_rule_review(
        evidence_packets,
        module_name="Revenue Model Review",
        review_path="generic",
        rules=[
            {
                "flag": "revenue_basis_evidenced",
                "terms": ["revenue", "offtake", "contract", "tariff", "price", "customer", "payment"],
                "weak_terms": ["under negotiation", "not executed", "not included", "no independent", "pending"],
                "required": True,
                "finding": "Revenue basis is evidenced.",
                "explanation": "Retrieved evidence includes revenue, contract, tariff, customer or payment support.",
                "weak_explanation": "Revenue basis is evidenced, but execution or support appears incomplete.",
                "missing_finding": "Revenue basis is not clearly evidenced.",
                "missing_explanation": "No clear revenue, contract, tariff, customer or payment support was found.",
            },
        ],
    )


def run_revenue_model_review(evidence_packets, transaction_profile):
    asset_classes = transaction_profile.get("asset_classes") or [transaction_profile.get("primary_asset_class") or transaction_profile.get("asset_class")]
    asset_classes = [normalize_text(asset_class) for asset_class in asset_classes if clean_text(asset_class)]
    revenue_models = transaction_profile.get("revenue_models") or [transaction_profile.get("revenue_model")]
    revenue_models = [normalize_text(revenue_model) for revenue_model in revenue_models if clean_text(revenue_model)]
    revenue_model_text = " ".join(revenue_models)

    if "hydrogen_offtake" in revenue_models or "hydrogen" in revenue_model_text:
        return run_hydrogen_offtake_review(evidence_packets, transaction_profile)

    if "tolling" in revenue_models or "capacity_payment" in revenue_models:
        return run_storage_revenue_review(evidence_packets, transaction_profile)

    if "regulated_grid_distribution" in revenue_models or "regulated_return" in revenue_models or "regulated" in revenue_model_text:
        return run_regulated_revenue_review(evidence_packets, transaction_profile)

    if any(model in revenue_models for model in ["ppa", "cfd", "merchant", "hybrid_contract_merchant"]):
        return run_power_revenue_review(evidence_packets, transaction_profile)

    if "bess" in asset_classes:
        return run_storage_revenue_review(evidence_packets, transaction_profile)

    if "hydrogen" in asset_classes:
        return run_hydrogen_offtake_review(evidence_packets, transaction_profile)

    if "regulated_grid_distribution" in asset_classes:
        return run_regulated_revenue_review(evidence_packets, transaction_profile)

    if any(asset_class in asset_classes for asset_class in ["solar_pv", "onshore_wind", "offshore_wind"]):
        return run_power_revenue_review(evidence_packets, transaction_profile)

    return run_generic_revenue_review(evidence_packets, transaction_profile)


# ---------------------------------------------------------------------
# Risk register
# ---------------------------------------------------------------------

def extract_risk_table_rows(evidence_results):
    risk_rows = []

    for result in evidence_results:
        chunk_text = clean_text(result.get("chunk_text"))
        row_texts = extract_row_texts(chunk_text)

        for row_text in row_texts:
            values = cell_values_from_row(row_text)

            if len(values) < 2:
                continue

            first_cell = clean_text(values[0])
            second_cell = clean_text(values[1])

            if normalize_text(first_cell) in ["risk", "principal risks"]:
                continue

            # Risk table rows generally have:
            # cell 0 = risk, cell 1 = rating/assessment, cell 2 = mitigation/status
            if has_any(second_cell, ["high", "medium", "low", "not assessed"]):
                mitigation_status = values[2] if len(values) >= 3 else ""

                risk_rows.append(
                    {
                        "risk_text": first_cell,
                        "rating": second_cell,
                        "mitigation_status": clean_text(mitigation_status),
                        "source_chunk": get_source_reference(result),
                    }
                )

    return risk_rows


def find_best_risk_row(risk_rows, terms):
    for row in risk_rows:
        if has_any(row.get("risk_text"), terms):
            return row

    for row in risk_rows:
        if has_any(row.get("mitigation_status"), terms):
            return row

    return None


def find_narrative_risk_evidence(evidence_results, terms):
    matched_results = []
    weak_terms = [
        "not assessed",
        "not included",
        "not modelled",
        "pending",
        "under review",
        "under negotiation",
        "open",
        "missing",
        "not quantified",
    ]
    weak_match = False

    for result in evidence_results:
        chunk_text = clean_text(result.get("chunk_text"))
        if not has_any(chunk_text, terms):
            continue
        matched_results.append(result)
        if has_any(chunk_text, weak_terms):
            weak_match = True

    return matched_results, weak_match


def risk_rating_to_traffic_light(rating, mitigation_status):
    rating_lower = normalize_text(rating)
    mitigation_lower = normalize_text(mitigation_status)

    if "not assessed" in rating_lower:
        return STATUS_RED

    if "not assessed" in mitigation_lower or "no dedicated" in mitigation_lower or "no funded" in mitigation_lower:
        return STATUS_RED

    if "high" in rating_lower:
        return STATUS_AMBER

    if "medium-high" in rating_lower:
        return STATUS_AMBER

    if "medium" in rating_lower:
        return STATUS_AMBER

    if "low" in rating_lower:
        return STATUS_GREEN

    return STATUS_GREY


def run_risk_review(review_config, evidence_packets):
    risk_taxonomy = review_config["asset_class_risk_taxonomy"]
    evidence_results = evidence_packets.get("risk_review", {}).get("selected_results", [])
    risk_rows = extract_risk_table_rows(evidence_results)

    rows = []

    for risk_item in risk_taxonomy:
        best_row = find_best_risk_row(risk_rows, risk_item["terms"])

        if not best_row:
            narrative_results, weak_match = find_narrative_risk_evidence(evidence_results, risk_item["terms"])
            if narrative_results:
                rows.append(
                    {
                        "risk_id": risk_item["risk_id"],
                        "risk": risk_item["risk"],
                        "asset_class": risk_item["asset_class"],
                        "rating": "Narrative evidence only",
                        "mitigation_status": "Risk topic appears in selected narrative evidence, but no structured risk-table rating was found.",
                        "reviewer_challenge": risk_item["reviewer_challenge"],
                        "open_gap": "Request structured risk rating / mitigation status if this risk is material.",
                        "traffic_light": STATUS_AMBER if weak_match else STATUS_GREY,
                        "source_chunks": [get_source_reference(item) for item in narrative_results[:5]],
                    }
                )
                continue

            rows.append(
                {
                    "risk_id": risk_item["risk_id"],
                    "risk": risk_item["risk"],
                    "asset_class": risk_item["asset_class"],
                    "rating": "Not identified",
                    "mitigation_status": "No row-level evidence found in retrieved risk tables.",
                    "reviewer_challenge": risk_item["reviewer_challenge"],
                    "open_gap": "Risk topic not clearly addressed in retrieved evidence.",
                    "traffic_light": STATUS_GREY,
                    "source_chunks": [],
                }
            )
            continue

        rating = best_row.get("rating")
        mitigation_status = best_row.get("mitigation_status")
        traffic_light = risk_rating_to_traffic_light(rating, mitigation_status)

        rows.append(
            {
                "risk_id": risk_item["risk_id"],
                "risk": risk_item["risk"],
                "asset_class": risk_item["asset_class"],
                "rating": rating,
                "mitigation_status": mitigation_status,
                "reviewer_challenge": risk_item["reviewer_challenge"],
                "open_gap": "Review the mitigation/status evidence and request support where weak language is present.",
                "traffic_light": traffic_light,
                "source_chunks": [best_row.get("source_chunk")],
            }
        )

    module_status = STATUS_GREEN
    if any(row["traffic_light"] == STATUS_RED for row in rows):
        module_status = STATUS_RED
    elif any(row["traffic_light"] == STATUS_AMBER for row in rows):
        module_status = STATUS_AMBER
    elif any(row["traffic_light"] == STATUS_GREY for row in rows):
        module_status = STATUS_AMBER

    return {
        "module": "Risk Register and Reviewer Challenge Prompts",
        "traffic_light": module_status,
        "summary": "Risk register is built from row-level risk table parsing to avoid contamination across unrelated risk rows.",
        "risk_register": rows,
        "raw_risk_rows_extracted": risk_rows,
    }


# ---------------------------------------------------------------------
# Valuation and benchmark review
# ---------------------------------------------------------------------

def run_client_valuation_review(evidence_packets):
    evidence_results = evidence_packets.get("client_valuation", {}).get("selected_results", [])
    blob = get_evidence_blob(evidence_results)

    valuation_metrics_present = has_any(blob, ["npv", "ev / ebitda", "ev ebitda", "minimum dscr", "equity irr"])
    cost_challenge_present = has_any(blob, ["equipment", "grid interconnection", "overhead", "contingency", "advisory", "upper quartile", "high"])

    findings = []

    if valuation_metrics_present:
        findings.append(
            {
                "finding": "Client-side valuation metrics are evidenced.",
                "traffic_light": STATUS_GREEN,
                "explanation": "Retrieved evidence includes valuation/return metrics such as NPV, DSCR and EV/EBITDA.",
                "source_chunks": find_source_chunks(evidence_results, ["NPV", "EV / EBITDA", "minimum DSCR", "equity IRR"]),
            }
        )
    else:
        findings.append(
            {
                "finding": "Client-side valuation metrics not clearly evidenced.",
                "traffic_light": STATUS_RED,
                "explanation": "The retrieval did not find sufficient client-side valuation metric evidence.",
                "source_chunks": [],
            }
        )

    if cost_challenge_present:
        findings.append(
            {
                "finding": "Project-cost challenge areas are evidenced.",
                "traffic_light": STATUS_AMBER,
                "explanation": (
                    "Evidence points to equipment, grid interconnection, contingency, overhead "
                    "or advisory items requiring challenge. These are client-side assertions unless "
                    "independently benchmarked through corpus_data."
                ),
                "source_chunks": find_source_chunks(
                    evidence_results,
                    ["equipment", "grid interconnection", "overhead", "contingency", "advisory", "upper quartile", "high"],
                ),
            }
        )

    return {
        "module": "Client-Side Valuation Review",
        "traffic_light": STATUS_AMBER if cost_challenge_present else STATUS_GREEN,
        "summary": "Client-side valuation review separates reported valuation metrics from independent benchmark confirmation.",
        "important_note": "Client-side benchmark statements are not treated as independent external benchmark confirmation.",
        "findings": findings,
    }


def get_asset_benchmark_terms(transaction_profile):
    asset_classes = transaction_profile.get("asset_classes") or [transaction_profile.get("primary_asset_class") or transaction_profile.get("asset_class")]
    asset_classes = [normalize_text(asset_class) for asset_class in asset_classes if clean_text(asset_class)]
    benchmark_terms = {
        "solar_pv": ["utility-scale pv", "solar", "pv", "photovoltaic", "inverter", "module", "solar - utility pv"],
        "onshore_wind": ["onshore wind", "wind turbine", "wind farm", "turbine", "wake loss"],
        "offshore_wind": ["offshore wind", "foundation", "export cable", "marine installation", "seabed"],
        "bess": ["utility-scale battery storage", "bess", "battery storage", "augmentation", "storage"],
        "hydrogen": ["hydrogen", "electrolyser", "electrolyzer", "lcoh", "ammonia"],
        "regulated_grid_distribution": ["distribution network", "regulated grid", "grid modernisation", "grid modernization", "network capex"],
    }
    output = []
    for asset_class in asset_classes:
        for term in benchmark_terms.get(asset_class, []):
            if term not in output:
                output.append(term)
    return output


def run_external_benchmark_review(evidence_packets, transaction_profile=None, review_config=None):
    transaction_profile = transaction_profile or {}
    review_config = review_config or {}
    cost_results = evidence_packets.get("external_cost_benchmark", {}).get("selected_results", [])
    ev_results = evidence_packets.get("external_ev_ebitda_benchmark", {}).get("selected_results", [])

    cost_blob = get_evidence_blob(cost_results)
    ev_blob = get_evidence_blob(ev_results)
    asset_class = clean_text(transaction_profile.get("asset_class")) or "not_identified"
    asset_terms = review_config.get("benchmark_term_library") or get_asset_benchmark_terms(transaction_profile)
    generic_cost_terms = [
        "overnight capital cost",
        "grid connection cost",
        "fixed operating expenses",
        "variable operating expense",
        "capex benchmark",
        "cost benchmark",
    ]

    asset_benchmark_available = bool(asset_terms) and has_any(cost_blob, asset_terms)
    cost_benchmark_available = asset_benchmark_available and has_any(cost_blob, generic_cost_terms)

    ev_ebitda_benchmark_available = has_all(ev_blob, ["ev", "ebitda"]) and has_any(
        ev_blob,
        ["enterprise value", "valuation multiple", "transaction benchmark", "comparable transaction", "listed company"],
    )

    findings = []

    if cost_benchmark_available:
        findings.append(
            {
                "benchmark_area": "External cost benchmark",
                "traffic_light": STATUS_GREEN,
                "finding": f"External cost benchmark categories are available for asset class: {asset_class}.",
                "allowed_use": "Use for cost, grid and O&M benchmarking only.",
                "source_chunks": find_source_chunks(
                    cost_results,
                    asset_terms + generic_cost_terms,
                ),
            }
        )
    else:
        finding = "External cost benchmark evidence was not found."
        if asset_terms and has_any(cost_blob, generic_cost_terms) and not asset_benchmark_available:
            finding = f"Cost benchmark evidence was found, but not for inferred asset class: {asset_class}."

        findings.append(
            {
                "benchmark_area": "External cost benchmark",
                "traffic_light": STATUS_GREY,
                "finding": finding,
                "allowed_use": "Do not make external cost benchmark claims.",
                "source_chunks": [],
            }
        )

    if ev_ebitda_benchmark_available:
        findings.append(
            {
                "benchmark_area": "External EV/EBITDA benchmark",
                "traffic_light": STATUS_GREEN,
                "finding": "External EV/EBITDA valuation multiple evidence appears available.",
                "allowed_use": "May be used cautiously if sources are verified.",
                "source_chunks": find_source_chunks(ev_results, ["EV", "EBITDA", "enterprise value", "valuation multiple"]),
            }
        )
    else:
        findings.append(
            {
                "benchmark_area": "External EV/EBITDA benchmark",
                "traffic_light": STATUS_GREY,
                "finding": "External valuation multiple benchmark evidence was not found in the current corpus.",
                "allowed_use": "Do not claim EV/EBITDA is externally benchmarked.",
                "source_chunks": [],
            }
        )

    module_status = STATUS_AMBER
    if cost_benchmark_available and ev_ebitda_benchmark_available:
        module_status = STATUS_GREEN

    return {
        "module": "External Benchmark Review",
        "traffic_light": module_status,
        "summary": "External benchmark review only allows cost benchmarking where corpus evidence matches the inferred asset class.",
        "asset_class": asset_class,
        "asset_benchmark_terms": asset_terms,
        "cost_benchmark_available": cost_benchmark_available,
        "ev_ebitda_benchmark_available": ev_ebitda_benchmark_available,
        "findings": findings,
    }


# ---------------------------------------------------------------------
# Conditions and grey modules
# ---------------------------------------------------------------------

def priority_from_traffic_light(traffic_light):
    if traffic_light == STATUS_RED:
        return "High"
    if traffic_light == STATUS_AMBER:
        return "Medium-high"
    return "Medium"


def append_open_item(rows, seen, category, condition, traffic_light, evidence_text="", source_chunks=None, recommended_action=None):
    key = (category, condition)
    if key in seen:
        return
    seen.add(key)
    rows.append(
        {
            "category": category,
            "condition": condition,
            "priority": priority_from_traffic_light(traffic_light),
            "status": "Open / requires closure",
            "traffic_light": traffic_light,
            "evidence_text": clean_text(evidence_text)[:700],
            "source_chunks": source_chunks or [],
            "recommended_action": recommended_action or "Track to closure before final approval.",
        }
    )


def build_open_items_register(
    evidence_packets,
    completeness_readiness=None,
    risk_review=None,
    sensitivity_review=None,
):
    evidence_results = evidence_packets.get("conditions_open_items", {}).get("selected_results", [])
    rows = []
    seen = set()
    completeness_readiness = completeness_readiness or {}
    risk_review = risk_review or {}
    sensitivity_review = sensitivity_review or {}

    for item in completeness_readiness.get("checklist", []):
        if item.get("traffic_light") not in [STATUS_RED, STATUS_AMBER]:
            continue
        append_open_item(
            rows=rows,
            seen=seen,
            category="Submission readiness",
            condition=f"Resolve checklist item: {item.get('required_item')}",
            traffic_light=item.get("traffic_light"),
            evidence_text=item.get("evidence_text"),
            source_chunks=item.get("source_chunks"),
            recommended_action=item.get("recommended_action"),
        )

    for item in risk_review.get("risk_register", []):
        if item.get("traffic_light") not in [STATUS_RED, STATUS_AMBER]:
            continue
        append_open_item(
            rows=rows,
            seen=seen,
            category="Risk",
            condition=f"Close risk gap: {item.get('risk')}",
            traffic_light=item.get("traffic_light"),
            evidence_text=item.get("mitigation_status"),
            source_chunks=item.get("source_chunks"),
            recommended_action=item.get("reviewer_challenge"),
        )

    for failure in sensitivity_review.get("validation_failures", []):
        append_open_item(
            rows=rows,
            seen=seen,
            category="Sensitivity",
            condition=f"Address sensitivity gap: {failure}",
            traffic_light=STATUS_RED,
            evidence_text="",
            source_chunks=sensitivity_review.get("source_chunks", [])[:3],
            recommended_action="Provide profile-relevant sensitivity evidence or confirm why the scenario is not applicable.",
        )

    weak_terms = [
        "not appended",
        "not included",
        "under negotiation",
        "under discussion",
        "not executed",
        "not assessed",
        "pending",
        "remain open",
        "not approved",
        "not quantified",
    ]
    for result in evidence_results:
        chunk_text = clean_text(result.get("chunk_text"))
        matched_terms = [term for term in weak_terms if normalize_text(term) in normalize_text(chunk_text)]
        if not matched_terms:
            continue
        append_open_item(
            rows=rows,
            seen=seen,
            category="Explicit weak evidence",
            condition=f"Resolve weak evidence language: {', '.join(matched_terms[:3])}",
            traffic_light=STATUS_AMBER,
            evidence_text=chunk_text,
            source_chunks=[get_source_reference(result)],
            recommended_action="Confirm status and attach final supporting evidence where applicable.",
        )

    module_status = STATUS_GREY
    if any(row["traffic_light"] == STATUS_RED for row in rows):
        module_status = STATUS_RED
    elif any(row["traffic_light"] == STATUS_AMBER for row in rows):
        module_status = STATUS_AMBER

    return {
        "module": "Conditions Precedent and Open Items",
        "traffic_light": module_status,
        "summary": "Open items are generated from readiness gaps, risk gaps, sensitivity validation failures and explicit weak evidence language.",
        "open_items": rows,
    }


def run_previous_request_accommodation_check(evidence_packets):
    evidence_results = evidence_packets.get("previous_request_accommodation", {}).get("selected_results", [])
    blob = get_evidence_blob(evidence_results)

    available = has_any(blob, ["previous request", "prior request", "accommodated", "follow-up action"])

    if not available:
        return {
            "module": "Previous Request Accommodation",
            "traffic_light": STATUS_GREY,
            "status": READINESS_NOT_ASSESSED,
            "summary": (
                "Prior request / follow-up action evidence was not found in the current corpus. "
                "The report does not assess whether previous requests were accommodated."
            ),
            "required_future_data": "Load previous_ic_requests or follow_up_actions table / corpus.",
            "findings": [],
        }

    return {
        "module": "Previous Request Accommodation",
        "traffic_light": STATUS_AMBER,
        "status": READINESS_PARTIAL,
        "summary": "Potential prior request evidence found. Manual validation recommended.",
        "source_chunks": [get_source_reference(item) for item in evidence_results[:8]],
    }


def run_prior_deal_comparison(evidence_packets):
    evidence_results = evidence_packets.get("prior_deal_comparison", {}).get("selected_results", [])
    blob = get_evidence_blob(evidence_results)

    available = has_any(blob, ["comparable prior", "historical deal", "previous transaction", "similarities", "differences"])

    if not available:
        return {
            "module": "Historical Deal Comparison",
            "traffic_light": STATUS_GREY,
            "status": READINESS_NOT_ASSESSED,
            "summary": (
                "Comparable prior-deal evidence was not found in the current corpus. "
                "The report does not claim automatic historical deal comparison."
            ),
            "required_future_data": "Load deal_metadata / prior IC materials with asset class, geography, size, stage, ownership and revenue model.",
            "comparables": [],
        }

    return {
        "module": "Historical Deal Comparison",
        "traffic_light": STATUS_AMBER,
        "status": READINESS_PARTIAL,
        "summary": "Potential comparable-deal evidence found. Manual validation recommended.",
        "source_chunks": [get_source_reference(item) for item in evidence_results[:8]],
    }


def run_macro_context_review(evidence_packets):
    evidence_results = evidence_packets.get("macro_context", {}).get("selected_results", [])
    blob = get_evidence_blob(evidence_results)

    available = has_any(blob, ["fx", "exchange rate", "gdp", "inflation", "macroeconomic", "country risk"])

    if not available:
        return {
            "module": "Macro / FX / GDP Context",
            "traffic_light": STATUS_GREY,
            "status": READINESS_NOT_ASSESSED,
            "summary": (
                "Macro / FX / GDP evidence was not found in the current corpus. "
                "The report does not fabricate macroeconomic conclusions."
            ),
            "required_future_data": "Connect or load whitelisted macro / FX / GDP source data.",
            "findings": [],
        }

    return {
        "module": "Macro / FX / GDP Context",
        "traffic_light": STATUS_AMBER,
        "status": READINESS_PARTIAL,
        "summary": "Potential macro evidence found. Manual validation recommended.",
        "source_chunks": [get_source_reference(item) for item in evidence_results[:8]],
    }


# ---------------------------------------------------------------------
# Executive summary and dashboard
# ---------------------------------------------------------------------

def build_traffic_light_dashboard(report_sections):
    dashboard = []

    for section_key, section_value in report_sections.items():
        if not isinstance(section_value, dict):
            continue

        dashboard.append(
            {
                "section_key": section_key,
                "section_name": section_value.get("module", section_key),
                "traffic_light": section_value.get("traffic_light", STATUS_GREY),
                "summary": section_value.get("summary", ""),
            }
        )

    return dashboard


def build_executive_summary(report_sections):
    dashboard = build_traffic_light_dashboard(report_sections)

    red_items = [item for item in dashboard if item["traffic_light"] == STATUS_RED]
    amber_items = [item for item in dashboard if item["traffic_light"] == STATUS_AMBER]
    grey_items = [item for item in dashboard if item["traffic_light"] == STATUS_GREY]

    if red_items:
        overall_status = STATUS_RED
    elif amber_items:
        overall_status = STATUS_AMBER
    elif grey_items:
        overall_status = STATUS_AMBER
    else:
        overall_status = STATUS_GREEN

    top_findings = []

    for item in red_items[:5]:
        top_findings.append({"traffic_light": STATUS_RED, "finding": f"{item['section_name']} requires immediate reviewer attention."})

    for item in amber_items[:5]:
        top_findings.append({"traffic_light": STATUS_AMBER, "finding": f"{item['section_name']} is partially addressed or contains unresolved issues."})

    for item in grey_items[:5]:
        top_findings.append({"traffic_light": STATUS_GREY, "finding": f"{item['section_name']} could not be fully assessed due to unavailable source/config data."})

    return {
        "module": "Executive IC Review Summary",
        "overall_readiness_status": overall_status,
        "summary": (
            "This output is an AI-enabled first-line IC review pack. It supports reviewer judgment "
            "and does not replace Investment Committee decision-making."
        ),
        "top_findings": top_findings,
        "red_count": len(red_items),
        "amber_count": len(amber_items),
        "grey_count": len(grey_items),
        "green_count": len([item for item in dashboard if item["traffic_light"] == STATUS_GREEN]),
    }


# ---------------------------------------------------------------------
# Evidence appendix
# ---------------------------------------------------------------------

def build_evidence_appendix(evidence_packets):
    source_map = {}

    for module_key, packet in evidence_packets.items():
        for result in packet.get("selected_results", []):
            source = get_source_reference(result)
            chunk_id = source.get("chunk_id")

            if not chunk_id:
                continue

            if chunk_id not in source_map:
                source_map[chunk_id] = source
                source_map[chunk_id]["used_in_modules"] = []

            source_map[chunk_id]["used_in_modules"].append(module_key)

    appendix = list(source_map.values())
    appendix = sorted(appendix, key=lambda x: str(x.get("chunk_id")))

    return appendix


# ---------------------------------------------------------------------
# Optional LLM summary
# ---------------------------------------------------------------------

def generate_llm_executive_summary(config_base, report_core):
    openai_api_key = config_base.get("openai_api_key")
    llm_model = config_base.get("llm_model")

    if not openai_api_key:
        return {
            "status": "not_generated",
            "reason": "OPENAI_API_KEY is not set.",
            "summary_text": "",
        }

    if not llm_model:
        return {
            "status": "not_generated",
            "reason": "llm_model is not set in config.",
            "summary_text": "",
        }

    prompt = f"""
You are preparing an AI first-line Investment Committee review summary.

Rules:
- Do not write a market research report.
- Do not make an investment recommendation.
- Do not invent facts.
- Use only the structured findings provided.
- Clearly flag Grey areas as not assessed due to missing source/config data.
- Focus on reviewer concerns, gaps, inconsistencies and follow-up questions.
- Keep the summary concise and IC-review oriented.

Structured findings:
{json.dumps(report_core, ensure_ascii=False, indent=2)[:25000]}

Generate:
1. One short executive paragraph.
2. Top red/amber issues.
3. Key IC follow-up questions.
4. Data limitations.
"""

    client = OpenAI(api_key=openai_api_key)

    try:
        response = client.responses.create(
            model=llm_model,
            input=prompt,
        )

        summary_text = getattr(response, "output_text", "")

        return {
            "status": "generated",
            "model": llm_model,
            "summary_text": summary_text,
        }

    except Exception as responses_error:
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You generate concise IC first-line review summaries from structured evidence.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            summary_text = response.choices[0].message.content

            return {
                "status": "generated",
                "model": llm_model,
                "summary_text": summary_text,
            }

        except Exception as chat_error:
            return {
                "status": "not_generated",
                "reason": (
                    f"Responses API error: {type(responses_error).__name__}: {str(responses_error)[:500]} | "
                    f"Chat Completions fallback error: {type(chat_error).__name__}: {str(chat_error)[:500]}"
                ),
                "summary_text": "",
            }



# ---------------------------------------------------------------------
# Terms and definitions
# ---------------------------------------------------------------------

def build_terms_and_definitions():
    terms = [
        ("AI", "Artificial intelligence; software that can perform tasks that normally require human reasoning or pattern recognition."),
        ("API", "Application programming interface; a structured way for software systems to call each other."),
        ("Amber", "Traffic-light status meaning partly addressed, weakly evidenced or requiring reviewer follow-up."),
        ("Audit log", "A record of the report run, model used, source chunks used and output generated."),
        ("BESS", "Battery Energy Storage System; batteries used to store electricity and release it later."),
        ("Benchmark", "A comparison point used to assess whether a value, assumption or cost is reasonable."),
        ("Brownfield", "An investment in an existing or operating asset, or expansion of an existing project."),
        ("Capex", "Capital expenditure; upfront investment cost to build or acquire an asset."),
        ("COD", "Commercial Operations Date; the date on which the project is expected to start commercial operation."),
        ("Contracted revenue", "Revenue backed by a contract such as a PPA rather than exposed to market prices."),
        ("Curtailment", "Reduction in electricity generation because the grid or buyer cannot accept all available output."),
        ("Data room", "A structured repository of transaction documents used for diligence and review."),
        ("Decommissioning", "Activities and costs required to dismantle an asset at the end of its useful life."),
        ("Debt / total cost", "Debt funding as a percentage of total project cost."),
        ("DSCR", "Debt Service Coverage Ratio; cash available for debt service divided by required debt payments."),
        ("EBITDA", "Earnings before interest, taxes, depreciation and amortization; a common operating-profit measure."),
        ("EMS", "Energy Management System; software used to monitor and optimize energy assets such as BESS."),
        ("Equity IRR", "Internal rate of return to equity investors after debt financing effects."),
        ("ESG", "Environmental, social and governance considerations."),
        ("EV / EBITDA", "Enterprise value divided by EBITDA; a valuation multiple."),
        ("External benchmark", "Benchmark evidence from outside the submitted transaction pack."),
        ("FX", "Foreign exchange; currency movement or exchange-rate exposure."),
        ("GDP", "Gross domestic product; a measure of economic output."),
        ("Green", "Traffic-light status meaning adequately addressed with clear evidence."),
        ("Greenfield", "A project developed and built from a new site or early-stage development base."),
        ("Grey", "Traffic-light status meaning not assessable because source data or configuration is unavailable."),
        ("Grid interconnection", "The connection between the project and the electricity grid."),
        ("IC", "Investment Committee; governance body that reviews and approves investment proposals."),
        ("IC memo", "Investment Committee memorandum; the core written submission for investment review."),
        ("IRR", "Internal Rate of Return; discount rate at which projected cash flows produce zero net present value."),
        ("JSON", "JavaScript Object Notation; structured data format used by APIs and front ends."),
        ("Letter of credit", "Bank-backed payment-security instrument used to support counterparty obligations."),
        ("LLM", "Large language model; AI model used to read, summarize and generate text."),
        ("Macro context", "Macroeconomic context such as GDP growth, inflation, exchange rates and country risk."),
        ("Majority stake", "Ownership interest above 50%, usually giving control or strong governance influence."),
        ("Merchant exposure", "Revenue exposure to market prices rather than fixed contracted prices."),
        ("MWh", "Megawatt-hour; unit of energy equal to one megawatt generated or consumed for one hour."),
        ("MW", "Megawatt; unit of power capacity."),
        ("MWac", "Megawatt alternating-current capacity; grid-side power capacity measure."),
        ("MWp", "Megawatt-peak; peak solar panel capacity under standard test conditions."),
        ("NPV", "Net Present Value; present value of expected cash flows minus investment cost."),
        ("O&M", "Operations and maintenance costs incurred to operate an asset."),
        ("Offtake", "Sale or purchase arrangement for the electricity or output from a project."),
        ("Offtaker", "Counterparty that buys the project output under an offtake agreement."),
        ("Payment security", "Contractual or bank-backed protection that supports payment obligations."),
        ("PPA", "Power Purchase Agreement; contract for sale and purchase of electricity."),
        ("POC", "Proof of concept; early version built to validate feasibility."),
        ("Project IRR", "Internal rate of return of the project before equity financing effects."),
        ("PV", "Photovoltaic solar technology that converts sunlight into electricity."),
        ("RAG", "Retrieval-augmented generation; AI approach that retrieves source evidence before generating answers."),
        ("Red", "Traffic-light status meaning missing, contradictory or approval-critical issue."),
        ("Reserved matters", "Important decisions requiring specified shareholder or board approval."),
        ("SCADA", "Supervisory Control and Data Acquisition; system used to monitor and control technical assets."),
        ("Sensitivity analysis", "Testing how investment outputs change when key assumptions change."),
        ("SharePoint", "Document management platform where source files may be stored."),
        ("Source chunk", "A retrieved text or table segment used as evidence for a finding."),
        ("SOW", "Statement of Work; document defining the client requirements and scope."),
        ("Submission gap", "Required or expected information that is missing or weak in the current submission."),
        ("Traffic-light status", "Green, Amber, Red or Grey rating used to summarize review status."),
        ("Traceability", "Ability to link a finding back to the exact source document, page, table or row."),
        ("Utility-scale", "Large project designed to supply electricity at grid scale rather than household scale."),
        ("Workbook", "Spreadsheet model or Excel file containing assumptions, calculations and outputs."),
    ]
    return [{"term": term, "definition": definition} for term, definition in terms]


def build_quality_status(report, review_config, transaction_profile):
    issues = []

    required_profile_fields = ["asset_class", "geography", "project_stage", "revenue_model"]
    for field_name in required_profile_fields:
        value = clean_text(transaction_profile.get(field_name))
        if not value or value == "not_identified":
            issues.append(f"Transaction profile field not identified: {field_name}")

    sensitivity_review = report.get("sensitivity_review", {})
    if sensitivity_review.get("validation_failures"):
        issues.extend(sensitivity_review.get("validation_failures", []))

    financial = report.get("financial_reconciliation", {})
    reconciliation_table = financial.get("reconciliation_table", [])
    if not reconciliation_table:
        issues.append("Financial reconciliation produced no extracted metric rows.")

    for row in financial.get("reconciliation_table", []):
        values_by_source = row.get("values_by_source", {})
        extracted_values = []
        for source_values in values_by_source.values():
            extracted_values.extend(source_values)

        if not extracted_values:
            issues.append(f"Financial metric has no extracted values: {row.get('metric_key')}")

        if row.get("traffic_light") == STATUS_RED or normalize_text(row.get("issue")).find("inconsistent") >= 0:
            issues.append(f"Financial metric conflict requires reconciliation: {row.get('metric_key')}")

    if not report.get("evidence_appendix"):
        issues.append("Evidence appendix is empty; source traceability is not available.")

    risk_ids = [
        row.get("risk_id")
        for row in report.get("risk_review", {}).get("risk_register", [])
    ]
    if "R08_LEGAL_TRANSACTION" not in risk_ids:
        issues.append("Legal transaction risk was not included in the risk register.")

    content_ready = len(issues) == 0
    return {
        "schema_ready": True,
        "content_ready": content_ready,
        "quality_issues": issues,
    }


def build_report_quality_status(report):
    return build_quality_status(
        report=report,
        review_config=report.get("review_config", {}),
        transaction_profile=report.get("transaction_profile", {}),
    )


# ---------------------------------------------------------------------
# Markdown formatter
# ---------------------------------------------------------------------

def traffic_icon(status):
    if status == STATUS_GREEN:
        return "🟢"
    if status == STATUS_AMBER:
        return "🟠"
    if status == STATUS_RED:
        return "🔴"
    if status == STATUS_GREY:
        return "⚪"
    return "⚪"


def format_sources_md(source_chunks):
    if not source_chunks:
        return "No source reference available."

    parts = []

    for src in source_chunks[:5]:
        parts.append(
            f"{src.get('chunk_id')} | {src.get('document_id')} | "
            f"{src.get('section_heading')} | {src.get('source_reference')}"
        )

    return "<br>".join(parts)


def format_markdown_report(report):
    lines = []

    lines.append("# AI First-Line IC Review Pack")
    lines.append("")
    lines.append(f"**Classification:** {report.get('classification')}")
    lines.append(f"**Transaction ID:** {report.get('transaction_id')}")
    lines.append(f"**Run ID:** {report.get('run_id')}")
    lines.append(f"**Generated at:** {report.get('generated_at')}")
    lines.append("")

    executive_summary = report.get("executive_summary", {})
    lines.append("## 1. Executive IC Review Summary")
    lines.append("")
    lines.append(f"**Overall status:** {traffic_icon(executive_summary.get('overall_readiness_status'))} {executive_summary.get('overall_readiness_status')}")
    lines.append("")
    lines.append(executive_summary.get("summary", ""))
    lines.append("")

    llm_summary = report.get("llm_executive_summary", {})
    if llm_summary.get("summary_text"):
        lines.append("### LLM-generated executive summary")
        lines.append("")
        lines.append(llm_summary.get("summary_text"))
        lines.append("")

    lines.append("### Top findings")
    lines.append("")
    for finding in executive_summary.get("top_findings", []):
        lines.append(f"- {traffic_icon(finding.get('traffic_light'))} {finding.get('finding')}")
    lines.append("")

    lines.append("## 2. Traffic-Light Dashboard")
    lines.append("")
    lines.append("| Section | Status | Summary |")
    lines.append("|---|---|---|")
    for item in report.get("traffic_light_dashboard", []):
        lines.append(
            f"| {item.get('section_name')} | {traffic_icon(item.get('traffic_light'))} {item.get('traffic_light')} | {clean_text(item.get('summary'))} |"
        )
    lines.append("")

    completeness = report.get("completeness_readiness", {})
    lines.append("## 3. Completeness & Readiness Check")
    lines.append("")
    lines.append("| Required item | Status | Weakness | Recommended action | Sources |")
    lines.append("|---|---|---|---|---|")
    for row in completeness.get("checklist", []):
        lines.append(
            f"| {row.get('required_item')} | {traffic_icon(row.get('traffic_light'))} {row.get('status')} | "
            f"{clean_text(row.get('weakness'))} | {clean_text(row.get('recommended_action'))} | "
            f"{format_sources_md(row.get('source_chunks'))} |"
        )
    lines.append("")

    strategy = report.get("strategy_fit_assessment", {})
    lines.append("## 4. Strategy & Fit Assessment")
    lines.append("")
    lines.append("| Criterion | Status | Finding | Sources |")
    lines.append("|---|---|---|---|")
    for row in strategy.get("criteria_assessment", []):
        lines.append(
            f"| {row.get('criterion')} | {traffic_icon(row.get('traffic_light'))} {row.get('status')} | "
            f"{clean_text(row.get('finding'))} | {format_sources_md(row.get('source_chunks'))} |"
        )
    lines.append("")

    icq = report.get("historical_ic_question_coverage", {})
    lines.append("## 5. Historical IC Question Coverage")
    lines.append("")
    lines.append(f"**Limitation:** {icq.get('limitation', '')}")
    lines.append("")
    lines.append("| Theme | Likely IC question | Coverage | Sources |")
    lines.append("|---|---|---|---|")
    for row in icq.get("question_coverage", []):
        lines.append(
            f"| {row.get('theme')} | {row.get('likely_ic_question')} | "
            f"{traffic_icon(row.get('traffic_light'))} {row.get('coverage_status')} | "
            f"{format_sources_md(row.get('source_chunks'))} |"
        )
    lines.append("")

    fin = report.get("financial_reconciliation", {})
    lines.append("## 6. Key Inconsistencies & Financial Metric Reconciliation")
    lines.append("")
    lines.append("| Metric | Values by source | Preferred review value | Status | Issue |")
    lines.append("|---|---|---:|---|---|")
    for row in fin.get("reconciliation_table", []):
        lines.append(
            f"| {row.get('metric_name')} | {json.dumps(row.get('values_by_source'), ensure_ascii=False)} | "
            f"{row.get('preferred_value_for_review')} {row.get('unit')} ({row.get('preferred_source')}) | "
            f"{traffic_icon(row.get('traffic_light'))} {row.get('traffic_light')} | {row.get('issue')} |"
        )
    lines.append("")

    market = report.get("market_offtake_revenue_review", {})
    lines.append("## 7. Market, Offtake and Revenue Review")
    lines.append("")
    for finding in market.get("findings", []):
        lines.append(f"- {traffic_icon(finding.get('traffic_light'))} **{finding.get('finding')}** — {finding.get('explanation')}")
    lines.append("")

    sens = report.get("sensitivity_review", {})
    lines.append("## 8. Sensitivity and Downside Protection Review")
    lines.append("")
    lines.append(f"**Submission gap flag:** {sens.get('submission_gap_flag')}")
    lines.append("")
    lines.append("| Scenario | Equity IRR | Minimum DSCR | Included in submission | Source |")
    lines.append("|---|---:|---:|---|---|")
    for row in sens.get("sensitivity_cases", []):
        src = row.get("source_chunk", {})
        lines.append(
            f"| {row.get('scenario')} | {row.get('equity_irr_pct')} | {row.get('minimum_dscr_x')} | "
            f"{row.get('included_in_submission')} | {src.get('chunk_id')} |"
        )
    lines.append("")

    risk = report.get("risk_review", {})
    lines.append("## 9. Risk Register and Reviewer Prompts")
    lines.append("")
    lines.append("| Risk | Rating | Mitigation/status | Reviewer challenge | Status | Sources |")
    lines.append("|---|---|---|---|---|---|")
    for row in risk.get("risk_register", []):
        lines.append(
            f"| {row.get('risk')} | {row.get('rating')} | {clean_text(row.get('mitigation_status'))} | "
            f"{row.get('reviewer_challenge')} | {traffic_icon(row.get('traffic_light'))} {row.get('traffic_light')} | "
            f"{format_sources_md(row.get('source_chunks'))} |"
        )
    lines.append("")

    valuation = report.get("valuation_review", {})
    external = report.get("external_benchmark_review", {})
    lines.append("## 10. Valuation and Benchmark Review")
    lines.append("")
    lines.append("### Client-side valuation")
    for finding in valuation.get("findings", []):
        lines.append(f"- {traffic_icon(finding.get('traffic_light'))} **{finding.get('finding')}** — {finding.get('explanation')}")
    lines.append("")
    lines.append("### External benchmark review")
    for finding in external.get("findings", []):
        lines.append(f"- {traffic_icon(finding.get('traffic_light'))} **{finding.get('benchmark_area')}** — {finding.get('finding')} Allowed use: {finding.get('allowed_use')}")
    lines.append("")

    previous = report.get("previous_request_accommodation", {})
    prior = report.get("prior_deal_comparison", {})
    macro = report.get("macro_context_review", {})

    lines.append("## 11. Grey / Not Assessed SOW Modules")
    lines.append("")
    lines.append("| Module | Status | Reason | Required future data |")
    lines.append("|---|---|---|---|")
    for section in [previous, prior, macro]:
        lines.append(
            f"| {section.get('module')} | {traffic_icon(section.get('traffic_light'))} {section.get('status')} | "
            f"{clean_text(section.get('summary'))} | {clean_text(section.get('required_future_data'))} |"
        )
    lines.append("")

    open_items = report.get("conditions_precedent_open_items", {})
    lines.append("## 12. Conditions Precedent and Open Items")
    lines.append("")
    lines.append("| Category | Condition | Priority | Status | Sources |")
    lines.append("|---|---|---|---|---|")
    for row in open_items.get("open_items", []):
        lines.append(
            f"| {row.get('category')} | {row.get('condition')} | {row.get('priority')} | "
            f"{traffic_icon(row.get('traffic_light'))} {row.get('status')} | "
            f"{format_sources_md(row.get('source_chunks'))} |"
        )
    lines.append("")

    lines.append("## 13. Evidence Appendix")
    lines.append("")
    lines.append("| Chunk ID | Document ID | Corpus zone | Section | Source reference | Used in modules |")
    lines.append("|---|---|---|---|---|---|")
    for src in report.get("evidence_appendix", []):
        lines.append(
            f"| {src.get('chunk_id')} | {src.get('document_id')} | {src.get('corpus_zone')} | "
            f"{src.get('section_heading')} | {src.get('source_reference')} | "
            f"{', '.join(src.get('used_in_modules', []))} |"
        )
    lines.append("")

    lines.append("## 14. Audit Metadata")
    lines.append("")
    audit = report.get("audit_metadata", {})
    lines.append("```json")
    lines.append(json.dumps(audit, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    lines.append("## 15. Terms and Definitions")
    lines.append("")
    lines.append("| Term | Simple definition |")
    lines.append("|---|---|")
    for item in report.get("terms_and_definitions", []):
        lines.append(f"| {item.get('term')} | {item.get('definition')} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Audit log and output persistence
# ---------------------------------------------------------------------

def write_audit_log(config_base, report):
    db_url = config_base.get("db_url")

    if not db_url:
        return {"status": "not_written", "reason": "db_url is not configured."}

    try:
        engine = create_engine(url=db_url, pool_pre_ping=True)

        create_sql = text("""
            CREATE TABLE IF NOT EXISTS report_generation_audit_log (
                audit_id BIGSERIAL PRIMARY KEY,
                run_id TEXT,
                transaction_id TEXT,
                report_type TEXT,
                report_version TEXT,
                classification TEXT,
                llm_model TEXT,
                generated_at TIMESTAMP WITH TIME ZONE,
                source_chunk_count INTEGER,
                report_json JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)

        insert_sql = text("""
            INSERT INTO report_generation_audit_log (
                run_id,
                transaction_id,
                report_type,
                report_version,
                classification,
                llm_model,
                generated_at,
                source_chunk_count,
                report_json
            )
            VALUES (
                :run_id,
                :transaction_id,
                :report_type,
                :report_version,
                :classification,
                :llm_model,
                :generated_at,
                :source_chunk_count,
                CAST(:report_json AS jsonb)
            );
        """)

        with engine.begin() as conn:
            conn.execute(create_sql)
            conn.execute(
                insert_sql,
                {
                    "run_id": report.get("run_id"),
                    "transaction_id": report.get("transaction_id"),
                    "report_type": report.get("report_type"),
                    "report_version": report.get("report_generation_version"),
                    "classification": report.get("classification"),
                    "llm_model": report.get("audit_metadata", {}).get("llm_model"),
                    "generated_at": report.get("generated_at"),
                    "source_chunk_count": len(report.get("evidence_appendix", [])),
                    "report_json": json.dumps(report, ensure_ascii=False),
                },
            )

        return {"status": "written", "table": "report_generation_audit_log"}

    except Exception as e:
        return {"status": "not_written", "reason": f"{type(e).__name__}: {str(e)[:1000]}"}


def save_report_outputs(config_pack, report, markdown_report):
    draft_report_dir = config_pack["draft_report_dir"]
    run_id = report["run_id"]

    json_path = draft_report_dir / f"{run_id}.json"
    markdown_path = draft_report_dir / f"{run_id}.md"
    output_files = {"json_path": str(json_path), "markdown_path": str(markdown_path)}
    report["output_files"] = output_files

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)

    return output_files


# ---------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------

def generate_investment_ic_review_report(
    transaction_id,
    use_llm_summary=True,
    write_audit=True,
):
    config_pack = get_config_pack(client_data=transaction_id)
    config_base = config_pack["config_base"]

    run_id = make_run_id(transaction_id)

    profile_packet = run_neutral_profile_retrieval(transaction_id)
    transaction_profile = infer_transaction_profile(
        transaction_id=transaction_id,
        profile_results=profile_packet.get("selected_results", []),
    )
    review_config = load_review_config(transaction_profile)

    evidence_packets = run_retrieval_plan(
        transaction_profile=transaction_profile,
        review_config=review_config,
        profile_packet=profile_packet,
    )

    financial_reconciliation = run_financial_reconciliation(evidence_packets, transaction_profile)
    sensitivity_review = run_sensitivity_review(evidence_packets, review_config, transaction_profile)
    completeness_readiness = run_completeness_check(
        review_config,
        evidence_packets,
        financial_reconciliation=financial_reconciliation,
        sensitivity_review=sensitivity_review,
    )
    strategy_fit_assessment = run_strategy_fit_assessment(review_config, evidence_packets)
    historical_ic_question_coverage = run_historical_ic_question_check(review_config, evidence_packets)
    market_offtake_revenue_review = run_revenue_model_review(evidence_packets, transaction_profile)
    risk_review = run_risk_review(review_config, evidence_packets)
    valuation_review = run_client_valuation_review(evidence_packets)
    external_benchmark_review = run_external_benchmark_review(evidence_packets, transaction_profile, review_config)
    previous_request_accommodation = run_previous_request_accommodation_check(evidence_packets)
    prior_deal_comparison = run_prior_deal_comparison(evidence_packets)
    macro_context_review = run_macro_context_review(evidence_packets)
    conditions_precedent_open_items = build_open_items_register(
        evidence_packets=evidence_packets,
        completeness_readiness=completeness_readiness,
        risk_review=risk_review,
        sensitivity_review=sensitivity_review,
    )

    report_sections_for_dashboard = {
        "completeness_readiness": completeness_readiness,
        "strategy_fit_assessment": strategy_fit_assessment,
        "historical_ic_question_coverage": historical_ic_question_coverage,
        "financial_reconciliation": financial_reconciliation,
        "market_offtake_revenue_review": market_offtake_revenue_review,
        "sensitivity_review": sensitivity_review,
        "risk_review": risk_review,
        "valuation_review": valuation_review,
        "external_benchmark_review": external_benchmark_review,
        "previous_request_accommodation": previous_request_accommodation,
        "prior_deal_comparison": prior_deal_comparison,
        "macro_context_review": macro_context_review,
        "conditions_precedent_open_items": conditions_precedent_open_items,
    }

    executive_summary = build_executive_summary(report_sections_for_dashboard)
    traffic_light_dashboard = build_traffic_light_dashboard(report_sections_for_dashboard)
    evidence_appendix = build_evidence_appendix(evidence_packets)
    terms_and_definitions = build_terms_and_definitions()

    audit_metadata = {
        "run_id": run_id,
        "transaction_id": transaction_id,
        "report_type": REPORT_TYPE,
        "report_generation_version": REPORT_GENERATION_VERSION,
        "classification": CLASSIFICATION,
        "generated_at": utc_now_iso(),
        "llm_provider": config_base.get("llm_provider"),
        "llm_model": config_base.get("llm_model"),
        "retrieval_modules": list(evidence_packets.keys()),
        "source_chunk_count": len(evidence_appendix),
        "important_controls": [
            "No investment recommendation is made.",
            "Historical IC learning is not claimed if historical logs are unavailable.",
            "External EV/EBITDA benchmark support is not claimed unless evidence exists.",
            "Macro / FX / GDP conclusions are not fabricated where source data is unavailable.",
            "Every finding should be traceable through chunk_id/document_id/source_reference.",
        ],
    }

    report = {
        "run_id": run_id,
        "transaction_id": transaction_id,
        "report_type": REPORT_TYPE,
        "classification": CLASSIFICATION,
        "report_generation_version": REPORT_GENERATION_VERSION,
        "generated_at": audit_metadata["generated_at"],
        "transaction_profile": transaction_profile,
        "review_config": review_config,
        "executive_summary": executive_summary,
        "traffic_light_dashboard": traffic_light_dashboard,
        "completeness_readiness": completeness_readiness,
        "strategy_fit_assessment": strategy_fit_assessment,
        "historical_ic_question_coverage": historical_ic_question_coverage,
        "financial_reconciliation": financial_reconciliation,
        "market_offtake_revenue_review": market_offtake_revenue_review,
        "sensitivity_review": sensitivity_review,
        "risk_review": risk_review,
        "valuation_review": valuation_review,
        "external_benchmark_review": external_benchmark_review,
        "previous_request_accommodation": previous_request_accommodation,
        "prior_deal_comparison": prior_deal_comparison,
        "macro_context_review": macro_context_review,
        "conditions_precedent_open_items": conditions_precedent_open_items,
        "evidence_appendix": evidence_appendix,
        "audit_metadata": audit_metadata,
        "terms_and_definitions": terms_and_definitions,
    }

    if use_llm_summary:
        llm_summary = generate_llm_executive_summary(
            config_base=config_base,
            report_core={
                "executive_summary": executive_summary,
                "traffic_light_dashboard": traffic_light_dashboard,
                "completeness_readiness": completeness_readiness,
                "financial_reconciliation": financial_reconciliation,
                "market_offtake_revenue_review": market_offtake_revenue_review,
                "sensitivity_review": sensitivity_review,
                "risk_review": risk_review,
                "valuation_review": valuation_review,
                "external_benchmark_review": external_benchmark_review,
                "conditions_precedent_open_items": conditions_precedent_open_items,
                "grey_modules": {
                    "previous_request_accommodation": previous_request_accommodation,
                    "prior_deal_comparison": prior_deal_comparison,
                    "macro_context_review": macro_context_review,
                },
            },
        )
    else:
        llm_summary = {"status": "not_requested", "summary_text": ""}

    report["llm_executive_summary"] = llm_summary
    report["report_quality_status"] = build_quality_status(
        report=report,
        review_config=review_config,
        transaction_profile=transaction_profile,
    )

    audit_write_result = {"status": "not_requested"}
    if write_audit:
        audit_write_result = write_audit_log(config_base, report)

    report["audit_write_result"] = audit_write_result

    markdown_report = format_markdown_report(report)
    saved_outputs = save_report_outputs(config_pack, report, markdown_report)
    report_status = "ok"
    if not report.get("report_quality_status", {}).get("content_ready"):
        report_status = "validation_failed"

    return {
        "message": "AI first-line IC review report generated.",
        "status": report_status,
        "run_id": run_id,
        "transaction_id": transaction_id,
        "overall_readiness_status": executive_summary.get("overall_readiness_status"),
        "report_quality_status": report.get("report_quality_status"),
        "output_files": saved_outputs,
        "audit_write_result": audit_write_result,
        "report": report,
    }


if __name__ == "__main__":
    output = generate_investment_ic_review_report(
        transaction_id="TXN_ADDC_001",
        use_llm_summary=True,
        write_audit=True,
    )

    print(json.dumps(output, indent=2, ensure_ascii=False))
