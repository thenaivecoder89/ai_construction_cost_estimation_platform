"""
Country macro-risk clustering -> LLM narrative orchestrator

Purpose:
- Imports and runs country_macro_clustering.run_country_macro_clustering().
- Sends the generated clustering JSON + base64 graph images to the configured OpenAI LLM.
- Returns the LLM narrative as JSON plus the original base64 graph images for front-end display.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Required project-package imports.
from global_rag.scripts import config
from global_rag.scripts import country_macro_clustering


DEFAULT_ASSET_CLASSES = ["Solar PV", "BESS", "Onshore Wind", "Offshore Wind"]
DEFAULT_REGIONAL_LENS = [
    "UAE",
    "GCC stable-peg economies",
    "high-growth emerging markets",
    "inflation / currency stress economies",
    "low-growth / volatile economies",
]


LLM_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "executive_summary": {"type": "string"},
        "model_interpretation": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kmeans": {"type": "string"},
                "hierarchical_clustering": {"type": "string"},
                "gaussian_mixture_model": {"type": "string"},
                "cross_model_consensus": {"type": "string"},
                "cross_model_divergence": {"type": "string"},
            },
            "required": [
                "kmeans",
                "hierarchical_clustering",
                "gaussian_mixture_model",
                "cross_model_consensus",
                "cross_model_divergence",
            ],
        },
        "regional_implications": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "region_or_country_segment": {"type": "string"},
                    "countries": {"type": "array", "items": {"type": "string"}},
                    "macro_risk_readout": {"type": "string"},
                    "project_development_implication": {"type": "string"},
                    "asset_class_implications": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "asset_class": {
                                    "type": "string",
                                    "enum": DEFAULT_ASSET_CLASSES,
                                },
                                "what_to_modify_or_consider": {"type": "string"},
                                "commercial_structuring": {"type": "string"},
                                "financing_and_bankability": {"type": "string"},
                                "risk_mitigants": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "asset_class",
                                "what_to_modify_or_consider",
                                "commercial_structuring",
                                "financing_and_bankability",
                                "risk_mitigants",
                            ],
                        },
                    },
                },
                "required": [
                    "region_or_country_segment",
                    "countries",
                    "macro_risk_readout",
                    "project_development_implication",
                    "asset_class_implications",
                ],
            },
        },
        "uae_focus": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "uae_cluster_position": {"type": "string"},
                "why_uae_matters_for_project_screening": {"type": "string"},
                "asset_class_actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "asset_class": {
                                "type": "string",
                                "enum": DEFAULT_ASSET_CLASSES,
                            },
                            "modifications_or_considerations": {"type": "string"},
                            "tariff_or_revenue_model_implication": {"type": "string"},
                            "capex_opex_and_contingency_implication": {"type": "string"},
                            "due_diligence_focus_areas": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "asset_class",
                            "modifications_or_considerations",
                            "tariff_or_revenue_model_implication",
                            "capex_opex_and_contingency_implication",
                            "due_diligence_focus_areas",
                        ],
                    },
                },
            },
            "required": [
                "uae_cluster_position",
                "why_uae_matters_for_project_screening",
                "asset_class_actions",
            ],
        },
        "country_watchlist": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "country": {"type": "string"},
                    "reason_for_attention": {"type": "string"},
                    "renewables_implication": {"type": "string"},
                },
                "required": [
                    "country",
                    "reason_for_attention",
                    "renewables_implication",
                ],
            },
        },
        "front_end_cards": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": ["informational", "low", "medium", "high"],
                    },
                },
                "required": ["title", "body", "severity"],
            },
        },
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "executive_summary",
        "model_interpretation",
        "regional_implications",
        "uae_focus",
        "country_watchlist",
        "front_end_cards",
        "limitations",
    ],
}


SYSTEM_PROMPT = """
You are a senior infrastructure and energy investment advisor.

You convert macro-risk clustering outputs into practical implications for renewable-energy and energy-transition projects.
Focus on these asset classes only:
- Solar PV
- BESS
- Onshore Wind
- Offshore Wind

Your interpretation must be grounded in the supplied clustering JSON and graph images.
Do not invent cluster assignments. Use the clustering datasets as the source of truth.
If K-Means, Hierarchical Clustering, and Gaussian Mixture Model disagree, explicitly explain the disagreement and what it means for investment decision-making.

Write for a client-facing investment, strategy, and due-diligence audience.
Connect macro-risk signals to tariff design, FX assumptions, offtake structure, EPC/O&M risk, financing, contingency, grid risk, permitting, import exposure, and downside-case modelling.
Maintain a specific UAE focus while still explaining implications by broader country segment / region.
Return only JSON matching the requested schema.
Keep all narrative fields concise. Avoid long paragraphs; prefer 2-4 sentences per field.
""".strip()


def _get_openai_client_and_model() -> tuple[OpenAI, str]:
    """Read OpenAI API key and model from deployed config.py."""
    cfg = config.config_base()

    api_key = cfg.get("openai_api_key")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from config.config_base().")

    model = cfg.get("llm_model")
    if not model:
        raise RuntimeError("llm_model is missing from config.config_base().")

    # Timeout gives the LLM enough time for the larger structured JSON response.
    return OpenAI(api_key=api_key, timeout=180.0), model


def _normalise_base64_image(graph_payload: Any) -> Optional[str]:
    """Extract raw base64 from graph payload generated by country_macro_clustering."""
    value = graph_payload

    if isinstance(value, dict):
        value = value.get("image_base64") or value.get("base64") or value.get("data")

    if not isinstance(value, str) or not value.strip():
        return None

    image_text = value.strip()
    if image_text.startswith("data:image/"):
        image_text = image_text.split(",", 1)[-1]

    if not re.fullmatch(r"[A-Za-z0-9+/=\n\r]+", image_text):
        return None

    return image_text.replace("\n", "").replace("\r", "")


def _strip_graph_images_for_text_payload(clustering_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep metadata and model datasets in the JSON text payload.
    Exclude base64 graph strings from text to keep the prompt compact.
    Graphs are provided separately as image inputs.
    """
    return {
        "metadata": clustering_result.get("metadata", {}),
        "datasets": clustering_result.get("datasets", {}),
    }


def _safe_json_dumps(payload: Any, max_chars: int = 120_000) -> str:
    """Compact JSON serializer with a basic safety cap for LLM input size."""
    text = json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
    if len(text) > max_chars:
        return text[:max_chars] + "...<TRUNCATED_FOR_LLM_CONTEXT>"
    return text


def _build_llm_input(
    clustering_json: Dict[str, Any],
    graph_images: Dict[str, Any],
    focus_country: str,
    include_graphs_in_llm: bool,
) -> List[Dict[str, Any]]:
    """Build OpenAI Responses API input using clustering JSON plus graph images."""
    user_payload = {
        "task": "Interpret country macro-risk clustering for renewable energy project implications.",
        "focus_country": focus_country,
        "asset_classes": DEFAULT_ASSET_CLASSES,
        "regional_lens": DEFAULT_REGIONAL_LENS,
        "clustering_json": clustering_json,
    }

    user_content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": _safe_json_dumps(user_payload)}
    ]

    if include_graphs_in_llm:
        for graph_name, graph_payload in graph_images.items():
            image_base64 = _normalise_base64_image(graph_payload)
            if not image_base64:
                continue

            user_content.append(
                {
                    "type": "input_text",
                    "text": f"Graph image generated by the {graph_name} analytical model.",
                }
            )
            user_content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image_base64}",
                }
            )

    return [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
        },
        {"role": "user", "content": user_content},
    ]


def _extract_response_text(response: Any) -> str:
    """Return response.output_text only when the OpenAI response completed cleanly."""
    status = getattr(response, "status", None)
    incomplete_details = getattr(response, "incomplete_details", None)

    if status == "incomplete":
        raise RuntimeError(
            "OpenAI response was incomplete before valid JSON was produced. "
            f"Incomplete details: {incomplete_details}. "
            "Increase max_output_tokens or reduce the requested narrative length."
        )

    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise RuntimeError(
            "OpenAI response did not contain output_text. "
            f"Response status: {status}. Response object: {response}"
        )

    return output_text


def _parse_llm_json(output_text: str) -> Dict[str, Any]:
    """Convert model output text into a Python dictionary."""
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"LLM returned non-JSON output: {exc}. Raw output: {output_text[:1000]}"
        ) from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("LLM output JSON was not a JSON object.")

    return parsed


def llm_call(
    n_clusters: int = 4,
    schema: str = "public",
    table_name: str = "country_features_raw",
    focus_country: str = "UAE",
    include_graphs_in_llm: bool = True,
    max_output_tokens: int = 16000,
) -> Dict[str, Any]:
    """
    Run clustering, send clustering outputs to the LLM, and return JSON narrative + graphs.

    Returns:
        {
            "llm_output": {...},
            "graphs": {
                "kmeans": {"mime_type": "image/png", "encoding": "base64", "image_base64": "..."},
                "hierarchical": {"mime_type": "image/png", "encoding": "base64", "image_base64": "..."},
                "gmm": {"mime_type": "image/png", "encoding": "base64", "image_base64": "..."}
            },
            "metadata": {...}
        }
    """

    # 1. Run the clustering program and capture JSON + graph output.
    clustering_result = country_macro_clustering.run_country_macro_clustering(
        n_clusters=n_clusters,
        schema=schema,
        table_name=table_name,
    )

    clustering_json = _strip_graph_images_for_text_payload(clustering_result)
    graph_images = clustering_result.get("graphs", {})

    # 2. Call the OpenAI LLM configured in config.py.
    client, model = _get_openai_client_and_model()
    response_input = _build_llm_input(
        clustering_json=clustering_json,
        graph_images=graph_images,
        focus_country=focus_country,
        include_graphs_in_llm=include_graphs_in_llm,
    )

    response = client.responses.create(
        model=model,
        input=response_input,
        text={
            "format": {
                "type": "json_schema",
                "name": "country_macro_risk_asset_implications",
                "schema": LLM_OUTPUT_SCHEMA,
                "strict": True,
            }
        },
        max_output_tokens=max_output_tokens,
    )

    # 3. Convert LLM output into JSON.
    response_text = _extract_response_text(response)
    llm_output = _parse_llm_json(response_text)

    # 4. Return LLM JSON + graph images for further processing / front-end display.
    return {
        "llm_output": llm_output,
        "graphs": graph_images,
        "metadata": {
            "llm_model": model,
            "focus_country": focus_country,
            "n_clusters": n_clusters,
            "source_table": f"{schema}.{table_name}",
            "graphs_sent_to_llm": include_graphs_in_llm,
            "max_output_tokens": max_output_tokens,
            "graphs_returned": list(graph_images.keys()),
            "clustering_metadata": clustering_result.get("metadata", {}),
        },
    }