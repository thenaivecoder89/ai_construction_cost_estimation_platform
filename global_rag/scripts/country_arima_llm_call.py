"""
Country ARIMA forecasts -> LLM narrative orchestrator

Purpose:
- Imports and runs country_arima_forecasting.run_country_arima_forecasts().
- Sends the generated forecast JSON + base64 graph images to the configured OpenAI LLM.
- Returns the LLM narrative as JSON plus the original base64 graph images for front-end display.

Important:
- No FastAPI code is included here.
- The API layer should import and call llm_call().
- This module does not query the forecast source tables directly; it delegates that to
  country_arima_forecasting.run_country_arima_forecasts().
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

# Required project-package imports.
from global_rag.scripts import config
from global_rag.scripts import country_arima_forecasting


DEFAULT_COUNTRY_CODES: List[str] = ["ARE", "AUS", "EGY", "IND", "JOR", "KHM", "SAU", "THA", "ZAF"]
DEFAULT_TARGETS: List[str] = ["gdp_growth", "inflation", "fx_depreciation"]
DEFAULT_MAX_OUTPUT_TOKENS = 16000


LLM_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "executive_summary": {"type": "string"},
        "methodology_readout": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "model_used": {"type": "string"},
                "why_simple_arima": {"type": "string"},
                "naive_benchmark_interpretation": {"type": "string"},
                "forecast_band_interpretation": {"type": "string"},
                "fx_modelling_note": {"type": "string"},
            },
            "required": [
                "model_used",
                "why_simple_arima",
                "naive_benchmark_interpretation",
                "forecast_band_interpretation",
                "fx_modelling_note",
            ],
        },
        "target_level_findings": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "gdp_growth": {"type": "string"},
                "inflation": {"type": "string"},
                "fx_depreciation": {"type": "string"},
            },
            "required": ["gdp_growth", "inflation", "fx_depreciation"],
        },
        "country_forecast_readout": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "countryiso3code": {"type": "string"},
                    "country": {"type": "string"},
                    "overall_macro_forecast_signal": {"type": "string"},
                    "gdp_growth_outlook": {"type": "string"},
                    "inflation_outlook": {"type": "string"},
                    "fx_depreciation_outlook": {"type": "string"},
                    "key_forecast_risks": {"type": "array", "items": {"type": "string"}},
                    "investment_implication": {"type": "string"},
                },
                "required": [
                    "countryiso3code",
                    "country",
                    "overall_macro_forecast_signal",
                    "gdp_growth_outlook",
                    "inflation_outlook",
                    "fx_depreciation_outlook",
                    "key_forecast_risks",
                    "investment_implication",
                ],
            },
        },
        "uae_focus": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "forecast_position": {"type": "string"},
                "gdp_growth_implication": {"type": "string"},
                "inflation_implication": {"type": "string"},
                "fx_implication": {"type": "string"},
                "recommended_due_diligence_actions": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "forecast_position",
                "gdp_growth_implication",
                "inflation_implication",
                "fx_implication",
                "recommended_due_diligence_actions",
            ],
        },
        "forecast_watchlist": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "countryiso3code": {"type": "string"},
                    "country": {"type": "string"},
                    "watchlist_reason": {"type": "string"},
                    "trigger_metric": {"type": "string"},
                    "commercial_or_financing_implication": {"type": "string"},
                },
                "required": [
                    "countryiso3code",
                    "country",
                    "watchlist_reason",
                    "trigger_metric",
                    "commercial_or_financing_implication",
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
                    "severity": {"type": "string", "enum": ["informational", "low", "medium", "high"]},
                },
                "required": ["title", "body", "severity"],
            },
        },
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "executive_summary",
        "methodology_readout",
        "target_level_findings",
        "country_forecast_readout",
        "uae_focus",
        "forecast_watchlist",
        "front_end_cards",
        "limitations",
    ],
}


SYSTEM_PROMPT = """
You are a senior macroeconomic forecasting and infrastructure investment advisor.

You interpret annual ARIMA forecast outputs for country-level GDP growth, inflation, and FX depreciation.
Use the forecast datasets and graph images as the source of truth.
Do not invent countries, targets, forecast values, model orders, or benchmark results.

Important interpretation rules:
- GDP growth forecasts indicate expected growth momentum and demand backdrop.
- Inflation forecasts indicate cost escalation, O&M pressure, and tariff indexation need.
- FX depreciation forecasts indicate currency weakness, imported capex exposure, debt-service mismatch, and hedging need.
- For FX, positive depreciation means the local currency weakens against USD.
- Compare ARIMA forecasts against the naive benchmark where relevant.
- Use forecast bands to discuss uncertainty, not just point forecasts.
- Treat these as screening outputs because annual time series are short.

Write for a client-facing investment, strategy, and due-diligence audience.
Return only JSON matching the requested schema.
""".strip()


def _get_openai_client_and_model() -> Tuple[OpenAI, str]:
    """Read OpenAI API key and model from deployed config.py."""
    cfg = config.config_base()

    api_key = cfg.get("openai_api_key")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from config.config_base().")

    model = cfg.get("llm_model")
    if not model:
        raise RuntimeError("llm_model is missing from config.config_base().")

    return OpenAI(api_key=api_key), model


def _normalise_base64_image(graph_payload: Any) -> Optional[str]:
    """Extract raw base64 from graph payload generated by country_arima_forecasting."""
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


def _strip_graph_images_for_text_payload(forecast_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep metadata, datasets, and model summary in the JSON text payload.
    Exclude base64 graph strings from text to keep the prompt compact.
    Graphs are provided separately as image inputs.
    """
    return {
        "metadata": forecast_result.get("metadata", {}),
        "datasets": forecast_result.get("datasets", {}),
        "model_summary": forecast_result.get("model_summary", []),
    }


def _safe_json_dumps(payload: Any, max_chars: int = 180_000) -> str:
    """Compact JSON serializer with a safety cap for LLM input size."""
    text = json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
    if len(text) > max_chars:
        return text[:max_chars] + "...<TRUNCATED_FOR_LLM_CONTEXT>"
    return text


def _build_llm_input(
    forecast_json: Dict[str, Any],
    graph_images: Dict[str, Any],
    focus_country: str,
    include_graphs_in_llm: bool,
    max_graphs_to_send: Optional[int],
) -> List[Dict[str, Any]]:
    """Build OpenAI Responses API input using forecast JSON plus graph images."""
    user_payload = {
        "task": "Interpret country ARIMA forecast outputs for GDP growth, inflation, and FX depreciation.",
        "focus_country": focus_country,
        "forecast_json": forecast_json,
        "instructions": [
            "Use forecast datasets and model summary as the primary source of truth.",
            "Use graph images as visual confirmation only; do not infer unsupported values from visuals.",
            "Comment on ARIMA forecast, naive benchmark, 80% and 95% forecast bands where material.",
            "Emphasize implications for country screening, tariff indexation, FX assumptions, financing, and downside-case modelling.",
        ],
    }

    user_content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": _safe_json_dumps(user_payload)}
    ]

    if include_graphs_in_llm:
        graph_items = list(graph_images.items())
        if max_graphs_to_send is not None:
            graph_items = graph_items[: max(0, int(max_graphs_to_send))]

        for graph_name, graph_payload in graph_items:
            image_base64 = _normalise_base64_image(graph_payload)
            if not image_base64:
                continue

            country = graph_payload.get("country") if isinstance(graph_payload, dict) else None
            target_name = graph_payload.get("target_name") if isinstance(graph_payload, dict) else None
            label = f"{graph_name}"
            if country or target_name:
                label = f"{country or ''} - {target_name or ''} ({graph_name})".strip()

            user_content.append(
                {"type": "input_text", "text": f"Forecast graph image: {label}"}
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
    """Safely extract output text from an OpenAI Responses API response."""
    status = getattr(response, "status", None)
    if status == "incomplete":
        details = getattr(response, "incomplete_details", None)
        raise RuntimeError(
            "OpenAI response was incomplete before valid JSON was produced. "
            f"Incomplete details: {details}"
        )

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    # Fallback extraction for SDK/version differences.
    parts: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if isinstance(text, str):
                parts.append(text)

    text = "".join(parts).strip()
    if not text:
        raise RuntimeError("OpenAI response did not contain output text.")
    return text


def _parse_llm_json(output_text: str) -> Dict[str, Any]:
    """Convert model output text into a Python dictionary."""
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"LLM returned non-JSON or truncated output: {exc}. Raw output starts: {output_text[:1000]}"
        ) from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("LLM output JSON was not a JSON object.")

    return parsed


def llm_call(
    forecast_years: int = 3,
    schema: str = "public",
    country_codes: Optional[List[str]] = None,
    focus_country: str = "ARE",
    include_graphs_in_llm: bool = True,
    max_graphs_to_send: Optional[int] = 27,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> Dict[str, Any]:
    """
    Run ARIMA forecasts, send forecast outputs to the LLM, and return JSON narrative + graphs.

    Args:
        forecast_years:
            Number of forward years to forecast. Default is 3.
        schema:
            PostgreSQL schema containing the WDI tables. Default is public.
        country_codes:
            Optional list of ISO3 country codes. Defaults to the 9-country list in
            country_arima_forecasting.
        focus_country:
            ISO3 code or country name to emphasize in the narrative. Default ARE/UAE.
        include_graphs_in_llm:
            If True, sends the base64 graph images to the LLM in addition to the JSON datasets.
            Graphs are always returned to the caller either way.
        max_graphs_to_send:
            Optional cap on graph images sent to the LLM. Default 27.
            Set None to send all returned graphs.
        max_output_tokens:
            Output token budget for the structured JSON narrative.

    Returns:
        {
            "llm_output": {...},
            "graphs": {
                "ARE_gdp_growth": {"mime_type": "image/png", "encoding": "base64", "image_base64": "..."},
                ...
            },
            "metadata": {...}
        }
    """

    # 1. Run the forecasting program and capture JSON datasets + graph output.
    forecast_result = country_arima_forecasting.run_country_arima_forecasts(
        forecast_years=forecast_years,
        schema=schema,
        country_codes=country_codes,
    )

    forecast_json = _strip_graph_images_for_text_payload(forecast_result)
    graph_images = forecast_result.get("graphs", {})

    # 2. Call the OpenAI LLM configured in config.py.
    client, model = _get_openai_client_and_model()
    response_input = _build_llm_input(
        forecast_json=forecast_json,
        graph_images=graph_images,
        focus_country=focus_country,
        include_graphs_in_llm=include_graphs_in_llm,
        max_graphs_to_send=max_graphs_to_send,
    )

    response = client.responses.create(
        model=model,
        input=response_input,
        text={
            "format": {
                "type": "json_schema",
                "name": "country_arima_forecast_interpretation",
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
            "forecast_years": forecast_years,
            "source_schema": schema,
            "country_codes": country_codes or DEFAULT_COUNTRY_CODES,
            "graphs_sent_to_llm": include_graphs_in_llm,
            "max_graphs_sent_to_llm": max_graphs_to_send,
            "graphs_returned": list(graph_images.keys()),
            "graph_count_returned": len(graph_images),
            "max_output_tokens": max_output_tokens,
            "forecast_metadata": forecast_result.get("metadata", {}),
        },
    }
