
"""
Country-level ARIMA forecasting for WDI macro indicators.

Purpose:
- Reads WDI tables directly from PostgreSQL using the deployed project config.
- Forecasts the next 3 years by country for:
    1) GDP growth
    2) Inflation
    3) FX depreciation percentage, derived from official exchange-rate levels
- Returns FastAPI-safe JSON:
    - 27 base64-encoded PNG graphs: 9 countries x 3 forecast targets
    - 27 JSON-ready output datasets, one dataset per graph
    - model summary / benchmark metadata

Important:
- This module intentionally does NOT include FastAPI route code.
- The API layer should import and call run_country_arima_forecasts().
- This module does NOT read CSV files. It queries the database tables directly.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from global_rag.scripts import config


COUNTRY_CODES: List[str] = ["ARE", "AUS", "EGY", "IND", "JOR", "KHM", "SAU", "THA", "ZAF"]

DEFAULT_SCHEMA = os.getenv("MACRO_SCHEMA", "public")
DEFAULT_FORECAST_YEARS = 3

SOURCE_TABLES = {
    "gdp_growth": "wdi_gdp_growth",
    "inflation": "wdi_inflation",
    "fx_level": "wdi_official_exchange_rate",
}

TARGETS = {
    "gdp_growth": {
        "display_name": "GDP growth",
        "unit": "%",
        "y_axis_label": "GDP growth (%)",
        "source_table_key": "gdp_growth",
        "source_value_definition": "GDP growth annual percentage",
        "modelling_definition": "ARIMA on annual GDP growth rate",
    },
    "inflation": {
        "display_name": "Inflation",
        "unit": "%",
        "y_axis_label": "Inflation (%)",
        "source_table_key": "inflation",
        "source_value_definition": "Inflation, consumer prices annual percentage",
        "modelling_definition": "ARIMA on annual inflation rate",
    },
    "fx_depreciation": {
        "display_name": "FX depreciation",
        "unit": "%",
        "y_axis_label": "FX depreciation vs USD (%)",
        "source_table_key": "fx_level",
        "source_value_definition": "Official exchange rate, LCU per USD, period average",
        "modelling_definition": "ARIMA on annual percentage change in LCU per USD",
    },
}

# Simple specifications only. Annual data has very few observations; avoid SARIMA and complex orders.
ARIMA_CANDIDATE_ORDERS: List[Tuple[int, int, int]] = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 0),
    (1, 1, 0),
    (0, 1, 1),
]

MIN_OBSERVATIONS_FOR_ARIMA = 6


def _get_db_url() -> str:
    """Read the database URL from global_rag.scripts.config."""
    cfg = config.config_base()
    db_url = cfg["db_url"]

    # Railway often provides postgres:// or postgresql://.
    # SQLAlchemy is more explicit with postgresql+psycopg2://.
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

    return db_url


def _quote_identifier(identifier: str) -> str:
    """Safely quote a PostgreSQL schema/table identifier."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier}")
    return f'"{identifier}"'


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-safe Python values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _records_json_safe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a dataframe to FastAPI-safe records."""
    safe = df.replace([np.inf, -np.inf], np.nan)
    records = safe.to_dict(orient="records")
    return [{key: _json_safe(value) for key, value in row.items()} for row in records]


def load_wdi_table(
    table_name: str,
    schema: str = DEFAULT_SCHEMA,
    country_codes: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load one WDI table from PostgreSQL."""
    codes = [c.upper().strip() for c in (country_codes or COUNTRY_CODES)]
    if not codes:
        raise ValueError("country_codes cannot be empty.")

    params = {f"c{i}": code for i, code in enumerate(codes)}
    placeholders = ", ".join([f":c{i}" for i in range(len(codes))])

    qualified_table = f"{_quote_identifier(schema)}.{_quote_identifier(table_name)}"

    sql = text(
        f"""
        SELECT
            countryiso3code,
            country,
            country_id,
            indicator_id,
            indicator_name,
            year,
            value
        FROM {qualified_table}
        WHERE countryiso3code IN ({placeholders})
        ORDER BY countryiso3code, year
        """
    )

    engine = create_engine(_get_db_url())
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn, params=params)

    if df.empty:
        raise ValueError(f"No rows returned from {schema}.{table_name} for countries: {codes}")

    df["countryiso3code"] = df["countryiso3code"].astype(str).str.upper()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def load_all_source_tables(
    schema: str = DEFAULT_SCHEMA,
    country_codes: Optional[Iterable[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Load all three raw WDI source tables."""
    return {
        "gdp_growth": load_wdi_table(SOURCE_TABLES["gdp_growth"], schema=schema, country_codes=country_codes),
        "inflation": load_wdi_table(SOURCE_TABLES["inflation"], schema=schema, country_codes=country_codes),
        "fx_level": load_wdi_table(SOURCE_TABLES["fx_level"], schema=schema, country_codes=country_codes),
    }


def _base_series_from_table(df: pd.DataFrame, country_code: str) -> pd.DataFrame:
    """Extract a clean annual time series from a raw WDI table."""
    out = (
        df.loc[df["countryiso3code"].eq(country_code)]
        .copy()
        .sort_values("year")
    )

    out = out[["countryiso3code", "country", "year", "value"]]
    out = out.dropna(subset=["year", "value"])
    out["year"] = out["year"].astype(int)
    out["value"] = out["value"].astype(float)
    out = out.drop_duplicates(subset=["year"], keep="last")
    return out.reset_index(drop=True)


def prepare_target_series(
    source_tables: Dict[str, pd.DataFrame],
    country_code: str,
    target_key: str,
) -> pd.DataFrame:
    """
    Prepare the model-ready time series for a country and target.

    For GDP and inflation, this is simply the WDI annual value.
    For FX, this converts official exchange-rate levels into annual depreciation percentage:
        FX depreciation % = percentage change in LCU per USD.
    Positive value = local currency depreciation against USD.
    Negative value = local currency appreciation against USD.
    """
    country_code = country_code.upper().strip()

    if target_key not in TARGETS:
        raise ValueError(f"Unknown target_key: {target_key}")

    source_key = TARGETS[target_key]["source_table_key"]
    raw = _base_series_from_table(source_tables[source_key], country_code=country_code)

    if raw.empty:
        raise ValueError(f"No source data found for {country_code} / {target_key}")

    if target_key == "fx_depreciation":
        raw = raw.sort_values("year").copy()
        raw["fx_level_lcu_per_usd"] = raw["value"]
        raw["value"] = raw["fx_level_lcu_per_usd"].pct_change() * 100.0
        raw = raw.dropna(subset=["value"]).reset_index(drop=True)
        raw["metric_note"] = "Annual % change in official exchange rate, LCU per USD. Positive means local-currency depreciation vs USD."
    else:
        raw["metric_note"] = TARGETS[target_key]["source_value_definition"]

    raw["target_key"] = target_key
    raw["target_name"] = TARGETS[target_key]["display_name"]
    raw["unit"] = TARGETS[target_key]["unit"]
    return raw[["countryiso3code", "country", "target_key", "target_name", "year", "value", "unit", "metric_note"]]


def _fit_arima_candidate(y: np.ndarray, order: Tuple[int, int, int]) -> Any:
    """Fit a single ARIMA candidate."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)

        model = ARIMA(
            y,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit()

    return fitted


def _select_arima_model(y: np.ndarray) -> Tuple[Any, Tuple[int, int, int], float]:
    """Select the best simple ARIMA candidate using AIC."""
    best_model = None
    best_order: Optional[Tuple[int, int, int]] = None
    best_aic = np.inf

    for order in ARIMA_CANDIDATE_ORDERS:
        # Avoid fitting models that are too parameter-heavy for the sample.
        p, d, q = order
        if len(y) <= (p + q + d + 3):
            continue

        try:
            fitted = _fit_arima_candidate(y, order)
            aic = float(fitted.aic)
            if np.isfinite(aic) and aic < best_aic:
                best_model = fitted
                best_order = order
                best_aic = aic
        except Exception:
            continue

    if best_model is None or best_order is None:
        # Last-resort simple random-walk specification.
        fitted = _fit_arima_candidate(y, (0, 1, 0))
        return fitted, (0, 1, 0), float(fitted.aic)

    return best_model, best_order, best_aic


def _forecast_constant_series(
    historical_df: pd.DataFrame,
    forecast_years: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return a deterministic flat forecast for a constant or near-constant series."""
    last_year = int(historical_df["year"].max())
    last_value = float(historical_df.sort_values("year")["value"].iloc[-1])
    forecast_rows = []

    for i in range(1, forecast_years + 1):
        year = last_year + i
        forecast_rows.append(
            {
                "year": year,
                "observed_value": None,
                "arima_forecast": last_value,
                "arima_lower_80": last_value,
                "arima_upper_80": last_value,
                "arima_lower_95": last_value,
                "arima_upper_95": last_value,
                "naive_forecast": last_value,
                "row_type": "forecast",
            }
        )

    forecast_df = pd.DataFrame(forecast_rows)
    model_info = {
        "model_status": "constant_series_flat_forecast",
        "selected_arima_order": "deterministic_flat",
        "selected_aic": None,
        "arima_holdout_rmse": 0.0,
        "naive_holdout_rmse": 0.0,
        "benchmark_comment": "Series was constant or near-constant; ARIMA and naive forecasts are equivalent.",
    }
    return forecast_df, model_info


def _benchmark_against_naive(y: np.ndarray) -> Dict[str, Any]:
    """
    Compare ARIMA against a naive last-observation benchmark using a small holdout.

    With only annual observations, this is deliberately simple:
    - Hold out the last min(3, roughly 20%) observations.
    - Fit ARIMA on the earlier period.
    - Compare RMSE against a flat naive forecast.
    """
    if len(y) < 9 or np.nanstd(y) < 1e-10:
        return {
            "arima_holdout_rmse": None,
            "naive_holdout_rmse": None,
            "benchmark_comment": "Holdout benchmark skipped due to limited or near-constant observations.",
        }

    test_size = min(3, max(1, len(y) // 5))
    train = y[:-test_size]
    test = y[-test_size:]

    try:
        fitted, order, _ = _select_arima_model(train)
        arima_pred = np.asarray(fitted.get_forecast(steps=test_size).predicted_mean, dtype=float)
        naive_pred = np.repeat(float(train[-1]), test_size)

        arima_rmse = float(np.sqrt(np.mean((test - arima_pred) ** 2)))
        naive_rmse = float(np.sqrt(np.mean((test - naive_pred) ** 2)))

        if arima_rmse < naive_rmse:
            comment = f"ARIMA holdout RMSE is lower than naive benchmark. Holdout order used: {order}."
        elif arima_rmse > naive_rmse:
            comment = f"Naive benchmark RMSE is lower than ARIMA holdout RMSE. Holdout order used: {order}."
        else:
            comment = f"ARIMA and naive benchmark RMSE are equal on holdout. Holdout order used: {order}."

        return {
            "arima_holdout_rmse": arima_rmse,
            "naive_holdout_rmse": naive_rmse,
            "benchmark_comment": comment,
        }
    except Exception as exc:
        return {
            "arima_holdout_rmse": None,
            "naive_holdout_rmse": None,
            "benchmark_comment": f"Holdout benchmark could not be completed: {type(exc).__name__}: {exc}",
        }


def forecast_arima_series(
    series_df: pd.DataFrame,
    forecast_years: int = DEFAULT_FORECAST_YEARS,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fit a simple ARIMA model and generate forecasts with 80% and 95% bands.

    Returns:
        forecast_df: rows for forecast horizon only.
        model_info: selected order, AIC, benchmark information.
    """
    if forecast_years < 1:
        raise ValueError("forecast_years must be >= 1")

    historical = series_df.dropna(subset=["year", "value"]).sort_values("year").copy()
    if len(historical) < MIN_OBSERVATIONS_FOR_ARIMA:
        raise ValueError(
            f"Need at least {MIN_OBSERVATIONS_FOR_ARIMA} observations for ARIMA forecasting; got {len(historical)}."
        )

    y = historical["value"].astype(float).to_numpy()
    last_year = int(historical["year"].max())
    last_value = float(y[-1])

    if np.nanstd(y) < 1e-10:
        return _forecast_constant_series(historical, forecast_years)

    fitted, selected_order, selected_aic = _select_arima_model(y)
    forecast_result = fitted.get_forecast(steps=forecast_years)

    predicted_mean = np.asarray(forecast_result.predicted_mean, dtype=float)

    ci80 = forecast_result.conf_int(alpha=0.20)
    ci95 = forecast_result.conf_int(alpha=0.05)

    # conf_int may be numpy ndarray or DataFrame depending on statsmodels internals.
    ci80_arr = np.asarray(ci80, dtype=float)
    ci95_arr = np.asarray(ci95, dtype=float)

    benchmark = _benchmark_against_naive(y)

    rows = []
    for idx in range(forecast_years):
        forecast_year = last_year + idx + 1
        rows.append(
            {
                "year": forecast_year,
                "observed_value": None,
                "arima_forecast": float(predicted_mean[idx]),
                "arima_lower_80": float(ci80_arr[idx, 0]),
                "arima_upper_80": float(ci80_arr[idx, 1]),
                "arima_lower_95": float(ci95_arr[idx, 0]),
                "arima_upper_95": float(ci95_arr[idx, 1]),
                "naive_forecast": last_value,
                "row_type": "forecast",
            }
        )

    model_info = {
        "model_status": "arima_fitted",
        "selected_arima_order": str(tuple(int(x) for x in selected_order)),
        "selected_aic": float(selected_aic),
        **benchmark,
    }

    return pd.DataFrame(rows), model_info


def build_output_dataset(
    series_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    model_info: Dict[str, Any],
) -> pd.DataFrame:
    """Combine historical observations and forecast rows into one JSON-ready output dataset."""
    country_code = str(series_df["countryiso3code"].iloc[0])
    country_name = str(series_df["country"].iloc[0])
    target_key = str(series_df["target_key"].iloc[0])
    target_name = str(series_df["target_name"].iloc[0])
    unit = str(series_df["unit"].iloc[0])
    metric_note = str(series_df["metric_note"].iloc[0])

    hist = series_df[["year", "value"]].copy()
    hist = hist.rename(columns={"value": "observed_value"})
    hist["arima_forecast"] = np.nan
    hist["arima_lower_80"] = np.nan
    hist["arima_upper_80"] = np.nan
    hist["arima_lower_95"] = np.nan
    hist["arima_upper_95"] = np.nan
    hist["naive_forecast"] = np.nan
    hist["row_type"] = "historical"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        combined = pd.concat([hist, forecast_df], ignore_index=True, sort=False)
    combined["countryiso3code"] = country_code
    combined["country"] = country_name
    combined["target_key"] = target_key
    combined["target_name"] = target_name
    combined["unit"] = unit
    combined["metric_note"] = metric_note

    for key, value in model_info.items():
        combined[key] = value

    ordered = [
        "countryiso3code",
        "country",
        "target_key",
        "target_name",
        "year",
        "row_type",
        "observed_value",
        "arima_forecast",
        "arima_lower_80",
        "arima_upper_80",
        "arima_lower_95",
        "arima_upper_95",
        "naive_forecast",
        "unit",
        "metric_note",
        "model_status",
        "selected_arima_order",
        "selected_aic",
        "arima_holdout_rmse",
        "naive_holdout_rmse",
        "benchmark_comment",
    ]

    return combined[ordered].sort_values(["year", "row_type"]).reset_index(drop=True)


def _figure_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure into raw base64 PNG text."""
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def make_forecast_graph(
    output_df: pd.DataFrame,
    title: str,
    y_axis_label: str,
) -> str:
    """Create a base64 forecast graph with historical series, ARIMA forecast, naive benchmark, and bands."""
    hist = output_df.loc[output_df["row_type"].eq("historical")].copy()
    fcst = output_df.loc[output_df["row_type"].eq("forecast")].copy()

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    ax.plot(
        hist["year"].astype(int),
        hist["observed_value"].astype(float),
        marker="o",
        label="Historical",
    )

    ax.plot(
        fcst["year"].astype(int),
        fcst["arima_forecast"].astype(float),
        marker="o",
        label="ARIMA forecast",
    )

    ax.plot(
        fcst["year"].astype(int),
        fcst["naive_forecast"].astype(float),
        linestyle="--",
        marker="x",
        label="Naive benchmark",
    )

    ax.fill_between(
        fcst["year"].astype(int).to_numpy(),
        fcst["arima_lower_95"].astype(float).to_numpy(),
        fcst["arima_upper_95"].astype(float).to_numpy(),
        alpha=0.12,
        label="95% forecast band",
    )

    ax.fill_between(
        fcst["year"].astype(int).to_numpy(),
        fcst["arima_lower_80"].astype(float).to_numpy(),
        fcst["arima_upper_80"].astype(float).to_numpy(),
        alpha=0.20,
        label="80% forecast band",
    )

    if not hist.empty:
        ax.axvline(int(hist["year"].max()), linestyle=":", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(y_axis_label)
    ax.grid(True, alpha=0.30)
    ax.legend(loc="best", fontsize=9)

    return _figure_to_base64(fig)


def run_country_arima_forecasts(
    forecast_years: int = DEFAULT_FORECAST_YEARS,
    schema: str = DEFAULT_SCHEMA,
    country_codes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Main API-ready callable.

    Returns:
        {
            "metadata": {...},
            "graphs": {
                "ARE_gdp_growth": {"mime_type": "image/png", "encoding": "base64", "image_base64": "...", ...},
                ...
            },
            "datasets": {
                "ARE_gdp_growth": [...],
                ...
            },
            "model_summary": [...]
        }

    Expected count:
        9 countries x 3 targets = 27 graphs and 27 datasets.
    """
    codes = [c.upper().strip() for c in (country_codes or COUNTRY_CODES)]

    source_tables = load_all_source_tables(schema=schema, country_codes=codes)

    graphs: Dict[str, Dict[str, Any]] = {}
    datasets: Dict[str, List[Dict[str, Any]]] = {}
    model_summary: List[Dict[str, Any]] = []

    for country_code in codes:
        for target_key in ["gdp_growth", "inflation", "fx_depreciation"]:
            series_df = prepare_target_series(
                source_tables=source_tables,
                country_code=country_code,
                target_key=target_key,
            )

            forecast_df, model_info = forecast_arima_series(
                series_df=series_df,
                forecast_years=forecast_years,
            )

            output_df = build_output_dataset(
                series_df=series_df,
                forecast_df=forecast_df,
                model_info=model_info,
            )

            country_name = str(output_df["country"].iloc[0])
            target_name = TARGETS[target_key]["display_name"]
            output_key = f"{country_code}_{target_key}"

            graph_title = f"{country_name} - {target_name}: ARIMA {forecast_years}-year forecast"
            image_base64 = make_forecast_graph(
                output_df=output_df,
                title=graph_title,
                y_axis_label=TARGETS[target_key]["y_axis_label"],
            )

            graphs[output_key] = {
                "countryiso3code": country_code,
                "country": country_name,
                "target_key": target_key,
                "target_name": target_name,
                "mime_type": "image/png",
                "encoding": "base64",
                "image_base64": image_base64,
            }

            datasets[output_key] = _records_json_safe(output_df)

            model_summary.append(
                {
                    "output_key": output_key,
                    "countryiso3code": country_code,
                    "country": country_name,
                    "target_key": target_key,
                    "target_name": target_name,
                    "historical_start_year": int(series_df["year"].min()),
                    "historical_end_year": int(series_df["year"].max()),
                    "historical_observations": int(len(series_df)),
                    "forecast_start_year": int(series_df["year"].max()) + 1,
                    "forecast_end_year": int(series_df["year"].max()) + forecast_years,
                    **{k: _json_safe(v) for k, v in model_info.items()},
                }
            )

    return {
        "metadata": {
            "use_case": "ARIMA forecasting for GDP growth, inflation, and FX depreciation",
            "source_tables": {
                "gdp_growth": f"{schema}.{SOURCE_TABLES['gdp_growth']}",
                "inflation": f"{schema}.{SOURCE_TABLES['inflation']}",
                "official_exchange_rate": f"{schema}.{SOURCE_TABLES['fx_level']}",
            },
            "country_codes": codes,
            "forecast_years": int(forecast_years),
            "expected_graph_count": len(codes) * 3,
            "actual_graph_count": len(graphs),
            "expected_dataset_count": len(codes) * 3,
            "actual_dataset_count": len(datasets),
            "targets": TARGETS,
            "arima_candidate_orders": [str(order) for order in ARIMA_CANDIDATE_ORDERS],
            "fx_method": "FX depreciation % = annual percentage change in official exchange rate, LCU per USD. Positive value means local-currency depreciation versus USD.",
            "forecast_bands": ["80%", "95%"],
            "naive_benchmark": "Flat forecast equal to the last observed value.",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "graphs": graphs,
        "datasets": datasets,
        "model_summary": model_summary,
    }


if __name__ == "__main__":
    # Optional local / deployment smoke run. This will query the configured PostgreSQL database.
    result = run_country_arima_forecasts()
    with open("country_arima_forecasts_output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("Done. Output saved to country_arima_forecasts_output.json")
