# File_name: wb_scraper.py

# Purpose
# -------
# Pull selected World Bank WDI indicators from the World Bank Indicators API and
# load them into PostgreSQL tables. The script also consolidates the indicators
# into the country_features_raw table used by the macro-risk clustering workflow.

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import json

import pandas as pd
from sqlalchemy import create_engine, text

from global_rag.scripts import config


WDI_START_YEAR = 2010
WDI_END_YEAR = 2024
WORLD_BANK_API_BASE_URL = "https://api.worldbank.org/v2"

INDICATORS = {
    "wdi_inflation": {
        "indicator_id": "FP.CPI.TOTL.ZG",
        "indicator_name": "Inflation, consumer prices (annual %)",
    },
    "wdi_official_exchange_rate": {
        "indicator_id": "PA.NUS.FCRF",
        "indicator_name": "Official exchange rate (LCU per US$, period average)",
    },
    "wdi_gdp_growth": {
        "indicator_id": "NY.GDP.MKTP.KD.ZG",
        "indicator_name": "GDP growth (annual %)",
    },
}

FEATURE_COLUMNS = [
    "countryiso3code",
    "country",
    "country_id",
    "gdp_obs_count",
    "inflation_obs_count",
    "fx_obs_count",
    "data_completeness_pct",
    "avg_gdp_growth_2010_2024",
    "gdp_growth_sd_2010_2024",
    "min_gdp_growth_2010_2024",
    "covid_2020_gdp_growth",
    "pre_covid_avg_gdp_growth_2017_2019",
    "post_covid_avg_gdp_growth_2021_2024",
    "recovery_lift_post_covid_vs_2020_pp",
    "avg_inflation_2010_2024",
    "inflation_sd_2010_2024",
    "max_inflation_2010_2024",
    "avg_fx_depreciation_pct",
    "fx_depreciation_sd_pct",
    "max_fx_depreciation_pct",
    "cumulative_fx_depreciation_pct_first_to_2024",
    "fx_first_available_year",
    "fx_last_available_year",
    "fx_stable_peg_flag",
    "suggested_include_in_clustering",
    "data_quality_note",
]


def get_reference_csv_path():
    project_root = Path(__file__).resolve().parent.parent
    return (
        project_root
        / "corpus"
        / "03_ML_Input_Datasets"
        / "WDI_Macro_Risk_Clustering_Input.csv"
    )


def get_default_country_codes():
    reference_csv_path = get_reference_csv_path()

    if not reference_csv_path.exists():
        return ["AUS", "KHM", "EGY", "IND", "JOR", "SAU", "ZAF", "THA", "ARE"]

    reference_df = pd.read_csv(reference_csv_path, encoding="utf-8-sig")
    return (
        reference_df["countryiso3code"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )


def clean_country_codes(country_codes):
    if country_codes is None:
        country_codes = get_default_country_codes()

    cleaned_codes = []
    for country_code in country_codes:
        country_code = str(country_code).strip().upper()
        if country_code:
            cleaned_codes.append(country_code)

    if not cleaned_codes:
        raise ValueError("At least one country code is required.")

    return sorted(set(cleaned_codes))


def to_float(value):
    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def fetch_wdi_indicator(indicator_id, indicator_name, country_codes, start_year, end_year):
    country_path = ";".join(country_codes)
    page = 1
    pages = 1
    rows = []
    retrieved_at = datetime.now(timezone.utc).replace(tzinfo=None)

    while page <= pages:
        query_params = urlencode(
            {
                "format": "json",
                "per_page": 20000,
                "date": f"{start_year}:{end_year}",
                "page": page,
            }
        )
        api_url = (
            f"{WORLD_BANK_API_BASE_URL}/country/{country_path}"
            f"/indicator/{indicator_id}?{query_params}"
        )

        with urlopen(api_url, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))

        if not isinstance(payload, list) or len(payload) < 2:
            raise RuntimeError(f"Unexpected World Bank API response for {indicator_id}.")

        metadata = payload[0] or {}
        data = payload[1] or []
        pages = int(metadata.get("pages") or 1)

        for item in data:
            country = item.get("country") or {}

            rows.append(
                {
                    "countryiso3code": item.get("countryiso3code"),
                    "country": country.get("value"),
                    "country_id": country.get("id"),
                    "indicator_id": indicator_id,
                    "indicator_name": indicator_name,
                    "year": int(item.get("date")),
                    "value": to_float(item.get("value")),
                    "unit": item.get("unit"),
                    "obs_status": item.get("obs_status"),
                    "decimal": item.get("decimal"),
                    "retrieved_at": retrieved_at,
                    "source_api_url": api_url,
                }
            )

        page += 1

    output_df = pd.DataFrame(rows)

    if output_df.empty:
        return pd.DataFrame(
            columns=[
                "countryiso3code",
                "country",
                "country_id",
                "indicator_id",
                "indicator_name",
                "year",
                "value",
                "unit",
                "obs_status",
                "decimal",
                "retrieved_at",
                "source_api_url",
            ]
        )

    output_df = output_df.sort_values(["countryiso3code", "year"]).reset_index(drop=True)
    return output_df


def get_series_value(df, countryiso3code, year):
    value_series = df.loc[
        (df["countryiso3code"] == countryiso3code) & (df["year"] == year),
        "value",
    ].dropna()

    if value_series.empty:
        return None

    return float(value_series.iloc[0])


def summarize_series(df, countryiso3code):
    values = df.loc[df["countryiso3code"] == countryiso3code, "value"].dropna()

    if values.empty:
        return {
            "obs_count": 0,
            "avg": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {
        "obs_count": int(values.count()),
        "avg": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def summarize_period_average(df, countryiso3code, start_year, end_year):
    values = df.loc[
        (df["countryiso3code"] == countryiso3code)
        & (df["year"] >= start_year)
        & (df["year"] <= end_year),
        "value",
    ].dropna()

    if values.empty:
        return None

    return float(values.mean())


def calculate_fx_depreciation_features(fx_df, countryiso3code):
    country_fx_df = (
        fx_df.loc[
            (fx_df["countryiso3code"] == countryiso3code) & fx_df["value"].notna(),
            ["year", "value"],
        ]
        .sort_values("year")
        .copy()
    )

    if country_fx_df.empty:
        return {
            "fx_obs_count": 0,
            "avg_fx_depreciation_pct": None,
            "fx_depreciation_sd_pct": None,
            "max_fx_depreciation_pct": None,
            "cumulative_fx_depreciation_pct_first_to_2024": None,
            "fx_first_available_year": None,
            "fx_last_available_year": None,
            "fx_stable_peg_flag": 0,
        }

    country_fx_df["fx_depreciation_pct"] = country_fx_df["value"].pct_change() * 100
    depreciation_values = country_fx_df["fx_depreciation_pct"].dropna()

    first_row = country_fx_df.iloc[0]
    last_row = country_fx_df.iloc[-1]

    cumulative_depreciation = None
    if first_row["value"] not in [None, 0]:
        cumulative_depreciation = ((last_row["value"] / first_row["value"]) - 1) * 100

    stable_peg_flag = 0
    if depreciation_values.count() > 0:
        if depreciation_values.abs().max() <= 1:
            stable_peg_flag = 1

    return {
        "fx_obs_count": int(country_fx_df["value"].count()),
        "avg_fx_depreciation_pct": (
            float(depreciation_values.mean()) if not depreciation_values.empty else None
        ),
        "fx_depreciation_sd_pct": (
            float(depreciation_values.std(ddof=0)) if not depreciation_values.empty else None
        ),
        "max_fx_depreciation_pct": (
            float(depreciation_values.max()) if not depreciation_values.empty else None
        ),
        "cumulative_fx_depreciation_pct_first_to_2024": (
            float(cumulative_depreciation) if cumulative_depreciation is not None else None
        ),
        "fx_first_available_year": int(first_row["year"]),
        "fx_last_available_year": int(last_row["year"]),
        "fx_stable_peg_flag": stable_peg_flag,
    }


def get_country_metadata(country_codes, indicator_dfs):
    metadata_rows = []

    for indicator_df in indicator_dfs:
        if indicator_df.empty:
            continue

        metadata_rows.append(
            indicator_df[["countryiso3code", "country", "country_id"]].drop_duplicates()
        )

    if not metadata_rows:
        return pd.DataFrame(
            {
                "countryiso3code": country_codes,
                "country": country_codes,
                "country_id": None,
            }
        )

    metadata_df = pd.concat(metadata_rows, ignore_index=True)
    metadata_df = metadata_df.dropna(subset=["countryiso3code"])
    metadata_df = metadata_df.drop_duplicates(subset=["countryiso3code"], keep="first")

    return metadata_df


def build_country_features(inflation_df, fx_df, gdp_df, country_codes):
    metadata_df = get_country_metadata(
        country_codes=country_codes,
        indicator_dfs=[inflation_df, fx_df, gdp_df],
    )
    metadata_map = metadata_df.set_index("countryiso3code").to_dict("index")

    expected_obs_count = WDI_END_YEAR - WDI_START_YEAR + 1
    expected_total_obs_count = expected_obs_count * 3
    feature_rows = []

    for countryiso3code in country_codes:
        metadata = metadata_map.get(countryiso3code, {})
        gdp_summary = summarize_series(gdp_df, countryiso3code)
        inflation_summary = summarize_series(inflation_df, countryiso3code)
        fx_features = calculate_fx_depreciation_features(fx_df, countryiso3code)

        pre_covid_avg = summarize_period_average(gdp_df, countryiso3code, 2017, 2019)
        post_covid_avg = summarize_period_average(gdp_df, countryiso3code, 2021, 2024)
        covid_2020_gdp_growth = get_series_value(gdp_df, countryiso3code, 2020)

        recovery_lift = None
        if post_covid_avg is not None and covid_2020_gdp_growth is not None:
            recovery_lift = post_covid_avg - covid_2020_gdp_growth

        observed_total = (
            gdp_summary["obs_count"]
            + inflation_summary["obs_count"]
            + fx_features["fx_obs_count"]
        )
        data_completeness_pct = observed_total / expected_total_obs_count * 100

        quality_notes = []
        if gdp_summary["obs_count"] < expected_obs_count:
            quality_notes.append("GDP growth series incomplete")
        if inflation_summary["obs_count"] < expected_obs_count:
            quality_notes.append("Inflation series incomplete")
        if fx_features["fx_obs_count"] < expected_obs_count:
            quality_notes.append("FX series incomplete")

        feature_rows.append(
            {
                "countryiso3code": countryiso3code,
                "country": metadata.get("country") or countryiso3code,
                "country_id": metadata.get("country_id"),
                "gdp_obs_count": gdp_summary["obs_count"],
                "inflation_obs_count": inflation_summary["obs_count"],
                "fx_obs_count": fx_features["fx_obs_count"],
                "data_completeness_pct": data_completeness_pct,
                "avg_gdp_growth_2010_2024": gdp_summary["avg"],
                "gdp_growth_sd_2010_2024": gdp_summary["std"],
                "min_gdp_growth_2010_2024": gdp_summary["min"],
                "covid_2020_gdp_growth": covid_2020_gdp_growth,
                "pre_covid_avg_gdp_growth_2017_2019": pre_covid_avg,
                "post_covid_avg_gdp_growth_2021_2024": post_covid_avg,
                "recovery_lift_post_covid_vs_2020_pp": recovery_lift,
                "avg_inflation_2010_2024": inflation_summary["avg"],
                "inflation_sd_2010_2024": inflation_summary["std"],
                "max_inflation_2010_2024": inflation_summary["max"],
                "avg_fx_depreciation_pct": fx_features["avg_fx_depreciation_pct"],
                "fx_depreciation_sd_pct": fx_features["fx_depreciation_sd_pct"],
                "max_fx_depreciation_pct": fx_features["max_fx_depreciation_pct"],
                "cumulative_fx_depreciation_pct_first_to_2024": fx_features[
                    "cumulative_fx_depreciation_pct_first_to_2024"
                ],
                "fx_first_available_year": fx_features["fx_first_available_year"],
                "fx_last_available_year": fx_features["fx_last_available_year"],
                "fx_stable_peg_flag": fx_features["fx_stable_peg_flag"],
                "suggested_include_in_clustering": (
                    "Yes" if data_completeness_pct >= 80 else "No"
                ),
                "data_quality_note": "; ".join(quality_notes),
            }
        )

    features_df = pd.DataFrame(feature_rows)
    return features_df[FEATURE_COLUMNS]


def create_country_features_raw_table(engine):
    with engine.begin() as conn:
        conn.execute(
            text("""
                CREATE TABLE IF NOT EXISTS country_features_raw (
                    countryiso3code TEXT PRIMARY KEY,
                    country TEXT,
                    country_id TEXT,
                    gdp_obs_count INTEGER,
                    inflation_obs_count INTEGER,
                    fx_obs_count INTEGER,
                    data_completeness_pct DOUBLE PRECISION,
                    avg_gdp_growth_2010_2024 DOUBLE PRECISION,
                    gdp_growth_sd_2010_2024 DOUBLE PRECISION,
                    min_gdp_growth_2010_2024 DOUBLE PRECISION,
                    covid_2020_gdp_growth DOUBLE PRECISION,
                    pre_covid_avg_gdp_growth_2017_2019 DOUBLE PRECISION,
                    post_covid_avg_gdp_growth_2021_2024 DOUBLE PRECISION,
                    recovery_lift_post_covid_vs_2020_pp DOUBLE PRECISION,
                    avg_inflation_2010_2024 DOUBLE PRECISION,
                    inflation_sd_2010_2024 DOUBLE PRECISION,
                    max_inflation_2010_2024 DOUBLE PRECISION,
                    avg_fx_depreciation_pct DOUBLE PRECISION,
                    fx_depreciation_sd_pct DOUBLE PRECISION,
                    max_fx_depreciation_pct DOUBLE PRECISION,
                    cumulative_fx_depreciation_pct_first_to_2024 DOUBLE PRECISION,
                    fx_first_available_year INTEGER,
                    fx_last_available_year INTEGER,
                    fx_stable_peg_flag INTEGER,
                    suggested_include_in_clustering TEXT,
                    data_quality_note TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        )


def load_dataframe_replace(engine, table_name, df):
    df.to_sql(
        name=table_name,
        con=engine,
        index=False,
        if_exists="replace",
        method="multi",
    )


def load_country_features_raw(engine, features_df):
    create_country_features_raw_table(engine)

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM country_features_raw;"))

    features_df.to_sql(
        name="country_features_raw",
        con=engine,
        index=False,
        if_exists="append",
        method="multi",
    )


def scrape_world_bank_wdi(country_codes=None, start_year=WDI_START_YEAR, end_year=WDI_END_YEAR):
    country_codes = ["AUS", "KHM", "EGY", "IND", "JOR", "SAU", "ZAF", "THA", "ARE"]
    config_base = config.config_base()

    engine = create_engine(
        url=config_base["db_url"],
        pool_pre_ping=True,
    )

    indicator_dfs = {}

    for table_name, indicator_config in INDICATORS.items():
        indicator_df = fetch_wdi_indicator(
            indicator_id=indicator_config["indicator_id"],
            indicator_name=indicator_config["indicator_name"],
            country_codes=country_codes,
            start_year=start_year,
            end_year=end_year,
        )
        indicator_dfs[table_name] = indicator_df
        load_dataframe_replace(engine, table_name, indicator_df)

    features_df = build_country_features(
        inflation_df=indicator_dfs["wdi_inflation"],
        fx_df=indicator_dfs["wdi_official_exchange_rate"],
        gdp_df=indicator_dfs["wdi_gdp_growth"],
        country_codes=country_codes,
    )
    load_country_features_raw(engine, features_df)

    return {
        "message": "World Bank WDI scrape and database load completed.",
        "country_count": len(country_codes),
        "countries": country_codes,
        "start_year": start_year,
        "end_year": end_year,
        "tables_loaded": {
            table_name: int(len(indicator_df))
            for table_name, indicator_df in indicator_dfs.items()
        },
        "country_features_raw_rows": int(len(features_df)),
    }


if __name__ == "__main__":
    print(json.dumps(scrape_world_bank_wdi(), indent=2, ensure_ascii=False))
