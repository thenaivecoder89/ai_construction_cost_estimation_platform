-- SQL script to create table for Country clustering based on macro-risk segmentation use case

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

SELECT * FROM country_features_raw;