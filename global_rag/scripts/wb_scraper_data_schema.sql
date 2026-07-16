-- ============================================================
-- 1. WDI GDP GROWTH
-- Source indicator: NY.GDP.MKTP.KD.ZG
-- ============================================================

CREATE TABLE wdi_gdp_growth (
    wdi_gdp_growth_id_pk BIGSERIAL PRIMARY KEY,

    countryiso3code      CHAR(3) NOT NULL,
    country              VARCHAR(150) NOT NULL,
    country_id           CHAR(2) NOT NULL,

    indicator_id         VARCHAR(50) NOT NULL,
    indicator_name       VARCHAR(255) NOT NULL,

    year                 SMALLINT NOT NULL,
    value                DOUBLE PRECISION NOT NULL,

    unit                 VARCHAR(100),
    obs_status           VARCHAR(50),
    decimal              SMALLINT NOT NULL,

    retrieved_at         TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    source_api_url       TEXT NOT NULL,

    CONSTRAINT uq_wdi_gdp_growth_country_indicator_year
        UNIQUE (countryiso3code, indicator_id, year),

    CONSTRAINT chk_wdi_gdp_growth_year
        CHECK (year BETWEEN 1800 AND 2200),

    CONSTRAINT chk_wdi_gdp_growth_decimal
        CHECK (decimal >= 0)
);


CREATE INDEX idx_wdi_gdp_growth_country_year
    ON wdi_gdp_growth (countryiso3code, year);

CREATE INDEX idx_wdi_gdp_growth_indicator
    ON wdi_gdp_growth (indicator_id);

-- ============================================================
-- 2. WDI INFLATION
-- Source indicator: FP.CPI.TOTL.ZG
-- ============================================================

CREATE TABLE wdi_inflation (
    wdi_inflation_id_pk  BIGSERIAL PRIMARY KEY,

    countryiso3code      CHAR(3) NOT NULL,
    country              VARCHAR(150) NOT NULL,
    country_id           CHAR(2) NOT NULL,

    indicator_id         VARCHAR(50) NOT NULL,
    indicator_name       VARCHAR(255) NOT NULL,

    year                 SMALLINT NOT NULL,
    value                DOUBLE PRECISION NOT NULL,

    unit                 VARCHAR(100),
    obs_status           VARCHAR(50),
    decimal              SMALLINT NOT NULL,

    retrieved_at         TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    source_api_url       TEXT NOT NULL,

    CONSTRAINT uq_wdi_inflation_country_indicator_year
        UNIQUE (countryiso3code, indicator_id, year),

    CONSTRAINT chk_wdi_inflation_year
        CHECK (year BETWEEN 1800 AND 2200),

    CONSTRAINT chk_wdi_inflation_decimal
        CHECK (decimal >= 0)
);


CREATE INDEX idx_wdi_inflation_country_year
    ON wdi_inflation (countryiso3code, year);

CREATE INDEX idx_wdi_inflation_indicator
    ON wdi_inflation (indicator_id);

-- ============================================================
-- 3. WDI OFFICIAL EXCHANGE RATE
-- Source indicator: PA.NUS.FCRF
-- ============================================================

CREATE TABLE wdi_official_exchange_rate (
    wdi_exchange_rate_id_pk BIGSERIAL PRIMARY KEY,

    countryiso3code         CHAR(3) NOT NULL,
    country                 VARCHAR(150) NOT NULL,
    country_id              CHAR(2) NOT NULL,

    indicator_id            VARCHAR(50) NOT NULL,
    indicator_name          VARCHAR(255) NOT NULL,

    year                    SMALLINT NOT NULL,
    value                   DOUBLE PRECISION NOT NULL,

    unit                    VARCHAR(100),
    obs_status              VARCHAR(50),
    decimal                 SMALLINT NOT NULL,

    retrieved_at            TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    source_api_url          TEXT NOT NULL,

    CONSTRAINT uq_wdi_exchange_rate_country_indicator_year
        UNIQUE (countryiso3code, indicator_id, year),

    CONSTRAINT chk_wdi_exchange_rate_year
        CHECK (year BETWEEN 1800 AND 2200),

    CONSTRAINT chk_wdi_exchange_rate_decimal
        CHECK (decimal >= 0)
);


CREATE INDEX idx_wdi_exchange_rate_country_year
    ON wdi_official_exchange_rate (countryiso3code, year);

CREATE INDEX idx_wdi_exchange_rate_indicator
    ON wdi_official_exchange_rate (indicator_id);