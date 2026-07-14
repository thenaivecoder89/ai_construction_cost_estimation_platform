# File_name: country_macro_clustering.py
# Use_case: Country clustering / macro-risk segmentation
# Run_mechanics: Run this program before running the llm call

# Purpose:
# - Reads the `country_features_raw` table from PostgreSQL using existing config package.
# - Builds macro-risk features.
# - Runs K-Means, Hierarchical Clustering, and Gaussian Mixture Model clustering.
# - Returns base64-encoded PNG graphs and JSON-ready output datasets.

from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text


# Preferred project-package import requested by you.
# Fallback keeps local testing simple if the file is run beside config.py.
try:
    from global_rag.scripts import config
except ImportError:  # pragma: no cover
    import config  # type: ignore


RAW_TABLE = "country_features_raw"
DEFAULT_SCHEMA = os.getenv("MACRO_SCHEMA", "public")
DEFAULT_N_CLUSTERS = 4


# Mapping from raw DB columns to model feature names.
# These are the eight features requested in the use case.
FEATURE_MAP = {
    "avg_gdp_growth": "avg_gdp_growth_2010_2024",
    "gdp_growth_volatility": "gdp_growth_sd_2010_2024",
    "avg_inflation": "avg_inflation_2010_2024",
    "inflation_volatility": "inflation_sd_2010_2024",
    # Long-term currency weakness from first available FX year to 2024.
    "fx_depreciation": "cumulative_fx_depreciation_pct_first_to_2024",
    "fx_volatility": "fx_depreciation_sd_pct",
    # Positive number = deeper 2020 COVID shock.
    "covid_shock_depth": "covid_2020_gdp_growth",
    # Larger number = stronger rebound from 2020 to post-COVID period.
    "post_covid_recovery": "recovery_lift_post_covid_vs_2020_pp",
}

MODEL_FEATURES = list(FEATURE_MAP.keys())
ID_COLUMNS = ["countryiso3code", "country", "country_id"]


def _get_db_url() -> str:
    """Read the database URL from config.config_base()."""
    cfg = config.config_base()
    db_url = cfg["db_url"]

    # Railway sometimes provides postgres://; SQLAlchemy prefers postgresql://.
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

    return db_url


def _quote_identifier(identifier: str) -> str:
    """Safely quote a PostgreSQL identifier such as schema or table name."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier}")
    return f'"{identifier}"'


def load_country_features(
    schema: str = DEFAULT_SCHEMA,
    table_name: str = RAW_TABLE,
) -> pd.DataFrame:
    """Load country_features_raw from PostgreSQL."""
    engine = create_engine(_get_db_url())
    qualified_table = f"{_quote_identifier(schema)}.{_quote_identifier(table_name)}"
    sql = text(f"SELECT * FROM {qualified_table}")

    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn)

    return df


def build_model_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Create the modelling dataset from the raw country feature table."""
    missing_cols = [col for col in FEATURE_MAP.values() if col not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in country_features_raw: {missing_cols}")

    output = pd.DataFrame()

    for col in ID_COLUMNS:
        if col in raw_df.columns:
            output[col] = raw_df[col]

    for feature_name, raw_col in FEATURE_MAP.items():
        output[feature_name] = pd.to_numeric(raw_df[raw_col], errors="coerce")

    # Convert 2020 GDP growth into shock depth.
    # Example: -8.7% growth becomes 8.7 shock depth; positive growth becomes 0.
    output["covid_shock_depth"] = (-output["covid_shock_depth"]).clip(lower=0)

    # Optional useful context for labelling clusters; not used as a model feature.
    if "fx_stable_peg_flag" in raw_df.columns:
        output["fx_stable_peg_flag"] = pd.to_numeric(raw_df["fx_stable_peg_flag"], errors="coerce")

    if "suggested_include_in_clustering" in raw_df.columns:
        include_mask = raw_df["suggested_include_in_clustering"].astype(str).str.upper().isin(["Y", "YES", "TRUE", "1"])
        output = output.loc[include_mask].copy()

    if output.empty:
        raise ValueError("No rows available for clustering after applying inclusion filter.")

    return output.reset_index(drop=True)


def prepare_matrix(feature_df: pd.DataFrame) -> Dict[str, Any]:
    """Impute missing values, standardise features, and compute PCA coordinates for plotting."""
    X_raw = feature_df[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X_raw)
    X_scaled = scaler.fit_transform(X_imputed)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)

    plot_df = feature_df.copy()
    plot_df["pc1"] = coords[:, 0]
    plot_df["pc2"] = coords[:, 1]

    return {
        "X_scaled": X_scaled,
        "plot_df": plot_df,
        "imputer": imputer,
        "scaler": scaler,
        "pca": pca,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }


def _choose_n_clusters(n_requested: int, n_rows: int) -> int:
    """Keep cluster count valid for small country lists."""
    if n_rows < 2:
        raise ValueError("At least two countries are required for clustering.")
    return max(2, min(int(n_requested), n_rows))


def _segment_labels(result_df: pd.DataFrame, cluster_col: str) -> Dict[int, str]:
    """Create client-friendly labels for each cluster based on cluster-level feature means."""
    labels: Dict[int, str] = {}
    med = result_df[MODEL_FEATURES].median(numeric_only=True)
    q75 = result_df[MODEL_FEATURES].quantile(0.75, numeric_only=True)
    q25 = result_df[MODEL_FEATURES].quantile(0.25, numeric_only=True)

    for cluster_id, group in result_df.groupby(cluster_col):
        avg = group[MODEL_FEATURES].mean(numeric_only=True)
        peg_share = group.get("fx_stable_peg_flag", pd.Series([0])).mean()

        high_currency_stress = (
            avg["avg_inflation"] >= q75["avg_inflation"]
            or avg["fx_depreciation"] >= q75["fx_depreciation"]
            or avg["fx_volatility"] >= q75["fx_volatility"]
        )
        stable_peg_profile = (
            peg_share >= 0.50
            or (
                abs(avg["fx_depreciation"]) <= abs(med["fx_depreciation"])
                and avg["fx_volatility"] <= med["fx_volatility"]
                and avg["avg_inflation"] <= med["avg_inflation"]
            )
        )
        high_growth_profile = (
            avg["avg_gdp_growth"] >= med["avg_gdp_growth"]
            and avg["post_covid_recovery"] >= med["post_covid_recovery"]
            and not high_currency_stress
        )
        low_growth_or_volatile = (
            avg["avg_gdp_growth"] <= q25["avg_gdp_growth"]
            or avg["gdp_growth_volatility"] >= q75["gdp_growth_volatility"]
            or avg["covid_shock_depth"] >= q75["covid_shock_depth"]
        )

        if high_currency_stress:
            label = "Inflation / currency stress economies"
        elif stable_peg_profile:
            label = "Stable peg / low-FX-risk economies"
        elif high_growth_profile:
            label = "High-growth emerging markets"
        elif low_growth_or_volatile:
            label = "Low-growth / volatile economies"
        else:
            label = "Balanced / moderate-risk economies"

        labels[int(cluster_id)] = label

    return labels


def _scatter_plot_base64(
    df: pd.DataFrame,
    cluster_col: str,
    title: str,
) -> str:
    """Create a base64 PNG scatter plot from PCA coordinates."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for cluster_id, group in df.groupby(cluster_col):
        ax.scatter(group["pc1"], group["pc2"], label=f"Cluster {cluster_id}", s=80)
        for _, row in group.iterrows():
            label = row.get("countryiso3code") or row.get("country")
            ax.annotate(str(label), (row["pc1"], row["pc2"]), xytext=(5, 5), textcoords="offset points", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    return _figure_to_base64(fig)


def _dendrogram_base64(X_scaled: np.ndarray, labels: List[str]) -> str:
    """Create a base64 PNG hierarchical clustering dendrogram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    Z = linkage(X_scaled, method="ward")
    dendrogram(Z, labels=labels, leaf_rotation=45, leaf_font_size=10, ax=ax)
    ax.set_title("Hierarchical Clustering - Country Macro-Risk Dendrogram")
    ax.set_ylabel("Ward Distance")
    ax.grid(True, axis="y", alpha=0.3)

    return _figure_to_base64(fig)


def _figure_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _json_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a dataframe to JSON-safe records for FastAPI response payloads."""
    safe_df = df.replace([np.inf, -np.inf], np.nan)
    records = safe_df.to_dict(orient="records")

    def convert(value: Any) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.ndarray,)):
            return value.tolist()
        return value

    return [{key: convert(value) for key, value in row.items()} for row in records]


def _attach_cluster_output(
    plot_df: pd.DataFrame,
    labels: np.ndarray,
    model_prefix: str,
    probabilities: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Attach cluster IDs, segment labels, and optional GMM probabilities."""
    out = plot_df.copy()
    cluster_col = f"{model_prefix}_cluster"
    segment_col = f"{model_prefix}_segment"

    # Use one-based cluster IDs for client readability.
    out[cluster_col] = labels.astype(int) + 1

    segment_map = _segment_labels(out, cluster_col)
    out[segment_col] = out[cluster_col].map(segment_map)

    if probabilities is not None:
        for i in range(probabilities.shape[1]):
            out[f"{model_prefix}_probability_cluster_{i + 1}"] = probabilities[:, i]

    ordered_cols = [col for col in ID_COLUMNS if col in out.columns]
    ordered_cols += MODEL_FEATURES
    if "fx_stable_peg_flag" in out.columns:
        ordered_cols.append("fx_stable_peg_flag")
    ordered_cols += ["pc1", "pc2", cluster_col, segment_col]
    if probabilities is not None:
        ordered_cols += [f"{model_prefix}_probability_cluster_{i + 1}" for i in range(probabilities.shape[1])]

    return out[ordered_cols]


def run_country_macro_clustering(
    n_clusters: int = DEFAULT_N_CLUSTERS,
    schema: str = DEFAULT_SCHEMA,
    table_name: str = RAW_TABLE,
    raw_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Main FastAPI-ready function.

    Returns:
        {
          "metadata": {...},
          "graphs": {
              "kmeans": {"mime_type": "image/png", "image_base64": "..."},
              "hierarchical": {"mime_type": "image/png", "image_base64": "..."},
              "gmm": {"mime_type": "image/png", "image_base64": "..."}
          },
          "datasets": {
              "kmeans": [...],
              "hierarchical": [...],
              "gmm": [...]
          }
        }
    """
    if raw_df is None:
        raw_df = load_country_features(schema=schema, table_name=table_name)

    feature_df = build_model_features(raw_df)
    n_clusters = _choose_n_clusters(n_clusters, len(feature_df))

    prepared = prepare_matrix(feature_df)
    X_scaled = prepared["X_scaled"]
    plot_df = prepared["plot_df"]

    # 1) K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    kmeans_output = _attach_cluster_output(plot_df, kmeans_labels, "kmeans")
    kmeans_graph = _scatter_plot_base64(kmeans_output, "kmeans_cluster", "K-Means Country Macro-Risk Segments")

    # 2) Hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    hierarchical_labels = hierarchical.fit_predict(X_scaled)
    hierarchical_output = _attach_cluster_output(plot_df, hierarchical_labels, "hierarchical")
    dendro_labels = feature_df["countryiso3code"].astype(str).tolist() if "countryiso3code" in feature_df.columns else feature_df["country"].astype(str).tolist()
    hierarchical_graph = _dendrogram_base64(X_scaled, dendro_labels)

    # 3) Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)
    gmm_probabilities = gmm.predict_proba(X_scaled)
    gmm_output = _attach_cluster_output(plot_df, gmm_labels, "gmm", probabilities=gmm_probabilities)
    gmm_graph = _scatter_plot_base64(gmm_output, "gmm_cluster", "Gaussian Mixture Model Country Macro-Risk Segments")

    return {
        "metadata": {
            "source_table": f"{schema}.{table_name}",
            "country_count": int(len(feature_df)),
            "n_clusters": int(n_clusters),
            "model_features": MODEL_FEATURES,
            "feature_source_columns": FEATURE_MAP,
            "pca_explained_variance_ratio": prepared["pca_explained_variance_ratio"],
        },
        "graphs": {
            "kmeans": {
                "mime_type": "image/png",
                "encoding": "base64",
                "image_base64": kmeans_graph,
            },
            "hierarchical": {
                "mime_type": "image/png",
                "encoding": "base64",
                "image_base64": hierarchical_graph,
            },
            "gmm": {
                "mime_type": "image/png",
                "encoding": "base64",
                "image_base64": gmm_graph,
            },
        },
        "datasets": {
            "kmeans": _json_records(kmeans_output),
            "hierarchical": _json_records(hierarchical_output),
            "gmm": _json_records(gmm_output),
        },
    }


if __name__ == "__main__":
    # CLI smoke run. In production, call run_country_macro_clustering() from FastAPI.
    result = run_country_macro_clustering()
    with open("country_macro_clustering_output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Done. Output saved to country_macro_clustering_output.json")