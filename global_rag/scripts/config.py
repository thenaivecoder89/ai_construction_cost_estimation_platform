# File_name: config.py
# Purpose: To store all common paths, database settings, model settings,
# and project-level constants used by the RAG pipeline

from pathlib import Path
import base64
import json
import os
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(dotenv_path=None, override=False, *args, **kwargs):
        if dotenv_path is None:
            return False

        dotenv_path = Path(dotenv_path)
        if not dotenv_path.exists():
            return False

        for raw_line in dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if not key:
                continue

            if override or key not in os.environ:
                os.environ[key] = value

        return True


def load_project_dotenv():
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    return load_dotenv(dotenv_path=env_path)

def discover_immediate_subdirectories(base_dir):
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return {}

    return {
        path.name: path
        for path in sorted(base_dir.iterdir())
        if path.is_dir()
    }


def bool_env(name, default_value=False):
    raw_value = os.getenv(name)
    if raw_value is None:
        return bool(default_value)

    return raw_value.strip().upper() in ["1", "Y", "YES", "TRUE", "ON"]


def required_env(name):
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        raise RuntimeError(f"{name} is not set. Please define it in the environment or .env file.")

    return str(value).strip().strip('"').strip("'")


def strip_gs_scheme(value):
    value = str(value or "").strip()
    if value.startswith("gs://"):
        return value[len("gs://"):]

    return value


def split_gs_bucket_and_prefix(gs_uri_or_bucket, prefix=""):
    bucket_and_path = strip_gs_scheme(gs_uri_or_bucket).strip("/")
    path_prefix = str(prefix or "").strip("/")

    if "/" in bucket_and_path:
        bucket_name, embedded_prefix = bucket_and_path.split("/", 1)
        if path_prefix:
            path_prefix = f"{embedded_prefix.strip('/')}/{path_prefix}"
        else:
            path_prefix = embedded_prefix.strip("/")
    else:
        bucket_name = bucket_and_path

    return bucket_name, path_prefix.strip("/")


def get_storage_client():
    """
    Creates an authenticated Google Cloud Storage client using a Base64-encoded
    service-account JSON from the environment.
    """
    encoded_credentials = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_BASE64")

    if not encoded_credentials:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON_BASE64 is not configured.")

    try:
        from google.cloud import storage
        from google.oauth2 import service_account
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "google-cloud-storage is required for Firebase Storage sync. "
            "Install requirements.txt before running the ingestion pipeline."
        ) from e

    try:
        decoded_credentials = base64.b64decode(encoded_credentials).decode("utf-8")
        service_account_info = json.loads(decoded_credentials)
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info
        )
        project_id = (
            os.getenv("GOOGLE_CLOUD_PROJECT")
            or service_account_info["project_id"]
        )

        return storage.Client(
            project=project_id,
            credentials=credentials
        )

    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError("The Google service-account credentials are invalid.") from exc


def sync_firebase_storage_prefix(firebase_storage_bucket, firebase_prefix, local_dir):
    """
    Download a Firebase Storage / GCS prefix into a local directory.
    Authentication uses GOOGLE_SERVICE_ACCOUNT_JSON_BASE64.
    """
    if not firebase_storage_bucket:
        raise RuntimeError("FIREBASE_STORAGE_BUCKET is not configured.")

    bucket_name, prefix = split_gs_bucket_and_prefix(
        gs_uri_or_bucket=firebase_storage_bucket,
        prefix=firebase_prefix,
    )

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blobs = list(storage_client.list_blobs(bucket, prefix=prefix))

    downloaded_files = []
    skipped_blobs = []

    prefix_with_slash = f"{prefix.rstrip('/')}/" if prefix else ""

    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        if prefix_with_slash and blob.name.startswith(prefix_with_slash):
            relative_blob_path = blob.name[len(prefix_with_slash):]
        else:
            relative_blob_path = Path(blob.name).name

        if not relative_blob_path:
            skipped_blobs.append(blob.name)
            continue

        local_file_path = local_dir / relative_blob_path
        local_file_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_file_path))
        downloaded_files.append(str(local_file_path))

    return {
        "bucket_name": bucket_name,
        "prefix": prefix,
        "local_dir": str(local_dir),
        "downloaded_files": len(downloaded_files),
        "skipped_blobs": skipped_blobs,
    }


def sync_client_data_from_firebase(config_settings):
    if not config_settings.get("firebase_sync_client_data", False):
        return {
            "status": "skipped",
            "reason": "FIREBASE_SYNC_CLIENT_DATA is disabled.",
        }

    return sync_firebase_storage_prefix(
        firebase_storage_bucket=config_settings["firebase_storage_bucket"],
        firebase_prefix=config_settings["firebase_client_data_active_prefix"],
        local_dir=config_settings["active_client_data_dir"],
    )


def sync_corpus_from_firebase(config_settings):
    if not config_settings.get("firebase_sync_corpus", False):
        return {
            "status": "skipped",
            "reason": "FIREBASE_SYNC_CORPUS is disabled.",
        }

    return sync_firebase_storage_prefix(
        firebase_storage_bucket=config_settings["firebase_storage_bucket"],
        firebase_prefix=config_settings["firebase_corpus_prefix"],
        local_dir=config_settings["corpus_dir"],
    )


def config_base():
    # Database
    load_project_dotenv()
    db_url = os.getenv("VECTOR_DB")
    if not db_url:
        raise RuntimeError(
            "Database URL variable is not set. Please set to railway DB using public connection string."
        )
    document_inv = os.getenv("DOCUMENT_INV")

    # Middleware
    allowed_origins = os.getenv("ALLOWED_ORIGINS")
    port = os.getenv("PORT")
    
    # LLM
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm_provider = "openai"
    llm_model = "gpt-5.5" 

    # Embeddings
    embedding_provider = "openai"
    embedding_model = "text-embedding-3-small"
    embedding_dimension = 1536

    # Chunking
    chunk_size_tokens = 1000
    chunk_overlap_tokens = 150

    # Basic project checks
    if __name__ == "__main__":
        print(f"Database URL: {db_url}")
        print(f"Embedding model: {embedding_model}")
        print(f"LLM model: {llm_model}")
        print(f"LLM provider: {llm_provider}")
        print(f"Path for document inventory: {document_inv}")
        if openai_api_key:
            print("Open API key found in environment.")
        else:
            print("WARNING! Open API key not found.")

    # Runtime
    default_currency = "AED"
    default_jurisdiction = "UAE"
    project_name = "AI Construction Cost Estimation Platform"
    
    # Function return
    return_pack = {
        "db_url": db_url,
        "document_inv": document_inv,
        "allowed_origins": allowed_origins,
        "port": port,
        "openai_api_key": openai_api_key,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "chunk_size_tokens": chunk_size_tokens,
        "chunk_overlap_tokens": chunk_overlap_tokens,
        "default_currency": default_currency,
        "default_jurisdiction": default_jurisdiction,
        "project_name": project_name
    }

    return return_pack

    

def config_paths(client_data: str):
    load_project_dotenv()

    # Project paths
    project_root = Path(__file__).resolve().parent.parent
    corpus_dir = Path(os.getenv("CORPUS_DIR", project_root / "corpus")).expanduser()
    client_data_dir = project_root / "client_data"
    output_dir = project_root / "output"
    draft_report_dir = output_dir / "draft_report"
    exceptions_dir = output_dir / "exceptions"
    extracted_tables_dir = output_dir / "extracted_tables"
    extracted_text_dir = output_dir / "extracted_text"
    findings_register_dir = output_dir / "findings_register"

    # Resolving active client data folder
    active_client_data_dir = client_data_dir / client_data

    # Corpus packs are discovered dynamically after Firebase sync/local download.
    # The inventory scanner uses corpus_dir as the recursive scan root.
    corpus_packs = discover_immediate_subdirectories(corpus_dir)
    corpus_scan_dirs = [corpus_dir]

    firebase_storage_bucket = required_env("FIREBASE_STORAGE_BUCKET")
    firebase_corpus_prefix = required_env("FIREBASE_CORPUS_PREFIX").strip("/")
    firebase_client_data_prefix = required_env("FIREBASE_CLIENT_DATA_PREFIX").strip("/")
    firebase_client_data_active_prefix = (
        f"{firebase_client_data_prefix}/{client_data}".strip("/")
    )
    firebase_sync_client_data = bool_env("FIREBASE_SYNC_CLIENT_DATA", False)
    firebase_sync_corpus = bool_env("FIREBASE_SYNC_CORPUS", False)

    # Client data packs
    client_data_packs = {
        client_data: active_client_data_dir
    }

    # File handling
    supported_text_extensions = [".txt", ".md"]
    supported_document_extensions = [".pdf", ".docx", ".pptx"]
    supported_table_extensions = [".csv", ".xlsx", ".xls"]
    supported_web_extensions = [".json", ".xml", ".html", ".svg"]
    supported_design_extensions = [".dxf"]
    supported_image_extensions = [".png", ".jpg", ".jpeg", ".webp"]
    supported_archive_extensions = [".zip"]
    exclude_from_rag = {
        "validation_ground_truth_findings.csv",
        ".DS_Store",
        "Thumbs.db",
        ".identifier"
    }
    supported_file_types = (
        supported_text_extensions
        + supported_document_extensions
        + supported_table_extensions
        + supported_web_extensions
        + supported_design_extensions
        + supported_image_extensions
        + supported_archive_extensions
    )
    store_raw_files_in_db = False # To avoid storing large files like the ACFE Manual as raw binary in DB

    #  Basic project checks - to confirm required project folders exist
    required_input_folders = [
        client_data_dir,
        corpus_dir,
        active_client_data_dir,
    ]

    if __name__ == "__main__":
        for folder in required_input_folders:
            if not folder.exists():
                print(f"WARNING! Folder does not exist: {folder}")
            else:
                print("Folder checks passed!")

    # Print configuration summary - 1st if condition added to ensure this codebase
    # runs only when executed not when called.
    if __name__ == "__main__":
        print("Forensic RAG configuration loaded successfully!")
        print(f"Project root: {project_root}")
        print(f"Client data folder: {client_data_dir}")
        print(f"Active client data folder: {active_client_data_dir}")
        print(f"Corpus data folder: {corpus_dir}")
        print(f"Outputs folder:  {output_dir}")

    # Function return
    return_pack = {
        "project_root": project_root,
        "corpus_dir": corpus_dir,
        "client_data_dir": client_data_dir,
        "active_client_data_dir": active_client_data_dir,
        "output_dir": output_dir,
        "draft_report_dir": draft_report_dir,
        "exceptions_dir": exceptions_dir,
        "extracted_tables_dir": extracted_tables_dir,
        "extracted_text_dir": extracted_text_dir,
        "findings_register_dir": findings_register_dir,
        "corpus_packs": corpus_packs,
        "corpus_scan_dirs": corpus_scan_dirs,
        "firebase_storage_bucket": firebase_storage_bucket,
        "firebase_corpus_prefix": firebase_corpus_prefix,
        "firebase_client_data_prefix": firebase_client_data_prefix,
        "firebase_client_data_active_prefix": firebase_client_data_active_prefix,
        "firebase_sync_client_data": firebase_sync_client_data,
        "firebase_sync_corpus": firebase_sync_corpus,
        "client_data_packs": client_data_packs,
        "exclude_from_rag": exclude_from_rag,
        "supported_file_types": supported_file_types,
        "store_raw_files_in_db": store_raw_files_in_db,
        "supported_image_extensions": supported_image_extensions
    }

    return return_pack
