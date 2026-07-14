# File_name: config.py
# Purpose: To store all common paths, database settings, model settings,
# and project-level constants used by the RAG pipeline

from pathlib import Path
import os
from dotenv import load_dotenv

def config_base():
    # Database
    load_dotenv()
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
    project_name = "AI Investment RAG"
    
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
    # Project paths
    project_root = Path(__file__).resolve().parent.parent
    corpus_dir =  project_root / "corpus"
    client_data_dir = project_root / "client_data"
    output_dir = project_root / "output"
    draft_report_dir = output_dir / "draft_report"
    exceptions_dir = output_dir / "exceptions"
    extracted_tables_dir = output_dir / "extracted_tables"
    extracted_text_dir = output_dir / "extracted_text"
    findings_register_dir = output_dir / "findings_register"

    # Resolving active client data folder
    active_client_data_dir = client_data_dir / client_data

    # Corpus packs
    corpus_packs = {
        "01_RAG_Public_Deal_Documents": corpus_dir / "01_RAG_Public_Deal_Documents",
        "02_Benchmark_and_Market_Data": corpus_dir / "02_Benchmark_and_Market_Data",
        "03_ML_Input_Datasets": corpus_dir / "03_ML_Input_Datasets",
        "04_Seed_Libraries": corpus_dir / "04_Seed_Libraries"
    }

    # Client data packs
    client_data_packs = {
        client_data: active_client_data_dir
    }

    # File handling
    supported_text_extensions = [".txt", ".md"]
    supported_document_extensions = [".pdf", ".docx", ".pptx"]
    supported_table_extensions = [".csv", ".xlsx", ".xls"]
    supported_web_extensions = [".json", ".xml", ".html"]
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
        + supported_archive_extensions
    )
    store_raw_files_in_db = False # To avoid storing large files like the ACFE Manual as raw binary in DB

    # Corpus folder mapping - used for classifying documents based on folder location
    corpus_folder_map = {
        "client_data": "client_evidence",
        "01_RAG_Public_Deal_Documents": "01_RAG_Public_Deal_Documents",
        "02_Benchmark_and_Market_Data": "02_Benchmark_and_Market_Data",
        "03_ML_Input_Datasets": "03_ML_Input_Datasets",
        "04_Seed_Libraries": "04_Seed_Libraries"
    }

    #  Basic project checks - to confirm required project folders exist
    required_input_folders = [
        client_data_dir,
        corpus_dir,
        active_client_data_dir,
        corpus_packs["01_RAG_Public_Deal_Documents"],
        corpus_packs["02_Benchmark_and_Market_Data"],
        corpus_packs["03_ML_Input_Datasets"],
        corpus_packs["04_Seed_Libraries"]
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
        "client_data_packs": client_data_packs,
        "exclude_from_rag": exclude_from_rag,
        "supported_file_types": supported_file_types,
        "store_raw_files_in_db": store_raw_files_in_db,
        "corpus_folder_map": corpus_folder_map
    }

    return return_pack