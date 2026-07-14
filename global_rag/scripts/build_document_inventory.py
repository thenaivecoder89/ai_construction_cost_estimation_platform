# File_name: build_document_inventory.py
# Purpose: To scan the client data and corpus
# folders to create a master document inventory
# This inventory becomes the control file for the
# rest of the pipeline:
# - extraction
# - database loading
# - chunking
# - embedding
# - evidence traceability

import pandas as pd
import hashlib
from datetime import datetime
from sqlalchemy import create_engine, inspect, text

# Import paths and settings from the config file
from global_rag.scripts import config


def build_document_inventory(client_data: str, rebuild_inventory: str = "Y"):
    config_settings = config.config_paths(client_data=client_data)
    config_base = config.config_base()
    rebuild_inventory = rebuild_inventory.strip().upper()

    if rebuild_inventory not in ["Y", "N"]:
        raise ValueError("rebuild_inventory must be 'Y' or 'N'.")
    
    # Establish DB connection
    engine = create_engine(
        url=config_base["db_url"],
        pool_pre_ping=True
    )

    # Define which root files to scan
    # We only scan corpus and client_data
    # not output and scripts since these do not contain evidence data
    folders_to_scan = [
        config_settings["active_client_data_dir"],
        config_settings["corpus_dir"]
    ]

    # Define files and folders to ignore when scanning
    ignore_file_names = config_settings["exclude_from_rag"]
    ignore_file_prefixes = [
        "$", # Temporary office files
        "." # Hidden files
    ]
    ignore_folder_names = [
        "__pycache__",
        ".git"
    ]

    # Define inventory output columns
    inventory_columns = [
        "document_id", # Unique evidence reference
        "source_group", # Separates corpus from client data
        "corpus_pack", # Classifies knowledge area
        "file_name", # Human-readable source name
        "file_extension", # Determines extraction method
        "relative_path", # Portable source traceability
        "absolute_path", # Exact local file location
        "file_size_bytes", # Basic file completeness check
        "last_modified_datetime", # Source version indicator
        "sha256_checksum", # Detects file tampering
        "supported_file_type", # Confirms processability
        "extraction_method_hint", # Guides extraction logic
        "index_in_rag", # Controls embedding eligibility
        "ingest_status", # Tracks pipeline progress
        "notes", # Captures manual observations
    ]

    inventory_table_name = "build_document_inventory"
    table_exists = inspect(engine).has_table(inventory_table_name)
    existing_relative_paths = set()
    max_document_number = 0

    if rebuild_inventory == "N" and table_exists:
        existing_inventory_df = pd.read_sql(
            text(f"""
                SELECT
                    document_id,
                    relative_path
                FROM {inventory_table_name}
            """),
            engine
        )

        existing_relative_paths = set(
            existing_inventory_df["relative_path"].dropna().astype(str).tolist()
        )

        with engine.begin() as conn:
            max_document_number = conn.execute(
                text(f"""
                    SELECT COALESCE(
                        MAX(CAST(REPLACE(document_id, 'DOC_', '') AS INTEGER)),
                        0
                    ) AS max_document_number
                    FROM {inventory_table_name}
                    WHERE document_id LIKE 'DOC_%'
                """)
            ).scalar()

    # Scan files and build inventory for RAG
    inventory_rows = []
    document_counter = int(max_document_number) + 1
    scan_folders = [str(scan_root) for scan_root in folders_to_scan]
    missing_folders = []

    for scan_root in folders_to_scan:

        # Skip if folder does not exist
        if not scan_root.exists():
            missing_folders.append(str(scan_root))
            print(f"WARNING: Folder does not exist and will be skipped: {scan_root}")
            continue

        # Recursively scan all files in the folder
        for file_path in scan_root.rglob("*"):

            # Skip folders
            if file_path.is_dir():
                continue

            # Skip files inside ignored folders
            if any(part in ignore_folder_names for part in file_path.parts):
                continue

            # Skip ignored filenames
            if file_path.name in ignore_file_names:
                continue

            # Skip hidden/ temporary files
            if any(file_path.name.startswith(prefix) for prefix in ignore_file_prefixes):
                continue

            # Get  file extension in lowercase
            file_ext = file_path.suffix.lower()

            # Determine if this is a supported file type
            supported_file_type = "Yes" if file_ext in config_settings["supported_file_types"] else "No"

            # Section A: Classify source group - tells us whether the document is a client evidence or methodology/ context corpus
            if config_settings["active_client_data_dir"] in file_path.parents:
                source_group = "client_data"
                relative_to_group = file_path.relative_to(config_settings["active_client_data_dir"])
            elif config_settings["corpus_dir"] in file_path.parents:
                source_group = "corpus_data"
                relative_to_group = file_path.relative_to(config_settings["corpus_dir"])
            else:
                source_group = "unknown"
                relative_to_group = file_path.relative_to(config_settings["project_root"])

            # Section B: Identify corpus/ client data pack
            if source_group == "client_data":
                corpus_pack = client_data
            elif len(relative_to_group.parts) > 1:
                corpus_pack = relative_to_group.parts[0]
            else:
                corpus_pack = source_group

            # Section C: Decide extraction method hint
            if file_ext == ".pdf":
                extraction_method_hint = "extract_pdf_text"
            elif file_ext == ".docx":
                extraction_method_hint = "extract_docx_text"
            elif file_ext in [".pptx", ".ppt"]:
                extraction_method_hint = "extract_ppt_text"
            elif file_ext in [".xlsx", ".xls"]:
                extraction_method_hint = "extract_excel_tables"
            elif file_ext == ".csv":
                extraction_method_hint = "read_csv_table"
            elif file_ext in [".txt", ".md"]:
                extraction_method_hint = "read_text_file"
            elif file_ext == ".html":
                extraction_method_hint = "read_html_file"
            elif file_ext == ".json":
                extraction_method_hint = "read_json_file"
            elif file_ext == ".zip":
                extraction_method_hint = "extract_file_and_parse"
            else:
                extraction_method_hint = "unknown"

            # Section D: Decide wether file should be indexed in RAG
            # - PDFs, DOCXs, TXTx, MDs, HTMLs and JSONs are indexed in RAG
            # - CSVs, XLSX/ XLSs are not directly indexed in RAG, they are used for analytical purposes
            # - ZIP is not indexed directly, it is extracted first
            if file_ext in [".pdf", ".docx", ".txt", ".md", ".html", ".json", ".pptx"]:
                index_in_rag = "Yes"
            elif file_ext in [".csv", ".xlsx", ".xls", ".identifier", ".pdf:mshield", ".json:mshield", ".xml:mshield", ".txt:mshield", ".csv:mshield"]:
                index_in_rag = "No"
            elif file_ext == ".zip":
                index_in_rag = "No_extract_archive_first"
            else:
                index_in_rag = "No"
            
            # Section E: Calculate file size and modified date
            file_size = file_path.stat().st_size
            last_modified_datetime = datetime.fromtimestamp(
                file_path.stat().st_mtime
            ).strftime("%Y-%m-%d %H:%M:%S")

            # Section F: Calculate SHA256 checksum to detect if a source file changes later - for traceability
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as file:
                while True:
                    file_chunk = file.read(1024 * 1024) # Read 1MB of a file each time
                    if not file_chunk:
                        break
                    sha256_hash.update(file_chunk)
            
            sha256_checksum = sha256_hash.hexdigest() # Finalize a 64-character hexadecimal string after completion of hash calculation (sha256_hash parameter)

            # Section G: Create document ID
            document_ID = f"DOC_{document_counter:06d}"
            document_counter += 1

            # Section H: Create inventory row
            inventory_rows.append(
                {
                    "document_id": document_ID,
                    "source_group": source_group,
                    "corpus_pack": corpus_pack,
                    "file_name": file_path.name,
                    "file_extension": file_ext,
                    "relative_path": str(file_path.relative_to(config_settings["project_root"])),
                    "absolute_path": str(file_path),
                    "file_size_bytes": file_size,
                    "last_modified_datetime": last_modified_datetime,
                    "sha256_checksum": sha256_checksum,
                    "supported_file_type": supported_file_type,
                    "extraction_method_hint": extraction_method_hint,
                    "index_in_rag": index_in_rag,
                    "ingest_status": "pending",
                    "notes": "",
                }
            )

    # Write inventory to DB
    inventory_df = pd.DataFrame(inventory_rows, columns=inventory_columns)
    inventory_df = inventory_df[inventory_columns]
    inventory_df.index.name = "SNo."
    columns_to_insert  = [
        "document_id",
        "source_group",
        "corpus_pack",
        "file_name",
        "file_extension",
        "relative_path",
        "absolute_path",
        "file_size_bytes",
        "last_modified_datetime",
        "sha256_checksum",
        "supported_file_type",
        "extraction_method_hint",
        "index_in_rag",
        "ingest_status",
        "notes"
    ]

    scanned_files_count = len(inventory_df)

    if rebuild_inventory == "N":
        inventory_df = inventory_df[
            ~inventory_df["relative_path"].isin(existing_relative_paths)
        ].copy()

        inventory_df["document_id"] = [
            f"DOC_{document_number:06d}"
            for document_number in range(
                int(max_document_number) + 1,
                int(max_document_number) + 1 + len(inventory_df)
            )
        ]

    rows_written_count = len(inventory_df)
    source_group_counts = {}
    pack_counts = {}

    for row in inventory_rows:
        source_group = row["source_group"]
        corpus_pack = row["corpus_pack"]
        source_group_counts[source_group] = source_group_counts.get(source_group, 0) + 1
        pack_counts[corpus_pack] = pack_counts.get(corpus_pack, 0) + 1

    with engine.begin() as conn:
        inventory_df[columns_to_insert].to_sql(
            name=inventory_table_name,
            con=conn,
            index=False,
            method="multi",
            if_exists="replace" if rebuild_inventory == "Y" else "append"
        )

    if __name__ == "__main__":
        # Print basic summary
        print(f"Document inventory successfully loaded.")
        print(f"Inventory mode: {'rebuild' if rebuild_inventory == 'Y' else 'update'}")
        print(f"Total files scanned: {scanned_files_count}")
        print(f"Total files written: {rows_written_count}")
        print("\nFiles by source group:")
        for source_group, count in source_group_counts.items():
            print(f"- {source_group}: {count}")

        print("\nFiles by corpus/client pack:")
        for corpus_pack, count in pack_counts.items():
            print(f"- {corpus_pack}: {count}")

        # Show unsupported files, if any
        unsupported_files = [
            row for row in inventory_rows
            if row["supported_file_type"] == "No"
        ]

        if unsupported_files:
            print("\nWARNING: Unsupported files found:")
            for row in unsupported_files:
                print(f"- {row['relative_path']}")
        else:
            print("\nNo unsupported files found.")

    # Return confirmation values
    return {
        "message": "Document inventory successfully loaded.",
        "client_data": client_data,
        "mode": "rebuild" if rebuild_inventory == "Y" else "update",
        "project_root": str(config_settings["project_root"]),
        "active_client_data_dir": str(config_settings["active_client_data_dir"]),
        "scan_folders": scan_folders,
        "missing_folders": missing_folders,
        "scanned_files": scanned_files_count,
        "source_group_counts": source_group_counts,
        "pack_counts": pack_counts,
        "existing_files_skipped": scanned_files_count - rows_written_count,
        "rows_written": rows_written_count,
        "output_table": inventory_table_name,
    }

if __name__ == "__main__":
    print("Executing the build program")
    print(build_document_inventory(client_data="TXN_ADDC_001", rebuild_inventory="Y"))
