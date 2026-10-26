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
import re
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, inspect, text

# Import paths and settings from the config file
from global_rag.scripts import config
from global_rag.scripts.project_paths import validate_project_id, contained_path



def document_ids_for_paths(relative_paths, known_documents):
    """Keep existing identities and allocate above all retained numeric IDs."""
    ids_by_path = {}
    paths_by_id = {}
    max_document_number = 0
    for row in known_documents:
        document_id, relative_path = row["document_id"], row["relative_path"]
        if not document_id or not relative_path:
            raise ValueError("Existing document identities contain an empty ID or path; review them before rebuilding.")
        if (relative_path in ids_by_path and ids_by_path[relative_path] != document_id
                or document_id in paths_by_id and paths_by_id[document_id] != relative_path):
            raise ValueError("Conflicting existing document IDs/relative paths; inventory was not changed.")
        ids_by_path[relative_path] = document_id
        paths_by_id[document_id] = relative_path
        match = re.fullmatch(r"DOC_([0-9]+)", document_id)
        if match:
            max_document_number = max(max_document_number, int(match.group(1)))

    result = []
    for relative_path in relative_paths:
        if relative_path not in ids_by_path:
            max_document_number += 1
            ids_by_path[relative_path] = f"DOC_{max_document_number:06d}"
        result.append(ids_by_path[relative_path])
    return result


def build_document_inventory(client_data: str, rebuild_inventory: str = "Y"):
    config_settings = config.config_paths(client_data=client_data)
    config_base = config.config_base()
    rebuild_inventory = rebuild_inventory.strip().upper()

    if rebuild_inventory not in ["Y", "N"]:
        raise ValueError("rebuild_inventory must be 'Y' or 'N'.")

    firebase_corpus_sync = config.sync_corpus_from_firebase(config_settings)
    firebase_client_data_sync = config.sync_client_data_from_firebase(config_settings)
    # Consume the listing without changing the existing API's sync summaries.
    source_files = {
        config_settings["active_client_data_dir"]: firebase_client_data_sync.pop("source_files", None),
        config_settings["corpus_scan_dirs"][0]: firebase_corpus_sync.pop("source_files", None),
    }
    
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
        *config_settings["corpus_scan_dirs"],
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
    if not table_exists:
        raise RuntimeError("The existing build_document_inventory table is required; this program does not create tables.")
    # Scan files and build inventory for RAG
    inventory_rows = []
    scan_folders = [str(scan_root) for scan_root in folders_to_scan]
    missing_folders = []

    for scan_root in folders_to_scan:

        # Skip if folder does not exist
        if not scan_root.exists():
            missing_folders.append(str(scan_root))
            print(f"WARNING: Folder does not exist and will be skipped: {scan_root}")
            continue

        # With synchronization enabled, scan exactly this run's Firebase listing.
        # Without it, preserve the existing local-folder workflow.
        downloaded = source_files.get(scan_root)
        files = (Path(value) for value in downloaded) if downloaded is not None else scan_root.rglob("*")
        for file_path in files:

            # Skip folders
            if file_path.is_dir():
                continue

            # Skip files inside ignored folders
            if any(part in ignore_folder_names for part in file_path.parts):
                continue

            # Skip ignored filenames
            if file_path.name in ignore_file_names:
                continue

            # Skip Windows alternate data stream metadata copied into Linux files.
            if file_path.name.lower().endswith(":zone.identifier"):
                continue

            # Skip hidden/ temporary files
            if any(file_path.name.startswith(prefix) for prefix in ignore_file_prefixes):
                continue

            contained_path(scan_root, file_path.relative_to(scan_root).as_posix())

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
                if len(relative_to_group.parts) < 2:
                    # Files directly under projects/ have no project attribution.
                    continue
                corpus_pack = validate_project_id(relative_to_group.parts[0])
            elif len(relative_to_group.parts) > 2:
                corpus_pack = relative_to_group.parts[1]
            elif len(relative_to_group.parts) > 1:
                corpus_pack = relative_to_group.parts[0]
            else:
                corpus_pack = source_group

            if source_group == "corpus_data":
                inventory_relative_path = str(Path("corpus") / relative_to_group)
            else:
                inventory_relative_path = str(file_path.relative_to(config_settings["project_root"]))

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
            elif file_ext == ".msg":
                extraction_method_hint = "extract_msg_email_text"
            elif file_ext == ".html":
                extraction_method_hint = "read_html_file"
            elif file_ext == ".json":
                extraction_method_hint = "read_json_file"
            elif file_ext == ".svg":
                extraction_method_hint = "extract_svg_text"
            elif file_ext == ".dxf":
                extraction_method_hint = "extract_dxf_text"
            elif file_ext in config_settings["supported_image_extensions"]:
                extraction_method_hint = "extract_image_description"
            elif file_ext == ".zip":
                extraction_method_hint = "extract_file_and_parse"
            else:
                extraction_method_hint = "unknown"

            # Section D: Decide wether file should be indexed in RAG
            # - PDFs, DOCXs, TXTs, MDs, HTMLs, JSONs, PPTX, tables,
            #   SVG/DXF design text, and PNG image descriptions are indexed in RAG.
            # - ZIP is not indexed directly, it is extracted first
            if file_ext in config_settings["supported_file_types"] and file_ext != ".zip":
                index_in_rag = "Yes"
            elif file_ext in [".identifier", ".pdf:mshield", ".json:mshield", ".xml:mshield", ".txt:mshield", ".csv:mshield"]:
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

            # Section H: Create inventory row
            inventory_rows.append(
                {
                    "document_id": None,  # Assigned from retained identities in the write transaction.
                    "source_group": source_group,
                    "corpus_pack": corpus_pack,
                    "file_name": file_path.name,
                    "file_extension": file_ext,
                    "relative_path": inventory_relative_path,
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

    source_group_counts = {}
    pack_counts = {}

    for row in inventory_rows:
        source_group = row["source_group"]
        corpus_pack = row["corpus_pack"]
        source_group_counts[source_group] = source_group_counts.get(source_group, 0) + 1
        pack_counts[corpus_pack] = pack_counts.get(corpus_pack, 0) + 1

    with engine.begin() as conn:
        # Serialize inventory ID allocation and writes without any schema changes.
        conn.execute(text("LOCK TABLE build_document_inventory IN SHARE ROW EXCLUSIVE MODE"))
        existing_inventory = conn.execute(text(
            "SELECT document_id, relative_path FROM build_document_inventory"
        )).mappings().all()
        historical_documents = conn.execute(text(
            "SELECT document_id, relative_path FROM documents"
        )).mappings().all()
        if rebuild_inventory == "N":
            existing_relative_paths = {row["relative_path"] for row in existing_inventory}
            inventory_df = inventory_df[
                ~inventory_df["relative_path"].isin(existing_relative_paths)
            ].copy()
        # Include historical documents so removed/reintroduced paths keep their
        # identities and new files cannot reuse an ID still used downstream.
        inventory_df["document_id"] = document_ids_for_paths(
            inventory_df["relative_path"].tolist(),
            list(existing_inventory) + list(historical_documents),
        )
        # Preserve full Y replacement of inventory DATA, without pandas dropping
        # and recreating the existing table (which would alter its schema).
        if rebuild_inventory == "Y":
            conn.execute(text("DELETE FROM build_document_inventory"))
        if not inventory_df.empty:
            column_sql = ", ".join(columns_to_insert)
            value_sql = ", ".join(f":{column}" for column in columns_to_insert)
            conn.execute(
                text(f"INSERT INTO build_document_inventory ({column_sql}) VALUES ({value_sql})"),
                inventory_df[columns_to_insert].to_dict(orient="records"),
            )

    rows_written_count = len(inventory_df)

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
        "client_data_layout": config_settings["client_data_layout"],
        "active_client_data_dir": str(config_settings["active_client_data_dir"]),
        "firebase_corpus_sync": firebase_corpus_sync,
        "firebase_client_data_sync": firebase_client_data_sync,
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
    print(build_document_inventory(client_data="synthetic_construction_cost_rag_pack", rebuild_inventory="Y"))
