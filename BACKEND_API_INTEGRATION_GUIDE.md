# Firebase upload and existing pipeline — Claude handover

Implemented in source code; runtime verification not performed. Examples are illustrative, not live Railway responses. The revised scope adds **one** endpoint and retains the existing pipeline/API architecture.

## New upload API

**`POST /upload_files`**, multipart form data. Uploads complete within the request; this route returns a result, not a background job ID. Allow sufficient request time for archive validation and Firebase writes.

| Field | Required | Meaning |
|---|---|---|
| `destination` | Yes | Exactly `projects` or `corpus`. |
| `project_id` | For `projects` | Project folder identifier: 1–80 ASCII letters/digits/underscores/hyphens, starting with a letter or digit. Case is preserved. |
| `target_folder` | For flat corpus files | Relative folder beneath the exact BOK/BOK root, e.g. `BOK-016` or `BOK-016/Reference`. Omit only when every uploaded file/archive entry already has a relative folder. Do not include a leading/trailing slash. |
| `files` | Yes | Repeated file fields, or one ZIP containing documents. Do not mix ZIPs and individual files. |
| `overwrite` | No | `false` by default. Send `true` only when replacement of existing objects is intended. |

The bucket is taken from the existing `FIREBASE_STORAGE_BUCKET` setting. The caller cannot select a bucket or arbitrary storage root:

- `projects` → `gs://ai-construction-cost-est.firebasestorage.app/projects/<project_id>/`
- `corpus` → `gs://ai-construction-cost-est.firebasestorage.app/corpus/ai_construction_cost_estimation_platform/BOK/BOK/<target_folder>/`

Project uploads reject `target_folder`; corpus uploads reject `project_id`. Other form fields are rejected. No project registration, database upload records or source relocation occurs.

ZIP folder hierarchy is preserved exactly. A wrapper folder in a ZIP is not stripped: packaging `new_villa_002/Architectural/A101.pdf` inside the ZIP adds that folder beneath the selected project prefix. To get `projects/new_villa_002/Architectural/A101.pdf`, the archive entry should be `Architectural/A101.pdf`. Likewise, use either `target_folder=BOK-016` with files relative to that folder, or no target folder with archive entries such as `BOK-016/reference.pdf`. Numbered BOK folders are never automatically renamed or renumbered. Individual multipart filenames can carry relative paths; otherwise use the corpus target folder for placement.

Supported file extensions: PDF, DOCX, PPTX, XLS/XLSX, CSV, TXT, MD, MSG, JSON, XML, HTML, SVG, DXF, IFC, PNG, JPG/JPEG, WebP. ZIPs are unpacked before publication; nested ZIPs are rejected. Existing downstream parsers and extension handling are unchanged: accepting an IFC file for takeoff does not add IFC text extraction to the original RAG program.

Validation rejects empty files, traversal/absolute paths, unsafe path characters, symlinks/special ZIP entries, encrypted/invalid archives, and duplicate/conflicting file paths. The entire batch is staged and validated before Firebase writes. Fixed upload bounds: 500 files, 100 MiB per document, 256 MiB per ZIP, 512 MiB expanded total, 1,500 ZIP entries and 100:1 per-entry compression ratio. These checks run after multipart parsing; owner-managed request-size/time limits at the trusted upload entry point remain necessary. Document contents are processed by the existing downstream parsers, not parsed during upload.

### Upload-only access dependency

The pre-feature backend has **no suitable administrative access protection** to reuse. The new upload route therefore fails closed until the owner sets `UPLOAD_API_TOKEN` to a high-entropy secret of at least 32 characters. Trusted administrative callers send:

```http
X-Upload-Token: <upload-only secret>
```

This is a check inside `/upload_files`, before parsing the form. It adds no middleware, Firebase user-role framework, registry or authentication requirement to existing routes. Existing route authorization and `ALLOWED_ORIGINS` behavior are unchanged.

The Claude integration must use a trusted server/admin gateway to supply this secret; do not place it in a public frontend bundle. The owner must arrange that trusted access before enabling uploads. Existing Google service-account configuration is still used by the backend to authenticate to Firebase. Neither credential is returned in API responses.

### Project upload example

```bash
curl -X POST "$API_BASE/upload_files" \
  -H "X-Upload-Token: $UPLOAD_TOKEN" \
  -F 'destination=projects' \
  -F 'project_id=new_villa_002' \
  -F 'files=@new_villa_002.zip'
```

### Corpus upload example

```bash
curl -X POST "$API_BASE/upload_files" \
  -H "X-Upload-Token: $UPLOAD_TOKEN" \
  -F 'destination=corpus' \
  -F 'target_folder=BOK-016' \
  -F 'files=@new_reference_document.pdf'
```

### Result and errors

```json
{
  "status": "success",
  "destination": "corpus",
  "firebase_path": "gs://ai-construction-cost-est.firebasestorage.app/corpus/ai_construction_cost_estimation_platform/BOK/BOK/BOK-016/",
  "files_uploaded": 1,
  "files_failed": 0,
  "errors": []
}
```

A partial result uses `status=partial` and supplies errors such as:

```json
{"file":"reference.pdf","code":"object_conflict","error":"Object exists or changed during upload; replacement requires overwrite=true."}
```

| HTTP code | Meaning |
|---|---|
| 200 | `success` or `partial`; always inspect counts/errors. |
| 400 | Multipart parsing error, including multipart field/file-count limits. |
| 401 | Upload token missing or invalid. |
| 409 | All files failed because objects already exist or changed during conditional replacement. |
| 413 | File/archive/expanded-size limits exceeded. |
| 422 | Missing/invalid destination, identifier/path, unsupported input, empty/corrupt archive, or invalid overwrite value. |
| 502 | All file writes failed and at least one failure was a storage-service error. |
| 503 | Upload access token or Firebase credential configuration unavailable. |

Validation/configuration errors use `{"detail":"..."}`. Publication failures use the result structure above. Successful file writes are not rolled back when another file fails. With `overwrite=false`, Firebase generation-zero preconditions prevent accidental replacement even if another caller uploads concurrently. With `overwrite=true`, replacement is conditional on the current object's generation, preventing unnoticed concurrent changes. Retry only failed files as appropriate; resubmitting successful paths without explicit overwrite reports conflicts.

## Existing RAG workflow

All original routes, argument defaults, responses, streaming and in-memory job handling are restored. In particular, `rebuild_inventory` again defaults to **`Y`**. There are no `/projects` or `/projects/{project_id}` APIs, and no new project/scope parameters on chunking or embedding.

Run the existing stages in order, waiting for each job before submitting the next:

1. `GET /build_document_inventory/start?client_data=ai_construction_cost_estimation_platform&rebuild_inventory=Y`
2. `GET /extract_documents/start?client_data=ai_construction_cost_estimation_platform&rebuild_inventory=Y`
3. `GET /chunk_documents/start?rebuild_inventory=Y`
4. `GET /embed_chunks/start?rebuild_inventory=Y`

For an incremental run, use `N` on the same four calls. Original `/pipeline_jobs/{job_id}` and `/pipeline_jobs/{job_id}/result` endpoints still monitor the original in-memory jobs. HTTP 202 means queued, not completed. Original base routes and live embedding streams are unchanged. Existing jobs are not durable across process restart; inspect the existing operation result before treating a job as fully processed.

| Stage | Source and retained modes |
|---|---|
| Inventory | Synchronizes the full `projects/` tree and exact `corpus/ai_construction_cost_estimation_platform/BOK/BOK/` tree. `Y` replaces all inventory **rows**; `N` appends previously unseen relative paths. Existing relative paths retain their document IDs in both modes; new paths receive numeric `DOC_000001`-style IDs above the highest retained ID in inventory and `documents`. |
| Extraction | Reads the existing inventory for both client and corpus evidence. `Y` processes supported inventory rows; `N` processes supported rows with `ingest_status='pending'`. Existing extraction SQL/parsers are restored unchanged. |
| Chunking | Reads existing extracted text/tables for both sources. `Y` clears/rebuilds the existing chunks; `N` processes eligible documents without existing chunks. Original algorithms/SQL are restored unchanged. |
| Embedding | Updates existing chunks using the original model, dimensions, batching and retry loop. `N` selects missing vectors; `Y` also selects vectors labelled with another model. Original implementation is restored unchanged. |

`client_data` remains accepted/required where it was before and is returned in the same response fields. Full-source inventory runs include **all project folders for both Y and N**, irrespective of that compatibility argument; it is no longer used to relabel every client file. `CLIENT_DATA_LAYOUT` cannot narrow these required full-source scans. Actual project attribution comes from the first folder beneath `projects/`.

| Firebase object | Local inventory relative path | Source group | Corpus pack |
|---|---|---|---|
| `projects/new_villa_002/Architectural/A101.pdf` | `client_data/new_villa_002/Architectural/A101.pdf` | `client_data` | `new_villa_002` |
| `projects/private_villa_001/Structural/S101.pdf` | `client_data/private_villa_001/Structural/S101.pdf` | `client_data` | `private_villa_001` |
| `corpus/ai_construction_cost_estimation_platform/BOK/BOK/BOK-016/book.pdf` | `corpus/BOK/BOK/BOK-016/book.pdf` | `corpus_data` | `BOK` |

The original classification uses the second relative folder for deep corpus paths, so these BOK/BOK documents retain `corpus_pack=BOK`; the numbered `BOK-016` remains in the relative path for existing retrieval. Syncing the deeper Firebase root into local `corpus/BOK/BOK/` preserves that convention. The wider Firebase/local corpus tree is not scanned. With synchronization enabled, inventory uses this run's downloaded source listing rather than stale cache files. With synchronization disabled, it retains local-folder scanning within the same intended local roots. Files placed directly at `projects/<filename>` have no project folder and are excluded.

**Preserved baseline limitations:** `N` does not refresh changed content at an already-known path or automatically reclassify previously mislabelled inventory. Overwriting an object does not change that rule; use the established full `Y` workflow when required. Extraction `Y` upserts/processes current inventory but does not purge historical documents absent from it; the original full chunk rebuild can still see their existing extracted evidence. No new clearing of those downstream tables or checksum/revision architecture was added. These are baseline behaviors, not a promise that each stage clears every old record uniformly.

## Existing takeoff API

`GET /generate_boq_takeoff/start?project_id=new_villa_002`

Combines `TAKEOFF_FIREBASE_ROOT=gs://ai-construction-cost-est.firebasestorage.app/projects` with the validated project ID to list **`projects/new_villa_002/`** recursively. The trailing slash excludes siblings such as `projects/new_villa_002_extra/`. A fixed-project takeoff root is rejected rather than redirected to another project.

Document matching, PDF/image rendering, DXF/IFC parsing, vision, quantity formulas, confidence validation, provisional allowances, result structures and writes to `takeoff_layer` are unchanged. BOQ v2, reviewer and chatbot implementations remain unchanged. Use `write_workbook` and `max_items_per_division` only with the existing BOQ v2 API, not takeoff.

## Required Railway changes

| Setting | Action |
|---|---|
| `FIREBASE_CORPUS_PREFIX` | Change to `corpus/ai_construction_cost_estimation_platform/BOK/BOK`. |
| `UPLOAD_API_TOKEN` | Configure only to enable the new upload route for trusted administrative callers; secret of at least 32 characters. No existing route uses it. |

Retain the supplied values of `FIREBASE_STORAGE_BUCKET`, `FIREBASE_CLIENT_DATA_PREFIX=projects`, both `FIREBASE_SYNC_*` flags (`Y`), `GOOGLE_CLOUD_PROJECT`, and `TAKEOFF_FIREBASE_ROOT` pointing to the projects root. Retain `VECTOR_DB`, the existing service-account/OpenAI credential variable names, OCR/model/embedding/takeoff/build settings, `CORPUS_DIR` and `ALLOWED_ORIGINS`. No new database or pipeline setting is required. The only added package is `python-multipart>=0.0.18` for form parsing.

**No new PostgreSQL tables, columns, indexes or migration scripts are required.** Inventory rebuilds delete/reinsert data within the existing table and transaction, rather than using pandas to drop/recreate its structure. ID assignment reads existing inventory and historical `documents` mappings under an inventory write lock before any deletion; conflicting path/ID mappings abort without changing inventory. In the checked-in schemas, downstream foreign keys reference `documents.document_id`, not the inventory table, so this inventory DELETE/INSERT does not delete or cascade into historical evidence. A missing inventory table is reported; it is not created automatically. Existing checked-in SQL schemas are unchanged. Do not apply the abandoned migration files or their older archived copies.

Owner runtime verification remains necessary for upload access, multipart/provider behavior, overwrite handling, actual source folders, database compatibility, full/incremental pipeline results and takeoff/BOQ output. No deployment or external-service testing was performed. See [implementation and verification report](docs/BACKEND_IMPLEMENTATION_REPORT.md).
