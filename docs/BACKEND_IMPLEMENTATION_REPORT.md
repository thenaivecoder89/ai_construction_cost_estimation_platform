# Simplified backend revision report

Branch: `feature/project-onboarding-rag-takeoff`. Functional baseline: commit `1a95f34` (pre-feature implementation). The revised attachment was read in full before changing code.

**Implemented in source code; runtime verification not performed.**

## Programs changed or restored

| Python program | Final change relative to the baseline |
|---|---|
| `global_rag/scripts/config.py` | Synchronize the full projects tree and exact BOK/BOK tree. Retain the original local project/corpus/output parents and all model/database/OCR/extension settings. Preserve BOK/BOK in local corpus paths, use slash-terminated prefixes, validate download containment, and provide a per-run file list to inventory without persistent metadata. |
| `global_rag/scripts/build_document_inventory.py` | Scan both required source roots in Y and N; derive each client ID from its project folder. Preserve original corpus classification, columns and path-based N logic; retain existing document IDs by relative path in both modes and reserve IDs present in historical `documents`. Y deletes/reloads inventory rows within the existing table instead of allowing pandas to drop/recreate its schema. No new constraints, fields, revisions or registration. |
| `global_rag/scripts/boq_takeoff_layer.py` | Only discovery changes: combine the configured projects root with the requested project ID. All other functions, parsers/calculations and result shapes match the baseline. |
| `global_rag/scripts/main.py` | Restore every pre-existing function, route/default, response, stream, job-global and CORS-global statement. Add only `POST /upload_files`, its imports and an upload-only token check. No global authentication behavior changes. |
| `global_rag/scripts/project_uploads.py` | Simplified to direct Firebase ZIP/multi-file uploads to projects or corpus. Includes validation, bounded safe archive extraction, explicit conditional overwrite and result counts/errors. Contains no SQL, registry or persistent tracking. |
| `global_rag/scripts/project_paths.py` | Reduced to a small standard-library path helper used by uploads, synchronization and takeoff. No identity hashing, registry mapping or job functionality. |
| `global_rag/scripts/extract_documents.py` | Restored byte-for-byte to `1a95f34`; no remaining final diff. |
| `global_rag/scripts/chunk_documents.py` | Restored byte-for-byte to `1a95f34`; no remaining final diff. |
| `global_rag/scripts/embed_chunks.py` | Restored byte-for-byte to `1a95f34`; no remaining final diff. |

`requirements.txt` retains only the added `python-multipart>=0.0.18` requirement; the prior explicit `google-auth` addition is removed. Existing storage-client dependencies remain unchanged. BOQ v2, reviewer, chatbot, checked-in SQL schemas and `railway.json` were not modified.

## Removed files

- `global_rag/scripts/backend_auth.py`
- `global_rag/scripts/project_registry.py`
- `global_rag/scripts/pipeline_jobs.py`
- `migrations/001_project_onboarding.sql`
- `migrations/002_legacy_registration.sql.example`
- `tools/check_backend_source.py`
- `docs/API_ROUTE_REFERENCE.md`
- `docs/BACKEND_DEPLOYMENT_NOTES.md`

The API guide and this report were rewritten for the simplified implementation. Remaining Python source has no imports of the deleted modules and no dependencies on their tables/columns. The original `/pipeline_jobs/...` **routes and in-memory functions in main.py** remain, as requested; they do not use the deleted module or any database job table.

The pre-existing untracked `daily_sql_script.sql` and user-added `migrations.zip` were left untouched. That archive is not part of the revised implementation and must not be used as deployment instructions. No production rollback, deletion, migration, commit, push or deployment was performed.

## Necessary choices and baseline ambiguities

- The original inventory `to_sql(if_exists='replace')` would drop/recreate the table, contradicting the revised strict schema-preservation requirement. The replacement uses one transaction to delete existing inventory rows for Y and insert the same original columns. N inserts only the original newly identified paths. This preserves intended full/update **record** behavior without schema DDL. Existing production-only constraints/triggers still require owner runtime review; no schema change is supplied or required by the checked-in definitions.
- Full-source inventory includes both required trees for Y and N. The retained `client_data` parameter is not allowed to assign one ID to files from different project folders. No new project-scoped parameters or registry lookup are used.
- For the supplied BOK/BOK layout, baseline classification yields `corpus_pack=BOK`; numbered folder names remain in the existing relative paths. Keeping the local BOK/BOK suffix preserves this result while narrowing the remote root.
- Original incremental inventory is **new-relative-path based**, not changed-checksum based. Original incremental extraction selects pending rows; chunking selects eligible documents without chunks. Original embedding Y selects missing/model-mismatched vectors. These rules, batch/retry behavior and mode defaults are restored, not replaced with revision tracking.
- Original extraction Y does not remove historical documents absent from current inventory; original chunk Y can include their extracted evidence. Strict source-only purging across all downstream tables would require changing those baseline semantics, which the revised instructions prohibit. No such purge was added. Full inventory rebuilds now preserve existing relative-path IDs; only new paths receive fresh numeric IDs.
- No suitable existing admin access protection was found. The single new write route is disabled until the owner configures `UPLOAD_API_TOKEN`; the token check is local to that route. Existing endpoints and CORS retain their original behavior. The Claude integration needs trusted server/admin-gateway access; no browser credential distribution or platform authentication framework is included.

## Permitted checks performed

1. **Compilation:** `python3 -m py_compile` succeeded for the nine modified/restored Python modules listed above. No application modules were imported or started.
2. **Import consistency:** AST/source inspection confirmed no imports of the three deleted modules and no remaining abandoned registry/file-table/revision-column dependencies.
3. **Self-contained path checks:** confirmed project and corpus upload prefixes, full projects inventory root, narrow BOK/BOK local/remote roots, correct attribution for two different project folders, retained BOK corpus classification, selected-project takeoff prefixes, traversal rejection and sibling-prefix exclusion. Only pure helpers and isolated path/classification logic with literal settings were evaluated; no `.env`, credentials or external clients were loaded.
4. **API preservation:** AST comparison confirmed all **27 original routes**, their defaults and function bodies, and all existing job/CORS global statements match the baseline. Exactly one new route/function, `/upload_files`, is added.
5. **Database compatibility:** source comparison confirmed identical inventory column lists and the existing N new-relative-path filter, with the ID-allocation correction described below; extraction/chunking/embedding are byte-identical to baseline; all takeoff functions except Firebase discovery are unchanged. Existing SQL schema files have no diff. No new schema objects or migration definitions remain in the implementation.
6. **Git review:** reviewed the final diff and ran `git diff --check`; only the intended backend/dependency/documentation changes remain, with unrelated user files preserved.

A first local checker process terminated with a Python interpreter segmentation fault; the unchanged source checker then passed when rerun with Python's fault handler enabled. No application or external-service execution occurred in either run. No persistent test suite or new checking framework was added.

No frontend/browser simulation, database/PgVector mock, database SQL execution, Firebase/Railway/OpenAI request, package installation or deployment was performed. The owner must verify multipart/access behavior, Firebase uploads/overwrites, actual production schema and source layouts, Y/N results and takeoff output after their own deployment.

**No new PostgreSQL tables, columns, indexes or migration scripts are required.** The [API guide](../BACKEND_API_INTEGRATION_GUIDE.md) contains the upload contract and the only required configuration changes.


## Final document-ID correction

Only `build_document_inventory.py` changed in this correction, alongside these documentation updates. After scanning, the existing write transaction locks the inventory table, reads ID/path mappings from inventory and `documents`, retains known IDs, and assigns new numeric IDs above the highest retained number. This covers changed scan order, newly added paths, and paths returning after removal from inventory but still present in historical `documents`. Existing nonnumeric IDs are retained too. Ambiguous ID/path mappings cause a validation error before DELETE/INSERT; no historical records are automatically relabelled. Y still replaces all inventory rows; N still inserts only paths absent from current inventory. APIs, other pipeline programs and schema definitions are unchanged.

Foreign-key review: `global_rag/scripts/global_rag_schema.sql` defines inventory with its own `s_no` primary key and no foreign keys to/from inventory. `documents.document_id` is the referenced primary key for extracted text/tables/rows, chunks and other evidence; takeoff references it in `takeoff_layer_schema.sql`. None of the checked-in schema definitions references inventory as a foreign-key target. Therefore the inventory-only DELETE/INSERT is compatible with those definitions and cannot cascade into downstream evidence. Production-only constraints/triggers were not inspected.

Local verification for this correction: Python compilation, isolated standard-library checks of the ID allocator (reordered/new/restored paths, reserved historical IDs, numeric allocation, existing nonnumeric IDs and conflicting mappings), static transaction/filter/schema inspection, confirmation that all other Python modules remained unchanged, and `git diff --check`. No application import, database mock, SQL execution or production-service access was performed.
