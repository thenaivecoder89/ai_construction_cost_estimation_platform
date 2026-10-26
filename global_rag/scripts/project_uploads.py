"""Upload validated documents to projects/ or the exact BOK/BOK Firebase root.

No database operations, registration or job tracking. Uses config.get_storage_client.
"""
import logging
import mimetypes
import stat
import tempfile
import zipfile
from pathlib import Path
from google.api_core.exceptions import NotFound, PreconditionFailed
from global_rag.scripts import config
from global_rag.scripts.project_paths import contained_path, safe_relative_path, upload_prefix

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".csv", ".txt", ".md",
    ".msg", ".json", ".xml", ".html", ".svg", ".dxf", ".ifc",
    ".png", ".jpg", ".jpeg", ".webp",
}
MAX_FILES = 500
MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_EXPANDED_BYTES = 512 * 1024 * 1024
MAX_ZIP_RATIO = 100


class UploadTooLarge(ValueError):
    pass


def _stage_files(uploads, root, destination, target_folder):
    """Validate the entire batch before any remote writes; never use extractall."""
    if not uploads or len(uploads) > MAX_FILES:
        raise ValueError(f"Supply between 1 and {MAX_FILES} files, or one ZIP archive.")
    names = [safe_relative_path(upload.filename or "") for upload in uploads]
    is_zip = len(uploads) == 1 and Path(names[0]).suffix.lower() == ".zip"
    if not is_zip and any(Path(name).suffix.lower() == ".zip" for name in names):
        raise ValueError("Supply one ZIP archive or individual files, not a mixture.")
    staged = []
    seen = set()
    total_bytes = 0

    def copy_file(source, relative, declared_size=None):
        nonlocal total_bytes
        relative = safe_relative_path(relative)
        if Path(relative).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {Path(relative).suffix}")
        if destination == "corpus" and not target_folder and "/" not in relative:
            raise ValueError("Corpus uploads require target_folder or a folder preserved in each uploaded path.")
        key = relative.casefold()
        if key in seen or any(key.startswith(other + "/") or other.startswith(key + "/") for other in seen):
            raise ValueError("Duplicate, case-ambiguous or conflicting file/folder paths within the upload.")
        seen.add(key)
        if len(staged) >= MAX_FILES:
            raise UploadTooLarge("Too many files in the archive.")
        path = contained_path(root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        with path.open("xb") as output:
            while block := source.read(1024 * 1024):
                size += len(block)
                total_bytes += len(block)
                if size > MAX_FILE_BYTES or total_bytes > MAX_EXPANDED_BYTES:
                    raise UploadTooLarge("Upload exceeds the file or total expanded size limit.")
                output.write(block)
        if not size or (declared_size is not None and size != declared_size):
            raise ValueError("Empty file or invalid archive entry length.")
        staged.append((relative, path))

    if is_zip:
        source = uploads[0].file
        source.seek(0, 2)
        if source.tell() > MAX_ARCHIVE_BYTES:
            raise UploadTooLarge("ZIP archive exceeds 256 MiB.")
        source.seek(0)
        try:
            with zipfile.ZipFile(source) as archive:
                entries = archive.infolist()
                if len(entries) > MAX_FILES * 3:
                    raise UploadTooLarge("Archive contains too many entries.")
                declared_total = 0
                for entry in entries:
                    safe_relative_path(entry.filename.rstrip("/"))
                    kind = stat.S_IFMT(entry.external_attr >> 16)
                    if kind not in {0, stat.S_IFREG, stat.S_IFDIR} or entry.flag_bits & 1:
                        raise ValueError("Links, special files and encrypted ZIP entries are not supported.")
                    if entry.is_dir():
                        continue
                    declared_total += entry.file_size
                    if (entry.file_size > MAX_FILE_BYTES or declared_total > MAX_EXPANDED_BYTES
                            or entry.file_size / max(1, entry.compress_size) > MAX_ZIP_RATIO):
                        raise UploadTooLarge("Archive exceeds decompression limits.")
                for entry in entries:
                    if not entry.is_dir():
                        with archive.open(entry) as content:
                            copy_file(content, entry.filename, entry.file_size)
        except (zipfile.BadZipFile, zipfile.LargeZipFile, NotImplementedError, RuntimeError) as exc:
            raise ValueError("Invalid, encrypted or unsupported ZIP archive.") from exc
    else:
        for upload, name in zip(uploads, names):
            upload.file.seek(0)
            copy_file(upload.file, name)
    if not staged:
        raise ValueError("Archive contains no supported documents.")
    return staged


def upload_files(destination, files, project_id=None, target_folder=None, overwrite=False):
    prefix = upload_prefix(destination, project_id, target_folder)
    config.load_project_dotenv()
    bucket_name = config.required_env("FIREBASE_STORAGE_BUCKET")
    if "/" in bucket_name:
        raise ValueError("FIREBASE_STORAGE_BUCKET must be a bucket name, without a path.")
    with tempfile.TemporaryDirectory(prefix="alchemy-upload-") as temporary:
        staged = _stage_files(files, Path(temporary), destination, target_folder)
        for relative, _ in staged:
            safe_relative_path(prefix + relative)
        client = config.get_storage_client()
        bucket = client.bucket(bucket_name)
        uploaded = 0
        errors = []
        for relative, path in staged:
            blob = bucket.blob(prefix + relative)
            try:
                generation = 0  # Create only, including when another writer races us.
                if overwrite:
                    try:
                        blob.reload()
                        generation = int(blob.generation)
                    except NotFound:
                        pass
                blob.upload_from_filename(
                    str(path), content_type=mimetypes.guess_type(relative)[0] or "application/octet-stream",
                    if_generation_match=generation,
                )
                uploaded += 1
            except PreconditionFailed:
                errors.append({"file": relative, "code": "object_conflict",
                               "error": "Object exists or changed during upload; replacement requires overwrite=true."})
            except Exception:
                logger.exception("Firebase upload failed for %s", relative)
                errors.append({"file": relative, "code": "upload_failed",
                               "error": "Firebase upload failed; the owner should check storage permissions and server logs."})
    return {"status": "success" if not errors else "partial" if uploaded else "failed",
            "destination": destination, "firebase_path": f"gs://{bucket_name}/{prefix}",
            "files_uploaded": uploaded, "files_failed": len(errors), "errors": errors}
