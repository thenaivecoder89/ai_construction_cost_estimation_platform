"""Small, dependency-free path helpers for the two allowed Firebase source trees."""
import re
import unicodedata
from pathlib import Path

PROJECTS_PREFIX = "projects"
CORPUS_PREFIX = "corpus/ai_construction_cost_estimation_platform/BOK/BOK"
CORPUS_LOCAL_PATH = Path("BOK") / "BOK"


def validate_project_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("project_id must be 1–80 letters, digits, underscores or hyphens, starting with a letter or digit.")
    return value


def safe_relative_path(value):
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 900:
        raise ValueError("A nonempty relative path of at most 900 UTF-8 bytes is required.")
    if any(unicodedata.category(c).startswith("C") for c in value) or any(c in value for c in '\\:%'):
        raise ValueError("Unsafe characters in relative path.")
    for part in value.split("/"):
        if not part or part in {".", ".."} or part != part.strip() or part.endswith("."):
            raise ValueError("Unsafe relative path component.")
        if len(part.encode("utf-8")) > 255:
            raise ValueError("Path component exceeds 255 UTF-8 bytes.")
    return value


def contained_path(root, relative):
    relative = safe_relative_path(relative)
    root = Path(root).resolve()
    path = root
    for part in relative.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError("Symlink source paths are not allowed.")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Source path leaves its processing directory.")
    return path


def upload_prefix(destination, project_id=None, target_folder=None):
    if destination == "projects":
        if target_folder:
            raise ValueError("target_folder is only used for corpus uploads.")
        return f"{PROJECTS_PREFIX}/{validate_project_id(project_id)}/"
    if destination == "corpus":
        if project_id:
            raise ValueError("project_id is not used for corpus uploads.")
        suffix = safe_relative_path(target_folder) + "/" if target_folder else ""
        return f"{CORPUS_PREFIX}/{suffix}"
    raise ValueError("destination must be projects or corpus.")


def takeoff_prefix(firebase_root, bucket_name, project_id):
    expected = f"gs://{bucket_name}/{PROJECTS_PREFIX}"
    if firebase_root.rstrip("/") != expected:
        raise ValueError("TAKEOFF_FIREBASE_ROOT must be the configured bucket's projects root, without a project suffix.")
    return f"{PROJECTS_PREFIX}/{validate_project_id(project_id)}/"
