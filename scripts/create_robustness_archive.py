"""Create the allowlisted input archive for the DeBERTa robustness notebook."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_PATH = ROOT / "promptshield-deberta-robustness-input.zip"
EXPLICIT_FILES = (
    Path("src/promptshield/baseline.py"),
    Path("src/promptshield/inference.py"),
    Path("src/promptshield/robustness.py"),
    Path("data/raw/validation.parquet"),
    Path("models/baseline.joblib"),
    Path("models/linear-svm.joblib"),
)
MODEL_DIRECTORY = Path("models/deberta-v3-small")


def required_members() -> list[Path]:
    members = list(EXPLICIT_FILES)
    checkpoint_files = sorted(
        path.relative_to(ROOT)
        for path in (ROOT / MODEL_DIRECTORY).rglob("*")
        if path.is_file()
    )
    return members + checkpoint_files


def validate_required_paths(members: list[Path]) -> None:
    missing = [member.as_posix() for member in members if not (ROOT / member).is_file()]
    if missing:
        raise FileNotFoundError(
            "required archive paths are missing: " + ", ".join(missing)
        )


def is_unsafe_member(name: str) -> bool:
    normalized = PurePosixPath(name)
    parts = {part.casefold() for part in normalized.parts}
    lowered = name.casefold()
    return (
        "test.parquet" in lowered
        or "test_labels" in lowered
        or "test-split" in lowered
        or "test_split" in lowered
        or "test" in parts
    )


def create_archive(members: list[Path]) -> tuple[int, int]:
    if ARCHIVE_PATH.exists():
        raise FileExistsError(
            f"refusing to overwrite existing archive: {ARCHIVE_PATH}"
        )
    with zipfile.ZipFile(
        ARCHIVE_PATH,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for member in members:
            archive.write(ROOT / member, arcname=member.as_posix())

    with zipfile.ZipFile(ARCHIVE_PATH, mode="r") as archive:
        names = archive.namelist()
        unsafe = [name for name in names if is_unsafe_member(name)]
        expected = {member.as_posix() for member in members}
        actual = set(names)
        if unsafe:
            ARCHIVE_PATH.unlink(missing_ok=True)
            raise RuntimeError(
                "archive safety check failed for: " + ", ".join(unsafe)
            )
        if actual != expected:
            ARCHIVE_PATH.unlink(missing_ok=True)
            raise RuntimeError("archive member allowlist check failed")
        return len(names), ARCHIVE_PATH.stat().st_size


def main() -> int:
    members = required_members()
    validate_required_paths(members)
    file_count, archive_size = create_archive(members)
    print(f"archive: {ARCHIVE_PATH}")
    print(f"size_bytes: {archive_size}")
    print(f"files: {file_count}")
    print("safety_check: passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"archive creation failed: {error}", file=sys.stderr)
        raise
