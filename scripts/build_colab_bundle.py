"""Build the allowlisted, portable Colab bundle for the hard-negative run.

The bundle is the only thing transferred to Google Colab. It contains exactly
the paths listed in ``docs/hard-negative-colab-runbook.md`` and nothing else.

Windows ``Compress-Archive`` stores path separators as ``\\`` in the ZIP
directory. Linux tools (including Colab) treat ``\\`` as a literal filename
character, so the files land flat at the extraction root instead of in real
directories. This builder always writes POSIX ``/`` entry names and then
verifies the archive before atomically replacing the output file.

Safety checks performed here (and again by the notebook after extraction):

* exact approved-file allowlist;
* ``test.parquet`` is never present (the test split is never opened);
* no ``.git``, ``__pycache__``, ``.venv``, or ``.pyc`` entries;
* ZIP CRC test;
* all 12 pinned SHA-256 values match both the source files and the ZIP
  members (train, validation, both dev-eval sources, control script, and the
  seven frozen baseline checkpoint files).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = Path(tempfile.gettempdir()) / "promptshield-colab-bundle.zip"

SRC_DIRECTORY = Path("src/promptshield")
BASELINE_DIRECTORY = Path("models/deberta-v3-small")

EXPLICIT_FILES = (
    Path("data/raw/train.parquet"),
    Path("data/raw/validation.parquet"),
    Path("data/raw-eval/dolly/databricks-dolly-15k.jsonl"),
    Path("data/raw-eval/oasst1/train.parquet"),
    Path("scripts/fp_benign_diagnostic.py"),
)

BASELINE_FILES = (
    Path("models/deberta-v3-small/added_tokens.json"),
    Path("models/deberta-v3-small/config.json"),
    Path("models/deberta-v3-small/model.safetensors"),
    Path("models/deberta-v3-small/special_tokens_map.json"),
    Path("models/deberta-v3-small/spm.model"),
    Path("models/deberta-v3-small/tokenizer.json"),
    Path("models/deberta-v3-small/tokenizer_config.json"),
)

PINNED_SHA256 = {
    "data/raw/train.parquet": (
        "c0724f8e00eebb65c001be9ba94291d1608e5da0374012e53dfd4077c5f92558"
    ),
    "data/raw/validation.parquet": (
        "bbd417bfd0df1553dd769e814a0b62ca58ad5b1d08d153c90f66937a4af74deb"
    ),
    "data/raw-eval/dolly/databricks-dolly-15k.jsonl": (
        "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
    ),
    "data/raw-eval/oasst1/train.parquet": (
        "bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e"
    ),
    "scripts/fp_benign_diagnostic.py": (
        "e262d65ad966eac03777f77ff402d197f27a210ebd5f71400a8a03f8d600c244"
    ),
    "models/deberta-v3-small/added_tokens.json": (
        "dc046d04c9b0ada7ae6f1dc89c465801799acdf0c9a6aab8c15a1b2d5ca4e91f"
    ),
    "models/deberta-v3-small/config.json": (
        "44e8fe36a9ea5659b039545691a0896e21ab5629f37b2c7f4480354433902f0d"
    ),
    "models/deberta-v3-small/model.safetensors": (
        "2cefb8e40ade17940899ef523c7dea2a1bd1813ba0373edda84481a32c783614"
    ),
    "models/deberta-v3-small/special_tokens_map.json": (
        "9463f61e1b109a8eb4688b829260d7c6b1e6dff04c98ff7269bb89e2b92369b9"
    ),
    "models/deberta-v3-small/spm.model": (
        "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd"
    ),
    "models/deberta-v3-small/tokenizer.json": (
        "37fe3823ad5737f0033c832905c8d4089ed4008da79682ab3fd85c8ce8e5b040"
    ),
    "models/deberta-v3-small/tokenizer_config.json": (
        "e6c4c771911c211618a2d46488dc7e9e499f051773cd4ff4faeca6fe55fdf569"
    ),
}

NOTEBOOK_REQUIRED_FILES = {
    "src/promptshield/__init__.py",
    "src/promptshield/hard_negative_experiment.py",
    "src/promptshield/deberta_decision_audit.py",
    "src/promptshield/inference.py",
    "src/promptshield/benign_chat_eval.py",
    "src/promptshield/review_suggestions.py",
    "src/promptshield/error_review.py",
    "src/promptshield/gateway.py",
    "src/promptshield/baseline.py",
    "data/raw/train.parquet",
    "data/raw/validation.parquet",
    "data/raw-eval/dolly/databricks-dolly-15k.jsonl",
    "data/raw-eval/oasst1/train.parquet",
    "models/deberta-v3-small/model.safetensors",
    "scripts/fp_benign_diagnostic.py",
}

FORBIDDEN_PARTS = {".git", "__pycache__", ".venv", "venv", "node_modules"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo"}


def approved_members() -> list[Path]:
    """Return the exact approved file list (POSIX-style relative paths)."""
    src_files = sorted(
        path.relative_to(ROOT)
        for path in (ROOT / SRC_DIRECTORY).glob("*.py")
        if path.is_file()
    )
    members = src_files + list(EXPLICIT_FILES) + list(BASELINE_FILES)
    if len(set(members)) != len(members):
        raise RuntimeError("approved member list contains duplicates")
    return members


def is_safe_member_name(name: str) -> bool:
    """Reject absolute paths, traversal, drive prefixes, and Windows separators."""
    if not name or "\\" in name:
        return False
    path = PurePosixPath(name)
    parts = path.parts
    if path.is_absolute() or ".." in parts:
        return False
    if parts and ":" in parts[0]:
        return False
    if any(part.casefold() in FORBIDDEN_PARTS for part in parts):
        return False
    if path.suffix.casefold() in FORBIDDEN_SUFFIXES:
        return False
    return path.name.casefold() != "test.parquet"


def validate_source(members: list[Path]) -> None:
    missing = [
        member.as_posix()
        for member in members
        if not (ROOT / member).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "required source paths are missing: " + ", ".join(missing)
        )

    unsafe = [
        member.as_posix()
        for member in members
        if not is_safe_member_name(member.as_posix())
    ]
    if unsafe:
        raise RuntimeError("unsafe source member paths: " + ", ".join(unsafe))

    absent = NOTEBOOK_REQUIRED_FILES - {member.as_posix() for member in members}
    if absent:
        raise RuntimeError(
            "notebook-required paths missing from allowlist: "
            + ", ".join(sorted(absent))
        )

    for relative, expected in PINNED_SHA256.items():
        actual = sha256_path(ROOT / relative)
        if actual != expected:
            raise RuntimeError(
                f"source hash mismatch for {relative}: {actual} != {expected}"
            )


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_zip_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    with archive.open(info) as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, members: list[Path]) -> dict[str, object]:
    expected = {member.as_posix() for member in members}
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        raw_names = [info.orig_filename for info in infos]
        normalized_names = [info.filename for info in infos]

        if any("\\" in name for name in raw_names):
            raise RuntimeError("archive has non-portable backslash entry names")
        if raw_names != normalized_names:
            raise RuntimeError("archive entry-name normalization mismatch")
        if len(raw_names) != len(set(raw_names)):
            raise RuntimeError("archive contains duplicate entry names")
        if set(raw_names) != expected:
            extra = sorted(set(raw_names) - expected)
            missing = sorted(expected - set(raw_names))
            raise RuntimeError(
                "archive allowlist mismatch; "
                f"extra={extra}, missing={missing}"
            )

        unsafe = [name for name in raw_names if not is_safe_member_name(name)]
        if unsafe:
            raise RuntimeError("unsafe archive entries: " + ", ".join(unsafe))

        bad_crc = archive.testzip()
        if bad_crc is not None:
            raise RuntimeError(f"archive CRC check failed at {bad_crc}")

        for relative, expected_hash in PINNED_SHA256.items():
            info = archive.getinfo(relative)
            actual_hash = sha256_zip_member(archive, info)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"archive hash mismatch for {relative}: "
                    f"{actual_hash} != {expected_hash}"
                )

        return {
            "file_count": len(raw_names),
            "crc_verified": True,
            "allowlist_verified": True,
            "layout_verified": True,
            "pinned_hashes_verified": len(PINNED_SHA256),
            "test_parquet_absent": not any(
                PurePosixPath(name).name.casefold() == "test.parquet"
                for name in raw_names
            ),
            "git_metadata_absent": not any(
                ".git" in PurePosixPath(name).parts for name in raw_names
            ),
        }


def build(output: Path) -> dict[str, object]:
    members = approved_members()
    validate_source(members)

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(
            partial,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for member in members:
                archive.write(ROOT / member, arcname=member.as_posix())

        checks = verify_archive(partial, members)
        os.replace(partial, output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    result: dict[str, object] = {
        "archive": str(output),
        "size_bytes": output.stat().st_size,
        "sha256": sha256_path(output),
    }
    result.update(checks)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the verified portable PromptShield Colab bundle."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output ZIP path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args(argv)

    result = build(args.output.expanduser().resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"bundle build failed: {error}", file=sys.stderr)
        raise
