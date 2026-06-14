"""Download and set up the EditCLEVR Phase-1 dataset from Hugging Face.

Dataset setup runs entirely in Python:

    python -m editclevr.download                 # default dataset location
    editclevr-download --data-dir /tmp/editclevr # console entry point

Or from Python:

    from editclevr.download import setup_dataset
    setup_dataset()
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .huggingface import download_dataset, hf_repo_id, hf_revision
from .paths import dataset_dir


def _resolve_data_dir(data_dir: str | os.PathLike[str] | None) -> Path:
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()
    return dataset_dir().resolve()


def is_ready(data_dir: str | os.PathLike[str]) -> bool:
    """Return True if splits.json exists and its first image path resolves on disk."""
    splits = Path(data_dir) / "splits.json"
    if not splits.exists():
        return False
    try:
        first_row = next(iter(json.loads(splits.read_text()).values()))[0]
    except (StopIteration, IndexError, ValueError):
        return False
    before_image = first_row.get("before_image", "")
    return bool(before_image) and os.path.exists(before_image)


def setup_dataset(
    data_dir: str | os.PathLike[str] | None = None,
    *,
    force: bool = False,
    repo_id: str | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Download the dataset from Hugging Face (if needed) and rebase ``splits.json`` paths."""
    target = _resolve_data_dir(data_dir)

    if not force and is_ready(target):
        print(f"[setup] Dataset already set up at {target} — skipping download.")
        return target

    print(f"[setup] Downloading from Hugging Face into {target} ...")
    download_dataset(
        target,
        repo_id=repo_id or hf_repo_id(),
        revision=revision or hf_revision(),
        token=token,
    )
    print(f"[setup] Done. Dataset ready at {target}/")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download the EditCLEVR Phase-1 dataset from Hugging Face."
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Destination directory (default: standard dataset location).",
    )
    parser.add_argument("--repo", default=None, help="Hugging Face dataset repo id.")
    parser.add_argument("--revision", default=None, help="Hugging Face revision.")
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if already present."
    )
    args = parser.parse_args(argv)

    try:
        setup_dataset(
            args.data_dir,
            force=args.force,
            repo_id=args.repo,
            revision=args.revision,
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error
        print(f"[setup] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
