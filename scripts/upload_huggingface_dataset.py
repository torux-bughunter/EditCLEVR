#!/usr/bin/env python3
"""Upload the EditCLEVR Phase-1 dataset to Hugging Face."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from editclevr.huggingface import DEFAULT_HF_REPO, upload_dataset
from editclevr.paths import dataset_dir


def _summarize_dataset(dataset_path: Path) -> dict[str, int]:
    splits = json.loads((dataset_path / "splits.json").read_text(encoding="utf-8"))
    return {split: len(rows) for split, rows in splits.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=dataset_dir(),
        help="Local dataset directory containing splits.json",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_HF_REPO,
        help="Hugging Face dataset repo (owner/name)",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Target branch or tag on the Hugging Face dataset repo",
    )
    parser.add_argument(
        "--dataset-card",
        type=Path,
        default=None,
        help="Optional dataset card README to upload as README.md",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update the dataset repo as private",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print upload plan without contacting Hugging Face",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_dir.expanduser().resolve()
    if not (dataset_path / "splits.json").exists():
        raise SystemExit(
            f"Missing splits.json under {dataset_path}. "
            "Download the dataset first with python -m editclevr.download."
        )

    split_counts = _summarize_dataset(dataset_path)
    total_pairs = sum(split_counts.values())
    print(f"Dataset: {dataset_path}")
    print(f"Pairs: {total_pairs} across {len(split_counts)} splits")
    for split, count in sorted(split_counts.items()):
        print(f"  {split}: {count}")

    upload_dataset(
        dataset_path,
        repo_id=args.repo_id,
        revision=args.revision,
        private=args.private,
        dataset_card=args.dataset_card,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("Dry run complete.")
    else:
        print(f"Upload complete: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
