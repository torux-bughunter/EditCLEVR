from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from editclevr.paths import resolve_blender_executable

from .blender_adapter import BlenderRenderer
from .build_dataset import PairRecord, assemble_split_metadata, iterative_stratified_split
from .real_metadata import pair_metadata_to_record, write_validation_report
from .schema import FACTORS


def build_real_atomic_dataset(
    output_dir: Path,
    total_pairs: int,
    width: int = 320,
    height: int = 240,
    samples: int = 4,
    blender_executable: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    blender_executable = resolve_blender_executable(blender_executable)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    renderer = BlenderRenderer(blender_executable=blender_executable)

    records: list[PairRecord] = []
    pair_summaries: list[dict[str, Any]] = []
    for pair_idx in range(total_pairs):
        factor = FACTORS[pair_idx % len(FACTORS)]
        pair_dir = output_dir / f"pair_{pair_idx:04d}_{factor}"
        pair_metadata = renderer.generate_atomic_pair(
            output_dir=pair_dir,
            object_index=None,
            factor=factor,
            new_value="auto",
            width=width,
            height=height,
            render_num_samples=samples,
            min_objects=3,
            max_objects=3,
            min_pixels_per_object=50,
            start_idx=pair_idx,
            scene_seed=100_000 + pair_idx,
            selection_seed=110_000 + pair_idx,
        )
        records.append(pair_metadata_to_record(pair_metadata, blender_executable=blender_executable))
        pair_summaries.append(
            {
                "pair_id": pair_metadata["pair_id"],
                "factor": factor,
                "dir": str(pair_dir),
            }
        )

    train = max(1, int(total_pairs * 0.5))
    val = max(1, int(total_pairs * 0.25))
    test_id = total_pairs - train - val
    if test_id < 1:
        test_id = 1
        if train > val:
            train -= 1
        else:
            val -= 1

    assignments = iterative_stratified_split(
        records,
        split_targets={"train": train, "val": val, "test_id": test_id},
    )
    split_metadata = assemble_split_metadata(assignments)
    (output_dir / "splits.json").write_text(json.dumps(split_metadata, indent=2))
    validation = write_validation_report(output_dir / "validation_report.json", split_metadata)
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "total_pairs": total_pairs,
                "pairs": pair_summaries,
                "splits": {name: len(rows) for name, rows in split_metadata.items()},
                "validation": validation,
            },
            indent=2,
        )
    )
    return split_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small real Blender-backed Atomic-ID dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/real_atomic_dataset"))
    parser.add_argument("--total-pairs", type=int, default=4)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument(
        "--blender-executable",
        type=str,
        default=None,
        help="Blender executable (default: EDITCLEVR_BLENDER, PATH, or platform fallback).",
    )
    args = parser.parse_args()
    build_real_atomic_dataset(
        output_dir=args.output_dir,
        total_pairs=args.total_pairs,
        width=args.width,
        height=args.height,
        samples=args.samples,
        blender_executable=resolve_blender_executable(args.blender_executable),
    )


if __name__ == "__main__":
    main()
