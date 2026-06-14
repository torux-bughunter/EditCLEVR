from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from editclevr.paths import resolve_blender_executable

from .blender_adapter import BlenderRenderer
from .build_dataset import PairRecord, assemble_split_metadata, iterative_stratified_split
from .real_metadata import pair_metadata_to_record, write_validation_report
from .schema import FACTORS
from .seed_utils import derive_seed

logger = logging.getLogger(__name__)

OBJECT_COUNTS = (3, 4, 5, 6)
ATOMIC_SCHEDULE_SEED = 10_001
NOOP_SCHEDULE_SEED = 20_001
HARD_SCHEDULE_SEED = 30_001
COGENT_SCHEDULE_SEED = 40_001


def metadata_to_record(pair_metadata: dict[str, Any], blender_executable: str | None = None) -> PairRecord:
    return pair_metadata_to_record(pair_metadata, blender_executable=blender_executable)


def make_factor_count_schedule(total: int, seed: int) -> list[tuple[str, int]]:
    if total < 0:
        raise ValueError("total must be non-negative.")
    grid = [(factor, object_count) for factor in FACTORS for object_count in OBJECT_COUNTS]
    schedule = [grid[index % len(grid)] for index in range(total)]
    random.Random(seed).shuffle(schedule)
    return schedule


def make_object_count_schedule(total: int, seed: int) -> list[int]:
    if total < 0:
        raise ValueError("total must be non-negative.")
    schedule = [OBJECT_COUNTS[index % len(OBJECT_COUNTS)] for index in range(total)]
    random.Random(seed).shuffle(schedule)
    return schedule


def _atomic_targets(total_pairs: int) -> dict[str, int]:
    if total_pairs < 3:
        raise ValueError("Atomic-ID suite needs at least 3 pairs to populate train, val, and test_id.")
    desired = {"train": 10, "val": 1, "test_id": 3}
    total_weight = sum(desired.values())
    train = max(1, int(round(total_pairs * desired["train"] / total_weight)))
    val = max(1, int(round(total_pairs * desired["val"] / total_weight)))
    test_id = total_pairs - train - val
    while test_id < 1:
        if train > 1:
            train -= 1
        elif val > 1:
            val -= 1
        test_id = total_pairs - train - val
    return {"train": train, "val": val, "test_id": test_id}


def _delete_blendfiles(directory: Path) -> int:
    """Remove all .blend and .blend1 files under *directory*. Returns bytes freed."""
    freed = 0
    for pattern in ("*.blend", "*.blend1"):
        for f in directory.rglob(pattern):
            freed += f.stat().st_size
            f.unlink()
    return freed


def _render_task(task: dict[str, Any]) -> dict[str, Any]:
    renderer = BlenderRenderer(blender_executable=task["blender_executable"])
    method = getattr(renderer, task["method"])
    output_dir = Path(task["output_dir"])
    kwargs = dict(task["kwargs"])
    pair_metadata = method(output_dir=output_dir, **kwargs)
    if task.get("delete_blendfiles", False):
        _delete_blendfiles(output_dir)
    return {
        "suite_split": task["suite_split"],
        "suite_name": task["suite_name"],
        "pair_dir": str(output_dir),
        "pair_metadata": pair_metadata,
    }


def _run_tasks(tasks: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    total = len(tasks)
    t0 = time.monotonic()
    results: list[dict[str, Any]] = []

    def run_serial() -> list[dict[str, Any]]:
        for i, task in enumerate(tasks, 1):
            result = _render_task(task)
            results.append(result)
            elapsed = time.monotonic() - t0
            per_task = elapsed / i
            eta = per_task * (total - i)
            logger.info(
                "[%d/%d] %s done  (%.1fs elapsed, ~%.0fs remaining)",
                i, total, result["suite_name"], elapsed, eta,
            )
        return results

    if workers <= 1:
        return run_serial()

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {executor.submit(_render_task, t): idx for idx, t in enumerate(tasks)}
            completed = 0
            ordered = [None] * total
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                ordered[idx] = future.result()
                completed += 1
                elapsed = time.monotonic() - t0
                per_task = elapsed / completed
                eta = per_task * (total - completed)
                logger.info(
                    "[%d/%d] pair done  (%.1fs elapsed, ~%.0fs / ~%.1fh remaining)",
                    completed, total, elapsed, eta, eta / 3600,
                )
            return [r for r in ordered if r is not None]
    except (OSError, PermissionError) as exc:
        logger.warning(
            "Falling back to sequential rendering because worker pool startup failed: %s",
            exc,
        )
        return run_serial()


def _atomic_task(
    output_dir: Path,
    index: int,
    factor: str,
    object_count: int,
    blender_executable: str,
    width: int,
    height: int,
    samples: int,
    delete_blendfiles: bool = False,
) -> dict[str, Any]:
    return {
        "suite_split": "atomic",
        "suite_name": "atomic_id",
        "output_dir": str(output_dir / f"atomic_id_{index:05d}_{factor}"),
        "blender_executable": blender_executable,
        "method": "generate_atomic_pair",
        "delete_blendfiles": delete_blendfiles,
        "kwargs": {
            "object_index": None,
            "factor": factor,
            "new_value": "auto",
            "width": width,
            "height": height,
            "render_num_samples": samples,
            "min_objects": object_count,
            "max_objects": object_count,
            "min_pixels_per_object": 50,
            "condition": "A",
            "start_idx": index,
            "scene_seed": 100_000 + index,
            "selection_seed": derive_seed(100_000 + index, "select_target"),
        },
    }


def _no_edit_task(
    output_dir: Path,
    index: int,
    object_count: int,
    blender_executable: str,
    width: int,
    height: int,
    samples: int,
    delete_blendfiles: bool = False,
) -> dict[str, Any]:
    return {
        "suite_split": "test_noop",
        "suite_name": "no_edit",
        "output_dir": str(output_dir / f"no_edit_{index:05d}"),
        "blender_executable": blender_executable,
        "method": "generate_no_edit_pair",
        "delete_blendfiles": delete_blendfiles,
        "kwargs": {
            "object_index": None,
            "width": width,
            "height": height,
            "render_num_samples": samples,
            "min_objects": object_count,
            "max_objects": object_count,
            "min_pixels_per_object": 50,
            "seed": 210_000 + index,
            "condition": "A",
            "start_idx": index,
            "scene_seed": 200_000 + index,
            "selection_seed": derive_seed(200_000 + index, "select_target"),
        },
    }


def _hard_task(
    output_dir: Path,
    index: int,
    factor: str,
    object_count: int,
    blender_executable: str,
    width: int,
    height: int,
    samples: int,
    delete_blendfiles: bool = False,
) -> dict[str, Any]:
    return {
        "suite_split": "test_hard",
        "suite_name": "hard_distractor",
        "output_dir": str(output_dir / f"hard_distractor_{index:05d}"),
        "blender_executable": blender_executable,
        "method": "generate_hard_distractor_pair",
        "delete_blendfiles": delete_blendfiles,
        "kwargs": {
            "factor": factor,
            "width": width,
            "height": height,
            "render_num_samples": samples,
            "min_objects": object_count,
            "max_objects": object_count,
            "min_pixels_per_object": 50,
            "max_attempts": 40,
            "condition": "A",
            "start_idx": index,
            "scene_seed": 300_000 + (index * 100),
            "selection_seed": derive_seed(300_000 + (index * 100), "select_target"),
        },
    }


def _cogent_task(
    output_dir: Path,
    index: int,
    factor: str,
    object_count: int,
    blender_executable: str,
    width: int,
    height: int,
    samples: int,
    delete_blendfiles: bool = False,
) -> dict[str, Any]:
    condition = "B"
    return {
        "suite_split": "test_cogent",
        "suite_name": "cogent_ood",
        "output_dir": str(output_dir / f"cogent_ood_{index:05d}_{condition.lower()}_{factor}"),
        "blender_executable": blender_executable,
        "method": "generate_cogent_pair",
        "delete_blendfiles": delete_blendfiles,
        "kwargs": {
            "object_index": None,
            "factor": factor,
            "new_value": "auto",
            "condition": condition,
            "width": width,
            "height": height,
            "render_num_samples": samples,
            "min_objects": object_count,
            "max_objects": object_count,
            "min_pixels_per_object": 50,
            "start_idx": index,
            "scene_seed": 400_000 + index,
            "selection_seed": derive_seed(400_000 + index, "select_target"),
        },
    }


def build_real_phase1_dataset(
    output_dir: Path,
    blender_executable: str | None = None,
    width: int = 320,
    height: int = 240,
    samples: int = 8,
    atomic_pairs: int = 12,
    noop_pairs: int = 4,
    hard_pairs: int = 4,
    cogent_pairs: int = 4,
    workers: int = 1,
    delete_blendfiles: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    blender_executable = resolve_blender_executable(blender_executable)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    total = atomic_pairs + noop_pairs + hard_pairs + cogent_pairs
    logger.info(
        "Building %d pairs (%d atomic, %d noop, %d hard, %d cogent) at %dx%d, %d samples, %d workers",
        total, atomic_pairs, noop_pairs, hard_pairs, cogent_pairs, width, height, samples, workers,
    )

    atomic_schedule = make_factor_count_schedule(atomic_pairs, seed=ATOMIC_SCHEDULE_SEED)
    noop_schedule = make_object_count_schedule(noop_pairs, seed=NOOP_SCHEDULE_SEED)
    hard_schedule = make_factor_count_schedule(hard_pairs, seed=HARD_SCHEDULE_SEED)
    cogent_schedule = make_factor_count_schedule(cogent_pairs, seed=COGENT_SCHEDULE_SEED)

    tasks: list[dict[str, Any]] = []
    tasks.extend(
        _atomic_task(
            output_dir / "atomic_id",
            index,
            factor,
            object_count,
            blender_executable,
            width,
            height,
            samples,
            delete_blendfiles,
        )
        for index, (factor, object_count) in enumerate(atomic_schedule)
    )
    tasks.extend(
        _no_edit_task(
            output_dir / "no_edit",
            index,
            object_count,
            blender_executable,
            width,
            height,
            samples,
            delete_blendfiles,
        )
        for index, object_count in enumerate(noop_schedule)
    )
    tasks.extend(
        _hard_task(
            output_dir / "hard_distractor",
            index,
            factor,
            object_count,
            blender_executable,
            width,
            height,
            samples,
            delete_blendfiles,
        )
        for index, (factor, object_count) in enumerate(hard_schedule)
    )
    tasks.extend(
        _cogent_task(
            output_dir / "cogent_ood",
            index,
            factor,
            object_count,
            blender_executable,
            width,
            height,
            samples,
            delete_blendfiles,
        )
        for index, (factor, object_count) in enumerate(cogent_schedule)
    )

    rendered = _run_tasks(tasks, workers=workers)

    atomic_records: list[PairRecord] = []
    atomic_pair_dirs: dict[str, str] = {}
    split_metadata: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test_id": [],
        "test_noop": [],
        "test_hard": [],
        "test_cogent": [],
    }
    pair_index: dict[str, list[dict[str, Any]]] = {key: [] for key in split_metadata}

    for item in rendered:
        record = metadata_to_record(item["pair_metadata"], blender_executable=blender_executable)
        pair_directory = str(Path(record.payload["before_image"]).resolve().parents[2])
        if item["suite_split"] == "atomic":
            atomic_records.append(record)
            atomic_pair_dirs[record.pair_id] = pair_directory
            continue
        split_metadata[item["suite_split"]].append(record.payload | {"split": item["suite_split"]})
        pair_index[item["suite_split"]].append(
            {
                "pair_id": record.pair_id,
                "suite": item["suite_name"],
                "directory": pair_directory,
            }
        )

    atomic_assignments = iterative_stratified_split(atomic_records, _atomic_targets(atomic_pairs))
    atomic_split_metadata = assemble_split_metadata(atomic_assignments)
    for split, rows in atomic_split_metadata.items():
        split_metadata[split].extend(rows)
        pair_index[split].extend(
            {
                "pair_id": row["pair_id"],
                "suite": "atomic_id",
                "directory": atomic_pair_dirs[row["pair_id"]],
            }
            for row in rows
        )

    validation = write_validation_report(output_dir / "validation_report.json", split_metadata)
    (output_dir / "splits.json").write_text(json.dumps(split_metadata, indent=2))
    (output_dir / "phase1_manifest.json").write_text(
        json.dumps(
            {
                "requested_pairs": {
                    "atomic_id": atomic_pairs,
                    "no_edit": noop_pairs,
                    "hard_distractor": hard_pairs,
                    "cogent_ood": cogent_pairs,
                },
                "splits": {split: len(rows) for split, rows in split_metadata.items()},
                "pairs": pair_index,
                "validation": validation,
            },
            indent=2,
        )
    )
    return split_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real Blender-backed Phase 1 dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/real_phase1_dataset"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--atomic-pairs", type=int, default=12)
    parser.add_argument("--noop-pairs", type=int, default=4)
    parser.add_argument("--hard-pairs", type=int, default=4)
    parser.add_argument("--cogent-pairs", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--blender-executable",
        type=str,
        default=None,
        help="Blender executable (default: EDITCLEVR_BLENDER, PATH, or platform fallback).",
    )
    parser.add_argument(
        "--delete-blendfiles",
        action="store_true",
        default=False,
        help="Delete .blend files after each pair to save disk space.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.output_dir / "build.log"),
        ],
    )

    build_real_phase1_dataset(
        output_dir=args.output_dir,
        blender_executable=resolve_blender_executable(args.blender_executable),
        width=args.width,
        height=args.height,
        samples=args.samples,
        atomic_pairs=args.atomic_pairs,
        noop_pairs=args.noop_pairs,
        hard_pairs=args.hard_pairs,
        cogent_pairs=args.cogent_pairs,
        workers=args.workers,
        delete_blendfiles=args.delete_blendfiles,
    )


if __name__ == "__main__":
    main()
