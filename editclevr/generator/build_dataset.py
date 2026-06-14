from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .difficulty_tagger import tag_difficulty
from .scene import ScenePair
from .schema import FACTORS, SPLITS, SUITES


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    suite: str
    edit_factor: str
    num_objects: int
    difficulty_bucket: str
    payload: dict[str, Any]


def validate_pair_record(record: PairRecord) -> None:
    if record.suite not in SUITES:
        raise ValueError(f"Unknown suite '{record.suite}'.")
    if record.suite == "no_edit":
        valid_factors = set(FACTORS) | {"none"}
    else:
        valid_factors = set(FACTORS)
    if record.edit_factor not in valid_factors:
        raise ValueError(f"Unknown edit factor '{record.edit_factor}'.")
    if record.num_objects < 1:
        raise ValueError("num_objects must be positive.")
    required_keys = {"pair_id", "suite", "edit_factor"}
    missing = required_keys.difference(record.payload)
    if missing:
        raise ValueError(f"Missing required payload fields: {sorted(missing)}")


def _stratum_key(record: PairRecord) -> tuple[str, int, str]:
    return (record.edit_factor, record.num_objects, record.difficulty_bucket)


def iterative_stratified_split(
    records: list[PairRecord],
    split_targets: dict[str, int],
) -> dict[str, list[PairRecord]]:
    for split in split_targets:
        if split not in SPLITS:
            raise ValueError(f"Unknown split '{split}'.")

    total_requested = sum(split_targets.values())
    if total_requested != len(records):
        raise ValueError(
            f"Split targets sum to {total_requested}, but received {len(records)} records."
        )

    for record in records:
        validate_pair_record(record)

    groups: dict[tuple[str, int, str], list[PairRecord]] = defaultdict(list)
    for record in sorted(records, key=lambda item: item.pair_id):
        groups[_stratum_key(record)].append(record)

    assignments: dict[str, list[PairRecord]] = {split: [] for split in split_targets}
    counts = Counter()

    for _, group in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        for record in group:
            available = [
                split
                for split, target in split_targets.items()
                if counts[split] < target
            ]
            if not available:
                raise RuntimeError("No split capacity remaining during assignment.")
            split = min(available, key=lambda name: (counts[name] / split_targets[name], counts[name], name))
            assignments[split].append(record)
            counts[split] += 1

    return assignments


def pair_to_record(pair: ScenePair, difficulty_bucket: str | None = None) -> PairRecord:
    pair.validate()
    difficulty = dict(pair.difficulty) if pair.difficulty else tag_difficulty(pair)
    bucket = difficulty_bucket or str(difficulty.get("occlusion_level", "unknown"))
    payload = pair.to_metadata()
    return PairRecord(
        pair_id=pair.pair_id,
        suite=pair.suite,
        edit_factor=pair.edit.factor,
        num_objects=len(pair.before.objects),
        difficulty_bucket=bucket,
        payload=payload,
    )


def assemble_split_metadata(
    assignments: dict[str, list[PairRecord]],
) -> dict[str, list[dict[str, Any]]]:
    metadata: dict[str, list[dict[str, Any]]] = {}
    for split, records in assignments.items():
        rows = []
        for record in records:
            validate_pair_record(record)
            row = dict(record.payload)
            row["split"] = split
            rows.append(row)
        metadata[split] = rows
    return metadata
