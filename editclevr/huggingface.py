"""Hugging Face Hub helpers for EditCLEVR dataset distribution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .paths import DATASET_PATH_KEYS, cache_dir, rebase_splits_json_paths, repo_root

DEFAULT_HF_REPO = "torux/EditCLEVR"
DEFAULT_HF_REVISION = "main"


def hf_repo_id() -> str:
    return os.environ.get("EDITCLEVR_HF_REPO", DEFAULT_HF_REPO).strip() or DEFAULT_HF_REPO


def hf_revision() -> str:
    return os.environ.get("EDITCLEVR_HF_REVISION", DEFAULT_HF_REVISION).strip() or DEFAULT_HF_REVISION


def relativize_splits_json_paths(
    splits_json: str | os.PathLike[str],
    dataset_dir: str | os.PathLike[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return splits metadata with paths relative to the dataset root."""

    splits_path = Path(splits_json).expanduser().resolve()
    root = Path(dataset_dir).expanduser().resolve() if dataset_dir else splits_path.parent
    with splits_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected splits.json to decode to a dict, got {type(data).__name__}.")

    for rows in data.values():
        for row in rows:
            for key in DATASET_PATH_KEYS:
                if key not in row:
                    continue
                value = row[key]
                if not isinstance(value, str) or not value:
                    continue
                path = Path(value).expanduser()
                if path.is_absolute():
                    try:
                        row[key] = str(path.resolve().relative_to(root))
                    except ValueError as exc:
                        raise ValueError(
                            f"Path for {key!r} is outside dataset root {root}: {value}"
                        ) from exc
                else:
                    row[key] = str(path)
    return data


def download_dataset(
    output_dir: str | os.PathLike[str],
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Download the EditCLEVR tarballs from Hugging Face, extract, and rebase paths."""

    try:
        import tarfile

        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Install huggingface_hub to download from Hugging Face: "
            "pip install huggingface_hub"
        ) from exc

    resolved_repo = repo_id or hf_repo_id()
    resolved_revision = revision or hf_revision()

    target = Path(output_dir).expanduser()
    if not target.is_absolute():
        target = repo_root() / target
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)

    for asset in TARBALL_ASSETS:
        local = hf_hub_download(
            repo_id=resolved_repo,
            repo_type="dataset",
            revision=resolved_revision,
            filename=asset,
            token=token,
        )
        with tarfile.open(local, "r:gz") as tar:
            tar.extractall(target)

    splits = target / "splits.json"
    if not splits.exists():
        raise FileNotFoundError(f"Downloaded dataset is missing splits.json at {splits}")
    rebase_splits_json_paths(splits)
    return target


# Suite directories packaged as individual tarballs. Each archive extracts at
# the dataset root, so a single `tar xzf` rebuilds the original layout.
SUITE_DIRS = ("atomic_id", "no_edit", "hard_distractor", "cogent_ood")
SPLITS_BUNDLE = ("splits.json", "validation_report.json", "phase1_manifest.json")
TARBALL_ASSETS = (
    "editclevr_splits.tar.gz",
    "editclevr_atomic_id.tar.gz",
    "editclevr_no_edit.tar.gz",
    "editclevr_hard_distractor.tar.gz",
    "editclevr_cogent_ood.tar.gz",
)


def build_dataset_tarballs(
    dataset_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> list[Path]:
    """Package a dataset directory into per-suite tarballs for distribution.

    `splits.json` is relativized so the archives are portable across machines.
    Returns the list of created tarball paths.
    """

    import tarfile
    import tempfile

    source = Path(dataset_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    splits = source / "splits.json"
    if not splits.exists():
        raise FileNotFoundError(f"Dataset directory must contain splits.json: {source}")

    relativized = relativize_splits_json_paths(splits, source)
    created: list[Path] = []

    with tempfile.TemporaryDirectory() as staging_str:
        staging = Path(staging_str)
        (staging / "splits.json").write_text(json.dumps(relativized, indent=2))
        for name in SPLITS_BUNDLE[1:]:
            src = source / name
            if src.exists():
                (staging / name).write_bytes(src.read_bytes())

        splits_tar = out / "editclevr_splits.tar.gz"
        with tarfile.open(splits_tar, "w:gz") as tar:
            for name in SPLITS_BUNDLE:
                member = staging / name
                if member.exists():
                    tar.add(member, arcname=name)
        created.append(splits_tar)

    for suite in SUITE_DIRS:
        suite_dir = source / suite
        if not suite_dir.exists():
            continue
        suite_tar = out / f"editclevr_{suite}.tar.gz"
        with tarfile.open(suite_tar, "w:gz") as tar:
            tar.add(suite_dir, arcname=suite)
        created.append(suite_tar)

    return created


def upload_dataset(
    dataset_dir: str | os.PathLike[str],
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    private: bool = False,
    token: str | None = None,
    dataset_card: str | os.PathLike[str] | None = None,
    tarball_dir: str | os.PathLike[str] | None = None,
    dry_run: bool = False,
) -> None:
    """Upload a local EditCLEVR dataset to Hugging Face as per-suite tarballs.

    Distributing a handful of tarballs (one LFS object each) avoids the
    rate-limiting and per-file overhead of pushing ~200k loose files.
    """

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "Install huggingface_hub to upload to Hugging Face: "
            "pip install huggingface_hub"
        ) from exc

    source = Path(dataset_dir).expanduser().resolve()
    splits = source / "splits.json"
    if not splits.exists():
        raise FileNotFoundError(f"Dataset directory must contain splits.json: {source}")

    resolved_repo = repo_id or hf_repo_id()
    resolved_revision = revision or hf_revision()
    tar_out = (
        Path(tarball_dir).expanduser().resolve()
        if tarball_dir
        else cache_dir() / "hf_tarballs"
    )

    if dry_run:
        relativized = relativize_splits_json_paths(splits, source)
        pair_count = sum(len(rows) for rows in relativized.values())
        print(f"[dry-run] repo={resolved_repo} revision={resolved_revision}")
        print(f"[dry-run] source={source}")
        print(f"[dry-run] pairs={pair_count} splits={list(relativized)}")
        print(f"[dry-run] would upload tarballs: {list(TARBALL_ASSETS)}")
        return

    existing = {p.name: p for p in tar_out.glob("*.tar.gz")} if tar_out.exists() else {}
    if all(name in existing for name in TARBALL_ASSETS):
        tarballs = [existing[name] for name in TARBALL_ASSETS]
        print(f"Reusing prebuilt tarballs in {tar_out}")
    else:
        print(f"Building tarballs in {tar_out} ...")
        tarballs = build_dataset_tarballs(source, tar_out)

    api = HfApi(token=token)
    api.create_repo(
        repo_id=resolved_repo,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    card_path = Path(dataset_card).expanduser().resolve() if dataset_card else None
    if card_path is None:
        default_card = repo_root() / "artifacts" / "huggingface_dataset" / "README.md"
        if default_card.exists():
            card_path = default_card

    if card_path is not None:
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=resolved_repo,
            repo_type="dataset",
            revision=resolved_revision,
            commit_message="Add EditCLEVR dataset card",
        )

    for tarball in tarballs:
        size_mb = tarball.stat().st_size / 1024 / 1024
        print(f"Uploading {tarball.name} ({size_mb:.0f} MB) ...")
        api.upload_file(
            path_or_fileobj=str(tarball),
            path_in_repo=tarball.name,
            repo_id=resolved_repo,
            repo_type="dataset",
            revision=resolved_revision,
            commit_message=f"Upload {tarball.name}",
        )
