# EditCLEVR Datasheet

## Motivation

Existing object-centric benchmarks emphasize discovery. EditCLEVR targets intervention faithfulness: when one factor of one object changes, does the representation change correctly and only there?

## Dataset Creation

- **Production path:** Blender-backed paired renders via `editclevr.generator.build_real_phase1_dataset`.

## Instances

Each pair contains:

- `before_image` / `after_image`
- per-object attributes and instance masks
- edited object id, factor, and transition
- difficulty metadata (occlusion, target area, object count)

## Distribution

- **Canonical:** Hugging Face dataset `torux/EditCLEVR`, downloaded by `python -m editclevr.download` (override with `EDITCLEVR_DATASET_DIR`).

## License

- **Code:** MIT (see repository root `LICENSE`).
- **Dataset:** CC-BY-4.0 (intended; see release notes).

## Maintenance

ICML 2026 workshop release (Combining Theory and Benchmarks).
