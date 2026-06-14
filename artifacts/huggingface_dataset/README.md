---
language: en
license: cc-by-4.0
task_categories:
  - image-to-image
tags:
  - object-centric-learning
  - intervention-faithfulness
  - CLEVR
  - synthetic
  - computer-vision
pretty_name: EditCLEVR Phase 1
size_categories:
  - 10K<n<100K
---

# EditCLEVR Phase 1

EditCLEVR is a synthetic benchmark for evaluating **intervention faithfulness** in object-centric representations. Each example is a paired before/after scene where exactly one object-level factor may change.

## Dataset summary

- **~20k paired edits** across six evaluation splits
- **Suites:** atomic single-factor edits, no-edit controls, hard distractors, and CoGenT-OOD combinations
- **Per-object factors:** `color`, `material`, `size`, `shape`
- **Artifacts per pair:** before/after RGB images, instance masks (`.npz`), scene JSON, object attributes, edit metadata, and difficulty tags

## Splits

| Split | Purpose |
|-------|---------|
| `train` | Probe training |
| `val` | Validation |
| `test_id` | In-distribution atomic edits |
| `test_noop` | No-edit control pairs |
| `test_hard` | Hard distractor edits |
| `test_cogent` | CoGenT-OOD edits |

## Download

The dataset ships as a small set of `.tar.gz` archives (one per suite plus a
`splits` bundle). The helpers below download and extract them into the original
directory layout automatically.

```bash
git clone https://github.com/torux-bughunter/EditCLEVR.git
cd EditCLEVR
pip install -e ".[hub]"
python -m editclevr.download
```

Or from Python:

```python
from editclevr.download import setup_dataset

setup_dataset()
```

## Evaluate

```bash
pip install -e .
python3 -m editclevr.evaluation.run_evaluation
```

## File layout

```text
splits.json
atomic_id/
no_edit/
hard_distractor/
cogent_ood/
validation_report.json
phase1_manifest.json
```

`splits.json` stores relative paths to images, masks, and scene JSON files so the dataset can be moved across machines.

## License

- **Dataset:** [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
- **Code:** [MIT](https://github.com/torux-bughunter/EditCLEVR/blob/main/LICENSE)

## Citation

If you use this dataset, please cite the EditCLEVR ICML 2026 paper when available. See [`CITATION.cff`](https://github.com/torux-bughunter/EditCLEVR/blob/main/CITATION.cff) in the code repository.
