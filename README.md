# EditCLEVR

**[ICML 2026]** An official source code for paper *"EditCLEVR: A Paired-Scene Intervention Benchmark for Compositional Faithfulness of Object-Centric Representations"*.

EditCLEVR is a benchmark for evaluating **intervention faithfulness** in object-centric representations: when exactly one object-level factor changes between two images, does the model localize and represent that change correctly?

This repository includes a Blender-backed CLEVR generator, evaluation metrics, and a reference oracle encoder.

## Quick start

```bash
git clone https://github.com/torux-bughunter/EditCLEVR.git
cd EditCLEVR

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[hub]"

python -m editclevr.download   # downloads the Phase-1 dataset from Hugging Face
```

The dataset (~20k paired edits, ~5.4 GB) is hosted on Hugging Face at
[`torux/EditCLEVR`](https://huggingface.co/datasets/torux/EditCLEVR);
`python -m editclevr.download` downloads, extracts, and rebases paths for your machine
(equivalently, the `editclevr-download` console command).

## Evaluate on the benchmark

The dataset ships as paired before/after scenes with instance masks and per-object
attributes (`splits.json`). Run the evaluation pipeline (CPU-only) over a dataset directory:

```bash
python3 -m editclevr.evaluation.run_evaluation
```

This trains attribute probes on the train split and reports the headline metrics with
bootstrap confidence intervals, writing `results.json` and per-split/suite CSVs to the
output directory.

### Headline metrics

| Metric | Suite | Better | Definition |
|--------|-------|:------:|------------|
| **SGIA** | edited | ↑ | Semantic-Graph Intervention Accuracy: decoded after-scene graph is exactly correct **and** only the edited object-factor changed |
| **ΔSGIA** | edited | ↑ | Edited factor decoded correctly after the edit **and** change is localized to that object-factor |
| **CLS** | edited | ↑ | Change-Locality Score: fraction of total embedding change attributable to the edited object |
| **EOA** | edited | ↑ | Edited-Object Accuracy: the most-changed object is the edited one |
| **NED** | no-edit | ↓ | No-Edit Drift: mean embedding change on control pairs where nothing changed |

Supporting metrics (`TFA`, `NFP`, `UOP`, `UOP_rate`, `SceneGraphExact`) are also reported per split, suite, factor, and object count. See [`artifacts/benchmark_card.md`](artifacts/benchmark_card.md) for protocol details.

### Scoring your own model

Encode each object under the provided instance masks, then use the standardized
`Evaluator` so results are directly comparable across methods:

```python
from editclevr.evaluation import Evaluator

evaluator = Evaluator(n_bootstrap=10000)
# print(evaluator.expected_input_format)   # documents the per-pair prediction schema
results = evaluator.eval(predictions)       # list of per-pair dicts
print(results["overall"]["SGIA"])           # {"mean": ..., "ci_lower": ..., "ci_upper": ...}
```

> Reference numbers: run the command above on the full Phase-1 dataset to reproduce the
> oracle-encoder headline scores reported in the paper.

## Generate data (optional)

The CLEVR Blender generation scripts and assets are vendored in
`editclevr/generator/clevr_blender/`, so the only external requirement is
[Blender 2.92](https://www.blender.org/) on your PATH (or `EDITCLEVR_BLENDER`):

```bash
python3 -m editclevr.generator.render_real --num-images 1
python3 -m editclevr.generator.build_real_phase1_dataset --atomic-pairs 4 --noop-pairs 2 --hard-pairs 2 --cogent-pairs 2
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `EDITCLEVR_REPO_DIR` | Repository root (auto-detected if unset) |
| `EDITCLEVR_DATASET_DIR` | Dataset directory containing `splits.json` |
| `EDITCLEVR_BLENDER` | Blender executable for rendering |
| `EDITCLEVR_HF_REPO` | Hugging Face dataset repo (default: `torux/EditCLEVR`) |
| `EDITCLEVR_HF_REVISION` | Hugging Face dataset revision (default: `main`) |

## Layout

```text
editclevr/          Python package (generator, evaluation, reference encoder, paths)
  download.py                Dataset download CLI (python -m editclevr.download)
  generator/clevr_blender/   Vendored CLEVR Blender scripts + assets (for rendering)
scripts/            Maintainer helper (HF dataset upload)
artifacts/          Benchmark card, datasheet, Croissant metadata
```

## Dependencies

Install the package and its core dependencies with `pip install -e .` (`numpy`, `scipy`,
`scikit-learn`, `Pillow`) — enough to run `run_evaluation`, use the `Evaluator`, and the
metric utilities.

**Optional extras:**

```bash
pip install -e ".[hub]"    # huggingface_hub for dataset download/upload
pip install -e ".[torch]"  # torch for GT-mask pooling (editclevr.evaluation.oracle_pooling)
pip install -e ".[all]"    # both
```

The legacy `requirements.txt` / `requirements-optional.txt` files are kept for `pip install -r` workflows.

**Generation**: only Blender 2.92. The CLEVR generation code (adapted from [`facebookresearch/clevr-dataset-gen`](https://github.com/facebookresearch/clevr-dataset-gen), BSD-licensed — see `editclevr/generator/clevr_blender/LICENSE-CLEVR`) is vendored in-repo, so no extra clone is needed.

## Dataset

The Phase-1 dataset (~20k paired edits, ~5.4 GB) is hosted on Hugging Face at
[`torux/EditCLEVR`](https://huggingface.co/datasets/torux/EditCLEVR). Each pair includes
before/after RGB images, instance masks (`.npz`), scene JSON, object attributes, edit
metadata, and difficulty tags, organized across the atomic, no-edit, hard-distractor, and
CoGenT-OOD evaluation suites plus `splits.json`.

| Split | Pairs | Purpose |
|-------|------:|---------|
| `train` | 10,000 | Probe training |
| `val` | 1,000 | Validation |
| `test_id` | 3,000 | In-distribution atomic edits |
| `test_noop` | 2,000 | No-edit control pairs |
| `test_hard` | 2,000 | Hard-distractor edits |
| `test_cogent` | 2,000 | CoGenT-OOD attribute combinations |

Download (extracts and rebases `splits.json` paths automatically):

```bash
pip install -e ".[hub]"
python -m editclevr.download
```

Or from Python:

```python
from editclevr.download import setup_dataset
setup_dataset()
```

Maintainers can (re)upload a local copy with:

```bash
huggingface-cli login
python3 scripts/upload_huggingface_dataset.py
```

**License:** code is [MIT](LICENSE). Dataset metadata and images are released under
**CC-BY-4.0** (see `artifacts/datasheet.md`).

## Citation

See [`CITATION.cff`](CITATION.cff). Update with final ICML workshop paper metadata before submission.

## Known limitations

- CLEVR generator targets Blender 2.92; newer Blender versions may differ.
- Reported metrics depend on probe calibration and the native/oracle evaluation protocol; see [`artifacts/benchmark_card.md`](artifacts/benchmark_card.md).
