# EditCLEVR

**[ICML 2026]** Official source code for *"EditCLEVR: A Paired-Scene Intervention Benchmark for Compositional Faithfulness of Object-Centric Representations"* (ICML workshop on Combining Theory and Benchmarks; paper URL work in progress).

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
> bundled oracle reference baseline. For paper baseline settings, see
> [Reproducing paper baselines](#reproducing-paper-baselines).

## Reproducing paper baselines

This repository provides the benchmark, metrics, and evaluation protocol. Baseline training
code is not included; the settings below match those reported in the paper.

**Slot Attention (`sa_native`)** — trained from scratch on EditCLEVR train before-images.
Architecture follows
[`google-research/slot_attention`](https://github.com/google-research/google-research/tree/master/slot_attention):
CNN encoder → Slot Attention → spatial decoder (8×8 broadcast grid, four stride-2 transposed
convs to 128×128). `num_slots=7`, `slot_dim=64`, 3 iterations, 128×128 input, 100k steps,
batch 64, Adam `lr=4e-4`, 10k-step warmup + cosine decay.

**DINOSAUR (`dinosaur_native`)** — MOVi-C recipe from
[`amazon-science/object-centric-learning-framework`](https://github.com/amazon-science/object-centric-learning-framework)
(`movi_c_feat_rec.yaml`). Frozen **DINO ViT-S/8** (`dino_vits8` via
`torch.hub.load('facebookresearch/dino:main', 'dino_vits8')`) at 224×224 → 784 patch tokens
(384-dim, CLS stripped), with train features pre-computed once and cached. Slot head:
`num_slots=7`, `slot_dim=128`, 3 iterations; MLP decoder `[1024, 1024, 1024]` with slot
softmax alpha. 150k steps, batch 64, Adam `lr=4e-4`, 10k warmup, exponential decay,
z-scored cached features.

**DINO oracle (`dino_oracle_s8`)** — same `dino_vits8` @ 224 backbone as DINOSAUR, with
ground-truth mask pooling instead of learned slots (matched encoder and resolution; mask
source is the only difference).

**Additional frozen oracles:** DINOv2 ViT-B/14 @ 224 (`dinov2_oracle`), SigLIP2 ViT-B/16 @
384 (`siglip2_oracle`). SAM2 hybrids used SAM2 ViT-Hiera-tiny automatic masks with the same
backbones at native resolution.

**Native matching (headline protocol).** For learned-slot and SAM 2 rows, each
ground-truth object is assigned the single predicted slot or mask proposal with highest
**best-overlap (MatchBO)** in that frame (strict one-to-one assignment; unused slots and
extra proposals are ignored). Semantic metrics are **conditional** on the edited object
having MatchBO ≥ 0.5 in **both** before and after frames. Linear logistic-regression probes
(one per factor) are fit on train object vectors; all object vectors are L2-normalized.
An IoU-weighted soft mixture over slot/proposal features is reported in the paper as an
ablation (Appendix D), not the headline setting. `editclevr/evaluation/slot_matching.py`
implements Hungarian IoU matching as a utility; the paper tables use strict MatchBO assignment.

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

If you use EditCLEVR, please cite:

> Anuraag Gadehothur Karnam and Tarunesh Sathish. *EditCLEVR: A Paired-Scene Intervention Benchmark for Compositional Faithfulness of Object-Centric Representations.* ICML 2026 Workshop on Combining Theory and Benchmarks: Towards A Virtuous Cycle to Understand and Guarantee Foundation Model Performance, 2026. Paper URL work in progress.

Full machine-readable metadata: [`CITATION.cff`](CITATION.cff).

## Known limitations

- CLEVR generator targets Blender 2.92; newer Blender versions may differ.
- Reported metrics depend on probe calibration and the native/oracle evaluation protocol; see [`artifacts/benchmark_card.md`](artifacts/benchmark_card.md).
