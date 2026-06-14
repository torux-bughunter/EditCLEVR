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
> **simple oracle reference** scores. The competitive paper baselines use the protocol below
> (native/oracle encoders, probe training, and slot matching) and are **not shipped as
> runnable code** in this release — see [Reproducing paper baselines](#reproducing-paper-baselines).

## Reproducing paper baselines

This section documents how the paper baselines were implemented in the EditCLEVR research
pipeline. **This public repository ships the benchmark, metrics, and a lightweight oracle
reference run only.** It does not include baseline training notebooks, external framework
vendoring, or feature-extraction scripts. Use the details below to reimplement the same
protocol with your own codebase, then score outputs with `Evaluator` or the metric utilities
in `editclevr/evaluation/`.

### Pipeline overview

Paper numbers were produced with a four-stage Colab-oriented workflow:

| Stage | Purpose | Typical runtime |
|-------|---------|-----------------|
| **1. Train native models** | Fine-tune or train object-centric encoders on EditCLEVR `train` before-images | A100 GPU (hours per model) |
| **2. Extract features** | Run every baseline on all splits; save per-pair object features + masks to disk | A100 GPU (~1 hr for full extraction pass) |
| **3. Train probes & score** | Fit factor probes on `train`, decode attributes, compute headline metrics + bootstrap CIs | CPU (free tier OK) |
| **4. Ablations (optional)** | Strict-native objectization, MLP-vs-linear probes, SAM2/DINOSAUR hybrids, compositional oracles | Mix of GPU + CPU |

Conceptual notebook order from the development pipeline:

1. **Slot Attention** — fine-tune a CLEVR-pretrained object-discovery autoencoder on EditCLEVR train images.
2. **DINOSAUR** — train a slot-attention grouping head on **frozen DINO patch features** (cached once on disk for speed).
3. **SlotDiffusion / SPOT** — train vendored native slot baselines (image-only SlotDiffusion path + official SPOT teacher/student stack).
4. **Feature extraction** — run all native, oracle, and hybrid models over every split; write one `.npz` cache per model per split.
5. **Metrics & tables** — train probes, Hungarian-align native slots to GT, compute SGIA / ΔSGIA / CLS / EOA / NED (+ supporting metrics), bootstrap CIs, export JSON/CSVs.
6. **Optional ablations** — strict argmax slot assignment (vs soft mixture), MLP probe sensitivity check, DINOSAUR-mask + frozen-DINO pooling, extra frozen-backbone oracles (CLIP, MAE, I-JEPA, etc.).

### Dual evaluation protocol

Every baseline is evaluated under one of two protocols:

**Native protocol (end-to-end discovery)**  
The model receives a single RGB image and outputs `K` slot representations plus soft attention
masks. Predicted slots are aligned to ground-truth objects with **Hungarian matching on mask
IoU** (implemented in `editclevr/evaluation/slot_matching.py`). Metrics are computed on
matched object pairs across the before/after edit. This measures discovery + representation
together.

**Oracle protocol (discovery removed)**  
Ground-truth instance masks pool features from a **frozen pretrained spatial encoder**
(resize mask → multiply feature map → spatial average pool → L2 normalize per object).
This isolates whether the encoder represents object factors faithfully when object identity
is given. The central paper comparison is **DINOSAUR native vs DINO oracle with the same
backbone**: any SGIA gap is attributed to discovery corrupting representations.

### Probe training & semantic metrics

1. Collect L2-normalized per-object features from the **`train`** split (before images).
2. Train **one probe per factor** (`color`, `material`, `size`, `shape`) — both **linear**
   (logistic regression) and **MLP** heads were used in development to check probe sensitivity.
3. Decode factor predictions for all objects on before/after images.
4. Compute semantic intervention metrics on edited pairs:
   - **TFA** — edited factor decoded correctly on the edited object after the edit
   - **NFP** — non-edited factors on the edited object unchanged
   - **UOP / UOP_rate** — untouched objects unchanged (strict vs rate-normalized)
   - **SceneGraphExact** — full after-scene graph matches ground truth
   - **ΔSGIA** — correct edited factor + localized graph change (soft intervention metric)
   - **SGIA** — SceneGraphExact **and** exclusive change at the correct object/factor (strict headline metric)
5. Compute change-locality metrics from embedding deltas: **EOA**, **CLS**, and **NED** on no-edit controls.
6. Report **mean ± 95% bootstrap CI** (10,000 resamples in the shipped evaluator; 1,000 in early development runs).

### Core paper baselines

| Model | Protocol | Trains on EditCLEVR? | Implementation summary |
|-------|----------|:--------------------:|------------------------|
| **Slot Attention** | Native | Yes (fine-tune) | Start from the official Google Research CLEVR object-discovery checkpoint (`gs://gresearch/slot-attention/object-discovery/` or a PyTorch conversion). Fine-tune on EditCLEVR **train before-images** only. Typical config: `K=7` slots, slot dim `64`, `128×128` input, ~50k extra steps, batch 64. Built on the upstream [object-centric-learning-framework](https://github.com/amazon-science/object-centric-learning-framework) Slot Attention modules. |
| **DINOSAUR** | Native | Yes (slot head only) | Frozen **DINO ViT-B/16** patch features from `torch.hub` → slot-attention grouping (`K=7`, dim `256`, 3 iterations) → MLP decoder reconstructing DINO tokens. **Pre-compute and cache** all train-set DINO features once (~2 GB) so training only updates the slot head (~3× faster). Train ~100k steps, batch 64, `224×224`, Adam `lr=4e-4` cosine decay. Development runs also used a **ViT-S/8** DINOSAUR variant for SAM2 hybrid experiments. |
| **DINO ViT-B/16 oracle** | Oracle | No | Same DINO backbone as DINOSAUR. Forward pass → `14×14` patch grid (768-d) → GT mask pooling → per-object features. No slot training. |
| **DINOv2 ViT-B/14 oracle** | Oracle | No | `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')` → `16×16` patches (768-d) → same mask pooling. Upper-bound style control with a stronger frozen encoder. |

The **DINOSAUR native vs DINO-oracle gap** is the paper's main discovery-vs-representation
diagnostic: same encoder family, different mask source (learned slots vs ground truth).

### Extended baseline family (development pipeline)

The full research pipeline also evaluated additional models reported in appendix /
ablation tables:

| Family | Model IDs (examples) | Mask source | Feature source |
|--------|---------------------|-------------|----------------|
| **SAM2 + frozen backbone** | `sam2_dino_s8`, `sam2_dinov2`, `sam2_siglip2_384` | SAM2 ViT-Hiera-tiny automatic masks (cached once per run) | Frozen DINO ViT-S/8, DINOv2, or SigLIP2 patch tokens |
| **Extra frozen oracles** | `dino_oracle_s8`, `dinov2_oracle`, `siglip2_oracle` | Ground truth | Matching frozen backbones at 224 px |
| **Vendored native slots** | `slotdiffusion_native`, `spot_native` | Learned (SlotDiffusion / SPOT) | Trained end-to-end on EditCLEVR train |
| **Hybrid ablation** | `dinosaur_masks_dino_s8` | Soft masks from trained DINOSAUR decoder | Frozen DINO ViT-S/8 pooled under those masks (isolates mask quality vs GT oracle) |
| **Compositional-pretraining oracles** | CLIP ViT-B/16, OpenCLIP B/16 (LAION-2B), SigLIP-1 B/16, MAE ViT-B/16, I-JEPA H/14 | Ground truth | Each frozen backbone + linear probes |

**SAM2 note:** SAM2 masks were computed once per dataset run and reused so later cells only
re-ran the cheap backbone forward pass. `sam2_dino_s8` uses the **same** DINO ViT-S/8
backbone as `dino_oracle_s8`; only the mask source differs (automatic vs GT).

### External dependencies (not in this release)

Reimplementing the paper baselines requires, beyond this repo:

- **[object-centric-learning-framework](https://github.com/amazon-science/object-centric-learning-framework)** — Slot Attention + DINOSAUR building blocks
- **Google Slot Attention CLEVR checkpoint** — initialization for native SA fine-tuning
- **`torch.hub` DINO / DINOv2** — frozen backbones for DINOSAUR and oracle rows
- **SAM2** — automatic mask generation for hybrid baselines
- **SlotDiffusion (image path) & SPOT** — vendored first-party copies were used for native slot-diffusion / SPOT rows
- **Optional compositional encoders** — CLIP, OpenCLIP, SigLIP, MAE, I-JEPA via `torch.hub` or `timm`

### Suggested reproduction checklist

1. Download Phase-1 data: `python -m editclevr.download`
2. Train or load each baseline; extract **per-object before/after features** (+ native soft masks if applicable) for every split.
3. Hungarian-match native slots to GT masks (IoU threshold 0.5; flag low-confidence matches below IoU 0.1).
4. Train linear (and optionally MLP) factor probes on **train** features.
5. Build per-pair prediction records matching `Evaluator.expected_input_format` (change magnitudes, decoded factors, edited-object index).
6. Score with `Evaluator(n_bootstrap=10000)` or `editclevr/evaluation/run_evaluation`-equivalent probe + metric code.
7. Compare suites separately: `atomic_id`, `no_edit`, `hard_distractor`, `cogent_ood`.

**Compute ballpark (development runs on Colab A100):** Slot Attention fine-tune ~3 hr; DINOSAUR
feature cache ~0.5 hr + slot-head train ~4.5 hr; full multi-baseline feature extraction ~1 hr;
probe + metric pass is CPU-only.

### What you can run in this repository today

| Goal | Command / API |
|------|----------------|
| Reference end-to-end eval (simple GT-mask oracle) | `python3 -m editclevr.evaluation.run_evaluation` |
| Score your own extracted features | `from editclevr.evaluation import Evaluator` |
| GT mask pooling over a torch feature map | `editclevr.evaluation.oracle_pooling.mask_pool_features` (requires `pip install -e ".[torch]"`) |
| Slot–GT Hungarian alignment | `editclevr.evaluation.slot_matching.match_masks_by_iou` |

For protocol semantics and intended use, see [`artifacts/benchmark_card.md`](artifacts/benchmark_card.md).


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
