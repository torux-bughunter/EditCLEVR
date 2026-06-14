# EditCLEVR Benchmark Card

## Summary

EditCLEVR is a synthetic benchmark for testing whether object-centric representations respond faithfully to controlled object-level interventions.

## Intended Use

- Diagnose whether models localize representational change to the edited object.
- Compare native object discovery against oracle-mask evaluation.
- Stress-test intervention faithfulness under distractors and OOD attribute combinations (CoGenT).

## Data Composition

- **Primary renderer:** Blender-backed CLEVR extension (`editclevr.generator`).
- **Phase-1 release:** ~20k paired edits across atomic-ID, no-edit, hard-distractor, and CoGenT-OOD suites.
- Objects expose four intrinsic factors: `color`, `material`, `size`, `shape`.

## Distribution

- Hugging Face dataset `torux/EditCLEVR`, installed via `python -m editclevr.download`.

## Current Artifact Status

- Blender dataset generation and evaluation pipeline are implemented in-repo.

## Limitations

- CLEVR generator targets Blender 2.92; newer Blender versions may differ.
- Reported SGIA scores depend on probe calibration and the native/oracle evaluation protocol; compare models under the same protocol.
