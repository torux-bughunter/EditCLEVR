from __future__ import annotations

from pathlib import Path


def clevr_blender_root() -> Path:
    """Return the vendored CLEVR Blender generation directory.

    Holds the EditCLEVR-modified CLEVR scripts (``render_images.py``,
    ``apply_atomic_edit.py``, ``render_existing_scene.py``, ``utils.py``), the
    ``editclevr_scene_ops.py`` dispatcher, and the ``data/`` Blender assets.
    """

    root = Path(__file__).resolve().parent / "clevr_blender"
    if not root.exists():
        raise FileNotFoundError(f"Missing vendored CLEVR Blender scripts at {root}")
    return root
