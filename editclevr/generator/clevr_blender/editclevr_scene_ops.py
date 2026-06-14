"""EditCLEVR Blender entry point dispatcher.

Run inside Blender, e.g.:

    blender --background --python editclevr_scene_ops.py -- --mode random [args]

`--mode` selects the operation and the remaining arguments are forwarded to the
corresponding script's own argument parser:

    random    -> render_images.py        (render fresh random base scenes)
    edit      -> apply_atomic_edit.py    (apply a single-factor atomic edit)
    rerender  -> render_existing_scene.py (re-render a scene with light jitter)
"""

from __future__ import print_function

import os
import sys

INSIDE_BLENDER = True
try:
    import bpy  # noqa: F401
except ImportError:
    INSIDE_BLENDER = False

if INSIDE_BLENDER:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import utils
    import render_images
    import apply_atomic_edit
    import render_existing_scene

MODES = ("random", "edit", "rerender")


def _extract_mode(argv):
    if "--mode" not in argv:
        raise SystemExit("editclevr_scene_ops.py requires --mode {%s}" % ",".join(MODES))
    idx = argv.index("--mode")
    if idx + 1 >= len(argv):
        raise SystemExit("--mode requires a value")
    mode = argv[idx + 1]
    remaining = argv[:idx] + argv[idx + 2 :]
    return mode, remaining


def main():
    mode, remaining = _extract_mode(utils.extract_args())
    if mode == "random":
        render_images.main(render_images.parser.parse_args(remaining))
    elif mode == "edit":
        apply_atomic_edit.main(apply_atomic_edit.parser.parse_args(remaining))
    elif mode == "rerender":
        render_existing_scene.main(render_existing_scene.parser.parse_args(remaining))
    else:
        raise SystemExit("Unknown --mode %r; expected one of %s" % (mode, MODES))


if __name__ == "__main__":
    if INSIDE_BLENDER:
        main()
    elif "--help" in sys.argv or "-h" in sys.argv:
        print(
            "Usage: blender --background --python editclevr_scene_ops.py -- "
            "--mode {%s} [script args]" % ",".join(MODES)
        )
