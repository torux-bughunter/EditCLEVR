from __future__ import annotations

import argparse
from pathlib import Path

from editclevr.paths import resolve_blender_executable

from .blender_adapter import BlenderRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a real Blender-backed before/after atomic pair.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/clevr_pair"))
    parser.add_argument("--object-index", type=int, default=0)
    parser.add_argument("--factor", type=str, required=True, choices=["color", "material", "size", "shape"])
    parser.add_argument("--new-value", type=str, default="auto")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--min-objects", type=int, default=3)
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--min-pixels", type=int, default=50)
    parser.add_argument(
        "--blender-executable",
        type=str,
        default=resolve_blender_executable(),
    )
    args = parser.parse_args()

    renderer = BlenderRenderer(blender_executable=args.blender_executable)
    renderer.generate_atomic_pair(
        output_dir=args.output_dir,
        object_index=args.object_index,
        factor=args.factor,
        new_value=args.new_value,
        width=args.width,
        height=args.height,
        render_num_samples=args.samples,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        min_pixels_per_object=args.min_pixels,
    )


if __name__ == "__main__":
    main()
