from __future__ import annotations

import argparse
from pathlib import Path

from editclevr.paths import resolve_blender_executable

from .blender_adapter import BlenderRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="Render real CLEVR scenes through Blender.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/clevr_real"))
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--split", type=str, default="editclevr")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument(
        "--blender-executable",
        type=str,
        default=resolve_blender_executable(),
    )
    args = parser.parse_args()

    renderer = BlenderRenderer(blender_executable=args.blender_executable)
    renderer.render_random_scenes(
        output_dir=args.output_dir,
        num_images=args.num_images,
        split=args.split,
        width=args.width,
        height=args.height,
        render_num_samples=args.samples,
    )


if __name__ == "__main__":
    main()
