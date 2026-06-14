from __future__ import annotations

import json
from dataclasses import dataclass
import os
from pathlib import Path
import random
import subprocess
from typing import Any

import numpy as np
from PIL import Image

from .renderer_interface import SceneRenderer
from .scene import SceneDescription, ScenePair
from editclevr.generator.clevr_upstream import clevr_blender_root
from editclevr.paths import repo_root
from .schema import FACTOR_VALUES, available_scene_edit_values, objects_match_condition
from .seed_utils import derive_seed


@dataclass
class BlenderRenderer(SceneRenderer):
    blender_executable: str = "blender"
    generator_root: Path | None = None
    name: str = "blender"

    def render_scene(self, scene: SceneDescription, output_dir: Path) -> SceneDescription:
        raise NotImplementedError(
            "BlenderRenderer currently wraps random CLEVR scene generation, not direct SceneDescription rendering."
        )

    def render_pair(self, pair: ScenePair, output_dir: Path) -> dict[str, object]:
        raise NotImplementedError(
            "Paired SceneDescription -> Blender rendering is not implemented yet."
        )

    def _generator_root(self) -> Path:
        return self.generator_root or clevr_blender_root()

    def _scene_ops_script(self) -> Path:
        script_path = self._generator_root() / "editclevr_scene_ops.py"
        if not script_path.exists():
            raise FileNotFoundError(f"Missing Blender scene ops script: {script_path}")
        return script_path

    def _blender_site_packages(self) -> Path | None:
        executable = self.blender_executable.replace("\\", "/").lower()
        candidate = repo_root() / "external" / ("blender292-python-site" if "2.92" in executable else "blender-python-site")
        return candidate if candidate.exists() else None

    def _blender_subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        python_paths = [str(self._generator_root())]
        site_packages = self._blender_site_packages()
        if site_packages is not None:
            python_paths.append(str(site_packages))
        if env.get("PYTHONPATH"):
            python_paths.append(str(env["PYTHONPATH"]))
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        return env

    def _run_blender(self, script_args: list[str]) -> None:
        cmd = [
            self.blender_executable,
            "--background",
            "--python-use-system-env",
            "--python",
            str(self._scene_ops_script()),
            "--",
            *script_args,
        ]
        subprocess.run(cmd, cwd=self._generator_root(), env=self._blender_subprocess_env(), check=True)

    @staticmethod
    def _condition_combo_file(generator_root: Path, condition: str | None) -> str | None:
        if condition is None:
            return None
        return str(generator_root / "data" / f"CoGenT_{condition}.json")

    @staticmethod
    def _pick_auto_edit_value(
        candidates: tuple[str, ...],
        *,
        factor: str,
        object_index: int,
        selection_seed: int | None = None,
        scene_seed: int | None = None,
    ) -> str:
        if not candidates:
            raise ValueError(f"No alternate values available for factor '{factor}'.")
        salt = sum((idx + 1) * ord(char) for idx, char in enumerate(factor))
        seed = (
            (selection_seed if selection_seed is not None else 0)
            ^ ((scene_seed if scene_seed is not None else 0) << 1)
            ^ ((object_index + 1) * 1009)
            ^ salt
        )
        rng = random.Random(seed)
        return rng.choice(list(candidates))

    @staticmethod
    def _pick_object_index(
        objects: list[dict[str, Any]],
        object_index: int | None = None,
        selection_seed: int | None = None,
    ) -> int:
        if object_index is not None:
            if not 0 <= object_index < len(objects):
                raise IndexError(f"object_index={object_index} is out of range for {len(objects)} objects.")
            return object_index
        if not objects:
            raise ValueError("Cannot choose an object from an empty scene.")
        rng = random.Random(
            selection_seed if selection_seed is not None else derive_seed(len(objects), "select_target")
        )
        return rng.randrange(len(objects))

    @staticmethod
    def _hard_distractor_candidates(
        objects: list[dict[str, Any]],
        preferred_factor: str | None = None,
        condition: str | None = None,
    ) -> list[tuple[int, tuple[str, ...]]]:
        candidates: list[tuple[int, tuple[str, ...]]] = []
        for target_index, target in enumerate(objects):
            shared_factors = set()
            for other_index, other in enumerate(objects):
                if other_index == target_index:
                    continue
                shared = [factor for factor in FACTOR_VALUES if other[factor] == target[factor]]
                if len(shared) >= 2:
                    shared_factors.update(shared)
            if preferred_factor is not None:
                candidate_factors = tuple(f for f in shared_factors if f == preferred_factor)
            else:
                candidate_factors = tuple(sorted(shared_factors))
            if not candidate_factors:
                continue
            valid_factors = tuple(
                factor
                for factor in candidate_factors
                if available_scene_edit_values(
                    shape=target["shape"],
                    color=target["color"],
                    factor=factor,
                    current_value=target[factor],
                    condition=condition,
                )
            )
            if valid_factors:
                candidates.append((target_index, valid_factors))
        return candidates

    def generate_base_scene(
        self,
        output_dir: Path,
        num_images: int = 1,
        split: str = "editclevr",
        width: int = 320,
        height: int = 240,
        render_num_samples: int = 32,
        min_objects: int = 3,
        max_objects: int = 6,
        min_pixels_per_object: int = 200,
        save_blendfiles: bool = False,
        shape_color_combos_json: str | None = None,
        start_idx: int = 0,
        seed: int | None = None,
    ) -> dict[str, object]:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = output_dir / "images"
        scene_dir = output_dir / "scenes"
        mask_dir = output_dir / "masks"
        mask_npz_dir = output_dir / "masks_npz"
        blend_dir = output_dir / "blends"
        scene_file = output_dir / "CLEVR_scenes.json"

        cmd = [
            "--mode",
            "random",
            "--num_images",
            str(num_images),
            "--start_idx",
            str(start_idx),
            "--split",
            split,
            "--width",
            str(width),
            "--height",
            str(height),
            "--min_objects",
            str(min_objects),
            "--max_objects",
            str(max_objects),
            "--min_pixels_per_object",
            str(min_pixels_per_object),
            "--render_num_samples",
            str(render_num_samples),
            "--output_image_dir",
            str(image_dir),
            "--output_scene_dir",
            str(scene_dir),
            "--output_mask_dir",
            str(mask_dir),
            "--output_blend_dir",
            str(blend_dir),
            "--output_scene_file",
            str(scene_file),
        ]
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        if shape_color_combos_json:
            cmd.extend(["--shape_color_combos_json", shape_color_combos_json])
        if save_blendfiles:
            cmd.extend(["--save_blendfiles", "1"])
        self._run_blender(cmd)
        payload = json.loads(scene_file.read_text())
        self._convert_mask_images_to_npz(payload, mask_dir, mask_npz_dir)
        return payload

    def render_random_scenes(
        self,
        output_dir: Path,
        num_images: int = 1,
        split: str = "editclevr",
        width: int = 320,
        height: int = 240,
        render_num_samples: int = 32,
        min_objects: int = 3,
        max_objects: int = 6,
        min_pixels_per_object: int = 200,
        save_blendfiles: bool = False,
        shape_color_combos_json: str | None = None,
        start_idx: int = 0,
        seed: int | None = None,
    ) -> dict[str, object]:
        return self.generate_base_scene(
            output_dir=output_dir,
            num_images=num_images,
            split=split,
            width=width,
            height=height,
            render_num_samples=render_num_samples,
            min_objects=min_objects,
            max_objects=max_objects,
            min_pixels_per_object=min_pixels_per_object,
            save_blendfiles=save_blendfiles,
            shape_color_combos_json=shape_color_combos_json,
            start_idx=start_idx,
            seed=seed,
        )

    def _base_scene_paths(self, base_dir: Path, scene_payload: dict[str, Any]) -> dict[str, Path]:
        base_stem = Path(scene_payload["image_filename"]).stem
        return {
            "blend": base_dir / "blends" / f"{base_stem}.blend",
            "scene_json": base_dir / "scenes" / f"{base_stem}.json",
            "mask_npz": base_dir / "masks_npz" / f"{Path(scene_payload['mask_filename']).stem}.npz",
        }

    def apply_atomic_edit(
        self,
        input_blendfile: Path | str,
        input_scene_json: Path | str,
        output_dir: Path | str,
        object_index: int,
        factor: str,
        new_value: str,
    ) -> dict[str, object]:
        output_dir = Path(output_dir).resolve()
        image_dir = output_dir / "images"
        scene_dir = output_dir / "scenes"
        mask_dir = output_dir / "masks"
        mask_npz_dir = output_dir / "masks_npz"

        input_scene_json = Path(input_scene_json).resolve()
        base_name = input_scene_json.stem + "_edit"
        output_image = image_dir / f"{base_name}.png"
        output_scene = scene_dir / f"{base_name}.json"
        output_mask = mask_dir / f"{base_name}.png"

        cmd = [
            "--mode",
            "edit",
            "--input_blendfile",
            str(Path(input_blendfile).resolve()),
            "--input_scene_json",
            str(input_scene_json),
            "--output_image",
            str(output_image),
            "--output_scene",
            str(output_scene),
            "--output_mask",
            str(output_mask),
            "--object_index",
            str(object_index),
            "--factor",
            factor,
            "--new_value",
            new_value,
        ]
        self._run_blender(cmd)
        if not output_scene.exists():
            raise FileNotFoundError(f"Expected edited scene JSON at {output_scene}, but Blender did not produce it.")
        payload = json.loads(output_scene.read_text())
        self._convert_mask_images_to_npz({"scenes": [payload]}, mask_dir, mask_npz_dir)
        return payload

    def rerender_scene(
        self,
        input_blendfile: Path | str,
        input_scene_json: Path | str,
        output_dir: Path | str,
        seed: int = 0,
    ) -> dict[str, object]:
        output_dir = Path(output_dir).resolve()
        image_dir = output_dir / "images"
        scene_dir = output_dir / "scenes"
        mask_dir = output_dir / "masks"
        mask_npz_dir = output_dir / "masks_npz"

        input_scene_json = Path(input_scene_json).resolve()
        base_name = input_scene_json.stem + "_rerender"
        output_image = image_dir / f"{base_name}.png"
        output_scene = scene_dir / f"{base_name}.json"
        output_mask = mask_dir / f"{base_name}.png"

        cmd = [
            "--mode",
            "rerender",
            "--input_blendfile",
            str(Path(input_blendfile).resolve()),
            "--input_scene_json",
            str(input_scene_json),
            "--output_image",
            str(output_image),
            "--output_scene",
            str(output_scene),
            "--output_mask",
            str(output_mask),
            "--seed",
            str(seed),
        ]
        self._run_blender(cmd)
        payload = json.loads(output_scene.read_text())
        self._convert_mask_images_to_npz({"scenes": [payload]}, mask_dir, mask_npz_dir)
        return payload

    def generate_atomic_pair(
        self,
        output_dir: Path | str,
        object_index: int | None,
        factor: str,
        new_value: str | None = None,
        width: int = 320,
        height: int = 240,
        render_num_samples: int = 8,
        min_objects: int = 3,
        max_objects: int = 3,
        min_pixels_per_object: int = 50,
        condition: str | None = None,
        start_idx: int = 0,
        scene_seed: int | None = None,
        selection_seed: int | None = None,
    ) -> dict[str, object]:
        output_dir = Path(output_dir).resolve()
        before_dir = output_dir / "before"
        after_dir = output_dir / "after"
        combo_file = self._condition_combo_file(self._generator_root(), condition)

        before_payload = self.render_random_scenes(
            output_dir=before_dir,
            num_images=1,
            split="editclevr_atomic_a" if condition == "A" else "editclevr_before",
            width=width,
            height=height,
            render_num_samples=render_num_samples,
            min_objects=min_objects,
            max_objects=max_objects,
            min_pixels_per_object=min_pixels_per_object,
            save_blendfiles=True,
            shape_color_combos_json=combo_file,
            start_idx=start_idx,
            seed=scene_seed,
        )
        before_scene = before_payload["scenes"][0]
        base_stem = Path(before_scene["image_filename"]).stem
        blend_path = before_dir / "blends" / f"{base_stem}.blend"
        scene_json_path = before_dir / "scenes" / f"{base_stem}.json"
        actual_object_index = self._pick_object_index(
            before_scene["objects"],
            object_index=object_index,
            selection_seed=selection_seed,
        )
        actual_old_value = before_scene["objects"][actual_object_index][factor]
        candidates = available_scene_edit_values(
            shape=before_scene["objects"][actual_object_index]["shape"],
            color=before_scene["objects"][actual_object_index]["color"],
            factor=factor,
            current_value=actual_old_value,
            condition=condition,
        )
        if new_value is None or new_value == "auto":
            resolved_new_value = self._pick_auto_edit_value(
                candidates,
                factor=factor,
                object_index=actual_object_index,
                selection_seed=selection_seed,
                scene_seed=scene_seed,
            )
        else:
            resolved_new_value = new_value
            if resolved_new_value not in candidates:
                raise ValueError(
                    f"Value '{resolved_new_value}' is invalid for factor '{factor}' under condition {condition or 'none'}."
                )

        after_scene = self.apply_atomic_edit(
            input_blendfile=blend_path,
            input_scene_json=scene_json_path,
            output_dir=after_dir,
            object_index=actual_object_index,
            factor=factor,
            new_value=resolved_new_value,
        )
        if condition is not None and not objects_match_condition(after_scene["objects"], condition):
            raise ValueError(f"Atomic pair edit breaks condition {condition}.")

        pair_metadata = {
            "pair_id": f"{base_stem}_{factor}_{actual_object_index}",
            "suite": "atomic_id",
            "base_scene_seed": scene_seed,
            "before": {
                "image": str(before_dir / "images" / before_scene["image_filename"]),
                "scene_json": str(scene_json_path),
                "mask_png": str(before_dir / "masks" / before_scene["mask_filename"]),
                "mask_npz": str(before_dir / "masks_npz" / f"{Path(before_scene['mask_filename']).stem}.npz"),
                "blendfile": str(blend_path),
                "scene": before_scene,
            },
            "after": {
                "image": str(after_dir / "images" / after_scene["image_filename"]),
                "scene_json": str(after_dir / "scenes" / f"{Path(after_scene['image_filename']).stem}.json"),
                "mask_png": str(after_dir / "masks" / after_scene["mask_filename"]),
                "mask_npz": str(after_dir / "masks_npz" / f"{Path(after_scene['mask_filename']).stem}.npz"),
                "scene": after_scene,
            },
            "edit": {
                "object_index": actual_object_index,
                "factor": factor,
                "old_value": actual_old_value,
                "new_value": after_scene["objects"][actual_object_index][factor],
            },
        }
        if condition is not None:
            pair_metadata["condition"] = condition
        (output_dir / "pair_metadata.json").write_text(json.dumps(pair_metadata, indent=2))
        return pair_metadata

    def generate_no_edit_pair(
        self,
        output_dir: Path | str,
        object_index: int | None = None,
        width: int = 320,
        height: int = 240,
        render_num_samples: int = 8,
        min_objects: int = 3,
        max_objects: int = 3,
        min_pixels_per_object: int = 50,
        seed: int = 1,
        condition: str | None = None,
        start_idx: int = 0,
        scene_seed: int | None = None,
        selection_seed: int | None = None,
    ) -> dict[str, object]:
        output_dir = Path(output_dir).resolve()
        before_dir = output_dir / "before"
        after_dir = output_dir / "after"
        combo_file = self._condition_combo_file(self._generator_root(), condition)

        before_payload = self.generate_base_scene(
            output_dir=before_dir,
            num_images=1,
            split="editclevr_noedit_a" if condition == "A" else "editclevr_noedit_before",
            width=width,
            height=height,
            render_num_samples=render_num_samples,
            min_objects=min_objects,
            max_objects=max_objects,
            min_pixels_per_object=min_pixels_per_object,
            save_blendfiles=True,
            shape_color_combos_json=combo_file,
            start_idx=start_idx,
            seed=scene_seed,
        )
        before_scene = before_payload["scenes"][0]
        paths = self._base_scene_paths(before_dir, before_scene)
        actual_object_index = self._pick_object_index(
            before_scene["objects"],
            object_index=object_index,
            selection_seed=selection_seed,
        )
        after_scene = self.rerender_scene(
            input_blendfile=paths["blend"],
            input_scene_json=paths["scene_json"],
            output_dir=after_dir,
            seed=seed,
        )
        if condition is not None and not objects_match_condition(after_scene["objects"], condition):
            raise ValueError(f"No-edit rerender breaks condition {condition}.")

        pair_metadata = {
            "pair_id": f"{Path(before_scene['image_filename']).stem}_noop",
            "suite": "no_edit",
            "base_scene_seed": scene_seed,
            "rerender_seed": seed,
            "before": {
                "image": str(before_dir / "images" / before_scene["image_filename"]),
                "scene_json": str(paths["scene_json"]),
                "mask_png": str(before_dir / "masks" / before_scene["mask_filename"]),
                "mask_npz": str(paths["mask_npz"]),
                "blendfile": str(paths["blend"]),
                "scene": before_scene,
            },
            "after": {
                "image": str(after_dir / "images" / after_scene["image_filename"]),
                "scene_json": str(after_dir / "scenes" / f"{Path(after_scene['image_filename']).stem}.json"),
                "mask_png": str(after_dir / "masks" / after_scene["mask_filename"]),
                "mask_npz": str(after_dir / "masks_npz" / f"{Path(after_scene['mask_filename']).stem}.npz"),
                "scene": after_scene,
            },
            "edit": {
                "object_index": actual_object_index,
                "factor": "none",
                "old_value": "none",
                "new_value": "none",
            },
        }
        if condition is not None:
            pair_metadata["condition"] = condition
        (output_dir / "pair_metadata.json").write_text(json.dumps(pair_metadata, indent=2))
        return pair_metadata

    def generate_cogent_pair(
        self,
        output_dir: Path | str,
        object_index: int | None,
        factor: str,
        new_value: str | None = None,
        condition: str = "A",
        start_idx: int = 0,
        scene_seed: int | None = None,
        selection_seed: int | None = None,
        **kwargs: Any,
    ) -> dict[str, object]:
        output_dir = Path(output_dir).resolve()
        combo_file = self._generator_root() / "data" / f"CoGenT_{condition}.json"
        before_dir = output_dir / "before"
        after_dir = output_dir / "after"

        before_payload = self.generate_base_scene(
            output_dir=before_dir,
            num_images=1,
            split=f"editclevr_cogent_{condition.lower()}",
            save_blendfiles=True,
            shape_color_combos_json=str(combo_file),
            start_idx=start_idx,
            seed=scene_seed,
            **kwargs,
        )
        before_scene = before_payload["scenes"][0]
        paths = self._base_scene_paths(before_dir, before_scene)
        actual_object_index = self._pick_object_index(
            before_scene["objects"],
            object_index=object_index,
            selection_seed=selection_seed,
        )
        actual_old_value = before_scene["objects"][actual_object_index][factor]
        candidates = available_scene_edit_values(
            shape=before_scene["objects"][actual_object_index]["shape"],
            color=before_scene["objects"][actual_object_index]["color"],
            factor=factor,
            current_value=actual_old_value,
            condition=condition,
        )
        if new_value is None or new_value == "auto":
            resolved_new_value = self._pick_auto_edit_value(
                candidates,
                factor=factor,
                object_index=actual_object_index,
                selection_seed=selection_seed,
                scene_seed=scene_seed,
            )
        else:
            resolved_new_value = new_value
            if resolved_new_value not in candidates:
                raise ValueError(
                    f"Value '{resolved_new_value}' is invalid for factor '{factor}' under condition {condition}."
                )
        after_scene = self.apply_atomic_edit(
            input_blendfile=paths["blend"],
            input_scene_json=paths["scene_json"],
            output_dir=after_dir,
            object_index=actual_object_index,
            factor=factor,
            new_value=resolved_new_value,
        )
        if not objects_match_condition(after_scene["objects"], condition):
            raise ValueError(f"CoGenT edit breaks condition {condition}.")
        pair_metadata = {
            "pair_id": f"{Path(before_scene['image_filename']).stem}_{factor}_{actual_object_index}",
            "suite": "cogent_ood",
            "condition": condition,
            "base_scene_seed": scene_seed,
            "before": {
                "image": str(before_dir / "images" / before_scene["image_filename"]),
                "scene_json": str(paths["scene_json"]),
                "mask_png": str(before_dir / "masks" / before_scene["mask_filename"]),
                "mask_npz": str(paths["mask_npz"]),
                "blendfile": str(paths["blend"]),
                "scene": before_scene,
            },
            "after": {
                "image": str(after_dir / "images" / after_scene["image_filename"]),
                "scene_json": str(after_dir / "scenes" / f"{Path(after_scene['image_filename']).stem}.json"),
                "mask_png": str(after_dir / "masks" / after_scene["mask_filename"]),
                "mask_npz": str(after_dir / "masks_npz" / f"{Path(after_scene['mask_filename']).stem}.npz"),
                "scene": after_scene,
            },
            "edit": {
                "object_index": actual_object_index,
                "factor": factor,
                "old_value": actual_old_value,
                "new_value": after_scene["objects"][actual_object_index][factor],
            },
        }
        (output_dir / "pair_metadata.json").write_text(json.dumps(pair_metadata, indent=2))
        return pair_metadata

    def generate_hard_distractor_pair(
        self,
        output_dir: Path | str,
        factor: str | None = None,
        width: int = 320,
        height: int = 240,
        render_num_samples: int = 8,
        min_objects: int = 3,
        max_objects: int = 6,
        min_pixels_per_object: int = 50,
        max_attempts: int = 10,
        condition: str | None = None,
        start_idx: int = 0,
        scene_seed: int | None = None,
        selection_seed: int | None = None,
    ) -> dict[str, object]:
        output_dir = Path(output_dir).resolve()
        combo_file = self._condition_combo_file(self._generator_root(), condition)
        for attempt in range(max_attempts):
            attempt_dir = output_dir / f"attempt_{attempt:02d}"
            attempt_seed = None if scene_seed is None else scene_seed + attempt
            before_payload = self.generate_base_scene(
                output_dir=attempt_dir / "before",
                num_images=1,
                split="editclevr_hard_a" if condition == "A" else "editclevr_hard_before",
                width=width,
                height=height,
                render_num_samples=render_num_samples,
                min_objects=min_objects,
                max_objects=max_objects,
                min_pixels_per_object=min_pixels_per_object,
                save_blendfiles=True,
                shape_color_combos_json=combo_file,
                start_idx=start_idx,
                seed=attempt_seed,
            )
            before_scene = before_payload["scenes"][0]
            candidates = self._hard_distractor_candidates(
                before_scene["objects"],
                preferred_factor=factor,
                condition=condition,
            )
            if not candidates:
                continue
            base_selection_seed = (
                selection_seed
                if selection_seed is not None
                else derive_seed(attempt_seed, "select_target")
            )
            rng = random.Random(derive_seed(base_selection_seed, "hard_attempt", attempt))
            target_index, factor_choices = rng.choice(candidates)
            chosen_factor = factor or rng.choice(list(factor_choices))
            target = before_scene["objects"][target_index]
            paths = self._base_scene_paths(attempt_dir / "before", before_scene)
            new_value = rng.choice(
                available_scene_edit_values(
                    shape=target["shape"],
                    color=target["color"],
                    factor=chosen_factor,
                    current_value=target[chosen_factor],
                    condition=condition,
                )
            )
            after_scene = self.apply_atomic_edit(
                input_blendfile=paths["blend"],
                input_scene_json=paths["scene_json"],
                output_dir=attempt_dir / "after",
                object_index=target_index,
                factor=chosen_factor,
                new_value=new_value,
            )
            if condition is not None and not objects_match_condition(after_scene["objects"], condition):
                raise ValueError(f"Hard-distractor edit breaks condition {condition}.")
            pair_metadata = {
                "pair_id": f"{Path(before_scene['image_filename']).stem}_{chosen_factor}_{target_index}",
                "suite": "hard_distractor",
                "base_scene_seed": attempt_seed,
                "before": {
                    "image": str((attempt_dir / 'before' / 'images' / before_scene["image_filename"])),
                    "scene_json": str(paths["scene_json"]),
                    "mask_png": str((attempt_dir / 'before' / 'masks' / before_scene["mask_filename"])),
                    "mask_npz": str(paths["mask_npz"]),
                    "blendfile": str(paths["blend"]),
                    "scene": before_scene,
                },
                "after": {
                    "image": str((attempt_dir / 'after' / 'images' / after_scene["image_filename"])),
                    "scene_json": str(attempt_dir / "after" / "scenes" / f"{Path(after_scene['image_filename']).stem}.json"),
                    "mask_png": str((attempt_dir / 'after' / 'masks' / after_scene["mask_filename"])),
                    "mask_npz": str(attempt_dir / "after" / "masks_npz" / f"{Path(after_scene['mask_filename']).stem}.npz"),
                    "scene": after_scene,
                },
                "edit": {
                    "object_index": target_index,
                    "factor": chosen_factor,
                    "old_value": target[chosen_factor],
                    "new_value": after_scene["objects"][target_index][chosen_factor],
                },
            }
            if condition is not None:
                pair_metadata["condition"] = condition
            final_dir = output_dir
            final_dir.mkdir(parents=True, exist_ok=True)
            (final_dir / "pair_metadata.json").write_text(json.dumps(pair_metadata, indent=2))
            return pair_metadata
        raise RuntimeError("Failed to generate a hard-distractor pair within the attempt limit.")

    def _convert_mask_images_to_npz(
        self,
        payload: dict[str, object],
        mask_dir: Path,
        mask_npz_dir: Path,
    ) -> None:
        mask_npz_dir.mkdir(parents=True, exist_ok=True)
        for scene in payload.get("scenes", []):
            mask_filename = scene.get("mask_filename")
            if not mask_filename:
                continue
            mask_image = np.asarray(Image.open(mask_dir / mask_filename).convert("RGB"), dtype=np.uint8)
            object_count = len(scene.get("objects", []))
            instance_masks = []
            blue_channel = mask_image[..., 2].astype(np.int16)
            unique_values = sorted(int(v) for v in np.unique(blue_channel) if v > 0)
            for object_idx in range(object_count):
                if object_idx < len(unique_values):
                    target_value = unique_values[object_idx]
                    instance_masks.append((np.abs(blue_channel - target_value) <= 4).astype(np.uint8))
                else:
                    instance_masks.append(np.zeros_like(blue_channel, dtype=np.uint8))
            np.savez_compressed(
                mask_npz_dir / (Path(mask_filename).stem + ".npz"),
                masks=np.stack(instance_masks, axis=0),
                object_ids=np.arange(object_count, dtype=np.int32),
            )
            mapping = {
                "mask_file": mask_filename,
                "mask_to_object_id": {str(index): index for index in range(object_count)},
                "source": "blue-channel sorted instance mask conversion",
            }
            (mask_npz_dir / (Path(mask_filename).stem + "_mask_to_object_id.json")).write_text(
                json.dumps(mapping, indent=2)
            )
