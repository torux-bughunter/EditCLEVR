from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .scene import SceneDescription, ScenePair


class SceneRenderer(ABC):
    name: str

    @abstractmethod
    def render_scene(self, scene: SceneDescription, output_dir: Path) -> SceneDescription:
        """Render one scene and return an updated scene with artifact paths."""

    @abstractmethod
    def render_pair(self, pair: ScenePair, output_dir: Path) -> dict[str, object]:
        """Render a pair and return the emitted metadata payload."""
