from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class GameConfig:
    width: int = 1100
    height: int = 760
    fps: int = 30
    history_length: int = 90
    sim_seconds_per_minute: int = 30
    ticks_per_minute: int = 60
    prediction_minutes: int = 5
    stream_seconds: int = 20
    post_display_seconds: int = 6
    board_levels: int = 5
    log_root: str = "runs"

    @property
    def ticks_per_second(self) -> float:
        return self.ticks_per_minute / max(self.sim_seconds_per_minute, 1)

    @property
    def ticks_per_frame_target(self) -> float:
        return self.ticks_per_second / max(self.fps, 1)

    @property
    def stream_frames(self) -> int:
        return int(self.stream_seconds * self.fps)

    @property
    def post_display_frames(self) -> int:
        return int(self.post_display_seconds * self.fps)


class ConfigLoader:
    """Load YAML backed configuration with fallback defaults."""

    def __init__(self, path: Optional[Path] = None, defaults: Optional[Dict[str, Any]] = None):
        self.path = Path(path) if path is not None else Path("config/game.yaml")
        self.defaults = defaults or {
            "width": 1100,
            "height": 760,
            "fps": 30,
            "history_length": 90,
            "sim_seconds_per_minute": 30,
            "ticks_per_minute": 60,
            "prediction_minutes": 5,
            "stream_seconds": 20,
            "post_display_seconds": 6,
            "board_levels": 5,
            "log_root": "runs",
        }

    def load(self) -> GameConfig:
        data: Dict[str, Any] = copy.deepcopy(self.defaults)
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
            if not isinstance(loaded, dict):
                raise TypeError(f"Configuration at {self.path} must be a mapping")
            data.update(loaded)
        return GameConfig(**data)


__all__ = ["GameConfig", "ConfigLoader"]
