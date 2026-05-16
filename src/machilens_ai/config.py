from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any
import random

import numpy as np
import torch
import yaml


@dataclass
class MachiLensConfig:
    """Configuration for the MachiLens AI prediction simulator."""

    project_name: str = "MachiLens AI"
    seed: int = 42
    n_samples: int = 1600
    window_steps: int = 24
    city_dim: int = 5
    context_dim: int = 6
    hidden_size: int = 64
    fusion_units: int = 128
    head_units: int = 64
    dropout: float = 0.25
    scenario_kinds: int = 3
    batch_size: int = 64
    epochs: int = 8
    base_lr: float = 1e-3
    weight_decay: float = 1e-5
    train_ratio: float = 0.75
    val_ratio: float = 0.15
    loss_success_weight: float = 1.0
    loss_scenario_weight: float = 0.4
    loss_regression_weight: float = 0.5
    mc_dropout_samples: int = 16
    artifact_dir: str = "artifacts"
    output_dir: str = "reports"
    torch_num_threads: int = 1

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["device"] = self.device
        return payload


def _filter_config_keys(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = {f.name for f in fields(MachiLensConfig)}
    return {k: v for k, v in raw.items() if k in allowed}


def load_config(path: str | Path | None = None) -> MachiLensConfig:
    """Load YAML config, falling back to dataclass defaults."""
    if path is None:
        return MachiLensConfig()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return MachiLensConfig(**_filter_config_keys(raw))


def config_from_dict(raw: dict[str, Any]) -> MachiLensConfig:
    """Reconstruct a MachiLensConfig from checkpoint/config dictionaries."""
    return MachiLensConfig(**_filter_config_keys(raw))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
