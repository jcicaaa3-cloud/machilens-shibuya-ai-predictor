from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import MachiLensConfig, config_from_dict


class TemporalAttention(nn.Module):
    """Simple temporal attention layer over LSTM outputs."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.score = nn.Linear(feature_dim, 1)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # sequence: (B, T, H)
        weights = torch.softmax(self.score(sequence), dim=1)  # (B, T, 1)
        pooled = torch.sum(weights * sequence, dim=1)  # (B, H)
        return pooled, weights.squeeze(-1)


class MachiLensPredictor(nn.Module):
    """AI predictor for MachiLens scenario forecasting.

    Upgrade over the original conceptual LSTM+MLP:
    - bidirectional LSTM encoder
    - temporal attention pooling
    - multi-task prediction heads
    """

    def __init__(self, cfg: MachiLensConfig):
        super().__init__()
        self.cfg = cfg
        encoded_dim = cfg.hidden_size * 2
        self.flow_encoder = nn.LSTM(
            input_size=cfg.city_dim,
            hidden_size=cfg.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = TemporalAttention(encoded_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(encoded_dim + cfg.context_dim, cfg.fusion_units),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_units, cfg.head_units),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.head_success = nn.Linear(cfg.head_units, 1)
        self.head_scenario = nn.Linear(cfg.head_units, cfg.scenario_kinds)
        self.head_regress = nn.Linear(cfg.head_units, 2)

    def forward(self, flow_window: torch.Tensor, context_vec: torch.Tensor) -> dict[str, torch.Tensor]:
        flow_out, _ = self.flow_encoder(flow_window)
        attended, attention_weights = self.attention(flow_out)
        fused = torch.cat([attended, context_vec], dim=1)
        shared = self.fusion_mlp(fused)
        logit_success = self.head_success(shared).squeeze(-1)
        logits_scenario = self.head_scenario(shared)
        reg_pair = self.head_regress(shared)
        return {
            "logit_success": logit_success,
            "prob_success": torch.sigmoid(logit_success),
            "logits_scenario": logits_scenario,
            "delta_flow_pred": reg_pair[:, 0],
            "delta_vacancy_pred": reg_pair[:, 1],
            "attention_weights": attention_weights,
        }


def save_checkpoint(
    model: MachiLensPredictor,
    cfg: MachiLensConfig,
    path: str | Path,
    metrics: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.to_dict(),
            "metrics": metrics or {},
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[MachiLensPredictor, MachiLensConfig, dict[str, Any]]:
    path = Path(path)
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    cfg = config_from_dict(checkpoint.get("config", {}))
    model = MachiLensPredictor(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(map_location)
    model.eval()
    return model, cfg, checkpoint.get("metrics", {})
