from __future__ import annotations

import numpy as np
import torch

from .models import MachiLensPredictor


def mc_dropout_predict(
    model: MachiLensPredictor,
    flow_window: torch.Tensor,
    context_vec: torch.Tensor,
    samples: int = 64,
) -> dict[str, float]:
    """Monte Carlo dropout prediction for uncertainty-aware scenario outputs."""
    was_training = model.training
    model.train()  # keep dropout active
    preds = []
    with torch.no_grad():
        for _ in range(samples):
            out = model(flow_window, context_vec)
            preds.append(
                torch.stack(
                    [
                        out["prob_success"],
                        out["delta_flow_pred"],
                        out["delta_vacancy_pred"],
                    ],
                    dim=1,
                )
            )
    if not was_training:
        model.eval()
    stacked = torch.stack(preds, dim=0).detach().cpu().numpy()  # (S, B, 3)
    mean = stacked.mean(axis=0)[0]
    std = stacked.std(axis=0)[0]

    risk_score = float(np.clip(0.35 * std[0] + 1.2 * abs(std[1]) + 2.0 * abs(std[2]), 0.0, 1.0))
    return {
        "success_probability_mean": float(mean[0]),
        "success_probability_std": float(std[0]),
        "delta_flow_mean": float(mean[1]),
        "delta_flow_std": float(std[1]),
        "delta_vacancy_mean": float(mean[2]),
        "delta_vacancy_std": float(std[2]),
        "risk_score": risk_score,
    }


def add_intervals(pred: dict[str, float]) -> dict[str, float]:
    """Add approximate 95% intervals from MC-dropout means/stds."""
    result = dict(pred)
    for base in ["success_probability", "delta_flow", "delta_vacancy"]:
        mean = result[f"{base}_mean"]
        std = result[f"{base}_std"]
        result[f"{base}_low95"] = float(mean - 1.96 * std)
        result[f"{base}_high95"] = float(mean + 1.96 * std)
    result["success_probability_low95"] = float(np.clip(result["success_probability_low95"], 0.0, 1.0))
    result["success_probability_high95"] = float(np.clip(result["success_probability_high95"], 0.0, 1.0))
    return result
