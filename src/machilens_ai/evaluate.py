from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .models import MachiLensPredictor


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    success_loss_fn: torch.nn.Module,
    scenario_loss_fn: torch.nn.Module,
    reg_loss_fn: torch.nn.Module,
    success_weight: float = 1.0,
    scenario_weight: float = 0.4,
    regression_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    reg_target = torch.stack([batch["delta_flow"], batch["delta_vacancy"]], dim=1)
    reg_pred = torch.stack([outputs["delta_flow_pred"], outputs["delta_vacancy_pred"]], dim=1)

    loss_success = success_loss_fn(outputs["logit_success"], batch["success_flag"])
    loss_scenario = scenario_loss_fn(outputs["logits_scenario"], batch["scenario_index"])
    loss_reg = reg_loss_fn(reg_pred, reg_target)
    total = success_weight * loss_success + scenario_weight * loss_scenario + regression_weight * loss_reg
    return total, {
        "loss_success": float(loss_success.detach().cpu()),
        "loss_scenario": float(loss_scenario.detach().cpu()),
        "loss_regression": float(loss_reg.detach().cpu()),
        "loss_total": float(total.detach().cpu()),
    }


def evaluate_model(
    model: MachiLensPredictor,
    loader: DataLoader,
    device: str,
    success_weight: float = 1.0,
    scenario_weight: float = 0.4,
    regression_weight: float = 0.5,
) -> dict[str, float]:
    model.eval()
    success_loss_fn = torch.nn.BCEWithLogitsLoss()
    scenario_loss_fn = torch.nn.CrossEntropyLoss()
    reg_loss_fn = torch.nn.MSELoss()

    totals = {"loss_total": 0.0, "loss_success": 0.0, "loss_scenario": 0.0, "loss_regression": 0.0}
    n_batches = 0
    n_correct = 0
    n = 0
    brier_sum = 0.0
    flow_abs = 0.0
    vac_abs = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["flow_window"], batch["context_vec"])
            loss, parts = compute_loss(
                outputs,
                batch,
                success_loss_fn,
                scenario_loss_fn,
                reg_loss_fn,
                success_weight,
                scenario_weight,
                regression_weight,
            )
            for k in totals:
                totals[k] += parts[k]
            probs = outputs["prob_success"]
            pred_flag = (probs >= 0.5).float()
            n_correct += int((pred_flag == batch["success_flag"]).sum().detach().cpu())
            n += int(batch["success_flag"].shape[0])
            brier_sum += float(torch.sum((probs - batch["success_flag"]) ** 2).detach().cpu())
            flow_abs += float(torch.sum(torch.abs(outputs["delta_flow_pred"] - batch["delta_flow"])).detach().cpu())
            vac_abs += float(torch.sum(torch.abs(outputs["delta_vacancy_pred"] - batch["delta_vacancy"])).detach().cpu())
            n_batches += 1

    metrics = {k: v / max(n_batches, 1) for k, v in totals.items()}
    metrics.update(
        {
            "success_accuracy": n_correct / max(n, 1),
            "success_brier": brier_sum / max(n, 1),
            "delta_flow_mae": flow_abs / max(n, 1),
            "delta_vacancy_mae": vac_abs / max(n, 1),
        }
    )
    return metrics
