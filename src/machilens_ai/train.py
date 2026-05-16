from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .baselines import run_baselines
from .config import load_config, set_seed
from .data import generate_synthetic_dataset, make_dataloaders, save_synthetic_csv
from .evaluate import compute_loss, evaluate_model
from .models import MachiLensPredictor, save_checkpoint


def train_model(args: argparse.Namespace) -> dict[str, float]:
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.n_samples is not None:
        cfg.n_samples = args.n_samples
    if args.artifact_dir is not None:
        cfg.artifact_dir = args.artifact_dir
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    torch.set_num_threads(max(1, int(cfg.torch_num_threads)))
    set_seed(cfg.seed)
    device = cfg.device
    data = generate_synthetic_dataset(cfg)
    train_loader, val_loader, test_loader = make_dataloaders(cfg, data)

    output_dir = Path(cfg.output_dir)
    artifact_dir = Path(cfg.artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_synthetic_csv(data, Path("data/synthetic/synthetic_machilens_samples.csv"), n_rows=min(750, cfg.n_samples))

    model = MachiLensPredictor(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    success_loss_fn = torch.nn.BCEWithLogitsLoss()
    scenario_loss_fn = torch.nn.CrossEntropyLoss()
    reg_loss_fn = torch.nn.MSELoss()

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_path = artifact_dir / "demo_model.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["flow_window"], batch["context_vec"])
            loss, _ = compute_loss(
                outputs,
                batch,
                success_loss_fn,
                scenario_loss_fn,
                reg_loss_fn,
                cfg.loss_success_weight,
                cfg.loss_scenario_weight,
                cfg.loss_regression_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += float(loss.detach().cpu())
            train_batches += 1
        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            cfg.loss_success_weight,
            cfg.loss_scenario_weight,
            cfg.loss_regression_weight,
        )
        row = {"epoch": epoch, "train_loss": train_loss / max(train_batches, 1), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if val_metrics["loss_total"] < best_val:
            best_val = val_metrics["loss_total"]
            save_checkpoint(model, cfg, best_path, metrics={"best_val_loss": best_val})

    # Reload best model for final test metrics.
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model_state_dict"])
    test_metrics = evaluate_model(
        model,
        test_loader,
        device,
        cfg.loss_success_weight,
        cfg.loss_scenario_weight,
        cfg.loss_regression_weight,
    )
    metrics = {"best_val_loss": best_val, **{f"test_{k}": v for k, v in test_metrics.items()}}

    if args.run_baselines:
        baseline_metrics = run_baselines(data, cfg, output_dir / "baseline_metrics.json")
        metrics.update(baseline_metrics)

    (output_dir / "demo_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_checkpoint(model, cfg, best_path, metrics=metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MachiLens AI predictor.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--n-samples", type=int, default=None, help="Override synthetic sample count.")
    parser.add_argument("--artifact-dir", default=None, help="Override artifact directory.")
    parser.add_argument("--output-dir", default=None, help="Override report/output directory.")
    parser.add_argument("--run-baselines", action="store_true", help="Run random-forest baselines.")
    return parser.parse_args()


if __name__ == "__main__":
    train_model(parse_args())
