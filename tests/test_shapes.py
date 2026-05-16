from __future__ import annotations

import torch

from machilens_ai.config import MachiLensConfig
from machilens_ai.data import generate_synthetic_dataset, ShibuyaMachiLensDataset
from machilens_ai.features import scenario_to_context
from machilens_ai.models import MachiLensPredictor


def test_context_shape() -> None:
    ctx = scenario_to_context("B", 0.8, 0.9, 0.7)
    assert ctx.shape == (6,)
    assert ctx[1] == 1.0


def test_dataset_shapes() -> None:
    cfg = MachiLensConfig(n_samples=16, window_steps=12)
    data = generate_synthetic_dataset(cfg)
    ds = ShibuyaMachiLensDataset(data)
    item = ds[0]
    assert item["flow_window"].shape == (12, 5)
    assert item["context_vec"].shape == (6,)


def test_model_forward_shapes() -> None:
    cfg = MachiLensConfig(n_samples=8, window_steps=12)
    model = MachiLensPredictor(cfg)
    flow = torch.rand(4, cfg.window_steps, cfg.city_dim)
    context = torch.rand(4, cfg.context_dim)
    out = model(flow, context)
    assert out["prob_success"].shape == (4,)
    assert out["logits_scenario"].shape == (4, 3)
    assert out["delta_flow_pred"].shape == (4,)
    assert out["delta_vacancy_pred"].shape == (4,)
    assert out["attention_weights"].shape == (4, cfg.window_steps)
