from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from .config import MachiLensConfig, set_seed
from .features import CITY_FEATURES, CONTEXT_FEATURES, SCENARIO_NAMES, scenario_to_context


@dataclass
class MachiLensGeneratedData:
    flow_seq: np.ndarray
    context: np.ndarray
    success_prob: np.ndarray
    success_flag: np.ndarray
    scenario_index: np.ndarray
    delta_flow: np.ndarray
    delta_vacancy: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        """Create a compact, human-readable frame for inspection/reporting."""
        rows: list[dict[str, Any]] = []
        last = self.flow_seq[:, -1, :]
        for i in range(self.flow_seq.shape[0]):
            row: dict[str, Any] = {f"last_{name}": float(last[i, j]) for j, name in enumerate(CITY_FEATURES)}
            row.update({name: float(self.context[i, j]) for j, name in enumerate(CONTEXT_FEATURES)})
            row["scenario"] = SCENARIO_NAMES[int(self.scenario_index[i])]
            row["success_prob_target"] = float(self.success_prob[i])
            row["success_flag"] = int(self.success_flag[i])
            row["delta_flow_target"] = float(self.delta_flow[i])
            row["delta_vacancy_target"] = float(self.delta_vacancy[i])
            rows.append(row)
        return pd.DataFrame(rows)


class ShibuyaMachiLensDataset(Dataset):
    def __init__(self, data: MachiLensGeneratedData):
        self.data = data

    def __len__(self) -> int:
        return int(self.data.flow_seq.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "flow_window": torch.from_numpy(self.data.flow_seq[idx]).float(),
            "context_vec": torch.from_numpy(self.data.context[idx]).float(),
            "success_prob_target": torch.tensor(self.data.success_prob[idx]).float(),
            "success_flag": torch.tensor(self.data.success_flag[idx]).float(),
            "scenario_index": torch.tensor(self.data.scenario_index[idx]).long(),
            "delta_flow": torch.tensor(self.data.delta_flow[idx]).float(),
            "delta_vacancy": torch.tensor(self.data.delta_vacancy[idx]).float(),
        }


def _scenario_dependent_policy(rng: np.random.Generator, scenario_idx: int) -> np.ndarray:
    """Generate policy sliders with different priors for A/B/C."""
    if scenario_idx == 0:  # A: minimal intervention
        night = rng.beta(2.0, 5.0)
        kpop = rng.beta(1.5, 6.0)
        support = rng.beta(2.5, 4.5)
    elif scenario_idx == 1:  # B: K-culture + night economy
        night = rng.beta(5.0, 2.0)
        kpop = rng.beta(6.0, 1.8)
        support = rng.beta(4.0, 2.2)
    else:  # C: community-oriented
        night = rng.beta(3.0, 3.5)
        kpop = rng.beta(2.4, 4.2)
        support = rng.beta(5.0, 2.0)
    return np.array([night, kpop, support], dtype=np.float32)


def _make_city_window(
    rng: np.random.Generator,
    cfg: MachiLensConfig,
    scenario_idx: int,
    policy: np.ndarray,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, cfg.window_steps, dtype=np.float32)
    seasonal = 0.025 * np.sin(2 * np.pi * np.arange(cfg.window_steps) / 12.0 + rng.uniform(-0.4, 0.4))
    pandemic_dip = -0.11 * np.exp(-0.5 * ((t - 0.36) / 0.10) ** 2)
    recovery = 1.0 - np.exp(-3.2 * t)

    b_boost = 1.0 if scenario_idx == 1 else 0.0
    c_boost = 1.0 if scenario_idx == 2 else 0.0

    base_pop = rng.uniform(0.49, 0.61)
    base_vacancy = rng.uniform(0.30, 0.44)
    base_tourists = rng.uniform(0.36, 0.50)
    base_kpop = rng.uniform(0.20, 0.35)
    base_hotel = rng.uniform(0.42, 0.56)

    night, kpop, support = policy
    pop_flow = (
        base_pop
        + 0.17 * t
        + 0.035 * night
        + 0.030 * support
        + 0.025 * b_boost
        + 0.012 * c_boost
        + pandemic_dip
        + 0.040 * recovery
        + seasonal
    )
    vacancy = (
        base_vacancy
        - 0.10 * t
        - 0.020 * support
        - 0.015 * b_boost
        - 0.020 * c_boost
        - 0.010 * night
        + 0.012 * rng.normal(size=cfg.window_steps)
    )
    tourists = (
        base_tourists
        + 0.22 * t
        + 0.050 * night
        + 0.040 * kpop
        + 0.035 * b_boost
        + pandemic_dip * 1.15
        + 0.070 * recovery
        + seasonal
    )
    sns_kpop = (
        base_kpop
        + 0.16 * t
        + 0.095 * kpop
        + 0.055 * b_boost
        + 0.020 * np.sin(3 * np.pi * t + rng.uniform(-0.2, 0.2))
    )
    hotel_rate = (
        base_hotel
        + 0.16 * t
        + 0.045 * tourists
        + 0.025 * night
        + 0.035 * b_boost
        + pandemic_dip
        + 0.035 * recovery
    )

    noise = rng.normal(0.0, 0.012, size=(cfg.window_steps, cfg.city_dim))
    window = np.stack([pop_flow, vacancy, tourists, sns_kpop, hotel_rate], axis=1) + noise
    return np.clip(window, 0.0, 1.0).astype(np.float32)


def generate_synthetic_dataset(cfg: MachiLensConfig | None = None) -> MachiLensGeneratedData:
    """Generate synthetic urban time-series for model development.

    This synthetic generator is intentionally explicit. It lets the project demonstrate
    AI forecasting mechanics while making every assumption inspectable.
    """
    cfg = cfg or MachiLensConfig()
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    flow_seq = np.zeros((cfg.n_samples, cfg.window_steps, cfg.city_dim), dtype=np.float32)
    context = np.zeros((cfg.n_samples, cfg.context_dim), dtype=np.float32)
    scenario_index = rng.choice(np.arange(cfg.scenario_kinds), size=cfg.n_samples, p=[0.34, 0.33, 0.33])

    success_prob = np.zeros(cfg.n_samples, dtype=np.float32)
    success_flag = np.zeros(cfg.n_samples, dtype=np.float32)
    delta_flow = np.zeros(cfg.n_samples, dtype=np.float32)
    delta_vacancy = np.zeros(cfg.n_samples, dtype=np.float32)

    for i, scen_idx in enumerate(scenario_index):
        policy = _scenario_dependent_policy(rng, int(scen_idx))
        scenario_name = SCENARIO_NAMES[int(scen_idx)]
        context[i] = scenario_to_context(scenario_name, *policy)
        flow_seq[i] = _make_city_window(rng, cfg, int(scen_idx), policy)

        last = flow_seq[i, -1, :]
        momentum = flow_seq[i, -1, 0] - flow_seq[i, 0, 0]
        tourism_momentum = flow_seq[i, -1, 2] - flow_seq[i, 0, 2]
        pop, vacancy, tourists, sns_kpop, hotel_rate = last
        scen_a, scen_b, scen_c = context[i, :3]
        night, kpop, support = policy

        # Moderate, deliberately non-deterministic success rule.
        # The coefficients are kept conservative so the demo does not imply
        # unrealistic certainty from synthetic data.
        logit = (
            -1.20
            + 0.45 * pop
            - 0.65 * vacancy
            + 0.30 * tourists
            + 0.35 * sns_kpop
            + 0.25 * hotel_rate
            + 0.25 * night
            + 0.35 * kpop
            + 0.30 * support
            - 0.25 * scen_a
            + 0.45 * scen_b
            + 0.10 * scen_c
            + 0.20 * scen_b * night * kpop
            + 0.20 * scen_c * support
            + 0.20 * momentum
            + rng.normal(0.0, 0.20)
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        success_prob[i] = np.float32(np.clip(prob, 0.02, 0.98))
        success_flag[i] = np.float32(rng.binomial(1, success_prob[i]))

        delta_flow[i] = np.float32(
            np.clip(
                0.005
                + 0.045 * success_prob[i]
                + 0.030 * night
                + 0.025 * kpop
                + 0.020 * support
                + 0.025 * scen_b
                + 0.015 * scen_c
                + 0.035 * tourism_momentum
                - 0.030 * vacancy
                + rng.normal(0.0, 0.018),
                -0.04,
                0.18,
            )
        )
        delta_vacancy[i] = np.float32(
            np.clip(
                -0.002
                - 0.012 * success_prob[i]
                - 0.010 * support
                - 0.004 * night
                - 0.004 * scen_b
                - 0.008 * scen_c
                + 0.020 * vacancy
                + rng.normal(0.0, 0.008),
                -0.04,
                0.02,
            )
        )

    return MachiLensGeneratedData(
        flow_seq=flow_seq,
        context=context,
        success_prob=success_prob,
        success_flag=success_flag,
        scenario_index=scenario_index.astype(np.int64),
        delta_flow=delta_flow,
        delta_vacancy=delta_vacancy,
    )


def make_dataloaders(cfg: MachiLensConfig, data: MachiLensGeneratedData | None = None) -> tuple[DataLoader, DataLoader, DataLoader]:
    data = data or generate_synthetic_dataset(cfg)
    dataset = ShibuyaMachiLensDataset(data)
    n_total = len(dataset)
    n_train = int(n_total * cfg.train_ratio)
    n_val = int(n_total * cfg.val_ratio)
    n_test = n_total - n_train - n_val
    generator = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)
    return (
        DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False),
    )


def save_synthetic_csv(data: MachiLensGeneratedData, path: str | Path, n_rows: int = 500) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_frame().head(n_rows).to_csv(path, index=False)
