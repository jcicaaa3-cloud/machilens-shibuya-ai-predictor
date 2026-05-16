from __future__ import annotations

from typing import Literal

import numpy as np

from .config import MachiLensConfig

CITY_FEATURES = ["pop_flow", "vacancy", "tourists", "sns_kpop", "hotel_rate"]
CONTEXT_FEATURES = [
    "scen_A",
    "scen_B",
    "scen_C",
    "night_economy",
    "kpop_fandom",
    "policy_support",
]
SCENARIO_NAMES = ["A", "B", "C"]
SCENARIO_ID = {name: idx for idx, name in enumerate(SCENARIO_NAMES)}

ScenarioName = Literal["A", "B", "C"]


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def scenario_to_context(
    scenario: ScenarioName | str,
    night_economy: float,
    kpop_fandom: float,
    policy_support: float,
) -> np.ndarray:
    """Convert scenario and policy sliders into the 6D MachiLens context vector."""
    scenario = scenario.upper().strip()
    if scenario not in SCENARIO_ID:
        raise ValueError(f"Unknown scenario '{scenario}'. Use one of: {SCENARIO_NAMES}")
    one_hot = np.zeros(3, dtype=np.float32)
    one_hot[SCENARIO_ID[scenario]] = 1.0
    policy = np.array(
        [_clip01(night_economy), _clip01(kpop_fandom), _clip01(policy_support)],
        dtype=np.float32,
    )
    return np.concatenate([one_hot, policy], axis=0)


def build_default_recent_window(cfg: MachiLensConfig | None = None) -> np.ndarray:
    """Create a deterministic recent-Shibuya-like 24-step feature window.

    The values are normalized indices, not official statistics. This function exists so the
    CLI and dashboard can run before real open-data integration.
    """
    cfg = cfg or MachiLensConfig()
    t = np.linspace(0.0, 1.0, cfg.window_steps, dtype=np.float32)
    pandemic_recovery = 1.0 - np.exp(-3.0 * t)
    seasonality = 0.03 * np.sin(2 * np.pi * np.arange(cfg.window_steps) / 12.0)

    pop_flow = 0.54 + 0.18 * t + 0.06 * pandemic_recovery + seasonality
    vacancy = 0.36 - 0.09 * t - 0.03 * pandemic_recovery + 0.015 * np.cos(2 * np.pi * t)
    tourists = 0.42 + 0.27 * t + 0.08 * pandemic_recovery + seasonality
    sns_kpop = 0.30 + 0.18 * t + 0.05 * np.sin(3 * np.pi * t)
    hotel_rate = 0.48 + 0.22 * t + 0.04 * pandemic_recovery

    window = np.stack([pop_flow, vacancy, tourists, sns_kpop, hotel_rate], axis=1)
    return np.clip(window, 0.0, 1.0).astype(np.float32)


def flatten_window_for_baseline(flow_seq: np.ndarray, context: np.ndarray) -> np.ndarray:
    """Flatten time-series and concatenate context for tabular baselines."""
    if flow_seq.ndim != 3:
        raise ValueError("flow_seq must have shape (N, T, D)")
    return np.concatenate([flow_seq.reshape(flow_seq.shape[0], -1), context], axis=1)
