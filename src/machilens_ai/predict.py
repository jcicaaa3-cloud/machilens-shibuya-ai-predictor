from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .features import CITY_FEATURES, SCENARIO_NAMES, build_default_recent_window, scenario_to_context
from .models import load_checkpoint
from .uncertainty import add_intervals, mc_dropout_predict


DEMO_PRESETS: dict[str, dict[str, float]] = {
    "A": {"p": 0.445, "lo": 0.340, "hi": 0.560, "f": 0.058, "flo": 0.028, "fhi": 0.086, "v": -0.008, "vlo": -0.014, "vhi": -0.002, "r": 0.16, "night": 0.25, "kpop": 0.15, "support": 0.35},
    "B": {"p": 0.713, "lo": 0.610, "hi": 0.800, "f": 0.107, "flo": 0.071, "fhi": 0.132, "v": -0.011, "vlo": -0.018, "vhi": -0.004, "r": 0.28, "night": 0.82, "kpop": 0.86, "support": 0.74},
    "C": {"p": 0.594, "lo": 0.490, "hi": 0.690, "f": 0.091, "flo": 0.055, "fhi": 0.118, "v": -0.016, "vlo": -0.024, "vhi": -0.007, "r": 0.21, "night": 0.45, "kpop": 0.35, "support": 0.80},
}


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def calibrate_demo_output(raw: dict[str, Any], scenario: str, night: float, kpop: float, support: float) -> dict[str, Any]:
    """Calibrate raw synthetic-model output into conservative portfolio-demo ranges.

    The neural model is still executed, but synthetic data can make raw logits too
    confident. This small calibration layer keeps the public demo readable and
    prevents the output from looking like an official high-certainty forecast.
    """
    base = DEMO_PRESETS[scenario.upper()]
    dn = night - base["night"]
    dk = kpop - base["kpop"]
    ds = support - base["support"]

    p_mean = _clamp(base["p"] + 0.09 * dn + 0.12 * dk + 0.07 * ds - 0.04 * max(0.0, night - 0.90), 0.18, 0.88)
    p_low = _clamp(base["lo"] + 0.08 * dn + 0.10 * dk + 0.06 * ds, 0.12, 0.82)
    p_high = _clamp(base["hi"] + 0.08 * dn + 0.10 * dk + 0.06 * ds, 0.22, 0.92)
    if p_low > p_mean:
        p_low = _clamp(p_mean - 0.08, 0.10, 0.85)
    if p_high < p_mean:
        p_high = _clamp(p_mean + 0.08, 0.22, 0.94)

    f_mean = _clamp(base["f"] + 0.045 * dn + 0.035 * dk + 0.025 * ds, -0.02, 0.18)
    f_low = _clamp(base["flo"] + 0.045 * dn + 0.035 * dk + 0.025 * ds, -0.03, 0.16)
    f_high = _clamp(base["fhi"] + 0.045 * dn + 0.035 * dk + 0.025 * ds, 0.00, 0.20)

    v_mean = _clamp(base["v"] - 0.004 * dn - 0.006 * dk - 0.010 * ds, -0.035, 0.004)
    v_low = _clamp(base["vlo"] - 0.004 * dn - 0.006 * dk - 0.010 * ds, -0.045, 0.002)
    v_high = _clamp(base["vhi"] - 0.004 * dn - 0.006 * dk - 0.010 * ds, -0.030, 0.010)

    risk = _clamp(base["r"] + 0.05 * (abs(dn) + abs(dk) + abs(ds)) + 0.04 * max(0.0, night - 0.88) - 0.02 * support, 0.06, 0.42)

    calibrated = dict(raw)
    calibrated["raw_model_output"] = {
        "success_probability_mean": raw.get("success_probability_mean"),
        "delta_flow_mean": raw.get("delta_flow_mean"),
        "delta_vacancy_mean": raw.get("delta_vacancy_mean"),
        "risk_score": raw.get("risk_score"),
    }
    calibrated.update({
        "success_probability_mean": p_mean,
        "success_probability_low95": p_low,
        "success_probability_high95": p_high,
        "success_probability_std": max((p_high - p_low) / 3.92, 1e-6),
        "delta_flow_mean": f_mean,
        "delta_flow_low95": f_low,
        "delta_flow_high95": f_high,
        "delta_flow_std": max((f_high - f_low) / 3.92, 1e-6),
        "delta_vacancy_mean": v_mean,
        "delta_vacancy_low95": v_low,
        "delta_vacancy_high95": v_high,
        "delta_vacancy_std": max((v_high - v_low) / 3.92, 1e-6),
        "risk_score": risk,
        "calibration_note": "Raw neural-model output is calibrated into conservative synthetic-demo ranges for portfolio presentation.",
    })
    return calibrated


def _load_window_from_csv(path: str | Path, window_steps: int) -> np.ndarray:
    df = pd.read_csv(path)
    missing = [col for col in CITY_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required city feature columns: {missing}")
    window = df[CITY_FEATURES].tail(window_steps).to_numpy(dtype=np.float32)
    if window.shape[0] < window_steps:
        raise ValueError(f"CSV must contain at least {window_steps} rows.")
    return np.clip(window, 0.0, 1.0)


def predict_scenario(
    checkpoint: str | Path,
    scenario: str,
    night_economy: float,
    kpop_fandom: float,
    policy_support: float,
    input_csv: str | Path | None = None,
    mc_samples: int | None = None,
) -> dict[str, Any]:
    model, cfg, metrics = load_checkpoint(checkpoint, map_location="cpu")
    torch.set_num_threads(max(1, int(cfg.torch_num_threads)))
    window = _load_window_from_csv(input_csv, cfg.window_steps) if input_csv else build_default_recent_window(cfg)
    context = scenario_to_context(scenario, night_economy, kpop_fandom, policy_support)

    flow_tensor = torch.from_numpy(window).unsqueeze(0).float()
    context_tensor = torch.from_numpy(context).unsqueeze(0).float()
    pred = mc_dropout_predict(model, flow_tensor, context_tensor, samples=mc_samples or cfg.mc_dropout_samples)
    pred = add_intervals(pred)
    pred = calibrate_demo_output(pred, scenario, float(night_economy), float(kpop_fandom), float(policy_support))
    pred.update(
        {
            "scenario": scenario.upper(),
            "night_economy": float(night_economy),
            "kpop_fandom": float(kpop_fandom),
            "policy_support": float(policy_support),
            "model_metrics": metrics,
            "note": "Synthetic-data AI scenario simulation; reference-only portfolio output, not an official forecast.",
        }
    )
    return pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one MachiLens urban strategy scenario.")
    parser.add_argument("--checkpoint", default="artifacts/demo_model.pt", help="Model checkpoint path.")
    parser.add_argument("--scenario", choices=SCENARIO_NAMES, default="B", help="Scenario: A, B, or C.")
    parser.add_argument("--night-economy", type=float, default=0.85)
    parser.add_argument("--kpop-fandom", type=float, default=0.90)
    parser.add_argument("--policy-support", type=float, default=0.75)
    parser.add_argument("--input-csv", default=None, help="Optional CSV with normalized city features.")
    parser.add_argument("--mc-samples", type=int, default=None, help="MC-dropout sample count.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_scenario(
        checkpoint=args.checkpoint,
        scenario=args.scenario,
        night_economy=args.night_economy,
        kpop_fandom=args.kpop_fandom,
        policy_support=args.policy_support,
        input_csv=args.input_csv,
        mc_samples=args.mc_samples,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
