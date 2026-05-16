from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("$", " ".join(command))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, cwd=ROOT, check=True, env=env)


def main() -> None:
    # Train a small demo checkpoint quickly.
    run([
        sys.executable,
        "-m",
        "machilens_ai.train",
        "--config",
        "configs/base.yaml",
        "--epochs",
        "3",
        "--n-samples",
        "700",
        "--run-baselines",
    ])

    presets = [
        ("A", 0.25, 0.15, 0.35),
        ("B", 0.82, 0.86, 0.74),
        ("C", 0.45, 0.35, 0.80),
    ]
    rows = []
    for scenario, night, kpop, support in presets:
        out_path = ROOT / "reports" / f"prediction_{scenario}.json"
        run([
            sys.executable,
            "-m",
            "machilens_ai.predict",
            "--checkpoint",
            "artifacts/demo_model.pt",
            "--scenario",
            scenario,
            "--night-economy",
            str(night),
            "--kpop-fandom",
            str(kpop),
            "--policy-support",
            str(support),
            "--mc-samples",
            "16",
            "--output",
            str(out_path),
        ])
        rows.append(json.loads(out_path.read_text(encoding="utf-8")))
    df = pd.DataFrame(rows)
    keep = [
        "scenario",
        "success_probability_mean",
        "success_probability_low95",
        "success_probability_high95",
        "delta_flow_mean",
        "delta_vacancy_mean",
        "risk_score",
    ]
    df[keep].to_csv(ROOT / "reports" / "demo_predictions.csv", index=False)
    print(df[keep].to_string(index=False))


if __name__ == "__main__":
    main()
