from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error
from sklearn.model_selection import train_test_split

from .config import MachiLensConfig
from .data import MachiLensGeneratedData
from .features import flatten_window_for_baseline


def run_baselines(data: MachiLensGeneratedData, cfg: MachiLensConfig, output_path: str | Path | None = None) -> dict[str, float]:
    """Train simple random-forest baselines for comparison."""
    x = flatten_window_for_baseline(data.flow_seq, data.context)
    y_class = data.success_flag
    y_reg = np.stack([data.delta_flow, data.delta_vacancy], axis=1)
    x_train, x_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        x, y_class, y_reg, test_size=0.20, random_state=cfg.seed, stratify=y_class
    )
    clf = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=cfg.seed)
    reg = RandomForestRegressor(n_estimators=120, max_depth=8, random_state=cfg.seed)
    clf.fit(x_train, yc_train)
    reg.fit(x_train, yr_train)
    probs = clf.predict_proba(x_test)[:, 1]
    pred_flag = (probs >= 0.5).astype(float)
    reg_pred = reg.predict(x_test)
    metrics = {
        "rf_success_accuracy": float(accuracy_score(yc_test, pred_flag)),
        "rf_success_brier": float(brier_score_loss(yc_test, probs)),
        "rf_delta_flow_mae": float(mean_absolute_error(yr_test[:, 0], reg_pred[:, 0])),
        "rf_delta_vacancy_mae": float(mean_absolute_error(yr_test[:, 1], reg_pred[:, 1])),
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
