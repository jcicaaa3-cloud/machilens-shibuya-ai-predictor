from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from machilens_ai.predict import predict_scenario  # noqa: E402

st.set_page_config(page_title="MachiLens AI", page_icon="🌆", layout="wide")
st.title("MachiLens AI — Shibuya Scenario Forecast Lab")
st.caption("시부야 도시전략 시나리오를 비교하는 AI 포트폴리오 데모입니다. 합성 데이터 기반 참고용 화면입니다.")
st.warning("참고용 포트폴리오 데모입니다. 실제 정책, 투자, 행정, 부동산 판단에 사용하면 안 됩니다.")

checkpoint = ROOT / "artifacts" / "demo_model.pt"
if not checkpoint.exists():
    st.warning("No checkpoint found. Run `PYTHONPATH=src python -m machilens_ai.train --epochs 3 --n-samples 700` first.")
    st.stop()

left, right = st.columns([1, 2])
with left:
    scenario = st.selectbox(
        "Scenario",
        options=["A", "B", "C"],
        format_func=lambda x: {
            "A": "A — Minimal intervention",
            "B": "B — K-culture + night economy",
            "C": "C — Community-oriented",
        }[x],
        index=1,
    )
    night = st.slider("Night economy intensity", 0.0, 1.0, 0.85 if scenario == "B" else 0.45, 0.05)
    kpop = st.slider("K-culture / fandom activation", 0.0, 1.0, 0.90 if scenario == "B" else 0.35, 0.05)
    support = st.slider("Policy support", 0.0, 1.0, 0.75 if scenario == "B" else 0.70, 0.05)
    mc_samples = st.slider("Uncertainty samples", 8, 64, 16, 8)
    run = st.button("Run AI prediction", type="primary")

if run:
    result = predict_scenario(
        checkpoint=checkpoint,
        scenario=scenario,
        night_economy=night,
        kpop_fandom=kpop,
        policy_support=support,
        mc_samples=mc_samples,
    )
    with right:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Success probability", f"{result['success_probability_mean'] * 100:.1f}%")
        c2.metric("Flow change", f"{result['delta_flow_mean'] * 100:.1f}%")
        c3.metric("Vacancy change", f"{result['delta_vacancy_mean'] * 100:.2f}%p")
        c4.metric("Risk score", f"{result['risk_score']:.2f}")

        interval_df = pd.DataFrame(
            [
                {
                    "metric": "success probability",
                    "mean": result["success_probability_mean"],
                    "low95": result["success_probability_low95"],
                    "high95": result["success_probability_high95"],
                },
                {
                    "metric": "delta flow",
                    "mean": result["delta_flow_mean"],
                    "low95": result["delta_flow_low95"],
                    "high95": result["delta_flow_high95"],
                },
                {
                    "metric": "delta vacancy",
                    "mean": result["delta_vacancy_mean"],
                    "low95": result["delta_vacancy_low95"],
                    "high95": result["delta_vacancy_high95"],
                },
            ]
        )
        st.subheader("Uncertainty-aware output")
        st.dataframe(interval_df, use_container_width=True)
        st.info("Use these values for scenario comparison only. They are generated from synthetic assumptions.")
else:
    with right:
        st.write("Choose scenario inputs and click **Run AI prediction**.")
        st.markdown(
            """
            **Outputs**
            - success probability
            - expected pedestrian flow change
            - expected vacancy-rate change
            - MC-dropout uncertainty/risk score
            """
        )
