# MachiLens

[English](README.md) · [日本語](README_JA.md)

시부야 도시전략 시나리오를 비교하는 작은 모델입니다. 학습에는 합성 데이터를 썼습니다.

시작은 시부야 재개발을 다룬 대학 수업이었습니다. 그때는 기획 보고서로 끝났고, 실제로 돌아가는 건 만들지 못했습니다. 이 레포는 그걸 다시 만든 것으로, 합성 데이터 생성기, PyTorch 모델, 예측 스크립트, 브라우저 데모가 들어 있습니다.

실제 시부야를 예측하는 모델이 아닙니다. 데이터는 전부 코드로 만든 것이고, 결과를 정책·투자·부동산 판단에 쓰면 안 됩니다.

데모: https://jcicaaa3-cloud.github.io/machilens-shibuya-ai-predictor/

## 시나리오

| 시나리오 | 내용 |
| --- | --- |
| A | 최소 개입 |
| B | K-컬처와 야간경제 |
| C | 로컬 커뮤니티 |

입력은 시나리오와 슬라이더 3개(야간경제, K-pop 팬덤, 정책 지원)입니다. 출력은 성공 확률, 유동인구 변화, 공실률 변화, 리스크 점수입니다.

## 모델

- 도시 지표 5개, 24스텝 시계열을 읽는 BiLSTM + 시간축 어텐션
- 시계열 인코딩과 6차원 시나리오·정책 벡터를 합치는 MLP
- 헤드 3개: 성공 분류, 시나리오 분류, 유동인구·공실률 변화 회귀
- MC dropout(16회 샘플링)으로 출력 범위 계산

코드는 `src/machilens_ai/`에 있습니다.

## 결과

레포에 들어 있는 데모 실행(합성 샘플 700개, 3 에폭) 결과입니다. 파일: `reports/demo_metrics.json`, `reports/baseline_metrics.json`.

| 모델 | 성공 정확도 | Brier | 유동인구 MAE | 공실률 MAE |
| --- | ---: | ---: | ---: | ---: |
| BiLSTM + 어텐션 | 0.60 | 0.248 | 0.020 | 0.043 |
| 랜덤 포레스트(베이스라인) | 0.85 | 0.117 | 0.017 | 0.010 |

이 실행에서는 모든 지표에서 랜덤 포레스트 베이스라인이 신경망 모델보다 좋습니다.

## 데모 숫자에 대해

브라우저 데모와 `reports/prediction_*.json`의 값은 신경망의 원래 출력이 아닙니다. `predict.py`가 시나리오별 프리셋 값으로 바꾸고 슬라이더 값에 따라 선형으로 조정합니다(`calibrate_demo_output`). 브라우저 데모도 같은 식을 JavaScript로 계산할 뿐 모델을 돌리지 않습니다. 신경망의 원래 출력은 각 예측 파일의 `raw_model_output`에 남겨 뒀습니다.

## 실행

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

PYTHONPATH=src pytest
PYTHONPATH=src python -m machilens_ai.train --epochs 3 --n-samples 900
PYTHONPATH=src python -m machilens_ai.predict --scenario B --night-economy 0.82 --kpop-fandom 0.86 --policy-support 0.74
streamlit run app/streamlit_app.py   # 선택
```

## 한계

- 데이터가 합성입니다. 결과는 생성 코드에 넣은 가정에 따라 달라집니다.
- 블록 단위 지리 정보가 없고, 도시를 집계 지표 5개로만 표현합니다.
- 실제로 쓰려면 검증된 데이터와 실제 결과 대조가 필요합니다.

## 라이선스

MIT
