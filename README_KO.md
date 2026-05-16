# MachiLens AI — Shibuya Scenario Forecast Lab

**MachiLens AI**는 시부야 도시전략 시나리오를 비교해 보기 위한 작은 AI 예측 데모입니다.

처음 아이디어는 대학 수업에서 시부야 도시공간 재편을 다루며 떠올린 기획이었습니다. 당시에는 보고서와 계획 단계에서 멈췄고, 실제로 작동하는 서비스나 데모까지 만들지는 못했습니다. 이번 버전은 그 아이디어를 다시 꺼내서, 합성 데이터 생성 → 모델 학습 → 예측 결과 → 브라우저 데모까지 이어지는 형태로 정리한 리빌드입니다.

이 프로젝트는 “시부야의 미래를 정확히 맞힌다”는 식의 예측기가 아닙니다. 관광 회복, 야간경제, K-컬처, 공실률 개선, 로컬 커뮤니티 같은 도시전략 키워드를 모델이 읽을 수 있는 입력값으로 바꾸고, 시나리오별 결과를 비교하는 방법을 보여주는 프로토타입입니다.

## Live HTML demo

브라우저에서 바로 열 수 있는 정적 데모가 포함되어 있습니다.

```text
index.html
demo.html
docs/index.html
```

GitHub Pages를 켜면 `index.html`이 바로 데모 페이지로 열립니다.

## 참고용 안내

이 저장소의 예측값과 HTML 데모는 **방법론 설명과 포트폴리오 검토를 위한 참고용**입니다. 실제 정책, 행정, 도시계획, 투자, 부동산, 상권 의사결정에 사용하면 안 됩니다.

공개 패키지의 데이터는 코드로 생성한 **합성 데이터**입니다. 외부 지도 이미지, 상용 리포트, 유료 데이터, 민간 원자료, 개인정보가 포함된 원본 문서는 포함하지 않았습니다.

## 이 프로젝트에서 AI / 머신러닝 / 딥러닝은 무엇인가

- **AI**: 도시전략 가정을 입력으로 넣고, 결과를 비교해 보는 의사결정 보조 흐름입니다.
- **머신러닝**: 합성 도시 시계열과 정책 벡터를 학습 데이터로 사용해 입력과 결과 사이의 관계를 학습합니다.
- **딥러닝**: PyTorch 기반 BiLSTM + Attention 모델이 시간에 따라 변하는 도시 지표를 읽고, MLP가 정책 벡터와 결합해 결과를 예측합니다.

## What it does

MachiLens compares three Shibuya strategy scenarios:

| Scenario | Meaning |
|---|---|
| A | Minimal intervention / 현재 흐름을 크게 바꾸지 않는 전략 |
| B | K-culture + night economy / 콘텐츠·관광·야간 체류 강화 |
| C | Community-oriented / 로컬 커뮤니티와 생활 밀도 강화 |

The demo predicts:

- strategy success probability
- expected pedestrian-flow change
- expected vacancy-rate change
- simple risk score
- uncertainty-style output range

## Model architecture

The model is implemented in PyTorch.

```text
1. Synthetic data generator
   Generates normalized urban time-series windows.

2. Feature builder
   Converts A/B/C scenario and policy sliders into a 6D context vector.

3. BiLSTM encoder
   Reads recent urban indicator patterns.

4. Temporal attention
   Highlights important time steps in the sequence.

5. Fusion MLP
   Combines city trend representation with scenario/policy context.

6. Multi-task heads
   - success probability classification
   - scenario classification
   - pedestrian-flow / vacancy-rate regression
```

## Repository structure

```text
machilens-shibuya-ai-predictor/
├── index.html                    # static browser demo
├── demo.html                     # same demo, direct-open version
├── docs/index.html                # GitHub Pages entry
├── src/machilens_ai/              # PyTorch model and data pipeline
├── app/streamlit_app.py           # optional Streamlit dashboard
├── data/synthetic/                # generated synthetic sample data
├── reports/                       # demo prediction outputs and notes
├── scripts/                       # demo runner and package checker
├── tests/                         # pytest checks
└── README_KO.md                   # Korean README
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

Train a demo model:

```bash
PYTHONPATH=src python -m machilens_ai.train --epochs 3 --n-samples 900
```

Predict one scenario:

```bash
PYTHONPATH=src python -m machilens_ai.predict   --scenario B   --night-economy 0.82   --kpop-fandom 0.86   --policy-support 0.74
```

Run Streamlit:

```bash
streamlit run app/streamlit_app.py
```

## GitHub Pages

After pushing the repository:

```text
Settings → Pages → Deploy from a branch → main → /root → Save
```

Recommended repository name:

```text
machilens-shibuya-ai-predictor
```

## Limitations

- The current data is synthetic and should not be interpreted as official Shibuya data.
- The model is a portfolio prototype, not a production forecasting system.
- Scenario outputs depend on the assumptions built into the generator.
- Real deployment would require verified public/private datasets, geospatial granularity, validation, and expert review.
- Results should be used for scenario exploration only.

## License

MIT License.
