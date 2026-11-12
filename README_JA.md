# MachiLens

[English](README.md) · [한국어](README_KO.md)

渋谷の都市戦略シナリオを比べるための小さなモデルです。学習には合成データを使っています。

きっかけは、渋谷の再開発を扱った大学の授業でした。そのときは企画レポートで終わり、実際に動くものは作れませんでした。このリポジトリはその作り直しで、合成データの生成、PyTorchモデル、予測スクリプト、ブラウザデモまでをまとめています。

実際の渋谷を予測するものではありません。データはすべてコードで生成したもので、出力を政策・投資・不動産の判断に使わないでください。

デモ：https://jcicaaa3-cloud.github.io/machilens-shibuya-ai-predictor/

## シナリオ

| シナリオ | 内容 |
| --- | --- |
| A | 最小限の介入 |
| B | K-カルチャーとナイトタイムエコノミー |
| C | 地域コミュニティ |

入力はシナリオと3つのスライダー（ナイトタイムエコノミー、K-POPファンダム、政策支援）です。出力は成功確率、歩行者流動の変化、空室率の変化、リスクスコアです。

## モデル

- 都市指標5種・24ステップの系列を読むBiLSTM＋時間方向のアテンション
- 系列の表現と6次元のシナリオ・政策ベクトルを結合するMLP
- 3つのヘッド：成功の分類、シナリオの分類、流動・空室率変化の回帰
- MC dropout（16サンプル）による出力の幅

コードは`src/machilens_ai/`にあります。

## 結果

同梱のデモ実行（合成サンプル700件、3エポック）の結果です。ファイル：`reports/demo_metrics.json`、`reports/baseline_metrics.json`。

| モデル | 成功の正解率 | Brier | 流動 MAE | 空室率 MAE |
| --- | ---: | ---: | ---: | ---: |
| BiLSTM＋アテンション | 0.60 | 0.248 | 0.020 | 0.043 |
| ランダムフォレスト（ベースライン） | 0.85 | 0.117 | 0.017 | 0.010 |

この実行では、すべての指標でランダムフォレストのベースラインがニューラルモデルを上回っています。

## デモの数値について

ブラウザデモと`reports/prediction_*.json`の値は、ネットワークの生の出力ではありません。`predict.py`がシナリオごとのプリセット値に置き換え、スライダーの値で線形に調整しています（`calibrate_demo_output`）。ブラウザデモも同じ式をJavaScriptで計算していて、モデルは動かしていません。ネットワークの生の出力は、各予測ファイルの`raw_model_output`に残しています。

## 実行

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

PYTHONPATH=src pytest
PYTHONPATH=src python -m machilens_ai.train --epochs 3 --n-samples 900
PYTHONPATH=src python -m machilens_ai.predict --scenario B --night-economy 0.82 --kpop-fandom 0.86 --policy-support 0.74
streamlit run app/streamlit_app.py   # 任意
```

## 限界

- データは合成です。結果は生成コードに入れた仮定に左右されます。
- 街区レベルの地理情報はなく、都市を5つの集計指標で表しています。
- 実用にするには、検証済みのデータと、実際の結果との照合が必要です。

## ライセンス

MIT
