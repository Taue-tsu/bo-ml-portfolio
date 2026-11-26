# ベイズ最適化

ガウス過程とExpected Improvementを用いた逐次探索。合成関数上でランダム探索と比較する。

## ファイル

- `bayesian_optimization.py`：GP予測、EI、候補選択、逐次更新。
- `bayesian_optimization_demo.ipynb`：実行例と比較図。
- `requirements.txt`：使用ライブラリ。

## 実行

```bash
pip install -r requirements.txt
jupyter lab bayesian_optimization_demo.ipynb
```

## 条件・制約

- 初期点3点、追加観測20回、10 seed。
- 固定カーネル、有限候補集合、単一目的の最大化。
- 実データでの有効性は未検証。
