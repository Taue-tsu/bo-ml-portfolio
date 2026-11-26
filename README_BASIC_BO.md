# Bayesian Optimization Portfolio

ガウス過程（GP）とExpected Improvement（EI）を用いた、単一目的の逐次ベイズ最適化の実装です。研究用コードから基本部分を抽出し、公開可能な合成関数で動作を確認できる形に整理しました。

## 掲載内容

- 観測値の標準化、Cholesky分解によるGP学習・予測
- EIによる未観測候補の選択と逐次更新
- 同一初期点・同一観測予算でのランダム探索との比較
- 複数seedでのsimple regretの平均とばらつきの可視化

## 実行方法

```bash
pip install -r requirements-basic-bo.txt
jupyter lab bayesian_optimization_demo.ipynb
```

外部データや認証情報は不要です。ノートブックは出力を削除した状態で掲載しています。実装は `bayesian_optimization.py` を参照してください。

## 範囲と限界

固定カーネル・有限候補集合・単一目的の基本実装です。実際の創薬での効果や手法の新規性を主張するものではありません。研究の発展部分、未公開データ、初歩的な文法練習は含めていません。依存関係は固定していないため、利用環境によって差が生じる可能性があります。
