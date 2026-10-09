# 単一 CDF の対数密度回帰

この拡張は，正規化済みの位相関数の弱い角度構造を近似するための研究実装です．
論文の実験一式を再現したものでも，実際の虹で改善が確認済みの最適設定でもありません．
HG，円周・区間 RQS coupling，鏡映と軸入射の制約，保存 CDF の定義を維持し，
学習点の分布・損失・検証と重みの選択を明示的に変更します．

## 目的関数と測度

一つの入射方向・一つの波長を固定します．`p` は保存 CDF が定めるセル内一定の
立体角密度，`q` は NF の立体角密度，`u=1/(4*pi)` は球面一様密度です．

$$
J = w_{\mathrm{NLL}} E_p[-\log q]
  + \beta E_u[(\log q-\log p)^2].
$$

最初の項は forward KL と教師だけに依存する定数の差です．全て自然対数を使用し，
`log1p(pdf)` の MSLE とは区別します．`|cos(theta)|` などの追加角度重みは使いません．
損失を変えても可逆な NF の解析的正規化は維持されます．外部 `hg.g` は固定ですが，
最終 NF の平均余弦まで自動的に一致するわけではありません．

`sampling="target_uniform"` は厳密に半々の固定 pool です．総点数 N は偶数とし，
CDF 点 N/2 と球面一様点 N/2 を交互に並べます．学習の各 minibatch はこの pool から
従来どおり復元抽出します．`r=(p+u)/2` に対し，

$$
\widehat J = \frac1B\sum_i
\left[w_{\mathrm{NLL}}\frac{p_i}{r_i}(-\log q_i)
 +\beta\frac{u_i}{r_i}(\log q_i-\log p_i)^2\right].
$$

`p/r` と `u/r` はそれぞれ 2 以下です．重みは logaddexp を使って安定に求め，
教師値とともに勾配を流しません．minibatch 内の重みの和による再正規化はしません．
点だけを混合して無重み NLL に渡すと `r` を学習してしまうため，この補正を省けません．

`sampling="target"` は beta=0 の従来 NLL 対照だけに使います．この場合の
`target_fraction=0.5` は混合方式用の固定設定であり，実際の pool の割合は
provenance に `target_fraction=1.0` と記録します．

## 球面一様点と教師値

幾何学的な `a=(1-cos(theta))/2` と方位角を一様に生成します．HG の累積確率座標を
一様にして HG の逆変換に渡す処理ではありません．記録された source frame で方向を
作り，既存の source_to_nf 回転を適用します．元の CDF の非一様な u_edges と，
セル内の正しい立体角密度を保持します．

PCG64 の SeedSequence は `[data_seed, stream_id]` です．乱数は既存と同じ 52-bit
open midpoint grid を使います．学習点の source label は CDF=0，uniform=1 です．

| stream | ID | 用途 |
|---|---:|---|
| CDF train | 0 | 固定学習点；従来 sampler のまま |
| CDF validation | 1 | 独立した NLL / KL 検証 |
| minibatches | 2 | 初期化から独立した torch.Generator の seed |
| CDF test | 3 | 完了時の独立評価 |
| proposal | 4 | 実際の NF サンプルによる ESS と平均余弦 |
| uniform train | 5 | 固定学習 pool の球面一様成分 |
| uniform validation | 6 | 独立した立体角平均の対数誤差 |
| uniform test | 7 | 完了時の独立した対数誤差 |

新しい回帰用の教師値は，方向を実際の geometry dtype に変換した後の方向で再評価します．
これにより，FP32 丸めでセル境界を越えた際に元セルの値を教師にしてしまうことを避けます．
FP64 の変換前方向・教師値・source label と，実際の方向・教師値・source label の両方に
SHA-256 を記録します．球面積分の不偏性は連続分布の定式化についての性質であり，
有限精度の方向量子化が数学的な連続サンプルと同一であるという主張ではありません．

## ゼロ密度を改変しない

新しい経路は，beta=0 の対照も含めて全球面の log 誤差を評価します．そのため，
開始前に保存 CDF の全セルをチャンク走査します．正の面積を持つゼロ密度セルが一つでも
あれば，面積割合とセル数を示して停止します．サンプル点だけの検査ではありません．
周辺 CDF のゼロ質量行も，その行全体をゼロ密度として検査します．

epsilon，PDF floor，CDF の平滑化，点の再抽選，領域の除外は行いません．
ゼロを含む教師の従来 NLL 学習は schema 2 のまま使用できます．この境界は
`zero_policy="error"` と設定に保存します．

## 設定と実行

以下は objective schema 1 の混合点群の設定です．全学習点を球面一様にする
objective schema 2 は，[全点一様の学習](UNIFORM_LOG_TRAINING.md) を参照してください．
旧 schema 1 の比率・重み・乱数・結果の意味は保持します．

従来の schema 2 は NLL 専用のままです．拡張は top-level `schema_version=3` と
`objective` を明示します．`model`，`training`，`visualization` は従来の設定です．

```json
"objective": {
  "schema_version": 1,
  "nll_weight": 1.0,
  "beta": 0.1,
  "sampling": "target_uniform",
  "target_fraction": 0.5,
  "selection_metric": "log_rmse",
  "validation_uniform_samples": 32768,
  "test_uniform_samples": 65536,
  "zero_policy": "error"
}
```

未知フィールドや半々以外の比率は拒否します．`nll_weight=0,beta>0` は純粋な log 回帰，
`nll_weight>0,beta=0` は NLL 対照です．両方ゼロにはできません．

Windows CMD で一つの候補を学習する例です．RECORD は実際の一条件に変更してください．

```bat
call environment.bat
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow train-rainbow ^
  --record "..\rainbow\datasets\drop_a1_i20_700nm_q1800x3600_c90x1800" ^
  --config "configs\rainbow_log_loss.json" ^
  --output "runs\rainbow_log_single_gpu" --device cuda:0
```

同じコマンドに `--resume "runs\rainbow_log_single_gpu\checkpoint.pt"` を付けて再開します．
中断を計画する場合は `--max-steps-this-run 200` を使い，予定した steps を変更しません．
設定を変える比較は新しい出力先で開始します．

学習経路は CUDA が既定です．提供設定は MLP・方向・HG・RQS・学習損失を FP32 で計算し，
教師値と検証の統計集計は FP64 を保持します．固定点と損失用の FP32 教師値を GPU に
一度転送し，更新ごとの CPU 参照や点のコピーを避けます．AMP / TF32 / CPU fallback は
自動で有効にしません．FP16 / CoopVec 推論の一致や速度は今回の実装範囲ではありません．

## 検証・選択・保存

NLL / forward KL は独立した CDF 点で，log MSE / RMSE と RMS 相対誤差は独立した
球面一様点で評価します．RMS 相対誤差は次の定義です．

$$
h_{\mathrm{rel}}=\sqrt{E_u[((q-p)/p)^2]}.
$$

相対誤差は log 空間で集計します．最終結果が FP64 の範囲を超えた場合だけ，
`relative_rmse=null` と `relative_rmse_status="overflow_float64"` を保存します．
有効な対数誤差をこの任意の補助統計の overflow によって削除しません．

| 出力 | 意味 |
|---|---|
| checkpoint.pt | 現在の重み，Adam，RNG，履歴，二種類の選択状態を保存する再開用 |
| best.pt | objective.selection_metric に従って選んだ推論用 |
| best_by_log.pt | 独立検証 log RMSE が最小の推論用 |
| best_by_nll.pt | 独立検証 NLL が最小の推論用 |
| sample_split.json | 全 stream・点数・pool のハッシュと生成元 |
| history.jsonl / history.json | 総目的関数と実測成分，独立検証，確定した履歴 |
| metrics.json | 主選択の検証・最終 test，両選択の step と検証値 |

同値なら最初の検証時点を選びます．test は予定した更新を完了した主選択に対してだけ
実施し，重み・beta の選択に使いません．異なる beta の総目的関数を候補間で比べません．

新規経路は checkpoint version 5 です．version 2/3/4 は評価・再描画の互換性を維持します．
厳密再開では元の目的関数・モデル・実装ハッシュ・入力・runtime・RNG を要求します．
本パッチ前の実装で走らせた実験へ，新しい損失を追加してそのまま再開はできません．

## 監視・図・独立した再評価

`monitor.bat` で TensorBoard を開けます．総 loss は NLL と呼ばず，NLL，log MSE，
係数を分離します．混合点の beta=0 / 純 log 対照でも，可能な両成分は実測して記録します．
CDF-only 対照の minibatch に無効な log MSE のゼロ値は表示しません．全候補の独立検証
log 誤差は常に測定します．混合 pool の固定監視部分は偶数点の均衡した prefix です．

学習曲線には対数誤差のパネルを加えます．中断再開時には，checkpoint より先に記録された
未確定な JSONL / TensorBoard のイベントを，確定履歴から再構成します．

三種類の比較図は元と同じ source frame，軸，アスペクト比を使います．両 PDF は同じ
対数色尺度です．混合点群は CDF / uniform を識別し，実際の全 N 点を表示します．
training_scatter.npz は方向，教師値，生成元ラベルと，検証済みの provenance を含みます．

```bat
call environment.bat
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow plot-rainbow ^
  --record "YOUR_RECORD" --checkpoint "runs\rainbow_log_single_gpu\best_by_nll.pt" ^
  --output "runs\rainbow_log_single_gpu\plots_by_nll" --device cuda:0
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow evaluate-rainbow ^
  --record "YOUR_RECORD" --checkpoint "runs\rainbow_log_single_gpu\best.pt" ^
  --samples 65536 --uniform-samples 65536 --seed 3026 --device cuda:0 ^
  --output "runs\rainbow_log_single_gpu\independent_evaluation.json"
```

`evaluate-rainbow` で --uniform-samples を省略すると version-5 の保存設定を使います．
0 を明示すれば従来の NLL / proposal 評価だけを実行します．再描画で既存の完了済み
sweep 結果を書き換えないため，sweep では専用の replot コマンドと別の出力先を使います．

角度診断に objective を渡すと，混合分布の期待点数 N*(p(R)+u(R))/2 を使います．
旧 CDF-only の sampler audit は version 5 を明示的に拒否します．その二項分布の
検査と旧 hash を，固定比率の混合 pool に黙って流用しません．

## 文献との関係

- Jendersie and d'Eon (2023), [An Approximate Mie Scattering Function for Fog and Cloud Rendering](https://research.nvidia.com/labs/rtr/approximate-mie/publications/approximate-mie.pdf): log PDF の二乗誤差を位相関数 fitting に用いる先行例．今回は追加の |cos(theta)| 重みを使わず，立体角に関する一様平均を使います．
- Draine (2003), [Scattering by Interstellar Dust Grains. I. Optical and Ultraviolet](https://arxiv.org/abs/astro-ph/0304060): 弱い方向の相対誤差を反映する RMS 相対誤差の定義．
- Li et al. (2025), [Normalizing Flow Regression for Bayesian Inference with Offline Likelihood Evaluations](https://arxiv.org/abs/2504.11554): 教師 log density 値による NF 回帰．未知の正規化定数や Tobit likelihood は今回の実装に含めません．
- Schopmans et al. (2026), [Efficient Training of Boltzmann Generators Using Off-Policy Log-Dispersion Regularization](https://proceedings.mlr.press/v306/schopmans26a.html): 通常の学習目的と対数密度に関する正則化の併用．今回は既知の正規化済み教師なので，平均を引いた分散ではなく，log density 比の非中心二乗平均を使います．

この組み合わせと球面上の教師への適用は，本プロジェクトの実験設計です．
損失だけの新規性，細かな干渉縞の完全な再現，レンダリング改善は主張しません．
実験では log RMSE / KL / 角度断面の改善と，最終的な画像の変化を別々に検証します．
