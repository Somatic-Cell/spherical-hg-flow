# Log PDF の二乗誤差と beta の比較

この実験は，一つの Rainbow CDF の位相関数を **値の評価にも使える形で近似する**ため，
従来の NLL と，立体角について平均した対数密度誤差を比較します．
過去の NLL 選択による sweep とは出力・設定・checkpoint 形式を分けます．
条件間補間，OptiX 実装，実データでの有効性を検証済みとするものではありません．

## 損失と学習点

教師を `p`，NF を `q`，球面一様を `u=1/(4*pi)` とすると，学習目的は次です．

```math
J_\beta=E_p[-\log q]+\beta E_u[(\log q-\log p)^2].
```

既定では `r=0.5*p+0.5*u` から固定学習点を作ります．その点での推定は，

```math
\widehat J_\beta=\frac1B\sum_i\left[
\frac{p_i}{r_i}(-\log q_i)+\beta\frac{u_i}{r_i}(\log q_i-\log p_i)^2\right]
```

です．両重みは 2 以下です．バッチ内の自己正規化を行わず，教師値と重みは固定します．
一様点を混ぜて無重みの NLL を適用すると教師分布が変わるため，その処理は行いません．
球面一様は `cos(theta)` と azimuth についての一様です．HG CDF 座標の一様とは異なります．
密度は立体角当たりであり，画像の等間隔行・列や theta を等重みにする目的ではありません．

`log` は自然対数です．`log1p` による MSLE，教師 PDF の floor，平滑化は使用しません．
正の面積を持つ教師のゼロ密度セルでは log 誤差が有限にならないため，`zero_policy="error"`
によって学習前に停止します．正の教師に対する現在の比較の仕様です．
HG の `g`，source/NF の座標契約，全周・鏡映の RQS 構成は維持します．

## バッチから開始する

まず log 損失の core patch，次に sweep patch を適用してください．
`sweep_log_loss.bat` の `RECORD` を，動作している `execute.bat` と同じ値にします．
`environment.bat` の Python を共用し，GPU での forward/backward 検査後に開始します．
CUDA が利用できなくても CPU へ自動的に切り替わることはありません．

```bat
call sweep_log_loss.bat plan
call sweep_log_loss.bat
```

`plan` は設定と更新数を表示するだけで，CDF の読み込み，GPU の確保，学習，出力先の作成を
行いません．`run` は同じ設定で再度実行すると，完了した結果を検証して再利用し，
未完了の trial を `checkpoint.pt` から再開します．

別の CMD で監視できます．

```bat
call monitor.bat "runs\rainbow_log_loss_gpu"
```

新しい config は `schema_version=3`，最上位に `objective` を持ちます．
従来の `schema_version=2` で `objective` のない設定も sweep の基礎設定として読めます．
保存された互換な実験の `config.json` を `CONFIG` に指定すれば，そのモデルと学習設定を
そのまま使用できます．未対応のモデル・encoding フィールドを黙って取り除くことはしません．
この sweep は beta と control の損失・sampling 軸を明示的に設定し，全候補の
`selection_metric` を `log_rmse` に揃えます．基礎設定の beta は sweep の候補値に置換されます．

## 既定の比較条件

| 要素 | 設定 |
|---|---|
| beta | 0，0.01，0.03，0.1，0.3，1 |
| 共通の混合比 | CDF 1/2，球面一様 1/2 |
| 追加の対照実験 | CDF 点のみ・NLL 学習を 1 本 |
| 任意の対照実験 | `include_pure_log_control=true` で NLL 係数 0，beta 1 を追加 |
| モデル | L=16，K=64，hidden=[64,64]，既存の conditioner |
| 学習率・optimizer | 0.001・既存の Adam |
| 学習点数 | 合計 262,144；混合 trial は各 131,072 点 |
| 更新数・minibatch | 20,000 更新・1,024 点 |
| 精度 | MLP・HG・RQS・方向は FP32；教師参照と統計集計は FP64 |
| seed / data seed | 415 / 2026 |
| 教師分布の validation / test | 32,768 / 65,536 点 |
| 球面一様の validation / test | 別 stream の 32,768 / 65,536 点 |
| 評価・checkpoint 間隔 | 100 更新 |

既定は 7 本，合計 140,000 更新です．pure-log 対照を追加すると 8 本・160,000 更新です．
実時間は，GPU，モデル，評価と図の出力に依存します．夜間に終わるという実測保証はありません．
beta と学習率を同時に変更せず，まずこの条件で対数形状と KL の変化を確認します．
beta が大きいほど必ず有利という前提はありません．

混合 trial と pure-log 対照では，方向・教師値・成分ラベルを含む pool の SHA-256 が一致します．
CDF のみの対照は意図的に異なる pool ですが，同じ総点数・初期化 seed・更新数・holdout を使います．
これは「昔の NLL 最良 checkpoint をそのまま再使用する」対照ではありません．
全 trial で同じ validation log RMSE を使って checkpoint を選び直す，比較可能な新規実験です．

## 結果と選択規則

| 出力 | 内容 |
|---|---|
| `manifest.json` | 全設定，入力とコード・実行環境，独立 stream と共通 pool の識別情報 |
| `summary.json` / `summary.csv` | 全 trial の状態，beta，log RMSE，KL，step，時間，pool hash |
| `summary.md` / `summary.png` | beta 比較，KL と log RMSE の関係，検証曲線 |
| `selection.json` | 全 trial 完了後の validation log RMSE による選択 |
| `trials/beta_*/best.pt` | validation log RMSE で選ばれた推論用重み |
| `best_by_log.pt` / `best_by_nll.pt` | 同じ学習履歴で各基準により選ばれた二種類の重み |
| `checkpoint.pt` | optimizer，RNG と途中状態を含む厳密な再開用状態 |
| `plots/` | 教師 PDF，全学習点，NF PDF の共通座標の可視化 |
| `diagnostics/` | 保存 source 座標での角度断面；報告用でモデル選択には使わない |
| `log_sweep_completed.json` | 完了 trial の変更を検出する成果物の hash |

すべての候補で **同じ独立 validation log RMSE** を最小にする checkpoint を比較します．
同値なら小さい beta，さらに trial ID の辞書順で選びます．総 loss は beta により定義が
変わるので，その大小で順位を決めません．test も選択や停止には使用しません．
検証 KL と検証 log RMSE の Pareto 候補も記録します．NLL 選択時の step・NLL・log RMSE は
別列で残し，形状重視により前方の確率質量をどれだけ犠牲にしているかを確認できます．
負の連続 NLL や，有限標本でわずかに負となる KL は，表示のためにゼロへ切り上げません．

全点の散布図は実際の学習 pool を使い，CDF と一様の点を区別します．混合点全体は `r` の
点群であり，教師 `p` からの点群とは呼びません．過去の固定 32,768 点への置換は行いません．

## 制御された途中停止と再描画

一つの trial の最初の 500 更新だけを確認する場合の例です．予定の 20,000 更新は維持します．

```bat
call environment.bat
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow.log_sweep run ^
  --record "YOUR_RECORD" ^
  --config "configs\rainbow_log_loss.json" ^
  --sweep-config "configs\rainbow_log_sweep.json" ^
  --output "runs\rainbow_log_loss_gpu" ^
  --device cuda:0 --max-steps-this-trial 500
```

次は制限を外し，同じ設定・同じ出力先で実行します．設定・コード・データ・記録した実行環境が
違う場合は，別実験として新しい出力先を指定してください．過去の完了済み trial は上書きしません．
CSV は LF 改行で出力し，CRLF を含む生成 CSV の patch による whitespace 警告を避けます．

```bat
call sweep_log_loss.bat replot
```

完了した `best.pt` を検証し，全学習点と角度断面を `OUTPUT/replots/日時/` に再描画します．
元の trial の重み・数値・図は変えません．一部だけ完了した sweep にも使えます．
再描画は追加の最適化や test を行いません．

## 文献との対応と主張の範囲

- Jendersie and d'Eon (2023), *An Approximate Mie Scattering Function for Fog and Cloud Rendering*，
  [§1.1](https://research.nvidia.com/labs/rtr/approximate-mie/publications/approximate-mie.pdf)．
  log PDF の二乗誤差を使う先行例です．この実験では追加の `abs(cos(theta))` を採用しません．
- Draine (2003), *Scattering by Interstellar Dust Grains. I. Optical and Ultraviolet*，
  [§4](https://arxiv.org/abs/astro-ph/0304060)．立体角に関する RMS 相対誤差を評価にも記録します．
- Li et al. (2025), *Normalizing Flow Regression for Bayesian Inference with Offline Likelihood
  Evaluations*，[論文](https://arxiv.org/abs/2504.11554)．NF を教師 log density に回帰する先行例です．

上の複合目的・混合 query・現在の HG 球面モデルの組合せは，今回明示的に設計した実験です．
いずれかの論文の実装・全実験をそのまま再現したとは扱いません．
形状の誤差が減るか，レンダリングの虹が改善するか，FP16 重みで維持できるかは，
実際の CDF と GPU・レンダラで別途測定する必要があります．
