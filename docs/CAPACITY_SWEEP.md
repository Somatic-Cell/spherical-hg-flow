# 単一 CDF：サンプラの確認，bin 数と最適化ステップ数の比較

この追加実験は，本プロジェクトのための実験計画です．関連論文の実験の再現や，
実データで検証済みの最適設定を意味しません．CDF，HG，対称性，RQS の構成と
target-distributed iid 点群を使う等重みの NLL は維持します．

## 実行順序

`sweep.bat` は `environment.bat` が選んだ Python と CUDA を使います．
既存の引数なしの A/B 探索，`diagnose`，`replot` も引き続き使用できます．
今回の追加機能は，それぞれ名前を指定して実行します．

```bat
call execute.bat check
call sweep.bat audit
call sweep.bat capacity
```

点数の再比較は capacity の完了後に実行します．

```bat
call sweep.bat samples
```

別の CMD ウィンドウから監視できます．

```bat
call monitor.bat runs\rainbow_capacity_gpu
```

`RECORD` は動作確認した CDF レコードに合わせてください．`AUDIT_RUN` は
`best.pt`，`checkpoint.pt`，`config.json`，`sample_split.json` を持つ元の学習
ディレクトリです．再描画した図だけを置いた `training_point_plots` ではありません．
既定値は前回の `OUTPUT\stage_b\n_262144` です．

| バッチの変数 | 用途／既定値 |
|---|---|
| `OUTPUT` | 既存 A/B の `runs\rainbow_sweep_gpu` |
| `AUDIT_OUTPUT` | 追加診断の `runs\rainbow_sampling_audit_n262144` |
| `CAPACITY_CONFIG` | `configs\rainbow_capacity.json` |
| `CAPACITY_SWEEP_CONFIG` | `configs\rainbow_capacity_sweep.json` |
| `CAPACITY_OUTPUT` | 新しい探索の `runs\rainbow_capacity_gpu` |
| `SAMPLES_OUTPUT` | 点数再比較の `runs\rainbow_capacity_samples_gpu` |

新しい探索に古い A/B の出力先を指定しないでください．設定を変える場合も新しい
出力先を使います．同じ出力先に複数のプロセスを同時起動しないでください．

## 1．点群は CDF を再現しているか

対数 PDF の色と通常の散布図は，低確率領域を同じ強さで表示しません．
緯度・経度に似た角度の長方形上では点の面密度に `sin(theta)` も含まれます．
この違いだけから，サンプラが正しいとも誤っているとも判定できません．

今回のサンプラは保存された非一様な `u_edges` を使います．

$$
u=(1-\cos\theta)/2,\qquad d\Omega=2\,du\,d\phi.
$$

周辺 CDF の差分と条件付き CDF の差分の積がセルの確率質量です．
セルを選んだ後，CDF の残りの累積確率を使って `u` と `phi` のセル内部を
一様にサンプルします．`theta` に関する一様補間ではありません．

`audit` は保存された学習 pool のハッシュを検証してから，全点を再生成します．
CDF のセルと検査領域の重なりを `u,phi` で積分して，厳密な保存 CDF の確率と
実際の点数を比較します．セル中心での PDF 値を積分の代用にはしません．

| 検査領域 | 目的 |
|---|---|
| theta 120–150 度，全 phi | 虹付近の広い領域の比較 |
| theta 149–151 度，全 phi | theta 150 度付近の薄い帯の点数 |
| theta 85–95 度，phi −5–5 度 | theta,phi = 90,0 付近の領域 |
| theta 89.5–90.5 度，phi −0.5–0.5 度 | 同じ場所の小領域に期待される点数 |

これらは相談された位置の周辺を検査する固定領域で，ピークの厳密な境界を
推定したものではありません．各領域について，教師確率，期待点数，観測点数，
二項分布の標準偏差，観測割合の Wilson 95% 区間を保存します．複数領域をまとめた
合否検定や，同時信頼区間とは呼びません．

サンプラが生成した FP64 方向での点数と，学習時の精度に変換した後の点数を
別々に記録します．これにより，逆 CDF の誤りと浮動小数点丸めによるセル境界の
移動を区別できます．元の checkpoint，評価値，図は書き換えません．

| 出力 | 内容 |
|---|---|
| `sampling_audit.json` | 検査領域の確率・点数・区間，点群と入力の識別情報 |
| `sampling_histogram.npz` | 等立体角の粗いセルの期待・観測質量と点数 |
| `sampling_histogram.png` | 同じ粗いセルで比較した教師 PDF，経験 PDF，標準化残差 |

ヒストグラムの既定値は `u` 方向 32，phi 方向 64 セルです．これはサンプラの
大域的な整合性を調べる診断で，元の高解像度 PDF や全学習点の散布図の代わりでは
ありません．経験 PDF は `点数 / (N * セルの立体角)` です．教師も同じセルで
積分した質量を立体角で割ります．期待点数の小さいセルを明示します．
疎な高解像度セルに漸近的な chi-square 検定をそのまま適用しません．

点数が足りるかを考える基準は，画像上の明るさよりも対象領域の確率質量です．

$$
E[n_R]=NP_R,\qquad
P(n_R=0)=(1-P_R)^N,\qquad
\sigma(n_R)=\sqrt{NP_R(1-P_R)}.
$$

例えば `P_R=1e-5` という仮の領域では，262,144 点でも期待値は約 2.62 点です．
1,048,576 点なら約 10.49 点になりますが，まだ細かい密度形状が十分分かるとは
限りません．この数値は実際の干渉縞を測定した結果ではありません．

## 2．bin 数と最適化ステップ数

今回の設定は次のとおりです．前回の N=262,144 が最良だったという意味ではなく，
容量比較の共通 pool を大きめに取る設計です．

| 項目 | 設定 |
|---|---|
| RQS bin 数 | 16，32，64，128 |
| 比較する最適化ステップ数 | 2,000，6,000，10,000，20,000 |
| 学習点数 | 262,144，全候補で同じ pool |
| coupling | 4 層 |
| hidden features | `[64,64]` |
| 学習率 | 0.003，一定 |
| optimizer | 既存の Adam |
| minibatch | 1,024 点 |
| 精度 | 重み・HG・RQS・方向は FP32；教師・統計集計は FP64 |
| seed / data seed | 415 / 2026 |
| validation / test | 32,768 / 65,536 点，独立な共通 stream |
| validation 評価 | 100 更新ごと |

モデルの形が違うため，seed が同じでも初期の全パラメータが同じという意味では
ありません．学習点群，validation，test の比較条件はそろえます．

各 bin 数で 20,000 更新の計画を最初から固定し，途中の 2,000，6,000，10,000 更新で
checkpoint を保存して続行します．4 本で合計 80,000 更新です．16 通りをすべて
最初から独立に学習する場合の 152,000 更新に比べ，重複する前半の学習を省きます．
GPU での経過時間はこれに比例するとは限りません．モデル容量，評価，描画も影響します．

既存 trainer の optimizer，RNG，minibatch RNG と履歴の再開処理をそのまま使います．
一定の学習率と固定の評価間隔なので，予定した最終更新数に応じて前半の学習率を
変えることはありません．比較点は全て validation の評価間隔にそろえます．
各時点の選択対象は，その時点までの validation で評価した重みだけです．
後の時点で選んだ重みを前の時点の結果として表示しません．

これは前回の 2,000 更新で完了した実験の `steps` を書き換えて再開する機能では
ありません．新しい探索を始め，その中で計画済みの途中保存を行う仕組みです．

`summary.csv`，`summary.json`，`summary.md`，`summary.png` に，各 bin 数・各時点の
validation NLL / KL，選択された更新数，現在の validation，処理点数，parameter 数などを
まとめます．各時点には凍結した重み，履歴，3 種類のマップ，角度断面の診断を残します．
散布図は各実験の固定 pool の全点です．

最良値を保存する validation の曲線は，定義上増加しません．収束や過学習の判断には
`latest_validation` と学習履歴も使ってください．途中時点の test は評価せず，
20,000 更新時点の test は報告だけに使います．bin 数は共通の 20,000 更新時点での
最小の validation NLL で選び，同値なら小さい bin 数を選びます．

自動選択は大域的な分布の適合であり，干渉縞の再現を自動的に保証しません．
角度断面図と元の PDF／NF PDF で，問題にしている弱いピークも確認してください．
大きい bin 数で validation が不安定な場合は，学習率 0.001 の別実験を比較します．
損失やサンプルの重みを暗黙に変えて補正することはありません．

### 再開と制御された途中停止

同じ設定・入力・コード・実行環境のまま，同じコマンドを再実行します．
設定や実装が変わった場合は，元の探索への上書きを拒否します．
Python CLI では，一度の起動で進める途中保存の数を制限できます．

```bat
call environment.bat
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow sweep-capacity-rainbow ^
  --record "YOUR_RECORD" ^
  --config "configs\rainbow_capacity.json" ^
  --sweep-config "configs\rainbow_capacity_sweep.json" ^
  --output "runs\rainbow_capacity_gpu" ^
  --device cuda:0 ^
  --max-milestones-this-run 1
```

中断後も，最初に決めた最終ステップ数を保持します．途中まで完了した探索では
最終的な bin の選択は確定しません．

## 3．選ばれた bin 数で点数を再比較する

`call sweep.bat samples` は，完了した容量探索の選択結果を検証し，
選ばれた bin 数と同じ 20,000 更新で N=65,536，262,144，1,048,576 を比較します．
容量探索と同一の N=262,144 の結果は検証して再利用し，新しく学習するのは
残りの二つです．test ではなく validation で選択します．

これは bin=16 で得た以前の点数比較を，大きい bin 数へそのまま一般化しないための
追加実験です．全 bin 数・全点数・全ステップ数の総当たりを最初に行う必要はありません．

更新数を T，batch size を B とすると，処理点数は `T*B` です．`T*B/N` は pool の
平均的な使用回数の目安で，重複なく全点を一巡する通常の epoch とは異なります．
固定 pool の更新回数を増やしても，新しい点が追加されるわけではありません．
N を増やすと小領域の独立な学習点は増えますが，同じ iid 損失と B の下で
その領域に入る minibatch の期待割合は増えません．

## 関連研究と LUT の範囲

NPIS の巨大な emitter tail は，一つの環境画像に対する条件なしの写像です．
その順写像・逆写像・PDF の二次元の表へのベイクは，今回の全条件を含む
NF を密な表へ変換することと同じ規模にはなりません．

粒子の形状・大きさを固定し，入射方位角の対称性を使う場合，PDF の LUT の
独立な軸は波長，入射傾斜角，出射方向の二座標の計四つです．PDF 値や
サンプル写像の出力座標はチャンネルであり，追加の格子軸ではありません．
`g` が条件から決まるなら，それも独立な格子軸にはしません．

無圧縮の容量は `条件の波長数 * 入射角数 * 角度セル数 * チャンネル数 * byte数` です．
例えば角度 1800×900，FP32，PDF 一つなら一条件で約 6.18 MiB，条件が 64×64 なら
約 24.72 GiB，128×128 なら約 98.88 GiB です．これらは仮定からの計算値で，
実測値や必要解像度の推定ではありません．当面は条件付き NF を直接評価する方針を
保ち，モデルの精度と計算量を確認してから native 推論を設計します．

参考：

- Litalien et al.，*Neural Product Importance Sampling via Warp Composition*，
  [§3.2，§4](https://arxiv.org/html/2409.18974v2)．
- Sadeghi et al.，*Physically-Based Simulation of Rainbows*，2012，§5．
  1800×14400 は位相関数の角度評価格子で，NF 用 iid 学習点数ではありません．
- [Mitsuba 3 の検査の説明](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/testing.html)．
  PDF を積分した期待度数とサンプラの観測度数を比較する Pearson chi-square 検定を説明しています．

CPU の小さな合成 fixture によるテストはサンプラ・制御処理の検証です．
実際の Rainbow の収束，Windows，CUDA，OptiX の動作や速度の実測を代用しません．
