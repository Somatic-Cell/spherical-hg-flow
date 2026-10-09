# 全点を球面一様に生成する log PDF 回帰

本設定は，固定した一条件の保存 CDF を教師として，全学習点を球面一様に生成し，
その点での log PDF 二乗誤差を学習する経路です．混合点群を使う以前の pure_log と
比較して，有限の学習点をどこへ配分するかの影響を調べます．

HG 基底，円周・区間 RQS coupling，対称性，保存 CDF，FP32 の学習計算は既存仕様です．
全点一様による halo の精度改善を実データで確認済みという意味ではありません．

## サンプリングと目的関数

学習方向は球面一様密度 `u=1/(4*pi)` から生成します．`cos(theta)` と方位角を
一様に生成するので，`theta` の等間隔な角度図の上では面積当たり一様には見えません．
source frame で作った方向を記録された NF frame へ回転し，実際の geometry dtype
にした後の方向で教師 `p` を評価します．HG の確率座標を一様にした点ではありません．

既定の損失は，次の単純平均です．

$$
\widehat L_{\log}=\frac1B\sum_{i=1}^{B}
\left[\log q(\omega_i)-\log p(\omega_i)\right]^2,
\qquad \omega_i\sim u.
$$

log 項に教師 PDF や混合分布の重要度重みは掛けません．すべて自然対数です．
保存 CDF に真のゼロ密度セルがある場合は，従来の log 経路と同じく明示的に停止し，
floor・平滑化・教師分布の変更は行いません．

全点一様と半々の混合点群＋適切な重要度重みは，連続分布については同じ立体角平均の
log 誤差を目的とします．一方，有限点群の配置と勾配推定のばらつきは異なります．
同じ総点数なら，全点一様は一様生成点を二倍確保できます．これだけで物理角度上の
knot の均等配置や微細構造の完全な再現を保証するわけではありません．

## 明示的なバージョンと設定

top-level の設定は従来どおり schema 3，objective 内だけ schema 2 を使います．

```json
"objective": {
  "schema_version": 2,
  "nll_weight": 0.0,
  "beta": 1.0,
  "sampling": "uniform",
  "target_fraction": 0.0,
  "selection_metric": "log_rmse",
  "validation_uniform_samples": 8192,
  "test_uniform_samples": 16384,
  "zero_policy": "error"
}
```

objective schema 1 は従来の target / target_uniform のままです．schema 2 は
uniform / target_fraction=0 の組だけを受け入れます．以前の設定を黙って別の
サンプリングへ読み替えません．この一様設定には単独の `train-rainbow` 経路を使い，
混合点群による beta 比較用の `phaseflow.log_sweep` には渡しません．

`nll_weight>0` を指定する場合の NLL 項は，数学的に正しい `p/u` 重みで計算します．
この重みは混合方式の `p/r` と異なり上限 2 を持たないため，今回の既定では NLL を
学習に加えません．学習ログの NLL 診断値には実測の `p/u` を使いますが，係数 0 の
ときは勾配へ加えません．独立 validation/test の NLL/KL は引き続き CDF 点で測ります．
log 誤差と相対誤差の独立評価は，学習と別 stream の球面一様点を使います．

## 既定の比較条件

| 項目 | 新しい単独実験 |
|---|---|
| モデル | L=4，hidden=64×64，K=64 |
| 学習率・更新数 | lr=0.003，2,000 更新 |
| 学習点 | 球面一様 262,144 点，CDF 由来 0 点 |
| Minibatch | 1,024，固定 pool から復元抽出 |
| 学習精度 | CUDA / FP32 |
| Validation | CDF 8,192 点＋独立球面一様 8,192 点 |
| Test | CDF 16,384 点＋独立球面一様 16,384 点 |
| 重みの選択 | validation log RMSE |
| 図 | 完了時に全解像度比較図と全実学習点を自動保存 |

先に点群の配分だけを比較するため，前回の quick pure_log と同じ 2,000 更新に
揃えます．学習率・モデル・点数・seed・評価点数を同時に変更しないでください．
5,000 更新を調べる場合は，その更新数を最初から設定した別の出力先で比較します．

同じ data seed では，一様点の最初の 131,072 点が，以前の混合 pool の一様成分と
一致します．一様点をさらに 131,072 点追加した構成です．全点は既存の stream 5
から生成し，CDF training stream 0 は呼びません．`sample_split.json` の
`training_pool` に `cdf=0`，`uniform=262144`，`target_fraction=0` と実際の
source streams を記録します．歴史的な `streams.train=0` は stream 名の登録情報で，
新しい点群に CDF 点が含まれることを示すものではありません．

## 実行と再開

前回の log-density objective / beta sweep / quick 設定のパッチを適用した checkout に，
今回の追加パッチを適用します．`train_log_uniform.bat` の `RECORD` を，動作している
`execute.bat` の一条件のディレクトリに合わせてください．

```bat
call train_log_uniform.bat
```

新しい `runs\rainbow_log_uniform_gpu` に一試行を実行し，独立評価・全解像度の三図・
角度断面まで保存します．各角度で元 CDF の PDF を照会するため，CDF レコード自体は
引き続き必要です．環境は既存の `environment.bat` の Python と CUDA を使用します．

途中で区切る場合：

```bat
call train_log_uniform.bat probe
call train_log_uniform.bat
```

probe は最大 250 追加更新で止まり，後の通常実行が optimizer と RNG を再開します．
通常実行は同じ出力先の `checkpoint.pt` があれば自動で再開します．既存の混合点群の
run を新しい OUTPUT として指定しないでください．サンプラ，コード，設定を変えた
実験は別の出力先へ開始し，既存の再現性確認を迂回しません．

```bat
call monitor.bat "runs\rainbow_log_uniform_gpu"
```

この監視は別の CMD から実行できます．`history.jsonl` と `learning_curves.png` も
保存します．学習中の合成 loss は今回 log MSE と一致しますが，別の目的関数で学習した
run と比較するときは，同じ独立評価点上の log RMSE / KL / 相対誤差を使います．

## 可視化

学習完了時の図は `runs\rainbow_log_uniform_gpu\plots` にあります．

- `comparison.png`：保存 CDF の log PDF，実学習点群，NF の log PDF．
- `training_samples.png`：今回の全一様学習点群を表示する単体図．
- `plots.json` / `training_scatter.npz`：座標，色尺度，点数，生成元と実際の点群．

点群は全 262,144 点を表示し，32,768 点などに間引きません．CDF の分布から
生成した点群として表示することもしません．両 PDF は同じ記録座標，アスペクト比，
対数色尺度を使います．`diagnostics` には既存の 120–150 度帯と方位 −90/0/90 度の
断面が保存されます．

学習途中の選択重みや完了済み重みを手動で描画する場合：

```bat
call train_log_uniform.bat plot
```

`best.pt` を使い，`plots_manual` に出力します．学習はしません．同じ手動描画を
繰り返す場合，この手動描画先の図を更新します．自動生成の `plots` は別の場所です．

### すでに終わった quick sweep の図を作る

今回の一様学習とは独立して，既存結果は以下で描画できます．

```bat
call sweep_log_loss_quick.bat replot
```

`runs\rainbow_log_quick_gpu\replots\<timestamp>\<trial_id>\plots`
に各試行の比較図を作ります．新しいパッチの適用前でも使える既存機能です．
学習は行わず，既存の checkpoint・summary を変更しません．元の混合実験の点群は
CDF 成分と一様成分を区別したまま描画し，全点一様だったかのように表示しません．

図の log 色尺度は各比較図の教師と NF の間で共有します．別 run の画像を並べる際は，
色バーの範囲も確認してください．異なる run の色バーまで固定する処理ではありません．
