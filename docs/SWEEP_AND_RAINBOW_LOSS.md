# 一条件の A/B 探索と虹の角度帯の診断

この追加機能は，現在の球面 HG + RQS coupling モデルと，CDF 点群に対する NLL を用います．
学習率と固定学習点群の大きさを順に比較し，保存結果を自動集計します．
新しい損失関数の導入や，教師の平滑化・ゼロ密度の置換は行っていません．

## 1. Windows で実行する

前提は v0.4 の `execute.bat setup` / `check` が通り，更新版 Rainbow ソルバの一条件の
CDF を読み込めることです．本追加機能に追加の依存パッケージはありません．
`sweep.bat` は既存の `environment.bat` で選んだ Python と CUDA を使います．

まず，`sweep.bat` の次の設定を合わせます．

| 設定 | 指定するもの |
|---|---|
| `RECORD` | 動作確認済みの `execute.bat` と同じ一条件の CDF ディレクトリ |
| `CONFIG` | 基準の学習設定．既定 `configs\rainbow_single.json` |
| `SWEEP_CONFIG` | 探索候補と角度診断．既定 `configs\rainbow_sweep.json` |
| `OUTPUT` | 新しい探索の保存先．既定 `runs\rainbow_sweep_gpu` |
| `EXISTING_RUN` | 診断する既存の学習結果．既定 `runs\rainbow_single_gpu` |
| `DEVICE` | 使用する CUDA device．既定 `cuda:0` |

既存結果の角度断面を調べるだけなら，次を実行します．

```bat
call sweep.bat diagnose
```

`EXISTING_RUN/config.json` と validation が選んだ `best.pt` を読み，
`EXISTING_RUN/diagnostics/` に診断を保存します．最適化の追加更新はありません．
学習記録と異なる CDF を指定すると停止します．

探索の開始・続行には同じコマンドを使います．

```bat
call sweep.bat
```

別の CMD ウィンドウで学習を監視できます．

```bat
call monitor.bat runs\rainbow_sweep_gpu
```

ブラウザで `http://127.0.0.1:6006` を開きます．TensorBoard の各 run は
`stage_a/lr_...` または `stage_b/n_...` として区別できます．
GPU がない場合，自動的に CPU 学習へ切り替わることはありません．

## 2. 比較する条件と選択規則

`configs/rainbow_sweep.json` の初期値は次のとおりです．

| 段階 | 変更する項目 | 候補 | 固定する項目 |
|---|---|---|---|
| A | 学習率 | `3e-4, 1e-3, 3e-3` | 学習点数 `65536` |
| B | 固定学習点群の点数 | `4096, 16384, 65536, 262144` | A で選ばれた学習率 |

それ以外は `CONFIG` を共有します．提供されている v0.4 の基準設定では，FP32，
batch size = 1024，更新回数 = 2000，validation = 32768 点，test = 65536 点です．
利用者が基準設定を編集していれば，編集後の値をそのまま使います．探索スクリプトが
モデルの層数・bin 数・更新回数・dtype などを上書きすることはありません．

各 A 試行の `best.pt` はその試行の validation NLL で選ばれます．その値を A の
3 候補で比較し，最小の試行の学習率を B に渡します．完全な同値のときは小さい学習率を選びます．
同じ validation 点群なので，validation NLL と forward KL による順位は一致します．
最終 test と角度診断は結果報告用であり，この自動選択には使用しません．

小さい学習 pool は最大 pool の prefix です．初期化・minibatch 用 seed，データ生成用 seed，
独立な validation / test の stream を共通にし，各 pool と評価点群のハッシュを確認します．
N が変わると同じ乱数 seed でも抽出される点の列は変わります．同一の minibatch を保証する
比較ではありません．各候補の独立な反復も 1 回だけです．

B の N = 65536 は A の勝者と全学習設定が同一なので，検証済みのその結果を再利用します．
既定では **6 回の実学習・7 行の結果**になります．再利用行には `reuse_of` と
実際の `run_directory` を記録します．別 seed の再実験として数えてはいけません．

### 点数と epoch を混同しない

この比較では，各候補の最適化更新回数 T と batch size B を固定します．
既定なら，各候補が処理する点は合計 2000 × 1024 = 2,048,000 点です．
固定 pool から復元抽出しており，次の値は厳密な shuffle epoch 数ではなく平均再使用回数です．

| 固定 pool の N | 処理点数 / N |
|---:|---:|
| 4096 | 500 |
| 16384 | 125 |
| 65536 | 31.25 |
| 262144 | 7.8125 |

これにより，まず同じ更新予算で点群を増やす効果を比較します．同じ wall-clock 時間の比較でも
収束後の精度だけの比較でもありません．点数の大きい候補で validation がまだ改善していれば，
その候補について更新回数を増やした別実験が必要です．基準 N で最良の学習率が，全ての N で
最良とは限らないため，最終候補が絞れた後に学習率を近傍で再確認できます．

## 3. 保存される結果

探索ルートに以下を保存します．

| ファイル | 内容 |
|---|---|
| `manifest.json` | 候補，モデル・学習設定，データ・コード・実行環境の識別情報，点群ハッシュ |
| `selection.json` | A の各 validation 結果と，B に渡した学習率の選択根拠 |
| `summary.csv` | 表計算ソフト用の比較表．UTF-8 BOM 付き |
| `summary.json` | 各試行の設定・結果・角度診断を含む機械可読の集計 |
| `summary.md` | 閲覧用の比較表と出力先 |
| `summary.png` | A/B の validation KL 比較と学習曲線 |

表の主な列は `validation_nll`，`validation_kl`，`validation_kl_standard_error`，
`test_kl`，`test_kl_standard_error`，`test_relative_ess`，`selected_step`，
`processed_examples`，`effective_passes`，実行時間，重み数です．
角度診断の数値は `diag_` で始まる列に入ります．

`summary.png` の誤差棒は各 KL 推定の ±1.96 Monte Carlo SE です．
seed 間のばらつきや，2 モデルの差の paired confidence interval ではありません．
validation に繰り返し適合させた選択の不確かさも表していません．
小さな差を有意差と決めつけず，最終候補は seed を変えて確認してください．

各試行は，既存の `config.json`，`best.pt`，`checkpoint.pt`，`history.jsonl`，
`learning_curves.png`，TensorBoard，`metrics.json`，`plots/comparison.png` なども保存します．
比較図の CDF 点群は描画用の独立 stream で，学習 pool 自体の散布図ではありません．
PDF の色尺度は各試行内で教師と NF が共通です．試行をまたいだ色尺度の同一性は保証していません．

### 再開と既存結果の保護

同一の探索を再実行すると，完了した試行の設定・checkpoint・結果ファイルのハッシュを確認して
再利用し，途中の試行は保存済み `checkpoint.pt` から続行します．途中で停止した場合も
部分的な集計が残ります．最適化が終わった後の図作成・診断に失敗した場合は，最終 checkpoint
から追加更新なしで後処理をやり直します．

データ，記録対象のコード，実行環境，設定が変わる場合は新しい `OUTPUT` を指定します．
同じディレクトリを複数プロセスで同時に書くことは想定していません．
OS 強制終了では，最後の checkpoint 以降の未保存の更新は再実行されます．

## 4. 135 度付近を診断する

既定の報告用の帯域は，散乱角 theta = 120〜150 度，方位角は全周です．
これは今回の関心領域を含む設定であり，物理的な虹の境界を自動推定したものではありません．
学習対象をこの帯域に切り取る設定でもありません．

`diagnostics/angular_diagnostics.json` に次を保存します．

| 項目 | 意味 |
|---|---|
| `reference_band_mass_exact` | 保存 CDF の区分一定分布を帯域内で積分した確率 P_R |
| `expected_train_band_samples` | 固定学習 pool に含まれる帯域内の点数の期待値 N × P_R |
| `expected_minibatch_band_samples` | 1 minibatch に含まれる点数の期待値 B × P_R |
| `hg_band_probability_width` | 元の g の HG での帯域確率．HG 累積確率座標での帯域の幅 |
| `reference_zero_solid_angle_fraction` | 教師 PDF が厳密にゼロのセルが占める全球立体角の割合 |
| `reference_band_zero_solid_angle_fraction` | 帯域の立体角に占めるゼロ密度セルの割合 |

帯域質量は保存された実際の `u_edges` と階層 CDF を使い，境界で切られたセルも含めて
積分します．`exact` は保存セル分布について丸め誤差を除いて厳密，という意味です．
連続的な物理ソルバの真値の積分ではありません．点数は特定の実現済み pool の実測数ではなく
期待値です．帯域全体の点数が十分でも，その中の細いピークを十分に覆うとは限りません．

`angular_profiles.png` はソルバの source phi = −90，0，90 度に最も近いネイティブセル中心での
教師 / HG / NF の断面です．実際に使った方位角を図に記載します．上段は 0〜180 度，
下段は 120〜150 度を拡大し，縦軸は立体角 PDF の対数表示です．
教師は保存セルの値を階段状に描き，NF と HG はセルの u 中点で評価します．
元の非一様セルを間引かずに使いますが，NF のセル内部の変化を検出できる保証はありません．
数値は `angular_profiles.npz` に保存します．厳密なゼロ教師 PDF は灰色の帯で示し，
log PDF は −infinity のまま NPZ に保存します．小さな正数への置換はありません．

### 結果から次の実験を選ぶ

- pool 点数を増やすと独立 validation が改善し，断面の細い特徴も戻る場合：点群の不足が
  一因であると考えられます．更新回数だけを増やしても固定 pool に新しい点は増えません．
- 固定 pool 上の NLL と validation がともに改善し続けている場合：追加の最適化を比較します．
- training と validation の差が大きくなる場合：同じ点を反復するだけでなく，pool 増加を比較します．
- 全球の NLL は改善したのに帯域の断面が改善しない場合：NLL の質量に応じた配分，
  モデル容量，HG 座標での特徴の狭さを，それぞれ別実験で検討します．

同じ傾向は複数の原因で生じ得るので，この分類だけで因果関係を確定しません．
特に添付図の「CDF samples N = 32768」は可視化用の点数，「selected step 1200」は
validation が選んだ重みの時点です．図だけでは学習 pool の N や予定した T は分かりません．

## 5. Jendersie & d'Eon (2023) の損失をどう評価するか

一次資料：

- [プロジェクトページ](https://research.nvidia.com/labs/rtr/approximate-mie/)
- [An Approximate Mie Scattering Function for Fog and Cloud Rendering, 本文](https://research.nvidia.com/labs/rtr/approximate-mie/publications/approximate-mie.pdf)
- [Supplemental material](https://research.nvidia.com/labs/rtr/approximate-mie/publications/approximate-mie-supplemental.pdf)

本文 Section 1.1 の式 (1) は，軸対称な位相関数について次の損失です．

\[
E_{\rm AS}=\sum_\theta |\cos\theta|\sin\theta
\left[\log q(\theta)-\log p(\theta)\right]^2.
\]

log PDF の二乗差なので，相対的な誤差に敏感です．sin theta は球面の面積要素に関係します．
abs(cos theta) は前方と後方を重視しますが，135 度だけを特別扱いする重みではありません．
この論文の HG + Draine 近似については，fogbow / glory に関わる弱い後方ピークを捉えられない
ことも本文に明記されています．この損失を使えば細い虹のピークが必ず戻る，という根拠にはなりません．
対象も波長・粒径分布について平均した Mie 散乱であり，今回の一波長・方位角依存の教師とは異なります．

### 現在の NLL と違う点

以下の p と q は，いずれも立体角に関する正規化された密度です．

\[
L_{\rm NLL}=-E_p[\log q]=H(p)+D_{\rm KL}(p\Vert q).
\]

したがって現在の最尤学習は forward KL の最小化です．確率質量が小さい領域は，全体の
期待値への寄与も小さくなります．一方，log PDF 差の二乗を立体角上で積分すれば，同じ倍率の
誤差は密度の絶対値によらず同じ重みを受けます．これは細い低密度構造を重視する候補になります．
NLL が負になることは連続密度では正常で，学習率の順位を比較する妨げになりません．

### このプロジェクトへ拡張する場合の数式

以下は本プロジェクト向けの数式整理であり，論文がこの二次元 NF を実装したという意味ではありません．
mu = ki dot ko とし，正の教師 PDF に対して，例えば正規化した全球の目的を

\[
E_{\log}=\frac{1}{2\pi}\int_{S^2}|\mu|
\left[\log q(\omega)-\log p(\omega)\right]^2\,d\Omega
\]

と定義できます．積分係数は integral(abs(mu) dOmega) = 2 pi となるためです．
軸対称の場合は，元の角度和に対応する角度積分になります．

球面一様サンプルなら，次の推定量です．

\[
\widehat E_{\log}=\frac{2}{M}\sum_{n=1}^{M}|\mu_n|
\left[\log q(\omega_n)-\log p(\omega_n)\right]^2,
\qquad \omega_n\sim U(S^2).
\]

この場合に sin theta をさらに掛ける必要はありません．
現在と同じ CDF 由来の p サンプルを流用する場合は，正しい積分には

\[
\widehat E_{\log}=\frac{1}{2\pi M}\sum_{n=1}^{M}
\frac{|\mu_n|}{p(\omega_n)}
\left[\log q(\omega_n)-\log p(\omega_n)\right]^2
\]

という重要度補正が必要です．低密度域の 1/p により分散が大きくなり得ます．
CDF 点群上で無補正の log 二乗誤差を平均すると，依然として p による重みが残り，
論文に対応する立体角積分とは異なる目的になります．

表を使う場合も，u = (1 − cos theta)/2，v = (phi + pi)/(2 pi) では
dOmega = 4 pi du dv なので，実際の非一様 `u_edges` に対応するセル立体角で積分します．
角度画像の各ピクセルを等重みとした MSE ではありません．

### ゼロ密度，重み付け，実装前に決めること

教師 p = 0 のセルが正の立体角を占めると，正の q に対して log 二乗損失は定義できません．
本文・補足資料から，今回にそのまま適用できるゼロ処理の規約は確認できません．
まず追加したゼロ立体角診断を見て，対象と目的の定義を決める必要があります．
epsilon を入れることやゼロセルを捨てることは，無変更の実装ではありません．

後方の CDF 点群の NLL だけに大きい重みを掛ける方法にも注意が必要です．

\[
-\int w(\omega)p(\omega)\log q(\omega)\,d\Omega
\]

の最適な密度は一般に p ではなく，w p を正規化した密度になります．
元の位相関数を推定する目的なら，補正なしの重み付き NLL を黙って導入すべきではありません．

学習損失を維持したまま帯域内を多くサンプルするには，領域ごとに質量 m_j と点数 n_j を決め，
p を各領域に条件付けた分布からサンプルし，

\[
\widehat L=-\sum_j\frac{m_j}{n_j}\sum_{k=1}^{n_j}\log q(\omega_{jk})
\]

と補正する層化も候補です．目的は元の NLL ですが，分散が必ず減る保証はなく，
低質量域の誤差に全体の NLL が鈍感である性質も残ります．この追加機能では未実装です．

教師が正で目的を厳密に定義できる場合は，NLL と上の log 二乗項を組み合わせる比較もできます．
表現力が無限ならどちらも q = p を最適解に持ちますが，有限の NF では誤差の配分が変わります．
lambda，積分点の配置，更新予算を明示した別実験として比較すべきです．今回の A/B が
終わる前に同時に損失も変えると，改善の原因を切り分けにくくなります．

## 6. 検証範囲

提供コードの検証結果は `reports/AB_SWEEP_VALIDATION.md` を参照してください．
合成 CDF の小規模 CPU 学習による選択・再開・描画テストは，実際の Rainbow CDF についての
CUDA 学習精度や，135 度付近の再現改善を証明するものではありません．
GPU の既存テストは，CUDA が利用できる環境で `python -m pytest -m cuda -q` を実行して確認します．
