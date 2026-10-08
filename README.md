# Spherical HG Flow

[`Somatic-Cell/rainbow`](https://github.com/Somatic-Cell/rainbow) が出力する
**一つの波長・一つの入射方向の CDF** から，HG を基底分布にした球面 NF を学習します．
保存 CDF の一次モーメント `hg.g` を読み込み，その値を固定したまま残差 NF を学習する構成です．

主経路は `inspect-rainbow` → `train-rainbow` → `evaluate-rainbow` です．
学習と NF の評価は **NVIDIA GPU / CUDA を標準**にします．学習完了後には，教師 PDF，
CDF サンプル点群，NF の PDF を同じ座標とアスペクト比で比較する図も保存します．
既存の `demo` / `train` / `evaluate` / `export` は **旧 v1 モデル用**として残しています．
旧モデルの HG 用 MLP，方位角の絶対値への折り畳み，OptiX 向け export は，
新しい学習経路の仕様ではありません．

## 今回の構成

| 要素 | 単一条件 Rainbow ワークフロー |
|---|---|
| 教師 | 保存された全立体角 CDF のセル内一定な立体角密度 |
| 入力 | `metadata.json` と 3 個の `.npy` ファイルを持つ一条件のディレクトリ |
| HG | 外部の `metadata["hg"]["g"]` を固定；HG head と事前学習は不要 |
| 球面 NF | HG の累積確率座標の区間 RQS と，方位角の円周 RQS を交互に coupling |
| 円周 | 全周を保持し，周期境界で slope を共有，conditioner も周期化 |
| 対称性 | 鏡映をパラメータ共有で表現；上下の入射を同一視しない |
| 学習 | CDF から固定点群を生成し，立体角に関する NLL で最尤学習 |
| 実行 | CUDA が既定；固定点群を GPU に置き，MLP・HG・RQS・方向は FP32 |
| 評価 | 独立な validation / test 点群，HG と NF の forward KL，NF サンプルによる重要度 ESS |
| 可視化 | 保存 CDF の PDF，CDF 点群，NF PDF；ソルバ座標で共通の軸・対数色尺度 |
| 保存 | 推論・評価用の最良重みと，optimizer / RNG を含む再開用 checkpoint を分離 |
| 監視 | TensorBoard，JSONL，NLL / KL・勾配・学習率の learning curves |
| 精度比較 | 同じ重みの FP64 参照と FP16 重み丸めを固定点上で比較 |

円周・区間 RQS は Rezende et al. (2020),
[*Normalizing Flows on Tori and Spheres*](https://proceedings.mlr.press/v119/rezende20a.html)
の Sections 2.1.2，2.2，2.3.1 で示される構成に基づきます．HG による再パラメータ化と
粒子の鏡映・軸方向の入射対称性は，本プロジェクトで明示的に加える制約です．
球面全体での密度の連続性や，論文中の全構成・全実験の再現を主張するものではありません．
数式，データ契約と実装の対応は [単一条件の仕様](docs/RAINBOW_SINGLE_CONDITION.md) を参照してください．

## セットアップ

Windows のバッチは **Python 3.14 / PyTorch 2.14.1 / CUDA 13.0 wheel** を対象とします．
公式の [cu130 wheel 一覧](https://download.pytorch.org/whl/cu130/torch/) に
Python 3.14 / Windows x64 用ビルドがあります．既存の CUDA 版 PyTorch を使います．
インストール先・学習・監視の Python は，すべて `environment.bat` で一度だけ指定します．

```bat
set "PHASEFLOW_PYTHON=py"
set "PHASEFLOW_PYTHON_ARGS=-3.14"
```

既存の仮想環境を使う場合は，上の二行を次のように変更します．

```bat
set "PHASEFLOW_PYTHON=C:\path\to\your\environment\Scripts\python.exe"
set "PHASEFLOW_PYTHON_ARGS="
```

リポジトリ直下で実行してください．別のバッチから呼ぶ場合は `call` を付けます．

```bat
call execute.bat setup
call execute.bat check
```

`setup` は同じ Python へ本プロジェクトと `.[dev,monitor]` を editable install します．
既存の torch は正確なバージョンで指定して保持し，torch がない場合だけ要求する cu130 wheel を
インストールします．異なる既存ビルドを黙って入れ替えません．通常実行時にネットワーク経由の
インストールは行いません．`check` は実際の `sys.executable`，`phaseflow.__file__`，GPU を表示します．
本プロジェクトが別の Python や別 checkout に入っていた場合は，CDF 読み込み前に手順を示して止まります．

従来の `run_rainbow_python314.bat` も同じ `execute.bat` へ転送します．
`execute.bat` に別の `.venv\Scripts\python.exe` を直接書く必要はありません．
既定設定は `cuda` です．CUDA が利用できなければ理由を表示して停止し，自動的に CPU へ
切り替えません．複数の GPU がある場合は `--device cuda:1` のように選べます．
Zuko は既存プロジェクトと同じ **1.6.0** を使い，ライブラリ全体の fork は不要です．

Linux では使用する CUDA 版 PyTorch を入れた同じ Python から `python -m pip install -e ".[dev,monitor]"`
を実行し，`python -m phaseflow` を使います．ライブラリ自体の下限は Python 3.12 です．
`requirements-validation.txt` は以前の CPU 回帰検証の固定バージョン一覧で，Windows の
Python 3.14 / cu130 環境を上書きするためのファイルではありません．
CPU の数学・小規模テストを実行する場合だけ，対応する CPU 版 PyTorch と明示的な
`--device cpu` を使います．CPU での検証結果を CUDA の検証結果とは扱いません．

## 一条件から学習する

`--record` には，データセット全体の親ディレクトリではなく，次の 4 ファイルを直接含む
一条件のレコードを指定します．点群の NPZ への事前変換は不要です．

| ファイル | 内容 |
|---|---|
| `metadata.json` | 波長，入射方向，フレーム，`hg.g`，生成・品質情報 |
| `phi_cdf.npy` | 方位角の周辺 CDF |
| `theta_given_phi_cdf.npy` | 方位角セルごとの条件付き CDF |
| `u_edges.npy` | `u=(1-cos(theta))/2` の実際のセル境界 |

```bat
call execute.bat
```

`execute.bat` の `RECORD`，`CONFIG`，`OUTPUT`，`DEVICE` を実験に合わせて変更します．
現在の `RECORD` は push された `..\rainbow\datasets\drop_a1_i20_700nm_q1800x3600_c90x1800`
を保持しています．実験ごとに空の `OUTPUT` を指定してください．最初は波長と入射方向を両方固定し，
この一条件について CDF サンプラ，HG 基底，残差学習と独立評価を確認します．

`configs/rainbow_single.json` は再現可能な出発点です．実データで最適化済みの設定ではありません．
validation は同じ物理条件からの独立な点群で，条件間の汎化評価ではありません．
`g` は点群から再推定せず，保存された CDF に対応する値を使用します．

### GPU のデータ移動と精度

CDF の読み込み・検証と教師点群の初回生成は CPU 上で行い，固定学習点群と validation 点群を
GPU に一度転送します．minibatch の抽出と最適化は GPU 上で行い，更新ごとの点群の
CPU → GPU コピーを避けます．固定 pool が VRAM に収まる点数を指定してください．

提供設定の `training.dtype="float32"` は MLP の重みと計算精度です．
`model.spline_dtype="model"` と `model.geometry_dtype="model"` により，HG・RQS・方向と log PDF も
FP32 にします．極付近の微小角度は方向の横成分から計算し，`z` が 1 に丸まっただけでは
その情報を捨てません．外部 `g` の原値は metadata に保持し，実行用定数をその値から作ります．
CDF と教師の統計集計は FP64 のままです．TF32，AMP / FP16 は暗黙に有効化しません．

短い学習を終えたら，同じ重みで数値精度と FP16 重み丸めの影響を評価します．
次のバッチは `execute.bat` の `RECORD` / `OUTPUT` / `DEVICE` をそのまま使います．

```bat
call execute.bat precision
```

`precision.json` は同じ評価点での NLL / KL 差と標準誤差，log PDF 誤差分位点，
同じ乱数からの方向差，sample 時と eval 時の PDF 整合性を記録します．
`fp16_weights_vs_float32` は FP16 に丸めた重みを FP32 に戻して評価する比較です．
CoopVec の積和・活性化・丸め・行列レイアウトを再現した native 検証ではありません．

学習率，点数，batch size と更新回数は以前の出発点を維持しています．特定 GPU で速度を
最適化したハイパーパラメータではありません．CPU は小規模な正しさの検証用として残しています．

### 学習中の loss を監視する

学習とは別の CMD ウィンドウで実行します．

```bat
call monitor.bat
```

[http://127.0.0.1:6006](http://127.0.0.1:6006) で `runs/` 以下の実験を比較できます．
既定の `log_every=20` 更新で，各更新の生 NLL・勾配ノルム・学習率をまとめて反映します．
`eval_every=100` 更新で独立な validation と固定 training subset を評価します．
`nll/train_minibatch`，`nll/train_fixed_subset`，`nll/validation`，`kl/validation` と
`kl_standard_error/validation` が主な監視項目です．HG baseline，処理点数，経過時間も記録します．
勾配ノルムは clipping 前です．速度表示はその起動の評価・保存等を含む平均で，GPU kernel のベンチマークではありません．

`history.jsonl` は学習中から読み取れ，`history.json` は checkpoint 保存時の確定履歴です．
既定設定では終了・制御された中断の後に `learning_curves.png` を保存します．
図は次のコマンドでも再生成できます．

```bat
call environment.bat
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow plot-history ^
  --history "runs\rainbow_single_gpu\history.jsonl" ^
  --output "runs\rainbow_single_gpu\learning_curves.png"
```

連続密度の NLL は負になり得るため，loss と KL の縦軸は線形です．表示のためのクリップや
平滑化は保存データに加えません．KL の誤差棒は ±1.96 Monte Carlo SE で，seed 間のばらつきではありません．
TensorBoard が不要なら `training.tensorboard=false` としても JSON / JSONL は残ります．
[`SummaryWriter` の公式説明](https://docs.pytorch.org/docs/stable/tensorboard.html) も参照してください．

### 単一 CDF での探索

最初は全方向を含む一条件を用い，LR → 学習点数 → モデル容量の順に絞ります．
`training.data_seed=2026` を固定し，`training.seed` だけを変えれば，同じ CDF 点群で
初期化・minibatch の反復実験ができます．N を変えると現 sampler では小さい点群が大きい点群の prefix になり，
その性質をテストしています．条件エンコーディングは少数の入射角・波長を含む次段階で比較します．
探索候補，評価指標，鏡映の根拠と独自設計の範囲は [実験計画](docs/SINGLE_CDF_EXPERIMENTS.md) にまとめています．

### 学習率 A → 点数 B をバッチで自動比較する

`sweep.bat` の `RECORD` を，動作確認済みの `execute.bat` と同じ値にします．
`EXISTING_RUN` は既に学習した実験のディレクトリ，`OUTPUT` は新しい探索の保存先です．
最初のコマンドは既存の `best.pt` について，120〜150 度の教師確率質量と
教師 / HG / NF の角度断面図を追加します．学習は行いません．

```bat
call sweep.bat diagnose
call sweep.bat
```

探索は A で LR = `3e-4, 1e-3, 3e-3` を N = `65536` で比較し，
最小の **validation NLL** を得た LR で B の N = `4096, 16384, 65536, 262144` を比較します．
モデル，各 seed，batch size，更新回数と評価点群は基準設定を共有します．
B の N = `65536` は選ばれた A の同一実験を再利用するので，既定では 6 回の学習です．
学習損失は既存の NLL のままです．

結果は `runs/rainbow_sweep_gpu/summary.csv`，`summary.md`，`summary.json`，`summary.png` に集計し，
LR の選択根拠を `selection.json` に保存します．各試行には学習曲線，TensorBoard，
既存の 3 種類のマップと角度診断が残ります．別の CMD で次を実行すると全試行を監視できます．

```bat
call monitor.bat runs\rainbow_sweep_gpu
```

同じ `sweep.bat` を再実行すると，設定・データ・コード・実行環境が一致する探索を再開します．
同じ保存先へ複数プロセスを同時起動しないでください．設定を変える場合は新しい `OUTPUT` にします．
既定の B は更新回数を固定する比較で，epoch 数を固定する比較ではありません．
[探索手順・集計の読み方・135 度付近の診断と論文の損失](docs/SWEEP_AND_RAINBOW_LOSS.md)
に詳しい仕様を記載しています．

完了済みの A/B 探索について，**実際の固定学習点群を全点表示する図**を作る場合は，
`sweep.bat` の `RECORD` と `OUTPUT` をその探索に合わせて実行します．

```bat
call sweep.bat replot
```

保存済みの重みと設定を読み，`OUTPUT\training_point_plots\` に新しい図を作ります．
元の評価値，選択結果，旧図，学習記録は保持します．A と B が同じ試行を再利用した行は
同じ図を参照します．この可視化修正により探索コードの識別値が変わるため，旧探索を
通常の `sweep.bat` で再開する代わりに，図の更新には `replot` を使います．
新しい探索を始める場合は新しい `OUTPUT` を指定します．

### 点群を検査し，bin 数と最適化ステップ数を比較する

追加の `sweep.bat` モードで，実際の固定学習点群と CDF の確率質量を比較し，
続いて RQS bin 数と最適化ステップ数を探索できます．

```bat
call sweep.bat audit
call sweep.bat capacity
call sweep.bat samples
```

`audit` は `AUDIT_RUN` の元の学習結果を読みます．`capacity` は新しい出力先で
bin 数 16・32・64・128 を，同じ 262,144 点と 20,000 更新の計画で比較します．
2,000・6,000・10,000・20,000 更新時点の重みと図を保存して続けるため，短い実験を
最初からやり直しません．`samples` は完了した容量探索で選んだ bin 数を使い，
65,536・262,144・1,048,576 点を再比較します．損失は既存の NLL のままです．

[設定・結果の読み方・LUT の規模](docs/CAPACITY_SWEEP.md) を参照してください．
引数なしの `sweep.bat` は従来の学習率 A → 点数 B の探索を続けて使用できます．

### 完了後の 3 種類のマップ

予定した更新が完了すると，validation で選んだ最良重みから `runs/rainbow_single_gpu/plots/` に
次の図を自動保存します．

| ファイル | 内容 |
|---|---|
| `reference_pdf.png` | CDF のセル質量を立体角で割った教師 PDF，対数色表示 |
| `cdf_samples.png` | 実際の固定学習点群を，学習時と同じ N 点すべて表示 |
| `nf_pdf.png` | NF の PDF 評価，教師図と共通の対数色尺度 |
| `comparison.png` | 3 枚を並べた比較図 |
| `plots.json` | 描画設定，座標・密度の規約と入力情報 |
| `training_scatter.npz` | 描画した全学習点の NF 方向，ソルバ座標の角度，点群の検証情報 |

点群は保存された `training.train_samples`，`training.data_seed` と学習用 stream から再生成し，
学習時の SHA-256 と一致することを検証します．学習時と同じ geometry dtype に変換した方向を
全点描きます．N が `262144` ならその全点を表示し，点数の上限や間引きはありません．
これは固定 pool の各点を一度ずつ表示する図で，minibatch での反復回数を含む図ではありません．
`best.pt` と同じ実験の `checkpoint.pt`，`config.json`，`sample_split.json` が必要です．

横軸はソルバの方位角 `phi_s` の −180〜180 度，縦軸は散乱角 `theta` の 0〜180 度で，
前方の 0 度を上に表示します．全て同じ軸・範囲と横 : 縦 = 2 : 1 のアスペクト比です．
NF フレームの方位角はソルバ座標へ戻して描き，教師の厳密なゼロ密度は区別して表示します．

角度の長方形は等面積ではないので，点群の見かけの密集度には `sin(theta)` が含まれます．
散布図ではピークや谷の位置と方位角の対応を確認し，PDF の色の明るさと点の密集度が
そのまま一致すると解釈しないでください．定量的な精度には立体角に関する NLL / KL を使います．

図だけを再生成することもできます．このコマンドも NF の評価には既定で CUDA を使います．

```bat
call environment.bat
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow plot-rainbow ^
  --record "..\rainbow\datasets\drop_a1_i20_700nm_q1800x3600_c90x1800" ^
  --checkpoint "runs\rainbow_single_gpu\best.pt" ^
  --output "runs\rainbow_single_gpu\plots_training" --scatter training
```

`--write-pdf` で PDF 形式も保存します．学習時の自動出力を省略する場合は `--no-plots` を
指定します．描画の設定は学習設定の `visualization` にまとめています．既存設定の
`visualization.cdf_samples` と `seed` は，学習点群の点数・seed を変更しません．
独立なサンプラ診断を明示的に行う `--scatter independent` だけで使用します．
[図の仕様と設定](docs/RAINBOW_SINGLE_CONDITION.md#学習完了後の可視化) を参照してください．

### 中断して再開する

```bat
call execute.bat resume
```

再開には `checkpoint.pt` を指定します．`best.pt` は validation で選んだ評価・推論用の重みで，
optimizer と乱数状態を引き継ぐ再開用ファイルではありません．
再開時は同じ入力ファイルと設定を用い，別の実験では出力ディレクトリも分けます．
途中停止の時点では最終 test と三種類の PDF 比較図を作りません．学習曲線の履歴は残ります．
制御された停止には `train-rainbow --max-steps-this-run 200` を使い，予定した `steps` は変更しません．
学習を延長する可能性がある場合は，初めから例えば `steps=4000` として 2000 更新で一時停止し，
同じ設定で再開します．旧 version 2 / 3 checkpoint は評価・推論用に読み込めますが，
FP32 化した v0.4 の厳密な再開には新しい version 4 checkpoint を使います．

## PDF と座標の規約

PDF はすべて立体角 `sr^-1` に関する密度です．サンプリングした点に教師 PDF をもう一度
掛けることや，画像上の各セルを等重みとして NLL / KL を計算することはしません．
CDF のセル質量を立体角で割ることにより，任意方向で教師 PDF を評価できます．

`rainbow` の入射傾斜角を `alpha`，粒子の有向軸を `-y` とすると，入射条件は
`incident_cosine = sin(alpha)` です．フレームをメタデータから読み，方位角の対称面を
NF の局所座標に明示的に合わせます．詳細は
[座標と測度](docs/RAINBOW_SINGLE_CONDITION.md#座標と立体角の規約) を参照してください．

## 検証範囲と今後の接続

この変更では，合成した Rainbow 形式のレコードを用いてデータ契約と学習経路を検証します．
実行した検査と合成教師での数値結果は [FP32・監視の検証報告](reports/FP32_MONITORING_VALIDATION.md)
に記録します．[v0.3 の GPU 経路・可視化報告](reports/GPU_PLOTS_VALIDATION.md) と
[v0.2 のアダプタ・学習報告](reports/RAINBOW_VALIDATION.md) は以前の実装の記録です．
**実際のソルバ出力での近似精度，ソルバの物理的収束，CUDA / OptiX 上の速度は別の検証対象です．**
新しい単一条件モデルの checkpoint は，旧 `phaseflow export` / `model.pflow` 形式とは互換ではありません．
旧 export が新モデルを誤って書き出すことは拒否します．

複数条件の学習，条件エンコーディング，条件間の `g` の供給・補間，新モデルの native export と
OptiX 推論は今後の実装範囲です．今回のモデルから条件間の補間性能を推定しないでください．

| パス | 役割 |
|---|---|
| `docs/RAINBOW_SINGLE_CONDITION.md` | 現在のモデル・データ・学習・評価の仕様 |
| `configs/rainbow_single.json` | 単一条件の学習設定 |
| `src/phaseflow/rainbow.py` | CDF 読み込み・検証，教師 sample / PDF，フレーム変換 |
| `src/phaseflow/single_condition.py` | 単一条件の学習・評価・再開 |
| `src/phaseflow/plotting.py` | ソルバ座標に揃えた教師 PDF・CDF 点群・NF PDF の描画 |
| `src/phaseflow/monitoring.py` | TensorBoard / JSONL の学習監視・履歴からの図の再生成 |
| `src/phaseflow/precision.py` | 同じ重みの FP32 / FP64 比較・FP16 重み丸めの診断 |
| `environment.bat`, `execute.bat`, `monitor.bat` | 共通 Python によるセットアップ・学習・監視 |
| `src/phaseflow/sphere_model.py` | 外部 HG と円周・区間 coupling のモデル |
| `src/phaseflow/sphere_splines.py` | 円周境界と鏡映のパラメータ制約 |
| `src/phaseflow/hg.py`, `geometry.py` | HG と局所方向の数値処理 |
| `docs/MODEL.md`, `docs/DATA_CONTRACT.md` | 旧 v1 モデル・点群形式の記録 |
| `docs/EXPORT_FORMAT.md`, `cpp/` | 旧 v1 専用の native 形式・参照実装 |
| `examples/`, `reports/VALIDATION.md` | 旧 v0.1 の合成教師・検証記録 |

開発時の契約は [AGENTS.md](AGENTS.md) に記載しています．リポジトリ直下で
`pytest -q` と `ruff check .` を実行してください．C++ テストが通ることは，新モデルの
native 推論が実装されたことを意味しません．

GPU を利用できる環境では，次の検査で CUDA 上の精度・学習・再開・描画の評価経路を確認します．
最初に CUDA が使えることを必ず確認し，GPU テストが全て skip された結果を合格と扱いません．

```bat
call environment.bat
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m pytest -m cuda -q
```
