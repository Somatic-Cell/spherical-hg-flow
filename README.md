# Spherical HG Flow

[`Somatic-Cell/rainbow`](https://github.com/Somatic-Cell/rainbow) が出力する
**一つの波長・一つの入射方向の CDF** から，HG を基底分布にした球面 NF を学習します．
保存 CDF の一次モーメント `hg.g` を読み込み，その値を固定したまま残差 NF を学習する構成です．

主経路は `inspect-rainbow` → `train-rainbow` → `evaluate-rainbow` です．
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
| 評価 | 独立な validation / test 点群，HG と NF の forward KL，NF サンプルによる重要度 ESS |
| 保存 | 推論・評価用の最良重みと，optimizer / RNG を含む再開用 checkpoint を分離 |

円周・区間 RQS は Rezende et al. (2020),
[*Normalizing Flows on Tori and Spheres*](https://proceedings.mlr.press/v119/rezende20a.html)
の Sections 2.1.2，2.2，2.3.1 で示される構成に基づきます．HG による再パラメータ化と
粒子の鏡映・軸方向の入射対称性は，本プロジェクトで明示的に加える制約です．
球面全体での密度の連続性や，論文中の全構成・全実験の再現を主張するものではありません．
数式，データ契約と実装の対応は [単一条件の仕様](docs/RAINBOW_SINGLE_CONDITION.md) を参照してください．

## セットアップ

Python 3.12 と，実行環境に対応する PyTorch を使用します．CPU で開始する場合の
Windows PowerShell の例です．仮想環境の有効化を行わず，その実行ファイルを直接呼びます．

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

CUDA で学習する場合は，最初の PyTorch インストールを
[PyTorch 公式案内](https://pytorch.org/get-started/locally/) の対応ビルドに変更し，
学習コマンドに `--device cuda` を指定します．CPU での検証結果を CUDA の検証結果とは扱いません．
Zuko は既存プロジェクトと同じ **1.6.0** を使い，ライブラリ全体の fork は不要です．

Linux では `python -m venv .venv` の後に `source .venv/bin/activate` を実行し，
同じ `pip install` と，以下の `phaseflow` コマンドを使用できます．
`requirements-validation.txt` は旧 v0.1 の検証環境の記録です．

## 一条件から学習する

`--record` には，データセット全体の親ディレクトリではなく，次の 4 ファイルを直接含む
一条件のレコードを指定します．点群の NPZ への事前変換は不要です．

| ファイル | 内容 |
|---|---|
| `metadata.json` | 波長，入射方向，フレーム，`hg.g`，生成・品質情報 |
| `phi_cdf.npy` | 方位角の周辺 CDF |
| `theta_given_phi_cdf.npy` | 方位角セルごとの条件付き CDF |
| `u_edges.npy` | `u=(1-cos(theta))/2` の実際のセル境界 |

```powershell
$record = 'D:\rainbow\output\records\i0000\w0000'
.\.venv\Scripts\phaseflow.exe inspect-rainbow --record $record
.\.venv\Scripts\phaseflow.exe train-rainbow --record $record --config configs/rainbow_single.json --output runs/rainbow_single
.\.venv\Scripts\phaseflow.exe evaluate-rainbow --record $record --checkpoint runs/rainbow_single/best.pt --samples 65536 --seed 2026 --output runs/rainbow_single/evaluation.json
```

パスは実際に生成したレコードに置き換えてください．`i0000/w0000` はパス構造の例で，
推奨する物理条件を意味しません．最初は波長と入射方向を両方固定し，
この一条件について CDF サンプラ，HG 基底，残差学習と独立評価を確認します．

`configs/rainbow_single.json` は再現可能な出発点です．実データで最適化済みの設定ではありません．
validation は同じ物理条件からの独立な点群で，条件間の汎化評価ではありません．
`g` は点群から再推定せず，保存された CDF に対応する値を使用します．

### 中断して再開する

```powershell
.\.venv\Scripts\phaseflow.exe train-rainbow --record $record --config configs/rainbow_single.json --output runs/rainbow_resume --max-steps-this-run 100
.\.venv\Scripts\phaseflow.exe train-rainbow --record $record --config configs/rainbow_single.json --output runs/rainbow_resume --resume runs/rainbow_resume/checkpoint.pt
```

再開には `checkpoint.pt` を指定します．`best.pt` は validation で選んだ評価・推論用の重みで，
optimizer と乱数状態を引き継ぐ再開用ファイルではありません．
再開時は同じ入力ファイルと設定を用い，別の実験では出力ディレクトリも分けます．

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
実行した検査と合成教師での数値結果は [検証報告](reports/RAINBOW_VALIDATION.md) に記録します．
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
| `src/phaseflow/sphere_model.py` | 外部 HG と円周・区間 coupling のモデル |
| `src/phaseflow/sphere_splines.py` | 円周境界と鏡映のパラメータ制約 |
| `src/phaseflow/hg.py`, `geometry.py` | HG と局所方向の数値処理 |
| `docs/MODEL.md`, `docs/DATA_CONTRACT.md` | 旧 v1 モデル・点群形式の記録 |
| `docs/EXPORT_FORMAT.md`, `cpp/` | 旧 v1 専用の native 形式・参照実装 |
| `examples/`, `reports/VALIDATION.md` | 旧 v0.1 の合成教師・検証記録 |

開発時の契約は [AGENTS.md](AGENTS.md) に記載しています．リポジトリ直下で
`pytest -q` と `ruff check .` を実行してください．C++ テストが通ることは，新モデルの
native 推論が実装されたことを意味しません．
