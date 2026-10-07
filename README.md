# Spherical HG Flow

条件付き Henyey–Greenstein (HG) 分布と，その残差を表現する
**球面上の RQS coupling normalizing flow** の学習・参照推論実装です．
点群を入力にするので，物理シミュレーションや CDF バッファの実装とは独立に開発できます．

```python
g = model.hg_g(conditions)
directions, log_pdf = model.sample_and_log_prob(conditions, num_samples=1024)
log_pdf_at_other_directions = model.log_prob(other_directions, conditions)
```

`conditions` の末尾次元は 2 で，順序は `[wavelength_nm, incident_cosine]`，
出力は局所座標系の単位方向ベクトルです．確率変数の自由度は **2**，
ベクトルの格納成分数は **3** です．返す PDF の基準測度は立体角 `sr^-1` です．

## 実装した構成

| 部分 | 実装 |
|---|---|
| 条件 | 波長 nm と，入射伝搬方向・粒子軸のなす角の余弦 |
| エンコーディング | `[0,1]` への正規化値 + Gaussian one-blob の区間積分 |
| HG head | 小さな ReLU MLP，`g_limit * tanh(raw)` |
| 基底 | 条件付き HG；解析的な CDF／逆 CDF と固有の球面 PDF |
| 残差 | Zuko `GeneralCouplingTransform` を拡張した交互の 2D coupling |
| スカラー写像 | `[0,1]` 上の rational-quadratic spline；解析的な逆写像 |
| 端点 | 位置 0/1 を固定し，両端を含む `K+1` 個の正の slope を学習 |
| 鏡映対称性 | 散乱面に対する方位角の折り畳みと，等確率の符号復元 |
| 軸方向の入射 | 入射傾角の `sin` を gate として，軸方向で方位角依存性を消す |
| 学習 | 平均散乱余弦で HG を事前学習 → HG を固定して残差 NLL を学習 |
| 重み共有 | `weights.safetensors`，仕様 JSON，独立にロード可能な `model.pflow` |
| 参照推論 | C++20 の loader/CLI と，CUDA からも呼ぶための allocation-free 数値ヘッダ |

Zuko は **1.6.0 に固定**しています．ライブラリを丸ごと fork せず，
MLP，lazy distribution，coupling の仕組みを利用する拡張として実装しています．
標準の NSF を名前だけ変更したモデルではありません．境界条件と対称性を持つ
`BoundedRQSTransform` / `AxialSymmetricCouplingTransform` が実際の学習経路です．

## 対象と前提

粒子の形状・サイズ・姿勢モデルを固定した，軸対称粒子の**非偏光スカラー位相関数**を対象にします．
粒子の上下対称性は仮定しません．`incident_cosine` の符号を保存します．
サイズなども変える場合は，データ契約・conditioner・export 仕様を合わせて拡張してください．

球面密度は，折り畳んだ座標チャートと鏡映の 2 分枝から構成します．
球面全体で滑らかな単一の微分同相写像であることや，出射方向の極での密度の連続性までは
保証しません．全立体角で正規化された密度としての変数変換を実装し，極の評価は `phi=0`
に統一しています．数学的な定義は [MODEL.md](docs/MODEL.md) に記載しています．

同梱の合成教師は，二つの HG と解析的な二次元角度分布の混合です．
**2012 年の虹の物理モデルを実装したものではありません．** 実データに対する再現精度や，
OptiX 上の実行速度をこの合成テストから推定しないでください．

## セットアップ

Python 3.12 以上と，対象環境に対応する PyTorch を使います．
PyTorch のインストール方法は [公式案内](https://pytorch.org/get-started/locally/) に従ってください．
CPU だけで試す場合の例です．

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
pytest -q
```

検証に使った依存関係の固定値は `requirements-validation.txt` に記録しています．
C++ との比較テストは `g++` または `clang++` を使用します．コンパイラがない場合は
該当テストが skip になるため，結果の `skipped` も確認してください．

## まず波長を固定して動かす

以下は **550 nm に固定して入射方向を変える**，短い動作確認です．
学習条件を軽くするため，モデルも `configs/smoke.json` で小さくしています．

```bash
phaseflow demo \
  --output data/fixed_wavelength.npz \
  --wavelengths 550 \
  --incident-cosines -0.75 -0.25 0.25 0.75 \
  --points-per-condition 2048

phaseflow train \
  --data data/fixed_wavelength.npz \
  --config configs/smoke.json \
  --output runs/fixed_wavelength

phaseflow evaluate \
  --data data/fixed_wavelength.npz \
  --checkpoint runs/fixed_wavelength/checkpoint.pt \
  --split validation \
  --sample-count 512 \
  --output runs/fixed_wavelength/evaluation.json

phaseflow export \
  --checkpoint runs/fixed_wavelength/checkpoint.pt \
  --output exports/fixed_wavelength
```

一つの入射方向に固定する場合は `--incident-cosines 0.25` とし，
`--wavelengths 400 500 600 700` のように波長を変えられます．両方を固定する場合は
config の `validation_fraction` を明示的に `0` にしてください．その評価は in-sample です．

もう少し大きな出発点は `configs/example.json` です．設定は最適化済みの推奨値ではなく，
比較実験の基準です．RQS の幅・高さの下限も表現能力に影響するため，記録して比較します．

## CDF 側との接続

CDF を使って得た，全立体角の正規化された目標分布からの方向サンプルを渡します．
NPZ 保存を経由する必要はありません．

```python
from phaseflow.data import PhasePointCloud
from phaseflow.model import ModelConfig
from phaseflow.training import TrainingConfig, train_model

cloud = PhasePointCloud(
    conditions=conditions,  # [C,2]: wavelength_nm, incident_cosine
    outgoing=local_unit_directions,  # [N,3]
    condition_index=condition_index,  # [N], values in [0,C)
    mode="target_samples",
    metadata={"source": "your_cdf_sampler", "normalization": "full_sphere"},
)
result = train_model(cloud, ModelConfig(), TrainingConfig(), "runs/cdf_samples")
```

`target_samples` に教師 PDF をもう一度重みとして掛けません．
方向の積分グリッドを使う場合だけ，`mode="quadrature"` とし，
`weights = phase_pdf * solid_angle_weight` の**積分質量**を渡します．
二つの形式，単位，座標変換，必須メタデータは [DATA_CONTRACT.md](docs/DATA_CONTRACT.md) を参照してください．

## 学習と再開

HG は `E[cos(scattering_angle)]` を教師に学習します．その後 HG を固定することで，
残差 NF と HG が同じ偏りを取り合い，`g` の解釈が曖昧になることを抑えます．
残差を学習した最終分布の平均余弦が，HG head の `g` と完全一致する制約はありません．
評価では両方を区別します．
教師の平均余弦が設定した `g_limit` に達する場合は，事前学習を開始せず設定の見直しを要求します．
教師値の切り詰めは行いません．

`joint_steps > 0` にすると，共同学習を明示的に追加できます．
その場合は HG の log PDF と HG 座標への依存を含む完全な NLL を使い，
正の `moment_regularization` を必須にしています．

条件の組を丸ごと train / validation に分けます．validation 条件は HG の事前学習にも
使いません．checkpoint には重み，optimizer，学習段階，乱数，サンプラー状態，条件 split，
データ fingerprint，config を保存します．

```bash
phaseflow train --data data/fixed_wavelength.npz --config configs/smoke.json \
  --output runs/resume_example --max-steps-this-run 60

phaseflow train --data data/fixed_wavelength.npz --config configs/smoke.json \
  --output runs/resume_example --resume runs/resume_example/checkpoint.pt
```

同一実行環境の CPU で，残差学習中の float64 再開と，共同学習中の float32 再開が，
中断しない実行の重みと bit-for-bit で一致することをテストしています．
環境をまたぐ厳密再現性や CUDA の同等性は保証していません．

## C++ / OptiX への重み共有

```bash
g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Icpp/include \
  cpp/src/model.cpp cpp/src/phaseflow_cli.cpp -o phaseflow_cli
./phaseflow_cli exports/fixed_wavelength/model.pflow
```

CLI は標準入力から，例えば次の行を受け取ります．

```text
g 550 0.25
sample 550 0.25 0.3 0.7
eval 550 0.25 0.6 0 0.8
```

`sample` は方向，log PDF，PDF，g を返し，`eval` は log PDF，PDF，g を返します．
JSON の配列順を推測する必要がないように，バイナリ仕様には行列配置，offset，
feature 順序，座標，単位，version を記録しています．詳細と OptiX 側の upload 手順は
[EXPORT_FORMAT.md](docs/EXPORT_FORMAT.md) を参照してください．

現段階では，Python の MLP/RQS はモデル dtype に従い，HG と方向座標は float64 です．
C++ 参照実装は FP32 の重みを読み，演算を double で行います．
強い前方散乱の精度確認を優先した構成です．CUDA ヘッダの nvcc コンパイル，
OptiX device upload，実レンダラとの結合，速度測定は別途必要です．

レンダラでは，NF を proposal として使う場合の `q` と，輸送式に使う物理位相関数 `p` を
区別してください．NEE の MIS には任意方向で評価した同じ `q` を使います．
物理位相関数そのものを NF で置き換えれば，その近似誤差は描画結果にも入ります．

## ファイル構成

| パス | 内容 |
|---|---|
| `src/phaseflow/model.py` | HG head，球面の sample/PDF，Zuko Distribution API |
| `src/phaseflow/coupling.py` | 軸方向の入射極限を扱う Zuko coupling 拡張 |
| `src/phaseflow/splines.py` | 両端 slope を含む bounded RQS と解析的逆写像 |
| `src/phaseflow/hg.py`, `geometry.py` | 安定化した HG と散乱座標系 |
| `src/phaseflow/data.py`, `training.py` | 点群，重み付き積分，段階学習，再開 |
| `src/phaseflow/export.py`, `cpp/` | 重み形式と native 参照推論 |
| `configs/`, `tests/`, `reports/` | 設定，検証，実際の実行記録 |

開発時の契約は [AGENTS.md](AGENTS.md)，検証手順は [DEVELOPMENT.md](docs/DEVELOPMENT.md)，
実行した結果は [VALIDATION.md](reports/VALIDATION.md) に記録しています．

## 参照

- [Zuko documentation](https://zuko.readthedocs.io/stable/)
- Rezende et al., [Normalizing Flows on Tori and Spheres](https://proceedings.mlr.press/v119/rezende20a.html), 2020.
- Durkan et al., [Neural Spline Flows](https://arxiv.org/abs/1906.04032), 2019.
- Müller et al., [Neural Importance Sampling](https://tom94.net/data/publications/mueller19neural/mueller19neural.pdf), 2019.
- [PBRT: Phase Functions](https://pbr-book.org/4ed/Volume_Scattering/Phase_Functions)
- [NromFlowHG2Mie](https://github.com/Somatic-Cell/NromFlowHG2Mie): safetensors と JSON による重み共有の考え方を参照．モデルコードは引き継いでいません．
