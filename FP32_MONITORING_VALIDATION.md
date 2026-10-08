# v0.4 FP32・学習監視の検証

検証日：2026-10-08．基準 commit：`84ca5105c56855b0c8e64e70fede33e828104311`．
この報告は新しい FP32 経路，実行バッチ，監視・精度診断に関するものです．
**教師は合成した小さな CDF です．実際の Rainbow ソルバの光学精度や実データ近似の結果ではありません．**

## 実装した変更

- `execute.bat` の `.venv` と setup の `py -3.14` の不一致を解消し，`environment.bat` を共通入口にしました．
  setup は指定 Python へ editable install し，既存 torch の正確なビルドを保持します．
  `sys.executable` と `phaseflow.__file__` を表示し，欠落・別 checkout を CDF 読み込み前に検出します．
- 新しい提供設定は MLP・HG・方向・RQS・返却 log PDF を FP32 にしました．外部 g の原値を保持し，
  HG の端点距離から実行定数を作ります．微小角度を transverse 成分から保つため，z が ±1 に丸まっても
  非零の横成分を捨てません．FP64 教師と評価統計は維持します．旧 checkpoint の推論精度は保持します．
- TensorBoard と JSONL，学習曲線の PNG を追加しました．raw minibatch NLL，固定 training subset と
  validation の NLL/KL，HG baseline，gradient norm，learning rate，処理点数・時間を記録します．
  ログを block 単位で同期し，再開時は checkpoint にある履歴へ戻します．
- `data_seed` を初期化・minibatch seed と分け，固定教師点での seed 反復を可能にしました．
- `precision-rainbow` と `execute.bat precision` は，同じ固定点で FP32/FP64 と FP16 重み丸めを比較します．
  FP16 重みを FP32 へ戻して演算する診断であり，CoopVec のエミュレーションではありません．

## 実行環境と検査

| 項目 | 実行した環境／結果 |
|---|---|
| OS / Python | Linux / CPython 3.14.8 |
| PyTorch | 2.14.1+cpu |
| Zuko / NumPy | 1.6.0 / 2.5.3 |
| Matplotlib / TensorBoard | 3.11.2 / 2.21.0 |
| `python -m pytest -q` | **287 passed，8 skipped**；skip はすべて CUDA が必要な検査 |
| `python -m ruff check .` | 成功 |
| `git diff --check` | 成功 |
| `python -m build` | wheel と sdist の作成成功 |
| TensorBoard event inspection | step 0～100，101 個の scalar step，順序の逆転なし |

直接依存の固定値は `requirements-validation-py314.txt` に記載しました．
CPU の正しさを確認した結果であり，Windows CMD，CUDA 実行，NVCC，OptiX の成功や速度は測定していません．
Python 3.12 / PyTorch 2.8 の既存 CI に加え，3.14 / 2.14.1 の CPU CI を設定しました．
この作業中に GitHub Actions がリモートで成功した，という報告ではありません．

数値検査には，HG の独立な半角式との比較，強い前方／後方散乱，非恒等 NF の sample/PDF，
RQS の順逆変換・Jacobian，seam・鏡映・軸入射，有限勾配，正規化と一次モーメントを含みます．
`g=±(1-2^-40)` のように FP32 の g や z が ±1 に丸まる場合も，端点距離と横成分を検査しています．
任意の極端な spline 圧縮を FP32 で解像できるという保証はせず，非有限値や表現できない方向を検出したら停止します．

監視の統合検査は，実際に event を読み取り，JSONL の live 読み取りと中断再開を照合しています．
checkpoint より後の架空の log を挿入した後に再開し，履歴から除去され，optimizer と数値履歴が
中断しない実行と一致することを確認しました．未完成の JSONL 最終行だけは live reader が待てるようにし，
完成した不正な行は黙って修正せずエラーにします．

バッチは ASCII / CRLF，label・継続行・引数の確認を行いました．helper の欠落 module，別 checkout，
setup のインストール順序，既存 torch の保持，インストール失敗などは mock で検査しました．
この確認でユーザーの Windows や GPU を操作したわけではありません．

## 小規模学習の条件

`tests/test_single_condition.py::smooth_teacher` が作る，次の密度の厳密なセル積分を教師にしました．

$$
p(\mu,\phi_s)=\frac{(1+0.6\mu)(1-0.65\cos(2\phi_s))}{4\pi}.
$$

theta 16 セル × 方位角 24 セルの保存 CDF を読みます．学習対象は，保存セル内で一定な PDF です．
元の解析関数そのものを連続評価して学習しているわけではありません．
外部 g は `0.1987210964506126`，条件ラベルは 550 nm / 入射傾斜 20 度です．

| 設定 | 値 |
|---|---:|
| Scalar coupling 層数 / bin 数 | 2 / 8 |
| MLP | `[16,16]`，SiLU，1,254 parameters |
| 重み・HG・RQS・方向 | FP32 |
| train / validation / test | 各 2,048 点 |
| proposal 評価 | 512 点 |
| Minibatch / updates | 256 / 100 |
| LR / eval・checkpoint・log 間隔 | 0.003 / 20 更新 |
| 初期化 seed / data seed | 415 / 2026 |
| Best checkpoint | validation で選択した step 100 |

提供する実データ用設定 `[64,64]` / L4 / K16 / train65,536 の実測結果とは区別してください．

## 学習と精度比較の結果

同じ独立 test 点で HG と NF を比較した結果です．± は一標準誤差です．

| 指標 | 結果 |
|---|---:|
| HG forward KL | 0.1153937 ± 0.0096884 nats |
| 学習後 NF forward KL | 0.0078552 ± 0.0032123 nats |
| HG からの paired NLL 改善 | 0.1075385 ± 0.0088151 nats/sample |
| Relative ESS（proposal 512 点） | 0.9770761 |
| sample / eval log PDF 最大絶対差 | 1.4305115e-6 |

![Learning curves from a synthetic CDF](fp32-monitoring/learning_curves.png)

![Synthetic CDF and learned NF](fp32-monitoring/comparison.png)

独立した精度診断用の教師 4,096 点，共通乱数 4,096 点で，固定重みを比較しました．

| 比較 | NLL 増加量 ± SE | 固定点 log PDF の最大絶対差 |
|---|---:|---:|
| FP32 − FP64 参照 | 8.4658e-8 ± 7.9164e-9 | 2.3877e-6 |
| FP16 重み丸め＋FP32 演算 − 元の FP32 | 2.5326e-7 ± 9.8271e-7 | 2.2697e-4 |

これらはこの小さな合成例の値です．実 CDF の鋭いピーク，別のモデル容量，CoopVec の許容誤差を
この値で保証しません．精度診断用の点と最終 test 点は別なので，二つの JSON の KL 推定値も異なります．
格子のセル中央で NF を積分した値は描画用の近似積分で，厳密な正規化の証拠にはしません．

生の設定・履歴・統計は [fp32-monitoring/](fp32-monitoring/) にあります．
再現はリポジトリ直下で，開発用依存を導入した Python から以下を実行します．

```python
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0, str(Path("tests").resolve()))
from test_single_condition import small_config, small_model, smooth_teacher
from phaseflow.rainbow import RainbowReference
from phaseflow.single_condition import train_single_condition
from phaseflow.plotting import RainbowPlotConfig
from phaseflow.precision import evaluate_precision

record = smooth_teacher(Path("runs/reproduce_v04_teacher"))
output = Path("runs/reproduce_v04")  # 未使用の出力先
model = replace(small_model(), spline_dtype="model", geometry_dtype="model")
cfg = small_config(dtype="float32", data_seed=2026, steps=100, eval_every=20,
                   checkpoint_every=20, log_every=20, tensorboard=True)
with RainbowReference(record) as reference:
    result = train_single_condition(reference, model, cfg, output,
        plot_config=RainbowPlotConfig(cdf_samples=2048, dpi=120))
    evaluate_precision(result.best_path, reference, output / "precision.json", device="cpu")
```

この再現コードは CPU の小規模検証です．本番の GPU 学習は `configs/rainbow_single.json` と
`execute.bat` を使い，実 CDF ごとの近似精度を測定してください．
