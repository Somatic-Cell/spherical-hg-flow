# 単一 CDF：coupling 層数，MLP 幅，学習率の夜間比較

この追加実験は，現在の単一条件モデルで容量と学習率の組合せを比較します．
条件付き NF，入射方向・波長の特徴グリッド，Optuna は今回の変更に含みません．
保存 CDF，外部 HG，鏡映のパラメータ共有，円周・区間 RQS，等重みの NLL を維持します．

追加先の基準は，公開リポジトリのコミット
`d38f2df3934abd85440ea07c4bcf6b2e51f515fe` です．
既存 trainer，`sweep.bat`，基準設定ファイルは変更せず，新しい入口から既存の容量探索を呼び出します．

## 実行方法

追加ファイルを同じリポジトリへ配置してください．既に `execute.bat setup` で
editable install 済みなら，再インストールは不要です．`environment.bat` の Python を共通に使います．

`sweep_architecture.bat` の先頭にある次の四つの値を確認します．

| 変数 | 既定値 |
|---|---|
| `RECORD` | `..\rainbow\datasets\drop_a1_i20_700nm_q1800x3600_c90x1800` |
| `CONFIG` | `configs\rainbow_architecture.json` |
| `OUTPUT` | `runs\rainbow_architecture_gpu` |
| `DEVICE` | `cuda:0` |

リポジトリ直下の CMD で，まず設定，CDF，共有 Python と GPU の確認を実行します．

```bat
call sweep_architecture.bat check
```

`check` は予定する全構成を検証し，学習せずに終了します．出力先に新しい実験ファイルは作りません．
成功したら，次を実行します．引数なしと `run` は同じです．

```bat
call sweep_architecture.bat
```

このバッチは実行時にパッケージをインストールせず，CUDA が使えなければ停止します．
CPU への自動切替はありません．既定では完了時にキー入力を待たないため，夜間の実行に使えます．
明示的に `PHASEFLOW_PAUSE=1` を設定した場合は，その指定を優先します．

終了後，集計だけを再作成する場合は次です．

```bat
call sweep_architecture.bat report
```

`report` は既存の実験の識別情報と保存済み結果を検証し，学習せずに集計します．
このバッチの `check`，`run`，`report` はすべて同じ Python，CDF，設定，GPU 指定を使います．
全構成が完了していない場合，`report` も非ゼロの終了コードを返します．

学習とは別の CMD から監視できます．

```bat
call monitor.bat runs\rainbow_architecture_gpu
```

ブラウザで <http://127.0.0.1:6006> を開きます．コンソール全体も保存したい場合は，
学習の起動を次のようにできます．各試行のログは，これとは別に出力先の `logs` に残ります．

```bat
call sweep_architecture.bat > architecture_console.log 2>&1
```

## 今回実行する 10 組

`configs/rainbow_architecture.json` の `base_config` は，設定ファイルと同じ
ディレクトリにある `rainbow_capacity.json` を指します．作業ディレクトリを基準とした相対指定ではありません．
その基準設定を読み，K，N，L，hidden features，学習率と milestone の指定を適用します．

| trial | coupling 層数 L | hidden features | 学習率 |
|---:|---:|---|---:|
| 01 | 4 | `[64,64]` | 0.001 |
| 02 | 4 | `[64,64]` | 0.003 |
| 03 | 4 | `[128,128]` | 0.001 |
| 04 | 4 | `[128,128]` | 0.003 |
| 05 | 8 | `[64,64]` | 0.001 |
| 06 | 8 | `[64,64]` | 0.003 |
| 07 | 8 | `[128,128]` | 0.001 |
| 08 | 8 | `[128,128]` | 0.003 |
| 09 | 16 | `[64,64]` | 0.001 |
| 10 | 16 | `[64,64]` | 0.003 |

最初の 8 組は L，幅，学習率の組合せ，最後の 2 組は幅 64 を維持して L を増やす比較です．
L=16 の全幅を調べる総当たりではありません．**既存の baseline も今回の出力先で学習し直すため，
実際に 10 本の学習を行います．** 以前の capacity/samples 出力ディレクトリを参照・変更しません．

| 共通条件 | 値 |
|---|---|
| bin 数 K | 64 |
| 固定学習点群 N | 262,144 |
| minibatch B | 1,024 |
| 最終更新数 T | 20,000 |
| 保存・比較する更新数 | 2,000，6,000，10,000，20,000 |
| seed / data seed | 415 / 2026 |
| validation / test | 32,768 / 65,536 点 |
| validation 評価間隔 | 100 更新 |
| optimizer | 既存の Adam；一定の学習率 |
| MLP | 二隠れ層，SiLU，skip connection なし |
| 精度 | MLP，HG，方向，RQS は FP32；教師と統計集計は FP64 |
| 可視化・TensorBoard | 有効 |

各試行の処理点数は `T*B = 20,480,000` です．`T*B/N = 78.125` は平均的な
pool 使用回数で，重複なく全点を一巡する epoch ではありません．10 本で合計 200,000 更新です．
途中時点は各学習の同じ軌跡から保存し，前半を別途学習し直しません．

学習点群と検証点群を共通にしますが，形状が違う MLP の初期パラメータが同一になるという意味ではありません．
一条件・一 seed の探索なので，最終的な条件付きモデルの最適値や seed 間の頑健性は確定しません．

## 今回 N=262,144 を選ぶ根拠

利用者が共有した `summary(3).json` は，K=64，L=4，`[64,64]`，学習率 0.003，
20,000 更新で，次の結果でした．全条件・全設定への一般化ではなく，次の比較の基準として使います．

| 学習点数 N | 最良 validation KL | 採用更新数 | 最終時点の validation KL | test KL |
|---:|---:|---:|---:|---:|
| 65,536 | 0.032382 | 8,500 | 0.035270 | 0.031974 |
| 262,144 | 0.025514 | 12,600 | 0.028147 | 0.025653 |
| 1,048,576 | 0.027569 | 19,500 | 0.028992 | 0.027249 |

選択には validation を使い，test は報告だけです．1,048,576 点で最良の更新時点が終盤にあることは，
継続的な改善や未収束の証明ではありません．同じ更新数でのこの結果から，多い点数が一般に不要とも結論しません．
幅や層数を変えた後には，点数の効果が変わる可能性もあります．

この JSON の虹帯域の観測点数，NF 帯域確率質量，帯域内 conditional KL は `null` です．
教師の帯域確率と期待点数だけから，干渉縞の再現精度が確認できたとは扱いません．

参照ファイルの SHA256：

```text
668786e77276658d62bda526d3c7493119b16018c0196a85e4ab7c187d421a1c
```

このファイルは判断根拠の記録であり，夜間 sweep の実行時の入力ではありません．

## 保存場所と結果の読み方

全体の出力先は `runs\rainbow_architecture_gpu` です．

| ファイル・ディレクトリ | 内容 |
|---|---|
| `manifest.json` | 全試行の計画，設定と入力の識別情報 |
| `inputs` | 各試行に渡す具体的な設定 |
| `logs\<trial_id>.log` | 各試行のコンソール出力 |
| `trials\<trial_id>` | 既存 capacity 実行による各試行の結果 |
| `summary.json` / `.csv` / `.md` / `.png` | 各試行の進捗と比較結果 |
| `selection.json` | 全試行完了後の validation に基づく選択 |

trial ID は例えば `trial_01_l4_h64x64_lr0.001` です．既定設定の最初の試行の図は次で開けます．

```bat
start "" "runs\rainbow_architecture_gpu\trials\trial_01_l4_h64x64_lr0.001\bins\k_64\milestones\updates_20000\plots\comparison.png"
```

この `comparison.png` は，保存 CDF の対数 PDF，**固定学習 pool の全 262,144 点**，
NF の対数 PDF を同じソルバ座標上に並べます．`cdf_samples=32768` という旧互換の設定値が
基準 JSON に残っていますが，学習 pool の可視化ではこの値で間引きません．
点群のハッシュを検証し，学習時の geometry 精度を適用して再生成します．

同じ milestone ディレクトリには `learning_curves.png`，`diagnostics\angular_profiles.png`，
`best.pt`，`checkpoint.pt` などが保存されます．学習中の TensorBoard は
`trials\<trial_id>\bins\k_64\training` 以下の履歴を参照します．

`updates_20000` の図と test は，20,000 更新までの **validation 最良の `best.pt`** に対応します．
最後の更新の `checkpoint.pt` と同じ重みとは限りません．過去の milestone に，その後の重みを流用しません．
best-so-far の曲線は定義上悪化しないので，安定性は latest validation と固定 train subset の履歴も見て判断します．

選択は，全構成が共通の最終更新数まで完了した後，validation NLL に基づいて行います．
今回のように同じ validation 点群で比較する場合，validation KL と NLL の順位は一致します．
test，重要度 ESS，画像や虹帯域の診断で自動的にモデルを選び直しません．
大域的な KL の改善だけで，弱いピークや干渉縞の改善を保証しません．角度断面も比較してください．

同じ比較図内の教師 PDF と NF PDF は対数色尺度を共有します．異なる試行の図では
自動選択された色範囲が異なる可能性があるため，カラーバーも確認してください．
完了済みの図を上書きすると保存物のハッシュが変わるので，再描画は別の出力先へ保存します．

## 中断，再開，失敗時の挙動

同じ設定・CDF・コード・実行環境のまま，同じコマンドを再実行します．

```bat
call sweep_architecture.bat
```

検証済みの完了試行は飛ばし，保存済み checkpoint のある未完了試行は，既存 trainer の再開処理を使います．
一回の起動で，各未完了・失敗試行を試すのは一度です．ある試行が例外終了した場合はログを残し，
ほかの試行を続けます．学習率，batch size，精度，点数を自動で変更して再試行することはありません．
全 10 組が成功するまで全体を完了とは扱わず，失敗・未完了があれば非ゼロで終了します．
原因を確認して同じコマンドを再実行すると，成功済みの試行を重複して学習しません．

`Ctrl+C` は全体の実行を中断します．子プロセスも停止させ，後続の試行を起動しません．
再開は最後に正常保存された checkpoint からです．保存前の更新は，その checkpoint からやり直します．
同じ出力先に二つの sweep を同時起動しないでください．

設定・入力・コードを変更した比較には新しい `OUTPUT` を使います．既存の識別情報を
書き換えて再開したように見せたり，既存の完了ディレクトリを別設定で上書きしたりしません．
容量不足で設定変更が必要になった場合も，新しい実験として記録します．

## 時間，関連研究，今回の範囲

前回の K=64/L=4/幅64 の実測は一試行約 25～27 分でした．L を増やすと逐次的な
coupling の処理も増え，幅を増やすと MLP の計算と中間値が増えます．10 組の時間を
単純に `27分 × 10` と見積もることはできません．

参考となる仮定計算では，時間が L にだけ比例するなら約 9 時間，MLP の積和数だけに
比例するなら約 13.3 時間です．これは実測予測の上下限ではなく，別々の仮定による例です．
GPU の使用率，評価，保存，可視化，メモリ，温度などで変わり，起床までの終了を保証しません．
中断・再開できる構成で，必要なら翌日も続けてください．

NPIS 2024 の emitter tail にある 128 bins，16 coupling layers，幅256，two residual blocks は，
今回の二隠れ層 MLP と同じ構成ではありません．本実験は現在のモデル内の L/H/学習率比較です．
論文と同じ残差ブロックを再現したという主張はしません．同論文の tail は環境画像ごとの
条件なし flow で，学習後に順写像・逆写像・PDF を表にしています．その速度実績は，
今回の条件付き NF の直接推論の速度を保証しません．

論文：Litalien et al.，
[*Neural Product Importance Sampling via Warp Composition*，§3.2，§4](https://arxiv.org/html/2409.18974v2)．

今回の学習経過時間は PyTorch の評価・保存等も含みます．OptiX/CoopVec の推論時間ではありません．
候補を絞った後，複数 seed，少数条件の条件付き学習，任意方向 PDF と sampling の native 推論を
それぞれ別に検証します．提供時に実行した検証の範囲・結果は，
[検証報告](../reports/ARCHITECTURE_SWEEP_VALIDATION.md)を参照してください．

## Python からの起動

Windows のバッチと同じ入口は，専用モジュールです．従来の `phaseflow` CLI の subcommand を
追加する方法ではないため，次の `-m phaseflow.architecture_sweep` を使います．

```bat
call environment.bat
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow.architecture_sweep ^
  --record "..\rainbow\datasets\drop_a1_i20_700nm_q1800x3600_c90x1800" ^
  --config "configs\rainbow_architecture.json" ^
  --output "runs\rainbow_architecture_gpu" ^
  --device cuda:0
```

計画の検証だけなら `--dry-run`，既存結果の集計だけなら `--report-only` を付けます．
両方を同時には指定しません．
