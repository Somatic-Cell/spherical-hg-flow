# 球面一様・pure log の 2 分岐継続学習

## 対象と変更範囲

これまでの `04-uniform-log-training-v1.patch` を適用して得た，version 5 の
`runs/rainbow_log_uniform_gpu/checkpoint.pt` が対象です．GitHub main だけでは
log/uniform の学習経路がないため，uniform 拡張が入った作業ツリーで使用してください．

この追加は **既存ソースを一切編集しません**．学習・HG・RQS・conditioner・損失の
実装と，それらの code fingerprint は変更されません．以前の encoding 拡張も，
親 checkpoint と現在の作業ツリーが一致している限り，そのまま引き継ぎます．

元の run は読み取り専用として扱い，別の出力先に二つの分岐を作ります．
元 checkpoint のファイルを完全にコピーし，各分岐の初期状態をそこから作成します．
これは「設定を変えない通常の exact resume」ではなく，**設定変更を記録した fork**です．
フォーク時に変更するのは以下だけです．

- `training_config.steps`: 2000 → 5000
- `training_config.learning_rate`: 分岐 A は 0.003，B は 0.001
- Adam の全 param group の `lr`: 上と同じ値

重み，Adam の一次・二次モーメントと step，minibatch RNG，Python/NumPy/PyTorch/CUDA
の乱数状態，学習・検証点群の定義，それまでの history と選択済み重みは保持します．
過去の履歴を新しい学習率で書き換えません．以後は変更後の設定で通常の exact resume
を行います．`branch.json` と `manifest.json` に変更前後・分岐元の SHA-256 を残します．

**今回の追加には knot 制約も新しい loss も含みません**．純 log・beta=1・全点球面一様，
現在の小さいモデルのまま，追加の最適化と学習率の効果を調べる比較です．

## 適用

配布 ZIP の `files` の中身をリポジトリ直下へコピーするか，パッチを適用してください．
両方を実行する必要はありません．editable install 済みなら再インストール不要です．

```bat
git apply --check --whitespace=error-all 05-uniform-continuation-branches-v1.patch
git apply --whitespace=error-all 05-uniform-continuation-branches-v1.patch
```

追加ファイル:

```
sweep_log_uniform_continue.bat
src/phaseflow/uniform_branch_sweep.py
tests/test_uniform_branch_sweep.py
docs/UNIFORM_CONTINUATION.md
```

## 開始

BAT の `PARENT` を完了した uniform run に，`OUTPUT` を別の新しいディレクトリにします．
`RECORD` は既定で空です．空なら checkpoint に記録された CDF のパスを使います．
CDF を移動した場合だけ，動作中の `train_log_uniform.bat` と同じ `RECORD` を指定します．

```bat
call sweep_log_uniform_continue.bat plan
call sweep_log_uniform_continue.bat check
call sweep_log_uniform_continue.bat
```

`plan` は checkpoint のメタデータを検査して計画を表示します．GPU の初期化や学習は
しません．`check` は GPU/runtime/code/CDF の同一性と，全学習点・教師値・両検証点群を
再生成してハッシュ一致まで確認します．出力 run は作りません．

引数なしで二つの run を逐次実行します．既定値は以下です．

| 分岐 | 1–2000 更新 | 2001–5000 更新 |
|---|---|---|
| `lr_0.003` | 同じ親 checkpoint を利用 | 学習率 0.003 |
| `lr_0.001` | 同じ親 checkpoint を利用 | 学習率 0.001 |

更新番号は最初の学習からの通算です．追加 5000 更新ではなく，各分岐 3000 更新を
追加します．追加学習は合計 6000 更新です．モデル構成・encoding・点数・batch size・
precision・評価間隔・勾配 clipping は **親 checkpoint の値を読み取り**，変更しません．
CUDA が既定で，CPU へ自動フォールバックしません．別 GPU/runtime への自動変換も
行いません．この比較中はライブラリを更新しないでください．

3,000 と 5,000 更新で，重み，数値指標，history，実際の全学習点による比較図と角度
断面を保存します．test は 5,000 更新の最後にだけ既存 trainer が評価し，候補選択には
使用しません．

## 出力と可視化

```
runs/rainbow_log_uniform_continue_gpu/
  manifest.json
  parent_checkpoint.pt                 # 元のファイルの完全なコピー
  summary.json / summary.csv / summary.md / summary.png
  selection.json                       # 二分岐の学習完了後だけ生成
  lr_0.003/
    branch.json
    initial_checkpoint.pt              # 宣言した設定変更だけを施した開始状態
    checkpoint.pt                      # 最後に保存した再開状態
    best.pt / best_by_log.pt / best_by_nll.pt
    history.json / metrics.json / sample_split.json / config.json
    continuation_timing.jsonl
    tensorboard/
    milestones/
      updates_3000/
      updates_5000/
        checkpoint.pt / best.pt / ...
        snapshot.json
        plots_best/comparison.png
        plots_last/comparison.png
        diagnostics_best/angular_profiles.png
        diagnostics_last/angular_profiles.png
        diagnostics_best/profile_120_160_1.png  # 3方位角の追加ズーム
        diagnostics_best/profile_120_160_2.png
        diagnostics_best/profile_120_160_3.png
  lr_0.001/
    ...
```

`plots_best` はその時点までの検証 log RMSE 最良モデルです．**親の 2000 更新も選択対象**
なので，それ以後改善しなかった場合は親が best のまま残ります．`plots_last` は通算
3000/5000 更新そのものの重みです．両者を区別して比較してください．

既存の 120–150 度の診断はそのまま残します．ピークが150度を超えた場合にも見えるよう，
別の表示専用の120–160度ズームを追加します．報告帯域，学習損失，検証基準は変えません．

元の CDF の PDF，**親と同じ全262,144点**（親の点数が異なる場合はその全点），NF PDF を
描画します．間引きません．一つの比較図内では教師とNFに同じカラースケールを用います．
異なる図のカラースケール範囲まで一律固定する処理ではありません．数値と色バーも
確認してください．

```bat
start "" "runs\rainbow_log_uniform_continue_gpu\summary.png"
start "" "runs\rainbow_log_uniform_continue_gpu\lr_0.001\milestones\updates_5000\plots_best\comparison.png"
start "" "runs\rainbow_log_uniform_continue_gpu\lr_0.001\milestones\updates_5000\plots_last\comparison.png"
call monitor.bat runs\rainbow_log_uniform_continue_gpu
```

## 中断と再開

同じ BAT を再実行します．新しいseedで再学習はせず，各分岐の `checkpoint.pt` から
再開します．保存済み milestone はハッシュを検証して再利用します．描画中に失敗した
場合，保存済み重みから描画をやり直し，重複する最適化は行いません．

```bat
call sweep_log_uniform_continue.bat
```

追加学習なしで保存済み milestone を描画する場合:

```bat
call sweep_log_uniform_continue.bat plot
```

集計のみの場合:

```bat
call sweep_log_uniform_continue.bat report
```

二つのプロセスから同じ OUTPUT を同時実行しないでください．OS の排他ロックで
重複書き込みを拒否します．BAT の最後に入力待ちはありません．

## 拒否するケース

- `best.pt` しかない（AdamやRNGを持たないため）．
- 親が2000更新で完了していない，混合サンプリング，NLL併用，betaが1以外．
- 元のモデル／trainerのコード，Python/PyTorch/NumPy/Zuko，CUDA/runtimeが変化した．
- 学習点群・教師値・検証点群・CDFのハッシュが変化した．
- OUTPUTが親runと同じ，その内部，または親runを内包する場所である．
- 完了したmilestoneの状態ファイルを上書きした．

指紋不一致を避けるために checkpoint の `code_fingerprint` を手で書き換えないでください．
コード不一致を自動的に無視するオプションは設けていません．

## 検証範囲

今回の配布に対する検証結果は `TEST_REPORT.md` を参照してください．既存研究の再現条件を
簡略化したモデルに差し替える処理はありませんが，この環境ではZukoが利用できないため，
実際のNFによる通しの継続学習テストは実行できていません．その統合テストも同梱し，
依存関係のある環境では `tests/test_uniform_branch_sweep.py` の最後のテストが実行されます．
