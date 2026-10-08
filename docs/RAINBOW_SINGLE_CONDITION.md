# Rainbow の一条件から学習する

この文書は，外部 HG と円周・区間 RQS を使う **単一条件モデル**の仕様です．
旧 `docs/MODEL.md` の HG head・折り畳み座標を使うモデルとは区別します．
今回固定する条件は一つの波長と一つの入射方向で，条件エンコーディングや条件間補間は学習しません．

## 対応するソルバ出力

入力契約は [`Somatic-Cell/rainbow` のコミット a9538941](https://github.com/Somatic-Cell/rainbow/tree/a9538941dcb9df2de6a4130fa66f625f0133c949)
の公開形式 `rainbow.phase_cdf.numpy.v2` に対応します．同じ形式と座標契約を持つ後続の生成物も
読み込めますが，異なる形式を推測して受け入れることはしません．`rainbow` の Python パッケージや
GPU ソルバを，NF の学習時にインストール・実行する必要はありません．

| ファイル | 型・形状 | 内容 |
|---|---|---|
| `metadata.json` | JSON object | `complete=true`，形式・座標・波長・HG・生成品質の情報 |
| `phi_cdf.npy` | `<f8`，`[N_phi+1]` | 方位角の周辺 CDF，セル境界の値 |
| `theta_given_phi_cdf.npy` | `<f8`，`[N_phi,N_theta+1]` | 方位角セルごとの条件付き CDF |
| `u_edges.npy` | `<f8`，`[N_theta+1]` | 半角座標の境界；一般に非等間隔 |

配列は little-endian float64，C order です．各レコードは全球面で正規化された，非偏光の
スカラー位相関数を表します．一つのファイルが波長・入射条件の同時分布を表すわけではありません．
`.part` ディレクトリや未完了レコードは受け入れません．

`RainbowReference` は，形式と座標規約，配列サイズ・型，CDF の単調性・端点，フレームの
直交性と向き，保存質量と `hg.g` の整合性を検証します．不正な値を並べ替える，CDF を
再正規化する，欠けた領域を補う，密度に floor を足す，という修復は行いません．
データは read-only に memory map し，条件付き CDF の検証や逆変換で
`sample_count × N_theta` の一時配列を作りません．

`quality`，`storage_processing`，`diffraction_policy` などの生成情報は変更せず記録します．
`complete=true` やアダプタの検証成功は，ソルバの物理的収束を保証するものではありません．
生成品質による採否は実験条件として判断し，NF 側が独自の物理的閾値を暗黙に課すことはしません．

## 座標と立体角の規約

### 保存 CDF の座標

入射・出射方向はともに光の伝播方向です．散乱角を `theta` とし，

$$
\mu=\mathbf k_i\cdot\mathbf k_o=\cos\theta,
\qquad
u=\frac{1-\mu}{2},
\qquad
v_s=\frac{\phi_s+\pi}{2\pi}
$$

を使います．`theta=0` は前方，`theta=pi` は後方です．座標変換に伴う立体角の関係は

$$
d\Omega=4\pi\,du\,dv_s
$$

です．正の密度としての測度を記載しており，`u` が `mu` と逆向きに増えることは
追加の負符号や損失の重みを意味しません．

周辺 CDF を `A`，条件付き CDF を `T`，`u_edges` を `U` とすると，セルの質量と立体角は

$$
P_{j,i}
=(A_{j+1}-A_j)(T_{j,i+1}-T_{j,i}),
\qquad
\Delta\Omega_i=\frac{4\pi}{N_\phi}(U_{i+1}-U_i).
$$

教師密度はセル内で

$$
p_\Omega(\omega)=\frac{P_{j,i}}{\Delta\Omega_i}
$$

です．これは保存 CDF が定める分布そのもので，セル中心の元の物理関数値を復元する式では
ありません．ゼロ質量セルの PDF は 0，log PDF は `-inf` です．

逆 CDF は方位角の周辺分布を選んでから，そのセルの条件付き分布を使います．
CDF の平坦区間は右側への探索で飛ばし，選んだセル内を `u` と方位角について線形補間します．
`theta` についての線形補間や，`U_i=i/N_theta` への置換は行いません．

CDF の確率に沿って生成した教師点は，学習損失内では重み 1 です．
教師 PDF や `sin(theta)` をもう一度掛けると，学習対象の分布が変わります．

### 記録されたフレームから NF フレームへの変換

粒子座標の上方向は `+y`，有向粒子軸は `a=-y` です．ソルバの入射傾斜 `alpha` の規約は

$$
\mathbf k_i=(\cos\alpha,-\sin\alpha,0),
\qquad
\eta=\mathbf a\cdot\mathbf k_i=\sin\alpha.
$$

粒子の上下対称性は仮定しないので，`eta` は符号を保持します．実際のアダプタは，角度から
理想的な軸を作り直す代わりに，記録されたフレームの入射軸から `eta=-M[1,2]` を求めます．
ソルバ内部の FP32 方向と保存用 FP64 フレームのわずかな差を上書きしません．

`sampling_frame_columns` は，外側の JSON 配列が通常の行列の **行** です．

```python
M = np.asarray(metadata["sampling_frame_columns"], dtype=np.float64)
e0, e1, ki = M[:, 0], M[:, 1], M[:, 2]
particle_direction = M @ source_local_direction
```

NF フレームは，`+z` が入射伝播方向，`+x` が粒子軸の入射方向に垂直な成分，
`+y = z cross x` となる向きです．アダプタは記録された transverse basis の中で
回転を求め，`source_to_nf` と `nf_frame` を明示的に保持します．

ソルバの正準フレームでは，軸に平行な入射を除いて

$$
(x_s,y_s,z_s)\longmapsto(y_s,-x_s,z_s),
\qquad
\phi=\operatorname{wrap}(\phi_s-\pi/2)
$$

になります．この回転後，粒子軸を含む面の鏡映は局所 `y -> -y` です．
ソルバの方位角にそのまま `phi_s -> -phi_s` を仮定することはしません．
回転は立体角を保存するため，PDF の値は変わりません．

厳密な軸方向入射では，記録された `e1` を NF の `+x` とする fallback を使います．
モデルはその場合の方位角依存を消すので，この任意の基準方向は密度に影響しません．

### 外部 HG の係数

保存分布の一次モーメントを

$$
g=\mathbb E_p[\mu]
=\sum_{j,i}P_{j,i}\bigl(1-U_i-U_{i+1}\bigr)
$$

で検証し，`metadata["hg"]["g"]` を固定係数として使用します．`g_source` は教師が
保存分布である今回の用途とは区別します．有効範囲は厳密に `-1 < g < 1` です．
HG head，点群からの再推定，`g_limit` へのクリップはありません．

投影面積・散乱断面積は別の物理量としてメタデータに保持します．正規化済みの CDF や
位相関数に，それらを重ねて掛ける処理はしません．

## 円周・区間 RQS と HG 基底

モデルは `SingleConditionSphereFlow`，設定は `SphereFlowConfig` です．
採用した論文との対応は次のとおりです．

| 構成 | 採用範囲 |
|---|---|
| 円周 RQS | Rezende et al. §2.1.2 の正の slope と周期端点の共有 |
| Coupling | §2.2 の区間・円周の積空間，保持座標に対する周期 conditioner |
| 球面 | §2.3.1 の円筒を使う再帰構成を `S²` に適用 |
| HG 座標 | 本プロジェクトの明示的拡張；外部 g の解析 CDF・逆 CDF |
| 鏡映と軸方向入射 | 本プロジェクトの明示的な物理対称性の制約 |

根拠は [原論文](https://proceedings.mlr.press/v119/rezende20a/rezende20a.pdf) と
[補足資料 Appendix A.1](https://proceedings.mlr.press/v119/rezende20a/rezende20a-supp.pdf) です．
Möbius 変換や exponential-map という別の分岐，論文中の実験一式はこの実装範囲に含みません．

### 変数変換と密度

HG の立体角密度を `h_g`，散乱余弦の CDF を `H_g` とします．

$$
h_g(\mu)=\frac{1-g^2}{4\pi(1+g^2-2g\mu)^{3/2}},
\qquad
H'_g(\mu)=2\pi h_g(\mu).
$$

NF のデータ座標は

$$
t=H_g(\mu)\in[0,1],
\qquad
v=\frac{\phi+\pi}{2\pi}\in S^1.
$$

ソルバの `u=(1-mu)/2` と，HG による `t=H_g(mu)` は別の座標です．
保存 CDF は教師方向の生成・評価に使い，HG の CDF はモデルの変数変換に使います．

`N` をこのデータ座標から一様円筒への変換とすると，モデルの密度は

$$
\log q_\Omega(\omega)
=\log h_g(\mu)+\log\left|\det D N(t,v)\right|.
$$

逆写像で生成する場合は

$$
b\sim U([0,1]^2),
\qquad
(t,v)=N^{-1}(b),
\qquad
\mu=H_g^{-1}(t)
$$

として方向を作り，

$$
\log q_\Omega(\omega)
=\log h_g(\mu)-\log\left|\det D N^{-1}(b)\right|
$$

を同時に返します．残差写像の初期値は恒等写像で，浮動小数点の丸めの範囲内で
学習開始時は HG 分布です．
追加の `sin(theta)`，`2*pi`，鏡映の確率因子はありません．

### 層順序と周期境界

データから基底への評価方向では，`t` の更新，`v` の更新を交互に行います．
`t` の区間 spline は端点 0 と 1 を固定し，両端を含む正の slope を学習します．
`v` の円周 spline は区間の lift として端点を固定し，端点の slope を共有します．
学習可能な全周 offset は追加していません．逆方向は層順を逆にして解析的な RQS 逆写像を使います．

各層は，変更しない側の座標だけを conditioner に渡します．そのため三角 Jacobian の
対角成分で密度を計算でき，sampling と任意方向の PDF 評価のどちらでも conditioner は
一層につき一回です．これは演算構成の性質で，両操作の GPU 実行時間が等しいという測定結果ではありません．

conditioner には Zuko の MLP と滑らかな SiLU を使用します．円周を保持して `t` を更新する
conditioner は，方位角自体の数値に代えて周期的な特徴を受け取ります．

### 鏡映対称性のパラメータ共有

物理的な鏡映対称性や，不変な基底と equivariant な写像から不変な密度を作る原理は既存です．
後者は [Köhler et al. (2020), Theorem 1](https://proceedings.mlr.press/v119/kohler20a/kohler20a.pdf)
を参照してください．以下の HG 座標・反転 RQS パラメータ共有・偶関数 conditioner の組合せは，
その原理を今回のモデルに適用するために加えた具体的な設計です．Rezende et al. の論文に
この雨滴用の構成がそのまま書かれているわけではなく，新規性を確立したとの主張も行いません．

既定値 `mirror_symmetry=true` では，`num_bins=2H` を全周の bin 数とします．
独立な `H` 個の幅・高さから，それぞれ

$$
(w_0,\ldots,w_{H-1},w_{H-1},\ldots,w_0)
$$

を作ります．各半周の幅・高さの合計は `1/2` です．knot の slope は閉じた列

$$
(d_0,\ldots,d_H,d_{H-1},\ldots,d_1,d_0)
$$

になり，円周 spline が `F(1-v)=1-F(v)` を満たします．異なる対称子午線の
`d_0` と `d_H` は独立で，同じ値を要求しません．

`t` 側の conditioner は `A*cos(phi)` を受け取り，鏡映で同じ値になります．
これらを合成することで `q(x,y,z)=q(x,-y,z)` を保ちます．確率変数は常に全周 `v` のままで，
絶対値座標への折り畳みや，別の確率変数による符号の復元はありません．

`mirror_symmetry=false` では通常の円周 spline を使用し，`t` 側は
`[A*cos(phi), A*sin(phi)]` を受け取ります．対称性を課さない比較実験として使用できます．
**アダプタは教師 CDF を対称化しません．** ソルバの保存分布に数値的な非対称性が残っていれば，
対称モデルでの近似誤差にその差も含まれます．

### 軸方向入射と出射極

入射に関する gate は

$$
A=\sqrt{(1-\eta)(1+\eta)}
$$

です．円周 spline の正に制約した幅・高さを `A` と一様幅で，slope を `A` と 1 で
凸結合します．厳密に `eta=±1` なら，円周変換は恒等写像で，`t` の conditioner も
方位角に依存しません．上下それぞれの入射条件で方位角一様な密度になります．

軸入射での方位角一様性は物理的な要請ですが，非軸入射まで上記の凸結合で減衰させる形は
本プロジェクトの設計選択です．例えば幅には `(1-A)/K` の下限が加わり，非軸入射でも
表現族を制限します．この gate は，物理から一意に導かれた最適形でも原論文の指定でもありません．

出射の極は座標が退化する零測度の点です．モデルの明示的な極への PDF クエリは
NF 座標の `phi=0` を代表値にします．教師のセル PDF はソルバ座標の `phi_s=0` を代表値にします．
この代表値の違いは積分・学習対象に影響しませんが，極での点比較は滑らかな極限の検証にはなりません．

正の有限な区間 endpoint slope は密度の発散を避けますが，方位角によらない極限や
球面全体での密度の連続性を保証しません．これは原論文の補足資料でも区別されています．
正の HG と有限な spline を使うため，教師の開領域における厳密なゼロ密度も完全には表現できません．

## 学習と保存

`configs/rainbow_single.json` の既定値は次のとおりです．最適化済みのハイパーパラメータではなく，
最初の比較に使用する設定です．

| パラメータ | 既定値 |
|---|---:|
| Coupling 層数 | 4 |
| 各区間・全円周の bin 数 | 16 |
| Conditioner の隠れ層 | `[64,64]`，SiLU |
| Bin 幅・高さの下限 | `1e-5` |
| Knot slope の下限 | `1e-4` |
| 固定学習点群 | 65,536 点 |
| Validation | 32,768 点 |
| 最終 test | 65,536 点 |
| 最終 proposal 評価 | 16,384 点 |
| Minibatch | 1,024 点 |
| 更新回数 | 2,000 |
| Optimizer / 学習率 | Adam / `1e-3` |
| Validation / checkpoint 間隔 | 100 更新 |
| Device | CUDA；利用できなければ停止 |
| Conditioner の重み・MLP | float32 |
| HG・方向・RQS・返却 log PDF | float32 |
| 教師 CDF・教師 log PDF・評価統計の reduction | float64 |
| Live log 間隔 / TensorBoard | 20 更新 / 有効 |
| 固定 training monitor subset | 学習 pool の先頭 8,192 点まで |
| 初期化・minibatch seed / data seed | 415 / 2026 |

Bin 下限は数値上の設定であると同時に表現可能な集中度を制限します．値を変更した比較では
他の設定とともに記録します．強い前方散乱があることだけを理由に，教師の前方領域を削除しません．

学習点群は最初に生成し，固定 pool から復元抽出した minibatch で最尤学習します．
毎更新で教師 CDF から新しい点群を生成する実装ではありません．点数の実験では，
`train_samples` を変え，更新回数・batch size・validation/test の点数と seed を揃えます．
これにより教師点数の効果と最適化に投入する更新量を分けて比較できます．

### GPU 上の学習経路

設定の `training.device` は `cuda`，`training.dtype` は `float32`，
`model.spline_dtype` と `model.geometry_dtype` は `model` が出発点です．CUDA が利用できない場合に，
自動的に CPU へ切り替える処理はありません．`cuda:1` などで使用する GPU を選べます．
CPU 上の小規模な数学・統合テストには `--device cpu` を明示します．

保存 CDF の読み込み・検証と教師点群の初回生成は CPU 上で行います．生成した固定学習点群と
validation 点群を選択した device に一度転送し，学習時の minibatch の抽出も同じ device 上で
行います．更新のたびに NumPy の点群を切り出して CPU から GPU へ転送しません．
したがって固定点群は VRAM に収まる必要があります．既定の方向は一点 12 byte の FP32，
validation と固定 training monitor subset の教師 log PDF は一点 8 byte の FP64 です．
これに MLP，optimizer，batch と評価統計の中間値などのメモリが加わります．

損失と勾配の有限性を device 上で記録し，log・validation・checkpoint・中断の境界でまとめて
CPU 側へ取り出します．不正な更新を検出した場合は更新位置を報告し，失敗した重みを
成功した checkpoint として保存しません．パラメータごとに CPU と同期する検査は避けます．
CUDA の Adam は foreach を使い，TF32 と AMP / FP16 は自動的に有効化しません．
GPU 向けのデータ移動・実行構成ですが，特定 GPU での速度を測定した設定ではありません．

v0.4 の FP32 経路では，極付近の `1-mu` / `1+mu` を方向の横成分から安定に求めます．
HG の逆写像も端点までの距離を保持して方向を再構成します．`z` が FP32 で 1 に丸まったこと
だけを理由に，非零の横成分を消しません．`hg.g` の元の Python binary64 値は metadata と
checkpoint に保持し，`1-g` と `1+g` の実行用定数をその値から直接 FP32 化します．
g のクリップ・教師点群の jitter・密度の floor は加えていません．

`geometry_dtype` を持たない旧 checkpoint は従来の FP64 幾何計算を保持します．
新方式の FP64 参照は両 dtype を `model` としたまま重みと入力を FP64 にして作ります．
`precision-rainbow` は同じ固定点で FP32 / FP64 と FP16 重み丸めを比較します．
FP16 重みを FP32 に戻して行う計算は重み丸め誤差を調べるもので，OptiX CoopVec の実行を
再現するものではありません．既存の native export は引き続き旧 v1 モデル専用です．

教師点群と proposal の乱数は NumPy PCG64 と `SeedSequence([data_seed, stream_id])` から作ります．
train，validation，test，proposal は独立な stream です．minibatch は別の stream から得た seed を
持つ，選択 device 上の専用 `torch.Generator` で抽出し，その状態を checkpoint に保存します．
提供設定は `data_seed=2026` を固定し，初期化・minibatch の `seed=415` と分けています．
`data_seed=null` は従来の `seed` と同じ値を使います．モデル反復で `seed` だけを変更すれば
同じ教師点・validation / test で比較できます．N の違いによる教師点の prefix 一致も検査しています．
CPU と CUDA の乱数列が一致するとは仮定しません．教師点群は 52-bit midpoint grid 上の
開区間一様値から生成するので，乱数の端点を丸めて回避する処理はありません．

重みは **validation NLL の最小値**で選びます．選択する時点は初期状態，`eval_every` ごと，
予定された最終更新です．一時停止を挟んだことだけを理由に候補時点を増やさないため，
中断なしの実行と同じ選択規則を保ちます．予定更新数が完了した後にだけ，選択された重みで
独立な最終 test を評価します．

| 出力 | 用途 |
|---|---|
| `config.json` | 適用したモデル・学習設定 |
| `data_summary.json` | 条件，フレーム，生成情報，入力 4 ファイルの SHA-256，CDF 検証結果 |
| `sample_split.json` | stream，点数，学習・validation 点群の hash |
| `history.json` | checkpoint 保存時の全更新の NLL・勾配・予算と予定評価の履歴 |
| `history.jsonl` | 既定 20 更新ごとに追記する同じ数値履歴；学習中にも読める |
| `monitoring.jsonl` | 起動単位の経過時間・処理点数・速度；数値履歴とは分離 |
| `tensorboard/` | 任意の TensorBoard scalar event；提供設定では有効 |
| `learning_curves.png` | NLL・KL と Monte Carlo 誤差・勾配・学習率の図 |
| `checkpoint.pt` | 最新の更新地点，optimizer，RNG，最良重みを含む再開用状態 |
| `best.pt` | validation で選択した評価・推論用モデル |
| `metrics.json` | 初期・最良 validation，完了フラグ，完了後の最終 test |
| `plots/` | 完了時に選択重みで作る教師 PDF・CDF 点群・NF PDF と比較図 |

再開は `checkpoint.pt` を使います．データの hash，設定，実装，記録された実行環境を
照合してから，点群と乱数状態を復元します．別のコード・device・設定へ移ることを，
同じ実験の厳密な再開として扱いません．`best.pt` は optimizer を含まないので再開できません．
checkpoint は primitive / tensor の状態を `weights_only=True` で読み込みます．
新しい再開状態は checkpoint version 4 です．以前の version 2 / 3 は，保存されたモデルの
精度設定を保持した評価・推論用として読み込めます．計算・監視・状態の仕様が変わっているため，
旧 version からの再開を同じ実験の厳密な続行とは扱いません．

再開時は checkpoint 内の履歴で JSONL を作り直し，TensorBoard も `purge_step=0` を使って
その scalar 履歴を再送します．異常終了の直前に log へ出たが checkpoint にはない更新を，
再開後の結果に混在させません．時間計測は起動ごとに分け，bit 単位で一致する履歴へは含めません．
`training.tensorboard=false` でも JSON / JSONL の監視は使えます．有効時に TensorBoard が
未インストールなら，学習前にセットアップ手順を示して停止します．

PyTorch / CUDA の再現性には実行環境による範囲があります．この実装は同じ設定・入力・実装・
記録環境での再開を検証対象とし，別の GPU や PyTorch バージョンでの同一乱数列・bit 単位の
同一結果は主張しません．[PyTorch 2.8 の再現性の説明](https://docs.pytorch.org/docs/2.8/notes/randomness.html)
も参照してください．

## 学習完了後の可視化

既定では，予定した更新が全て終わった後に `best.pt` と同じ validation 選択重みを使い，
`plots/` に次の画像を保存します．中断中の重みや test の結果で選択した重みではありません．

| ファイル | 表示内容 |
|---|---|
| `reference_pdf.png` | 保存 CDF のセル質量を実際の立体角で割った教師 PDF，対数色表示 |
| `cdf_samples.png` | 学習に使った固定 pool の全 N 点の散布図 |
| `nf_pdf.png` | 学習済み NF の任意方向 PDF 評価，対数色表示 |
| `comparison.png` | 上記 3 枚を並べた比較図 |
| `plots.json` | 可視化設定，入力と checkpoint の hash，座標・密度・サンプルの情報 |
| `training_scatter.npz` | 実際に描画した全 N 点の方向，角度，検証情報 |

全てのマップの横軸はソルバの方位角 `phi_s`，範囲は `[-180,180]` 度です．
縦軸は入射伝播方向からの散乱角 `theta`，範囲は `[0,180]` 度で，前方散乱の 0 度を上に
表示します．1 度の横・縦の長さを揃え，表示領域のアスペクト比を横 : 縦 = 2 : 1 にします．
NF 座標へ回転した点をそのまま描かず，記録されたフレームとの変換を使ってソルバ座標に戻します．

教師図は保存された全セルを使い，実際の `u_edges` から求めた散乱角の境界で表示します．
NF も同じ全セルの `u` の中点・方位角の中点で PDF を評価し，同じ境界で色を配置します．
角度方向の間引きはしません．`plots.json` に保存される NF の格子積分はこの中点則による
診断値であり，解析的な正規化の値や保存 CDF の厳密な積分とは区別します．

2 枚の PDF 図では正の値に対する **同じ LogNorm と同じ色範囲**を使います．
両図に存在する正の値の最小・最大を色範囲にし，パーセンタイルによる切り捨てはしません．
教師の厳密なゼロは灰色で区別し，対数表示のために正の floor を足しません．
全ての正の値が同じ場合だけ，表示用の色範囲を上下に広げます．PDF 自体は変更しません．
両方とも単位は `sr^-1` であり，画像上の角度面積あたりの密度ではありません．

点群は **学習に使った固定 pool 自体を，全 N 点，元の順序で一度ずつ**表示します．
学習時の `train_samples` と `data_seed`（未指定なら `seed`），学習用 stream = 0 を使い，
元の FP64 の方向と教師 log PDF を再生成します．両配列の SHA-256 が学習時に記録された
`training_points_sha256` と一致した場合だけ，学習と同じ geometry dtype に変換して描きます．
FP32 学習では，GPU に入力したものと同じ FP32 の方向が対象です．GPU tensor を
保存していたという主張ではなく，保存記録に対して検証した再生成です．

`best.pt` と同じ実験の `checkpoint.pt`，`config.json`，`sample_split.json` が必要です．
checkpoint の選択重み・step・条件，設定，入力 CDF と点群ハッシュの一致を確認します．
不一致や記録の欠落があれば，別の独立点群に置き換えずに停止します．点数の上限や間引きは
ありません．学習で同じ点を再使用した回数や，監視用 subset の点数とは区別します．
`plots.json` の `scatter` には使用した N，seed，ハッシュの一致，描画時の dtype と，
120〜150 度に入った実際の点数を記録します．`training_scatter.npz` は同じ全点を保存します．

`theta,phi_s` の長方形は等面積投影ではないため，点の見かけの密集度は
`p_Omega * sin(theta)` に従います．前方の PDF が大きい場所ほど必ず画面上の点が密になる，
という読み方はせず，ピーク・谷の位置や方位角の対応を確認します．定量比較は立体角に関する
NLL / KL を使います．

設定ファイルの最上位にある `visualization` で描画だけを調整できます．

```json
"visualization": {
  "enabled": true,
  "eval_batch_size": 16384,
  "dpi": 160,
  "write_pdf": false
}
```

旧設定にある `cdf_samples` と `seed` は互換性のため読み込めますが，学習点群の描画では
使用しません．`plots.json` にも unused と記録します．独立な CDF サンプラの診断を行う場合
に限り，CLI の `--scatter independent --samples 32768 --seed 2027` で明示します．
その図には Independent CDF diagnostic と表示され，学習点群の図とは区別されます．

`--no-plots` は完了後の自動描画を明示的に省略します．`--max-steps-this-run` で予定の途中で
停止したときには，自動図は作りません．学習済み checkpoint から図だけを作り直す場合は，
次のコマンドを使用できます．NF の評価は既定で CUDA，画像の構成と保存は Matplotlib で行います．

```bat
call environment.bat
"%PHASEFLOW_PYTHON%" %PHASEFLOW_PYTHON_ARGS% -m phaseflow plot-rainbow ^
  --record "..\rainbow\datasets\drop_a1_i20_700nm_q1800x3600_c90x1800" ^
  --checkpoint "runs\rainbow_single_gpu\best.pt" ^
  --output "runs\rainbow_single_gpu\plots_training" --scatter training
```

`--batch-size` は描画時の NF 評価の batch size，`--dpi` は出力解像度です．
`--write-pdf` を付けると，PNG に加えて PDF 形式の図も出力します．
checkpoint をコピーして別の場所に置いている場合は，`--training-run` で元の学習結果の
ディレクトリを指定します．A/B 探索の完了済み結果は `call sweep.bat replot` でまとめて
再描画できます．[既存探索の再描画](SWEEP_AND_RAINBOW_LOSS.md#完了済み探索を実際の学習点群で再描画する)
を参照してください．

## 評価指標の読み方

教師を `p`，NF を `q` とします．教師から独立に生成した `N` 点に対し，

$$
\widehat L
=-\frac{1}{N}\sum_{n=1}^N\log q_\Omega(\omega_n),
\qquad
\widehat D_{\mathrm{KL}}(p\Vert q)
=\frac{1}{N}\sum_{n=1}^N
\bigl(\log p_\Omega(\omega_n)-\log q_\Omega(\omega_n)\bigr)
$$

を計算します．CDF アダプタが教師 PDF を評価できるため，単なる NLL と区別して
forward KL を推定できます．HG 基底についても **同じ評価点**で計算し，NF による改善量を比較します．

| JSON の主な項目 | 解釈 |
|---|---|
| `nll` | 立体角に関する平均負の対数尤度；小さいほど良い |
| `forward_kl_estimate` | 教師から NF への forward KL の Monte Carlo 推定 |
| `hg_forward_kl_estimate` | 同じ教師点で評価した HG 単体の forward KL |
| `nll_improvement_over_hg` | 同じ点での HG から NF への NLL 改善；正なら改善 |
| `*_standard_error` | 評価点の独立性に基づく標準誤差 |
| `proposal.relative_ess` | NF の proposal としての近似精度を示す重要度 ESS の比率 |
| `proposal.importance_weight_mean` | `p/q` の標本平均；十分なサンプルで 1 に近づく指標 |
| `proposal.learned_moment_estimate` | NF サンプルから求めた一次モーメント |
| `proposal.sample_eval_log_pdf_max_abs_error` | 同じ NF 方向の sample-PDF と独立な PDF 評価の最大差 |

連続分布の NLL は負になり得ます．その符号は不正な学習や指標を意味しません．
真の KL は非負ですが，有限標本の推定値は小さく負になることがあります．0 に切り上げず，
標準誤差と合わせて読みます．記録される標準誤差は，学習 seed によるモデル間変動や
ソルバの誤差を含まないので，研究の比較では別の seed でも実験します．

proposal 評価では，NF 自身から `M` 点を生成して

$$
w_m=\frac{p_\Omega(\omega_m)}{q_\Omega(\omega_m)},
\qquad
\widehat r_{\mathrm{ESS}}
=\frac{(\sum_m w_m)^2}{M\sum_m w_m^2}
$$

を求めます．これは 1 に近いほど良く，正規化された位相関数の重要サンプリングを評価します．
全ての重みがゼロの標本では ESS を定義できず，`relative_ess=null` と状態を返します．
照明，遮蔽，NEE / MIS，経路長を含むレンダラ全体の分散を直接測る指標ではありません．

`evaluate-rainbow --samples N` の `N` は教師からの評価点数です．proposal の点数は
`--proposal-samples M` で独立に変更でき，省略時は同じ `N`，`0` なら proposal 評価を省略します．
初期の学習設定の `proposal_samples` は，学習完了時の最終 test に使用します．

独立評価は同じ条件内の点の汎化を測ります．新しい波長・入射条件への汎化の測定ではありません．
`g` は教師の一次モーメントに固定されますが，残差 NF の一次モーメントまで自動的に
同じ値に拘束されるわけではないため，その差も記録します．

## Python API

以下はデータ境界と推論を確認するための例です．学習は CLI または
`phaseflow.single_condition.train_single_condition` を使用します．

```python
import numpy as np
import torch

from phaseflow.rainbow import RainbowReference
from phaseflow.sphere_model import SingleConditionSphereFlow, SphereFlowConfig

with RainbowReference("path/to/one_record") as reference:
    rng = np.random.default_rng(2026)
    teacher_uniforms = (rng.integers(0, 2**52, size=(4096, 2)) + 0.5) / 2**52
    directions, teacher_log_pdf = reference.sample(teacher_uniforms)
    model = SingleConditionSphereFlow(
        hg_g=reference.g,
        incident_cosine=float(reference.condition[1]),
        config=SphereFlowConfig(spline_dtype="model", geometry_dtype="model"),
        dtype=torch.float32,
        device="cuda",
    )
    model_log_pdf = model.log_prob(torch.as_tensor(directions, dtype=torch.float32, device="cuda"))
    sampled, sampled_log_pdf = model.sample_and_log_prob(4096)
    teacher_at_model_samples = reference.log_prob(sampled.detach().cpu().numpy())
```

`RainbowReference.sample` の一様値の順序は `[phi_marginal, u_given_phi]` で，
両成分が `[0,1)` です．`SingleConditionSphereFlow.sample_from_uniform` の順序は
基底円筒の `[t_base, v_base]` で，前者が `(0,1)`，後者が `[0,1)` です．
この二つのサンプラで，同じ一様値を同じ物理方向へ対応させることは要求しません．

CDF サンプラへ明示的に 0 を与えた場合はセル境界や極を指すことがあり，その零測度の
代表方向の log PDF が `-inf` になる場合があります．開区間の入力から丸めによって
極やゼロ密度セルに入った場合は `FloatingPointError` を返します．学習経路は開区間の
midpoint grid を使い，いずれの場合も端点の置換や点の引き直しはしません．

参照データを開いている間は入力ファイルを変更しないでください．`with` を抜けた後の
memory map は使用できません．metadata は copy として取得でき，入力の履歴は checkpoint 側にも残ります．

## 数値精度と今後の範囲

v0.4 の提供設定では，重み・MLP・HG・方向・RQS・返却する log PDF は FP32 です．
`training.dtype="float32"`，`model.spline_dtype="model"`，`model.geometry_dtype="model"`
を明示します．教師 CDF，教師 log PDF と統計の集計は FP64 です．極付近は方向の横成分と
端点までの距離を保持し，`z` の丸めだけで微小角度を消さない計算を使います．

旧設定・checkpoint の `geometry_dtype` 省略時は FP64 幾何計算として解釈します．
FP64 参照を作る場合は両 dtype を `model` とした同じ重みを FP64 に変換し，固定点で比較します．
FP64 で別に学習する場合は `training.dtype="float64"` とし，学習軌跡も変わる別実験として扱います．
重み，幾何と spline の精度設定は checkpoint に保存します．

外部 `g` は Python の binary64 値として保存し，ネットワークを `.float()` にしても値を
丸めたり `g_limit` に置き換えたりしません．NF の逆変換に渡す開区間一様値は，実際の spline 精度が
FP64 なら 52-bit，FP32 なら 23-bit の midpoint grid を使います．

強い集中により内部の逆写像が極へ丸められた場合は，数値的な失敗として報告します．
別の方向へ置き換える，サンプルを捨てる，自動で引き直す，という処理は行いません．
モデルの validation を無効にした呼び出しでは，不正な lane は NaN です．
精度・bin 下限・集中度を確認してから，描画用の精度や速度を選びます．

このバージョンは，単一条件で実際に最尤学習を行う実装です．ただし，今回の環境での
データ契約・学習・再開の検証には合成レコードを用いており，実際の Rainbow の出力での
近似精度や物理再現性はまだ測定していません．合成レコードを物理的教師データとして報告しません．
[GPU 経路・可視化の検証報告](../reports/GPU_PLOTS_VALIDATION.md) に，実行済みの範囲と結果を記録します．

CUDA 用の検査は `cuda` marker を持ちます．実機では `python -c "import torch; assert torch.cuda.is_available()"`
で利用可能性を確認してから，`python -m pytest -m cuda -q` を実行してください．
これらは GPU 上の精度・学習・再開・描画時のモデル評価を検査し，CPU しかない環境では skip されます．
skip を CUDA の成功例に数えず，速度の評価は同じ GPU 上で別に測定します．

次の段階は，実際の一条件でサンプル数とモデル容量を調べた後，片方の条件を変える実験，
二条件モデルとエンコーディング，外部 `g` の条件間での供給を設計することです．
新モデルの C++ / CUDA / OptiX 用 export は未実装です．旧 format v1 の native 実装は
そのまま保持されますが，新 checkpoint からの export は拒否します．
バイナリ形式の変更と Python / native の一致を検証してからレンダラへ接続します．
