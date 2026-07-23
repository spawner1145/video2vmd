# depth_tracking_vmd

このプログラムは、[FCRN-DepthPrediction](https://github.com/iro-cp/FCRN-DepthPrediction) \(Iro Laina様他\) を miu(miu200521358) がfork して、改造しました。

動作詳細等は上記URL、または [README-original.md](README-original.md) をご確認ください。

## 機能概要

- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) で検出された人体の骨格構造から、深度を推定します。
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)の関節XY位置情報と、深度推定を元に、複数人数のトレースで人物追跡を行います。

## 準備

詳細は、[Qiita](https://qiita.com/miu200521358/items/d826e9d70853728abc51)を参照して下さい。

### 依存関係

python3系 で以下をインストールして下さい

- [OpenCV](http://opencv.org/)
- [tensorflow](https://www.tensorflow.org/) 1.0 ～ 1.6
- numpy
- python-dateutil
- pytz
- pyparsing
- six
- matplotlib
- opencv-python
- imageio

補足）以下プログラムが動作する環境であれば、追加インストール不要です。
 - [miu200521358/3d-pose-baseline-vmd](https://github.com/miu200521358/3d-pose-baseline-vmd)
 - [miu200521358/VMD-3d-pose-baseline-multi](https://github.com/miu200521358/VMD-3d-pose-baseline-multi)

## モデルデータ

[tensorflow用モデルデータ](http://campar.in.tum.de/files/rupprecht/depthpred/NYU_FCRN-checkpoint.zip)を「`tensorflow/data`」ディレクトリを作成し、以下に配置する

## 実行方法

1. [Openpose簡易起動バッチ](https://github.com/miu200521358/openpose-simple) で データを解析する
1. [VideoToDepth.bat](VideoToDepth.bat) を実行する
	- [VideoToDepth_en.bat](VideoToDepth_en.bat) is in English. !! The logs remain in Japanese.
1. `解析対象映像ファイルパス` が聞かれるので、動画のファイルフルパスを入力する
1. `解析結果JSONディレクトリパス` が聞かれるので、1.の結果ディレクトリパスを指定する 
	- `{動画パス}/{動画ファイル名}_{実行年月日}/{動画ファイル名}_json` が対象ディレクトリパス
1. `深度推定間隔` が聞かれるので、深度推定を行うフレームの間隔(整数のみ)を指定する
    - 指定された間隔ごとに深度推定を行う
    - 未指定の場合、デフォルトで「10」とする
    - 値が小さいほど細かく深度推定を行うが、その分処理が遅くなる
1. `反転フレームリスト`が聞かれるので、Openposeが裏表を誤認識しているフレーム範囲を指定する。
	- ここで指定されたフレーム範囲内のみ、反転判定を行う。
	- `10,20` のように、カンマで区切って複数フレーム指定可能。
	- `10-15` のように、ハイフンで区切った場合、その範囲内のフレームが指定可能。
1. `順番指定リスト` が聞かれるので、交差後に人物追跡が間違っている場合に、フレームNoと人物インデックスの順番を指定する。
	- 人物インデックスは、0F目の左から0番目、1番目、と数える。
	- `[12:1,0]` と指定された場合、12F目は、画面左から、0F目の1番目、0F目の0番目と並び替える、とする。
	- `[12-15:1,0]` と指定された場合、12～15F目の範囲で、1番目・0番目と並び替える。
1. `詳細なログを出すか` 聞かれるので、出す場合、`yes` を入力する
    - 未指定 もしくは `no` の場合、通常ログ
1. 処理開始
1. 処理が終了すると、`解析結果JSONディレクトリパス`と同階層に以下結果が出力される
	- `{動画ファイル名}_json_{実行日時}_depth`
	    - depth.txt …　各関節位置の深度推定値リスト
	    - message.log …　出力順番等、パラメーター指定情報の出力ログ
	    - movie_depth.gif　…　深度推定の合成アニメーションGIF
	        - 白い点が関節位置として取得したポイントになる
	    - depth/depth_0000000000xx.png … 各フレームの深度推定結果
	    - ※複数人数のトレースを行った場合、全員分の深度情報が出力される
	- `{動画ファイル名}_json_{実行日時}_index{0F目の左からの順番}`
	    - depth.txt …　該当人物の各関節位置の深度推定値リスト
1. message.log に出力される情報
	- ＊＊05254F目の出力順番: [5254:1,0], 位置: {0: [552.915, 259.182], 1: [654.837, 268.902]}
		- 5254F目では、1, 0の順番に割り当てられた
			- 0番目に設定されている1は、[654.837, 268.902]の人物が推定された
			- 1番目に設定されている0は、[552.915, 259.182]の人物が推定された
		- このフレームの人物立ち位置が間違っている場合、[5254:0,1]を、`順番指定リスト`に指定すると、5254Fの出力順番が反転される
	- ※※03329F目 順番指定あり [1, 0]
		- 3229F目を、`順番指定リスト`で、[1,0]と指定してあり、それに準じて出力された
	- ※※04220F目 1番目の人物、下半身反転 [4220:1]
		- 4220F目を、`反転フレームリスト`で指定してあり、かつ反転判定された場合に反転出力された


## ライセンス
Simplified BSD License

MMD自動トレースの結果を公開・配布する場合は、必ずライセンスのご確認と明記をお願い致します。Unity等、他のアプリケーションの場合も同様です。

[MMDモーショントレース自動化キットライセンス](https://ch.nicovideo.jp/miu200521358/blomaga/ar1686913)
