import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import logging
import cv2
import datetime
import os
import re
import shutil
import imageio
import json
import sys
import csv
import sort_people

# tensorflow
import models
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ファイル出力ログ用
file_logger = logging.getLogger("message")

level = {0: logging.ERROR,
            1: logging.WARNING,
            2: logging.INFO,
            3: logging.DEBUG}

# 入力値
HEIGHT = 480
WIDTH = 640

def predict_video(now_str, model_path, centerz_model_path, video_path, depth_path, past_depth_path, interval, json_path, number_people_max, reverse_specific_dict, order_specific_dict, is_avi_output, end_frame_no, verbose):
    # 深度用サブディレクトリ
    subdir = '{0}/depth'.format(depth_path)
    if os.path.exists(subdir):
        # 既にディレクトリがある場合、一旦削除
        shutil.rmtree(subdir)
    os.makedirs(subdir)

    # ファイル用ログの出力設定
    log_file_path = '{0}/message.log'.format(depth_path)
    logger.debug(log_file_path)
    file_logger.addHandler(logging.FileHandler(log_file_path))
    file_logger.warning("深度推定出力開始 now: %s ---------------------------", now_str)

    logger.addHandler(logging.FileHandler('{0}/{1}.log'.format(depth_path, __name__)))

    # 映像情報取得
    org_width, org_height, scale_x, scale_y = get_video_info(video_path)

    logger.debug("org_width: %s, org_height: %s, scale_x: %s, scale_y: %s", org_width, org_height, scale_x, scale_y)

    for pidx in range(number_people_max):
        # 人数分サイズデータ出力
        size_idx_path = '{0}/{1}_{3}_idx{2:02d}/size.txt'.format(os.path.dirname(
            json_path), os.path.basename(json_path), pidx+1, now_str)
        os.makedirs(os.path.dirname(size_idx_path), exist_ok=True)
        sizef = open(size_idx_path, 'w')
        # 一行分を追記
        sizef.write("{0}\n".format(org_width))
        sizef.write("{0}\n".format(org_height))
        sizef.close()

    # フレーム開始INDEX取得
    start_json_name, start_frame, json_size = read_openpose_start_json(json_path)

    logger.debug("number_people_max: %s, json_size: %s, start_frame: %s", number_people_max, json_size, start_frame)

    # 深度アニメーションGIF用
    png_lib = []
    # 人数分の深度データ(実際に入っているのはintervalごと)
    pred_multi_ary = [[[[] for z in range(18)] for y in range(number_people_max)] for x in range(json_size - start_frame)]
    # 人数分のセンターZデータ(実際に入っているのはintervalごと)
    pred_multi_z_ary = [[[[] for z in range(18)] for y in range(number_people_max)] for x in range(json_size - start_frame)]
    # 人数分の深度推定XY位置データ(実際に入っているのはintervalごと)
    pred_multi_xy_ary = [[[[] for z in range(18)] for y in range(number_people_max)] for x in range(json_size - start_frame)]
    # 人数分の深度データ(実際に入っているのはintervalごと)
    pred_multi_frame_ary = [[] for x in range(json_size - start_frame)]

    # 深度用ファイル
    depthf_path = '{0}/depth.txt'.format(depth_path)
    depthzf_path = '{0}/depth_z.txt'.format(depth_path)

    past_depthf_path = None
    past_depthzf_path = None
    if past_depth_path is not None:
        past_depthf_path = '{0}/depth.txt'.format(past_depth_path)
        past_depthzf_path = '{0}/depth_z.txt'.format(past_depth_path)

    logger.info("past_depthf_path: %s", past_depthf_path)
    logger.info("past_depthzf_path: %s", past_depthzf_path)

    if past_depthf_path is not None and os.path.exists(past_depthf_path) and past_depthzf_path is not None and os.path.exists(past_depthzf_path):
        # 深度ファイルが両方ある場合、それを読み込む
        pdepthf = open(past_depthf_path, 'r')
        pdepthzf = open(past_depthzf_path, 'r')

        fkey = -1
        fnum = 0
        # カンマ区切りなので、csvとして読み込む
        reader = csv.reader(pdepthf)
        zreader = csv.reader(pdepthzf)

        for row in reader:
            fidx = int(row[0])
            if fkey != fidx:
                # キー値が異なる場合、インデックス取り直し
                fnum = 0

            pred_multi_ary[fidx][fnum] = [ float(x) for x in row[1:] ]

            # 人物インデックス加算
            fnum += 1
            # キー保持
            fkey = fidx
        
        fkey = -1
        fnum = 0
        for row in zreader:
            fidx = int(row[0])
            if fkey != fidx:
                # キー値が異なる場合、インデックス取り直し
                fnum = 0

            pred_multi_z_ary[fidx][fnum] = [ float(x) for x in row[1:] ]

            # 人物インデックス加算
            fnum += 1
            # キー保持
            fkey = fidx

        pdepthf.close()
        pdepthzf.close()
        
        # 自分の深度情報ディレクトリにコピー
        shutil.copyfile(past_depthf_path, depthf_path)
        shutil.copyfile(past_depthzf_path, depthzf_path)
    else:
        # --------------------------
        tf.reset_default_graph()

        # Default input size
        height = 288
        width = 512
        channels = 3
        batch_size = 1
        scale = 0

        # 縮小倍率
        scale = width / org_width
        logger.debug("scale: {0}".format(scale))

        height = int(org_height * scale)
        logger.debug("width: {0}, height: {1}".format(width, height))
        
        # FCRN用グラフ
        graph_FCRN = tf.Graph()
        with graph_FCRN.as_default():
            # 再設定したサイズでtensorflow準備
            # Create a placeholder for the input image
            input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

            # Construct the network
            net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
            saver_FCRN = tf.train.Saver()

            init_graph_FCRN = tf.global_variables_initializer()

        # FCRN用セッション
        sess_FCRN = tf.Session(graph=graph_FCRN)
        # 初期化
        sess_FCRN.run(init_graph_FCRN)
        # リストア
        saver_FCRN.restore(sess_FCRN, model_path)

        # # ---------------------------
        # # センターZ用グラフ
        # graph_centerz = tf.Graph()
        # with graph_centerz.as_default():
        #     # 深度用プレースホルダ
        #     phi_predict_ph = tf.placeholder(tf.float32, [None,1], name="phi_predict_ph")

        #     saver_centerz = tf.train.import_meta_graph(centerz_model_path + ".meta")
        #     ckpt = tf.train.get_checkpoint_state(os.path.dirname(centerz_model_path))

        #     init_graph_centerz = tf.global_variables_initializer()
        
        # # センターZ用セッション
        # sess_centerz = tf.InteractiveSession(graph=graph_centerz)
        # # 初期化
        # sess_centerz.run(init_graph_centerz)
        # # モデルリストア
        # saver_centerz.restore(sess_centerz, ckpt.model_checkpoint_path)
        # # 予測関数生成
        # centerz_y = create_centerz_model(sess_centerz, phi_predict_ph)
            
        # --------------------------

        # 深度ファイルがない場合、出力する
        depthf = open(depthf_path, 'w')
        depthzf = open(depthzf_path, 'w')

        # 動画を1枚ずつ画像に変換する
        cnt = 0
        cap = cv2.VideoCapture(video_path)
        while(cap.isOpened()):
            # 動画から1枚キャプチャして読み込む
            flag, frame = cap.read()  # Capture frame-by-frame

            # logger.debug("start_frame: %s, n: %s, len(openpose_2d): %s", start_frame, n, len(openpose_2d))

            # 深度推定のindex
            _idx = cnt - start_frame
            _display_idx = cnt - interval

            # 開始フレームより前は飛ばす
            if start_frame > cnt:
                cnt += 1
                continue

            # 終わったフレームより後は飛ばす
            # 明示的に終わりが指定されている場合、その時も終了する
            if flag == False or cnt >= json_size or (end_frame_no > 0 and cnt >= end_frame_no):
                break
            
            if ((_idx % interval == 0 and _idx < json_size) or (cnt >= json_size - 1)):
                logger.debug("_idx: %s", _idx)

                # 一定間隔フレームおきにキャプチャした画像を深度推定する
                logger.warning("深度推定 idx: %s(%s)", _idx, cnt)

                # キャプチャ画像を読み込む
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2Lab))
                img = img.resize([width, height], Image.ANTIALIAS)
                img = np.array(img).astype('float32')
                img = np.expand_dims(np.asarray(img), axis=0)

                # Use to load from npy file
                # net.load(model_path, sess_FCRN)

                # Evalute the network for the given image
                pred = sess_FCRN.run(net.get_output(), feed_dict={input_node: img})

                # 深度解析後の画像サイズ
                pred_width = len(pred[0][0])
                pred_height = len(pred[0])
                logger.debug("pred_width: %s, pred_height: %s", pred_width, pred_height)

                # 該当シーンのJSONデータを読み込む
                file_name = re.sub(r'\d{12}', "{0:012d}".format(_idx), start_json_name)
                _file = os.path.join(json_path, file_name)
                try:
                    data = json.load(open(_file))
                except Exception as e:
                    logger.warning("JSON読み込み失敗のため、空データ読み込み, %s %s", _file, e)
                    data = json.load(open("tensorflow/json/all_empty_keypoints.json"))

                for dpidx in range(len(data["people"]), number_people_max):
                    # 人数分のデータが無い場合、空データを読み込む
                    data["people"].append(json.load(open("tensorflow/json/one_keypoints.json")))

                for dpidx in range(number_people_max):
                    logger.debug("dpidx: %s, len(data[people]): %s", dpidx, len(data["people"]))
                    for o in range(0,len(data["people"][dpidx]["pose_keypoints_2d"]),3):
                        oidx = int(o/3)
                        if data["people"][dpidx]["pose_keypoints_2d"][o+2] > 0.3:
                            # 信頼度が足る場合

                            # オリジナルの画像サイズから、縮尺を取得
                            scale_org_x = data["people"][dpidx]["pose_keypoints_2d"][o] / org_width
                            scale_org_y = data["people"][dpidx]["pose_keypoints_2d"][o+1] / org_height
                            # logger.debug("scale_org_x: %s, scale_org_y: %s", scale_org_x, scale_org_y)

                            # 縮尺を展開して、深度解析後の画像サイズに合わせる
                            pred_x = int(pred_width * scale_org_x)
                            pred_y = int(pred_height * scale_org_y)
                            logger.debug("pred_x: %s, pred_y: %s", pred_x, pred_y)

                            if 0 <= pred_y < len(pred[0]) and 0 <= pred_x < len(pred[0][pred_y]):
                                depths = []
                                for x_shift in range(-3,4):
                                    for y_shift in range(-3, 4):
                                        if 0 <= pred_x + x_shift < pred_width and 0 <= pred_y + y_shift < pred_height:
                                            depths.append(pred[0][pred_y + y_shift][pred_x + x_shift][0])

                                # 周辺3ピクセルで平均値を取る
                                depth = np.average(np.array(depths))

                                # # センターZ再推定
                                # phi = np.array([np.array([depth])]).T
                                # yt = sess_centerz.run(centerz_y, feed_dict={phi_predict_ph: phi})

                                pred_multi_ary[_idx][dpidx][oidx] = depth
                                # とりあえず同じdepth値を出力
                                pred_multi_z_ary[_idx][dpidx][oidx] = depth
                                pred_multi_xy_ary[_idx][dpidx][oidx] = [pred_x, pred_y]
                                pred_multi_frame_ary[_idx] = pred[0]
                            else:
                                # たまにデータが壊れていて、「9.62965e-35」のように取れてしまった場合の対策
                                pred_multi_ary[_idx][dpidx][oidx] = 0
                                pred_multi_z_ary[_idx][dpidx][oidx] = 0
                                pred_multi_xy_ary[_idx][dpidx][oidx] = [0, 0]
                                pred_multi_frame_ary[_idx] = pred[0]
                        else:
                            # 信頼度が足りない場合
                            logger.debug("×信頼度 _idx: %s, dpidx: %s, o:%s, oidx: %s", _idx, dpidx, o, oidx)
                            pred_multi_ary[_idx][dpidx][oidx] = 0
                            pred_multi_z_ary[_idx][dpidx][oidx] = 0
                            pred_multi_xy_ary[_idx][dpidx][oidx] = [0, 0]
                            pred_multi_frame_ary[_idx] = pred[0]

                    # ------------------
                    # 深度データ
                    depthf.write("{0}, {1}\n".format(_idx, ','.join([ str(x) for x in pred_multi_ary[_idx][dpidx] ])))

                    # 深度データ(センターZ)
                    depthzf.write("{0}, {1}\n".format(_idx, ','.join([ str(x) for x in pred_multi_z_ary[_idx][dpidx] ])))

            # インクリメント        
            cnt += 1

        depthf.close()
        depthzf.close()

        cap.release()
        cv2.destroyAllWindows()

        sess_FCRN.close()
        # sess_centerz.close()

    # 基準深度で再計算
    # zファイルの方は基準深度再計算なし
    recalc_depth(pred_multi_ary, interval, json_size)

    # 並べ直したindex用配列
    sorted_idxs = [[-1 for y in range(number_people_max)] for x in range(json_size)]
    # フレームの画像（1区間分だけ保持）
    frame_imgs = [[] for x in range(interval) ]
    # 各関節の最も信頼度の高い値
    max_conf_ary = [[ 0 for x in range(18) ] for y in range(number_people_max)]
    max_conf_color_ary = [[ 0 for x in range(18) ] for y in range(number_people_max)]
    # 前回のXY位置情報
    past_data = [[] for y in range(number_people_max)]
    # 前回の深度
    past_depths = [[] for y in range(number_people_max)]
    # 前回の深度(センターZ)
    past_depths_z = [[] for y in range(number_people_max)]

    logger.info("人物ソート開始 ---------------------------")

    cnt = 0
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        # 動画から1枚キャプチャして読み込む
        flag, frame = cap.read()  # Capture frame-by-frame

        # logger.debug("start_frame: %s, n: %s, len(openpose_2d): %s", start_frame, n, len(openpose_2d))

        # 深度推定のindex
        _idx = cnt - start_frame
        _display_idx = cnt

        # 開始フレームより前は飛ばす
        if start_frame > cnt:
            cnt += 1
            continue

        # 終わったフレームより後は飛ばす
        # 明示的に終わりが指定されている場合、その時も終了する
        if flag == False or cnt >= json_size or (end_frame_no > 0 and _idx >= end_frame_no):
            break

        # フレームイメージをオリジナルのサイズで保持
        frame_imgs[_idx % interval] = np.array(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), dtype=np.float32)

        if _idx >= 0:
            # 一区間後になればソート処理実行
            all_now_data, all_now_depths, all_now_depths_z = sort_people.sort(cnt, _display_idx, _idx, sorted_idxs, now_str, interval, subdir, json_path, json_size, number_people_max, reverse_specific_dict, order_specific_dict, start_json_name, start_frame, pred_multi_ary, pred_multi_z_ary, pred_multi_xy_ary, pred_multi_frame_ary, frame_imgs, max_conf_ary, max_conf_color_ary, org_width, org_height, past_data, past_depths, past_depths_z, png_lib, verbose)                   

            # 出力し終わったら、今回データを過去データとして保持する。
            for pidx, sidx in enumerate(sorted_idxs[_idx]):
                past_data[sidx] = all_now_data[pidx]["people"][0]

                if _idx % interval == 0:
                    past_depths[sidx] = all_now_depths[pidx]
                    past_depths_z[sidx] = all_now_depths_z[pidx]

        # インクリメント        
        cnt += 1

    if is_avi_output:
        # MMD用背景AVI出力
        outputAVI(depth_path, json_path, number_people_max, now_str, start_frame, end_frame_no, start_json_name, org_width, org_height)

    if level[verbose] <= logging.INFO and len(pred_multi_frame_ary[0]) > 0:
        logger.info("creating Gif {0}/movie_depth.gif, please Wait!".format(os.path.dirname(depth_path)))
        imageio.mimsave('{0}/movie_depth.gif'.format(os.path.dirname(depth_path)), png_lib, fps=30)

# 基準深度で再計算
def recalc_depth(depth_ary, interval, json_size):
    # 基準となる深度(1人目の0F目平均値)
    # 深度0が含まれていると狂うので、ループしてチェックしつつ合算
    pred_sum = 0
    pred_cnt = 0
    for pred_joint in depth_ary[0][0]:
        if pred_joint > 0:
            pred_sum += pred_joint
            pred_cnt += 1

    # 1人目の0F目の場合、基準深度として平均値を保存
    base_depth = pred_sum / pred_cnt if pred_cnt > 0 else 0

    logger.info("基準深度取得: base_depth: %s, pred_sum: %s, pred_cnt: %s", base_depth, pred_sum, pred_cnt)   

    # 基準深度で入れ直し
    for fidx, pred_ary in enumerate(depth_ary):
        if fidx % interval == 0 or (fidx == json_size - 1):
            for pidx, pred_one in enumerate(pred_ary):
                # logger.info("fidx: %s, pidx: %s, len(pred_one) :%s", fidx, pidx, len(pred_one))

                for jidx, pred_joint in enumerate(pred_one):
                    depth_ary[fidx][pidx][jidx] -= base_depth



def outputAVI(depth_path, json_path, number_people_max, now_str, start_frame, end_frame_no, start_json_name, org_width, org_height):
    fourcc_names = ["I420"]

    if os.name == "nt":
        # Windows
        fourcc_names = ["IYUV"]

    # MMD用AVI出力 -----------------------------------------------------
    for fourcc_name in fourcc_names:
        try:
            # コーデックは実行環境によるので、自環境のMMDで確認できたfourccを総当たり
            # FIXME IYUVはAVI2なので、1GBしか読み込めない。ULRGは出力がULY0になってMMDで動かない。とりあえずIYUVを1GB以内で出力する
            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
            # 出力先AVIを設定する（MMD用に小さめ)
            out_path = '{0}/output_{1}.avi'.format(depth_path, fourcc_name)

            if os.name == "nt":
                # Windows
                op_avi_path = re.sub(r'json$', "openpose.avi", json_path)
            else:
                op_avi_path = re.sub(r'json/?', "openpose.avi", json_path)
            logger.info("op_avi_path: %s", op_avi_path)
            # Openopse結果AVIを読み込む
            cnt = 0
            cap = cv2.VideoCapture(op_avi_path)

            avi_width = int(org_width*0.32)
            avi_height = int(org_height*0.32)

            out = cv2.VideoWriter(out_path, fourcc, 30.0, (avi_width, avi_height))
            
            while(cap.isOpened()):
                # 動画から1枚キャプチャして読み込む
                flag, frame = cap.read()  # Capture frame-by-frame

                # 動画が終わっていたら終了
                if flag == False:
                    break

                # 開始フレームより前は飛ばす
                if start_frame > cnt:
                    cnt += 1
                    continue

                for pidx, lcolor, rcolor in zip(range(number_people_max) \
                        , [(51,255,51), (255,51,51), (255,255,255), (51,255,255), (255,51,255), (255,255,51), (0,127,0), (127,0,0), (102,102,102), (0,127,127), (127,0,127), (127,127,0)] \
                        , [(51,51,255), (51,51,255),   (51,51,255),  (51,51,255),  (51,51,255),  (51,51,255), (0,0,127), (0,0,127),     (0,0,127),   (0,0,127),   (0,0,127),   (0,0,127)]):
                    # 人物別に色を設定, colorはBGR形式
                    # 【00番目】 左:緑, 右: 赤
                    # 【01番目】 左:青, 右: 赤
                    # 【02番目】 左:白, 右: 赤
                    # 【03番目】 左:黄, 右: 赤
                    # 【04番目】 左:桃, 右: 赤
                    # 【05番目】 左:濃緑, 右: 赤
                    # 【06番目】 左:濃青, 右: 赤
                    # 【07番目】 左:灰色, 右: 赤
                    # 【08番目】 左:濃黄, 右: 赤
                    # 【09番目】 左:濃桃, 右: 赤
                    idx_json_path = '{0}/{1}_{3}_idx{2:02d}/json/{4}'.format(os.path.dirname(json_path), os.path.basename(json_path), pidx+1, now_str, re.sub(r'\d{12}', "{0:012d}".format(cnt), start_json_name))
                    # logger.warning("pidx: %s, color: %s, idx_json_path: %s", pidx, color, idx_json_path)

                    if os.path.isfile(idx_json_path):
                        data = json.load(open(idx_json_path))

                        for o in range(0,len(data["people"][0]["pose_keypoints_2d"]),3):
                            # 左右で色を分ける
                            color = rcolor if int(o/3) in [2,3,4,8,9,10,14,16] else lcolor

                            if data["people"][0]["pose_keypoints_2d"][o+2] > 0:
                                # 少しでも信頼度がある場合出力
                                # logger.debug("x: %s, y: %s", data["people"][0]["pose_keypoints_2d"][o], data["people"][0]["pose_keypoints_2d"][o+1])
                                # cv2.drawMarker( frame, (int(data["people"][0]["pose_keypoints_2d"][o]+5), int(data["people"][0]["pose_keypoints_2d"][o+1]+5)), color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=10)
                                # 座標のXY位置に点を置く。原点が左上なので、ちょっとずらす
                                cv2.circle( frame, (int(data["people"][0]["pose_keypoints_2d"][o]+1), int(data["people"][0]["pose_keypoints_2d"][o+1]+1)), 5, color, thickness=-1)
                
                # 縮小
                output_frame = cv2.resize(frame, (avi_width, avi_height))

                # 全人物が終わったら出力
                out.write(output_frame)

                # インクリメント
                cnt += 1

                if end_frame_no > 0 and cnt >= end_frame_no:
                    break

            logger.warning('MMD用AVI: {0}'.format(out_path))

            # 出力に成功したら終了
            # break
        except Exception as e:
            logger.warning("MMD用AVI出力失敗: %s, %s", fourcc_name, e)

        finally:
            # 終わったら開放
            cap.release()
            out.release()
            cv2.destroyAllWindows()


def create_centerz_model(sess, phi_predict_ph):
    # 隠れ層１の重み
    hidden1_dense_kernel = sess.graph.get_operation_by_name("hidden1_dense/kernel/Assign")
    hidden1_dense_kernel_weight = sess.run(hidden1_dense_kernel.inputs[0])
    logger.debug("hidden1_dense_kernel_weight: %s", hidden1_dense_kernel_weight)
    
    # 隠れ層１のバイアス
    hidden1_dense_bias = sess.graph.get_operation_by_name("hidden1_dense/bias/Assign")
    hidden1_dense_bias_weight = sess.run(hidden1_dense_bias.inputs[0])
    logger.debug("hidden1_dense_bias_weight: %s", hidden1_dense_bias_weight)
    
    # 出力層の重み
    outlayer_dense_kernel = sess.graph.get_operation_by_name("outlayer_dense/kernel/Assign")
    outlayer_dense_kernel_weight = sess.run(outlayer_dense_kernel.inputs[0])
    logger.debug("outlayer_dense_kernel_weight: %s", outlayer_dense_kernel_weight)

    # 出力層のバイアス
    outlayer_dense_bias = sess.graph.get_operation_by_name("outlayer_dense/bias/Assign")
    outlayer_dense_bias_weight = sess.run(outlayer_dense_bias.inputs[0])
    logger.debug("outlayer_dense_bias_weight: %s", outlayer_dense_bias_weight)

    # 予測関数の生成
    d_middle = 5000
    x = phi_predict_ph
    logger.debug("x: %s", x)
    hidden1 = tf.layers.Dense(units=d_middle,activation=tf.nn.relu,kernel_initializer=tf.constant_initializer(value=hidden1_dense_kernel_weight, dtype=hidden1_dense_kernel_weight.dtype),bias_initializer=tf.constant_initializer(value=hidden1_dense_bias_weight, dtype=hidden1_dense_bias_weight.dtype), name="restore_hidden1")
    x1 = hidden1(x)
    outlayer = tf.layers.Dense(units=1,activation=None,kernel_initializer=tf.constant_initializer(value=outlayer_dense_kernel_weight, dtype=outlayer_dense_kernel_weight.dtype),bias_initializer=tf.constant_initializer(value=outlayer_dense_bias_weight, dtype=outlayer_dense_bias_weight.dtype), name="restore_outlayer")
    y = outlayer(x1)

    # 一度重みを取らないとエラーになる？    
    ow = outlayer.get_weights()
    logger.debug("ow: %s", ow)
    
    return y

# Openposeの結果jsonの最初を読み込む
def read_openpose_start_json(json_path):
    # openpose output format:
    # [x1,y1,c1,x2,y2,c2,...]
    # ignore confidence score, take x and y [x1,y1,x2,y2,...]

    # load json files
    json_files = os.listdir(json_path)
    # check for other file types
    json_files = sorted([filename for filename in json_files if filename.endswith(".json")])

    # jsonのファイル数が読み取り対象フレーム数
    json_size = len(json_files)
    # 開始フレーム
    start_frame = 0
    # 開始フラグ
    is_started = False
    
    for file_name in json_files:
        logger.debug("reading {0}".format(file_name))
        _file = os.path.join(json_path, file_name)
        if not os.path.isfile(_file): raise Exception("No file found!!, {0}".format(_file))
        try:
            data = json.load(open(_file))
        except Exception as e:
            logger.warning("JSON読み込み失敗のため、空データ読み込み, %s %s", _file, e)
            data = json.load(open("tensorflow/json/all_empty_keypoints.json"))

        # 12桁の数字文字列から、フレームINDEX取得
        frame_idx = int(re.findall("(\d{12})", file_name)[0])
        
        if (frame_idx <= 0 or is_started == False) and len(data["people"]) > 0:
            # 何らかの人物情報が入っている場合に開始
            # 開始したらフラグを立てる
            is_started = True
            # 開始フレームインデックス保持
            start_frame = frame_idx

            # ループ終了
            break

    logger.warning("開始フレーム番号: %s", start_frame)

    return json_files[0], start_frame, json_size


# 映像解析縮尺情報
def get_video_info(video_path):
    # 映像サイズを取得する
    cap = cv2.VideoCapture(video_path)
    # 幅
    org_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # 高さ
    org_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    logger.debug("width: {0}, height: {1}".format(org_width, org_height))

    # 学習に渡すサイズの縮尺
    scale_x = HEIGHT / org_height
    scale_y = WIDTH / org_width

    return org_width, org_height, scale_x, scale_y



def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='data/nyu.h5', dest='model_path', help='Converted parameters for the model', type=str)
    parser.add_argument('--centerz_model_path', dest='centerz_model_path', help='Converted parameters for the centerz model', type=str)
    parser.add_argument('--video_path', dest='video_path', help='input video', type=str)
    parser.add_argument('--json_path', dest='json_path', help='openpose json result path', type=str)
    parser.add_argument('--now', dest='now', help='now', default=None, type=str)
    parser.add_argument('--past_depth_path', dest='past_depth_path', help='past_depth_path', default=None, type=str)
    parser.add_argument('--interval', dest='interval', help='interval', type=int)
    parser.add_argument('--number_people_max', dest='number_people_max', help='number_people_max', type=int)
    parser.add_argument('--reverse_specific', dest='reverse_specific', help='reverse_specific', default="", type=str)
    parser.add_argument('--order_specific', dest='order_specific', help='order_specific', default="", type=str)
    parser.add_argument('--end_frame_no', dest='end_frame_no', help='end_frame_no', default=-1, type=int)
    parser.add_argument('--avi_output', dest='avi_output', help='avi_output', default='yes', type=str)
    parser.add_argument('--verbose', dest='verbose', help='verbose', type=int)
    args = parser.parse_args()

    logger.setLevel(level[args.verbose])

    # 間隔は1以上の整数
    interval = args.interval if args.interval > 0 else 1

    # AVI出力有無
    is_avi_output = False if args.avi_output == 'no' else True

    # 出力用日付
    if args.now is None:
        now_str = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    else:
        now_str = args.now

    # 日付+depthディレクトリ作成
    depth_path = '{0}/{1}_{2}_depth'.format(os.path.dirname(args.json_path), os.path.basename(args.json_path), now_str)
    os.makedirs(depth_path, exist_ok=True)

    # 過去深度ディレクトリ
    past_depth_path = args.past_depth_path if args.past_depth_path is not None and len(args.past_depth_path) > 0 else None

    # 強制反転指定用辞書作成
    reverse_specific_dict = {}
    if args.reverse_specific is not None and len(args.reverse_specific) > 0:
        for frame in args.reverse_specific.split(']'):
            # 終わりカッコで区切る
            if ':' in frame:
                # コロンでフレーム番号と人物を区切る
                frames = frame.lstrip("[").split(':')[0]
                # logger.debug("frame: %s", frame)
                # logger.debug("frames: %s", frames)
                # logger.debug("frame.split(':')[1]: %s", frame.split(':')[1])
                # logger.debug("frame.split(':')[1].split(','): %s", frame.split(':')[1].split(','))
                if '-' in frames:
                    frange = frames.split('-')
                    if len(frange) >= 2 and frange[0].isdecimal() and frange[1].isdecimal():
                        for f in range(int(frange[0]), int(frange[1])+1):
                            # 指定フレームの辞書作成
                            if f not in reverse_specific_dict:
                                reverse_specific_dict[f] = {}

                            # 人物INDEXとその反転内容を保持
                            reverse_specific_dict[f][int(frame.split(':')[1].split(',')[0])] = frame.split(':')[1].split(',')[1]
                else:        
                    if frames not in reverse_specific_dict:
                        # 該当フレームがまだない場合、作成
                        reverse_specific_dict[int(frames)] = {}

                    # 人物INDEXとその反転内容を保持
                    reverse_specific_dict[int(frames)][int(frame.split(':')[1].split(',')[0])] = frame.split(':')[1].split(',')[1]

        logger.warning("反転指定リスト: %s", reverse_specific_dict)

        paramf = open( depth_path + "/reverse_specific.txt", 'w')
        paramf.write(args.reverse_specific)
        paramf.close()

    # 強制順番指定用辞書作成
    order_specific_dict = {}
    if args.order_specific is not None and len(args.order_specific) > 0:
        for frame in args.order_specific.split(']'):
            # 終わりカッコで区切る
            if ':' in frame:
                # コロンでフレーム番号と人物を区切る
                frames = frame.lstrip("[").split(':')[0]
                logger.info("frames: %s", frames)
                if '-' in frames:
                    frange = frames.split('-')
                    if len(frange) >= 2 and frange[0].isdecimal() and frange[1].isdecimal():
                        for f in range(int(frange[0]), int(frange[1])+1):
                            # 指定フレームの辞書作成
                            order_specific_dict[f] = []

                            for person_idx in frame.split(':')[1].split(','):
                                order_specific_dict[f].append(int(person_idx))
                else:        
                    if frames not in order_specific_dict:
                        # 該当フレームがまだない場合、作成
                        order_specific_dict[int(frames)] = []

                        for person_idx in frame.split(':')[1].split(','):
                            order_specific_dict[int(frames)].append(int(person_idx))

        logger.warning("順番指定リスト: %s", order_specific_dict)

        paramf = open( depth_path + "/order_specific.txt", 'w')
        paramf.write(args.order_specific)
        paramf.close()

    # Predict the image
    predict_video(now_str, args.model_path, args.centerz_model_path, args.video_path, depth_path, past_depth_path, interval, args.json_path, args.number_people_max, reverse_specific_dict, order_specific_dict, is_avi_output, args.end_frame_no, args.verbose)

    logger.info("Done!!")
    logger.info("深度推定結果: {0}".format(depth_path +'/depth.txt'))

if __name__ == '__main__':
    main()

        



