import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    # 引数に2枚の画像の名前を入れる
    img1 = cv2.imread(sys.argv[1]) # 画像1の読み込み
    img2 = cv2.imread(sys.argv[2]) # 画像2の読み込み

    #特徴抽出機の生成
    detector = cv2.xfeatures2d.SIFT_create() # SIFTを使うとき
    # detector = cv2.xfeatures2d.SURF_create() # SURFを使うとき

    #kpは位置座標 destは特徴を現すベクトル
    start = time.time()        # 計算時間を計りはじめる
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    print(time.time() - start) # 計算時間を表示

    #特徴点の比較機
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    #割合試験を適用
    good = []
    match_param = 0.6
    for m, n in matches:
        if m.distance < match_param*n.distance:
            good.append([m])
    #cv2.drawMatchesKnnは適合している点を結ぶ画像を生成する
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

    # 点の対応を表した画像を表示
    plt.imshow(img3)
    plt.show()

    # 画像を保存 (連結した画像と点の対応を表した画像)
    cv2.imwrite('img_conected.jpg', cv2.hconcat([img1, img2]))
    cv2.imwrite('img_detected.jpg', img3)