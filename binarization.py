#
#参考:
#https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Question_01_10/answers_py/answer_3.py
#https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Question_01_10/answers_py/answer_4.py
#

import numpy as np
import cv2
import grayscale as gs


def binarization(im, th=128):
    im[im < th] = 0
    im[im > th] = 255
    return im


# クラス間分散の値が最大になる値がしきい値になる
def binarization_Otsu(im):
    max_sigma = 0
    max_t = 0
    H, W = im.shape
    
    for t in range(1,256): 
        v0 = im[np.where(im < t)]  # しきい値t以下の画素の配列 (クラス0)
        w0 = len(v0)/ (H*W)  # クラス0の画素数
        m0 = np.mean(v0) if len(v0) > 0 else 0. # クラス0に属する画素値の平均
        v1 = im[np.where(im >= t)]
        w1 = len(v1) / (H*W)
        m1 = np.mean(v1) if len(v1) > 0 else 0.
        sigma = w0 * w1 / ((m0 - m1)**2)
        if sigma > max_sigma:
            max_sigma = sigma
            max_t = t
    
    th = max_t
    im[im < th]  = 0
    im[im >= th] = 255
    
    return im

def main():
    filename = 'lena.png'
    im = cv2.imread(filename).astype(np.float)

    gray = gs.toGray_BT601(im)
    bnr = binarization(gray)
    otsu = binarization_Otsu(gray)

    out = np.concatenate((bnr, otsu), axis=1)
    cv2.imshow("out", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
