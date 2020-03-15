#グレースケール化アルゴリズム
#
#1. RGB値にかかっているガンマ補正を外す（逆ガンマ補正してlinear-RGB化）
#2. R, G, B値に重みを付けて足す（CIE XYZ）
#3. ガンマ補正して表示
#
#参考: 
#    https://qiita.com/yoya/items/96c36b069e74398796f3
#    https://knowledge.shade3d.jp/knowledgebase/%E3%82%AC%E3%83%B3%E3%83%9E%E3%81%A8%E3%81%AF
#    https://github.com/yoyoyo-yo/Gasyori100knock/blob/master/Question_01_10/answers_py/answer_2.py
#


import numpy as np
import cv2

# CIE XYZ
def toGray_CIE(im):
    # 逆ガンマ補正
    degamma_r = degamma_table(im[:,:,0].copy())  
    degamma_g = degamma_table(im[:,:,1].copy())
    degamma_b = degamma_table(im[:,:,2].copy()) 

    # 重みを加算する
    r = degamma_r * 0.2126
    g = degamma_g * 0.7152 
    b = degamma_b * 0.0722

    # ガンマ補正
    gamma_r = gamma_table(r)
    gamma_g = gamma_table(g)
    gamma_b = gamma_table(b)
    res = gamma_r + gamma_g + gamma_b
    
    return res.astype(np.uint8)

# BT.601
def toGray_BT601(im):
    r = im[:,:,0].copy()  
    g = im[:,:,1].copy()
    b = im[:,:,2].copy()
    res = 0.299*r + 0.587*g + 0.114*b
    
    return res.astype(np.uint8)


def gamma_table(color, gamma=2.2, gain=1.0):    
    return np.power(color, gain/gamma)

def degamma_table(color, gamma=2.2, gain=1.0):
    return np.power(color, gain/(gain/gamma))


filename = 'lena.png'
im = cv2.imread(filename).astype(np.float) # 配列に画像を読み込む

cie = toGray_CIE(im)
bt601 = toGray_BT601(im)

out = np.concatenate((cie, bt601), axis=1)
cv2.imshow("out", out)
cv2.waitKey(0)
cv2.destroyAllWindows()







