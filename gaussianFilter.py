#
#参考: http://optie.hatenablog.com/entry/2018/03/21/185647
#

import numpy as np
import cv2

# sigma = 標準偏差
# kernel = カーネルのサイズ
def gaussianFilter(im, k_size=3):
    H, W, C = im.shape
    
    if k_size%2 == 0:
        print('kernel size should be odd')
        return
    # フィルタサイズ (2w + 1)(2w + 1) の場合，
    # sigma = w/2 とするのが一つの目安
    sigma = (k_size-1)/2
    
    kernel = getKernel(sigma, k_size)
    out, pad = zeroPadding(im, H, W, C, k_size)
    tmp = out.copy()
    
    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad + y, pad + x, c] = np.sum(kernel * tmp[y: y + k_size, x: x + k_size, c])
                
    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
    
    return out


### 補助関数 ###

def zeroPadding(im, H, W, C, k_size):
    # Zero Padding
    pad = k_size // 2 # 切り捨て除算
    zero = np.zeros((H + pad*2, W + pad*2, C), dtype=np.float)
    zero[pad: pad + H, pad: pad + W] = im.copy().astype(np.float)  
    return zero, pad
    
def getKernel(sigma, k_size):
    x = y = np.arange(0,k_size) - sigma
    X,Y = np.meshgrid(x,y)
    mat = norm2d(X,Y,sigma)
    kernel = mat / np.sum(mat) # 総和を1にする 
    return kernel
    
def norm2d(x, y, sigma):
    Z = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    return Z


### メイン関数 ###

def main():
    filename = 'lena_noisy.png'
    im = cv2.imread(filename).astype(np.float)

    gf = gaussianFilter(im)
    cv2.imshow("out", gf)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
