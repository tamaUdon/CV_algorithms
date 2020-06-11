#
#参考: http://optie.hatenablog.com/entry/2018/03/21/185647
#

import numpy as np
import cv2

def motionFilter(im, k_size=3):
    H, W, C = im.shape
    
    # Kernel
    K = np.diag([1] * k_size).astype(np.float)
    K /= k_size
    
    # zero padding
    pad = k_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = im.copy().astype(np.float)
    tmp = out.copy()
    
     # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + k_size, x: x + k_size, c])

    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
    
    return out

### メイン関数 ###

def main():
    filename = 'lena.png'
    im = cv2.imread(filename).astype(np.float)

    mf = motionFilter(im)
    cv2.imshow("out", mf)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
