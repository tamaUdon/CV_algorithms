import numpy as np
import cv2

def medianFilter(im, k_size=3):
    H, W, C = im.shape
    
    # zero padding
    pad = k_size // 2
    out = np.zeros((H + pad*2, W + pad*2,C), dtype=np.float)
    out[pad:pad+H, pad:pad+W] = im.copy().astype(np.float)
    
    tmp = out.copy()
    
    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad+y,pad+x,c] = np.median[tmp[y:y+k_size, x: x+k_size, c]]
    
    out = out[pad:pad+H, pad:pad+W].astype(np.uint8)
    
    return out 


def main():
    filename = 'lena.png'
    im = cv2.imread(filename).astype(np.float)
    mf = medianFilter(im)
    print(ap)
    cv2.imshow("out", mf)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()