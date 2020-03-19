import numpy as np
import cv2

def avgPooling(im, box=8): 
    H, W, C = im.shape
    out = im.copy()
    for y in range(1,H+1,box):
        for x in range(1,W+1,box):
            for c in range(C):
                out[y:y+box,x:x+box,c] = np.mean(out[y:y+box,x:x+box,c]).astype(np.int)/255
    return out
    
def maxPooling(im, box=8):
    H, W, C = im.shape
    out = im.copy()
    for y in range(1,H+1,box):
        for x in range(1,W+1,box):
            for c in range(C):
                out[y:y+box,x:x+box,c] = np.max(out[y:y+box,x:x+box,c]).astype(np.int)/255
    return out

def main():
    filename = 'lena.png'
    im = cv2.imread(filename).astype(np.float)

    ap = avgPooling(im)
    mp = maxPooling(im)
    print(ap)
    out = np.concatenate((ap, mp), axis=1)
    cv2.imshow("out", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
