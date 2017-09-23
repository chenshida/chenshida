import cv2
import numpy as np
def trans(item):
    data = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    for i in range(0,len(data)):
        if item == data[i]:
            return i

def strDataToInt(strdata):
    intdst = trans(strdata[0])*16 + trans(strdata[1])
    return intdst

if __name__ == "__main__":
    data = np.loadtxt('img.txt', dtype=str)
    data_dst = []
    for i in range(0,4800):
        data_dst.append(strDataToInt(data[i]))
    print data_dst
    data_img = np.array(data_dst, dtype=np.uint8).reshape(60,80)
    cv2.namedWindow('data_img', cv2.WINDOW_NORMAL)
    cv2.imshow('data_img', data_img)
    cv2.waitKey(0)
