import numpy as np
import cv2
import yaml
def getSrcPointIndex(event, x, y, flag, param):
    global src
    if event == cv2.EVENT_LBUTTONDOWN:
        print 'current point index is: ', [x, y]
        print 'press s add current point into src point.\n'
        if cv2.waitKey(0) & 0xff ==ord('s'):
            if len(src) < 4:
                src.append([x, y])
                print 'src point add success, current point is:\n', len(src)
            else:
                print 'src point is enough,press q exit.\n'
        else:
            print 'repick point'
def getDstPointIndex(event, x, y, flag, param):
    global dst
    if event == cv2.EVENT_LBUTTONDOWN:
        print 'current point index is: ', [x, y]
        print 'press s add current point into dst point.\n'
        if cv2.waitKey(0) & 0xff ==ord('s'):
            if len(dst) < 4:
                dst.append([x, y])
                print 'dst point add success, current point is:\n', len(dst)
            else:
                print 'dst point is enough,press q exit.\n'
        else:
            print 'repick point'

if __name__ == "__main__":
    src = []
    dst = []
    image = cv2.imread('test.png')
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', getSrcPointIndex)
    while True:
        cv2.imshow('image', image)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break

    cv2.destroyAllWindows()
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', getDstPointIndex)
    while True:
        cv2.imshow('image', image)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break
    if len(src) != 4:
        File = open('PointConfig.yaml', 'r')
        Data = yaml.load(File)
        src = Data['point_src']
        print src
        File.close()
    if len(dst) != 4:
        File = open('PointConfig.yaml', 'r')
        Data = yaml.load(File)
        dst = Data['point_dst']
        File.close()
    src = np.array(src, dtype=np.float32).reshape(4, 2)
    dst = np.array(dst, dtype=np.float32).reshape(4, 2)
    print "src points:\n", src
    print "dst points:\n", dst
    M = cv2.getPerspectiveTransform(src, dst)
    print 'Transform Matrix:\n', M
    print 'Transform Matrix Inverse:\n', np.linalg.inv(M)

