import numpy as np
import cv2
if __name__ == '__main__':
    img = cv2.imread('test.png')
    # img = cv2.imread('timg.png')
    cv2.imshow('image', img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print img.shape
    img_hight, img_width = img.shape
    # print len(img)
    img_trans = np.zeros((img_hight, img_width), np.uint8)
    # print img[0]
    # point_src = np.float32([[519, 19], [848, 19], [14, 406], [1169, 406]])
    # point_dst = np.float32([[14, 19], [1199, 20], [14, 422], [1199, 423]])
    point_src = np.float32([[30, 51], [156, 0], [0, 213], [135, 210]])
    point_dst = np.float32([[0, 0], [158, 0], [0, 210], [159, 213]])
    M = cv2.getPerspectiveTransform(point_src, point_dst)
    print M
    point = [0, 0]
    points = []
    points_1 = []
    for i in range(0, img_hight):
        for j in range(0, img_width):
            point[0] = j
            point[1] = i
            # print point
            points.extend(point)
            points_1.extend(point)

    points = np.array(points,dtype=int).reshape(-1,2)
    points_1 = np.array(points_1,dtype=float).reshape(-1,2)
    # print points
    #
    for i in range(0, len(points)):
        x = float(points[i][0])
        y = float(points[i][1])
        denominator = float(M[2][0] * x + M[2][1] * y + M[2][2])
        points_1[i][0] = float((M[0][0] * x + M[0][1] * y + M[0][2])) / denominator
        points_1[i][1] = float((M[1][0] * x + M[1][1] * y + M[1][2])) / denominator
    # print points[8139]
    # print points[156]
    # print points_1[8139]
    # print points_1[156]


    for i in range(0, len(points)):
        if points_1[i][0] < 0 or points_1[i][0] > (img_width-1) or points_1[i][1] < 0 or points_1[i][1] > (img_hight-1):
            continue
        img_trans[int(points_1[i][1])][int(points_1[i][0])] = img[points[i][1]][points[i][0]]
        # img_trans[int(points[i][1])][int(points[i][0])] = img[int(points_1[i][1])][int(points_1[i][0])]
    img_perspect = cv2.warpPerspective(img, M, (img_width, img_hight))
    cv2.imshow('image_1', img_perspect)
    cv2.namedWindow('image1', flags=cv2.WINDOW_NORMAL)
    # img_trans = cv2.GaussianBlur(img_trans, (3, 3), 0.2)

    # img_trans = cv2.blur(img_trans, (3, 3))
    # img_trans = cv2.medianBlur(img_trans, 3)
    cv2.imshow('image1', img_trans)
    cv2.waitKey(0)