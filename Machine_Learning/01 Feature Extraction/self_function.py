import cv2
import numpy as np
import os


def edge_demo(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)  # 高斯降噪，适度
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # 求梯度
    xgrd = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
    ygrd = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)

    edge_output = cv2.Canny(xgrd, ygrd, 30, 90)  # 50低阈值，150高阈值

    return edge_output


def FillHole(img):
    im_in = img
    # 复制 im_in 图像
    im_floodfill = im_in.copy()

    # Mask 用于 floodFill，官方要求长宽+2
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # floodFill函数中的seedPoint对应像素必须是背景
    isbreak = False
    seedPoint = (0, 0)
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if im_floodfill[i][j] == 0:
                seedPoint = (i, j)
                isbreak = True
                break
        if isbreak:
            break

    # 得到im_floodfill 255填充非孔洞值
    cv2.floodFill(im_floodfill, mask, seedPoint, 255)

    # 得到im_floodfill的逆im_floodfill_inv
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    im_out = im_in | im_floodfill_inv

    return im_out


def Circle(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray_img, 5)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 2, 100, param1=60,
                               param2=30, minRadius=20, maxRadius=35)
    circles = np.uint16(np.around(circles))
    rst = img * 0
    for i in circles[0, :]:
        cv2.circle(rst, (i[0], i[1]), i[2] + 16, (255, 255, 255), 2)
    return rst, circles


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        # print("---  new folder...  ---")
        # print("---  OK  ---")

    # else:
    #     print("---  There is this folder!  ---")


def GetPoint(circle, h, w, cut_h, cut_w):
    y = circle[1] - cut_h/2
    if (circle[1] - cut_h/2) < 0:
        y = 0
    elif (circle[1] - cut_h/2) > (h-1):
        y = h-cut_h

    x = circle[0] - cut_w / 2
    if (circle[0] - cut_w/2) < 0:
        x = 0
    elif (circle[0] - cut_w/2) > (w-1):
        x = w-cut_w
    y = int(y)
    x = int(x)
    return y, x


def Circle_v1(img, deta=4):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray_img, 5)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 2, 100, param1=60,
                               param2=30, minRadius=20, maxRadius=35)
    circles = np.uint16(np.around(circles))
    rst = img * 0
    for i in circles[0, :]:
        cv2.circle(rst, (i[0], i[1]), i[2] + deta, (255, 255, 255), 2)
    return rst, circles
