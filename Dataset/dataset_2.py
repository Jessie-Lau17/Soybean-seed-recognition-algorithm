import cv2
import numpy as np
import os


filepath = 'D:\corn\XX6'
Path = 'D:\lab_something\Agriculture\Corn\data1_1\XX6'
frame = 0

pathDir = os.listdir(filepath)
for allDir in pathDir:
    #print(allDir)
    child = os.path.join(filepath,allDir)
    #print(child)
    img = cv2.imread(child)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #data1_1
    retval, dst=cv2.threshold(gray, 65, 255, cv2.THRESH_TOZERO)
    #data1_2
    #retval, dst=cv2.threshold(gray, 65, 255, cv2.THRESH_TOZERO)
    rows,cols,channels = img.shape
    roi = img[0:rows, 0:cols]
    cv_contours = []
    contours, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 500 and area <= 3000:
            frame = frame +1
            mask = np.zeros((rows,cols),np.uint8)
            cv2.drawContours(mask, [contour], -1, (255,255,255),cv2.FILLED)
            M = cv2.moments(contour)
            x = int(M["m10"]/M["m00"])
            y = int(M["m01"]/M["m00"])
            # cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
            # cv2.imshow("mask", mask)
            # cv2.fillPoly(mask, contour, (255,255,255))
            # cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
            # cv2.imshow("mask", mask)
            img1 = cv2.bitwise_or(img, roi, mask=mask)
            a = x - 64
            b = x + 64
            c = y - 64
            d = y + 64
            if a < 0: 
                a = 0
                b = 128
            if b > cols:
                b = cols
                a = b - 128
            if c < 0:
                c = 0
                d = c + 128
            if d > rows:
                d = rows
                c = d - 128
            Img = img1[c:d,a:b]
            # cv2.imshow("Img", Img)
            # cv2.namedWindow('img1',cv2.WINDOW_NORMAL)
            # cv2.imshow("img1", img1)
            path = os.path.join(Path, str(frame)+'.png')
            cv2.imwrite(path, Img)
cv2.waitKey(0)
