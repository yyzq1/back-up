import cv2 as cv
import numpy as np
import math


def LeastSquaresCircleFitting(Points):      # 中心点轨迹拟合圆 最小二乘法
    Center = np.zeros(2)
    Radius = 0.
    if (len(Points) < 3):
        raise Exception('There are fewer than 3 elements in Points.')
    x1 = y1 = 0.
    x2 = y2 = 0.
    x3 = y3 = 0.
    x1y1 = x1y2 = x2y1 = 0.
    N = len(Points)
    for i in range(N):
        x = Points[i][0]
        y = Points[i][1]
        x1 += x
        y1 += y
        x2 += x**2
        y2 += y**2
        x3 += x**3
        y3 += y**3
        x1y1 += x * y
        x1y2 += x * y**2
        x2y1 += x**2 * y
    
    C = N * x2 - x1**2
    D = N * x1y1 - x1 * y1
    E = N * x3 + N * x1y2 - (x2 + y2) * x1
    G = N * y2 - y1 * y1
    H = N * x2y1 + N * y3 - (x2 + y2) * y1
    a = (H * D - E * G) / (C * G - D * D)
    b = (H * C - E * D) / (D * D - G * C)
    c = -(a * x1 + b * y1 + x2 + y2) / N

    Center[0] = int(a / (-2))
    Center[1] = int(b / (-2))
    Center = Center.astype(int)
    Radius = int(np.sqrt(a * a + b * b - 4 * c) / 2)
    return Center, Radius


def predict(point, center, w, t):       # 预测点位置 point当前点 center轨迹圆圆心 w角速度 t射出到击中时间
    vector = point - center             # 当前靶点向量（相对圆心）
    R = np.zeros((2,2))
    v = w*t
    R[0][0] = R[1][1] = math.cos(v)
    R[0][1] = (-1)*math.sin(v)
    R[1][0] = math.sin(v)
    
    pre = center + np.dot(R, vector)
    pre = pre.astype(int)
    return pre


# capture = cv.VideoCapture('fc002.mp4')
capture = cv.VideoCapture('test.avi')
points = []
while(True):
    ret, frame  = capture.read()
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)      # BGR->LAB 
    gray, g, b = cv.split(lab)                      # 取l通道作为gray
    _, mask = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)
    kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (9,9))    # 开运算kernel
    kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (15,15))  # 闭运算kernel
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel1)        # 开运算
    h, w = mask.shape[:2]
    mask1 = np.zeros([h+2, w+2], np.uint8)
    cv.floodFill(mask, mask1, (0,0), (0,0,0))   #漫水填充
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel2)   #闭运算
    cv.imshow('closed frame', mask)
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv.contourArea(contour)
        #print(area)
        if area <1500 and area>1300:
            (x, y), radius = cv.minEnclosingCircle(contour)                 # 闭合圆圆心和闭合圆半径
            cv.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)          # 画闭合圆圆心（原靶点）
            cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 3) # 画闭合圆
            points.append((int(x),int(y)))                    # 靶心列表

            if len(points) > 10:                # 画轨迹拟合圆
                center, radius = LeastSquaresCircleFitting(points)          # 轨迹圆圆心和轨迹圆半径
                cv.circle(frame, center, radius, (255, 0, 0), 3)
                cv.circle(frame, center, 3, (0, 0, 255), -1)

                #pre = predict((x,y), center, 1.0, 0.1)      # 预测靶点
                #cv.circle(frame, pre, 3, (255,255,255), -1)   # 画预测靶点

    cv.imshow('result', frame)

    c = cv.waitKey(1)
    if c == 27:
        break
cv.destroyAllWindows
