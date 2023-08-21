# %%
# 12.1 绘制一幅图像内所有的轮廓。
import cv2

o = cv2.imread("lesson_12\\contours.png")
cv2.imshow("original", o)
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
o = cv2.drawContours(o, contours, -1, (0, 0, 255), 5)
cv2.imshow("result", o)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.2 逐个显示一幅图像内的边缘信息
import cv2
import numpy as np

o = cv2.imread("lesson_12\\contours.png")
cv2.imshow("original", o)
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
n = len(contours)
contoursImg = []
for i in range(n):
    temp = np.zeros(o.shape, np.uint8)
    contoursImg.append(temp)
    contoursImg[i] = cv2.drawContours(contoursImg[i], contours, i, (255, 255, 255), 5)
    cv2.imshow(f"contours[{i}]", contoursImg[i])
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.3 使用轮廓绘制功能提取前景对象
import cv2
import numpy as np

o = cv2.imread("lesson_12\\loc3.png")
cv2.imshow("original", o)
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros(o.shape, np.uint8)
mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)  # 知识在原图上绘制轮廓信息，并不会显示图像
cv2.imshow("mask", mask)
loc = cv2.bitwise_and(o, mask)
cv2.imshow("location", loc)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.4 使用cv2.moments()提取一幅图像的特征
import cv2
import numpy as np

o = cv2.imread("lesson_12\\moments.png")
cv2.imshow("original", o)
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
n = len(contours)
contoursImg = []
for i in range(n):
    temp = np.zeros(o.shape, np.uint8)
    contoursImg.append(temp)
    contoursImg[i] = cv2.drawContours(contoursImg[i], contours, i, 255, 3)
    cv2.imshow(f"contours[{i}]", contoursImg[i])
print("观察各个轮廓的矩(moments):")
for i in range(n):
    print(f"轮廓{i}的矩形:\n", cv2.moments(contours[i]))
print("观察各个轮廓的面积：")
for i in range(n):
    print(f"轮廓{i}的面积:{cv2.moments(contours[i])['m00']}")
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.5 使用函数cv2.contourArea()计算各个轮廓的面积
import cv2
import numpy as np

o = cv2.imread("lesson_12\\contours.png")
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow("original", o)
n = len(contours)
contoursImg = []
for i in range(n):
    print(f"contours[{i}]面积=", cv2.contourArea(contours[i]))
    temp = np.zeros(o.shape, np.uint8)
    contoursImg.append(temp)
    contoursImg[i] = cv2.drawContours(contoursImg[i], contours, i, (255, 255, 255), 3)
    cv2.imshow(f"contours[{i}]", contoursImg[i])
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.6 在12.5的基础上将面积大于1500的筛选出来
import cv2
import numpy as np

o = cv2.imread("lesson_12\\contours.png")
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow("original", o)
n = len(contours)
contoursImg = []
for i in range(n):
    temp = np.zeros(o.shape, np.uint8)
    contoursImg.append(temp)
    contoursImg[i] = cv2.drawContours(contoursImg[i], contours, i, (255, 255, 255), 3)
    if cv2.contourArea(contours[i]) > 1500:
        cv2.imshow(f"contours[{i}]", contoursImg[i])
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.7 将一幅图像给内长度大于平均值的的轮廓显示出来
import cv2
import numpy as np

# --------------读取及显示原始图像--------------------
o = cv2.imread("lesson_12\\contours0.png")
cv2.imshow("original", o)
# --------------获取轮廓--------------------
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# --------------计算各轮廓的长度之和、平均长度--------------------
n = len(contours)
cnt_len = []
for i in range(n):
    cnt_len.append(cv2.arcLength(contours[i], True))
    print(f"第{i}个轮廓的长度:{cnt_len}")
cnt_len_sum = np.sum(cnt_len)
cnt_len_avg = cnt_len_sum / n
print(f"轮廓的总长度为{cnt_len_sum}")
print(f"轮廓的平均长度为{cnt_len_avg}")
# --------------显示长度超过平均值的轮廓--------------------
contoursImg = []
for i in range(n):
    temp = np.zeros(o.shape, np.uint8)
    contoursImg.append(temp)
    contoursImg[i] = cv2.drawContours(contoursImg[i],
                                      contours, i, (255, 255, 255), 3)
    if cv2.arcLength(contours[i], True) > cnt_len_avg:
        cv2.imshow("contours[" + str(i) + "]", contoursImg[i])
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.8 计算图像的Hu矩，对其中第0个矩关系进行演示
import cv2

o1 = cv2.imread("lesson_12\\contours.png")
gray = cv2.cvtColor(o1, cv2.COLOR_BGR2GRAY)
hu_m1 = cv2.HuMoments(cv2.moments(gray)).flatten()
print("cv2.moments(gray)=\n", cv2.moments(gray))
print("\nhu_m1=\n", hu_m1)
print(f"\ncv2.moments(gray)['nu20']+cv2.moments(gray)['nu02']="
      f"{cv2.moments(gray)['nu20']}+{cv2.moments(gray)['nu02']}="
      f"{cv2.moments(gray)['nu20'] + cv2.moments(gray)['nu02']}")
print("hu_m1[0]=", hu_m1[0])
print("\nHu[0]-(nu02+nu20)=", hu_m1[0] - (cv2.moments(gray)['nu20'] + cv2.moments(gray)['nu02']))

# %%
# 12.9 计算三幅不同图像的Hu矩
import cv2

# ----------------计算图像 o1 的 Hu 矩-------------------
o1 = cv2.imread("lesson_12\\cs1.png")
gray1 = cv2.cvtColor(o1, cv2.COLOR_BGR2GRAY)
hu_m1 = cv2.HuMoments(cv2.moments(gray1)).flatten()
# ----------------计算图像 o2 的 Hu 矩-------------------
o2 = cv2.imread("lesson_12\\cs3.png")
gray2 = cv2.cvtColor(o2, cv2.COLOR_BGR2GRAY)
hu_m2 = cv2.HuMoments(cv2.moments(gray2)).flatten()
# ----------------计算图像 o3 的 Hu 矩-------------------
o3 = cv2.imread("lesson_12\\danger.png")
gray3 = cv2.cvtColor(o3, cv2.COLOR_BGR2GRAY)
hu_m3 = cv2.HuMoments(cv2.moments(gray3)).flatten()
# ---------打印图像 o1、图像 o2、图像 o3 的特征值------------
print("o1.shape=", o1.shape)
print("o2.shape=", o2.shape)
print("o3.shape=", o3.shape)
print("cv2.moments(gray1)=\n", cv2.moments(gray1))
print("cv2.moments(gray2)=\n", cv2.moments(gray2))
print("cv2.moments(gray3)=\n", cv2.moments(gray3))
print("\nHuM1=\n", hu_m1)
print("\nHuM2=\n", hu_m2)
print("\nHuM3=\n", hu_m3)
# ---------计算图像 o1 与图像 o2、图像 o3 的 Hu 矩之差----------------
print("\nhu_m1-hu_m2=", hu_m1 - hu_m2)
print("\nhu_m1-hu_m3=", hu_m1 - hu_m3)
# ---------显示图像----------------
cv2.imshow("original1", o1)
cv2.imshow("original2", o2)
cv2.imshow("original3", o3)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.10 使用函数cv2.matchShapes()计算三幅不同图像的匹配度
import cv2

o1 = cv2.imread("lesson_12\\cs1.png")
o2 = cv2.imread("lesson_12\\cs2.png")
o3 = cv2.imread("lesson_12\\cc.png")
gray1 = cv2.cvtColor(o1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(o2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(o3, cv2.COLOR_BGR2GRAY)
ret, binary1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
ret, binary2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)
ret, binary3 = cv2.threshold(gray3, 127, 255, cv2.THRESH_BINARY)
contours1, hierarchy = cv2.findContours(binary1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours2, hierarchy = cv2.findContours(binary2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours3, hierarchy = cv2.findContours(binary3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnt1 = contours1[0]
cnt2 = contours2[0]
cnt3 = contours3[0]
ret0 = cv2.matchShapes(cnt1, cnt1, 1, 0.0)
ret1 = cv2.matchShapes(cnt1, cnt2, 1, 0.0)
ret2 = cv2.matchShapes(cnt1, cnt3, 1, 0.0)
print("相同图像的matchShape=", ret0)
print("相似图像的matchShape=", ret1)
print("不相似图像的matchShape=", ret2)

# %%
# 12.11 显示函数cv2.boundingRect()不同形式的返回值
import cv2

# ---------------读取并显示原始图像------------------
o = cv2.imread("lesson_12\\cc.png")
# ---------------提取图像轮廓------------------
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# ---------------返回顶点及边长------------------
x, y, w, h = cv2.boundingRect(contours[0])
print("顶点及长宽的点形式:")
print("x=", x)
print("y=", y)
print("w=", w)
print("h=", h)
# ---------------仅有一个返回值的情况------------------
rect = cv2.boundingRect(contours[0])
print("\n顶点及长宽的元组(tuple)形式:")
print("rect=", rect)

# %%
# 12.12 使用函数cv2.drawContours()绘制矩阵的包围框
import cv2
import numpy as np

# ---------------读取并显示原始图像------------------
o = cv2.imread("lesson_12\\cc.png")
cv2.imshow("original", o)
# ---------------提取图像轮廓------------------
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# ---------------构造矩形边界------------------
x, y, w, h = cv2.boundingRect(contours[0])
br_cnt = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])  # 为什么是三重括号?和np的设计有关吧
cv2.drawContours(o, [br_cnt], -1, (255, 255, 255), 2)
# ---------------显示矩形边界------------------
cv2.imshow("result", o)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.13 使用函数cv2.boundingRect()及cv2.rectangle()绘制矩形包围框
import cv2

# ---------------读取并显示原始图像------------------
o = cv2.imread("lesson_12\\cc.png")
cv2.imshow("original", o)
# ---------------提取图像轮廓------------------
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# ---------------构造矩形边界------------------
x, y, w, h = cv2.boundingRect(contours[0])
cv2.rectangle(o, (x, y), (x + w, y + h), (255, 255, 255), 2)  # 直接修改内存数据
# ---------------显示矩形边界------------------
cv2.imshow("result", o)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.14 使用函数cv2.minAreaRect()计算图像的最小包围矩形框
import cv2
import numpy as np

o = cv2.imread("lesson_12\\cc.png")
cv2.imshow("original", o)
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
rect = cv2.minAreaRect(contours[0])
print("返回值rect:\n", rect)
points = cv2.boxPoints(rect)
print("\n转换后的points:\n", points)
points = np.int0(points)
cv2.drawContours(o, [points], 0, (255, 255, 255), 2)  # 该函数接收各种顶点作为参数
cv2.imshow("result", o)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.15 使用函数cv2.minEnclosingCircle()构造图像的最小包围原型
import cv2

o = cv2.imread("lesson_12\\cc.png")
cv2.imshow("original", o)
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
(x, y), radius = cv2.minEnclosingCircle(contours[0])
center = (int(x), int(y))
radius = int(radius)
cv2.circle(o, center, radius, (255, 255, 255), 2)
cv2.imshow("result", o)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.16 使用函数cv2.fitEllipse()构造最优你和椭圆
import cv2

o = cv2.imread("lesson_12\\cc.png")
cv2.imshow("original", o)
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
ellipse = cv2.fitEllipse(contours[0])
print("ellipse=", ellipse)
cv2.ellipse(o, ellipse, (0, 255, 0), 3)
cv2.imshow("result", o)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.17 使用函数cv2.fitLine()构造最有你和直线
import cv2

o = cv2.imread("lesson_12\\cc.png")
cv2.imshow("original", o)
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
rows, cols = binary.shape[:2]
[vx, vy, x, y] = cv2.fitLine(contours[0], cv2.DIST_L2, 0, 0.01, 0.01)
lefty = int((-x * vy / vx) + y)
righty = int(((cols - x) * vy / vx) + y)
cv2.line(o, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
cv2.imshow("result", o)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.18 使用函数cv2.minEnclosingTriangle构造最小外包三角形
import cv2

o = cv2.imread("lesson_12\\cc.png")
cv2.imshow("original", o)
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
area, trgl = cv2.minEnclosingTriangle(contours[0])
print("area=", area)
print("trgl=", trgl)
# 问题代码？
# for i in range(0, 3):
#     cv2.line(o, tuple(trgl[i][0]), tuple(trgl[(i + 1) % 3][0]), (255, 255, 255), 2)
# cv2.imshow("result", o)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.19 使用函数cv2.approxPolyDP()构造不同精度的逼近多边形
import cv2

# ----------------读取并显示原始图像-------------------------------
o = cv2.imread("lesson_12\\cc.png")
cv2.imshow("original", o)
# ----------------获取轮廓-------------------------------
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# ----------------epsilon=0.1*周长-------------------------------
adp = o.copy()
epsilon = 0.1 * cv2.arcLength(contours[0], True)
approx = cv2.approxPolyDP(contours[0], epsilon, True)
adp = cv2.drawContours(adp, [approx], 0, (0, 0, 255), 2)
cv2.imshow("result0.1", adp)
# ----------------epsilon=0.09*周长-------------------------------
adp = o.copy()
epsilon = 0.09 * cv2.arcLength(contours[0], True)
approx = cv2.approxPolyDP(contours[0], epsilon, True)
adp = cv2.drawContours(adp, [approx], 0, (0, 0, 255), 2)
cv2.imshow("result0.09", adp)
# ----------------epsilon=0.055*周长-------------------------------
adp = o.copy()
epsilon = 0.055 * cv2.arcLength(contours[0], True)
approx = cv2.approxPolyDP(contours[0], epsilon, True)
adp = cv2.drawContours(adp, [approx], 0, (0, 0, 255), 2)
cv2.imshow("result0.055", adp)
# ----------------epsilon=0.05*周长-------------------------------
adp = o.copy()
epsilon = 0.05 * cv2.arcLength(contours[0], True)
approx = cv2.approxPolyDP(contours[0], epsilon, True)
adp = cv2.drawContours(adp, [approx], 0, (0, 0, 255), 2)
cv2.imshow("result0.05", adp)
# ----------------epsilon=0.02*周长-------------------------------
adp = o.copy()
epsilon = 0.02 * cv2.arcLength(contours[0], True)
approx = cv2.approxPolyDP(contours[0], epsilon, True)
adp = cv2.drawContours(adp, [approx], 0, (0, 0, 255), 2)
cv2.imshow("result0.02", adp)
# ----------------等待释放窗口-------------------------------
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.20 观察函数cv2.convexHull()内参数returnPoints的使用情况
import cv2

o = cv2.imread("lesson_12\\contours.png")
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
hull = cv2.convexHull(contours[0])
print("returnPoints为默认值True的返回值hull的值:\n", hull)
hull2 = cv2.convexHull(contours[0], returnPoints=False)
print("returnPoints为False时返回值hull的值:\n", hull2)

# %%
# 12.21 使用函数cv2.convexHull()获取轮廓的凸包
import cv2

o = cv2.imread("lesson_12\\hand.png")
cv2.imshow("original", o)
width, height = o.shape[:2]
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
hull = cv2.convexHull(contours[0])
cv2.polylines(o, [hull], True, (0, 255, 0), 2)
cv2.imshow("result", o)
cv2.destroyAllWindows()

# %%
# 12.22 使用函数cv2.convexityDefects()计算缺陷
import cv2

img = cv2.imread("lesson_12\\hand.png")
cv2.imshow("original", img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, 0)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
hull = cv2.convexHull(cnt, returnPoints=False)
defects = cv2.convexityDefects(cnt, hull)
print("defects=\n", defects)
# --------------------构造凸缺陷-------------------
for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(img, start, end, [0, 0, 255], 2)
    cv2.circle(img, far, 5, [255, 0, 0], -1)
cv2.imshow("result", img)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.23 使用函数cv2.isContourConvex()来判断轮廓是否是凸形的
import cv2

o = cv2.imread("lesson_12\\hand.png")
cv2.imshow("original", o)
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# --------------凸包----------------------
image1 = o.copy()
hull = cv2.convexHull(contours[0])
cv2.polylines(image1, [hull], True, (0, 255, 0), 2)
print("使用函数cv2.convexHull()构造的多边形是否是凸型的：", cv2.isContourConvex(hull))
cv2.imshow("result1", image1)
# ------------逼近多边形--------------------
image2 = o.copy()
epsilon = 0.01 * cv2.arcLength(contours[0], True)
approx = cv2.approxPolyDP(contours[0], epsilon, True)
image2 = cv2.drawContours(image2, [approx], 0, (0, 0, 255), 2)
print("使用函数cv2.approxPolyDP()构造的多边形是否是凸型的：", cv2.isContourConvex(approx))
cv2.imshow("result2", image2)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.24 使用函数cv2.pointPolygonTest()计算点到轮廓的最短距离
import cv2

# ----------------原始图像-------------------------
o = cv2.imread("lesson_12\\cs1.png")
cv2.imshow("original", o)
# ----------------获取凸包------------------------
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
hull = cv2.convexHull(contours[0])
image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
cv2.polylines(image, [hull], True, (0, 255, 0), 2)
# ----------------内部点 A 到轮廓的距离-------------------------
dist_A = cv2.pointPolygonTest(hull, (300, 150), True)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, 'A', (300, 150), font, 1, (0, 255, 0), 3)
print("dist_A=", dist_A)
# ----------------外部点 B 到轮廓的距离-------------------------
dist_B = cv2.pointPolygonTest(hull, (300, 250), True)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, 'B', (300, 250), font, 1, (0, 255, 0), 3)
print("dist_B=", dist_B)
# ----------------外部点 B 到轮廓的距离-------------------------
dist_C = cv2.pointPolygonTest(hull, (423, 112), True)
font = cv2.FONT_HERSHEY_TRIPLEX
cv2.putText(image, 'C', (423, 112), font, 1, (0, 255, 0), 3)
print("dist_C=", dist_C)
cv2.imshow("result", image)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.25 使用函数判断点与轮廓的关系代码同上

# %%
# 12.26 使用函数cv2.createShapeContextDistanceExtractor()计算形状场景距离
import cv2

# -----------原始图像 o1 的边缘--------------------
o1 = cv2.imread("lesson_12\\cs1.png")
cv2.imshow("original1", o1)
gray1 = cv2.cvtColor(o1, cv2.COLOR_BGR2GRAY)
ret, binary1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
contours1, hierarchy = cv2.findContours(binary1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contour1 = contours1[0]
# -----------原始图像 o2 的边缘--------------------
o2 = cv2.imread('lesson_12\\cs3.png')
cv2.imshow("original2", o2)
gray2 = cv2.cvtColor(o2, cv2.COLOR_BGR2GRAY)
ret, binary2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)
contours2, hierarchy = cv2.findContours(binary2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contour2 = contours2[0]
# -----------原始图像 o3 的边缘--------------------
o3 = cv2.imread('lesson_12\\hand.png')
cv2.imshow("original3", o3)
gray3 = cv2.cvtColor(o3, cv2.COLOR_BGR2GRAY)
ret, binary3 = cv2.threshold(gray3, 127, 255, cv2.THRESH_BINARY)
contours3, hierarchy = cv2.findContours(binary3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contour3 = contours3[0]
# -----------构造距离提取算子--------------------
sd = cv2.createShapeContextDistanceExtractor()
# -----------计算距离--------------------
d1 = sd.computeDistance(contour1, contour1)
print("与自身距离d1=", d1)
d2 = sd.computeDistance(contour1, contour2)
print("与cnt2距离d2=", d2)
d3 = sd.computeDistance(contour1, contour3)
print("与cnt3距离d3=")
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.27 使用cv2.createHausdorffDistanceExtractor()计算不同图像的Hausdorff距离
import cv2

# -----------读取原始图像--------------------
o1 = cv2.imread("lesson_12\\cs1.png")
o2 = cv2.imread('lesson_12\\cs3.png')
o3 = cv2.imread('lesson_12\\hand.png')
cv2.imshow("original1", o1)
cv2.imshow("original2", o2)
cv2.imshow("original", o3)
# -----------色彩转换--------------------
gray1 = cv2.cvtColor(o1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(o2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(o3, cv2.COLOR_BGR2GRAY)
# -----------阈值处理--------------------
ret, binary1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
ret, binary2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)
ret, binary3 = cv2.threshold(gray3, 127, 255, cv2.THRESH_BINARY)
# -----------提取轮廓--------------------
contours1, hierarchy = cv2.findContours(binary1,
                                        cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)
contours2, hierarchy = cv2.findContours(binary2,
                                        cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)
contours3, hierarchy = cv2.findContours(binary3,
                                        cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)
cnt1 = contours1[0]
cnt2 = contours2[0]
cnt3 = contours3[0]
# -----------构造距离提取算子--------------------
hd = cv2.createHausdorffDistanceExtractor()
# -----------计算距离--------------------
d1 = hd.computeDistance(cnt1, cnt1)
print("与自身图像的 Hausdorff 距离 d1=", d1)
d2 = hd.computeDistance(cnt1, cnt2)
print("与旋转缩放后的自身图像的 Hausdorff 距离 d2=", d2)
d3 = hd.computeDistance(cnt1, cnt3)
print("与不相似对象的 Hausdorff 距离 d3=", d3)
# -----------显示距离--------------------
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.28编写计算矩形轮廓宽高比
import cv2

o = cv2.imread("lesson_12\\cc.png")
cv2.imshow("original", o)
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv2.boundingRect(contours[0])
cv2.rectangle(o, (x, y), (x + w, y + h), (255, 255, 255), 3)
aspectRatio = float(w) / h
print(aspectRatio)
cv2.imshow("result", o)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.29 计算图像轮廓面积与其矩形边界的面积比
o = cv2.imread("lesson_12\\cc.png")
cv2.imshow("original", o)
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv2.boundingRect(contours[0])
cv2.drawContours(o, contours[0], -1, (0, 0, 255), 3)
cv2.rectangle(o, (x, y), (x + w, y + h), (255, 255, 255), 3)
rect_area = w * h
cnt_area = cv2.contourArea(contours[0])
extend = float(cnt_area) / rect_area
cv2.imshow("result", o)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.30 编写程序计算图像轮廓面积与凸包的面积之比
import cv2

o = cv2.imread("lesson_12\\hand.png")
cv2.imshow("original", o)
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(o, contours[0], -1, (0, 0, 255), 3)
cnt_area = cv2.contourArea(contours[0])
hull = cv2.convexHull(contours[0])
hull_area = cv2.contourArea(hull)
cv2.polylines(o, [hull], True, (0, 255, 0), 2)
solidity = float(cnt_area) / hull_area
print(solidity)
cv2.imshow("result", o)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.31 计算与轮廓面积相等的圆形的直径，并绘制与该轮廓等面积的员
import cv2
import numpy as np

o = cv2.imread("lesson_12\\cc.png")
cv2.imshow("original", o)
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(o, contours[0], -1, (0, 0, 255), 3)
cnt_area = cv2.contourArea(contours[0])
equiDiameter = np.sqrt(4 * cnt_area / np.pi)
print(equiDiameter)
cv2.circle(o, (100, 100), int(equiDiameter / 2), (0, 0, 255), 3)
cv2.imshow("result", o)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.32 编写程序观察函数cv2.fitEllipse()的不同返回值，就是使用函数没有必要思考

# %%
# 12.33 使用Numpy函数获取一个数组内的非零元素的位置信息
import numpy as np

# ------------生成一个元素都是零值的数组 a-------------------
a = np.zeros((5, 5), dtype=np.uint8)
# -------随机将其中 10 个位置上的数值设置为 1------------
# ---times 控制次数
# ---i,j 是随机生成的行、列位置
# ---a[i,j]=1,将随机挑选出来的位置上的值设置为 1
for times in range(10):
    i = np.random.randint(0, 5)
    j = np.random.randint(0, 5)
    a[i, j] = 1
# -------打印数组 a，观察数组 a 内值的情况-----------
print("a=\n", a)
# ------查找数组 a 内非零值的位置信息------------
loc = np.transpose(np.nonzero(a))
print("a内非零值的位置：\n", loc)

# %%
# 12.34 使用Numpy函数获取一个图像内的轮廓点的位置
import cv2
import numpy as np

o = cv2.imread("lesson_12\\cc.png")
cv2.imshow("original", o)
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contour = contours[0]
# -----------------绘制空心轮廓------------------------
mask1 = np.zeros(gray.shape, np.uint8)
cv2.drawContours(mask1, [contour], 0, 255, 2)
pixel_points1 = np.transpose(np.nonzero(mask1))
print("pixel_points1.shape=", pixel_points1.shape)
print("pixel_points1=\n", pixel_points1)
cv2.imshow("mask1", mask1)
# -----------------绘制实心轮廓---------------------
mask2 = np.zeros(gray.shape, np.uint8)
cv2.drawContours(mask2, [contour], 0, 255, -1)
pixel_points2 = np.transpose(np.nonzero(mask2))
print("pixel_points1.shape=", pixel_points2.shape)
print("pixel_points2=\n", pixel_points2)
cv2.imshow("mask2", mask2)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.35 使用Opencv函数cv2.findNonZero()获取一个数组内的非零值
import cv2
import numpy as np

# ------------生成一个元素都是零值的数组 a-------------------
a = np.zeros((5, 5), dtype=np.uint8)
# -------随机将其中 10 个位置上的值设置为 1------------
# ---times 控制次数
# ---i,j 是随机生成的行、列位置
# ---a[i,j]=1,将随机挑选出来的位置上的值设置为 1
for times in range(10):
    i = np.random.randint(0, 5)
    j = np.random.randint(0, 5)
    a[i, j] = 1
# -------打印数组 a，观察数组 a 内值的情况-----------
print("a=\n", a)
# ------查找数组 a 内非零值的位置信息------------
loc = cv2.findNonZero(a)
print("a 内非零值的位置:\n", loc)

# %%
# 12.36 使用OpenCV函数cv2.findNonZero()获取一个图像内的轮廓点的位置
import cv2
import numpy as np

o = cv2.imread("lesson_12\\cc.png")
cv2.imshow("original", o)
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contour = contours[0]
# -----------------绘制空心轮廓------------------------
mask1 = np.zeros(gray.shape, np.uint8)
cv2.drawContours(mask1, [contour], 0, 255, 2)
pixel_points1 = cv2.findNonZero(mask1)
print("pixel_points1.shape=", pixel_points1.shape)
print("pixel_points1=\n", pixel_points1)
cv2.imshow("mask1", mask1)
# -----------------绘制实心轮廓---------------------
mask2 = np.zeros(gray.shape, np.uint8)
cv2.drawContours(mask2, [contour], 0, 255, -1)
pixel_points2 = cv2.findNonZero(mask2)
print("pixel_points1.shape=", pixel_points2.shape)
print("pixel_points2=\n", pixel_points2)
cv2.imshow("mask2", mask2)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.37 使用cv2.minMaxLoc()在图像内查找掩模指定区域内的最大值、最小值及其位置
import cv2
import numpy as np

o = cv2.imread("lesson_12\\ct.png")
cv2.imshow("original", o)
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contour = contours[2]
# --------使用掩模获取感兴趣区域的最值-----------------
# 需要注意函数 minMaxLoc 处理的对象为灰度图像，本例中处理的对象为灰度图像 gray
# 如果希望获取彩色图像的最值，需要提取各个通道图像，为每个通道独立计算最值
mask = np.zeros(gray.shape, np.uint8)
mask = cv2.drawContours(mask, [contour], -1, 255, -1)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(gray, mask=mask)
print("minVal=", minVal)
print("maxVal=", maxVal)
print("minLoc=", minLoc)
print("maxLoc=", maxLoc)
# --------使用掩模获取感兴趣区域并显示-----------------
masko = np.zeros(o.shape, np.uint8)
masko = cv2.drawContours(masko, [contour], -1, (255, 255, 255), -1)
loc = cv2.bitwise_and(o, masko)
cv2.imshow("mask", loc)
# 显示灰度结果
loc2 = cv2.bitwise_and(gray, mask)
cv2.imshow("mask2", loc2)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.38 使用函数cv2.mean()计算一个对象的平均灰度值
import cv2
import numpy as np

o = cv2.imread("lesson_12\\ct.png")
cv2.imshow("original", o)
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contour = contours[2]
# --------使用掩模获取感兴趣区域的均值-----------------
mask = np.zeros(gray.shape, np.uint8)  # 构造 mean 所使用的掩模（必须是单通道的）
cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)
meanVal = cv2.mean(o, mask=mask)
print("meanVal=\n", meanVal)
# --------使用掩模获取感兴趣区域并显示-----------------
masko = np.zeros(o.shape, np.uint8)
masko = cv2.drawContours(masko, [contour], -1, (255, 255, 255), -1)
loc = cv2.bitwise_and(o, masko)
cv2.imshow("mask", loc)
# 显示灰度结果
loc2 = cv2.bitwise_and(gray, mask)
cv2.imshow("mask2", loc2)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 12.39 计算一幅图像内的极值点
import cv2
import numpy as np

o = cv2.imread("lesson_12\\cs1.png")
# --------获取并绘制轮廓-----------------
gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros(gray.shape, np.uint8)
cnt = contours[0]
cv2.drawContours(mask, [cnt], 0, 255, -1)
# --------计算极值-----------------
left_most = tuple(cnt[cnt[:, :, 0].argmin()][0])
right_most = tuple(cnt[cnt[:, :, 0].argmax()][0])
top_most = tuple(cnt[cnt[:, :, 1].argmin()][0])
bottom_most = tuple(cnt[cnt[:, :, 1].argmax()][0])
# --------打印极值-----------------
print("left_most=", left_most)
print("right_most=", right_most)
print("top_most=", top_most)
print("bottom_most=", bottom_most)
# --------绘制说明文字-----------------
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(o, 'A', left_most, font, 1, (0, 0, 255), 2)
cv2.putText(o, 'B', right_most, font, 1, (0, 0, 255), 2)
cv2.putText(o, 'C', top_most, font, 1, (0, 0, 255), 2)
cv2.putText(o, 'D', bottom_most, font, 1, (0, 0, 255), 2)
cv2.imshow("result", o)
cv2.waitKey()
cv2.destroyAllWindows()
