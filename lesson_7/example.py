# %%
# 7.1 读取噪声图像，使用函数cv2.blur()对图像进行均值滤波处理
import cv2

o = cv2.imread(".\\lesson_7\\img.png")
r = cv2.blur(o, (5, 5))
cv2.imshow("original", o)
cv2.imshow("result", r)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 7.2 使用不同大小的卷积和对其进行均值滤波
import cv2

o = cv2.imread(".\\lesson_7\\img.png")
r = cv2.blur(o, (5, 5))
r30 = cv2.blur(o,(30, 30))
cv2.imshow("original", o)
cv2.imshow("result", r)
cv2.imshow("result30", r30)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 7.3/7.4 针对噪声图像采用方框滤波
import cv2

o = cv2.imread(".\\lesson_7\\img.png")
r = cv2.boxFilter(o, -1, (5, 5), normalize=0)
cv2.imshow("original", o)
cv2.imshow("result", r)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 7.5 针对噪声图像，使用方框滤波去噪，将参数normalize值设为0， 卷积核大小设为2x2
import cv2

o = cv2.imread(".\\lesson_7\\img.png")
r = cv2.boxFilter(o, -1, (2, 2), normalize=0)
cv2.imshow("original", o)
cv2.imshow("result", r)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 7.6 是用高斯滤波来对图像进行处理
import cv2

o = cv2.imread(".\\lesson_7\\img.png")
r = cv2.GaussianBlur(o, (5, 5), 0, 0)
cv2.imshow("original", o)
cv2.imshow("result", r)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 7.7 针对噪声图像，对其进行中值滤波，显示滤波结果
import cv2

o = cv2.imread(".\\lesson_7\\img.png")
r = cv2.medianBlur(o, 3)
cv2.imshow("original", o)
cv2.imshow("result", r)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 7.8 针对噪声图像，对其进行双边滤波, 还是取了均值
import cv2

o = cv2.imread(".\\lesson_7\\img.png")
r = cv2.bilateralFilter(o, 25, 150, 150)
cv2.imshow("original", o)
cv2.imshow("result", r)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 7.9 对边缘图像分别进行高斯滤波和双边滤波，比较处理边缘的结果
import cv2
import  numpy as np

img = np.zeros((500, 500), dtype=np.uint8)
img[:, 250:500] = 255
g = cv2.GaussianBlur(img, (55, 55), 0, 0)
b = cv2.bilateralFilter(img, 55, 100, 100)
cv2.imshow("img", img)
cv2.imshow("Gaussian", g)
cv2.imshow("bilateral", b)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 7.10 通过cv2.filter2D()应用该卷积核对图像进行滤波操作
import cv2
import numpy as np

o = cv2.imread(".\\lesson_7\\img.png", 0)
kernel = np.ones((9, 9), np.float32) / 81
r = cv2.filter2D(o, -1, kernel)
cv2.imshow("original", o)
cv2.imshow("Gaussian", r)
cv2.waitKey()
cv2.destroyAllWindows()



