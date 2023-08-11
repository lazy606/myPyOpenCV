# %%
# 8.1 使用数组来演示腐蚀的基本原理
import cv2
import numpy as np

img = np.zeros((5, 5), np.uint8)
img[1:4, 1:4] = 1
kernel = np.ones((3, 1), np.uint8)
erosion = cv2.erode(img, kernel)
print("img=\n", img)
print("kernel=\n", kernel)
print("erosion=\n", erosion)

# %%
# 8.2 使用函数cv2.erode()完成图像腐蚀
import cv2
import numpy as np

o = cv2.imread("lesson_8\\erode.png", cv2.IMREAD_UNCHANGED)
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(o, kernel)
cv2.imshow("original", o)
cv2.imshow("erosion", erosion)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 8.3 调节参数控制不同参数控制下的图像腐蚀效果
import cv2
import numpy as np

o = cv2.imread("lesson_8\\erode.png", cv2.IMREAD_UNCHANGED)
kernel = np.ones((9, 9), np.uint8)
erosion = cv2.erode(o, kernel, iterations=5)
cv2.imshow("original", o)
cv2.imshow("erosion", erosion)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 8.4 数组演示膨胀的基本原理
import cv2
import numpy as np

img = np.zeros((5, 5), np.uint8)
img[2:3, 1:4] = 1
kernel = np.ones((3, 1), np.uint8)
dilation = cv2.dilate(img, kernel)
print("img=\n", img)
print("kernel=\n", kernel)
print("dilation=\n", dilation)

# %%
# 8.5 使用cv2.dilate()完成图像膨胀
import cv2
import numpy as np

o = cv2.imread("lesson_8\\dilation.png", cv2.IMREAD_UNCHANGED)
kernel = np.ones((9, 9), np.uint8)
dilation = cv2.dilate(o, kernel)
cv2.imshow("original", o)
cv2.imshow("dilation", dilation)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 8.6调节参数查看不同参数控制下图像的膨胀效果
import cv2
import numpy as np

o = cv2.imread("lesson_8\\dilation.png", cv2.IMREAD_UNCHANGED)
kernel = np.ones((5, 5), np.uint8)
dilation = cv2.dilate(o, kernel, iterations=9)
cv2.imshow("original", o)
cv2.imshow("dilation", dilation)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 8.7 使用开运算进行去噪和计数
import cv2
import numpy as np

img1 = cv2.imread("lesson_8\\opening.png")
img2 = cv2.imread("lesson_8\\opening2.png")
k = np.ones((10, 10), np.uint8)
r1 = cv2.morphologyEx(img1, cv2.MORPH_OPEN, k)
r2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, k)
cv2.imshow("img1", img1)
cv2.imshow("result1", r1)
cv2.imshow("img2", img2)
cv2.imshow("result2", r2)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 8.8 使用函数闭运算
import cv2
import numpy as np

img1 = cv2.imread("lesson_8\\closing.png")
img2 = cv2.imread("lesson_8\\closing2.png")
k = np.ones((15, 15), np.uint8)
r1 = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, k, iterations=3)
r2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, k, iterations=3)
cv2.imshow("img1", img1)
cv2.imshow("result1", r1)
cv2.imshow("img2", img2)
cv2.imshow("result2", r2)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 8.9
import cv2
import numpy as np

o = cv2.imread("lesson_8\\gradient.png")
k = np.ones((5, 5), np.uint8)
r = cv2.morphologyEx(o, cv2.MORPH_GRADIENT, k)
cv2.imshow("original", o)
cv2.imshow("result", r)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 8.10 使用礼貌运算处理图像,就是得到图像的边缘
import cv2
import numpy as np

o1 = cv2.imread("lesson_8\\erode.png", cv2.IMREAD_UNCHANGED)
o2 = cv2.imread("lesson_8\\danger.png", cv2.IMREAD_UNCHANGED)
k = np.ones((5, 5), np.uint8)
r1 = cv2.morphologyEx(o1, cv2.MORPH_TOPHAT, k)
r2 = cv2.morphologyEx(o2, cv2.MORPH_TOPHAT, k)
cv2.imshow("original1", o1)
cv2.imshow("original", o2)
cv2.imshow("result1", r1)
cv2.imshow("result2", r2)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 8.11
import cv2
import numpy as np

o1 = cv2.imread("lesson_8\\closing.png", cv2.IMREAD_UNCHANGED)
o2 = cv2.imread("lesson_8\\danger.png", cv2.IMREAD_UNCHANGED)
k = np.ones((5, 5), np.uint8)
r1 = cv2.morphologyEx(o1, cv2.MORPH_BLACKHAT, k)
r2 = cv2.morphologyEx(o2, cv2.MORPH_BLACKHAT, k)
cv2.imshow("original1", o1)
cv2.imshow("original", o2)
cv2.imshow("result1", r1)
cv2.imshow("result2", r2)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 8.12 使用函数cv2.getStructuringElement()生成不同结构的核
import cv2

kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
print("kernel1=\n", kernel1)
print("kernel2=\n", kernel2)
print("kernel3=\n", kernel3)

# %%
# 8.13 观察不同的核对形态学操作做的影响
import cv2

o = cv2.imread("lesson_8\\kernel.png", cv2.IMREAD_UNCHANGED)
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (59, 59))
kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (59, 59))
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (59, 59))
dst1 = cv2.dilate(o, kernel1)
dst2 = cv2.dilate(o, kernel2)
dst3 = cv2.dilate(o, kernel3)
cv2.imshow("original", o)
cv2.imshow("dst1", dst1)
cv2.imshow("dst2", dst2)
cv2.imshow("dst3", dst3)
cv2.waitKey()
cv2.destroyAllWindows()




















