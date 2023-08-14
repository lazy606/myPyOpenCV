# %%
# 9.1 使用cv2.convertScaleAbs()取绝对值
import cv2
import numpy as np

img = np.random.randint(-256, 256, size=[4, 5], dtype=np.int16)
rst = cv2.convertScaleAbs(img)
print("img=\n", img)
print("rst=\n", rst)

# %%
# 9.2 使用函数cv2.Sobel()获取水平方向的编译信息。
import cv2

o = cv2.imread("lesson_9\\sobel4.png", cv2.IMREAD_GRAYSCALE)
sobel_x = cv2.Sobel(o, -1, 1, 0)
print("sobel_x=\n", sobel_x)
cv2.imshow("original", o)
cv2.imshow("sobel_x", sobel_x)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 9.3 使用函数cv2.Sobel()获取图像水平方向的完整边缘信息,在本例中将参数ddpeth的值设置为cv2.CV_64
import cv2

o = cv2.imread("lesson_9\\sobel4.png", cv2.IMREAD_GRAYSCALE)
sobel_x = cv2.Sobel(o, cv2.CV_64F, 1, 0)
sobel_x = cv2.convertScaleAbs(sobel_x)
cv2.imshow("original", o)
cv2.imshow("x", sobel_x)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 9.4 使用函数cv2.Sobel()获取图像水平方向的完整边缘信息,在本例中将参数ddpeth的值设置为cv2.CV_64
import cv2

o = cv2.imread("lesson_9\\sobel4.png", cv2.IMREAD_GRAYSCALE)
sobel_y = cv2.Sobel(o, cv2.CV_64F, 0, 1)
sobel_y = cv2.convertScaleAbs(sobel_y)
cv2.imshow("original", o)
cv2.imshow("x", sobel_y)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 9.5 当参数dx和dy的值为dx=1, dy=1是，查看函数cv2.Sobel()的执行效果
import cv2

o = cv2.imread("lesson_9\\sobel4.png", cv2.IMREAD_GRAYSCALE)
sobel_xy = cv2.Sobel(o, cv2.CV_64F, 1, 1)
sobel_xy = cv2.convertScaleAbs(sobel_xy)
cv2.imshow("original", o)
cv2.imshow("xy", sobel_xy)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 9.6 计算cv2.Sobel()在水平、垂直两个方向叠加的边缘信息
import cv2

o = cv2.imread("lesson_9\\sobel4.png", cv2.IMREAD_GRAYSCALE)
sobel_x = cv2.Sobel(o, cv2.CV_64F, 1, 0)
sobel_y = cv2.Sobel(o, cv2.CV_64F, 0, 1)
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)
sobel_xy = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
cv2.imshow("original", o)
cv2.imshow("xy", sobel_xy)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 9.7 使用不同的方式处理图像在两个方向上的边缘信息
'''
方式一：分别使用“dx=1, dy=0”和“dx=0, dy=1”计算图像在水平方向和垂直方向的边
缘信息，然后将二者相加，构成两个方向的边缘信息。
方式 2：将参数 dx 和 dy 的值设为“dx=1, dy=1”，获取图像在两个方向的梯度。
'''
import cv2

o = cv2.imread("lesson_9\\danger.png", cv2.IMREAD_GRAYSCALE)
sobel_x = cv2.Sobel(o, cv2.CV_64F, 1, 0)
sobel_y = cv2.Sobel(o, cv2.CV_64F, 0, 1)
sobel_xy = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
sobel_xy11 = cv2.Sobel(o, cv2.CV_64F, 1, 1)
sobel_xy11 = cv2.convertScaleAbs(sobel_xy11)
cv2.imshow("original", o)
cv2.imshow("xy", sobel_xy)
cv2.imshow("xy11", sobel_xy11)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 9.8 使用函数cv2.Scharr()回去图像水平方向的边缘信息。
import cv2

o = cv2.imread("lesson_9\\scharr.png", cv2.IMREAD_GRAYSCALE)
scharrx = cv2.Scharr(o, cv2.CV_64F, 1, 0)
scharrx = cv2.convertScaleAbs(scharrx)
cv2.imshow("original", o)
cv2.imshow("x", scharrx)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 9.9 使用函数cv2.Scharr()回去图像垂直方向的边缘信息。
import cv2

o = cv2.imread("lesson_9\\scharr.png", cv2.IMREAD_GRAYSCALE)
scharrx = cv2.Scharr(o, cv2.CV_64F, 0, 1)
scharrx = cv2.convertScaleAbs(scharrx)
cv2.imshow("original", o)
cv2.imshow("x", scharrx)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 9.9 使用函数cv2.Scharr()回去图像边缘叠加效果
import cv2

o = cv2.imread("lesson_9\\scharr.png", cv2.IMREAD_GRAYSCALE)
scharrx = cv2.Scharr(o, cv2.CV_64F, 1, 0)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.Scharr(o, cv2.CV_64F, 0, 1)
scharry = cv2.convertScaleAbs(scharry)
scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
cv2.imshow("original", o)
cv2.imshow("xy", scharrxy)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 9.12 使用cv2.Sobel()完成Scharr算子运算。
import cv2

o = cv2.imread('Sobel4.bmp',cv2.IMREAD_GRAYSCALE)
Scharrx = cv2.Sobel(o,cv2.CV_64F,1,0,-1)
Scharry = cv2.Sobel(o,cv2.CV_64F,0,1,-1)
Scharrx = cv2.convertScaleAbs(Scharrx)
Scharry = cv2.convertScaleAbs(Scharry)
cv2.imshow("original",o)
cv2.imshow("x",Scharrx)
cv2.imshow("y",Scharry)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 9.13 分别使用Sobel算子和Scharr算子计算一幅图像水平边缘和垂直边缘的叠加结果
import cv2

o = cv2.imread("lesson_9\\danger.png", cv2.IMREAD_GRAYSCALE)
sobel_x = cv2.Sobel(o, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(o, cv2.CV_64F, 0, 1, ksize=3)
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)
sobel_xy = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

scharr_x = cv2.Scharr(o, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(o, cv2.CV_64F, 0, 1)
scharr_x = cv2.convertScaleAbs(scharr_x)
scharr_y = cv2.convertScaleAbs(scharr_y)
scharr_xy = cv2.addWeighted(scharr_x, 0.5, scharr_y, 0.5, 0)
cv2.imshow("original", o)
cv2.imshow("sobelxy", sobel_xy)
cv2.imshow("scharrxy", scharr_xy)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 9.14 使用函数cv2.Laplacian()计算图像的边缘信息,这个算子就不需要指定方向了，直接对每个像素使用
import cv2

o = cv2.imread("lesson_9\\danger.png", cv2.IMREAD_GRAYSCALE)
laplacian = cv2.Laplacian(o, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)
cv2.imshow("original", o)
cv2.imshow("laplacian", laplacian)
cv2.waitKey()
cv2.destroyAllWindows()