# %%
# 11.1使用函数cv2.pyrDown()对一幅图像进行向下取样
import cv2

o = cv2.imread("lesson_11\\danger.png", cv2.IMREAD_GRAYSCALE)
r1 = cv2.pyrDown(o)
r2 = cv2.pyrDown(r1)
r3 = cv2.pyrDown(r2)
print("o.shape=", o.shape)
print("r1.shape=", r1.shape)
print("r2.shape=", r2.shape)
print("r3.shape=", r3.shape)
cv2.imshow("original", o)
cv2.imshow("r1", r1)
cv2.imshow("r2", r2)
cv2.imshow("r3", r3)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 11.2 使用cv2.pyrUp()对图像进行向上采样
import cv2

o = cv2.imread("lesson_11\\danger.png", cv2.IMREAD_GRAYSCALE)
r1 = cv2.pyrUp(o)
r2 = cv2.pyrUp(r1)
r3 = cv2.pyrUp(r2)
print("o.shape=", o.shape)
print("r1.shape=", r1.shape)
print("r2.shape=", r2.shape)
print("r3.shape=", r3.shape)
cv2.imshow("original", o)
cv2.imshow("r1", r1)
cv2.imshow("r2", r2)
cv2.imshow("r3", r3)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 11.3 使用函数cv2.pyrDown()和cv2.pyrUp()，先后将一幅图像进行向下采用、向上采用观察采样结果即结果图像和原始图像的差异
import cv2

o = cv2.imread("lesson_11\\danger.png")
down = cv2.pyrDown(o)
up = cv2.pyrUp(down)
diff = up - o  # 大小不一致不能运算
print("o.shape=", o.shape)
print("up.shape=", up.shape)
cv2.imshow("original", o)
cv2.imshow("up", up)
cv2.imshow("difference", diff)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 11.4 使用函数cv2.pyrUp()和cv2.pyrDown()，先后将一幅图形进行向上采样和向下采样，观察结果
import cv2

o = cv2.imread("lesson_11\\danger.png")
up = cv2.pyrUp(o)
down = cv2.pyrDown(up)
diff = down - o  # 大小不一致不能运算
print("o.shape=", o.shape)
print("up.shape=", up.shape)
cv2.imshow("original", o)
cv2.imshow("down", down)
cv2.imshow("difference", diff)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 11.5 使用函数cv2.pyrDown()和cv2.pyrUp()构造拉普拉斯金字塔
import cv2

o = cv2.imread("lesson_11\\danger.png")
g0 = o
g1 = cv2.pyrDown(g0)
g2 = cv2.pyrDown(g1)
g3 = cv2.pyrDown(g2)
l0 = g0 - cv2.pyrUp(g1)
l1 = g1 - cv2.pyrUp(g2)
l2 = g2 - cv2.pyrUp(g3)
print("l0.shape", l0.shape)
print("l1.shape", l1.shape)
print("l2.shape", l2.shape)
cv2.imshow("l0", l0)
cv2.imshow("l1", l1)
cv2.imshow("l2", l2)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 11.6 编写程序使用拉普拉斯金字塔恢复原始图像
import cv2
import numpy as np

o = cv2.imread("lesson_11\\danger.png")
g0 = o
g1 = cv2.pyrDown(o)
l0 = g0 - cv2.pyrUp(g1)
r0 = l0 + cv2.pyrUp(g1)  # 通过拉普拉斯复原图像
print("o.shape=", o.shape)
print("r0.shape=", r0.shape)
result = r0 - o
result = abs(result)
print("原始图像o和恢复图像ro之差的绝对值和:", np.sum(result))

# %%
# 11.7 编写程序使用拉普拉斯金字塔和高斯金字塔恢复高斯金字塔内的多层图像
import cv2
import numpy as np

o = cv2.imread("lesson_11\\danger.png")
# ==============生成高斯金字塔===================
G0 = o
G1 = cv2.pyrDown(G0)
G2 = cv2.pyrDown(G1)
G3 = cv2.pyrDown(G2)
# ============生成拉普拉斯金字塔===================
L0 = G0 - cv2.pyrUp(G1)
L1 = G1 - cv2.pyrUp(G2)
L2 = G2 - cv2.pyrUp(G3)
# =================复原 G0======================
RG0 = L0 + cv2.pyrUp(G1)  # 通过拉普拉斯图像复原的原始图像 G0
print("G0.shape=", G0.shape)
print("RG0.shape=", RG0.shape)
result = RG0 - G0  # 将 RG0 和 G0 相减
result = RG0 - G0  # 将 RG0 和 G0 相减
# 计算 result 的绝对值，避免求和时负负为正，3+(-3)=0
result = abs(result)
# 计算 result 所有元素的和
print("原始图像 G0 与恢复图像 RG0 差值的绝对值和：", np.sum(result))
# =================复原 G1======================
RG1 = L1 + cv2.pyrUp(G2)  # 通过拉普拉斯图像复原 G1
print("G1.shape=", G1.shape)
print("RG1.shape=", RG1.shape)
result = RG1 - G1  # 将 RG1 和 G1 相减
print("原始图像 G1 与恢复图像 RG1 之差的绝对值和：", np.sum(abs(result)))
# =================复原 G2======================
RG2 = L2 + cv2.pyrUp(G3)  # 通过拉普拉斯图像复原 G2
print("G2.shape=", G2.shape)
print("RG2.shape=", RG2.shape)
result = RG2 - G2  # 将 RG2 和 G2 相减
print("原始图像 G2 与恢复图像 RG2 之差的绝对值和：", np.sum(abs(result)))
