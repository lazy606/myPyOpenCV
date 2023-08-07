# %%
# 5.1 使用函数cv2.resize()对一个数组进行简单放缩
import cv2
import numpy as np

img = np.ones([2, 4, 3], dtype=np.uint8)
size = img.shape[:2]
rst = cv2.resize(img, size)
print("img.shape=\n", img.shape)
print("img=\n", img)
print("rst.shape=\n", rst.shape)
print("rst=\n", rst)

# %%
# 5.2 使用函数cv2.resize()完成一个简单的图像放缩
import cv2

img = cv2.imread(".\\lesson_5\\danger.png")
rows, cols = img.shape[:2]
size = (int(cols * 0.9), int(rows * 0.5))
rst = cv2.resize(img, size)
print("img.shape=", img.shape)
print("rst.shape=", rst.shape)

# %%
# 5.3 控制函数cv2.resize()的fx参数、fy参数，完成图像缩放
import cv2

img = cv2.imread(".\\lesson_5\\danger.png")
rst = cv2.resize(img, None, fx=2, fy=0.5)
print("img.shape=", img.shape)
print("rst.shape=", rst.shape)

# %%
# 5.4 使用cv2.flip()完成图像的翻转
import cv2

img = cv2.imread(".\\lesson_5\\danger.png")
x = cv2.flip(img, 0)
y = cv2.flip(img, 1)
xy = cv2.flip(img, -1)
cv2.imshow("img", img)
cv2.imshow("x", x)
cv2.imshow("y", y)
cv2.imshow("xy", xy)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 5.5 利用自定义转换矩阵完成图像的平移
import cv2
import numpy as np

img = cv2.imread(".\\lesson_5\\danger.png")
height, width = img.shape[:2]
x = 100
y = 200
M = np.float32([[1, 0, x], [0, 1, y]])
move = cv2.warpAffine(img, M, (width, height))
cv2.imshow("original", img)
cv2.imshow("move", move)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 5.6 获取选择矩阵M，并将图像旋转
import cv2

img = cv2.imread(".\\lesson_5\\danger.png")
height, width = img.shape[:2]
M = cv2.getRotationMatrix2D((width / 2, height / 2), 45, 0.6)
rotate = cv2.warpAffine(img, M, (width, height))
cv2.imshow("original", img)
cv2.imshow("rotation", rotate)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 5.7
import cv2
import numpy as np

img = cv2.imread(".\\lesson_5\\danger.png")
rows, cols, ch = img.shape
p1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
p2 = np.float32([[0, rows * 0.33], [cols * 0.85, rows * 0.25], [cols * 0.15, rows * 0.7]])
M = cv2.getAffineTransform(p1, p2)
dst = cv2.warpAffine(img, M, (cols, rows))
cv2.imshow("original", img)
cv2.imshow("result", dst)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 5.8 完成从平行四边形到矩形的转换。
import cv2
import numpy as np

img = cv2.imread(".\\lesson_5\\danger.png")
rows, cols = img.shape[:2]
print(rows, cols)
pts1 = np.float32([[150, 50], [400, 50], [60, 450], [310, 450]])
pts2 = np.float32([[50, 50], [rows - 50, 50], [50, cols - 50], [rows - 50, cols - 50]])
M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M, (cols, rows))
cv2.imshow("img", img)
cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 5.9 将目标数组内所有像素点都映射为原始图像内第0行第3列上的像素点
import cv2
import numpy as np

img = np.random.randint(0, 256, size=[4, 5], dtype=np.uint8)
rows, cols = img.shape
mapx = np.ones(img.shape, np.float32) * 3
mapy = np.ones(img.shape, np.float32) * 0
rst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)  # 这个是反向映射，即结果图片中像素点再原图像中的位置
print("img=\n", img)
print("mapx=\n", mapx)
print("mapy=\n", mapy)
print("rst=\n", rst)

# %%
# 5.10 使用cv2.remap()完成数组复制，那就是构建两个与原图等大的数组，数组中的元素就是数组的序号
import cv2
import cv2

img = np.random.randint(0, 256, size=[4, 5], dtype=np.uint8)
rows, cols = img.shape
mapx = np.zeros(img.shape, np.float32)
mapy = np.zeros(img.shape, np.float32)
for i in range(rows):
    for j in range(cols):
        mapx.itemset((i, j), j)
        mapy.itemset((i, j), i)
rst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)  # mapx表示是参考的列， mapy表示参考的行
print("img=\n", img)
print("mapx=\n", mapx)
print("mapy=\n", mapy)
print("rst=\n", rst)

# %%
# 5.11 使用cv2.remap()完成图像复制
import cv2
import numpy as np

img = cv2.imread(".\\lesson_5\\danger.png")
rows, cols = img.shape[:2]
map_col = np.zeros(img.shape[:2], np.float32)
map_row = np.zeros(img.shape[:2], np.float32)
for i in range(rows):
    for j in range(cols):
        map_col.itemset((i, j), j)
        map_row.itemset((i, j), i)
rst = cv2.remap(img, map_col, map_row, cv2.INTER_LINEAR)
cv2.imshow("original", img)
cv2.imshow("result", rst)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 5.12 使用cv2.remap()实现绕x轴翻转
import cv2
import numpy as np

rows, cols = img.shape
mapx = np.zeros(img.shape, np.float32)
mapy = np.zeros(img.shape, np.float32)
for i in range(rows):
    for j in range(cols):
        mapx.itemset((i, j), j)
        mapy.itemset((i, j), rows - 1 - i)
rst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
print("img=\n", img)
print("mapx=\n", mapx)
print("mapy=\n", mapy)
print("rst=\n", rst)

# %%
# 5.14 数组翻转
import cv2
import numpy as np

img = np.random.randint(0, 256, size=[4, 5], dtype=np.uint8)
rows, cols = img.shape
mapx = np.zeros(img.shape, np.float32)
mapy = np.zeros(img.shape, np.float32)
for i in range(rows):
    for j in range(cols):
        mapx.itemset((i, j), cols - 1 - j)
        mapy.itemset((i, j), i)
rst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
print("img=\n", img)
print("mapx=\n", mapx)
print("mapy=\n", mapy)
print("rst=\n", rst)

# %%
# 5.15 图片翻转
import cv2
import numpy as np

img = cv2.imread("lena.bmp")
rows, cols = img.shape[:2]
mapx = np.zeros(img.shape[:2], np.float32)
mapy = np.zeros(img.shape[:2], np.float32)
for i in range(rows):
    for j in range(cols):
        mapx.itemset((i, j), cols - 1 - j)
        mapy.itemset((i, j), i)
rst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
cv2.imshow("original", img)
cv2.imshow("result", rst)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 5.16 数组绕xy轴翻转
import cv2
import numpy as np

img = np.random.randint(0, 256, size=[4, 5], dtype=np.uint8)
rows, cols = img.shape
mapx = np.zeros(img.shape, np.float32)
mapy = np.zeros(img.shape, np.float32)
for i in range(rows):
    for j in range(cols):
        mapx.itemset((i, j), cols - 1 - j)
        mapy.itemset((i, j), rows - 1 - i)
rst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
print("img=\n", img)
print("mapx=\n", mapx)
print("mapy=\n", mapy)
print("rst=\n", rst)

# %%
# 5.17 图片绕x、y轴翻转
import cv2
import numpy as np

img = cv2.imread("lena.bmp")
rows, cols = img.shape[:2]
mapx = np.zeros(img.shape[:2], np.float32)
mapy = np.zeros(img.shape[:2], np.float32)
for i in range(rows):
    for j in range(cols):
        mapx.itemset((i, j), cols - 1 - j)
        mapy.itemset((i, j), rows - 1 - i)
rst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
cv2.imshow("original", img)
cv2.imshow("result", rst)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 5.18 实现数组x、y轴互换
import cv2
import numpy as np

img = np.random.randint(0, 256, size=[4, 6], dtype=np.uint8)
rows, cols = img.shape
mapx = np.zeros(img.shape, np.float32)
mapy = np.zeros(img.shape, np.float32)
for i in range(rows):
    for j in range(cols):
        mapx.itemset((i, j), i)
        mapy.itemset((i, j), j)
rst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
print("img=\n", img)
print("mapx=\n", mapx)
print("mapy=\n", mapy)
print("rst=\n", rst)

# %%
# 5.19 实现数组互换
import cv2
import numpy as np

img = cv2.imread("lena.bmp")
rows, cols = img.shape[:2]
mapx = np.zeros(img.shape[:2], np.float32)
mapy = np.zeros(img.shape[:2], np.float32)
for i in range(rows):
    for j in range(cols):
        mapx.itemset((i, j), i)
        mapy.itemset((i, j), j)
rst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
cv2.imshow("original", img)
cv2.imshow("result", rst)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 5.20 实现图像缩小,这个不好用
import cv2
import numpy as np

img = cv2.imread(".\\lesson_5\\danger.png")
rows, cols = img.shape[:2]
mapx = np.zeros(img.shape[:2], np.float32)
mapy = np.zeros(img.shape[:2], np.float32)
for i in range(rows):
    for j in range(cols):
        if 0.25 * cols < i < 0.75 * cols and 0.25 * rows < j < 0.75 * rows:
            mapx.itemset((i, j), 2 * (j - cols * 0.25) + 0.5)
            mapy.itemset((i, j), 2 * (i - rows * 0.25) + 0.5)
        else:
            mapx.itemset((i, j), 0)
            mapy.itemset((i, j), 0)
print(mapx)
print(mapy)
rst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
cv2.imshow("original", img)
cv2.imshow("result", rst)
cv2.waitKey()
cv2.destroyAllWindows()
