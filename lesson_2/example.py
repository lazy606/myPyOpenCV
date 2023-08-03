# example.py - 第二章例题文件

# %%
# 2.1用Numpy库生成一个8X8的数组，用来模拟二值图像，并对其进行简单修改
import cv2
import numpy as np

img = np.zeros((64, 64), dtype=np.uint8)
print("image=\n", img)
cv2.imshow("one", img)
print("读取像素点img[0, 3]=", img[0, 3])
img[0, 3] = 255
print("修改后img=\n", img)
print("读取修改后像素点img[0, 3]", img[0, 3])
cv2.imshow("two", img)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 2.2读取一个灰度图像，并对其像素进行访问、修改
import cv2

img = cv2.imread(".\\lesson_2\\fiveHunderMiles.jpg", 0)
cv2.imshow("before", img)
for i in range(10, 100):
    for j in range(80, 100):
        img[i, j] = 0
cv2.imshow("after", img)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 2.3 使用Numpy生成三维数组，用来观察三个通道值的变化情况
# -----------蓝色通道值--------------
import cv2
import numpy as np

blue = np.zeros((300, 300, 3), dtype=np.uint8)
blue[:, :, 0] = 255
print("blue=\n", blue)
cv2.imshow("blue", blue)
# -----------绿色通道值--------------
green = np.zeros((300, 300, 3), dtype=np.uint8)
green[:, :, 1] = 255
print("green=\n", green)
cv2.imshow("green", green)
# -----------红色通道值--------------
red = np.zeros((300, 300, 3), dtype=np.uint8)
red[:, :, 2] = 255
print("red=\n", red)
cv2.imshow("red", red)
# -----------释放窗口--------------
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 2.4 生成三维数组，用来观察通道值的变化
import numpy as np
import cv2

img = np.zeros((300, 300, 3), dtype=np.uint8)
img[:, 0:100, 0] = 255
img[:, 100:200, 1] = 255
img[:, 200:300, 2] = 255
print("img=\n", img)
cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 2.5用numpy生成一个图像数组并操作
import numpy as np

img = np.zeros((2, 4, 3), dtype=np.uint8)
print("img=\n", img)
print("读取像素点img[0, 3]=", img[0, 3])
print("读取像素点img[1, 2, 2]", img[1, 2, 2])
img[0, 3] = 255
img[0, 0] = [66, 77, 88]
img[1, 1, 1] = 3
img[1, 2, 2] = 4
img[0, 2, 0] = 5
print("修改后img\n", img)
print("读取修改后像素点img[1, 2, 2]=", img[1, 2, 2])

# %%
# 2.6 读取一副彩色图像，并对其像素进行修改
import cv2

img = cv2.imread(".\\lesson_2\\fiveHunderMiles.jpg")
cv2.imshow("before", img)
print("访问img[0, 0]=", img[0, 0])
print("访问img[0, 0, 0]=", img[0, 0, 0])
print("访问img[0, 0, 1]=", img[0, 0, 1])
print("访问img[0, 0, 2]=", img[0, 0, 2])
print("访问img[50, 0]=", img[50, 2])
print("访问img[100, 0]=", img[100, 0])
# 区域1
for i in range(0, 50):
    for j in range(0, 100):
        for k in range(0, 3):
            img[i, j, k] = 255  # 白色
# 区域2
for i in range(50, 100):
    for j in range(0, 100):
        img[i, j] = [128, 128, 128]
# 区域3
for i in range(100, 150):
    for j in range(0, 100):
        img[i, j] = 0
cv2.imshow("after", img)
print("修改后 img[0,0]=", img[0, 0])
print("修改后 img[0,0,0]=", img[0, 0, 0])
print("修改后 img[0,0,1]=", img[0, 0, 1])
print("修改后 img[0,0,2]=", img[0, 0, 2])
print("修改后 img[50,0]=", img[50, 0])
print("修改后 img[100,0]=", img[100, 0])
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 2.7 使用Numpy生成二维随机数组并使用item和itemset来访问和修改图像
import numpy as np

img = np.random.randint(10, 99, size=[5, 5], dtype=np.uint8)
print("img=\n", img)
print("读取像素点img.item(3, 2)=", img.item(3, 2))
img.itemset((3, 2), 255)
print("修改后img=\n", img)
print("修改后像素img.item(3, 2)=", img.item(3, 2))

# %%
# 2.8 生成随机数灰度图像
import numpy as np
import cv2

img = np.random.randint(0, 256, size=[256, 256], dtype=np.uint8)
cv2.imshow("demo", img)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 2.9 读取一幅灰度图像，并赌气像素值进行修改
import cv2

img = cv2.imread(".\\lesson_2\\fiveHunderMiles.jpg", 0)
# 测试读取、修改单个像素值
print("读取像素点img.item(3, 2)=", img.item(3, 2))
img.itemset((3, 2), 255)
print("修改后像素点img.item(3, 2)=", img.item(3, 2))
# 测试修改一个区域的像素值
cv2.imshow("before", img)
for i in range(10, 100):
    for j in range(80, 100):
        img.itemset((i, j), 255)
cv2.imshow("after", img)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 2.10 使用numpy随机生成三维数组，模拟彩色空间
import numpy as np

img = np.random.randint(10, 99, size=[2, 4, 3], dtype=np.uint8)
print("img=\n", img)
print("读取像素点 img[1,2,0]=", img.item(1, 2, 0))
print("读取像素点 img[0,2,1]=", img.item(0, 2, 1))
print("读取像素点 img[1,0,2]=", img.item(1, 0, 2))
img.itemset((1, 2, 0), 255)
img.itemset((0, 2, 1), 255)
img.itemset((1, 0, 2), 255)
print("修改后img=\n", img)
print("读取像素点 img[1,2,0]=", img.item(1, 2, 0))
print("读取像素点 img[0,2,1]=", img.item(0, 2, 1))
print("读取像素点 img[1,0,2]=", img.item(1, 0, 2))

# %%
# 2.11生成一幅彩色图像，让其中的像素值均位随机数
import cv2
import numpy as np

img = np.random.randint(0, 256, size=[256, 256, 3], dtype=np.uint8)
cv2.imshow("demo", img)
cv2.waitKey()
cv2.destroyAllWindows()

# %% 读取一幅彩色图像对其像素进行访问
import cv2
import numpy as np

img = cv2.imread(".\\lesson_2\\fiveHunderMiles.jpg")
cv2.imshow("before", img)
print("访问 img.item(0,0,0)=", img.item(0, 0, 0))
print("访问 img.item(0,0,1)=", img.item(0, 0, 1))
print("访问 img.item(0,0,2)=", img.item(0, 0, 2))
for i in range(0, 50):
    for j in range(0, 100):
        for k in range(0, 3):
            img.itemset((i, j, k), 255)
cv2.imshow("after", img)
print("修改后 img.item(0,0,0)=", img.item(0, 0, 0))
print("修改后 img.item(0,0,1)=", img.item(0, 0, 1))
print("修改后 img.item(0,0,2)=", img.item(0, 0, 2))
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 2.13 获取图像脸部信息并显示
import cv2

a = cv2.imread(".\\lesson_2\\fiveHunderMiles.jpg", cv2.IMREAD_UNCHANGED)
face = a[220:400, 250:350]
cv2.imshow("original", a)
cv2.imshow("face", face)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 2.14 对图像进行打码
import cv2
import numpy as np

a = cv2.imread(".\\lesson_2\\fiveHunderMiles.jpg", cv2.IMREAD_UNCHANGED)
cv2.imshow("original", a)
face = np.random.randint(0, 256, size=[180, 100, 3])
a[220:400, 250:350] = face
cv2.imshow("result", a)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 2.15将一幅图像内的 ROI 复制到另一幅图像内。
import cv2

lena = cv2.imread("lena512.bmp", cv2.IMREAD_UNCHANGED)
dollar = cv2.imread("dollar.bmp", cv2.IMREAD_UNCHANGED)
cv2.imshow("lena", lena)
cv2.imshow("dollar", dollar)
face = lena[220:400, 250:350]
dollar[160:340, 200:300] = face
cv2.imshow("result", dollar)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 2.16 演示图像通道拆分及通道值改变对彩色图像的影响
import cv2
import numpy as np

lena = cv2.imread(".\\lesson_2\\fiveHunderMiles.jpg", cv2.IMREAD_UNCHANGED)
cv2.imshow("lena1", lena)
b = lena[:, :, 0]
g = lena[:, :, 1]
r = lena[:, :, 2]
cv2.imshow("b", b)
cv2.imshow("g", g)
cv2.imshow("r", r)
lena[:, :, 0] = 0
cv2.imshow("lenab0", lena)
lena[:, :, 1] = 0
cv2.imshow("lenab0g0", lena)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 2.17 使用cv2.split拆分图像通道
import cv2

lena = cv2.imread(".\\lesson_2\\fiveHunderMiles.jpg")
b, g, r = cv2.split(lena)
cv2.imshow("B", b)
cv2.imshow("G", g)
cv2.imshow("R", r)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 2.18 使用cv2.merge合并通道
import cv2

lena = cv2.imread(".\\lesson_2\\fiveHunderMiles.jpg")
b, g, r = cv2.split(lena)
bgr = cv2.merge([b, g, r])
rgb = cv2.merge([r, g, b])
cv2.imshow("lena", lena)
cv2.imshow("bgr", bgr)
cv2.imshow("rgb", rgb)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 2.19 观察图像常用属性
import cv2

gray = cv2.imread(".\\lesson_2\\fiveHunderMiles.jpg", 0)
color = cv2.imread(".\\lesson_2\\fiveHunderMiles.jpg")
print("图像gray属性：")
print("gray.shape=", gray.shape)
print("gray.size=", gray.size)
print("gray.dtype=", gray.dtype)
print("图像color属性：")
print("color.shape=", color.shape)
print("color.size=", color.size)
print("color.dtype=", color.dtype)