# %%
# 4.1 将BGR图像转为灰度图像
import cv2
import numpy as np

img = np.random.randint(0, 256, size=[2, 4, 3], dtype=np.uint8)
rst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("img=\n", img)
print("rst=\n", rst)
print("像素点(1, 0)直接计算得到的值=", img[1, 0, 0] * 0.114 + img[1, 0, 1] * 0.587 + img[1, 0, 2] * 0.299)
print("像素点(1, 0)使用公式cv2.cvtColor()转换值=", rst[1, 0])

# %%
# 4.2 灰度图像转BGR图像
import cv2
import numpy as np

img = np.random.randint(0, 256, size=[2, 4], dtype=np.uint8)
rst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
print("img=\n", img)
print("rst=\n", rst)

# %%
# 4.3 将图像在BGR和RGB模式之间进行相互转换
import cv2
import numpy as np

img = np.random.randint(0, 256, size=[2, 4, 3], dtype=np.uint8)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
print("img=\n", img)
print("rgb=\n", rgb)
print("bgr=\n", bgr)

# %%
# 4.4将图像在BGR模式和灰度图像之间相互转换
import cv2

lena = cv2.imread(".\\lesson_4\\img.png")
gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
# ==========打印 shape============
print("lena.shape=", lena.shape)
print("gray.shape=", gray.shape)
print("rgb.shape=", rgb.shape)
# ==========显示效果============
cv2.imshow("lena", lena)
cv2.imshow("gray", gray)
cv2.imshow("rgb", rgb)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 4.5 将图像从BGR模式转换为RGB模式
import cv2

lena = cv2.imread(".\\lesson_4\\img.png")
rgb = cv2.cvtColor(lena, cv2.COLOR_BGR2RGB)
cv2.imshow("lena", lena)
cv2.imshow("rgb", rgb)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# H(色调)色调是直接表示我们看到的颜色;
import cv2
import numpy as np

# =========测试一下 OpenCV 中蓝色的 HSV 模式值=============
img_blue = np.zeros([1, 1, 3], dtype=np.uint8)
img_blue[0, 0, 0] = 255
blue = img_blue
blue_HSV = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
print("blue=\n", blue)
print("blue_HSV=\n", blue_HSV)
# =========测试一下 OpenCV 中绿色的 HSV 模式值=============
img_green = np.zeros([1, 1, 3], dtype=np.uint8)
img_green[0, 0, 1] = 255
green = img_green
green_HSV = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
print("green=\n", green)
print("green_HSV=\n", green_HSV)
# =========测试一下 OpenCV 中红色的 HSV 模式值=============
img_red = np.zeros([1, 1, 3], dtype=np.uint8)
img_red[0, 0, 2] = 255
red = img_red
red_HSV = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
print("Red=\n", red)
print("RedHSV=\n", red_HSV)

# %%
# 4.7 使用cv2.inRange()将某个图像内的在[100, 200]内的值标注出来
import cv2
import numpy as np

img = np.random.randint(0, 256, size=[5, 5], dtype=np.uint8)
min = 100
max = 200
mask = cv2.inRange(img, min, max)
print("img=\n", img)
print("img=\n", mask)

# %%
# 4.8 正常显示某个图像内的感兴趣区域，而将其他区域显示为黑色
import cv2
import numpy as np

img = np.ones([5, 5], dtype=np.uint8) * 9
mask = np.zeros([5, 5], dtype=np.uint8)
mask[0:3, 0] = 1
mask[2:5, 2:4] = 1
roi = cv2.bitwise_and(img, img, mask=mask)
print("img=\n", img)
print("mask=\n", mask)
print("roi=\n", roi)

# %%
# 4.9 分别提取OpenCV的logo图的红绿蓝三色,其实可以直接通过RGB来提取，但是这不是实际操作的会使用的，实际会使用HSV来
import cv2
import numpy as np

opencv = cv2.imread(".\\lesson_4\\opencv.png")
hsv = cv2.cvtColor(opencv, cv2.COLOR_BGR2HSV)
cv2.imshow("opencv", opencv)
# =============指定蓝色值的范围=============
min_blue = np.array([110, 50, 50])
max_blue = np.array([130, 255, 255])
# 使用HSV表示蓝色来确认蓝色区域
mask = cv2.inRange(hsv, min_blue, max_blue)
# 通过掩码控制的按位运算，锁定蓝色区域；计算机不能识别HSV色彩空间，所以我们从图片得到HSV后，就使用他来得到掩码，再从掩码从图片中提取
blue = cv2.bitwise_and(opencv, opencv, mask=mask)
cv2.imshow("blue", blue)
# =============指定绿色值的范围=============
minGreen = np.array([50, 50, 50])
maxGreen = np.array([70, 255, 255])
# 确定绿色区域
mask = cv2.inRange(hsv, minGreen, maxGreen)
# 通过掩码控制的按位与运算，锁定绿色区域
green = cv2.bitwise_and(opencv, opencv, mask=mask)
cv2.imshow('green', green)
# =============指定红色值的范围=============
minRed = np.array([0, 50, 50])
maxRed = np.array([30, 255, 255])
# 确定红色区域
mask = cv2.inRange(hsv, minRed, maxRed)
# 通过掩码控制的按位与运算，锁定红色区域
red = cv2.bitwise_and(opencv, opencv, mask=mask)
cv2.imshow('red', red)
cv2.waitKey()
cv2.destroyAllWindows()


# %%
# 4.10 提取一幅图像内的肤色
import cv2

img = cv2.imread(".\\lesson_4\\img.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
min_hue = 5
max_hue = 170
hue_mask = cv2.inRange(h, min_hue, max_hue)
min_sat = 25
max_sat = 170
sat_mask = cv2.inRange(s, min_sat, max_sat)
mask = hue_mask & sat_mask
roi = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("img", img)
cv2.imshow("ROI", roi)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 4.11 调整HSV色彩空间内V通道的值，观察处理结果
import cv2

img = cv2.imread(".\\lesson_4\\img.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
v[:, :] = 255
new_hsv = cv2.merge([h, s, v])
art = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)
cv2.imshow("img", img)
cv2.imshow("art", art)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 4.12 分析alpha通道的值
import cv2
import numpy as np

img = np.random.randint(2, 256, size=[2, 3, 3], dtype=np.uint8)
bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
print("img=\n", img)
print("bgra=\n", bgra)
b, g, r, a = cv2.split(bgra)
print("a=\n", a)
a[:, :] = 125
new_bgra = cv2.merge([b, g, r, a])
print("new_bgra=\n", new_bgra)

# %%
# 4.13 编写一个程序对图像alpha通道进行处理
import cv2

img = cv2.imread(".\\lesson_4\\img.png")
bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
b, g, r, a = cv2.split(bgra)
a[:, :] = 125
bgra125 = cv2.merge([b, g, r, a])
a[:, :] = 0
bgra0 = cv2.merge([b, g, r, a])
cv2.imshow("img", img)
cv2.imshow("bgra", bgra)
cv2.imshow("bgra125", bgra125)
cv2.imshow("bgra0", bgra0)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite("bgra.png", bgra)
cv2.imwrite("bgra125.png", bgra125)
cv2.imwrite("bgra0.png", bgra0)