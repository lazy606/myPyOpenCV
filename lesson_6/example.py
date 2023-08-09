# 本章主要对使用阈值来进行不同亮度的处理，从而实现区分亮暗，区分前后景色

# %%
# 6.1 使用cv2.threshold()对数组进行二值化处理
import cv2
import numpy as np

img = np.random.randint(0, 256, size=[4, 5], dtype=np.uint8)
t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
print("img=\n", img)
print("t=", t)
print("rst=\n", rst)

# %%
# 6.2 对图像进行二值化阈值处理
import cv2

img = cv2.imread(".\\lesson_6\\danger.png", 0)
t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("img", img)
cv2.imshow("rst", rst)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 6.3 对数组进行反二值化阈值处理
import cv2

img = np.random.randint(0, 256, size=[4, 5], dtype=np.uint8)
t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
print("img=\n", img)
print("t=", t)
print("rst=\n", rst)

# %%
# 5.4 对图像进行反二值化阈值处理
import cv2

img = cv2.imread(".\\lesson_6\\danger.png", 0)
t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("img", img)
cv2.imshow("rst", rst)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 6.5 对数组进行截断阈值化处理
import cv2
import numpy as np

img = np.random.randint(0, 256, size=[4, 5], dtype=np.uint8)
t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
print("img=\n", img)
print("t=", t)
print("rst=\n", rst)

# %%
# 6.6 对图像进行阶段阈值化处理
import cv2

img = cv2.imread(".\\lesson_6\\danger.png", 0)
t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
cv2.imshow("img", img)
cv2.imshow("rst", rst)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 6.7 对数组进行超阈值0处理
import cv2
import numpy as np

img = np.random.randint(0, 256, size=[4, 5], dtype=np.uint8)
t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
print("img=\n", img)
print("t=", t)
print("rst=\n", rst)

# %%
# 6.8 对图像进行超阈值零处理（抠图？）
import cv2

img = cv2.imread(".\\lesson_6\\danger.png", 0)
t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
cv2.imshow("img", img)
cv2.imshow("rst", rst)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 6.9 对数组进行低阈值零处理
import cv2
import numpy as np

img = np.random.randint(0, 256, size=[4, 5], dtype=np.uint8)
t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
print("img=\n", img)
print("t=", t)
print("rst=\n", rst)

# %%
# 6.10 对图像进行低阈值零处理
import cv2

img = cv2.imread(".\\lesson_6\\danger.png", 0)
t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
cv2.imshow("img", img)
cv2.imshow("rst", rst)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 6.11 对一幅图像使用二值化阈值函数和自适应阈值函数进行处理
import cv2

img = cv2.imread(".\\lesson_6\\danger.png", 0)
t1, thd = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
a_thd_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 3)
a_thd_gaus = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,5, 3)
cv2.imshow("img", img)
cv2.imshow("thd", thd)
cv2.imshow("a_thd_mean", a_thd_mean)
cv2.imshow("a_thd_gaus",a_thd_gaus)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 6.12 测试Otsu阈值处理的实现
import cv2
import numpy as np

img = np.zeros((5, 5), dtype=np.uint8)
img[0:6, 0:6] = 123
img[2:6, 2:6] = 126
print("img=\n", img)
t1, thd = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
print("thd=\n", thd)
t2, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print("otsu=\n", otsu)

# %%
# 6.13 对一幅图像分别进行二值化阈值处理和Otsu阈值处理
import cv2

img = cv2.imread(".\\lesson_6\\danger.png", 0)
t1, thd = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
t2, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("img", img)
cv2.imshow("thd", thd)
cv2.imshow("otsu", otsu)
cv2.waitKey()
cv2.destroyAllWindows()