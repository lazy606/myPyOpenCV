# %%
# 3.1 使用+来做图像加法运算
import numpy as np

img1 = np.random.randint(0, 256, size=[3, 3], dtype=np.uint8)
img2 = np.random.randint(0, 256, size=[3, 3], dtype=np.uint8)
print("img1=\n", img1)
print("img2=\n", img2)
print("img1+img2=\n", img1 + img2)

# %%
# 3.2 使用+来做图像加法运算
import numpy as np
import cv2

img1 = np.random.randint(0, 256, size=[3, 3], dtype=np.uint8)
img2 = np.random.randint(0, 256, size=[3, 3], dtype=np.uint8)
print("img1=\n", img1)
print("img2=\n", img2)
img3 = cv2.add(img1, img2)
print("cv2.add(img1, img2)=\n", img3)

# %%
# 3.3 分别用记号计算加号运算符和函数cv2.add()计算两幅灰度图像像素这和
import cv2

a = cv2.imread(".\\lesson_3\\gipsy_danger.png", 0)
b = a
result1 = a + b
result2 = cv2.add(a, b)
cv2.imshow("original", a)
cv2.imshow("result1", result1)
cv2.imshow("result2", result2)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 3.4 使用数组演示cv2.addWeighted()的使用
import cv2
import numpy as np

img1 = np.ones((3, 4), dtype=np.uint8) * 100
img2 = np.ones((3, 4), dtype=np.uint8) * 10
gamma = 3
img3 = cv2.addWeighted(img1, 0.6, img2, 5, gamma)
print(img3)

# %%
# 3.5 使用函数cv2.addWeighted()对两幅图像进行加权混合
import cv2

a = cv2.imread(".\\lesson_3\\gipsy_danger.png")
b = cv2.imread(".\\lesson_3\\ocean.png")
result = cv2.addWeighted(a, 0.6, b, 0.4, 0)
cv2.imshow("gipsy_danger", a)
cv2.imshow("ocean", b)
cv2.imshow("result", result)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 3.6 使用函数cv2.addWeight将一幅图像的ROI混合在另外一幅图像内
import cv2

danger = cv2.imread("danger512.bmp", cv2.IMREAD_UNCHANGED)
dollar = cv2.imread("dollar.bmp", cv2.IMREAD_UNCHANGED)
cv2.imshow("danger", danger)
cv2.imshow("dollar", dollar)
face1 = danger[220:400, 250:350]
face2 = dollar[160:340, 200:300]
add = cv2.addWeighted(face1, 0.6, face2, 0.4, 0)
dollar[160:340, 200:300] = add
cv2.imshow("result", dollar)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 3.7 使用数组演示与掩模图像的按位与运算
import cv2
import numpy as np

a = np.random.randint(0, 255, (5, 5), dtype=np.uint8)
b = np.zeros((5, 5), dtype=np.uint8)
b[0:3, 0:3] = 255
b[4, 4] = 255
c = cv2.bitwise_and(a, b)
print("a=\n", a)
print("b=\n", b)
print("c=\n", c)

# %%
# 3.8 构造一个掩模图像，使用按位与运算保留途中被掩模指定的部分
import cv2
import numpy as np

a = cv2.imread(".\\lesson_3\\gipsy_danger.png", 0)
b = np.zeros(a.shape, dtype=np.uint8)
b[100:400, 200:400] = 255
b[100:500, 100:200] = 255
c = cv2.bitwise_and(a, b)
cv2.imshow("a", a)
cv2.imshow("b", b)
cv2.imshow("c", c)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 3.9 构造一个掩模图像，使用按照位与操作保留被掩模所指定的部分
import cv2
import numpy as np

a = cv2.imread(".\\lesson_3\\gipsy_danger.png", 1)
b = np.zeros(a.shape, dtype=np.uint8)
b[100:300, 200:400] = 255
b[100:400, 100:200] = 255
c = cv2.bitwise_and(a, b)
print("a.shape=", a.shape)
print("b.shape=", b.shape)
cv2.imshow("a", a)
cv2.imshow("b", b)
cv2.imshow("c", c)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 3.10 运算掩码的使用
import cv2
import numpy as np

img1 = np.ones((4, 4), dtype=np.uint8) * 3
img2 = np.ones((4, 4), dtype=np.uint8) * 5
mask = np.zeros((4, 4), dtype=np.uint8)
mask[2:4, 2:4] = 1
img3 = np.ones((4, 4), dtype=np.uint8) * 66
print("img1=\n", img1)
print("img2=\n", img2)
print("mask=\n", mask)
print("初始值img3=\n", img3)
img3 = cv2.add(img1, img2, mask=mask)
print("求和后的值为img3=\n", img3)

# %%
# 3.11 构造一个掩模图像，将该掩模图像作为安慰与函数的mask参数，实现保留图像指定部分
import cv2
import numpy as np

a = cv2.imread(".\\lesson_3\\gipsy_danger.png", 1)
w, h, l = a.shape
mask = np.zeros((w, h), dtype=np.uint8)
mask[100:200, 200:400] = 255
mask[100:300, 100:200] = 255
c = cv2.bitwise_and(a, a, mask=mask)  # 函数声明方式变了
print("a.shape", a.shape)
print("mask.shape", mask.shape)
cv2.imshow("a", a)
cv2.imshow("mask", mask)
cv2.imshow("c", c)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 3.12 演示图像与数值的运算结果, 数值无论前后没有区别
import cv2
import numpy as np

img1 = np.ones((4, 4), dtype=np.uint8) * 3
img2 = np.ones((4, 4), dtype=np.uint8) * 5
print("img1=\n", img1)
print("img2=\n", img2)
img3 = cv2.add(img1, img2)
print("cv2.add(img1, img2)=\n", img3)
img4 = cv2.add(img1, 6)
print("cv2.add(img1, 6)=\n", img4)
img5 = cv2.add(6, img2)
print("cv2.add(6, img2)=\n", img5)

# %%
# 3.13 观察灰度图像各个位平面
import cv2
import numpy as np

danger = cv2.imread(".\\lesson_3\\gipsy_danger.png", 0)
cv2.imshow("danger", danger)
r, c = danger.shape
x = np.zeros((r, c, 8), dtype=np.uint8)  # 这里创建8个通道，每一个通道就是一个对应的Mat
for i in range(8):
    x[:, :, i] = 2 ** i
r = np.zeros((r, c, 8), dtype=np.uint8)
for i in range(8):
    r[:, :, i] = cv2.bitwise_and(danger, x[:, :, i])
    mask = r[:, :, i] > 0
    r[mask] = 255
    cv2.imshow(str(i), r[:, :, i])
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 3.14 通过按位异或运算实现加密和解密
import cv2
import numpy as np

danger = cv2.imread(".\\lesson_3\\gipsy_danger.png", 0)
r, c = danger.shape
key = np.random.randint(0, 256, size=[r, c], dtype=np.uint8)
encryption = cv2.bitwise_xor(danger, key)
decryption = cv2.bitwise_xor(encryption, key)
cv2.imshow("danger", danger)
cv2.imshow("key", key)
cv2.imshow("encryption", encryption)
cv2.imshow("decryption", decryption)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 3.15 编写程序模拟数字水印的嵌入和提取
import cv2
import numpy as np

# 读取原始载体图像
danger = cv2.imread(".\\lesson_3\\gipsy_danger.png", 0)
# 读取水印图像
watermark = cv2.imread(".\\lesson_3\\fiveHunderMiles.jpg", 0)
# 将水印图像值处理为1，方便嵌入
w = watermark[:, :] > 0
watermark[w] = 1
# 读取原始载体图像的shape值
r, c = danger.shape
# ============嵌入过程============
# 生成元素值都是254的数组
t254 = np.ones((r, c), dtype=np.uint8) * 254
# 获取图像的高七位
dangerH7 = cv2.bitwise_and(danger, t254)
# 将watermark嵌入dangerH7
e = cv2.bitwise_or(dangerH7, watermark)
# ============提取过程============
t1 = np.ones((r, c), dtype=np.uint8)
# 从载体图像中提取水印图像
wm = cv2.bitwise_and(e, t1)
print(wm)
# 将水印内的值处理为255方便显示
w = wm[:, :] > 0
wm[w] = 255
# ============显示============
cv2.imshow("danger", danger)
cv2.imshow("watermark", watermark * 255)
cv2.imshow("e", e)
cv2.imshow("wm", wm)
cv2.waitKey()
cv2.destroyAllWindows()

# %%
# 3.16 使用掩码对脸部进行打码、解码
import cv2
import numpy as np

# 读取原始载体图像
danger = cv2.imread(".\\lesson_3\\gipsy_danger.png", 0)
# 读取原始载体图像的shape值
r, c = danger.shape
mask = np.zeros((r, c), dtype=np.uint8)
mask[220:400, 250:350] = 1
# 获取一个key，打码解码的密钥
key = np.random.randint(0, 256, size=[r, c], dtype=np.uint8)
# ============获取打码脸============
dangerXorKey = cv2.bitwise_xor(danger, key)
# 获取加密图像的脸部信息 encryptFace
encryptFace = cv2.bitwise_and(dangerXorKey, mask * 255)
# 将图像 danger 内的脸部值设置为 0，得到 noFace1
noFace1 = cv2.bitwise_and(danger, (1 - mask) * 255)
# 得到打码的 danger 图像
maskFace = encryptFace + noFace1
# ============将打码脸解码============
# 将脸部打码的 danger 与密钥 key 进行异或运算，得到脸部的原始信息
extractOriginal = cv2.bitwise_xor(maskFace, key)
# 将解码的脸部信息 extractOriginal 提取出来，得到 extractFace
extractFace = cv2.bitwise_and(extractOriginal, mask * 255)
# 从脸部打码的 danger 内提取没有脸部信息的 danger 图像，得到 noFace2
noFace2 = cv2.bitwise_and(maskFace, (1 - mask) * 255)
# 得到解码的 danger 图像
extractdanger = noFace2 + extractFace
# ============显示图像============
cv2.imshow("danger", danger)
cv2.imshow("mask", mask * 255)
cv2.imshow("1-mask", (1 - mask) * 255)
cv2.imshow("key", key)
cv2.imshow("dangerXorKey", dangerXorKey)
cv2.imshow("encryptFace", encryptFace)
cv2.imshow("noFace1", noFace1)
cv2.imshow("maskFace", maskFace)
cv2.imshow("extractOriginal", extractOriginal)
cv2.imshow("extractFace", extractFace)
cv2.imshow("noFace2", noFace2)
cv2.imshow("extractdanger", extractdanger)
cv2.waitKey()
cv2.destroyAllWindows()
