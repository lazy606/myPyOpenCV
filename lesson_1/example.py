# example.py - OpenCV例题文件


#%%
# 1.1读入图像
import cv2
lena = cv2.imread("fiveHunderMiles.jpg")
print(lena)

#%%
# 1.2在一个窗口内显示读取的图像
import cv2
lena = cv2.imread("fiveHunderMiles.jpg")
cv2.namedWindow("lesson")
cv2.imshow("lesson", lena)
cv2.waitKey(0)          # waitKey()为什么要加上？

#%%
# 1.3在一个窗口内显示图像，并针对不同的案件做出不同的反应
import cv2
lena = cv2.imread("fiveHunderMiles.jpg")
cv2.imshow("demo", lena)
cv2.waitKey(0)
key=cv2.waitKey()
if key == ord('A'):
    cv2.imshow("PressA", lena)
    cv2.waitKey(0)
elif key == ord('B'):
    cv2.imshow("PressB", lena)
    cv2.waitKey(0)

#%%
# 1.4 在一个窗口内显示图像，用cv2.waitKey()实现程序暂停，按下案件后让程序继续运行
import cv2
lena = cv2.imread("fiveHunderMiles.jpg")
cv2.imshow("demo", lena)
key = cv2.waitKey()
if key != -1:
    print("触发了按键")

#%%
# 1.5 使用destroyWindow来释放窗口
import cv2
lena = cv2.imread("fiveHunderMiles.jpg")
cv2.imshow("demo", lena)
cv2.waitKey()
cv2.destroyWindow("demo")

#%%
# 1.6 使用destroyAllWindows来释放所有窗口
import cv2
lena = cv2.imread("fiveHunderMiles.jpg")
cv2.imshow("demo", lena)
cv2.imshow("demo2", lena)
cv2.waitKey()
cv2.destroyAllWindows()

#%%
# 1.7 使用imwrite保存图像
import cv2
lena = cv2.imread("fiveHunderMiles.jpg")
r = cv2.imwrite("result.jpg", lena)

