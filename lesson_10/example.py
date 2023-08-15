# %%
# 10.1 使用函数cv2.Canny()获取图像的边缘，并尝试使用不同的thredshold1和thredshold2,来得到边缘

import cv2

o = cv2.imread("lesson_10\\danger.png", cv2.IMREAD_GRAYSCALE)
r1 = cv2.Canny(o, 150, 220)
r2 = cv2.Canny(o, 32, 128)
cv2.imshow("original", o)
cv2.imshow("result1", r1)
cv2.imshow("result2", r2)
cv2.waitKey()
cv2.destroyAllWindows()