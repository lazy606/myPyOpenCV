# %%
# 18.1 使用cv2.VideoCapture类捕获摄像头视频
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    c = cv2.waitKey(1)
    if c == 27: #ESC键盘
        break
cap.release()
cv2.destroyAllWindows()

# %%
# 18.2 使用cv2.VideoCapture类播放视频文件
import numpy as np
import cv2

cap = cv2.VideoCapture(".\\lesson_18\\video.mp4")
while (cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    c = cv2.waitKey(25)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()

# %%
# 18.3 使用cv2.VideoWriter类保存摄像头视频文件
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter.fourcc('I', '4', '2', '0')
out = cv2.VideoWriter("output.avi", fourcc, 20, (640, 480))
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        out.write(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == 27:
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()

# %%
# 18.4提取视频的Canny边缘检测结果
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.Canny(frame, 100, 200)
    cv2.imshow("frame", frame)
    c = cv2.waitKey(1)
    if c == 27: #ESC键盘
        break
cap.release()
cv2.destroyAllWindows()
