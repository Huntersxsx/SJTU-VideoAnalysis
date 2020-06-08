import numpy as np
import cv2

# opencv的版本需要为2或3，opencv 4版本会在image, cnts, hierarchy = cv2.findContours()处报错

cap = cv2.VideoCapture(0)  # 打开摄像头
#cap = cv2.VideoCapture("F:/SJTU_VideoAnalysis/Project1/Input/demo.avi")

size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

mog = cv2.createBackgroundSubtractorMOG2()  # 定义高斯混合模型对象 mog
#fourcc1 = cv2.VideoWriter_fourcc(*'XVID')
#fourcc2 = cv2.VideoWriter_fourcc(*'XVID')
#out_detect = cv2.VideoWriter('output_detect.avi', fourcc1, 20.0, size)
#out_bg = cv2.VideoWriter('output_bg.avi', fourcc1, 20.0, size)
i = 0
while (1):  # 摄像头正常，进入循环体，读取摄像头每一帧图像
    ret, frame = cap.read()  # 读取摄像头每一帧图像，frame是这一帧的图像
    print(frame.shape)
    fgmask = mog.apply(frame)  # 使用前面定义的高斯混合模型对象 mog 当前帧的运动目标检测，返回二值图像
    gray_frame = fgmask.copy()
    # 使用 findContours 检测图像轮廓框，具体原理有论文，但不建议阅读。会使用即可。
    # 返回值：image，轮廓图像，不常用。 cnts，轮廓的坐标。 hierarchy，各个框之间父子关系，不常用。
    image, cnts, hierarchy = cv2.findContours(gray_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制每一个 cnts 框到原始图像 frame 中
    for c in cnts:
        if cv2.contourArea(c) < 900:  # 计算候选框的面积，如果小于1500，跳过当前候选框
            continue
        (x, y, w, h) = cv2.boundingRect(c)  # 根据轮廓c，得到当前最佳矩形框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)  # 将该矩形框画在当前帧 frame 上
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)  # 将该矩形框画在当前帧 frame 上

    cv2.imshow("Origin", frame)  # 显示当前帧
    cv2.imshow("GMM", image)  # 显示运动前景图像
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # print(image.shape)
    # print(frame.shape)

    #out_detect.write(frame)
    #out_bg.write(image)
    cv2.waitKey(20)
    i = i + 1

cap.release()  # 释放候选框
out_detect.release()
out_bg.release()
cv2.destroyAllWindows()

