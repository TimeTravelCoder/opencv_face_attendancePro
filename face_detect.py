"""摄像头人脸检测演示脚本。

本文件负责：
1. 打开摄像头读取实时画面。
2. 使用 Haar 级联模型检测人脸位置。
3. 在窗口中绘制人脸框并显示检测数量。
"""

import cv2


def main():
    """运行基础人脸检测演示。"""
    # cv2.CascadeClassifier() 用于加载 OpenCV 自带的人脸检测模型。
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # empty() 用于判断模型文件是否成功加载。
    if face_cascade.empty():
        print("人脸检测模型加载失败。")
        return

    # cv2.VideoCapture(0) 打开默认摄像头设备。
    cap = cv2.VideoCapture(0)

    # isOpened() 用于确认摄像头是否可正常读取。
    if not cap.isOpened():
        print("无法打开摄像头，请检查权限或设备是否被占用。")
        return

    print("人脸检测已启动，按 q 键退出。")

    while True:
        # read() 返回当前帧读取是否成功以及对应的图像数据。
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面。")
            break

        # cv2.cvtColor() 把 BGR 彩色图转换为灰度图，便于后续检测。
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # cv2.equalizeHist() 用于增强灰度图对比度，提升检测稳定性。
        gray = cv2.equalizeHist(gray)

        # detectMultiScale() 会返回检测到的所有人脸矩形框。
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        # rectangle() 与 putText() 分别用于绘制人脸框和提示文字。
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # imshow() 把处理后的画面显示到桌面窗口。
        cv2.imshow("OpenCV Face Detection", frame)

        # waitKey() 用于处理窗口事件并读取键盘输入。
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # release() 释放摄像头；destroyAllWindows() 关闭全部 OpenCV 窗口。
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
