"""摄像头连通性测试脚本。

本文件负责：
1. 验证 OpenCV 是否能成功打开摄像头。
2. 实时显示摄像头画面。
3. 作为环境排查工具，不涉及人脸识别逻辑。
"""

import cv2


def main():
    """运行最小化摄像头测试。"""
    # cv2.VideoCapture(0) 用于创建摄像头采集对象；参数 0 通常表示默认摄像头。
    cap = cv2.VideoCapture(0)

    # cap.isOpened() 用于判断摄像头是否成功打开。
    if not cap.isOpened():
        print("无法打开摄像头，请检查权限或设备占用。")
        return

    print("摄像头已打开，按 q 键退出。")

    while True:
        # cap.read() 负责读取一帧图像，返回是否成功和当前帧数据。
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面。")
            break

        # cv2.putText() 在当前帧上绘制提示文字。
        cv2.putText(
            frame,
            "Press 'q' to quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # cv2.imshow() 将处理后的图像显示在指定窗口中。
        cv2.imshow("OpenCV Camera Test", frame)

        # cv2.waitKey(1) 等待键盘输入并处理窗口事件；按下 q 时退出循环。
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cap.release() 释放摄像头设备，cv2.destroyAllWindows() 关闭所有 OpenCV 窗口。
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
