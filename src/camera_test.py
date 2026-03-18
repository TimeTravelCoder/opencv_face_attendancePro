import cv2


def main():
    # 尝试连接默认摄像头设备，参数 0 通常表示系统中的第一个摄像头。
    cap = cv2.VideoCapture(0)

    # 在进入采集循环前先确认设备可用，避免后续 read() 持续失败。
    if not cap.isOpened():
        print("无法打开摄像头，请检查权限或设备占用。")
        return

    print("摄像头已打开，按 q 键退出。")

    while True:
        # 逐帧读取摄像头画面；ret 为 False 表示当前帧读取失败。
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面。")
            break

        # 在预览窗口叠加退出提示，方便直接从画面中看到操作方式。
        cv2.putText(
            frame,
            "Press 'q' to quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # 实时显示当前摄像头帧。
        cv2.imshow("OpenCV Camera Test", frame)

        # waitKey(1) 既用于处理窗口事件，也用于捕获键盘输入。
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 退出前释放摄像头并销毁窗口，避免设备被后续进程占用。
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
