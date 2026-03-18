"""命令行实时人脸识别脚本。

本文件负责：
1. 加载训练好的人脸识别模型与标签映射。
2. 打开摄像头识别当前画面中的人员。
3. 仅做识别展示，不写入签到数据库。
"""

import os
import json
import cv2


MODEL_PATH = os.path.join("data", "model", "lbph_face_model.yml")
LABEL_MAP_PATH = os.path.join("data", "model", "label_map.json")


def load_label_map():
    """读取标签 ID 到姓名的映射。"""
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(f"未找到标签映射文件: {LABEL_MAP_PATH}")

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    # json 读出来的 key 是字符串，这里转回 int
    return {int(k): v for k, v in label_map.items()}


def main():
    """运行命令行版实时人脸识别。"""
    if not hasattr(cv2, "face"):
        print("当前 OpenCV 没有 face 模块，请确认安装的是 opencv-contrib-python。")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"未找到模型文件: {MODEL_PATH}")
        return

    try:
        label_map = load_label_map()
    except Exception as e:
        print(f"加载标签映射失败: {e}")
        return

    # cv2.CascadeClassifier() 负责加载人脸检测模型。
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        print("人脸检测模型加载失败。")
        return

    # cv2.face.LBPHFaceRecognizer_create() 创建 LBPH 识别器。
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # read() 从磁盘加载训练完成的模型参数。
    recognizer.read(MODEL_PATH)

    # VideoCapture() 打开默认摄像头。
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头。")
        return

    print("实时识别已启动，按 q 退出。")

    while True:
        # read() 返回当前帧是否读取成功以及画面内容。
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面。")
            break

        # copy() 复制显示帧；cvtColor() 转灰度；equalizeHist() 增强对比度。
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # detectMultiScale() 在灰度图中检测所有人脸位置。
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            # resize() 统一尺寸，保持与训练阶段一致。
            face_img = cv2.resize(face_img, (200, 200))

            # predict() 输出最可能的标签和对应距离值。
            label, confidence = recognizer.predict(face_img)

            # 阈值需要根据你自己的数据微调
            # 对 LBPH，这里通常把 confidence 当“距离”看待：越小越像
            threshold = 70

            if confidence < threshold and label in label_map:
                name = label_map[label]
                color = (0, 255, 0)
                text = f"{name} ({confidence:.1f})"
            else:
                name = "Unknown"
                color = (0, 0, 255)
                text = f"{name} ({confidence:.1f})"

            # rectangle() 画框，putText() 绘制姓名与置信度说明。
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                display_frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        cv2.putText(
            display_frame,
            "Press q to quit",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )

        # imshow() 把识别结果显示到窗口中。
        cv2.imshow("Real-time Face Recognition", display_frame)

        # waitKey() 监听用户按键，这里按 q 结束程序。
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 释放摄像头并关闭窗口。
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
