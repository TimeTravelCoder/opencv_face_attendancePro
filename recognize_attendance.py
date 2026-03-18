"""命令行签到识别脚本。

本文件负责：
1. 加载训练好的人脸识别模型和标签映射。
2. 打开摄像头实时识别人脸。
3. 对识别成功的人员执行签到去重并写入数据库。
"""

import os
import json
import cv2
import sqlite3
import time
from datetime import datetime


MODEL_PATH = os.path.join("data", "model", "lbph_face_model.yml")
LABEL_MAP_PATH = os.path.join("data", "model", "label_map.json")
DB_PATH = os.path.join("data", "attendance.db")


def load_label_map():
    """读取标签 ID 到姓名的映射。"""
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(f"未找到标签映射文件: {LABEL_MAP_PATH}")

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    return {int(k): v for k, v in label_map.items()}


def init_db():
    """初始化签到数据库表。"""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            checkin_time TEXT NOT NULL,
            checkin_date TEXT NOT NULL,
            confidence REAL NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def has_checked_in_today(name):
    """判断指定人员今天是否已经签到。"""
    today = datetime.now().strftime("%Y-%m-%d")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM attendance
        WHERE name = ? AND checkin_date = ?
    """, (name, today))

    count = cursor.fetchone()[0]
    conn.close()

    return count > 0


def save_attendance(name, confidence):
    """保存签到记录。"""
    now = datetime.now()
    checkin_time = now.strftime("%Y-%m-%d %H:%M:%S")
    checkin_date = now.strftime("%Y-%m-%d")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO attendance (name, checkin_time, checkin_date, confidence)
        VALUES (?, ?, ?, ?)
    """, (name, checkin_time, checkin_date, float(confidence)))

    conn.commit()
    conn.close()


def main():
    """运行命令行版实时签到识别。"""
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

    init_db()

    # cv2.CascadeClassifier() 加载 Haar 级联检测器，用于先定位人脸。
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        print("人脸检测模型加载失败。")
        return

    # cv2.face.LBPHFaceRecognizer_create() 创建 LBPH 识别器对象。
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # read() 从磁盘加载训练好的识别模型。
    recognizer.read(MODEL_PATH)

    # VideoCapture() 打开默认摄像头，提供实时识别输入。
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头。")
        return

    print("签到识别已启动，按 q 退出。")

    # 防止同一张脸在短时间内被频繁触发
    last_seen = {}
    cooldown_seconds = 8

    while True:
        # read() 持续读取当前摄像头帧。
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面。")
            break

        # copy() 生成一份显示画面，避免在原始帧上重复叠加。
        display_frame = frame.copy()
        # cvtColor() 转灰度，equalizeHist() 增强对比度，便于检测与识别。
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # detectMultiScale() 返回当前画面中的人脸候选框列表。
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            # resize() 统一输入尺寸，保持与训练数据一致。
            face_img = cv2.resize(face_img, (200, 200))

            # predict() 返回预测标签和置信距离，距离越小通常代表越像。
            label, confidence = recognizer.predict(face_img)

            threshold = 70
            status_text = ""

            if confidence < threshold and label in label_map:
                name = label_map[label]
                color = (0, 255, 0)

                current_time = time.time()
                last_time = last_seen.get(name, 0)

                if current_time - last_time >= cooldown_seconds:
                    if not has_checked_in_today(name):
                        save_attendance(name, confidence)
                        status_text = "Checked in"
                        print(f"[签到成功] {name}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    else:
                        status_text = "Already checked today"

                    last_seen[name] = current_time
                else:
                    status_text = "Recognized"

                text = f"{name} ({confidence:.1f})"
            else:
                name = "Unknown"
                color = (0, 0, 255)
                text = f"{name} ({confidence:.1f})"
                status_text = "Unregistered"

            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

            # putText() 用于显示识别姓名、置信度和签到状态。
            cv2.putText(
                display_frame,
                text,
                (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2
            )

            cv2.putText(
                display_frame,
                status_text,
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
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

        # imshow() 把识别结果显示在桌面窗口。
        cv2.imshow("Face Attendance System", display_frame)

        # waitKey() 处理窗口消息并监听 q 键退出。
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 退出前释放摄像头和关闭窗口。
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
