"""命令行人脸采集脚本。

本文件负责：
1. 让用户输入人员姓名。
2. 打开摄像头采集该人员的人脸图片。
3. 自动裁剪、标准化并保存训练样本到 raw_faces 目录。
"""

import os
import cv2


def main():
    """运行命令行版人脸采集流程。"""
    # 1. 输入姓名
    person_name = input("请输入姓名或编号（建议用拼音/英文）: ").strip()
    if not person_name:
        print("姓名不能为空。")
        return

    # os.makedirs() 创建当前人员的人脸样本目录。
    save_dir = os.path.join("data", "raw_faces", person_name)
    os.makedirs(save_dir, exist_ok=True)

    # cv2.CascadeClassifier() 加载人脸检测模型。
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    if face_cascade.empty():
        print("人脸检测模型加载失败。")
        return

    # VideoCapture() 打开默认摄像头。
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头。")
        return

    print("开始采集人脸，按 q 可提前退出。")
    print("请正对摄像头，并稍微变化角度、表情和距离。")

    count = 0
    max_count = 40
    frame_index = 0

    while True:
        # read() 返回一帧图像，作为采集输入。
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面。")
            break

        # copy() 用于显示；cvtColor() 转灰度；equalizeHist() 增强对比度。
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # detectMultiScale() 检测当前帧中所有人脸位置。
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        # 如果检测到多张脸，只取最大的一张
        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            x, y, w, h = faces[0]

            # rectangle() 在预览窗口中标出当前采集到的人脸区域。
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 每隔几帧保存一次，避免重复太多
            if frame_index % 5 == 0 and count < max_count:
                face_img = gray[y:y + h, x:x + w]
                # resize() 把样本统一成训练所需尺寸。
                face_img = cv2.resize(face_img, (200, 200))

                file_path = os.path.join(save_dir, f"{count + 1:03d}.jpg")
                # imwrite() 把裁剪后的人脸图写入磁盘。
                cv2.imwrite(file_path, face_img)
                count += 1

        # putText() 在窗口中实时显示姓名、采集数量和退出提示。
        cv2.putText(
            display_frame,
            f"Name: {person_name}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )

        cv2.putText(
            display_frame,
            f"Saved: {count}/{max_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        cv2.putText(
            display_frame,
            "Press q to quit",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # imshow() 显示当前采集画面。
        cv2.imshow("Register Face", display_frame)

        frame_index += 1

        # 达到目标张数自动结束
        if count >= max_count:
            print(f"{person_name} 的人脸采集完成，共保存 {count} 张。")
            break

        # waitKey() 处理窗口事件并捕获 q 键退出。
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("已手动退出采集。")
            break

    # 释放摄像头并关闭窗口。
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
