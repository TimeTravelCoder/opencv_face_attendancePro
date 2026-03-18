"""模型训练脚本。

本文件负责：
1. 扫描 raw_faces 目录下各人员的人脸样本。
2. 生成训练数据与标签映射。
3. 训练 LBPH 人脸识别模型并保存到 data/model 目录。
"""

import os
import json
import cv2
import numpy as np


RAW_FACE_DIR = os.path.join("data", "raw_faces")
MODEL_DIR = os.path.join("data", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "lbph_face_model.yml")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")


def load_training_data():
    """读取训练样本目录，生成训练图像、标签数组和标签映射。"""
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    if not os.path.exists(RAW_FACE_DIR):
        raise FileNotFoundError(f"未找到数据目录: {RAW_FACE_DIR}")

    person_names = sorted(
        [name for name in os.listdir(RAW_FACE_DIR)
         if os.path.isdir(os.path.join(RAW_FACE_DIR, name))]
    )

    if not person_names:
        raise ValueError("raw_faces 目录下没有任何人员文件夹。")

    for person_name in person_names:
        person_dir = os.path.join(RAW_FACE_DIR, person_name)
        image_files = sorted(
            [f for f in os.listdir(person_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )

        if not image_files:
            print(f"[跳过] {person_name} 文件夹中没有图片。")
            continue

        label_map[current_label] = person_name

        for image_name in image_files:
            image_path = os.path.join(person_dir, image_name)

            # imread() 以灰度模式读取训练图片。
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[跳过] 无法读取图片: {image_path}")
                continue

            # resize() 统一保险处理，确保尺寸一致。
            img = cv2.resize(img, (200, 200))

            faces.append(img)
            labels.append(current_label)

        print(f"[已加载] {person_name}: {len(image_files)} 张")
        current_label += 1

    if not faces:
        raise ValueError("没有可用于训练的图片，请检查采集数据。")

    return faces, np.array(labels), label_map


def main():
    """执行模型训练并保存结果。"""
    # 检查 cv2.face 是否存在
    if not hasattr(cv2, "face"):
        print("当前 OpenCV 没有 face 模块。")
        print("请安装: opencv-contrib-python")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        faces, labels, label_map = load_training_data()
    except Exception as e:
        print(f"加载训练数据失败: {e}")
        return

    print(f"\n开始训练，共 {len(faces)} 张图片，{len(label_map)} 个人。")

    # LBPHFaceRecognizer_create() 创建 LBPH 识别器对象。
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # train() 使用图片样本和标签完成训练。
    recognizer.train(faces, labels)

    # save() 将训练后的模型参数保存到磁盘。
    recognizer.save(MODEL_PATH)

    # json.dump() 保存标签编号到姓名的映射关系。
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print("\n训练完成！")
    print(f"模型已保存到: {MODEL_PATH}")
    print(f"标签映射已保存到: {LABEL_MAP_PATH}")


if __name__ == "__main__":
    main()
