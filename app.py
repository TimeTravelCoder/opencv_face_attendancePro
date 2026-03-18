"""Flask Web 端入口。

本文件负责：
1. 初始化人员表、签到表、模型和标签映射。
2. 提供网页签到、网页注册、记录查询、统计和导出路由。
3. 支持浏览器相机采集与图片上传，并在注册后自动训练模型。
"""

import base64
import json
import os
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from werkzeug.exceptions import RequestEntityTooLarge
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)


app = Flask(__name__)
app.secret_key = "opencv-face-attendance-secret-key"
app.config["MAX_CONTENT_LENGTH"] = 24 * 1024 * 1024
app.config["MAX_FORM_MEMORY_SIZE"] = 24 * 1024 * 1024

MODEL_PATH = os.path.join("data", "model", "lbph_face_model.yml")
LABEL_MAP_PATH = os.path.join("data", "model", "label_map.json")
DB_PATH = os.path.join("data", "attendance.db")
EXPORT_DIR = os.path.join("data", "exports")
RAW_FACE_DIR = os.path.join("data", "raw_faces")

PERSON_CODE_PREFIX = "USR"
PERSON_CODE_SEQ_WIDTH = 4
DEFAULT_GENDER = "未填写"

recognizer = None
face_cascade = None
label_map = {}
last_seen = {}
cooldown_seconds = 8
threshold = 70
startup_error = ""


def get_db_connection():
    """创建 SQLite 连接，并将查询结果包装为 Row。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_dir(path):
    """确保目录存在。"""
    os.makedirs(path, exist_ok=True)


def list_face_directories():
    """返回当前样本人脸目录列表。"""
    ensure_dir(RAW_FACE_DIR)
    return sorted(
        [
            name
            for name in os.listdir(RAW_FACE_DIR)
            if os.path.isdir(os.path.join(RAW_FACE_DIR, name))
        ]
    )


def attendance_column_exists(conn, column_name):
    """检查 attendance 表是否包含指定字段。"""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(attendance)")
    columns = {row[1] for row in cursor.fetchall()}
    return column_name in columns


def generate_person_code(conn):
    """按规则生成人员编号：USR + 日期 + 4 位序号。"""
    prefix = datetime.now().strftime(f"{PERSON_CODE_PREFIX}%Y%m%d")
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT person_code
        FROM persons
        WHERE person_code LIKE ?
        ORDER BY person_code DESC
        LIMIT 1
        """,
        (f"{prefix}%",),
    )
    row = cursor.fetchone()

    if row and row["person_code"]:
        seq = int(row["person_code"][-PERSON_CODE_SEQ_WIDTH:]) + 1
    else:
        seq = 1

    return f"{prefix}{seq:0{PERSON_CODE_SEQ_WIDTH}d}"


def get_next_person_code():
    """获取当前页面可展示的下一个人员编号。"""
    conn = get_db_connection()
    try:
        return generate_person_code(conn)
    finally:
        conn.close()


def sync_persons_from_raw_faces(conn):
    """为历史样本目录自动补齐人员记录。"""
    cursor = conn.cursor()

    for face_key in list_face_directories():
        cursor.execute(
            "SELECT id FROM persons WHERE face_key = ?",
            (face_key,),
        )
        if cursor.fetchone():
            continue

        person_code = generate_person_code(conn)
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            INSERT INTO persons (person_code, name, gender, face_key, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (person_code, face_key, DEFAULT_GENDER, face_key, created_at),
        )


def backfill_attendance_fields(conn):
    """尽量为旧签到记录补齐人员编号和性别。"""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, name
        FROM attendance
        WHERE person_code IS NULL OR person_code = '' OR gender IS NULL OR gender = ''
        """
    )
    rows = cursor.fetchall()

    for row in rows:
        cursor.execute(
            """
            SELECT person_code, gender
            FROM persons
            WHERE name = ?
            ORDER BY id ASC
            LIMIT 1
            """,
            (row["name"],),
        )
        person = cursor.fetchone()
        if not person:
            continue

        cursor.execute(
            """
            UPDATE attendance
            SET person_code = COALESCE(NULLIF(person_code, ''), ?),
                gender = COALESCE(NULLIF(gender, ''), ?)
            WHERE id = ?
            """,
            (person["person_code"], person["gender"], row["id"]),
        )


def init_db():
    """初始化数据目录、人员表和签到表。"""
    ensure_dir("data")
    ensure_dir(EXPORT_DIR)
    ensure_dir(RAW_FACE_DIR)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_code TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            gender TEXT NOT NULL,
            face_key TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_code TEXT,
            name TEXT NOT NULL,
            gender TEXT,
            checkin_time TEXT NOT NULL,
            checkin_date TEXT NOT NULL,
            confidence REAL NOT NULL
        )
        """
    )

    if not attendance_column_exists(conn, "person_code"):
        cursor.execute("ALTER TABLE attendance ADD COLUMN person_code TEXT")

    if not attendance_column_exists(conn, "gender"):
        cursor.execute("ALTER TABLE attendance ADD COLUMN gender TEXT")

    conn.commit()
    sync_persons_from_raw_faces(conn)
    backfill_attendance_fields(conn)
    conn.commit()
    conn.close()


def load_label_map():
    """加载标签 ID 到 face_key 的映射。"""
    if not os.path.exists(LABEL_MAP_PATH):
        return {}

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    return {int(k): v for k, v in data.items()}


def get_person_by_face_key(face_key):
    """按 face_key 查询人员信息。"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT person_code, name, gender, face_key, created_at
        FROM persons
        WHERE face_key = ?
        LIMIT 1
        """,
        (face_key,),
    )
    person = cursor.fetchone()
    conn.close()
    return person


def get_person_by_code(person_code):
    """按人员编号查询人员信息。"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT person_code, name, gender, face_key, created_at
        FROM persons
        WHERE person_code = ?
        LIMIT 1
        """,
        (person_code,),
    )
    person = cursor.fetchone()
    conn.close()
    return person


def get_registered_people():
    """返回当前已注册人员列表。"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT person_code, name, gender, face_key, created_at
        FROM persons
        ORDER BY id DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()

    people = []
    for row in rows:
        face_dir = os.path.join(RAW_FACE_DIR, row["face_key"])
        sample_count = 0
        if os.path.isdir(face_dir):
            sample_count = len(
                [
                    file_name
                    for file_name in os.listdir(face_dir)
                    if file_name.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )

        people.append(
            {
                "person_code": row["person_code"],
                "name": row["name"],
                "gender": row["gender"],
                "created_at": row["created_at"],
                "sample_count": sample_count,
            }
        )

    return people


def has_checked_in_today(person_code):
    """判断指定人员今天是否已经签到。"""
    today = datetime.now().strftime("%Y-%m-%d")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM attendance
        WHERE person_code = ? AND checkin_date = ?
        """,
        (person_code, today),
    )
    count = cursor.fetchone()["cnt"]
    conn.close()

    return count > 0


def get_latest_attendance(person_code):
    """获取指定人员最近一条签到记录。"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, person_code, name, gender, checkin_time, checkin_date, confidence
        FROM attendance
        WHERE person_code = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (person_code,),
    )
    row = cursor.fetchone()
    conn.close()
    return row


def save_attendance(person, confidence):
    """写入一条签到记录。"""
    now = datetime.now()
    checkin_time = now.strftime("%Y-%m-%d %H:%M:%S")
    checkin_date = now.strftime("%Y-%m-%d")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO attendance (person_code, name, gender, checkin_time, checkin_date, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            person["person_code"],
            person["name"],
            person["gender"],
            checkin_time,
            checkin_date,
            float(confidence),
        ),
    )
    conn.commit()
    conn.close()


def update_person_info(person_code, new_name, new_gender):
    """更新人员基础信息，并同步历史签到中的姓名和性别。"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE persons
        SET name = ?, gender = ?
        WHERE person_code = ?
        """,
        (new_name, new_gender, person_code),
    )
    cursor.execute(
        """
        UPDATE attendance
        SET name = ?, gender = ?
        WHERE person_code = ?
        """,
        (new_name, new_gender, person_code),
    )
    conn.commit()
    conn.close()


def clear_model_files():
    """清空模型文件，并重置内存中的模型状态。"""
    global recognizer, label_map

    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    if os.path.exists(LABEL_MAP_PATH):
        os.remove(LABEL_MAP_PATH)

    recognizer = None
    label_map = {}


def prepare_model_artifacts():
    """基于当前样本目录准备新的识别器和标签映射。"""
    if not list_face_directories():
        return None, {}

    faces, labels, new_label_map = load_training_data()
    new_recognizer = cv2.face.LBPHFaceRecognizer_create()
    new_recognizer.train(faces, labels)
    return new_recognizer, new_label_map


def persist_model_artifacts(new_recognizer, new_label_map):
    """将训练结果原子替换到正式模型文件，并同步内存状态。"""
    global recognizer, label_map

    ensure_dir(os.path.dirname(MODEL_PATH))

    if new_recognizer is None:
        clear_model_files()
        return

    temp_dir = tempfile.mkdtemp(prefix="model_swap_", dir=os.path.dirname(MODEL_PATH))
    temp_model_path = os.path.join(temp_dir, "lbph_face_model.yml")
    temp_label_map_path = os.path.join(temp_dir, "label_map.json")

    try:
        new_recognizer.save(temp_model_path)
        with open(temp_label_map_path, "w", encoding="utf-8") as file:
            json.dump(new_label_map, file, ensure_ascii=False, indent=2)

        os.replace(temp_model_path, MODEL_PATH)
        os.replace(temp_label_map_path, LABEL_MAP_PATH)
        recognizer = new_recognizer
        label_map = new_label_map
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def delete_person_and_related_data(person_code):
    """删除人员、样本目录以及相关签到记录，并自动重训模型。"""
    conn = get_db_connection()
    cursor = conn.cursor()
    backup_root = tempfile.mkdtemp(prefix="person_delete_backup_")
    backup_face_dir = None
    face_dir = None
    transaction_started = False

    try:
        cursor.execute("BEGIN IMMEDIATE")
        transaction_started = True
        cursor.execute(
            """
            SELECT person_code, name, gender, face_key, created_at
            FROM persons
            WHERE person_code = ?
            LIMIT 1
            """,
            (person_code,),
        )
        person = cursor.fetchone()
        if person is None:
            raise ValueError("未找到该人员。")

        face_dir = os.path.join(RAW_FACE_DIR, person["face_key"])
        if os.path.isdir(face_dir):
            backup_face_dir = os.path.join(backup_root, person["face_key"])
            shutil.move(face_dir, backup_face_dir)

        cursor.execute("DELETE FROM attendance WHERE person_code = ?", (person_code,))
        cursor.execute("DELETE FROM persons WHERE person_code = ?", (person_code,))

        new_recognizer, new_label_map = prepare_model_artifacts()
        conn.commit()

        persist_model_artifacts(new_recognizer, new_label_map)
        last_seen.pop(person_code, None)
    except Exception:
        if transaction_started:
            conn.rollback()
        if backup_face_dir and os.path.isdir(backup_face_dir):
            shutil.move(backup_face_dir, face_dir)
        raise
    finally:
        conn.close()
        shutil.rmtree(backup_root, ignore_errors=True)


def register_person_with_samples(person_name, gender, uploaded_files, captured_images):
    """保存新人员、训练模型，并在失败时回滚样本和数据库。"""
    staging_dir = tempfile.mkdtemp(prefix="person_register_")
    conn = get_db_connection()
    final_dir = None
    transaction_started = False

    try:
        saved_count = extract_and_save_faces_to_dir(staging_dir, uploaded_files, captured_images)
        if saved_count == 0:
            raise ValueError("没有检测到有效人脸，请上传更清晰的正脸照片，或重新采集。")

        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        transaction_started = True

        person_code = generate_person_code(conn)
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        final_dir = os.path.join(RAW_FACE_DIR, person_code)
        shutil.move(staging_dir, final_dir)

        cursor.execute(
            """
            INSERT INTO persons (person_code, name, gender, face_key, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (person_code, person_name, gender, person_code, created_at),
        )

        new_recognizer, new_label_map = prepare_model_artifacts()
        conn.commit()
        persist_model_artifacts(new_recognizer, new_label_map)
        return person_code, saved_count
    except Exception:
        if transaction_started:
            conn.rollback()
        if final_dir and os.path.isdir(final_dir):
            shutil.rmtree(final_dir, ignore_errors=True)
        raise
    finally:
        conn.close()
        if os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)


def init_models():
    """初始化人脸检测器、识别器和标签映射。"""
    global recognizer, face_cascade, label_map

    if not hasattr(cv2, "face"):
        raise RuntimeError("当前 OpenCV 没有 face 模块，请安装 opencv-contrib-python。")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise RuntimeError("人脸检测模型加载失败。")

    label_map = load_label_map()

    if os.path.exists(MODEL_PATH):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_PATH)
    else:
        recognizer = None


def initialize_app_state():
    """初始化数据库和模型，并兼容 flask run 场景。"""
    global startup_error

    init_db()
    try:
        init_models()
        startup_error = ""
    except Exception as exc:
        startup_error = str(exc)


@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(_error):
    """表单或上传过大时给出友好提示。"""
    flash("提交的数据过大，请减少采集张数，或使用更少更清晰的样本后重试。", "error")
    return redirect(url_for("register_person"))


def load_training_data():
    """扫描原始人脸目录，组装训练图片、标签数组和标签映射。"""
    faces = []
    labels = []
    new_label_map = {}
    current_label = 0

    person_keys = list_face_directories()
    if not person_keys:
        raise ValueError("raw_faces 目录下没有任何人员文件夹。")

    for face_key in person_keys:
        person_dir = os.path.join(RAW_FACE_DIR, face_key)
        image_files = sorted(
            [
                file_name
                for file_name in os.listdir(person_dir)
                if file_name.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        if not image_files:
            continue

        new_label_map[current_label] = face_key

        valid_count = 0
        for image_name in image_files:
            image_path = os.path.join(person_dir, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (200, 200))
            faces.append(img)
            labels.append(current_label)
            valid_count += 1

        if valid_count > 0:
            current_label += 1

    if not faces:
        raise ValueError("没有可用于训练的图片，请检查采集数据。")

    return faces, np.array(labels), new_label_map


def retrain_model():
    """使用当前采集到的人脸样本重新训练并保存模型。"""
    new_recognizer, new_label_map = prepare_model_artifacts()
    persist_model_artifacts(new_recognizer, new_label_map)


def decode_data_url_image(data_url):
    """将 base64 data URL 转换为 OpenCV 图像。"""
    if not data_url or "," not in data_url:
        return None

    _, encoded = data_url.split(",", 1)
    try:
        image_bytes = base64.b64decode(encoded)
    except (ValueError, TypeError):
        return None

    file_bytes = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def detect_largest_face(gray_image):
    """检测并返回最大的人脸区域。"""
    if face_cascade is None or face_cascade.empty():
        raise RuntimeError("人脸检测器未初始化。")

    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    if len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda item: item[2] * item[3], reverse=True)
    return faces[0]


def save_face_crop(person_dir, gray_image, next_index):
    """从图像中提取最大人脸并保存。"""
    gray = cv2.equalizeHist(gray_image)
    largest_face = detect_largest_face(gray)
    if largest_face is None:
        return False

    x, y, w, h = largest_face
    face_img = gray[y:y + h, x:x + w]
    face_img = cv2.resize(face_img, (200, 200))

    file_path = os.path.join(person_dir, f"{next_index:03d}.jpg")
    cv2.imwrite(file_path, face_img)
    return True


def extract_and_save_faces_to_dir(person_dir, uploaded_files, captured_images):
    """从上传图片和网页采集图片中提取并保存人脸到指定目录。"""
    ensure_dir(person_dir)

    existing_files = [
        file_name
        for file_name in os.listdir(person_dir)
        if file_name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    next_index = len(existing_files) + 1
    saved_count = 0

    for file in uploaded_files:
        if not file or not file.filename:
            continue

        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if save_face_crop(person_dir, gray, next_index):
            next_index += 1
            saved_count += 1

    for captured_image in captured_images:
        image = decode_data_url_image(captured_image)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if save_face_crop(person_dir, gray, next_index):
            next_index += 1
            saved_count += 1

    return saved_count


def extract_and_save_faces(person_face_key, uploaded_files, captured_images):
    """兼容旧调用：按人员目录保存采集到的人脸样本。"""
    person_dir = os.path.join(RAW_FACE_DIR, person_face_key)
    return extract_and_save_faces_to_dir(person_dir, uploaded_files, captured_images)


def recognize_person_from_image(image):
    """识别单张图像中的最大人脸，并按需完成签到。"""
    frame_height, frame_width = image.shape[:2]

    if recognizer is None:
        return {
            "ok": False,
            "status": "model_not_ready",
            "message": "当前还没有训练模型，请先注册人员。",
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    largest_face = detect_largest_face(gray)

    if largest_face is None:
        return {
            "ok": False,
            "status": "no_face",
            "message": "没有检测到清晰人脸，请正对摄像头。",
            "face_box": None,
            "frame_size": {"width": frame_width, "height": frame_height},
        }

    x, y, w, h = largest_face
    face_box = {
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h),
    }
    face_img = gray[y:y + h, x:x + w]
    face_img = cv2.resize(face_img, (200, 200))
    label, confidence = recognizer.predict(face_img)

    if confidence >= threshold or label not in label_map:
        return {
            "ok": False,
            "status": "unregistered",
            "message": "未识别到已注册人员，请先完成注册。",
            "confidence": round(float(confidence), 1),
            "face_box": face_box,
            "frame_size": {"width": frame_width, "height": frame_height},
        }

    face_key = label_map[label]
    person = get_person_by_face_key(face_key)
    if person is None:
        return {
            "ok": False,
            "status": "person_missing",
            "message": "识别到了标签，但未找到对应人员信息，请重新训练模型。",
            "face_box": face_box,
            "frame_size": {"width": frame_width, "height": frame_height},
        }

    current_time = time.time()
    last_time = last_seen.get(person["person_code"], 0)

    if current_time - last_time < cooldown_seconds:
        latest_record = get_latest_attendance(person["person_code"])
        return {
            "ok": True,
            "status": "recognized",
            "message": "已识别到该人员，请勿重复靠近镜头。",
            "person": {
                "name": person["name"],
                "gender": person["gender"],
                "person_code": person["person_code"],
            },
            "checkin_time": latest_record["checkin_time"] if latest_record else "",
            "confidence": round(float(confidence), 1),
            "face_box": face_box,
            "frame_size": {"width": frame_width, "height": frame_height},
        }

    if has_checked_in_today(person["person_code"]):
        latest_record = get_latest_attendance(person["person_code"])
        last_seen[person["person_code"]] = current_time
        return {
            "ok": True,
            "status": "already_checked",
            "message": "今天已经签到过了。",
            "person": {
                "name": person["name"],
                "gender": person["gender"],
                "person_code": person["person_code"],
            },
            "checkin_time": latest_record["checkin_time"] if latest_record else "",
            "confidence": round(float(confidence), 1),
            "face_box": face_box,
            "frame_size": {"width": frame_width, "height": frame_height},
        }

    save_attendance(person, confidence)
    last_seen[person["person_code"]] = current_time
    latest_record = get_latest_attendance(person["person_code"])

    return {
        "ok": True,
        "status": "checked_in",
        "message": "签到成功。",
        "person": {
            "name": person["name"],
            "gender": person["gender"],
            "person_code": person["person_code"],
        },
        "checkin_time": latest_record["checkin_time"] if latest_record else "",
        "confidence": round(float(confidence), 1),
        "face_box": face_box,
        "frame_size": {"width": frame_width, "height": frame_height},
    }


@app.route("/")
def index():
    """首页：展示统计数据、最近签到记录和网页签到入口。"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) AS cnt FROM attendance")
    total_records = cursor.fetchone()["cnt"]

    today = datetime.now().strftime("%Y-%m-%d")
    cursor.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM attendance
        WHERE checkin_date = ?
        """,
        (today,),
    )
    today_records = cursor.fetchone()["cnt"]

    cursor.execute("SELECT COUNT(*) AS cnt FROM persons")
    registered_count = cursor.fetchone()["cnt"]

    cursor.execute(
        """
        SELECT id, person_code, name, gender, checkin_time, checkin_date, confidence
        FROM attendance
        ORDER BY id DESC
        LIMIT 5
        """
    )
    recent_records = cursor.fetchall()
    conn.close()

    return render_template(
        "index.html",
        total_records=total_records,
        today_records=today_records,
        registered_count=registered_count,
        recent_records=recent_records,
        model_ready=(recognizer is not None and face_cascade is not None),
        startup_error=startup_error,
    )


@app.route("/api/checkin/recognize", methods=["POST"])
def api_checkin_recognize():
    """网页端签到识别接口。"""
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image", "")
    image = decode_data_url_image(image_data)

    if image is None:
        return jsonify(
            {
                "ok": False,
                "status": "bad_image",
                "message": "未收到有效图像，请重试。",
            }
        ), 400

    try:
        return jsonify(recognize_person_from_image(image))
    except Exception as exc:
        return jsonify(
            {
                "ok": False,
                "status": "server_error",
                "message": f"识别失败：{exc}",
            }
        ), 500


@app.route("/records")
def records():
    """记录页：支持按姓名、编号、性别和日期筛选签到记录。"""
    keyword = request.args.get("keyword", "").strip()
    gender = request.args.get("gender", "").strip()
    checkin_date = request.args.get("checkin_date", "").strip()

    sql = """
        SELECT id, person_code, name, gender, checkin_time, checkin_date, confidence
        FROM attendance
        WHERE 1=1
    """
    params = []

    if keyword:
        sql += " AND (name LIKE ? OR person_code LIKE ?)"
        params.extend([f"%{keyword}%", f"%{keyword}%"])

    if gender:
        sql += " AND gender = ?"
        params.append(gender)

    if checkin_date:
        sql += " AND checkin_date = ?"
        params.append(checkin_date)

    sql += " ORDER BY id DESC"

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(sql, params)
    all_records = cursor.fetchall()
    conn.close()

    return render_template(
        "records.html",
        records=all_records,
        keyword=keyword,
        gender=gender,
        checkin_date=checkin_date,
    )


@app.route("/records/delete/<int:record_id>", methods=["POST"])
def delete_record(record_id):
    """删除指定签到记录。"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()

    flash(f"记录 #{record_id} 已删除。", "success")
    return redirect(url_for("records"))


@app.route("/register", methods=["GET", "POST"])
def register_person():
    """注册页：支持上传照片或浏览器采集，并自动训练模型。"""
    if request.method == "POST":
        person_name = request.form.get("person_name", "").strip()
        gender = request.form.get("gender", "").strip() or DEFAULT_GENDER
        uploaded_files = request.files.getlist("face_images")
        captured_images_raw = request.form.get("captured_images", "[]").strip()

        try:
            captured_images = json.loads(captured_images_raw) if captured_images_raw else []
        except json.JSONDecodeError:
            captured_images = []

        if not person_name:
            flash("姓名不能为空。", "error")
            return redirect(url_for("register_person"))

        if gender not in {"男", "女", "其他", DEFAULT_GENDER}:
            flash("请选择有效的性别。", "error")
            return redirect(url_for("register_person"))

        if (not uploaded_files or all(not file.filename for file in uploaded_files)) and not captured_images:
            flash("请上传照片，或先用网页相机采集若干张样本。", "error")
            return redirect(url_for("register_person"))

        try:
            person_code, saved_count = register_person_with_samples(
                person_name=person_name,
                gender=gender,
                uploaded_files=uploaded_files,
                captured_images=captured_images,
            )
            flash(
                f"注册成功：{person_name}，编号 {person_code}，性别 {gender}，"
                f"已保存 {saved_count} 张有效人脸并自动完成训练。",
                "success",
            )
            return redirect(url_for("register_person"))
        except Exception as exc:
            flash(f"注册失败：{exc}", "error")
            return redirect(url_for("register_person"))

    people = get_registered_people()
    return render_template(
        "register.html",
        people=people,
        next_person_code=get_next_person_code(),
        id_rule=f"{PERSON_CODE_PREFIX} + 日期(YYYYMMDD) + {PERSON_CODE_SEQ_WIDTH}位序号",
        startup_error=startup_error,
    )


@app.route("/persons/update/<person_code>", methods=["POST"])
def update_person(person_code):
    """更新人员姓名和性别。"""
    person = get_person_by_code(person_code)
    if person is None:
        flash("未找到要更新的人员。", "error")
        return redirect(url_for("register_person"))

    new_name = request.form.get("name", "").strip()
    new_gender = request.form.get("gender", "").strip() or DEFAULT_GENDER

    if not new_name:
        flash("姓名不能为空。", "error")
        return redirect(url_for("register_person"))

    if new_gender not in {"男", "女", "其他", DEFAULT_GENDER}:
        flash("请选择有效的性别。", "error")
        return redirect(url_for("register_person"))

    update_person_info(person_code, new_name, new_gender)
    flash(f"人员 {person_code} 的信息已更新。", "success")
    return redirect(url_for("register_person"))


@app.route("/persons/delete/<person_code>", methods=["POST"])
def delete_person(person_code):
    """删除人员、样本和相关签到记录。"""
    try:
        delete_person_and_related_data(person_code)
        flash(f"人员 {person_code} 已删除，相关样本和签到记录已同步清理。", "success")
    except Exception as exc:
        flash(f"删除失败：{exc}", "error")

    return redirect(url_for("register_person"))


@app.route("/stats")
def stats():
    """统计页：按日期和人员维度聚合签到次数。"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT checkin_date, COUNT(*) AS cnt
        FROM attendance
        GROUP BY checkin_date
        ORDER BY checkin_date
        """
    )
    daily_rows = cursor.fetchall()

    cursor.execute(
        """
        SELECT person_code, name, COUNT(*) AS cnt
        FROM attendance
        GROUP BY person_code, name
        ORDER BY cnt DESC, name ASC
        """
    )
    person_rows = cursor.fetchall()
    conn.close()

    daily_labels = [row["checkin_date"] for row in daily_rows]
    daily_counts = [row["cnt"] for row in daily_rows]

    person_labels = [
        f'{row["name"]} ({row["person_code"] or "未补全"})' for row in person_rows
    ]
    person_counts = [row["cnt"] for row in person_rows]
    top_person_rows = person_rows[:5]

    total_records = sum(daily_counts)
    total_days = len(daily_counts)
    average_daily = round(total_records / total_days, 1) if total_days else 0
    peak_day = max(daily_rows, key=lambda row: row["cnt"]) if daily_rows else None
    top_person = top_person_rows[0] if top_person_rows else None
    today = datetime.now().strftime("%Y-%m-%d")
    today_count = next(
        (row["cnt"] for row in daily_rows if row["checkin_date"] == today),
        0,
    )

    return render_template(
        "stats.html",
        daily_labels=daily_labels,
        daily_counts=daily_counts,
        person_labels=person_labels,
        person_counts=person_counts,
        top_labels=[
            f'{row["name"]} ({row["person_code"] or "未补全"})'
            for row in top_person_rows
        ],
        top_counts=[row["cnt"] for row in top_person_rows],
        total_records=total_records,
        total_days=total_days,
        average_daily=average_daily,
        today_count=today_count,
        peak_day_label=peak_day["checkin_date"] if peak_day else "暂无数据",
        peak_day_count=peak_day["cnt"] if peak_day else 0,
        top_person_name=top_person["name"] if top_person else "暂无数据",
        top_person_code=top_person["person_code"] if top_person else "",
        top_person_count=top_person["cnt"] if top_person else 0,
    )


@app.route("/export/csv")
def export_csv():
    """导出 CSV 文件。"""
    conn = get_db_connection()
    df = pd.read_sql_query(
        """
        SELECT id, person_code AS 人员编号, name AS 姓名, gender AS 性别,
               checkin_time AS 签到时间, checkin_date AS 签到日期, confidence AS 识别距离
        FROM attendance
        ORDER BY id DESC
        """,
        conn,
    )
    conn.close()

    file_path = os.path.join(EXPORT_DIR, "attendance_records.csv")
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    return send_file(file_path, as_attachment=True)


@app.route("/export/excel")
def export_excel():
    """导出 Excel 文件。"""
    conn = get_db_connection()
    df = pd.read_sql_query(
        """
        SELECT id, person_code AS 人员编号, name AS 姓名, gender AS 性别,
               checkin_time AS 签到时间, checkin_date AS 签到日期, confidence AS 识别距离
        FROM attendance
        ORDER BY id DESC
        """,
        conn,
    )
    conn.close()

    file_path = os.path.join(EXPORT_DIR, "attendance_records.xlsx")
    df.to_excel(file_path, index=False)
    return send_file(file_path, as_attachment=True)


initialize_app_state()


if __name__ == "__main__":
    app.run(debug=True)
