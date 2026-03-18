"""签到记录查看脚本。

本文件负责：
1. 连接 SQLite 签到数据库。
2. 查询全部签到记录。
3. 按倒序打印到控制台，便于快速检查数据。
"""

import sqlite3
import os

DB_PATH = os.path.join("data", "attendance.db")

# sqlite3.connect() 用于连接本地 SQLite 数据库文件。
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# execute() 执行 SQL 查询语句，获取签到记录。
cursor.execute("""
    SELECT id, name, checkin_time, checkin_date, confidence
    FROM attendance
    ORDER BY id DESC
""")

# fetchall() 一次性取出全部查询结果。
rows = cursor.fetchall()
conn.close()

# 遍历并打印每条签到记录。
for row in rows:
    print(row)
