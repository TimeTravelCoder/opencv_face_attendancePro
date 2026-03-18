# 基于 OpenCV 与 Flask 的网页端人脸签到系统

一个适合课程设计、项目展示和复试演示的人脸签到系统。  
项目将 `OpenCV` 人脸检测与识别、`Flask` Web 服务、`SQLite` 本地数据库结合起来，实现了网页端人员注册、自动训练、签到识别、记录查询、人员管理与统计可视化。

## 项目特色

- 网页端签到：点击按钮后才开启浏览器摄像头进行识别
- 签到结果展示：识别成功后显示姓名、性别、人员编号、签到时间
- 人员注册：支持上传照片和网页采集两种方式
- 自动训练：注册完成后自动更新 LBPH 人脸识别模型
- 人员管理：支持修改人员信息、删除人员、同步删除样本与相关记录
- 记录管理：支持按姓名、编号、性别、日期筛选签到记录
- 统计分析：提供趋势图、排行图、占比图和指标卡展示
- 数据导出：支持导出 CSV 与 Excel

## 技术栈

- Python 3
- Flask
- OpenCV Contrib
- NumPy
- Pandas
- SQLite
- HTML / CSS / JavaScript
- Chart.js

## 功能模块

### 1. 网页签到

- 首页点击“开始签到”后才打开摄像头
- 浏览器采集图像并发送到后端识别
- 页面实时显示识别框
- 识别成功后自动记录签到信息
- 同一人员同一天只签到一次

### 2. 人员注册

- 输入姓名和性别
- 系统按规则自动生成人员编号
- 支持两种采集方式：
  - 上传多张照片
  - 网页端相机采集样本
- 自动检测并裁剪最大人脸
- 保存样本后自动完成模型训练

### 3. 人员管理

- 查看已注册人员列表
- 修改姓名与性别
- 删除人员
- 删除时同步清理：
  - 人脸样本
  - 相关签到记录
  - 识别模型映射

### 4. 记录与统计

- 查看最近签到记录
- 条件筛选签到数据
- 统计每日签到趋势
- 统计人员签到次数排行
- 展示人员签到占比

## 系统页面

- `/`：首页 / 网页签到
- `/register`：人员注册与人员管理
- `/records`：签到记录查询
- `/stats`：统计可视化大屏
- `/export/csv`：导出 CSV
- `/export/excel`：导出 Excel

## 项目结构

```text
opencv_face_attendance/
├── app.py                         # Flask 主程序
├── train_model.py                 # 命令行训练脚本
├── register_face.py               # 命令行采集脚本
├── recognize_attendance.py        # 命令行识别签到脚本
├── requirements.txt               # 依赖列表
├── static/
│   └── style.css                  # 页面样式
├── templates/
│   ├── base.html                  # 基础模板
│   ├── index.html                 # 首页 / 签到页
│   ├── register.html              # 注册与人员管理页
│   ├── records.html               # 签到记录页
│   └── stats.html                 # 统计分析页
└── data/
    ├── attendance.db              # SQLite 数据库
    ├── raw_faces/                 # 原始人脸样本
    ├── model/                     # 训练好的模型与标签映射
    └── exports/                   # 导出的 CSV / Excel 文件
```

## 运行环境

建议使用虚拟环境运行。

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动项目

```bash
python app.py
```

如果你的 `5000` 端口被占用，可以改用其他端口，例如：

```bash
python -c "from app import app; app.run(debug=True, port=5001)"
```

如果你使用项目虚拟环境，推荐：

```bash
./.venv/bin/python app.py
```

## 首次使用流程

1. 启动 Flask 服务
2. 打开浏览器访问首页
3. 进入“人员注册”页面
4. 填写姓名、性别
5. 上传照片或使用网页采集人脸样本
6. 系统自动训练模型
7. 返回首页进行签到识别

## 人员编号规则

当前默认规则为：

```text
USR + 日期(YYYYMMDD) + 4位序号
```

例如：

```text
USR202603200001
```

## 数据存储说明

### persons 表

存储注册人员的基础信息：

- `person_code`：人员编号
- `name`：姓名
- `gender`：性别
- `face_key`：样本目录标识
- `created_at`：注册时间

### attendance 表

存储签到记录：

- `person_code`：人员编号
- `name`：姓名
- `gender`：性别
- `checkin_time`：签到时间
- `checkin_date`：签到日期
- `confidence`：识别距离

## 识别流程

1. 浏览器采集当前画面
2. 后端接收图像并转为 OpenCV 格式
3. 使用 Haar 级联检测人脸
4. 裁剪最大人脸并统一尺寸
5. 使用 `LBPHFaceRecognizer` 进行预测
6. 若识别成功且当天未签到，则写入数据库
7. 前端显示识别框与签到结果

## 可视化内容

统计页目前包含：

- 总签到记录
- 今日签到人数
- 最活跃人员
- 日均签到
- 每日签到趋势折线图
- 签到次数排行图
- 人员签到占比环形图
- 统计摘要卡片

## 常见问题

### 1. `ModuleNotFoundError: No module named 'cv2'`

说明当前 Python 环境没有安装 OpenCV Contrib。

解决方法：

```bash
pip install opencv-contrib-python
```

或者使用已安装依赖的虚拟环境启动：

```bash
./.venv/bin/python app.py
```

### 2. `Port 5000 is in use`

说明本机 `5000` 端口被其他程序占用。

可改用：

```bash
python -c "from app import app; app.run(debug=True, port=5001)"
```

### 3. 网页采集时提示 `Request Entity Too Large`

项目中已经对采集图像做了压缩处理，并提高了上传限制。  
如果仍然出现，建议：

- 减少一次提交的采集张数
- 保留清晰、有效的人脸样本
- 避免上传过大的原始图片

### 4. 网页无法打开摄像头

请检查：

- 浏览器是否允许摄像头权限
- 是否使用支持 `getUserMedia` 的现代浏览器
- 是否有其他程序占用摄像头

## 项目亮点

- 将传统 OpenCV 命令行程序改造成了网页端交互系统
- 实现了“注册后自动训练”的完整闭环
- 实现了人员信息管理与记录管理
- 对注册和删除流程做了数据一致性修复
- 优化了识别框显示和统计页面视觉效果

## 适用场景

- 课程设计
- 毕业设计前期原型
- 复试项目展示
- 小型实验室 / 班级签到演示

## 后续可扩展方向

- 增加活体检测，降低照片攻击风险
- 替换为深度学习特征提取模型
- 支持多人同时签到
- 增加管理员权限控制
- 接入 MySQL / PostgreSQL
- 支持云端部署与局域网访问

## 作者说明

如果你准备将本项目用于复试展示，建议重点演示以下流程：

1. 注册新人员
2. 自动训练模型
3. 首页点击签到并识别
4. 展示签到记录
5. 展示统计页面

这样能完整体现项目闭环和工程实现能力。
