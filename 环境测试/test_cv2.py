"""OpenCV 基础环境测试脚本。

本文件负责：
1. 输出当前 OpenCV 版本。
2. 作为最基础的安装校验入口。
"""

import cv2

# 输出当前安装的 OpenCV 版本，用于快速确认环境是否安装正确。
print(cv2.__version__)
