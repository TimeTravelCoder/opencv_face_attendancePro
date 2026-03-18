"""OpenCV face 模块检查脚本。

本文件负责：
1. 输出当前 OpenCV 版本。
2. 检查当前安装包是否包含 cv2.face 模块。
3. 用于快速判断是否安装了 opencv-contrib-python。
"""

import cv2

# __version__ 用于查看当前安装的 OpenCV 版本号。
print(cv2.__version__)
# hasattr(cv2, "face") 用于检查当前 OpenCV 是否带有人脸识别扩展模块。
print(hasattr(cv2, "face"))
