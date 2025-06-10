import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='输入图片路径')
parser.add_argument('--output', type=str, required=True, help='输出bin文件路径')
args = parser.parse_args()

img = cv2.imread(args.input)
img = cv2.resize(img, (640, 640))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.transpose(2, 0, 1)  # HWC to CHW
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, 0)  # 增加batch维
img.tofile(args.output) 